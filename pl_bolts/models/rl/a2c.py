import argparse
from collections import OrderedDict, deque
from typing import Tuple, List

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import log_softmax, softmax
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F

from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.models.rl.common import cli
from pl_bolts.models.rl.common.agents import PolicyAgent
from pl_bolts.models.rl.common.multiprocessing_env import SubprocVecEnv
from pl_bolts.models.rl.common.networks import MLP


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value


class A2C(pl.LightningModule):
    def __init__(
        self,
        env: str,
        gamma: float = 0.99,
        lr: float = 0.01,
        batch_size: int = 8,
        n_steps: int = 10,
        avg_reward_len: int = 100,
        num_envs: int = 4,
        entropy_beta: float = 0.01,
        epoch_len: int = 1000,
        **kwargs
    ) -> None:
        """
        PyTorch Lightning implementation of `Advantage Actor Critic
        <https://papers.nips.cc/paper/
        1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_
        Paper authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour
        Model implemented by:

            - `Donal Byrne <https://github.com/djbyrne>`

        Example:
            >>> from pl_bolts.models.rl.vanilla_policy_gradient_model import VanillaPolicyGradient
            ...
            >>> model = VanillaPolicyGradient("PongNoFrameskip-v4")

        Train::

            trainer = Trainer()
            trainer.fit(model)

        Args:
            env: gym environment tag
            gamma: discount factor
            lr: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            batch_episodes: how many episodes to rollout for each batch of training
            entropy_beta: dictates the level of entropy per batch
            avg_reward_len: how many episodes to take into account when calculating the avg reward

        Note:
            This example is based on:
            https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter11/04_cartpole_pg.py

        Note:
            Currently only supports CPU and single GPU training with `distributed_backend=dp`
        """
        super().__init__()

        # Hyperparameters
        self.lr = 0.001
        self.batch_size = 32 # this is scaled to the number of environments
        self.batches_per_epoch = 200
        self.entropy_beta = entropy_beta
        self.gamma = gamma
        self.n_steps = 5
        self.entropy = 0

        self.save_hyperparameters()

        # Model components
        num_envs = 16
        env_name = "CartPole-v0"
        envs = [self.make_env(env_name) for _ in range(num_envs)]
        self.envs_pool = SubprocVecEnv(envs)
        self.env = gym.make(env_name)

        num_inputs = self.env.observation_space.shape[0]
        num_outputs = self.env.action_space.n

        self.net = self.build_network(num_inputs, num_outputs, hidden=128)
        self.agent = PolicyAgent(self.net)

        # Tracking metrics
        self.total_steps = 0
        self.total_rewards = []
        self.episode_rewards = []
        self.done_episodes = 0
        self.avg_rewards = 0
        self.avg_reward_len = avg_reward_len
        self.eps = np.finfo(np.float32).eps.item()
        self.batch_states = []
        self.batch_actions = []

        self.episode_rewards = 0
        self.batches = 0

        self.state = self.envs_pool.reset()

    def build_network(self, num_inputs, num_outputs, hidden=256):
        model = ActorCritic(num_inputs, num_outputs, hidden)
        return model

    def make_env(self, env_name):
        def _thunk():
            env = gym.make(env_name)
            return env

        return _thunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def train_batch(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader

        Returns:
            yields a tuple of Lists containing tensors for states, actions and rewards of the batch.
        """

        while True:
            batch_states = []
            batch_actions = []
            batch_rewards = []
            rewards = []
            masks = []
            frame_idx = 0

            while frame_idx < self.batch_size:

                for _ in range(self.n_steps):
                    state = torch.FloatTensor(self.state).to(self.device)
                    dist, value = self.net(state)

                    action = dist.sample()
                    next_state, reward, done, _ = self.envs_pool.step(action.cpu().numpy())

                    self.episode_rewards += reward[0]

                    if done[0]:
                        self.total_rewards.append(self.episode_rewards)
                        self.episode_rewards = 0

                    batch_states.append(self.state)
                    batch_actions.append(action.cpu().numpy())
                    rewards.append(reward)
                    masks.append(1-done)

                    self.state = next_state
                    frame_idx += 1


                    if frame_idx >= self.batch_size:
                        break

                    self.total_steps += 1

            self.batches += 1

            for idx in range(len(batch_actions)):
                batch_rewards.extend(self.compute_returns(rewards, masks))
                yield batch_states[idx], batch_actions[idx], batch_rewards[idx]

            if self.batches >= self.batches_per_epoch:
                self.batches = 0
                break

    @staticmethod
    def compute_returns(rewards, masks, gamma=0.99):
        R = 0
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def loss(self, states, actions, scaled_rewards):

        dists, values = self.net(states.float())
        log_probs = dists.log_prob(actions)
        entropy = dists.entropy().mean()

        advantage = scaled_rewards - values.squeeze(dim=-1)

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        return loss, actor_loss, critic_loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """
        states, actions, scaled_rewards = batch

        loss, actor_loss, critic_loss = self.loss(states, actions, scaled_rewards)

        avg_rewards = float(
            np.mean(self.total_rewards[-100:])
        )

        log = {
            "reward": self.total_rewards[-1],
            "batches": self.batches,
            "avg_reward": avg_rewards,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss
        }

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": avg_rewards,
                "log": log,
                "progress_bar": log,
            }
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0][0][0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def add_model_specific_args(arg_parser) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model

        Note: these params are fine tuned for Pong env

        Args:
            arg_parser: the current argument parser to add to

        Returns:
            arg_parser with model specific cargs added
        """

        arg_parser.add_argument(
            "--entropy_beta", type=float, default=0.01, help="entropy value",
        )

        return arg_parser


# todo: covert to CLI func and add test
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = cli.add_base_args(parser)
    parser = A2C.add_model_specific_args(parser)
    args = parser.parse_args()

    model = A2C(**args.__dict__)

    # save checkpoints based on avg_reward
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="avg_reward", mode="max", period=1, verbose=True
    )

    seed_everything(123)
    trainer = pl.Trainer.from_argparse_args(
        args, deterministic=True, checkpoint_callback=checkpoint_callback
    )
    trainer.fit(model)
