import argparse
from collections import OrderedDict
from typing import Tuple, List
import numpy as np

import torch
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.functional import log_softmax, softmax, mse_loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.datamodules.experience_source import DiscountedExperienceSource
from pl_bolts.models.rl.common import cli
from pl_bolts.models.rl.common.agents import PolicyAgent
from pl_bolts.models.rl.common.networks import MLP, CNNActorCritic
from pl_bolts.models.rl.common.wrappers import make_env


class A2C(pl.LightningModule):
    def __init__(
        self,
        env: str,
        gamma: float = 0.99,
        lr: float = 0.01,
        batch_size: int = 128,
        n_steps: int = 4,
        avg_reward_len: int = 100,
        num_envs: int = 50,
        entropy_beta: float = 0.01,
        epoch_len: int = 1000,
        **kwargs
    ) -> None:
        """
        PyTorch Lightning implementation of `Vanilla Policy Gradient
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

        Note:
            This example is based on:
            https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter11/04_cartpole_pg.py

        Note:
            Currently only supports CPU and single GPU training with `distributed_backend=dp`
        """
        super().__init__()

        # Hyperparameters
        self.lr = 0.001
        self.batch_size = 128
        self.batches_per_epoch = self.batch_size * epoch_len
        self.entropy_beta = 0.01
        self.gamma = 0.99
        self.n_steps = 4

        self.save_hyperparameters()

        # Model components
        self.env = [make_env(env) for _ in range(30)]
        self.net = CNNActorCritic(self.env[0].observation_space.shape, self.env[0].action_space.n)
        self.agent = PolicyAgent(self.net)
        self.exp_source = DiscountedExperienceSource(
            self.env, self.agent, gamma=gamma, n_steps=self.n_steps
        )

        # Tracking metrics
        self.total_steps = 0
        self.total_rewards = [0]
        self.done_episodes = 0
        self.avg_rewards = 0
        self.reward_sum = 0.0
        self.baseline = 0
        self.avg_reward_len = avg_reward_len
        self.optimizer = None

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

        states = []
        actions = []
        rewards = []
        not_done_idx = []
        last_states = []
        local_count = 0

        for step_idx, exp in enumerate(self.exp_source.runner(self.device)):

            # self.reward_sum += exp.reward
            # self.baseline = self.reward_sum / (self.total_steps + 1)
            # scaled_reward = exp.reward - self.baseline
            #
            # new_rewards = self.exp_source.pop_total_rewards()
            # if new_rewards:
            #     for reward in new_rewards:
            #         self.done_episodes += 1
            #         self.total_rewards.append(reward)
            #         self.avg_rewards = float(
            #             np.mean(self.total_rewards[-self.avg_reward_len:])
            #         )
            #
            # yield exp.state, exp.action, scaled_reward

            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            if not exp.done:
                not_done_idx.append(local_count)
                last_states.append(exp.new_state)

            new_rewards = self.exp_source.pop_total_rewards()
            if new_rewards:
                for reward in new_rewards:
                    self.done_episodes += 1
                    self.total_rewards.append(reward)
                    self.avg_rewards = float(
                        np.mean(self.total_rewards[-self.avg_reward_len:])
                    )

            self.total_steps += 1
            local_count += 1

            if local_count == self.batch_size:

                if not_done_idx:
                    rewards_np = np.array(rewards, dtype=np.float32)
                    last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(self.device)
                    last_vals_v = self.net(last_states_v)[1]
                    last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
                    last_vals_np *= self.gamma ** self.n_steps
                    rewards_np[not_done_idx] += last_vals_np

                for idx in range(len(states)):
                    yield states[idx], actions[idx], rewards_np[idx]

                states = []
                actions = []
                rewards = []
                not_done_idx = []
                last_states = []
                local_count = 0

            if self.total_steps % self.batches_per_epoch == 0:
                break

    def loss(self, states, actions, scaled_rewards):
        self.optimizer.zero_grad()

        # logits, values = self.net(states)
        #
        # # value loss
        # value_loss = mse_loss(values.squeeze(-1), scaled_rewards.float())
        # # value_loss = nn.MSELoss()(values.squeeze(-1), scaled_rewards)
        #
        # # policy loss
        # log_prob = log_softmax(logits, dim=1)
        # advantage = scaled_rewards - values.detach()
        # log_prob_actions = advantage * log_prob[range(self.batch_size), actions]
        # policy_loss = -log_prob_actions.mean()
        #
        # # entropy loss
        # prob = softmax(logits, dim=1)
        # entropy = (prob * log_prob).sum(dim=1).mean()
        # entropy_loss = self.entropy_beta * entropy
        #
        # # calc policy gradients only
        # # policy_loss.backward(retain_graph=True)
        # # grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
        # #                         for p in self.net.parameters()
        # #                         if p.grad is not None])
        #
        # # add entropy and value gradients
        # loss = entropy_loss + value_loss + policy_loss
        #
        # # total loss
        # total_loss = loss
        #
        # return loss, total_loss, None

        logits_v, value_v = self.net(states)
        loss_value_v = mse_loss(value_v.squeeze(-1), scaled_rewards.float())

        log_prob_v = log_softmax(logits_v, dim=1)
        adv_v = scaled_rewards - value_v.detach()
        log_prob_actions_v = adv_v * log_prob_v[range(self.batch_size), actions]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = softmax(logits_v, dim=1)
        entropy_loss_v = self.entropy_beta * (prob_v * log_prob_v).sum(dim=1).mean()

        # calculate policy gradients only
        loss_policy_v.backward(retain_graph=True)
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in self.net.parameters()
                                if p.grad is not None])

        # apply entropy and value gradients
        loss = entropy_loss_v + loss_value_v
        # get full loss
        total_loss = loss + loss_policy_v

        return loss, total_loss, grads

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

        loss, total_loss, grads = self.loss(states, actions, scaled_rewards)

        log = {
            "episodes": self.done_episodes,
            "reward": self.total_rewards[-1],
            "avg_reward": self.avg_rewards,
            "baseline": self.baseline,
            "steps": self.total_steps
        }
        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_rewards,
                "log": log,
                "progress_bar": log,
            }
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-3)
        return [self.optimizer]

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
        args, deterministic=True, checkpoint_callback=checkpoint_callback, gradient_clip_val=0.1
    )
    trainer.fit(model)
