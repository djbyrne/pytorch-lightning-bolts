import torch
from pytorch_lightning import Callback
from torch import nn


class ConfusedLogitCallback(Callback):  # pragma: no-cover

    def __init__(
            self,
            top_k,
            projection_factor=3,
            min_logit_value=5.0,
            logging_batch_interval=20,
            max_logit_difference=0.1
    ):
        """
        Takes the logit predictions of a model and when the probabilities of two classes are very close, the model
        doesn't have high certainty that it should pick one vs the other class.

        This callback shows how the input would have to change to swing the model from one label prediction
        to the other.

        In this case, the network predicts a 5... but gives almost equal probability to an 8.
        The images show what about the original 5 would have to change to make it more like a 5 or more like an 8.

        For each confused logit the confused images are generated by taking the gradient from a logit wrt an input
        for the top two closest logits.

        Example::

            from pl_bolts.callbacks.vision import ConfusedLogitCallback
            trainer = Trainer(callbacks=[ConfusedLogitCallback()])


        .. note:: whenever called, this model will look for self.last_batch and self.last_logits in the LightningModule

        .. note:: this callback supports tensorboard only right now

        Args:
            top_k: How many "offending" images we should plot
            projection_factor: How much to multiply the input image to make it look more like this logit label
            min_logit_value: Only consider logit values above this threshold
            logging_batch_interval: how frequently to inspect/potentially plot something
            max_logit_difference: when the top 2 logits are within this threshold we consider them confused

        Authored by:

            - Alfredo Canziani

        """
        super().__init__()
        self.top_k = top_k
        self.projection_factor = projection_factor
        self.max_logit_difference = max_logit_difference
        self.logging_batch_interval = logging_batch_interval
        self.min_logit_value = min_logit_value

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # show images only every 20 batches
        if (trainer.batch_idx + 1) % self.logging_batch_interval != 0:
            return

        # pick the last batch and logits
        x, y = pl_module.last_batch
        logits = pl_module.last_logits

        # only check when it has opinions (ie: the logit > 5)
        if logits.max() > self.min_logit_value:
            # pick the top two confused probs
            (values, idxs) = torch.topk(logits, k=2, dim=1)

            # care about only the ones that are at most eps close to each other
            eps = self.max_logit_difference
            mask = (values[:, 0] - values[:, 1]).abs() < eps

            if mask.sum() > 0:
                # pull out the ones we care about
                confusing_x = x[mask, ...]
                confusing_y = y[mask]

                mask_idxs = idxs[mask]

                pl_module.eval()
                self._plot(confusing_x, confusing_y, trainer, pl_module, mask_idxs)
                pl_module.train()

    def _plot(self, confusing_x, confusing_y, trainer, model, mask_idxs):
        from matplotlib import pyplot as plt

        batch_size, c, w, h = confusing_x.size()

        confusing_x = confusing_x[:self.top_k]
        confusing_y = confusing_y[:self.top_k]

        x_param_a = nn.Parameter(confusing_x)
        x_param_b = nn.Parameter(confusing_x)

        for logit_i, x_param in enumerate((x_param_a, x_param_b)):
            logits = model(x_param.view(batch_size, -1))
            logits[:, mask_idxs[:, logit_i]].sum().backward()

        # reshape grads
        grad_a = x_param_a.grad.view(batch_size, w, h)
        grad_b = x_param_b.grad.view(batch_size, w, h)

        for img_i in range(len(confusing_x)):
            x = confusing_x[img_i].squeeze(0)
            y = confusing_y[img_i]
            ga = grad_a[img_i]
            gb = grad_b[img_i]

            mask_idx = mask_idxs[img_i]

            fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
            self.__draw_sample(fig, axarr, 0, 0, x, f'True: {y}')
            self.__draw_sample(fig, axarr, 0, 1, ga, f'd{mask_idx[0]}-logit/dx')
            self.__draw_sample(fig, axarr, 0, 2, gb, f'd{mask_idx[1]}-logit/dx')
            self.__draw_sample(fig, axarr, 1, 1, ga * 2 + x, f'd{mask_idx[0]}-logit/dx')
            self.__draw_sample(fig, axarr, 1, 2, gb * 2 + x, f'd{mask_idx[1]}-logit/dx')

            trainer.logger.experiment.add_figure('confusing_imgs', fig, global_step=trainer.global_step)

    @staticmethod
    def __draw_sample(fig, axarr, row_idx, col_idx, img, title):
        im = axarr[row_idx, col_idx].imshow(img)
        fig.colorbar(im, ax=axarr[row_idx, col_idx])
        axarr[row_idx, col_idx].set_title(title, fontsize=20)
