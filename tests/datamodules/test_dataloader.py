import torch
from torch.utils.data import DataLoader
from pl_bolts.datamodules.cifar10_dataset import CIFAR10
from pl_bolts.datamodules.async_dataloader import AsynchronousLoader


def test_async_dataloader(tmpdir):
    ds = CIFAR10(tmpdir)

    if torch.cuda.device_count() > 0:  # Can only run this test with a GPU
        device = torch.device('cuda', 0)
        dataloader = AsynchronousLoader(ds, device=device)

        for b in dataloader:
            pass

        dataloader = AsynchronousLoader(DataLoader(ds, batch_size=16), device=device)
        for b in dataloader:
            pass
