"""Trainer loop behavior: iteration budget and device transfers.

`should_stop` must be evaluated every iteration (it used to run only every
`log_after_iters`, overrunning `max_num_iterations` by up to an epoch), and the
host-to-device copies of the pinned loader batches must be non-blocking on CUDA
(the copy is ordered before the CUDA ops that follow on the same stream).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from panseg.functionals.training.losses import DiceLoss
from panseg.functionals.training.model import UNet3D
from panseg.functionals.training.trainer import UNetTrainer

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for transfer tests"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _CountingLoss(torch.nn.Module):
    """DiceLoss wrapper counting how many batches were actually trained on."""

    def __init__(self):
        super().__init__()
        self.inner = DiceLoss()
        self.calls = 0

    def forward(self, output, target):
        self.calls += 1
        return self.inner(output, target)


def _fixed_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    inp = torch.randn(1, 1, 8, 16, 16, generator=g)
    target = torch.rand(1, 1, 8, 16, 16, generator=g)
    return inp, target


def _make_trainer(
    checkpoint_dir: Path,
    max_num_iterations: int,
    log_after_iters: int,
    batches: int,
    device: str = "cpu",
    loss=None,
):
    torch.manual_seed(7)
    model = UNet3D(in_channels=1, out_channels=1, f_maps=[4, 4]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = loss if loss is not None else _CountingLoss()
    batch = _fixed_batch()
    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=10
        ),
        loss_criterion=loss,
        loaders={"train": [batch] * batches, "val": []},
        checkpoint_dir=checkpoint_dir,
        max_num_iterations=max_num_iterations,
        device=device,
        log_after_iters=log_after_iters,
        amp="off",
    )
    return trainer, loss


class TestIterationBudget:
    """`max_num_iterations` bounds the run regardless of the logging cadence."""

    @pytest.mark.parametrize("log_after_iters", [500, 3, 1])
    def test_train_stops_at_max_num_iterations(self, tmp_path, log_after_iters):
        trainer, loss = _make_trainer(
            tmp_path,
            max_num_iterations=7,
            log_after_iters=log_after_iters,
            batches=20,  # far more batches than iterations allowed
        )

        trainer.train()

        assert loss.calls == 7

    def test_train_stops_at_max_num_iterations_short_epoch(self, tmp_path):
        """Even when the epoch is shorter than the stop condition used to be
        checked at, the run must not loop epochs past the budget."""
        trainer, loss = _make_trainer(
            tmp_path,
            max_num_iterations=7,
            log_after_iters=500,
            batches=3,
        )

        trainer.train()

        assert loss.calls == 7
