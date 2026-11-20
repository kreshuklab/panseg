"""Trainer loop behavior: iteration budget and device transfers.

`should_stop` must be evaluated every iteration (it used to run only every
`log_after_iters`, overrunning `max_num_iterations` by up to an epoch), and the
host-to-device copy of a loader batch is non-blocking exactly when that batch is
pinned (on unpinned memory the flag is a silent no-op; see
`panseg.functionals.transfer`).
"""

from pathlib import Path

import pytest
import torch

from panseg.functionals.training.model import UNet3D
from panseg.functionals.training.trainer import UNetTrainer
from tests.conftest import requires_cuda
from tests.functionals.training.helpers import RecordingLoss, fixed_batch

BATCH_SHAPE = (1, 1, 8, 16, 16)


def _make_trainer(
    checkpoint_dir: Path,
    max_num_iterations: int,
    log_after_iters: int,
    batches: int,
    device: str = "cpu",
    loss=None,
    val_batches: int = 0,
    pin: bool = False,
):
    torch.manual_seed(7)
    model = UNet3D(in_channels=1, out_channels=1, f_maps=[4, 4]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = loss if loss is not None else RecordingLoss()
    batch = fixed_batch(BATCH_SHAPE, pin=pin)
    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=10
        ),
        loss_criterion=loss,
        loaders={
            "train": [batch] * batches,
            "val": [fixed_batch(BATCH_SHAPE, seed=1, pin=pin)] * val_batches,
        },
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


class TestHostToDeviceTransfers:
    """Batches copy non-blocking exactly when they are pinned."""

    @staticmethod
    def _run(tmp_path, tensor_to_spy, val_batches=1, **trainer_kwargs) -> list[dict]:
        trainer, loss = _make_trainer(
            tmp_path, 2, 1, batches=2, val_batches=val_batches, **trainer_kwargs
        )
        trainer.train()
        # 2 train + `val_batches` val batches reached the loss
        assert loss.calls == 2 + val_batches
        return [kw for shape, _, kw in tensor_to_spy if shape == BATCH_SHAPE]

    @pytest.mark.parametrize("pinned", [True, False])
    def test_batches_copy_non_blocking_exactly_when_pinned(
        self, tmp_path, tensor_to_spy, monkeypatch, pinned
    ):
        """Without pinning (`pin_memory=False`) the flag must not be set: it
        would only be a silent no-op."""
        if pinned:
            monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)

        calls = self._run(tmp_path, tensor_to_spy)

        assert calls, "the batch transfers must be observed"
        assert all(kw.get("non_blocking") is pinned for kw in calls)

    @requires_cuda
    def test_trainer_transfers_use_non_blocking_on_cuda(self, tmp_path, tensor_to_spy):
        """The real loader pins its batches on CUDA, so they copy asynchronously
        (the copy is ordered before the CUDA ops that follow on the same stream)."""
        calls = self._run(
            tmp_path, tensor_to_spy, val_batches=2, device="cuda", pin=True
        )

        assert calls
        assert all(kw.get("non_blocking") is True for kw in calls)
