"""Trainer loop behavior: iteration budget and device transfers.

`should_stop` must be evaluated every iteration (it used to run only every
`log_after_iters`, overrunning `max_num_iterations` by up to an epoch), and the
host-to-device copy of a loader batch is non-blocking exactly when that batch is
pinned (on unpinned memory the flag is a silent no-op; see
`panseg.functionals.transfer`).
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

BATCH_SHAPE = (1, 1, 8, 16, 16)


def _fixed_batch(seed: int = 0, pin: bool = False):
    g = torch.Generator().manual_seed(seed)
    inp = torch.randn(1, 1, 8, 16, 16, generator=g)
    target = torch.rand(1, 1, 8, 16, 16, generator=g)
    if pin:
        inp, target = inp.pin_memory(), target.pin_memory()
    return inp, target


class _CountingLoss(torch.nn.Module):
    """DiceLoss wrapper recording how many batches were trained/validated on."""

    def __init__(self):
        super().__init__()
        self.inner = DiceLoss()
        self.calls = 0
        self.losses: list[float] = []

    def forward(self, output, target):
        self.calls += 1
        loss = self.inner(output, target)
        self.losses.append(float(loss.detach()))
        return loss


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
    loss = loss if loss is not None else _CountingLoss()
    batch = _fixed_batch(pin=pin)
    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=10
        ),
        loss_criterion=loss,
        loaders={
            "train": [batch] * batches,
            "val": [_fixed_batch(1, pin=pin)] * val_batches,
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

    def test_pinned_batches_copy_non_blocking(
        self, tmp_path, tensor_to_spy, monkeypatch
    ):
        monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)

        calls = self._run(tmp_path, tensor_to_spy)

        assert calls, "the batch transfers must be observed"
        assert all(kw.get("non_blocking") is True for kw in calls)

    def test_unpinned_batches_copy_blocking(self, tmp_path, tensor_to_spy):
        """Without pinning (`pin_memory=False`) the flag must not be set: it
        would only be a silent no-op."""
        calls = self._run(tmp_path, tensor_to_spy)

        assert calls, "the batch transfers must be observed"
        assert all(kw.get("non_blocking") is False for kw in calls)

    @requires_cuda
    def test_trainer_transfers_use_non_blocking_on_cuda(self, tmp_path, tensor_to_spy):
        """The real loader pins its batches on CUDA, so they copy asynchronously
        (the copy is ordered before the CUDA ops that follow on the same stream)."""
        calls = self._run(
            tmp_path, tensor_to_spy, val_batches=2, device="cuda", pin=True
        )

        assert calls
        assert all(kw.get("non_blocking") is True for kw in calls)

    @requires_cuda
    def test_non_blocking_results_match_blocking(self, tmp_path, monkeypatch):
        """The async path must produce exactly the blocking path's losses."""
        real_to = torch.Tensor.to

        def run(force_blocking: bool) -> list[float]:
            def wrapped(self, *args, **kwargs):
                # only override keyword calls: torch internals pass non_blocking
                # positionally, which cannot take the kwarg again
                if force_blocking and len(args) < 3:
                    kwargs["non_blocking"] = False
                return real_to(self, *args, **kwargs)

            monkeypatch.setattr(torch.Tensor, "to", wrapped)
            trainer, loss = _make_trainer(
                tmp_path / f"run_{force_blocking}",
                3,
                1,
                batches=3,
                val_batches=1,
                device="cuda",
                pin=True,
            )
            trainer.train()
            return loss.losses

        blocking = run(True)
        non_blocking = run(False)

        assert len(blocking) == len(non_blocking) >= 3
        for step, (blocking_loss, nb_loss) in enumerate(zip(blocking, non_blocking)):
            assert nb_loss == pytest.approx(blocking_loss, rel=1e-4, abs=1e-5), step
