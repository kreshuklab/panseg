"""AMP (`amp: auto|off`) behavior of training: autocast, fp32 exports.

Exported artifacts (checkpoint tensors, packaged bioimageIO test tensors)
must remain fp32 regardless of the training precision, and the amp mode that
flows from the training YAML into `unet_training` must reach the batch-size
probe and the trainer alike.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from panseg.functionals.precision import AmpMode
from panseg.functionals.training.model import UNet3D
from panseg.functionals.training.train import unet_training
from panseg.functionals.training.trainer import UNetTrainer
from tests.conftest import requires_cuda
from tests.functionals.training.helpers import (
    RecordingLoss,
    fixed_batch,
    mocked_unet_training,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _run_training(
    amp: AmpMode, checkpoint_dir: Path, steps: int = 3, val_batches: int = 0
) -> tuple[UNetTrainer, RecordingLoss]:
    """Run a few real trainer steps on a fixed batch; returns trainer and recorder."""
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    model = UNet3D(in_channels=1, out_channels=1, f_maps=[4, 4]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    recorder = RecordingLoss()
    inp, target = fixed_batch((2, 1, 16, 16, 16))
    inp, target = inp.to(DEVICE), target.to(DEVICE)
    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=10
        ),
        loss_criterion=recorder,
        loaders={
            "train": [(inp, target)] * steps,
            "val": [(inp, target)] * val_batches,
        },
        checkpoint_dir=checkpoint_dir,
        max_num_iterations=2,
        device=DEVICE,
        log_after_iters=1,
        amp=amp,
    )
    trainer.train()
    return trainer, recorder


class TestUNetTrainingAmpOption:
    """`unet_training` validates the amp option and threads it through."""

    def test_unet_training_rejects_unknown_amp(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "train").mkdir(parents=True)
        (dataset_dir / "val").mkdir()

        with pytest.raises(ValueError, match="amp"):
            unet_training(
                dataset_dir=str(dataset_dir),
                model_name="test_model_bad_amp",
                in_channels=1,
                out_channels=1,
                feature_maps=[4, 4],
                patch_size=(1, 64, 64),
                max_num_iters=5,
                dimensionality="2D",
                sparse=False,
                device="cpu",
                amp="sometimes",  # pyright: ignore[reportArgumentType] -- deliberately invalid
            )

    @staticmethod
    def _run_mocked(tmp_path, model_name: str, amp: AmpMode):
        """Run `unet_training` with mocked training internals; returns the mocks."""
        with mocked_unet_training(tmp_path, model_name) as mocked:
            unet_training(
                dataset_dir=str(mocked.dataset_dir),
                model_name=model_name,
                in_channels=1,
                out_channels=1,
                feature_maps=(16, 16),
                patch_size=(1, 64, 64),
                max_num_iters=100,
                dimensionality="2D",
                sparse=False,
                device="cpu",
                amp=amp,
            )
        return mocked

    def test_unet_training_flows_amp_to_batch_probe_and_trainer(self, tmp_path):
        """The batch-size probe and the trainer must share the configured mode."""
        mocked = self._run_mocked(tmp_path, "test_model_amp_flow", "off")

        assert mocked.find_batch_size.call_args.kwargs["amp"] == "off"
        assert mocked.trainer.call_args.kwargs["amp"] == "off"

    def test_unet_training_test_tensors_are_fp32(self, tmp_path):
        """The exported tensors are fp32: a dtype contract, not an AMP claim.

        The export model is a fresh CPU instance, so the tensors say nothing
        about the training precision.
        """
        mocked = self._run_mocked(tmp_path, "test_model_amp_export", "auto")

        assert np.load(mocked.checkpoint_dir / "test_in.npy").dtype == np.float32
        assert np.load(mocked.checkpoint_dir / "test_out.npy").dtype == np.float32


@requires_cuda
class TestUNetTrainerAmp:
    """Real trainer steps on CUDA: autocast engaged, exports fp32, losses close."""

    def test_trainer_forward_runs_in_the_resolved_precision_mode(self, tmp_path):
        cases: list[tuple[AmpMode, torch.dtype]] = [
            ("auto", torch.bfloat16),
            ("off", torch.float32),
        ]
        for amp, expected_dtype in cases:
            _, recorder = _run_training(amp, tmp_path / amp, steps=1)
            assert recorder.output_dtypes[0] == expected_dtype, amp

    def test_checkpoint_tensors_stay_fp32_with_amp_on(self, tmp_path):
        _run_training("auto", tmp_path)

        state = torch.load(
            tmp_path / "last_checkpoint.pytorch", map_location="cpu", weights_only=True
        )
        assert state
        for name, tensor in state.items():
            assert tensor.dtype == torch.float32, name

    def test_losses_close_between_amp_auto_and_off(self, tmp_path):
        """The val loop shares the training precision mode (via `_forward_pass`)."""
        trainer_off, losses_off = _run_training("off", tmp_path / "off", val_batches=1)
        trainer_auto, losses_auto = _run_training(
            "auto", tmp_path / "auto", val_batches=1
        )

        assert len(losses_auto.losses) == len(losses_off.losses) >= 2
        for step, (auto, off) in enumerate(zip(losses_auto.losses, losses_off.losses)):
            assert auto == pytest.approx(off, rel=5e-2, abs=5e-2), f"step {step}"
        assert trainer_auto.best_eval_loss == pytest.approx(
            trainer_off.best_eval_loss, rel=5e-2, abs=5e-2
        )
