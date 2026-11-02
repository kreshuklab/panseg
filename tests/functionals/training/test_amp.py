"""AMP (`amp: auto|off`) behavior of training: autocast, fp32 exports.

Exported artifacts (checkpoint tensors, packaged bioimageIO test tensors)
must remain fp32 regardless of the training precision, and the amp mode that
flows from the training YAML into `unet_training` must reach the batch-size
probe and the trainer alike.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from panseg.core.zoo import model_zoo
from panseg.functionals.training.losses import DiceLoss
from panseg.functionals.training.model import UNet3D
from panseg.functionals.training.train import unet_training
from panseg.functionals.training.trainer import UNetTrainer

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for AMP tests"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _RecordingLoss(torch.nn.Module):
    """DiceLoss wrapper recording the dtype of every output it is asked to score."""

    def __init__(self):
        super().__init__()
        self.inner = DiceLoss()
        self.output_dtypes: list[torch.dtype] = []
        self.losses: list[float] = []

    def forward(self, output, target):
        self.output_dtypes.append(output.dtype)
        loss = self.inner(output, target)
        self.losses.append(float(loss.detach()))
        return loss


def _fixed_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    inp = torch.randn(2, 1, 16, 16, 16, generator=g)
    target = torch.rand(2, 1, 16, 16, 16, generator=g)
    return inp, target


def _run_training(amp: str, checkpoint_dir: Path, steps: int = 3) -> _RecordingLoss:
    """Run a few real trainer steps on a fixed batch; returns the loss recorder."""
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    model = UNet3D(in_channels=1, out_channels=1, f_maps=[4, 4]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    recorder = _RecordingLoss()
    inp, target = _fixed_batch()
    inp, target = inp.to(DEVICE), target.to(DEVICE)
    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=10
        ),
        loss_criterion=recorder,
        loaders={"train": [(inp, target)] * steps, "val": []},
        checkpoint_dir=checkpoint_dir,
        max_num_iterations=2,
        device=DEVICE,
        log_after_iters=1,
        amp=amp,
    )
    trainer.train()
    return recorder


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
                amp="sometimes",
            )

    @staticmethod
    def _write_checkpoints(mock_trainer_class, checkpoint_dir):
        """Let the mocked UNetTrainer leave checkpoint files behind like the real one."""

        def _train():
            model = mock_trainer_class.call_args.kwargs["model"]
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state = model.state_dict()
            torch.save(state, checkpoint_dir / "last_checkpoint.pytorch")
            torch.save(state, checkpoint_dir / "best_checkpoint.pytorch")

        mock_trainer_class.return_value.train.side_effect = _train

    def _run_mocked_unet_training(self, tmp_path, model_name: str, amp: str):
        """Run `unet_training` with mocked training internals; returns the mocks."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        batch = torch.from_numpy(np.random.rand(1, 1, 1, 64, 64).astype("float32"))

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "train").mkdir(parents=True)
        (dataset_dir / "val").mkdir()

        with (
            patch("panseg.functionals.training.train.test_model"),
            patch("panseg.functionals.training.train.make_model_description"),
            patch("panseg.functionals.training.train.UNetTrainer") as mock_trainer,
            patch(
                "panseg.functionals.training.train.create_datasets",
                return_value=[mock_dataset],
            ),
            patch("panseg.functionals.training.train.create_model_config"),
            patch(
                "panseg.functionals.training.train.find_batch_size", return_value=1
            ) as mock_find_batch_size,
            patch(
                "panseg.functionals.training.train.DataLoader",
                return_value=[(batch, None)],
            ),
            patch("panseg.functionals.training.train.ConcatDataset"),
            patch("panseg.functionals.training.train.PATH_PANSEG_MODELS", tmp_path),
            patch("panseg.core.zoo.PATH_PANSEG_MODELS", tmp_path),
            # keep the zoo records of this run out of the user's custom zoo
            patch.multiple(
                model_zoo,
                path_zoo=tmp_path / "models_zoo.yaml",
                path_zoo_custom=tmp_path / "custom_zoo.yaml",
            ),
        ):
            self._write_checkpoints(mock_trainer, tmp_path / model_name)
            unet_training(
                dataset_dir=str(dataset_dir),
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
        return mock_trainer, mock_find_batch_size, tmp_path / model_name

    def test_unet_training_flows_amp_to_batch_probe_and_trainer(self, tmp_path):
        """The batch-size probe and the trainer must share the configured mode."""
        mock_trainer, mock_find_batch_size, _ = self._run_mocked_unet_training(
            tmp_path, model_name="test_model_amp_flow", amp="off"
        )

        assert mock_find_batch_size.call_args.kwargs["amp"] == "off"
        assert mock_trainer.call_args.kwargs["amp"] == "off"

    def test_unet_training_test_tensors_stay_fp32_with_amp(self, tmp_path):
        """The packaged bioimageIO test tensors are fp32 whatever the amp mode."""
        _, _, checkpoint_dir = self._run_mocked_unet_training(
            tmp_path, model_name="test_model_amp_export", amp="auto"
        )

        assert np.load(checkpoint_dir / "test_in.npy").dtype == np.float32
        assert np.load(checkpoint_dir / "test_out.npy").dtype == np.float32


@requires_cuda
class TestUNetTrainerAmp:
    """Real trainer steps on CUDA: autocast engaged, exports fp32, losses close."""

    def test_trainer_forward_runs_in_the_resolved_precision_mode(self, tmp_path):
        for amp, expected_dtype in [
            ("auto", torch.bfloat16),
            ("off", torch.float32),
        ]:
            recorder = _run_training(amp, tmp_path / amp, steps=1)
            assert recorder.output_dtypes[0] == expected_dtype, amp

    def test_checkpoint_tensors_stay_fp32_with_amp_on(self, tmp_path):
        _run_training("auto", tmp_path)

        state = torch.load(
            tmp_path / "last_checkpoint.pytorch", map_location="cpu", weights_only=True
        )
        assert state
        for name, tensor in state.items():
            assert tensor.dtype == torch.float32, name

    def test_training_losses_close_between_amp_auto_and_off(self, tmp_path):
        losses_off = _run_training("off", tmp_path / "off").losses
        losses_auto = _run_training("auto", tmp_path / "auto").losses

        assert len(losses_auto) == len(losses_off) >= 2
        for step, (auto, off) in enumerate(zip(losses_auto, losses_off)):
            assert auto == pytest.approx(off, rel=5e-2, abs=5e-2), f"step {step}"

    def test_validation_loss_close_between_amp_auto_and_off(self, tmp_path):
        """The val loop shares the training precision mode (via _forward_pass)."""

        def _run_with_val(amp: str, checkpoint_dir: Path) -> float:
            torch.manual_seed(7)
            torch.cuda.manual_seed_all(7)
            model = UNet3D(in_channels=1, out_channels=1, f_maps=[4, 4]).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            inp, target = _fixed_batch()
            inp, target = inp.to(DEVICE), target.to(DEVICE)
            trainer = UNetTrainer(
                model=model,
                optimizer=optimizer,
                lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.2, patience=10
                ),
                loss_criterion=DiceLoss(),
                loaders={"train": [(inp, target)], "val": [(inp, target)]},
                checkpoint_dir=checkpoint_dir,
                max_num_iterations=2,
                device=DEVICE,
                log_after_iters=1,
                amp=amp,
            )
            trainer.train()
            return trainer.best_eval_loss

        val_off = _run_with_val("off", tmp_path / "off")
        val_auto = _run_with_val("auto", tmp_path / "auto")

        assert val_auto == pytest.approx(val_off, rel=5e-2, abs=5e-2)
