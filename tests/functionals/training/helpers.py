"""Shared helpers for the training test modules."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import torch

from panseg.functionals.training.losses import DiceLoss


def write_h5(file_path: Path, raw, label, raw_dtype=None, label_dtype=None) -> Path:
    """Write a training H5 file: a `raw` and a `label` dataset.

    Args:
        file_path: destination; parent directories are created.
        raw: the raw volume (ZYX or CZYX).
        label: the channel-less instance label volume (ZYX).
        raw_dtype (optional): dtype to cast `raw` to on the way in.
        label_dtype (optional): dtype to cast `label` to on the way in.

    Returns:
        The written path.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_dtype is not None:
        raw = np.asarray(raw, dtype=raw_dtype)
    if label_dtype is not None:
        label = np.asarray(label, dtype=label_dtype)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("raw", data=raw)
        f.create_dataset("label", data=label)
    return file_path


def model_dir_patches(root: Path) -> list[tuple[object, str, object]]:
    """``(target, attribute, value)`` triples redirecting model-zoo writes to `root`.

    Every module that holds its own reference to a model path must be
    redirected: `unet_training` writes the checkpoints to the training module's
    `PATH_PANSEG_MODELS`, while `model_zoo.add_custom_model` copies the packaged
    files into its own `PATH_PANSEG_MODELS` and appends the record to the custom
    zoo file. Missing one of them writes into the user's `~/.panseg_models`.
    """
    import panseg.core.zoo as zoo_module
    from panseg import FILE_MODEL_ZOO_CUSTOM
    from panseg.core.zoo import model_zoo
    from panseg.functionals.training import train as train_module

    custom_zoo = root / FILE_MODEL_ZOO_CUSTOM
    return [
        (train_module, "PATH_PANSEG_MODELS", root),
        (zoo_module, "PATH_PANSEG_MODELS", root),
        (zoo_module, "PATH_MODEL_ZOO_CUSTOM", custom_zoo),
        (model_zoo, "path_zoo", root),
        (model_zoo, "path_zoo_custom", custom_zoo),
    ]


@dataclass
class MockedTraining:
    """Mocks and paths of a `mocked_unet_training` run."""

    dataset_dir: Path
    checkpoint_dir: Path
    trainer: MagicMock
    find_batch_size: MagicMock
    create_datasets: MagicMock


@contextmanager
def mocked_unet_training(tmp_path: Path, model_name: str) -> Iterator[MockedTraining]:
    """Patch `unet_training`'s training internals for a CPU-only run.

    Creates ``tmp_path/dataset/{train,val}`` and lets the mocked trainer write
    `last_checkpoint.pytorch` and `best_checkpoint.pytorch` from the model it
    was handed, like the real trainer.
    """
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "train").mkdir(parents=True)
    (dataset_dir / "val").mkdir()

    dataset = MagicMock()
    dataset.__len__.return_value = 10
    dataset.raw.shape = (1, 8, 64, 64)
    batch = torch.from_numpy(np.random.rand(1, 1, 1, 64, 64).astype("float32"))
    checkpoint_dir = tmp_path / model_name

    with ExitStack() as stack:
        stack.enter_context(patch("panseg.functionals.training.train.test_model"))
        stack.enter_context(
            patch("panseg.functionals.training.train.make_model_description")
        )
        mock_trainer = stack.enter_context(
            patch("panseg.functionals.training.train.UNetTrainer")
        )
        mock_create_datasets = stack.enter_context(
            patch(
                "panseg.functionals.training.train.create_datasets",
                return_value=[dataset],
            )
        )
        stack.enter_context(
            patch("panseg.functionals.training.train.create_model_config")
        )
        mock_find_batch_size = stack.enter_context(
            patch("panseg.functionals.training.train.find_batch_size", return_value=1)
        )
        stack.enter_context(
            patch(
                "panseg.functionals.training.train.DataLoader",
                return_value=[(batch, None)],
            )
        )
        stack.enter_context(patch("panseg.functionals.training.train.ConcatDataset"))
        for target, attribute, value in model_dir_patches(tmp_path):
            stack.enter_context(patch.object(target, attribute, value))

        def _train() -> None:
            state = mock_trainer.call_args.kwargs["model"].state_dict()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state, checkpoint_dir / "last_checkpoint.pytorch")
            torch.save(state, checkpoint_dir / "best_checkpoint.pytorch")

        mock_trainer.return_value.train.side_effect = _train
        yield MockedTraining(
            dataset_dir=dataset_dir,
            checkpoint_dir=checkpoint_dir,
            trainer=mock_trainer,
            find_batch_size=mock_find_batch_size,
            create_datasets=mock_create_datasets,
        )


def fixed_batch(
    batch_shape, seed: int = 0, pin: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic `(input, target)` batch, optionally pinned host memory."""
    generator = torch.Generator().manual_seed(seed)
    inp = torch.randn(batch_shape, generator=generator)
    target = torch.rand(batch_shape, generator=generator)
    if pin:
        inp, target = inp.pin_memory(), target.pin_memory()
    return inp, target


class RecordingLoss(torch.nn.Module):
    """DiceLoss wrapper recording calls, losses, and output dtypes."""

    def __init__(self):
        super().__init__()
        self.inner = DiceLoss()
        self.calls = 0
        self.losses: list[float] = []
        self.output_dtypes: list[torch.dtype] = []

    def forward(self, output, target):
        self.calls += 1
        self.output_dtypes.append(output.dtype)
        loss = self.inner(output, target)
        self.losses.append(float(loss.detach()))
        return loss
