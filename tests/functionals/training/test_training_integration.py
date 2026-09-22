"""Integration tests for training."""

import h5py
import numpy as np
import pytest
import torch

from panseg import FILE_MODEL_ZOO_CUSTOM, PATH_PANSEG_MODELS
from panseg.core.zoo import model_zoo
from panseg.functionals.training.train import unet_training


def _write_h5_dataset(
    file_path, volume_size: int = 64, target_label_frac: float = 0.13, seed: int = 0
):
    """Write a single H5 file with ``raw`` (uint8 noise) and ``label`` (uint8
    ellipsoid instances) datasets.

    The label density is kept near the real mini files (~12-15 % nonzero):
    below ~10 % the slice filter drops all patches and the pipeline crashes.
    """
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(volume_size,) * 3).astype("uint8")
    label = np.zeros((volume_size,) * 3, dtype="uint8")
    zyx = np.ogrid[0:volume_size, 0:volume_size, 0:volume_size]
    total = volume_size**3
    instance_id = 1
    attempts = 0
    while np.count_nonzero(label) / total < target_label_frac:
        attempts += 1
        assert attempts < 10000, "Could not reach target label density"
        center = rng.integers(6, volume_size - 6, size=3)
        radii = rng.integers(3, 6, size=3)
        ellipsoid = (
            ((zyx[0] - center[0]) / radii[0]) ** 2
            + ((zyx[1] - center[1]) / radii[1]) ** 2
            + ((zyx[2] - center[2]) / radii[2]) ** 2
        ) <= 1
        if label[ellipsoid].any():
            continue
        label[ellipsoid] = instance_id
        instance_id += 1
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("raw", data=raw)
        f.create_dataset("label", data=label)


@pytest.fixture
def generated_dataset_dir(tmp_path):
    """Create a tiny self-generated H5 dataset (2 train + 1 val file, 64^3 each)."""
    data_dir = tmp_path / "data"
    for idx, seed in enumerate((0, 1)):
        _write_h5_dataset(data_dir / "train" / f"train_{idx:03d}.h5", seed=seed)
    _write_h5_dataset(data_dir / "val" / "val_000.h5", seed=2)
    return data_dir


class TestUnetTrainingIntegration:
    """Integration tests for unet_training function using a self-generated H5 dataset."""

    def test_training_integration_3d_cpu(self, mocker, generated_dataset_dir):
        """Test actual training."""
        test_data_dir = generated_dataset_dir
        train_dir = test_data_dir / "train"
        val_dir = test_data_dir / "val"
        assert train_dir.exists(), f"Train directory not found: {train_dir}"
        assert val_dir.exists(), f"Val directory not found: {val_dir}"

        train_files = list(train_dir.glob("*.h5"))
        val_files = list(val_dir.glob("*.h5"))
        assert len(train_files) > 0, f"No training H5 files found in {train_dir}"
        assert len(val_files) > 0, f"No validation H5 files found in {val_dir}"

        model_name = "test_integration_3d_cpu"
        # model should not be saved in the users models directory
        assert not (PATH_PANSEG_MODELS / model_name).exists()

        tmp_path = test_data_dir.parent
        mocker.patch(
            "panseg.functionals.training.train.PATH_PANSEG_MODELS",
            tmp_path,
        )
        mocker.patch.multiple(
            "panseg.core.zoo",
            PATH_PANSEG_MODELS=tmp_path,
            PATH_MODEL_ZOO_CUSTOM=tmp_path / FILE_MODEL_ZOO_CUSTOM,
        )
        mocker.patch.multiple(
            model_zoo,
            path_zoo=tmp_path,
            path_zoo_custom=tmp_path / FILE_MODEL_ZOO_CUSTOM,
        )
        unet_training(
            dataset_dir=str(test_data_dir),
            model_name=model_name,
            in_channels=1,
            out_channels=1,
            feature_maps=[2, 2],
            patch_size=(16, 64, 64),
            max_num_iters=5,
            dimensionality="3D",
            sparse=True,
            device="cpu",
        )

        model_dir = tmp_path / model_name
        assert model_dir.exists(), f"Model directory not created: {model_dir}"

        checkpoint_files = list(model_dir.glob("*.pytorch"))
        assert len(checkpoint_files) > 0, "No checkpoint files created"

        config_file = model_dir / "config_train.yml"
        assert config_file.exists(), f"Config file not created: {config_file}"

        assert (model_dir / "test_in.npy").exists()
        assert (model_dir / "test_out.npy").exists()

        mocker.stopall()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_integration_3d_gpu(self, mocker, generated_dataset_dir):
        """Test actual training on GPU if available."""
        test_data_dir = generated_dataset_dir
        train_dir = test_data_dir / "train"
        val_dir = test_data_dir / "val"
        assert train_dir.exists(), f"Train directory not found: {train_dir}"
        assert val_dir.exists(), f"Val directory not found: {val_dir}"

        train_files = list(train_dir.glob("*.h5"))
        val_files = list(val_dir.glob("*.h5"))
        assert len(train_files) > 0, f"No training H5 files found in {train_dir}"
        assert len(val_files) > 0, f"No validation H5 files found in {val_dir}"

        model_name = "test_integration_3d_gpu"

        tmp_path = test_data_dir.parent
        mocker.patch(
            "panseg.functionals.training.train.PATH_PANSEG_MODELS",
            tmp_path,
        )
        mocker.patch.multiple(
            "panseg.core.zoo",
            PATH_PANSEG_MODELS=tmp_path,
            PATH_MODEL_ZOO_CUSTOM=tmp_path / FILE_MODEL_ZOO_CUSTOM,
        )
        mocker.patch.multiple(
            model_zoo,
            path_zoo=tmp_path,
            path_zoo_custom=tmp_path / FILE_MODEL_ZOO_CUSTOM,
        )
        unet_training(
            dataset_dir=str(test_data_dir),
            model_name=model_name,
            in_channels=1,
            out_channels=1,
            feature_maps=[16, 16],
            patch_size=(16, 64, 64),
            max_num_iters=2,
            dimensionality="3D",
            sparse=False,
            device="cuda",
        )

        model_dir = tmp_path / model_name
        assert model_dir.exists(), f"Model directory not created: {model_dir}"

        assert (model_dir / "test_in.npy").exists()
        assert (model_dir / "test_out.npy").exists()

        mocker.stopall()
        assert not (PATH_PANSEG_MODELS / model_name).exists()
