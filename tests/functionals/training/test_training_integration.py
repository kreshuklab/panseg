"""Integration tests for training."""

import tempfile
from pathlib import Path

import pytest
import torch

from panseg import FILE_MODEL_ZOO_CUSTOM, PATH_PANSEG_MODELS
from panseg.core.zoo import model_zoo
from panseg.functionals.training.train import unet_training


class TestUnetTrainingIntegration:
    """Integration tests for unet_training function using real H5 data."""

    def test_training_integration_3d_cpu(self, mocker, tmp_path):
        """Test actual training."""
        test_data_dir = Path(__file__).parent.parent.parent / "resources" / "data"
        assert test_data_dir.exists(), f"Test data directory not found: {test_data_dir}"

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
    def test_training_integration_3d_gpu(self, mocker, tmp_path):
        """Test actual training on GPU if available."""
        test_data_dir = Path(__file__).parent.parent.parent / "resources" / "data"
        assert test_data_dir.exists(), f"Test data directory not found: {test_data_dir}"

        train_dir = test_data_dir / "train"
        val_dir = test_data_dir / "val"
        assert train_dir.exists(), f"Train directory not found: {train_dir}"
        assert val_dir.exists(), f"Val directory not found: {val_dir}"

        train_files = list(train_dir.glob("*.h5"))
        val_files = list(val_dir.glob("*.h5"))
        assert len(train_files) > 0, f"No training H5 files found in {train_dir}"
        assert len(val_files) > 0, f"No validation H5 files found in {val_dir}"

        model_name = "test_integration_3d_gpu"

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
            feature_maps=16,
            patch_size=(16, 64, 64),
            max_num_iters=100,
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
