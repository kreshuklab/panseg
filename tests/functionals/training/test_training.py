"""Unit tests for training functionality."""

import tempfile
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch
import yaml

from panseg import FILE_CONFIG_TRAIN_YAML
from panseg.functionals.training.augs import Augmenter
from panseg.functionals.training.h5dataset import HDF5Dataset, resolve_dataset_layouts
from panseg.functionals.training.losses import DiceLoss
from panseg.functionals.training.model import UNet2D, UNet3D
from panseg.functionals.training.train import (
    create_datasets,
    create_model_config,
    find_h5_files,
    unet_training,
    zero_mean_unit_variance,
)
from panseg.functionals.training.trainer import UNetTrainer, load_pretrained_weights


def _write_checkpoints(mock_trainer_class, checkpoint_dir, make_states=None):
    """Let the mocked UNetTrainer leave checkpoint files behind like the real one.

    `unet_training` loads the packaged weights to generate the bioimageio test
    tensors, so a checkpoint has to exist after `train()` returns.
    """

    def _train():
        model = mock_trainer_class.call_args.kwargs["model"]
        if make_states is None:
            states = {
                "best_checkpoint.pytorch": model.state_dict(),
                "last_checkpoint.pytorch": model.state_dict(),
            }
        else:
            states = make_states(model)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for name, state in states.items():
            torch.save(state, checkpoint_dir / name)

    mock_trainer_class.return_value.train.side_effect = _train


def _dense_label(shape):
    """Instance label with >= 50 % nonzero in every z-slice.

    `FilterSliceBuilder` drops patches with < 10 % nonzero labels (plus 1 %
    seeded random slack). With this fixture every candidate patch survives, so
    patch counts equal the plain `SliceBuilder` counts exactly -- deterministic,
    no filter bookkeeping needed in the assertions.
    """
    label = np.zeros(shape, dtype="uint8")
    label[..., shape[-2] // 2 :, :] = 1
    label[..., : shape[-2] // 2, : shape[-2] // 2] = 2
    return label


def _write_layout_h5(file_path, raw, label, axis_orders=None):
    """Write an h5 file with raw/label datasets and optional `axis_order` attrs.

    `axis_orders` maps dataset name ("raw"/"label") to a layout string, e.g.
    {"raw": "CZYX", "label": "ZYX"}, written as the dataset attr `axis_order`
    the way `panseg/io/h5.py` writes it on export.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as f:
        datasets = {
            "raw": f.create_dataset("raw", data=raw),
            "label": f.create_dataset("label", data=label),
        }
        for name, order in (axis_orders or {}).items():
            datasets[name].attrs["axis_order"] = order


class TestFindH5Files:
    """Tests for find_h5_files function."""

    def test_find_h5_files_with_valid_directory(self):
        """Test finding h5 files in a valid directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files with different extensions
            test_files = [
                temp_path / "test1.h5",
                temp_path / "test2.hdf",
                temp_path / "test3.hdf5",
                temp_path / "test4.hd5",
                temp_path / "test5.txt",  # Should not be found
            ]

            for file in test_files:
                file.touch()

            found_files = find_h5_files(temp_path)

            # Should find 4 h5-type files
            assert len(found_files) == 4

            # Check that only h5-type files are found
            found_names = [f.name for f in found_files]
            assert "test1.h5" in found_names
            assert "test2.hdf" in found_names
            assert "test3.hdf5" in found_names
            assert "test4.hd5" in found_names
            assert "test5.txt" not in found_names

    def test_find_h5_files_empty_directory(self):
        """Test finding h5 files in an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            found_files = find_h5_files(temp_dir)
            assert len(found_files) == 0

    def test_find_h5_files_invalid_directory(self):
        """Test finding h5 files with invalid directory."""
        with pytest.raises(AssertionError):
            find_h5_files("/non/existent/directory")


class TestCreateModelConfig:
    """Tests for create_model_config function."""

    def test_create_model_config_2d(self):
        """Test creating model config for 2D UNet."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "test_checkpoint"

            create_model_config(
                checkpoint_dir=checkpoint_dir,
                in_channels=1,
                out_channels=2,
                patch_size=[64, 64],
                dimensionality="2D",
                layer_order="bcr",
                sparse=False,
                f_maps=[16, 32, 64],
                max_num_iters=1000,
            )

            # Check that directory was created
            assert checkpoint_dir.exists()

            # Check that config file was created
            config_path = checkpoint_dir / FILE_CONFIG_TRAIN_YAML
            assert config_path.exists()

            # Check config content
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            assert config["model"]["in_channels"] == 1
            assert config["model"]["out_channels"] == 2
            assert config["model"]["f_maps"] == [16, 32, 64]
            assert config["model"]["name"] == "UNet2D"
            assert config["model"]["final_sigmoid"] is True  # not sparse
            assert config["trainer"]["checkpoint_dir"] == str(checkpoint_dir)
            assert config["trainer"]["max_num_iterations"] == 1000

    def test_create_model_config_3d(self):
        """Test creating model config for 3D UNet."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "test_checkpoint"

            create_model_config(
                checkpoint_dir=checkpoint_dir,
                in_channels=1,
                out_channels=3,
                patch_size=[32, 64, 64],
                dimensionality="3D",
                layer_order="bcr",
                sparse=True,
                f_maps=[8, 16, 32],
                max_num_iters=2000,
            )

            config_path = checkpoint_dir / FILE_CONFIG_TRAIN_YAML
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            assert config["model"]["name"] == "UNet3D"
            assert config["model"]["final_sigmoid"] is False  # sparse
            assert config["trainer"]["max_num_iterations"] == 2000
            assert config["loaders"]["train"]["slice_builder"]["patch_shape"] == [
                32,
                64,
                64,
            ]
            assert config["loaders"]["train"]["slice_builder"]["stride_shape"] == [
                16,
                32,
                32,
            ]

    def test_create_model_config_2d_per_slice_stride(self):
        """The train stride keeps the z-extent at 1 for per-slice 2D training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "test_checkpoint"

            create_model_config(
                checkpoint_dir=checkpoint_dir,
                in_channels=1,
                out_channels=1,
                patch_size=[1, 64, 64],
                dimensionality="2D",
                layer_order="bcr",
                sparse=True,
                f_maps=[2, 2],
                max_num_iters=5,
            )

            config_path = checkpoint_dir / FILE_CONFIG_TRAIN_YAML
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            assert config["model"]["name"] == "UNet2D"
            assert config["loaders"]["train"]["slice_builder"]["patch_shape"] == [
                1,
                64,
                64,
            ]
            # max(i // 2, 1): z stride is 1, never 0
            assert config["loaders"]["train"]["slice_builder"]["stride_shape"] == [
                1,
                32,
                32,
            ]


class TestCreateDatasets:
    """Tests for create_datasets function."""

    def test_create_datasets_train_3D(self):
        """Test creating training datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_dir = temp_path / "train"
            train_dir.mkdir()

            # Create dummy h5 files
            for i in range(3):
                h5_file = train_dir / f"data_{i}.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("raw", data=np.random.rand(10, 88, 88))
                    f.create_dataset("label", data=np.random.rand(10, 88, 88))

            datasets = create_datasets(str(temp_path), "train", (8, 64, 64), "3D")

            assert len(datasets) == 3
            assert len(datasets[0]) == 8

            for dataset in datasets:
                assert isinstance(dataset, HDF5Dataset)

    def test_create_datasets_train_3Dc(self):
        """Test creating training datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_dir = temp_path / "train"
            train_dir.mkdir()

            # Create dummy h5 files
            for i in range(3):
                h5_file = train_dir / f"data_{i}.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("raw", data=np.random.rand(3, 10, 88, 88))
                    f.create_dataset("label", data=np.random.rand(10, 88, 88))

            datasets = create_datasets(str(temp_path), "train", (8, 64, 64), "3D")

            assert len(datasets) == 3
            assert len(datasets[0]) == 8

            for dataset in datasets:
                assert isinstance(dataset, HDF5Dataset)

    def test_create_datasets_train_2Dc(self):
        """Test creating training datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_dir = temp_path / "train"
            train_dir.mkdir()

            # Create dummy h5 files
            for i in range(3):
                h5_file = train_dir / f"data_{i}.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("raw", data=np.random.rand(10, 88, 88))
                    f.create_dataset("label", data=np.random.rand(88, 88))

            datasets = create_datasets(str(temp_path), "train", (1, 64, 64), "2D")

            assert len(datasets) == 3
            assert len(datasets[0]) == 4

            for dataset in datasets:
                assert isinstance(dataset, HDF5Dataset)

    def test_create_datasets_train_2D(self):
        """Test creating training datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_dir = temp_path / "train"
            train_dir.mkdir()

            # Create dummy h5 files
            for i in range(3):
                h5_file = train_dir / f"data_{i}.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("raw", data=np.random.rand(88, 88))
                    f.create_dataset("label", data=np.random.rand(88, 88))

            datasets = create_datasets(str(temp_path), "train", (1, 64, 64), "2D")

            assert len(datasets) == 3
            assert len(datasets[0]) == 4

            for dataset in datasets:
                assert isinstance(dataset, HDF5Dataset)

    def test_create_datasets_val_3D(self):
        """Test creating validation datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            val_dir = temp_path / "val"
            val_dir.mkdir()

            # Create dummy h5 files
            for i in range(2):
                h5_file = val_dir / f"val_data_{i}.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("raw", data=np.random.rand(10, 88, 88))
                    f.create_dataset("label", data=np.random.rand(10, 88, 88))

            datasets = create_datasets(str(temp_path), "val", (8, 64, 64), "3D")

            assert len(datasets) == 2

    def test_create_datasets_val_2Dc(self):
        """Test creating validation datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            val_dir = temp_path / "val"
            val_dir.mkdir()

            # Create dummy h5 files
            for i in range(2):
                h5_file = val_dir / f"val_data_{i}.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("raw", data=np.random.rand(10, 88, 88))
                    f.create_dataset("label", data=np.random.rand(88, 88))

            datasets = create_datasets(str(temp_path), "val", (1, 64, 64), "2D")

            assert len(datasets) == 2

    def test_create_datasets_val_2D(self):
        """Test creating validation datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            val_dir = temp_path / "val"
            val_dir.mkdir()

            # Create dummy h5 files
            for i in range(2):
                h5_file = val_dir / f"val_data_{i}.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("raw", data=np.random.rand(88, 88))
                    f.create_dataset("label", data=np.random.rand(88, 88))

            datasets = create_datasets(str(temp_path), "val", (1, 64, 64), "2D")

            assert len(datasets) == 2

    def test_create_datasets_invalid_phase(self):
        """Test creating datasets with invalid phase."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            val_dir = temp_path / "val"
            val_dir.mkdir()

            # Create dummy h5 files
            for i in range(2):
                h5_file = val_dir / f"val_data_{i}.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("raw", data=np.random.rand(10, 88, 88))
                    f.create_dataset("label", data=np.random.rand(10, 88, 88))

            with pytest.raises(AssertionError):
                create_datasets(str(temp_dir), "invalid_phase", (8, 64, 64), "3D")

    def test_create_datasets_channel_mismatch(self):
        """The in_channels check names the offending file and both counts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_dir = temp_path / "train"
            train_dir.mkdir()
            h5_file = train_dir / "data_0.h5"
            with h5py.File(h5_file, "w") as f:
                f.create_dataset("raw", data=np.random.rand(10, 88, 88))
                f.create_dataset("label", data=np.random.rand(10, 88, 88))

            with pytest.raises(ValueError) as excinfo:
                create_datasets(
                    str(temp_path), "train", (8, 64, 64), "3D", in_channels=2
                )

            message = str(excinfo.value)
            assert "data_0.h5" in message
            assert "has 1 channels" in message
            assert "in_channels=2" in message


class TestUnetTraining:
    """Tests for unet_training function."""

    @patch("panseg.functionals.training.train.test_model")
    @patch("panseg.functionals.training.train.make_model_description")
    @patch("panseg.functionals.training.train.UNetTrainer")
    @patch("panseg.functionals.training.train.create_datasets")
    @patch("panseg.functionals.training.train.create_model_config")
    @patch("panseg.functionals.training.train.DataLoader")
    @patch("panseg.functionals.training.train.ConcatDataset")
    def test_unet_training_2d(
        self,
        mock_concat,
        mock_data_loader,
        mock_create_config,
        mock_create_datasets,
        mock_trainer,
        mock_model_desc,
        mock_test_model,
        tmp_path,
    ):
        """Test UNet training for 2D case."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10  # Mock non-empty dataset
        mock_create_datasets.return_value = [mock_dataset]

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        _write_checkpoints(mock_trainer, tmp_path / "test_model_2d")

        mock_data_loader.return_value = [
            (  # b, c, z,  x,  y
                torch.from_numpy(np.random.rand(1, 1, 1, 64, 64)).type(torch.float32),
                None,
            )
        ]

        mock_create_config.side_effect = create_model_config

        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "train").mkdir()
        (dataset_dir / "val").mkdir()

        # Test parameters
        model_name = "test_model_2d"

        # Patch PATH_PANSEG_MODELS to use temp directory
        with patch("panseg.functionals.training.train.PATH_PANSEG_MODELS", tmp_path):
            with patch("panseg.core.zoo.PATH_PANSEG_MODELS", tmp_path):
                unet_training(
                    dataset_dir=str(dataset_dir),
                    model_name=model_name,
                    in_channels=1,
                    out_channels=1,
                    feature_maps=(16, 32, 64),
                    patch_size=(1, 64, 64),
                    max_num_iters=100,
                    dimensionality="2D",
                    sparse=False,
                    device="cpu",
                )

        mock_trainer_instance.train.assert_called_once()
        mock_create_config.assert_called_once()

        # Verify that create_datasets was called for both train and val
        assert mock_create_datasets.call_count == 2

        mock_model_desc.assert_called_once()
        # validation runs on the training device (not the bioimageIO default)
        assert mock_test_model.call_args.kwargs["devices"] == ["cpu"]
        assert (tmp_path / model_name / "test_in.npy").exists()
        assert (tmp_path / model_name / "test_out.npy").exists()

    @patch("panseg.functionals.training.train.test_model")
    @patch("panseg.functionals.training.train.make_model_description")
    @patch("panseg.functionals.training.train.UNetTrainer")
    @patch("panseg.functionals.training.train.create_datasets")
    @patch("panseg.functionals.training.train.create_model_config")
    @patch("panseg.functionals.training.train.DataLoader")
    @patch("panseg.functionals.training.train.ConcatDataset")
    def test_unet_training_3d(
        self,
        mock_concat,
        mock_data_loader,
        mock_create_config,
        mock_create_datasets,
        mock_trainer,
        mock_model_desc,
        mock_test_model,
        tmp_path,
    ):
        """Test UNet training for 3D case."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10  # Mock non-empty dataset
        mock_create_datasets.return_value = [mock_dataset]

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        _write_checkpoints(mock_trainer, tmp_path / "test_model_3d")

        mock_data_loader.return_value = [
            (
                torch.from_numpy(np.random.rand(1, 1, 16, 64, 64)).type(torch.float32),
                None,
            )
        ]

        mock_create_config.side_effect = create_model_config

        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "train").mkdir()
        (dataset_dir / "val").mkdir()

        # Test parameters
        model_name = "test_model_3d"

        # Patch PATH_PANSEG_MODELS to use temp directory
        with patch("panseg.functionals.training.train.PATH_PANSEG_MODELS", tmp_path):
            with patch("panseg.core.zoo.PATH_PANSEG_MODELS", tmp_path):
                unet_training(
                    dataset_dir=str(dataset_dir),
                    model_name=model_name,
                    in_channels=1,
                    out_channels=1,
                    feature_maps=(16, 32, 64),
                    patch_size=(8, 64, 64),
                    max_num_iters=100,
                    dimensionality="3D",
                    sparse=True,
                    device="cpu",
                )

        mock_trainer_instance.train.assert_called_once()
        mock_create_config.assert_called_once()

        # Verify that create_datasets was called for both train and val
        assert mock_create_datasets.call_count == 2

        mock_model_desc.assert_called_once()
        assert mock_model_desc.call_args.kwargs["final_sigmoid"] is False
        assert (tmp_path / model_name / "test_in.npy").exists()
        assert (tmp_path / model_name / "test_out.npy").exists()

    @patch("panseg.functionals.training.train.test_model")
    @patch("panseg.functionals.training.train.make_model_description")
    @patch("panseg.functionals.training.train.UNetTrainer")
    @patch("panseg.functionals.training.train.create_datasets")
    @patch("panseg.functionals.training.train.create_model_config")
    @patch("panseg.functionals.training.train.DataLoader")
    @patch("panseg.functionals.training.train.ConcatDataset")
    def test_test_tensors_reproduced_from_packaged_weights(
        self,
        mock_concat,
        mock_data_loader,
        mock_create_config,
        mock_create_datasets,
        mock_trainer,
        mock_model_desc,
        mock_test_model,
        tmp_path,
    ):
        """test_out must come from the packaged weights and test_in must be raw.

        Regression test: test_out used to be generated from the live model (the
        last training state) while the package shipped `best_checkpoint.pytorch`,
        so `bioimageio.core.test_model` could not reproduce the test outputs.
        test_in used to be stored pre-normalized, although the model description
        declares ZeroMeanUnitVariance preprocessing.
        """
        torch.manual_seed(0)

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_create_datasets.return_value = [mock_dataset]
        mock_create_config.side_effect = create_model_config

        loader_batch = torch.from_numpy(
            np.random.rand(1, 1, 1, 64, 64).astype("float32")
        )
        mock_data_loader.return_value = [(loader_batch, None)]

        # the mocked trainer leaves two *different* checkpoints behind:
        # best_checkpoint holds zeroed weights, last_checkpoint the initial ones
        def make_states(model):
            last = model.state_dict()
            best = {k: torch.zeros_like(v) for k, v in last.items()}
            return {"best_checkpoint.pytorch": best, "last_checkpoint.pytorch": last}

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        model_name = "test_model_export"
        _write_checkpoints(mock_trainer, tmp_path / model_name, make_states)

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "train").mkdir(parents=True)
        (dataset_dir / "val").mkdir()

        with (
            patch("panseg.functionals.training.train.PATH_PANSEG_MODELS", tmp_path),
            patch("panseg.core.zoo.PATH_PANSEG_MODELS", tmp_path),
        ):
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
            )

        checkpoint_dir = tmp_path / model_name
        weights = checkpoint_dir / "best_checkpoint.pytorch"
        assert mock_model_desc.call_args.kwargs["weights"] == weights

        test_in = np.load(checkpoint_dir / "test_in.npy")
        test_out = np.load(checkpoint_dir / "test_out.npy")

        # test_in is stored as produced by the loader, un-normalized
        assert np.array_equal(test_in, loader_batch.squeeze(dim=-3).numpy())

        def _predict(state):
            model = UNet2D(
                in_channels=1,
                out_channels=1,
                f_maps=(16, 16),
                final_sigmoid=True,
                layer_order="bcr",
            )
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                out = model(zero_mean_unit_variance(torch.from_numpy(test_in)))
            return out.numpy()

        best_out = _predict(torch.load(weights, map_location="cpu"))
        last_out = _predict(
            torch.load(checkpoint_dir / "last_checkpoint.pytorch", map_location="cpu")
        )

        # test_out reproduces from the packaged (best) weights ...
        assert np.allclose(test_out, best_out, atol=1e-6)
        # ... and not from the last training state
        assert not np.allclose(test_out, last_out, atol=1e-3)

    @patch("panseg.functionals.training.train.test_model")
    @patch("panseg.functionals.training.train.make_model_description")
    @patch("panseg.functionals.training.train.UNetTrainer")
    @patch("panseg.functionals.training.train.create_datasets")
    @patch("panseg.functionals.training.train.create_model_config")
    @patch("panseg.functionals.training.train.find_batch_size")
    @patch("panseg.functionals.training.train.DataLoader")
    @patch("panseg.functionals.training.train.ConcatDataset")
    def test_test_tensors_hold_a_single_sample(
        self,
        mock_concat,
        mock_data_loader,
        mock_find_batch_size,
        mock_create_config,
        mock_create_datasets,
        mock_trainer,
        mock_model_desc,
        mock_test_model,
        tmp_path,
    ):
        """The packaged test tensors hold one sample, whatever the batch size.

        The training batch is fitted to the device memory; the bioimageIO test
        tensors must stay single-sample (one sample is all bioimageIO needs,
        and it keeps the package small).
        """
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_create_datasets.return_value = [mock_dataset]
        mock_create_config.side_effect = create_model_config

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        model_name = "test_model_batch"
        _write_checkpoints(mock_trainer, tmp_path / model_name)

        batch = torch.from_numpy(
            np.random.rand(2, 1, 1, 64, 64).astype("float32")  # b, c, z, y, x
        )
        mock_data_loader.return_value = [(batch, None)]
        mock_find_batch_size.return_value = 2

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "train").mkdir(parents=True)
        (dataset_dir / "val").mkdir()

        with (
            patch("panseg.functionals.training.train.PATH_PANSEG_MODELS", tmp_path),
            patch("panseg.core.zoo.PATH_PANSEG_MODELS", tmp_path),
        ):
            unet_training(
                dataset_dir=str(dataset_dir),
                model_name=model_name,
                in_channels=1,
                out_channels=1,
                feature_maps=(2, 2),
                patch_size=(1, 64, 64),
                max_num_iters=100,
                dimensionality="2D",
                sparse=False,
                device="cpu",
            )

        checkpoint_dir = tmp_path / model_name
        test_in = np.load(checkpoint_dir / "test_in.npy")
        test_out = np.load(checkpoint_dir / "test_out.npy")
        assert test_in.shape == (1, 1, 64, 64)
        assert test_out.shape == (1, 1, 64, 64)
        # the sample is the first of the training batch, stored un-normalized
        assert np.array_equal(test_in, batch[:1].squeeze(dim=-3).numpy())

    @patch("panseg.functionals.training.train.UNetTrainer")
    @patch("panseg.functionals.training.train.create_datasets")
    @patch("panseg.functionals.training.train.create_model_config")
    def test_unet_training_with_existing_checkpoint_dir(
        self, mock_create_config, mock_create_datasets, mock_trainer
    ):
        """Test UNet training with existing checkpoint directory should fail."""
        # Mock the datasets with non-zero length
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 3  # Mock non-empty dataset
        mock_create_datasets.return_value = [mock_dataset]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary dataset directory
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "train").mkdir()
            (dataset_dir / "val").mkdir()

            # Create the checkpoint directory beforehand
            checkpoint_dir = Path(temp_dir) / "test_model"
            checkpoint_dir.mkdir(parents=True)

            # Patch PATH_PANSEG_MODELS to use temp directory
            with patch(
                "panseg.functionals.training.train.PATH_PANSEG_MODELS",
                Path(temp_dir),
            ):
                with pytest.raises(
                    AssertionError, match="Checkpoint dir .* already exists"
                ):
                    unet_training(
                        dataset_dir=str(dataset_dir),
                        model_name="test_model",
                        in_channels=1,
                        out_channels=2,
                        feature_maps=(16, 32),
                        patch_size=(1, 64, 64),
                        max_num_iters=100,
                        dimensionality="2D",
                        sparse=False,
                        device="cpu",
                    )

    @patch("panseg.functionals.training.train.test_model")
    @patch("panseg.functionals.training.train.isinstance")
    @patch("panseg.functionals.training.train.make_model_description")
    @patch("panseg.functionals.training.train.DataLoader")
    @patch("panseg.functionals.training.train.ConcatDataset")
    @patch("panseg.functionals.training.train.UNet2D")
    @patch("panseg.functionals.training.train.UNetTrainer")
    @patch("panseg.functionals.training.train.create_datasets")
    @patch("panseg.functionals.training.train.create_model_config")
    @patch("torch.cuda.device_count")
    @patch("torch.nn.DataParallel")
    @patch("panseg.functionals.training.train.Adam")
    @patch("panseg.functionals.training.train.ReduceLROnPlateau")
    @patch("panseg.functionals.training.train.find_batch_size")
    def test_unet_training_multi_gpu(
        self,
        mock_find_batch_size,
        mock_reduce_lr,
        mock_adam,
        mock_data_parallel,
        mock_device_count,
        mock_create_config,
        mock_create_datasets,
        mock_trainer,
        mock_unet,
        mock_concat,
        mock_data_loader,
        mock_description,
        mock_isinstance,
        mock_test_model,
        tmp_path,
    ):
        """Test UNet training with multiple GPUs."""

        # Mock DataParallel to avoid CUDA initialization
        mock_device_count.return_value = 2
        mock_parallel_model = MagicMock()
        mock_parallel_model.to.return_value = mock_parallel_model
        mock_data_parallel.return_value = mock_parallel_model

        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer

        mock_scheduler = MagicMock()
        mock_reduce_lr.return_value = mock_scheduler

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 8  # Mock non-empty dataset
        mock_create_datasets.return_value = [mock_dataset]

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        _write_checkpoints(
            mock_trainer,
            tmp_path / "test_model_multi_gpu",
            lambda model: {
                "best_checkpoint.pytorch": {},
                "last_checkpoint.pytorch": {},
            },
        )

        mock_data_loader.return_value = [
            (
                # torch.from_numpy(np.random.rand(1, 1, 64, 64)).type(torch.float32),
                mock_parallel_model,
                None,
            )
        ]

        mock_isinstance.return_value = False

        mock_find_batch_size.return_value = 1

        # Create a temporary dataset directory
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "train").mkdir()
        (dataset_dir / "val").mkdir()

        # Patch PATH_PANSEG_MODELS to use temp directory
        with patch("panseg.functionals.training.train.PATH_PANSEG_MODELS", tmp_path):
            with patch("panseg.core.zoo.PATH_PANSEG_MODELS", tmp_path):
                unet_training(
                    dataset_dir=str(dataset_dir),
                    model_name="test_model_multi_gpu",
                    in_channels=1,
                    out_channels=2,
                    feature_maps=(16, 32),
                    patch_size=(1, 64, 64),
                    max_num_iters=100,
                    dimensionality="2D",
                    sparse=False,
                    device="cuda",
                )

        # Verify that trainer was called
        mock_trainer_instance.train.assert_called_once()

        # The batch size must be probed with a backward pass, otherwise the
        # first training iteration can run out of memory
        assert mock_find_batch_size.call_args.kwargs["for_training"] is True

        # Verify that DataParallel was called
        mock_data_parallel.assert_called_once()

        # validation is pinned to the training device (not the bioimageIO default)
        assert mock_test_model.call_args.kwargs["devices"] == ["cuda"]

        # Verify that Adam optimizer was called
        mock_adam.assert_called_once()

        # Verify that ReduceLROnPlateau scheduler was called
        mock_reduce_lr.assert_called_once()

    def test_unet_training_2d_rejects_patch_z_extent_above_1(self, tmp_path):
        """2D models train per z-slice: a patch z-extent > 1 is rejected early."""
        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "train").mkdir(parents=True)
        (dataset_dir / "val").mkdir()

        with pytest.raises(ValueError, match="patch z-extent must be 1"):
            unet_training(
                dataset_dir=str(dataset_dir),
                model_name="test_model_2d",
                in_channels=1,
                out_channels=1,
                feature_maps=(16, 16),
                patch_size=(8, 64, 64),
                max_num_iters=5,
                dimensionality="2D",
                sparse=False,
                device="cpu",
            )

    def test_unet_training_channel_mismatch_raises(self, tmp_path):
        """unet_training propagates the create_datasets channel check."""
        dataset_dir = tmp_path / "dataset"
        train_dir = dataset_dir / "train"
        train_dir.mkdir(parents=True)
        (dataset_dir / "val").mkdir()
        with h5py.File(train_dir / "data_0.h5", "w") as f:
            f.create_dataset("raw", data=np.random.rand(10, 88, 88))
            f.create_dataset("label", data=np.random.rand(10, 88, 88))

        with pytest.raises(ValueError) as excinfo:
            unet_training(
                dataset_dir=str(dataset_dir),
                model_name="test_model_mismatch",
                in_channels=2,
                out_channels=1,
                feature_maps=(8, 8),
                patch_size=(8, 64, 64),
                max_num_iters=5,
                dimensionality="3D",
                sparse=True,
                device="cpu",
            )

        message = str(excinfo.value)
        assert "data_0.h5" in message
        assert "has 1 channels" in message


class TestHDF5DatasetLayouts:
    """Dataset-level tests for the on-disk layout resolution in HDF5Dataset.

    2D training on volumetric data (ZYX/CZYX) is per z-slice: the real z axis
    is kept and every z-slice is an independent 2D sample with patch z-extent
    1. Legacy 2D data (YX/CYX) gets a fabricated singleton z instead. Raw
    patches follow the czyx contract, label patches zyx.
    """

    @staticmethod
    def _make_dataset(
        file_path,
        patch_shape=(1, 64, 64),
        dimensionality: Literal["2D", "3D"] = "2D",
    ):
        return HDF5Dataset(
            file_path=file_path,
            augmenter=Augmenter(),
            patch_shape=patch_shape,
            dimensionality=dimensionality,
        )

    def test_per_slice_zyx_fallback(self, tmp_path):
        """(3, 3) without attrs resolves as ZYX per-slice 2D training."""
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 256, size=(8, 64, 64)).astype("uint8")
        label = _dense_label((8, 64, 64))
        file_path = tmp_path / "zyx.h5"
        _write_layout_h5(file_path, raw, label)

        dataset = self._make_dataset(file_path)

        # loaded as czyx / zyx
        assert dataset.raw.shape == (1, 8, 64, 64)
        assert dataset.label.shape == (8, 64, 64)

        # 8 z-slices x 1 y-step x 1 x-step; the dense label keeps every patch
        assert len(dataset) == 8

        # patches come out in the training contract: raw czyx, label zyx
        assert dataset.raw[dataset.raw_slices[0]].shape == (1, 1, 64, 64)
        assert dataset.label[dataset.label_slices[0]].shape == (1, 64, 64)

        raw_patch, label_patch = dataset[0]
        assert raw_patch.shape == (1, 1, 64, 64)
        # the label transform stacks a boundary channel on the zyx patch
        assert label_patch.shape == (1, 1, 64, 64)

    def test_per_slice_czyx_two_channels(self, tmp_path):
        """(4, 3) without attrs resolves as CZYX per-slice 2D training."""
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 256, size=(2, 8, 64, 64)).astype("uint8")
        label = _dense_label((8, 64, 64))
        file_path = tmp_path / "czyx.h5"
        _write_layout_h5(file_path, raw, label)

        dataset = self._make_dataset(file_path)

        assert dataset.raw.shape == (2, 8, 64, 64)
        assert dataset.label.shape == (8, 64, 64)
        assert len(dataset) == 8
        assert dataset.raw[dataset.raw_slices[0]].shape == (2, 1, 64, 64)
        assert dataset.label[dataset.label_slices[0]].shape == (1, 64, 64)

        raw_patch, label_patch = dataset[0]
        assert raw_patch.shape == (2, 1, 64, 64)
        assert label_patch.shape == (1, 1, 64, 64)

    def test_axis_order_attrs_match_fallback(self, tmp_path):
        """`axis_order` attrs (as written by panseg/io/h5.py) change nothing."""
        rng = np.random.default_rng(0)
        label = _dense_label((8, 64, 64))

        for raw, raw_order in (
            (rng.integers(0, 256, size=(8, 64, 64)).astype("uint8"), "ZYX"),
            (rng.integers(0, 256, size=(2, 8, 64, 64)).astype("uint8"), "CZYX"),
        ):
            plain_path = tmp_path / f"plain_{raw_order}.h5"
            attr_path = tmp_path / f"attr_{raw_order}.h5"
            _write_layout_h5(plain_path, raw, label)
            _write_layout_h5(
                attr_path,
                raw,
                label,
                axis_orders={"raw": raw_order, "label": "ZYX"},
            )

            plain = self._make_dataset(plain_path)
            attributed = self._make_dataset(attr_path)

            assert np.array_equal(attributed.raw, plain.raw)
            assert np.array_equal(attributed.label, plain.label)
            assert attributed.raw_slices == plain.raw_slices
            assert attributed.label_slices == plain.label_slices

    def test_per_slice_keeps_real_z_axis_like_3d(self, tmp_path):
        """Per-slice 2D loading is byte-identical to the 3D branch."""
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 256, size=(8, 64, 64)).astype("uint8")
        label = _dense_label((8, 64, 64))
        file_path = tmp_path / "zyx.h5"
        _write_layout_h5(file_path, raw, label)

        dataset_2d = self._make_dataset(file_path)
        dataset_3d = self._make_dataset(file_path, dimensionality="3D")

        assert np.array_equal(dataset_2d.raw, dataset_3d.raw)
        assert np.array_equal(dataset_2d.label, dataset_3d.label)
        assert dataset_2d.raw_slices == dataset_3d.raw_slices
        assert dataset_2d.label_slices == dataset_3d.label_slices

    def test_legacy_fake_z_yx(self, tmp_path):
        """(2, 2) without attrs is legacy 2D data with a fabricated z."""
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 256, size=(64, 64)).astype("uint8")
        label = _dense_label((64, 64))
        file_path = tmp_path / "yx.h5"
        _write_layout_h5(file_path, raw, label)

        dataset = self._make_dataset(file_path)

        assert dataset.raw.shape == (1, 1, 64, 64)  # c=1, fabricated z=1
        assert dataset.label.shape == (1, 64, 64)
        assert len(dataset) == 1
        assert dataset.raw[dataset.raw_slices[0]].shape == (1, 1, 64, 64)
        assert dataset.label[dataset.label_slices[0]].shape == (1, 64, 64)

        raw_patch, label_patch = dataset[0]
        assert raw_patch.shape == (1, 1, 64, 64)
        assert label_patch.shape == (1, 1, 64, 64)

    def test_legacy_fake_z_cyx(self, tmp_path):
        """(3, 2) without attrs is channelled legacy 2D data."""
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 256, size=(2, 64, 64)).astype("uint8")
        label = _dense_label((64, 64))
        file_path = tmp_path / "cyx.h5"
        _write_layout_h5(file_path, raw, label)

        dataset = self._make_dataset(file_path)

        assert dataset.raw.shape == (2, 1, 64, 64)  # c=2, fabricated z=1
        assert dataset.label.shape == (1, 64, 64)
        assert len(dataset) == 1
        assert dataset.raw[dataset.raw_slices[0]].shape == (2, 1, 64, 64)

        raw_patch, label_patch = dataset[0]
        assert raw_patch.shape == (2, 1, 64, 64)
        assert label_patch.shape == (1, 1, 64, 64)

    def test_per_slice_patch_counts_follow_stride(self, tmp_path):
        """Patch counts follow the FilterSliceBuilder stride max(p // 2, 1)."""
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 256, size=(4, 88, 88)).astype("uint8")
        label = _dense_label((4, 88, 88))
        file_path = tmp_path / "zyx_88.h5"
        _write_layout_h5(file_path, raw, label)

        dataset = self._make_dataset(file_path)

        # patch (1, 64, 64) -> stride (1, 32, 32):
        # 4 z-steps x 2 y-steps (0, 24) x 2 x-steps (0, 24); dense label keeps all
        assert len(dataset) == 16


class TestResolveDatasetLayouts:
    """Tests for the pure layout-resolution helper in h5dataset."""

    @pytest.mark.parametrize(
        ("raw_ndim", "label_ndim", "expected"),
        [
            (2, 2, ("YX", "YX")),
            (3, 2, ("CYX", "YX")),
            (3, 3, ("ZYX", "ZYX")),
            (4, 3, ("CZYX", "ZYX")),
        ],
    )
    def test_fallback_pairing_2d(self, raw_ndim, label_ndim, expected):
        assert resolve_dataset_layouts("2D", raw_ndim, label_ndim) == expected

    @pytest.mark.parametrize(
        ("raw_ndim", "label_ndim", "expected"),
        [
            (3, 3, ("ZYX", "ZYX")),
            (4, 3, ("CZYX", "ZYX")),
        ],
    )
    def test_fallback_pairing_3d(self, raw_ndim, label_ndim, expected):
        assert resolve_dataset_layouts("3D", raw_ndim, label_ndim) == expected

    @pytest.mark.parametrize("dimensionality", ["2D", "2d", "2"])
    def test_dimensionality_spellings(self, dimensionality):
        assert resolve_dataset_layouts(dimensionality, 3, 3) == ("ZYX", "ZYX")

    @pytest.mark.parametrize(
        ("dimensionality", "raw_ndim", "label_ndim"),
        [
            ("2D", 2, 3),
            ("2D", 4, 2),
            ("2D", 5, 3),
            ("3D", 2, 2),
            ("3D", 3, 2),
            ("3D", 3, 4),
            ("4D", 3, 3),
        ],
    )
    def test_invalid_combinations_raise(self, dimensionality, raw_ndim, label_ndim):
        with pytest.raises(ValueError):
            resolve_dataset_layouts(dimensionality, raw_ndim, label_ndim)

    def test_invalid_combination_names_accepted_combos(self):
        with pytest.raises(ValueError, match=r"\(3, 3\), \(4, 3\)"):
            resolve_dataset_layouts("3D", 3, 2)

    def test_axis_order_attr_wins(self):
        # (3, 2) would fall back to CYX/YX legacy 2D; the attrs declare a
        # z-stack with a single 2D label instead
        assert resolve_dataset_layouts(
            "2D", 3, 2, raw_axis_order="ZYX", label_axis_order="YX"
        ) == ("ZYX", "YX")

    @pytest.mark.parametrize(
        ("raw_ndim", "raw_axis_order", "label_ndim", "label_axis_order", "expected"),
        [
            (2, "YX", 2, None, ("YX", "YX")),
            (3, "CYX", 2, None, ("CYX", "YX")),
            (3, None, 3, "ZYX", ("ZYX", "ZYX")),
            (4, "CZYX", 3, "ZYX", ("CZYX", "ZYX")),
        ],
    )
    def test_partial_attrs_use_pairing_fallback(
        self, raw_ndim, raw_axis_order, label_ndim, label_axis_order, expected
    ):
        assert (
            resolve_dataset_layouts(
                "2D", raw_ndim, label_ndim, raw_axis_order, label_axis_order
            )
            == expected
        )

    def test_time_axis_rejected(self):
        with pytest.raises(ValueError, match="Time axis not supported"):
            resolve_dataset_layouts("2D", 4, 3, raw_axis_order="TZYX")

    def test_axis_order_ndim_mismatch(self):
        with pytest.raises(ValueError, match="inconsistent"):
            resolve_dataset_layouts("2D", 4, 3, raw_axis_order="ZYX")

    def test_unsupported_axis_order(self):
        with pytest.raises(ValueError, match="Unsupported axis_order"):
            resolve_dataset_layouts("2D", 3, 3, raw_axis_order="XYZ")

    def test_label_channel_axis_rejected(self):
        with pytest.raises(ValueError, match="channel-less"):
            resolve_dataset_layouts(
                "2D", 3, 3, raw_axis_order="ZYX", label_axis_order="CYX"
            )

    def test_attr_conflicts_with_pairing(self):
        # (3, 3) pairs as ZYX/ZYX, but the raw dataset declares CYX
        with pytest.raises(ValueError, match="Cannot resolve"):
            resolve_dataset_layouts("2D", 3, 3, raw_axis_order="CYX")

    @pytest.mark.parametrize(
        ("raw_ndim", "raw_axis_order", "label_ndim", "label_axis_order"),
        [
            (3, "CYX", 2, "YX"),
            (2, "YX", 2, "YX"),
            (3, "CYX", 3, "ZYX"),
        ],
    )
    def test_3d_rejects_2d_layouts_with_attrs(
        self, raw_ndim, raw_axis_order, label_ndim, label_axis_order
    ):
        # attrs win over the pairing fallback, but 2D-layout data must not
        # silently get a fabricated z axis for 3D training
        with pytest.raises(ValueError, match="3D training requires volumetric"):
            resolve_dataset_layouts(
                "3D", raw_ndim, label_ndim, raw_axis_order, label_axis_order
            )

    def test_non_string_axis_order_rejected(self):
        bad_attr: Any = 5  # e.g. a corrupted h5 attribute
        with pytest.raises(ValueError, match="must be a string"):
            resolve_dataset_layouts("2D", 3, 3, raw_axis_order=bad_attr)


class TestLoadPretrainedWeights:
    """Tests for the fine-tuning weight loader in trainer."""

    @staticmethod
    def _make_model(in_channels=1):
        return UNet2D(
            in_channels=in_channels,
            out_channels=1,
            f_maps=(8, 16),
            final_sigmoid=True,
            layer_order="bcr",
        )

    def test_load_plain_state_dict(self, tmp_path):
        source = self._make_model()
        path = tmp_path / "weights.pytorch"
        torch.save(source.state_dict(), path)

        target = self._make_model()
        load_pretrained_weights(target, path)

        for expected, actual in zip(source.parameters(), target.parameters()):
            assert torch.equal(expected, actual)

    def test_load_wrapped_state_dict(self, tmp_path):
        source = self._make_model()
        path = tmp_path / "weights.pytorch"
        torch.save({"model_state_dict": source.state_dict()}, path)

        target = self._make_model()
        load_pretrained_weights(target, path)

        for expected, actual in zip(source.parameters(), target.parameters()):
            assert torch.equal(expected, actual)

    def test_load_into_data_parallel_wrapped_model(self, tmp_path):
        class _DataParallelLike(torch.nn.DataParallel):
            """nn.DataParallel stand-in that skips the GPU-dependent __init__."""

            def __init__(self, module):
                torch.nn.Module.__init__(self)
                self.module = module

        source = self._make_model()
        path = tmp_path / "weights.pytorch"
        torch.save(source.state_dict(), path)

        target = _DataParallelLike(self._make_model())
        load_pretrained_weights(target, path)

        for expected, actual in zip(source.parameters(), target.module.parameters()):
            assert torch.equal(expected, actual)

    def test_channel_mismatch_raises_clear_error(self, tmp_path):
        source = self._make_model(in_channels=2)
        path = tmp_path / "weights.pytorch"
        torch.save(source.state_dict(), path)

        with pytest.raises(ValueError, match="architecture or a different"):
            load_pretrained_weights(self._make_model(in_channels=1), path)


class TestUNetTrainerForwardPass:
    """The 2D singleton-z handling must also work for DataParallel-wrapped models."""

    class _DataParallelLike(torch.nn.DataParallel):
        """nn.DataParallel stand-in that skips the GPU-dependent __init__."""

        def __init__(self, module):
            torch.nn.Module.__init__(self)
            self.module = module

        def forward(self, *inputs):
            return self.module(*inputs)

    def _trainer(self, model):
        trainer = UNetTrainer.__new__(UNetTrainer)
        trainer.model = model
        trainer.loss_criterion = DiceLoss()
        trainer.amp = "auto"
        trainer.device = "cpu"
        return trainer

    def test_forward_pass_unwraps_data_parallel_2d_model(self):
        # Per-slice 2D training feeds (B, C, 1, Y, X); the model needs the z
        # axis squeezed even when it is wrapped in nn.DataParallel
        model = UNet2D(in_channels=1, out_channels=1, f_maps=[4, 4])
        _input = torch.rand(2, 1, 1, 32, 32)
        target = torch.rand(2, 1, 1, 32, 32)

        output, loss = self._trainer(self._DataParallelLike(model))._forward_pass(
            _input, target
        )

        assert output.shape == (2, 1, 1, 32, 32)
        assert torch.isfinite(loss)

    def test_forward_pass_keeps_z_axis_for_3d_model(self):
        model = UNet3D(in_channels=1, out_channels=1, f_maps=[4, 4])
        _input = torch.rand(1, 1, 16, 16, 16)
        target = torch.rand(1, 1, 16, 16, 16)

        output, loss = self._trainer(model)._forward_pass(_input, target)

        assert output.shape == (1, 1, 16, 16, 16)
        assert torch.isfinite(loss)


class TestUNetTrainerCleanup:
    """`cleanup()` frees the resources held after training and is idempotent."""

    @staticmethod
    def _trained_trainer(tmp_path):
        """A trainer that has seen one optimization step (Adam state allocated)."""
        model = UNet2D(in_channels=1, out_channels=1, f_maps=[4, 4])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model(torch.rand(1, 1, 32, 32)).sum().backward()
        optimizer.step()

        return UNetTrainer(
            model=model,
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.2, patience=10
            ),
            loss_criterion=DiceLoss(),
            loaders={"train": [None, None], "val": []},
            checkpoint_dir=tmp_path,
            max_num_iterations=10,
            device="cpu",
        )

    def test_cleanup_frees_training_resources(self, tmp_path):
        trainer = self._trained_trainer(tmp_path)
        optimizer = trainer.optimizer
        # precondition: the Adam step allocated its moment buffers
        assert len(optimizer.state) > 0

        trainer.cleanup()

        assert len(optimizer.state) == 0
        assert trainer.writer is None  # closed and dropped
        assert trainer.loaders == {}
        assert all(p.device.type == "cpu" for p in trainer.model.parameters())

    def test_cleanup_is_idempotent(self, tmp_path, mocker):
        writer_cls = mocker.patch("panseg.functionals.training.trainer.SummaryWriter")
        trainer = self._trained_trainer(tmp_path)

        trainer.cleanup()
        trainer.cleanup()  # must not raise or double-close

        writer_cls.return_value.close.assert_called_once()
