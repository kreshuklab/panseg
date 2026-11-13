"""Integration tests for training."""

import numpy as np
import pytest
import torch
import yaml

from panseg import FILE_CONFIG_TRAIN_YAML, FILE_MODEL_ZOO_CUSTOM, PATH_PANSEG_MODELS
from panseg.core.zoo import model_zoo
from panseg.functionals.training.train import unet_training
from tests.functionals.training.helpers import model_dir_patches, write_h5


def _write_h5_dataset(
    file_path,
    volume_size: int = 64,
    target_label_frac: float = 0.13,
    seed: int = 0,
    raw_channels: int = 1,
):
    """Write a single H5 file with ``raw`` (uint8 noise) and ``label`` (uint8
    ellipsoid instances) datasets.

    The label density is kept near the real mini files (~12-15 % nonzero):
    below ~10 % the slice filter drops all patches and the pipeline crashes.
    ``raw_channels`` > 1 writes a channelled (CZYX) raw dataset.
    """
    rng = np.random.default_rng(seed)
    raw_shape = (volume_size,) * 3
    if raw_channels > 1:
        raw_shape = (raw_channels,) + raw_shape
    raw = rng.integers(0, 256, size=raw_shape).astype("uint8")
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
    write_h5(file_path, raw, label)


@pytest.fixture
def generated_dataset_dir(tmp_path):
    """Create a tiny self-generated H5 dataset (2 train + 1 val file, 64^3 each)."""
    data_dir = tmp_path / "data"
    for idx, seed in enumerate((0, 1)):
        _write_h5_dataset(data_dir / "train" / f"train_{idx:03d}.h5", seed=seed)
    _write_h5_dataset(data_dir / "val" / "val_000.h5", seed=2)
    return data_dir


def _redirect_model_dirs(mocker, root):
    """Keep model-zoo writes out of the user's model directory."""
    for target, attribute, value in model_dir_patches(root):
        mocker.patch.object(target, attribute, value)


def _train_2d(dataset_dir, model_name, in_channels=1, pre_trained=None):
    """Tiny 2D training run on CPU (seconds-class): per z-slice on 3D data."""
    unet_training(
        dataset_dir=str(dataset_dir),
        model_name=model_name,
        in_channels=in_channels,
        out_channels=1,
        feature_maps=[2, 2],
        patch_size=(1, 64, 64),
        max_num_iters=5,
        dimensionality="2D",
        sparse=True,
        device="cpu",
        pre_trained=pre_trained,
    )


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

        test_in = np.load(model_dir / "test_in.npy")
        test_out = np.load(model_dir / "test_out.npy")
        # the packaged test tensors hold a single sample: b, c, z, y, x
        assert test_in.shape == (1, 1, 16, 64, 64)
        assert test_out.shape == (1, 1, 16, 64, 64)

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

        test_in = np.load(model_dir / "test_in.npy")
        test_out = np.load(model_dir / "test_out.npy")
        # single sample even though the GPU training batch may be larger
        assert test_in.shape == (1, 1, 16, 64, 64)
        assert test_out.shape == (1, 1, 16, 64, 64)

        mocker.stopall()
        assert not (PATH_PANSEG_MODELS / model_name).exists()

    def test_training_integration_2d_on_zyx_cpu(self, mocker, generated_dataset_dir):
        """2D UNet trained per z-slice on ZYX volumes, end to end.

        The files carry no `axis_order` attrs, so the (raw ndim, label ndim)
        fallback resolves (3, 3) as ZYX per-slice training.
        """
        tmp_path = generated_dataset_dir.parent
        model_name = "test_integration_2d_on_zyx_cpu"
        # model should not be saved in the users models directory
        assert not (PATH_PANSEG_MODELS / model_name).exists()

        _redirect_model_dirs(mocker, tmp_path)
        _train_2d(generated_dataset_dir, model_name)

        model_dir = tmp_path / model_name
        assert model_dir.exists(), f"Model directory not created: {model_dir}"
        assert (model_dir / "best_checkpoint.pytorch").exists()
        assert (model_dir / "last_checkpoint.pytorch").exists()

        with open(model_dir / FILE_CONFIG_TRAIN_YAML) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        assert config["model"]["name"] == "UNet2D"
        train_slices = config["loaders"]["train"]["slice_builder"]
        assert tuple(train_slices["patch_shape"]) == (1, 64, 64)
        # the train stride halves the patch per axis but keeps z at 1 (never 0)
        assert tuple(train_slices["stride_shape"]) == (1, 32, 32)

        test_in = np.load(model_dir / "test_in.npy")
        test_out = np.load(model_dir / "test_out.npy")
        # 2D test tensors are stored squeezed: batch, c, y, x -- no z axis
        assert test_in.shape == (1, 1, 64, 64)
        assert test_out.shape == (1, 1, 64, 64)

    def test_training_integration_2d_on_czyx_cpu(self, mocker, tmp_path):
        """2D UNet trained per z-slice on 2-channel CZYX raw data, end to end."""
        data_dir = tmp_path / "data_czyx"
        for idx, seed in enumerate((0, 1)):
            _write_h5_dataset(
                data_dir / "train" / f"train_{idx:03d}.h5", seed=seed, raw_channels=2
            )
        _write_h5_dataset(data_dir / "val" / "val_000.h5", seed=2, raw_channels=2)

        model_name = "test_integration_2d_on_czyx_cpu"
        assert not (PATH_PANSEG_MODELS / model_name).exists()

        _redirect_model_dirs(mocker, tmp_path)
        _train_2d(data_dir, model_name, in_channels=2)

        model_dir = tmp_path / model_name
        assert model_dir.exists(), f"Model directory not created: {model_dir}"
        assert (model_dir / "best_checkpoint.pytorch").exists()
        assert (model_dir / "last_checkpoint.pytorch").exists()

        test_in = np.load(model_dir / "test_in.npy")
        test_out = np.load(model_dir / "test_out.npy")
        # squeezed 2D test tensors: batch, c, y, x -- raw keeps both channels
        assert test_in.shape == (1, 2, 64, 64)
        assert test_out.shape == (1, 1, 64, 64)

    def test_training_integration_finetune_from_checkpoint_cpu(
        self, mocker, generated_dataset_dir
    ):
        """Fine-tuning from a previous run's checkpoint, end to end.

        Regression test for the multi-GPU ordering bug: the pre-trained weights
        must load into the unwrapped model inside `unet_training`, before any
        `nn.DataParallel` wrap. Covers plain and `{"model_state_dict": ...}`
        wrapped checkpoints.
        """
        tmp_path = generated_dataset_dir.parent
        model_name = "test_integration_finetune_base"
        assert not (PATH_PANSEG_MODELS / model_name).exists()

        _redirect_model_dirs(mocker, tmp_path)
        _train_2d(generated_dataset_dir, model_name)

        base_ckpt = tmp_path / model_name / "best_checkpoint.pytorch"
        assert base_ckpt.exists()
        wrapped_ckpt = tmp_path / "wrapped_checkpoint.pytorch"
        torch.save(
            {
                "model_state_dict": torch.load(
                    base_ckpt, map_location="cpu", weights_only=True
                )
            },
            wrapped_ckpt,
        )

        for suffix, pre_trained in (
            ("plain", base_ckpt),
            ("wrapped", wrapped_ckpt),
        ):
            model_name = f"test_integration_finetune_{suffix}"
            _train_2d(generated_dataset_dir, model_name, pre_trained=pre_trained)

            model_dir = tmp_path / model_name
            assert model_dir.exists(), f"Model directory not created: {model_dir}"
            assert (model_dir / "best_checkpoint.pytorch").exists()
            assert (model_dir / "last_checkpoint.pytorch").exists()
            assert (model_dir / "test_in.npy").exists()
            assert (model_dir / "test_out.npy").exists()

            with open(model_dir / FILE_CONFIG_TRAIN_YAML) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            assert config["trainer"]["pre_trained"] == str(pre_trained)
