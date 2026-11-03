"""Validation loader slice-grid behavior (findings 1+2 of the perf analysis).

Validation must run on a *non-overlapping*, *unfiltered* patch grid: every val
patch is evaluated exactly as extracted (no foreground threshold), and the grid
covers the whole volume without the 8x-overlapping train stride. The train
loader keeps its `FilterSliceBuilder` behavior (stride max(p // 2, 1),
threshold-filtered).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from panseg.core.zoo import model_zoo
from panseg.functionals.training.train import create_datasets, unet_training
from tests.functionals.training.helpers import model_dir_patches, write_h5

# The example volume from the analysis: 64x256x256 with patch 32x128x128.
# The train grid (stride patch // 2) yields 3x3x3 = 27 patches; the
# non-overlapping val grid must yield 2x2x2 = 8.
VOL_SHAPE = (64, 256, 256)
PATCH_SHAPE = (32, 128, 128)
TRAIN_PATCH_COUNT = 27
VAL_PATCH_COUNT = 8


def _write_h5(file_path: Path, raw: np.ndarray, label: np.ndarray):
    """Write a training H5 file with the analysis' dtypes (uint16 raw, uint8 label)."""
    write_h5(file_path, raw, label, raw_dtype="uint16", label_dtype="uint8")


def _dense_label(shape):
    """Instance label that is nonzero everywhere (survives the filter).

    Split along z so that *every* possible patch is >= 50 % nonzero.
    """
    label = np.zeros(shape, dtype="uint8")
    label[: shape[0] // 2] = 2
    label[shape[0] // 2 :] = 1
    return label


def _make_phase_dataset(dataset_dir, phase, **kwargs):
    return create_datasets(dataset_dir, phase, PATCH_SHAPE, "3D", **kwargs)[0]


def _coverage(dataset):
    """Voxel coverage counts of the patch grid (drops the channel slice)."""
    counts = np.zeros(dataset.raw.shape[1:], dtype=int)
    for raw_idx in dataset.raw_slices:
        counts[raw_idx[1:]] += 1
    return counts


def _expected_coverage(shape, patch):
    """Reference coverage of the documented val grid: regular non-overlapping
    steps plus one final patch anchored at the border when the regular steps
    stop short. Overlap then occurs exactly where that anchored patch reaches
    back over the previous one (and compounds across ragged dimensions)."""
    counts = np.zeros(shape, dtype=int)
    starts_per_dim = []
    for size, k in zip(shape, patch):
        starts = list(range(0, size - k + 1, k))
        if starts[-1] + k < size:
            starts.append(size - k)
        starts_per_dim.append(starts)
    for z in starts_per_dim[0]:
        for y in starts_per_dim[1]:
            for x in starts_per_dim[2]:
                counts[z : z + patch[0], y : y + patch[1], x : x + patch[2]] += 1
    return counts


@pytest.fixture()
def dense_dir(tmp_path):
    """A 64x256x256 volume whose label survives the foreground filter."""
    rng = np.random.default_rng(0)
    raw, label = rng.integers(0, 256, VOL_SHAPE), _dense_label(VOL_SHAPE)
    _write_h5(tmp_path / "train" / "vol.h5", raw, label)
    _write_h5(tmp_path / "val" / "vol.h5", raw, label)
    return tmp_path


@pytest.fixture()
def sparse_dir(tmp_path):
    """A volume whose label is almost pure background (the filter drops most patches)."""
    rng = np.random.default_rng(0)
    label = np.zeros(VOL_SHAPE, dtype="uint8")
    label[:2, :2, :2] = 1  # 8 foreground voxels in 4.2 M
    raw = rng.integers(0, 256, VOL_SHAPE)
    _write_h5(tmp_path / "train" / "vol.h5", raw, label)
    _write_h5(tmp_path / "val" / "vol.h5", raw, label)
    return tmp_path


class TestValSliceGrid:
    """The val grid is non-overlapping and unfiltered; train is unchanged."""

    def test_val_patches_form_the_non_overlapping_grid(self, dense_dir):
        val = _make_phase_dataset(dense_dir, "val")

        assert len(val) == VAL_PATCH_COUNT

    def test_train_patches_keep_the_overlapping_filtered_grid(self, dense_dir):
        train = _make_phase_dataset(dense_dir, "train")

        assert len(train) == TRAIN_PATCH_COUNT

    def test_val_grid_covers_a_divisible_volume_exactly_once(self, dense_dir):
        val = _make_phase_dataset(dense_dir, "val")

        counts = _coverage(val)
        assert counts.min() == 1
        assert counts.max() == 1

    def test_val_grid_covers_an_edge_volume_completely(self, tmp_path):
        """Patches must still cover the volume when it is not a multiple of the patch."""
        shape = (70, 300, 300)  # patch 32x128x128 leaves ragged edges
        rng = np.random.default_rng(0)
        _write_h5(
            tmp_path / "val" / "ragged.h5",
            rng.integers(0, 256, shape),
            _dense_label(shape),
        )
        val = _make_phase_dataset(tmp_path, "val")

        counts = _coverage(val)
        assert counts.min() >= 1, "every voxel must be validated"
        # covers the volume exactly once except where the border-anchored edge
        # patches overlap the previous grid cell (compounds over the 3 ragged
        # dims, up to 2^3 at the corner)
        assert np.array_equal(counts, _expected_coverage(shape, PATCH_SHAPE))
        assert counts.max() <= 2**3

    def test_val_keeps_background_patches(self, sparse_dir):
        """Val loss must not be biased towards foreground-rich patches."""
        val = _make_phase_dataset(sparse_dir, "val")

        assert len(val) == VAL_PATCH_COUNT

    def test_train_still_filters_background_patches(self, sparse_dir):
        train = _make_phase_dataset(sparse_dir, "train")

        assert len(train) < TRAIN_PATCH_COUNT


class TestValStrideConfig:
    """The val `stride_shape` of the training YAML is honored (it was dead config)."""

    def test_create_datasets_honors_val_stride(self, dense_dir):
        datasets = create_datasets(
            dense_dir,
            "val",
            PATCH_SHAPE,
            "3D",
            stride_shape=(16, 64, 64),
        )

        # the old 8x-overlapping grid: 3 x 3 x 3
        assert len(datasets[0]) == TRAIN_PATCH_COUNT

    def test_create_datasets_rejects_train_stride(self, dense_dir):
        """Train keeps FilterSliceBuilder; a train stride would silently do nothing."""
        with pytest.raises(ValueError, match="stride_shape"):
            create_datasets(
                dense_dir,
                "train",
                PATCH_SHAPE,
                "3D",
                stride_shape=(16, 64, 64),
            )


class TestUnetTrainingValSliceBuilder:
    """`unet_training` threads the val slice-builder section into the val loader."""

    @staticmethod
    def _run_mocked_unet_training(tmp_path, **extra_kwargs):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.raw = MagicMock(shape=(1, 8, 64, 64))
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
            ) as mock_create_datasets,
            patch("panseg.functionals.training.train.create_model_config"),
            patch("panseg.functionals.training.train.find_batch_size", return_value=1),
            patch(
                "panseg.functionals.training.train.DataLoader",
                return_value=[(batch, None)],
            ),
            patch("panseg.functionals.training.train.ConcatDataset"),
            patch("panseg.functionals.training.train.PATH_PANSEG_MODELS", tmp_path),
            patch("panseg.core.zoo.PATH_PANSEG_MODELS", tmp_path),
            patch.multiple(
                model_zoo,
                path_zoo=tmp_path / "models_zoo.yaml",
                path_zoo_custom=tmp_path / "custom_zoo.yaml",
            ),
        ):

            def _train():
                state = mock_trainer.call_args.kwargs["model"].state_dict()
                checkpoint_dir = tmp_path / "test_val_stride_cfg"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(state, checkpoint_dir / "best_checkpoint.pytorch")

            mock_trainer.return_value.train.side_effect = _train
            unet_training(
                dataset_dir=str(dataset_dir),
                model_name="test_val_stride_cfg",
                in_channels=1,
                out_channels=1,
                feature_maps=(16, 16),
                patch_size=(1, 64, 64),
                max_num_iters=5,
                dimensionality="2D",
                sparse=False,
                device="cpu",
                **extra_kwargs,
            )
        return mock_create_datasets

    def test_val_slice_builder_section_selects_the_stride(self, tmp_path):
        mock_create_datasets = self._run_mocked_unet_training(
            tmp_path, val_slice_builder={"stride_shape": [1, 32, 32]}
        )

        val_call = mock_create_datasets.call_args_list[1]
        assert val_call.kwargs["stride_shape"] == (1, 32, 32)

    def test_val_defaults_to_non_overlapping(self, tmp_path):
        mock_create_datasets = self._run_mocked_unet_training(tmp_path)

        val_call = mock_create_datasets.call_args_list[1]
        assert val_call.kwargs["stride_shape"] is None

    def test_train_call_has_no_val_stride(self, tmp_path):
        mock_create_datasets = self._run_mocked_unet_training(
            tmp_path, val_slice_builder={"stride_shape": [1, 32, 32]}
        )

        train_call = mock_create_datasets.call_args_list[0]
        assert "stride_shape" not in train_call.kwargs


class TestLaunchTrainingValStride:
    """A YAML's `loaders.val.slice_builder.stride_shape` reaches the real grid.

    End to end through the CLI entry point: a tiny CPU training run on a real
    H5 dataset, with the actual `create_datasets` call observed.
    """

    VOLUME = 64

    @classmethod
    def _write_dataset(cls, root: Path) -> Path:
        rng = np.random.default_rng(0)
        shape = (cls.VOLUME,) * 3
        raw = rng.integers(0, 256, shape)
        label = np.zeros(shape, dtype="uint8")
        label[8:56, 8:56, 8:56] = 1
        data_dir = root / "data"
        _write_h5(data_dir / "train" / "vol.h5", raw, label)
        _write_h5(data_dir / "val" / "vol.h5", raw, label)
        return data_dir

    def _run(self, tmp_path, monkeypatch, model_name: str, **config_extra) -> list[int]:
        from panseg.functionals.training import train as train_module
        from panseg.run_panseg import main

        data_dir = self._write_dataset(tmp_path)
        config = {
            "dataset_dir": str(data_dir),
            "model_name": model_name,
            "in_channels": 1,
            "out_channels": 1,
            "feature_maps": [2, 2],
            "patch_size": [16, self.VOLUME, self.VOLUME],
            "max_num_iters": 2,
            "dimensionality": "3D",
            "sparse": True,
            "device": "cpu",
            **config_extra,
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))

        real_create_datasets = train_module.create_datasets
        patch_counts: list[int] = []

        def spy(*args, **kwargs):
            datasets = real_create_datasets(*args, **kwargs)
            patch_counts.append(sum(len(dataset) for dataset in datasets))
            return datasets

        monkeypatch.setattr(train_module, "create_datasets", spy)
        monkeypatch.setattr(train_module, "test_model", MagicMock())
        monkeypatch.setattr(train_module, "make_model_description", MagicMock())
        monkeypatch.setattr("panseg.run_panseg.check_version", MagicMock())
        for target, attribute, value in model_dir_patches(tmp_path):
            monkeypatch.setattr(target, attribute, value)

        monkeypatch.setattr("sys.argv", ["panseg", "--train", str(config_path)])
        main()

        # train grid first, val grid second (see `unet_training`)
        assert len(patch_counts) == 2
        return patch_counts

    def test_configured_val_stride_changes_the_val_grid(self, tmp_path, monkeypatch):
        patch_counts = self._run(
            tmp_path,
            monkeypatch,
            model_name="test_launch_val_stride_cfg",
            loaders={
                "val": {
                    "slice_builder": {
                        "name": "SliceBuilder",
                        "patch_shape": [16, self.VOLUME, self.VOLUME],
                        # half the patch on y/x, half on z: 7 patches instead of 4
                        "stride_shape": [8, self.VOLUME // 2, self.VOLUME // 2],
                    }
                }
            },
        )

        assert patch_counts[1] == 7

    def test_omitted_val_stride_runs_the_non_overlapping_grid(
        self, tmp_path, monkeypatch
    ):
        patch_counts = self._run(
            tmp_path, monkeypatch, model_name="test_launch_val_stride_default"
        )

        # 64 / 16 == 4 non-overlapping z patches
        assert patch_counts[1] == 4


class TestModelConfigValStride:
    """The written training YAML records the *effective* val stride."""

    def test_default_val_stride_is_the_patch_shape(self, tmp_path):
        from panseg.functionals.training.train import create_model_config

        create_model_config(
            tmp_path / "ckpt",
            1,
            1,
            (8, 64, 64),
            "3D",
            "bcr",
            False,
            16,
            100,
        )

        with open(tmp_path / "ckpt" / "config_train.yml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        assert tuple(config["loaders"]["val"]["slice_builder"]["stride_shape"]) == (
            8,
            64,
            64,
        )

    def test_configured_val_stride_is_recorded(self, tmp_path):
        from panseg.functionals.training.train import create_model_config

        create_model_config(
            tmp_path / "ckpt",
            1,
            1,
            (8, 64, 64),
            "3D",
            "bcr",
            False,
            16,
            100,
            val_stride_shape=(4, 32, 32),
        )

        with open(tmp_path / "ckpt" / "config_train.yml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        assert tuple(config["loaders"]["val"]["slice_builder"]["stride_shape"]) == (
            4,
            32,
            32,
        )

    @pytest.mark.parametrize("val_stride_shape", [None, (4, 32, 32)])
    def test_written_config_round_trips_as_a_training_config(
        self, tmp_path, val_stride_shape
    ):
        """The record written into `config_train.yml` is a valid input config.

        `launch_training` reads `loaders.val.slice_builder.stride_shape` back, so
        the written record reproduces the run's validation grid instead of being
        dead config.
        """
        from panseg.functionals.training.train import create_model_config
        from panseg.run_panseg import _translate_loaders

        create_model_config(
            tmp_path / "ckpt",
            1,
            1,
            (8, 64, 64),
            "3D",
            "bcr",
            False,
            16,
            100,
            val_stride_shape=val_stride_shape,
        )

        with open(tmp_path / "ckpt" / "config_train.yml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["patch_size"] = [8, 64, 64]

        _translate_loaders(config, tmp_path / "config_train.yml")

        assert "loaders" not in config
        expected = list(val_stride_shape or (8, 64, 64))
        assert config["val_slice_builder"]["stride_shape"] == expected
