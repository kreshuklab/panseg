"""Validation loader slice-grid behavior (findings 1+2 of the perf analysis).

Validation must run on a *non-overlapping*, *unfiltered* patch grid: every val
patch is evaluated exactly as extracted (no foreground threshold), and the grid
covers the whole volume without the 8x-overlapping train stride. The train
loader keeps its `FilterSliceBuilder` behavior (stride max(p // 2, 1),
threshold-filtered).
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from panseg.functionals.training.train import create_datasets
from tests.functionals.training.helpers import write_h5

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

        starts = [
            sorted({raw_slice[axis + 1].start for raw_slice in val.raw_slices})
            for axis in range(3)
        ]
        assert starts == [[0, 32, 38], [0, 128, 172], [0, 128, 172]]

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
