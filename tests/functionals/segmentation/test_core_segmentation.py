import logging

import numpy as np
import pytest

from panseg.functionals.segmentation import dt_watershed

shapes = [(32, 64, 64), (64, 64)]
stacked_options = [True, False]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("stacked", stacked_options)
def test_dt_watershed(shape, stacked):
    mock_data = np.random.rand(*shape).astype("float32")

    if stacked and len(shape) == 2:
        with pytest.raises(ValueError):  # 2D data cannot be stacked
            dt_watershed(mock_data, stacked=stacked)
    else:
        result = dt_watershed(mock_data, stacked=stacked)
        assert isinstance(result, np.ndarray)
        assert result.shape == mock_data.shape
        assert result.dtype == np.uint64
        assert result.max() > result.min() >= 0


def test_dt_watershed_blockwise_3d():
    mock_data = np.random.rand(32, 64, 64).astype("float32")

    result = dt_watershed(mock_data, blockwise=True)
    assert isinstance(result, np.ndarray)
    assert result.shape == mock_data.shape
    assert result.dtype == np.uint64
    assert result.max() > result.min() >= 0
    assert len(np.unique(result)) > 1


def test_dt_watershed_blockwise_n_threads(caplog):
    # volume large enough (>= 2M voxels) for the real blockwise path to run
    mock_data = np.random.rand(64, 256, 128).astype("float32")

    with caplog.at_level(logging.WARNING):
        result = dt_watershed(mock_data, blockwise=True, n_threads=2)
    assert "falling back" not in caplog.text
    assert isinstance(result, np.ndarray)
    assert result.shape == mock_data.shape
    assert result.dtype == np.uint64
    assert result.max() > result.min() >= 0
    assert len(np.unique(result)) > 1


def test_dt_watershed_blockwise_default_threads(caplog):
    # volume large enough (>= 2M voxels); default n_threads targets all cores
    mock_data = np.random.rand(64, 256, 128).astype("float32")

    with caplog.at_level(logging.WARNING):
        result = dt_watershed(mock_data, blockwise=True)
    assert "falling back" not in caplog.text
    assert isinstance(result, np.ndarray)
    assert result.shape == mock_data.shape
    assert result.dtype == np.uint64
    assert result.max() > result.min() >= 0
    assert len(np.unique(result)) > 1


def test_dt_watershed_blockwise_path_taken(caplog):
    # volume large enough (>= 2M voxels) for the blockwise path to be taken
    mock_data = np.random.rand(64, 256, 128).astype("float32")

    with caplog.at_level(logging.WARNING):
        result = dt_watershed(mock_data, blockwise=True, n_threads=2)
    assert "falling back" not in caplog.text
    assert isinstance(result, np.ndarray)
    assert result.shape == mock_data.shape
    assert result.dtype == np.uint64
    assert result.max() > result.min() >= 0
    assert len(np.unique(result)) > 1


def test_dt_watershed_blockwise_explicit_block_shape_and_halo(caplog):
    # volume large enough (>= 2M voxels); block_shape yields 2x2x2 = 8 blocks
    mock_data = np.random.rand(64, 256, 128).astype("float32")

    with caplog.at_level(logging.WARNING):
        result = dt_watershed(
            mock_data, blockwise=True, block_shape=(32, 128, 64), halo=(8, 8, 8)
        )
    assert "falling back" not in caplog.text
    assert isinstance(result, np.ndarray)
    assert result.shape == mock_data.shape
    assert result.dtype == np.uint64
    assert result.max() > result.min() >= 0


def test_dt_watershed_blockwise_2d_fallback(caplog):
    mock_data = np.random.rand(64, 64).astype("float32")

    with caplog.at_level(logging.WARNING):
        result = dt_watershed(mock_data, blockwise=True)
    assert "falling back" in caplog.text
    assert result.shape == mock_data.shape
    assert result.dtype == np.uint64
    assert result.max() > result.min() >= 0


def test_dt_watershed_blockwise_small_volume_fallback(caplog):
    mock_data = np.random.rand(8, 16, 16).astype("float32")

    with caplog.at_level(logging.WARNING):
        result = dt_watershed(mock_data, blockwise=True)
    assert "falling back" in caplog.text
    assert result.shape == mock_data.shape
    assert result.dtype == np.uint64
    assert result.max() > result.min() >= 0


def test_dt_watershed_blockwise_single_block_fallback(caplog):
    mock_data = np.random.rand(64, 256, 128).astype("float32")

    with caplog.at_level(logging.WARNING):
        result = dt_watershed(
            mock_data, blockwise=True, n_threads=8, block_shape=(512, 512, 512)
        )
    assert "fewer than two blocks" in caplog.text
    assert result.shape == mock_data.shape
    assert result.dtype == np.uint64


def test_dt_watershed_blockwise_clamped_block_shape_and_halo(caplog):
    # block_shape/halo exceed the volume size; clamping keeps the blockwise path usable
    mock_data = np.random.rand(64, 256, 128).astype("float32")

    with caplog.at_level(logging.WARNING):
        result = dt_watershed(
            mock_data,
            blockwise=True,
            n_threads=8,
            block_shape=(32, 1024, 1024),
            halo=(100, 100, 100),
        )
    assert "falling back" not in caplog.text
    assert result.shape == mock_data.shape
    assert result.dtype == np.uint64


def _sphere_grid_pmap(
    n_objects: int, pitch: int, radius_sq: float
) -> tuple[np.ndarray, np.ndarray]:
    """Builds a deterministic boundary pmap of a cubic grid of spheres.

    The pmap is 1.0 on the surface voxels of the spheres and 0.0 everywhere else
    (inside and between the spheres).
    """
    n = n_objects * pitch
    centers = np.arange(pitch // 2, n, pitch)
    g = np.arange(n, dtype="float64")

    def axis_d2(c: np.ndarray) -> np.ndarray:
        d = np.abs(g[:, None] - c[None, :])
        return (d * d).min(axis=1)

    d2 = (
        axis_d2(centers)[:, None, None]
        + axis_d2(centers)[None, :, None]
        + axis_d2(centers)[None, None, :]
    )
    inside = d2 <= radius_sq
    eroded = inside.copy()
    eroded[1:, :, :] &= inside[:-1, :, :]
    eroded[:-1, :, :] &= inside[1:, :, :]
    eroded[:, 1:, :] &= inside[:, :-1, :]
    eroded[:, :-1, :] &= inside[:, 1:, :]
    eroded[:, :, 1:] &= inside[:, :, :-1]
    eroded[:, :, :-1] &= inside[:, :, 1:]
    return (inside & ~eroded).astype("float32"), centers


def test_dt_watershed_blockwise_quality_spheres(caplog):
    # 6x6x6 spheres with diameter ~13 and pitch 40 (240^3 volume): no object center or
    # surface lies on an auto-derived block boundary for n_threads=8 (block shape
    # (80, 80, 120)), so object interiors must stay in a single label.
    n_objects, pitch = 6, 40
    pmap, centers = _sphere_grid_pmap(n_objects, pitch, radius_sq=42.0)
    n_objects_total = n_objects**3

    with caplog.at_level(logging.WARNING):
        result = dt_watershed(pmap, blockwise=True, n_threads=8, min_size=10)
    assert "falling back" not in caplog.text
    assert result.shape == pmap.shape
    assert result.dtype == np.uint64

    # over-segmentation (block-boundary fragments) stays bounded
    assert result.max() <= 2.5 * n_objects_total

    # every object interior contains exactly one label
    r = 5
    g = np.arange(-r, r + 1, dtype="float64")
    ball_local = (
        (g * g)[:, None, None] + (g * g)[None, :, None] + (g * g)[None, None, :]
    ) <= r * r
    for cz in centers:
        for cy in centers:
            for cx in centers:
                sub = result[
                    cz - r : cz + r + 1, cy - r : cy + r + 1, cx - r : cx + r + 1
                ]
                labels = np.unique(sub[ball_local])
                labels = labels[labels != 0]
                assert labels.size == 1
