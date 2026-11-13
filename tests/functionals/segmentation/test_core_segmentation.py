import logging
import os

import numpy as np
import pytest

from panseg.functionals.segmentation import dt_watershed
from panseg.functionals.segmentation.segmentation import (
    _FIXED_BYTES_PER_VOXEL,
    _WS_BYTES_PER_VOXEL,
    _available_memory_bytes,
    _blockwise_n_workers,
    _effective_cpu_count,
    _sysconf_first,
)

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


def test_effective_cpu_count():
    assert _effective_cpu_count() >= 1


def test_available_memory_bytes():
    mem = _available_memory_bytes()
    assert mem is None or mem > 0


def test_sysconf_first_falls_back_to_alternate_names(monkeypatch):
    # simulate a platform exposing only the alternate sysconf names (e.g. macOS)
    values = {"SC_PHYS_PAGES": 100, "SC_PAGESIZE": 4096}

    def fake_sysconf(name):
        if name not in values:
            raise ValueError(name)
        return values[name]

    monkeypatch.setattr(os, "sysconf", fake_sysconf, raising=False)
    assert _sysconf_first("SC_AVPHYS_PAGES", "SC_PHYS_PAGES") == 100
    assert _sysconf_first("SC_PAGE_SIZE", "SC_PAGESIZE") == 4096
    assert _sysconf_first("SC_NOT_A_NAME") is None
    assert _available_memory_bytes() == 100 * 4096


def test_blockwise_n_workers_without_detected_memory(caplog, monkeypatch):
    monkeypatch.setattr(
        "panseg.functionals.segmentation.segmentation._available_memory_bytes",
        lambda: None,
    )
    with caplog.at_level(logging.WARNING):
        n_workers = _blockwise_n_workers(2_000_000, (32, 64, 128), (8, 8, 8), 8, None)
    assert "could not determine the available memory" in caplog.text
    assert n_workers == 8


def test_blockwise_n_workers_uncapped():
    assert (
        _blockwise_n_workers(
            2_000_000, (32, 64, 128), (8, 8, 8), 8, memory_budget=10**15
        )
        == 8
    )


def test_blockwise_n_workers_caps_concurrency():
    n_voxels, block_shape, halo, n_threads = 2_000_000, (32, 64, 128), (8, 8, 8), 8
    fixed_bytes = _FIXED_BYTES_PER_VOXEL * n_voxels
    per_block_bytes = 48 * 80 * 144 * _WS_BYTES_PER_VOXEL

    # budget fits exactly three concurrent blocks
    budget = fixed_bytes + 3 * per_block_bytes
    assert _blockwise_n_workers(n_voxels, block_shape, halo, n_threads, budget) == 3

    # budget fits one block, but never fewer than one worker
    assert (
        _blockwise_n_workers(n_voxels, block_shape, halo, n_threads, fixed_bytes + 1)
        == 1
    )


def test_blockwise_n_workers_budget_below_fixed_cost(caplog):
    n_voxels, block_shape, halo, n_threads = 2_000_000, (32, 64, 128), (8, 8, 8), 8
    fixed_bytes = _FIXED_BYTES_PER_VOXEL * n_voxels

    with caplog.at_level(logging.WARNING):
        n_workers = _blockwise_n_workers(
            n_voxels, block_shape, halo, n_threads, fixed_bytes // 2
        )
    assert "exceeds the memory budget" in caplog.text
    assert n_workers == n_threads


def test_dt_watershed_blockwise_memory_budget_cap(caplog):
    # volume large enough (>= 2M voxels) for the real blockwise path to run
    mock_data = np.random.rand(64, 256, 128).astype("float32")

    # budget just above the fixed full-volume cost: concurrency is capped to one block
    fixed_bytes = _FIXED_BYTES_PER_VOXEL * mock_data.size
    with caplog.at_level(logging.INFO):
        result = dt_watershed(
            mock_data,
            blockwise=True,
            n_threads=4,
            memory_budget=fixed_bytes + 1,
        )
    assert "falling back" not in caplog.text
    assert "limiting blockwise watershed to 1 concurrent blocks (from 4 threads)" in (
        caplog.text
    )
    assert result.shape == mock_data.shape
    assert result.dtype == np.uint64
    assert result.max() > result.min() >= 0
    assert len(np.unique(result)) > 1


def test_dt_watershed_blockwise_large_memory_budget(caplog):
    # volume large enough (>= 2M voxels) for the real blockwise path to run
    mock_data = np.random.rand(64, 256, 128).astype("float32")

    with caplog.at_level(logging.INFO):
        result = dt_watershed(
            mock_data, blockwise=True, n_threads=4, memory_budget=10**15
        )
    assert "falling back" not in caplog.text
    assert "limiting blockwise watershed" not in caplog.text
    assert result.shape == mock_data.shape
    assert result.dtype == np.uint64
    assert result.max() > result.min() >= 0
    assert len(np.unique(result)) > 1
