import h5py
import numpy as np
import pytest

from panseg.io.h5 import (
    create_h5,
    load_h5,
    read_h5_axis_order,
    read_h5_shape,
    read_h5_time_spacing,
)
from panseg.io.voxelsize import VoxelSize


def test_h5_roundtrip_small(tmp_path):
    out = tmp_path / "out.h5"
    data = np.array(np.random.random((50, 50, 10)), dtype="float32")
    create_h5(out, data, "raw", VoxelSize())
    assert out.exists()
    loaded = load_h5(out, key="raw")
    loaded_no_name = load_h5(out, key="raw")
    assert np.array_equal(loaded, data)
    assert np.array_equal(loaded_no_name, data)


@pytest.mark.slow
def test_h5_roundtrip_big(tmp_path):
    data = np.array(np.random.random((875, 700, 2000)), dtype="float32")
    out = tmp_path / "out.h5"
    create_h5(out, data, "raw", VoxelSize())
    assert out.exists()
    loaded = load_h5(out, key="raw")
    assert loaded.shape == data.shape
    assert np.array_equal(loaded, data)


@pytest.mark.parametrize(
    "in_shape,loaded_shape",
    [
        ((100, 100), (100, 100)),
        ((100, 100, 100), (100, 100, 100)),
        ((100, 1, 100), (100, 1, 100)),
        ((1, 100, 100), (1, 100, 100)),
    ],
)
def test_read_tiff_shape(tmp_path, in_shape, loaded_shape):
    data = np.empty(in_shape, dtype="float32")
    out = tmp_path / "out.h5"
    create_h5(out, data, "keyname", VoxelSize())

    shape = read_h5_shape(out)
    assert shape == loaded_shape


def test_read_h5_axis_order_absent(tmp_path):
    out = tmp_path / "out.h5"
    create_h5(out, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())
    assert read_h5_axis_order(out, key="raw") is None


def test_read_h5_axis_order_present(tmp_path):
    out = tmp_path / "out.h5"
    create_h5(out, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())
    with h5py.File(out, "a") as f:
        f["raw"].attrs["axis_order"] = "TZYX"
    assert read_h5_axis_order(out, key="raw") == "TZYX"
    assert read_h5_axis_order(out) == "TZYX"


# --- Time-aware export: PanSeg h5 exports carry the axis order
# on every export and the time spacing attrs when known. ---


def test_create_h5_writes_axis_order_on_every_export(tmp_path):
    out = tmp_path / "out.h5"
    create_h5(
        out,
        np.empty((5, 16, 16), dtype="float32"),
        "raw",
        VoxelSize(),
        axis_order="ZYX",
    )
    assert read_h5_axis_order(out, key="raw") == "ZYX"


def test_create_h5_writes_time_spacing_when_known(tmp_path):
    out = tmp_path / "out.h5"
    create_h5(
        out,
        np.empty((4, 5, 16, 16), dtype="float32"),
        "raw",
        VoxelSize(),
        axis_order="TZYX",
        t_spacing=10.0,
    )
    with h5py.File(out, "r") as f:
        assert f["raw"].attrs["t_spacing"] == 10.0
        assert f["raw"].attrs["t_spacing_unit"] == "s"
    assert read_h5_time_spacing(out, key="raw") == (10.0, "s")


def test_create_h5_unknown_t_spacing_writes_no_time_attrs(tmp_path):
    out = tmp_path / "out.h5"
    create_h5(
        out,
        np.empty((4, 5, 16, 16), dtype="float32"),
        "raw",
        VoxelSize(),
        axis_order="TZYX",
    )
    with h5py.File(out, "r") as f:
        assert "t_spacing" not in f["raw"].attrs
        assert "t_spacing_unit" not in f["raw"].attrs
    assert read_h5_time_spacing(out, key="raw") == (None, "s")


def test_create_h5_time_attrs_gated_on_time_axis(tmp_path):
    # a spacing without a T axis in the axis order writes no time attrs
    out = tmp_path / "out.h5"
    create_h5(
        out,
        np.empty((5, 16, 16), dtype="float32"),
        "raw",
        VoxelSize(),
        axis_order="ZYX",
        t_spacing=10.0,
    )
    with h5py.File(out, "r") as f:
        assert "t_spacing" not in f["raw"].attrs
        assert "t_spacing_unit" not in f["raw"].attrs


def test_read_h5_time_spacing_absent_for_older_files(tmp_path):
    out = tmp_path / "out.h5"
    create_h5(out, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())
    assert read_h5_time_spacing(out, key="raw") == (None, "s")
