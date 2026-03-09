import numpy as np
import pytest

from panseg.io.h5 import create_h5, load_h5, read_h5_shape
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
