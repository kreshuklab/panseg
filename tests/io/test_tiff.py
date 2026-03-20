import numpy as np
import pytest

from panseg.io.tiff import create_tiff, load_tiff, read_tiff_shape
from panseg.io.voxelsize import VoxelSize


@pytest.mark.parametrize("dtype", ["float32", "uint16", "uint8"])
def test_tiff_roundtrip_small(tmp_path):
    out = tmp_path / "out.tiff"
    data = np.array(np.random.random((100, 100, 10)), dtype="float32")
    create_tiff(out, data, VoxelSize())
    assert out.exists()
    loaded = load_tiff(out)
    assert np.array_equal(loaded, data)


def test_tiff_roundtrip_bigtiff(tmp_path):
    data = np.array(np.random.random((875, 100, 100)), dtype="float32")
    out = tmp_path / "out.tiff"
    create_tiff(out, data, VoxelSize(), force_bigtiff=True)
    assert out.exists()
    loaded = load_tiff(out)
    assert loaded.shape == data.shape
    assert np.array_equal(loaded, data)


@pytest.mark.parametrize(
    "in_shape,loaded_shape,layout",
    [
        ((100, 100), (100, 100), "YX"),
        ((100, 100, 100), (100, 100, 100), "ZYX"),
        ((100, 1, 100), (100, 100), "ZYX"),
        ((1, 100, 100), (100, 100), "ZYX"),
        ((1, 100, 100, 1), (100, 100), "CZYX"),
        ((10, 100, 100, 1), (100, 10, 100), "CZYX"),  # reshaped to (t)zcx(y)
    ],
)
def test_read_tiff_shape(tmp_path, in_shape, loaded_shape, layout):
    data = np.empty(in_shape, dtype="float32")
    out = tmp_path / "out.tiff"
    create_tiff(out, data, VoxelSize(), layout=layout, force_bigtiff=False)

    shape = read_tiff_shape(out)
    assert shape == loaded_shape
