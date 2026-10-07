from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from panseg.io.h5 import create_h5
from panseg.io.io import guess_stack_layout, shape_to_stack_layout, smart_load
from panseg.io.mesh import create_mesh
from panseg.io.tiff import create_tiff
from panseg.io.voxelsize import VoxelSize
from panseg.io.zarr import create_zarr, list_zarr_keys

OME_EXAMPLES = (
    Path(__file__).resolve().parent.parent / "resources" / "ome_tiff_examples"
)


class TestIO:
    voxel_size = VoxelSize(voxels_size=(0.235, 0.15, 0.15))

    def _check_voxel_size(self, voxel_size):
        assert isinstance(voxel_size, VoxelSize)
        np.testing.assert_allclose(
            voxel_size,
            self.voxel_size,
            rtol=1e-5,
            err_msg="Voxel size read from file is not equal to the original voxel size",
        )

    def test_create_read_h5(self, path_h5):
        from panseg.io.h5 import create_h5, load_h5, read_h5_shape, read_h5_voxel_size

        # Create an HDF5 file
        data = np.random.rand(10, 10, 10)
        create_h5(path_h5, data, "raw", voxel_size=self.voxel_size)

        # Create an HDF5 file with empty name causes an error
        with pytest.raises(ValueError):
            create_h5(path_h5, data, "", voxel_size=self.voxel_size)
        with pytest.raises(ValueError):
            create_h5(path_h5, data, None, voxel_size=self.voxel_size)

        # Read the HDF5 file
        data_read = load_h5(path_h5, "raw")
        assert np.array_equal(data, data_read), (
            "Data read from HDF5 file is not equal to the original data"
        )

        # Read the shape of the HDF5 file
        shape = read_h5_shape(path_h5, "raw")
        assert shape == data.shape, (
            "Shape read from HDF5 file is not equal to the original shape"
        )

        # Read the voxel size of the HDF5 file
        voxel_size = read_h5_voxel_size(path_h5, "raw")
        self._check_voxel_size(voxel_size)

        data_read2 = smart_load(path_h5, "raw")
        assert np.array_equal(data, data_read2), (
            "Data read from HDF5 file is not equal to the original data"
        )

    def test_create_read_zarr(self, path_zarr):
        from panseg.io.zarr import (
            create_zarr,
            load_zarr,
            read_zarr_shape,
            read_zarr_voxel_size,
        )

        # Create a Zarr file
        data = np.random.rand(10, 10, 10)
        create_zarr(path_zarr, data, "raw", voxel_size=self.voxel_size)

        # Create a Zarr file with empty name causes an error
        with pytest.raises(ValueError):
            create_zarr(path_zarr, data, "", voxel_size=self.voxel_size)
        with pytest.raises(ValueError):
            create_zarr(path_zarr, data, None, voxel_size=self.voxel_size)

        # Read the Zarr file
        data_read = load_zarr(path_zarr, "raw")
        assert np.array_equal(data, data_read), (
            "Data read from Zarr file is not equal to the original data"
        )

        # Read the shape of the Zarr file
        shape = read_zarr_shape(path_zarr, "raw")
        assert shape == data.shape, (
            "Shape read from Zarr file is not equal to the original shape"
        )

        # Read the voxel size of the Zarr file
        voxel_size = read_zarr_voxel_size(path_zarr, "raw")
        assert np.allclose(voxel_size, self.voxel_size), (
            "Voxel size read from Zarr file is not equal to the original voxel size"
        )

        data_read2 = smart_load(path_zarr, "raw")
        assert np.array_equal(data, data_read2), (
            "Data read from Zarr file is not equal to the original data"
        )

        assert list_zarr_keys(path_zarr) == ["raw"]

    def test_create_read_tiff(self, path_tiff):
        from panseg.io.tiff import create_tiff, load_tiff, read_tiff_voxel_size

        # Create a TIFF file
        data = 255 * np.random.rand(10, 10, 10)
        data = data.astype(np.uint8)
        create_tiff(path_tiff, data, voxel_size=self.voxel_size)

        # Read the TIFF file
        data_read = load_tiff(path_tiff)
        assert np.array_equal(data, data_read), (
            "Data read from TIFF file is not equal to the original data"
        )

        # Read the voxel size of the TIFF file
        voxel_size = read_tiff_voxel_size(path_tiff)
        self._check_voxel_size(voxel_size)

        data_read2 = smart_load(path_tiff)
        assert np.array_equal(data, data_read2), (
            "Data read from TIFF file is not equal to the original data"
        )

    def test_read_jpg(self, path_jpg):
        import numpy as np
        from PIL import Image

        from panseg.io.pil import load_pil

        # Create a JPG file
        data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        image = Image.fromarray(data)
        image.save(path_jpg)

        # Implicitly convrrted from RGB to grayscale
        data_read = load_pil(path_jpg)

        assert data.shape[:2] == data_read.shape, (
            "Data read from JPG file is not equal to the original data"
        )

    def test_mesh(self, tmp_path):
        data = np.zeros((10, 10, 5), dtype=np.uint8)
        data[0:4, 3:5, 0:2] = 1
        data[0:4, 3:5, 2:5] = 2
        data[6:10, 3:5, :] = 3

        out_file = tmp_path / "scene.glb"
        scene = create_mesh(
            path=out_file,
            stack=data,
            voxel_size=self.voxel_size,
            reduction_factor=1.0,
        )
        assert out_file.exists()
        assert len(scene.geometry) == 3
        vc0 = scene.geometry["geometry_0"].visual.vertex_colors
        vc1 = scene.geometry["geometry_1"].visual.vertex_colors
        assert not np.all(vc0[0] == vc1[0])


def test_shape_to_stack_layout_empty():
    assert shape_to_stack_layout(()) == ""


def test_shape_to_stack_layout_yx():
    assert shape_to_stack_layout((200, 300)) == "YX"


def test_shape_to_stack_layout_ycx():
    assert shape_to_stack_layout((200, 4, 300)) == "YCX"


def test_shape_to_stack_layout_zyx():
    assert shape_to_stack_layout((200, 50, 300)) == "ZYX"


def test_shape_to_stack_layout_czyx():
    assert shape_to_stack_layout((8, 200, 50, 300)) == "CZYX"


def test_shape_to_stack_layout_none():
    assert shape_to_stack_layout(None) == ""


@pytest.mark.parametrize(
    "shape",
    [
        (10, 20, 30, 40),
        (16, 32, 64, 128),
        (4, 5, 16, 16),
    ],
)
def test_shape_to_stack_layout_4d_no_channel_returns_empty(shape):
    assert shape_to_stack_layout(shape) == ""


@pytest.mark.parametrize(
    "file_name, axes",
    [
        ("time-series.ome.tif", "TYX"),
        ("4D-series.ome.tif", "TZYX"),
        ("multi-channel-4D-series.ome.tif", "TCZYX"),
    ],
)
def test_guess_stack_layout_ome_tiff(file_name, axes):
    assert guess_stack_layout(OME_EXAMPLES / file_name) == axes


def test_guess_stack_layout_synthetic_ome_axes(make_ome_timeseries):
    # a synthetic OME-TIFF is guessed from its reader axes, like the
    # committed anchors (the widget prefill only copies this guess)
    assert guess_stack_layout(make_ome_timeseries()) == "TZYX"


def test_guess_stack_layout_non_ome_tiff_shape_heuristic(tmp_path):
    path = tmp_path / "out.tiff"
    create_tiff(path, np.empty((10, 20, 30), dtype="float32"), VoxelSize())
    assert guess_stack_layout(path) == "ZYX"


def test_guess_stack_layout_non_ome_tiff_4d_no_small_dim(tmp_path):
    path = tmp_path / "out.tiff"
    create_tiff(
        path, np.empty((10, 12, 16, 24), dtype="float32"), VoxelSize(), layout="ZCYX"
    )
    assert guess_stack_layout(path) == ""


def test_guess_stack_layout_h5_axis_order(tmp_path):
    path = tmp_path / "out.h5"
    create_h5(path, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())
    with h5py.File(path, "a") as f:
        f["raw"].attrs["axis_order"] = "TZYX"
    assert guess_stack_layout(path, key="raw") == "TZYX"
    assert guess_stack_layout(path) == "TZYX"


def test_guess_stack_layout_h5_old_file_shape_heuristic(tmp_path):
    path = tmp_path / "timeseries.h5"
    create_h5(path, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())
    assert guess_stack_layout(path, key="raw") == ""

    path = tmp_path / "volume.h5"
    create_h5(path, np.empty((10, 20, 30), dtype="float32"), "raw", VoxelSize())
    assert guess_stack_layout(path, key="raw") == "ZYX"


def test_guess_stack_layout_zarr_axis_order(tmp_path):
    path = tmp_path / "out.zarr"
    create_zarr(path, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())
    zarr_file = zarr.open_group(store=str(path), mode="a")
    zarr_file["raw"].attrs["axis_order"] = "TZYX"
    assert guess_stack_layout(path, key="raw") == "TZYX"
    assert guess_stack_layout(path) == "TZYX"


def test_guess_stack_layout_zarr_old_file_shape_heuristic(tmp_path):
    path = tmp_path / "timeseries.zarr"
    create_zarr(path, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())
    assert guess_stack_layout(path, key="raw") == ""

    path = tmp_path / "volume.zarr"
    create_zarr(path, np.empty((10, 20, 30), dtype="float32"), "raw", VoxelSize())
    assert guess_stack_layout(path, key="raw") == "ZYX"


def test_guess_stack_layout_pil_shape_heuristic(tmp_path):
    from PIL import Image

    path = tmp_path / "out.png"
    Image.fromarray(np.zeros((128, 128), dtype="uint8")).save(path)
    assert guess_stack_layout(path) == "YX"
