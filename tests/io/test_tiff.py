import logging
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest
import tifffile

from panseg.io.tiff import create_tiff, load_tiff, read_tiff_shape, read_tiff_voxel_size
from panseg.io.voxelsize import VoxelSize

OME_DESCRIPTION_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    'xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 '
    'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd" '
    'UUID="urn:uuid:00000000-0000-0000-0000-000000000001">'
)


def _write_ome(path, stack, **pixels_attrs):
    metadata = {"axes": "ZYX"}
    metadata.update(pixels_attrs)
    tifffile.imwrite(path, stack, metadata=metadata, ome=True, photometric="minisblack")


def _overwrite_ome_description(path, body):
    with tifffile.TiffFile(path, mode="r+") as tiff:
        tiff.pages[0].tags["ImageDescription"].overwrite(
            (OME_DESCRIPTION_HEADER + body).encode("ascii")
        )


def _write_imagej_no_resolution_tags(path, x=4, y=3):
    """Minimal handcrafted single-page TIFF with an ImageJ description but no
    XResolution/YResolution tags (tifffile always writes those, so they cannot
    be omitted via imwrite)."""
    description = (
        "ImageJ=1.11a\nimages=1\nslices=1\nhyperstack=true\n"
        "mode=grayscale\nspacing=0.235\nunit=um\n"
    )
    desc = description.encode("ascii") + b"\x00"
    data = np.zeros((y, x), dtype="uint16").tobytes()
    ifd_offset = 8
    ifd_size = 2 + 10 * 12 + 4
    desc_offset = ifd_offset + ifd_size
    data_offset = desc_offset + len(desc)
    tags = [
        (256, 3, 1, struct.pack("<H", x)),
        (257, 3, 1, struct.pack("<H", y)),
        (258, 3, 1, struct.pack("<H", 16)),
        (259, 3, 1, struct.pack("<H", 1)),
        (262, 3, 1, struct.pack("<H", 1)),
        (270, 2, len(desc), struct.pack("<I", desc_offset)),
        (273, 4, 1, struct.pack("<I", data_offset)),
        (277, 3, 1, struct.pack("<H", 1)),
        (278, 4, 1, struct.pack("<I", y)),
        (279, 4, 1, struct.pack("<I", len(data))),
    ]
    out = b"II" + struct.pack("<HI", 42, ifd_offset) + struct.pack("<H", 10)
    for code, typ, count, value in tags:
        out += struct.pack("<HHI", code, typ, count)
        out += value + b"\x00" * (4 - len(value))
    out += struct.pack("<I", 0) + desc + data
    path.write_bytes(out)


def _assert_no_warnings(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args, **kwargs)


@pytest.mark.parametrize("dtype", ["float32", "uint16", "uint8"])
def test_tiff_roundtrip_small(tmp_path, dtype):
    out = tmp_path / "out.tiff"
    data = np.array(np.random.random((100, 100, 10)), dtype=dtype)
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


def test_create_tiff_roundtrip_voxel_size(tmp_path):
    data = np.random.random((10, 20, 30)).astype("float32")
    voxel_size = VoxelSize(voxels_size=(0.235, 0.15, 0.2))
    out = tmp_path / "out.tiff"
    create_tiff(out, data, voxel_size)
    assert _assert_no_warnings(read_tiff_voxel_size, out) == voxel_size


def test_create_tiff_bigtiff_roundtrip_voxel_size(tmp_path):
    data = np.random.random((10, 20, 30)).astype("float32")
    voxel_size = VoxelSize(voxels_size=(0.235, 0.15, 0.2))
    out = tmp_path / "out.tiff"
    create_tiff(out, data, voxel_size, force_bigtiff=True)
    assert _assert_no_warnings(read_tiff_voxel_size, out) == voxel_size
    assert read_tiff_shape(out) == (10, 20, 30)
    assert np.array_equal(load_tiff(out), data)


@pytest.mark.parametrize(
    "layout,shape",
    [("YX", (20, 30)), ("CYX", (2, 20, 30)), ("ZCYX", (10, 2, 20, 30))],
)
def test_create_tiff_bigtiff_roundtrip_voxel_size_layouts(tmp_path, layout, shape):
    data = np.random.random(shape).astype("float32")
    voxel_size = VoxelSize(voxels_size=(0.235, 0.15, 0.2))
    out = tmp_path / "out.tiff"
    create_tiff(out, data, voxel_size, layout=layout, force_bigtiff=True)
    assert _assert_no_warnings(read_tiff_voxel_size, out) == voxel_size
    assert read_tiff_shape(out) == shape
    assert np.array_equal(load_tiff(out), data)


def test_create_tiff_unsupported_layout(tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        create_tiff(
            tmp_path / "out.tiff",
            np.zeros((2, 3, 4), dtype="float32"),
            VoxelSize(),
            layout="XYZ",
        )


def test_create_tiff_wrong_ndim(tmp_path):
    with pytest.raises(AssertionError, match="ZYX order"):
        create_tiff(
            tmp_path / "out.tiff",
            np.zeros((3, 4), dtype="float32"),
            VoxelSize(),
            layout="ZYX",
        )


@pytest.mark.parametrize(
    "in_shape,loaded_shape,layout",
    [
        ((100, 100), (100, 100), "YX"),
        ((100, 100, 100), (100, 100, 100), "ZYX"),
        ((100, 1, 100), (100, 100), "ZYX"),
        ((1, 100, 100), (100, 100), "ZYX"),
        ((2, 100, 100), (2, 100, 100), "CYX"),
        ((10, 2, 100, 100), (10, 2, 100, 100), "ZCYX"),
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


def test_read_tiff_voxel_size_ome(tmp_path):
    stack = np.zeros((2, 3, 4), dtype="uint16")
    out = tmp_path / "out.tiff"
    _write_ome(
        out,
        stack,
        PhysicalSizeX=0.2,
        PhysicalSizeXUnit="um",
        PhysicalSizeY=0.15,
        PhysicalSizeYUnit="um",
        PhysicalSizeZ=0.235,
        PhysicalSizeZUnit="um",
    )
    assert _assert_no_warnings(read_tiff_voxel_size, out) == VoxelSize(
        voxels_size=(0.235, 0.15, 0.2)
    )


@pytest.mark.parametrize("unit", ["\u00b5m", "micrometer"])
def test_read_tiff_voxel_size_ome_unit_variants(tmp_path, unit):
    stack = np.zeros((2, 3, 4), dtype="uint16")
    out = tmp_path / "out.tiff"
    _write_ome(
        out,
        stack,
        PhysicalSizeX=0.2,
        PhysicalSizeXUnit=unit,
        PhysicalSizeY=0.15,
        PhysicalSizeYUnit=unit,
        PhysicalSizeZ=0.235,
        PhysicalSizeZUnit=unit,
    )
    voxel_size = _assert_no_warnings(read_tiff_voxel_size, out)
    assert voxel_size == VoxelSize(voxels_size=(0.235, 0.15, 0.2), unit="um")


def test_read_tiff_voxel_size_ome_missing_sizes(tmp_path):
    stack = np.zeros((2, 3, 4), dtype="uint16")
    out = tmp_path / "out.tiff"
    _write_ome(out, stack)
    with pytest.warns(UserWarning, match="Error parsing omero tiff meta"):
        assert read_tiff_voxel_size(out) == VoxelSize()


def test_read_tiff_voxel_size_ome_missing_image(tmp_path):
    stack = np.zeros((2, 3, 4), dtype="uint16")
    out = tmp_path / "out.tiff"
    _write_ome(out, stack)
    _overwrite_ome_description(out, "<StructuredAnnotations/></OME>")
    with pytest.warns(UserWarning, match="omero tiff meta Image"):
        assert read_tiff_voxel_size(out) == VoxelSize()


def test_read_tiff_voxel_size_ome_missing_pixels(tmp_path):
    stack = np.zeros((2, 3, 4), dtype="uint16")
    out = tmp_path / "out.tiff"
    _write_ome(out, stack)
    _overwrite_ome_description(out, '<Image ID="Image:0"/></OME>')
    with pytest.warns(UserWarning, match="omero tiff meta Pixels"):
        assert read_tiff_voxel_size(out) == VoxelSize()


def test_read_tiff_voxel_size_ome_mixed_units(tmp_path):
    stack = np.zeros((2, 3, 4), dtype="uint16")
    out = tmp_path / "out.tiff"
    _write_ome(
        out,
        stack,
        PhysicalSizeX=0.2,
        PhysicalSizeXUnit="um",
        PhysicalSizeY=150,
        PhysicalSizeYUnit="nm",
        PhysicalSizeZ=0.235,
        PhysicalSizeZUnit="um",
    )
    with pytest.warns(UserWarning, match="Units are not homogeneous") as record:
        voxel_size = read_tiff_voxel_size(out)
    assert any("nm" in str(w.message) for w in record.list)
    assert voxel_size == VoxelSize(voxels_size=(0.235, 150.0, 0.2))


def test_read_tiff_voxel_size_ome_unit_rejected(tmp_path):
    stack = np.zeros((2, 3, 4), dtype="uint16")
    out = tmp_path / "out.tiff"
    _write_ome(
        out,
        stack,
        PhysicalSizeX=200,
        PhysicalSizeXUnit="nm",
        PhysicalSizeY=150,
        PhysicalSizeYUnit="nm",
        PhysicalSizeZ=235,
        PhysicalSizeZUnit="nm",
    )
    with pytest.warns(UserWarning, match="Reverting to default voxel size"):
        assert read_tiff_voxel_size(out) == VoxelSize()


def test_read_tiff_voxel_size_imagej(tmp_path):
    stack = np.zeros((2, 3, 4), dtype="uint16")
    out = tmp_path / "out.tif"
    tifffile.imwrite(
        out,
        stack,
        imagej=True,
        photometric="minisblack",
        resolution=(1.0 / 0.2, 1.0 / 0.15),
        metadata={"axes": "ZYX", "spacing": 0.235, "unit": "um"},
        compression="zlib",
    )
    assert _assert_no_warnings(read_tiff_voxel_size, out) == VoxelSize(
        voxels_size=(0.235, 0.15, 0.2)
    )


def test_read_tiff_voxel_size_imagej_missing_resolution(tmp_path, caplog):
    out = tmp_path / "out.tif"
    _write_imagej_no_resolution_tags(out)
    with caplog.at_level(logging.WARNING, logger="panseg.io.tiff"):
        assert read_tiff_voxel_size(out) == VoxelSize()
    assert "Error parsing imagej tiff meta." in caplog.text


def test_read_tiff_voxel_size_no_metadata(tmp_path):
    stack = np.zeros((2, 3, 4), dtype="uint16")
    out = tmp_path / "out.tiff"
    tifffile.imwrite(out, stack, metadata=None, imagej=False, photometric="minisblack")
    with pytest.warns(UserWarning, match="No metadata found"):
        assert read_tiff_voxel_size(out) == VoxelSize()


def test_read_tiff_shape_resource_rgb_3d():
    path = Path(__file__).resolve().parent.parent / "resources" / "rgb_3D.tif"
    assert read_tiff_shape(path) == (75, 2, 75, 75)


def test_read_tiff_voxel_size_resource_rgb_3d():
    path = Path(__file__).resolve().parent.parent / "resources" / "rgb_3D.tif"
    with pytest.warns(UserWarning, match="No metadata found"):
        assert read_tiff_voxel_size(path) == VoxelSize()
