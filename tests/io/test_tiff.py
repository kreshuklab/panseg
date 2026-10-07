import logging
import struct
import warnings
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import tifffile

from panseg.io.tiff import (
    check_ome_single_file,
    create_tiff,
    load_tiff,
    read_ome_axes,
    read_ome_time_spacing,
    read_tiff_shape,
    read_tiff_voxel_size,
)
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
    out = tmp_path / "out.ome.tiff"
    create_tiff(out, data, VoxelSize(), force_bigtiff=True)
    assert out.exists()
    loaded = load_tiff(out)
    assert loaded.shape == data.shape
    assert np.array_equal(loaded, data)


def test_tiff_roundtrip_bigtiff_renaming(tmp_path):
    data = np.array(np.random.random((875, 100, 100)), dtype="float32")
    out = tmp_path / "out.tiff"
    written = tmp_path / "out.ome.tiff"
    create_tiff(out, data, VoxelSize(), force_bigtiff=True)
    assert not out.exists()
    assert written.exists()
    loaded = load_tiff(written)
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
    out = tmp_path / "out.ome.tiff"
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
    out = tmp_path / "out.ome.tiff"
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


# Committed resliced anchors: the three T-bearing OME-TIFFs resliced offline
# to T=4, C=3, Z=2, Y=X=32 (see ome_tiff_examples/reslice_anchors.py). Tests
# load the committed files and never reslice at runtime.
OME_EXAMPLES = (
    Path(__file__).resolve().parent.parent / "resources" / "ome_tiff_examples"
)

COMMITTED_ANCHORS = [
    pytest.param("time-series.ome.tif", "TYX", (4, 32, 32), id="TYX"),
    pytest.param("4D-series.ome.tif", "TZYX", (4, 2, 32, 32), id="TZYX"),
    pytest.param(
        "multi-channel-4D-series.ome.tif", "TCZYX", (4, 3, 2, 32, 32), id="TCZYX"
    ),
]


@pytest.mark.parametrize("file_name, axes, shape", COMMITTED_ANCHORS)
def test_committed_anchor_axes_and_shape(file_name, axes, shape):
    path = OME_EXAMPLES / file_name
    assert path.exists()
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        assert series.axes == axes
        assert series.shape == shape
        data = tiff.asarray()
    assert data.shape == shape
    assert data.dtype == np.int8


@pytest.mark.parametrize("file_name, axes, shape", COMMITTED_ANCHORS)
def test_committed_anchor_no_timing_metadata(file_name, axes, shape):
    path = OME_EXAMPLES / file_name
    with tifffile.TiffFile(path) as tiff:
        root = ElementTree.fromstring(tiff.ome_metadata)
    pixels = _ome_pixels(root)
    assert "TimeIncrement" not in pixels.attrib
    assert "TimeIncrementUnit" not in pixels.attrib
    assert not [e for e in pixels if e.tag.endswith("Plane")]


def test_committed_anchor_license_note():
    note = (OME_EXAMPLES / "LICENSE.md").read_text()
    assert "CC-BY-4.0" in note
    assert "The Open Microscopy Environment" in note


def test_committed_anchor_reslice_script_documented():
    script = OME_EXAMPLES / "reslice_anchors.py"
    assert script.exists()
    text = script.read_text()
    assert "ome_tiff_examples.tar.xz" in text
    assert "time-series.ome.tif" in text
    assert "4D-series.ome.tif" in text
    assert "multi-channel-4D-series.ome.tif" in text


# Synthetic OME-TIFF builders: every timing variant, written into tmp_path at
# test time (none of the committed anchors carries timing metadata).
def _ome_pixels(root):
    image = next(e for e in root if e.tag.endswith("Image"))
    return next(e for e in image if e.tag.endswith("Pixels"))


@pytest.mark.parametrize("unit", ["s", "ms", "min"])
def test_ome_time_increment(make_ome_timeseries, unit):
    path = make_ome_timeseries(t_increment=500, t_increment_unit=unit)
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        assert series.axes == "TZYX"
        assert series.shape == (4, 5, 16, 16)
        root = ElementTree.fromstring(tiff.ome_metadata)
    pixels = _ome_pixels(root)
    assert pixels.get("TimeIncrement") == "500"
    assert pixels.get("TimeIncrementUnit") == unit
    assert not [e for e in pixels if e.tag.endswith("Plane")]


def test_ome_timeseries_uniform_plane_delta_t(make_ome_timeseries):
    path = make_ome_timeseries(plane_delta_t=1000, plane_delta_t_unit="ms")
    with tifffile.TiffFile(path) as tiff:
        root = ElementTree.fromstring(tiff.ome_metadata)
    pixels = _ome_pixels(root)
    assert pixels.get("TimeIncrement") is None
    planes = [e for e in pixels if e.tag.endswith("Plane")]
    assert len(planes) == 4 * 5
    per_timepoint = {}
    for p in planes:
        per_timepoint.setdefault(p.get("TheT"), set()).add(p.get("DeltaT"))
    # realistic absolute acquisition times: the first plane of timepoint i
    # sits at i * 1000 ms
    assert per_timepoint == {
        "0": {"0"},
        "1": {"1000"},
        "2": {"2000"},
        "3": {"3000"},
    }
    assert all(p.get("DeltaTUnit") == "ms" for p in planes)


def test_ome_timeseries_nonuniform_plane_delta_t(make_ome_timeseries):
    path = make_ome_timeseries(nonuniform_plane_delta_t=True)
    with tifffile.TiffFile(path) as tiff:
        root = ElementTree.fromstring(tiff.ome_metadata)
    pixels = _ome_pixels(root)
    assert pixels.get("TimeIncrement") is None
    per_timepoint = {}
    for p in pixels:
        if p.tag.endswith("Plane"):
            per_timepoint.setdefault(p.get("TheT"), set()).add(p.get("DeltaT"))
    assert len(per_timepoint) == 4
    # absolute times whose first-plane differences are non-uniform
    assert per_timepoint == {
        "0": {"0"},
        "1": {"1000"},
        "2": {"3000"},
        "3": {"7000"},
    }


def test_ome_timeseries_no_timing_metadata(make_ome_timeseries):
    path = make_ome_timeseries()
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        root = ElementTree.fromstring(tiff.ome_metadata)
    assert series.axes == "TZYX"
    pixels = _ome_pixels(root)
    assert pixels.get("TimeIncrement") is None
    assert not [e for e in pixels if e.tag.endswith("Plane")]


def test_ome_timeseries_t1_squeezes(make_ome_timeseries):
    path = make_ome_timeseries(axes="TYX", shape=(1, 16, 16))
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        root = ElementTree.fromstring(tiff.ome_metadata)
    pixels = _ome_pixels(root)
    assert pixels.get("SizeT") == "1"
    assert series.axes == "YX"
    assert series.shape == (16, 16)


@pytest.mark.parametrize("file_name, axes, shape", COMMITTED_ANCHORS)
def test_read_ome_axes_committed_anchors(file_name, axes, shape):
    assert read_ome_axes(OME_EXAMPLES / file_name) == axes


def test_read_ome_axes_non_ome_tiff(tmp_path):
    out = tmp_path / "out.tiff"
    create_tiff(out, np.empty((10, 20, 30), dtype="float32"), VoxelSize())
    assert read_ome_axes(out) is None


def test_read_ome_axes_multifile_uses_first_position(ome_timeseries_multifile):
    first, second, _ = ome_timeseries_multifile
    assert read_ome_axes(first) == "TZYX"
    assert read_ome_axes(second) == "TZYX"


@pytest.mark.parametrize("unit", ["s", "ms", "min"])
def test_read_ome_time_spacing_time_increment(make_ome_timeseries, unit):
    path = make_ome_timeseries(t_increment=500, t_increment_unit=unit)
    assert read_ome_time_spacing(path) == (500.0, unit)


def test_read_ome_time_spacing_uniform_plane_delta_t(make_ome_timeseries):
    # DeltaT holds absolute times 0, 1000, 2000, 3000 ms; the spacing is
    # recovered as their uniform difference, not read directly
    path = make_ome_timeseries(plane_delta_t=1000, plane_delta_t_unit="ms")
    assert read_ome_time_spacing(path) == (1000.0, "ms")


def test_read_ome_time_spacing_delta_t_offset_is_differenced_away(
    make_ome_timeseries,
):
    # absolute times not starting at 0: only the differences matter
    path = make_ome_timeseries(
        plane_delta_t=[5, 1005, 2005, 3005], plane_delta_t_unit="ms"
    )
    assert read_ome_time_spacing(path) == (1000.0, "ms")


def test_read_ome_time_spacing_constant_zero_delta_t_warns_unknown(
    make_ome_timeseries,
):
    # constant DeltaT of 0.0 (all absolute times zero) is not a positive
    # spacing: warn and treat as missing, never return 0.0
    path = make_ome_timeseries(plane_delta_t=0.0, plane_delta_t_unit="ms")
    with pytest.warns(UserWarning, match="DeltaT"):
        assert read_ome_time_spacing(path) == (None, "s")


def test_read_ome_time_spacing_partial_delta_t_warns_unknown(make_ome_timeseries):
    # planes documented for every timepoint, but DeltaT absent on one of them
    path = make_ome_timeseries(
        plane_delta_t=[0, None, 2000, 3000], plane_delta_t_unit="ms"
    )
    with pytest.warns(UserWarning, match="DeltaT"):
        assert read_ome_time_spacing(path) == (None, "s")


def test_read_ome_time_spacing_single_timepoint_delta_t_warns_unknown(
    make_ome_timeseries,
):
    # a single timepoint carries no difference to recover a spacing from
    path = make_ome_timeseries(
        axes="TYX", shape=(1, 16, 16), plane_delta_t=1000, plane_delta_t_unit="ms"
    )
    with pytest.warns(UserWarning, match="DeltaT"):
        assert read_ome_time_spacing(path) == (None, "s")


def test_read_ome_time_spacing_nonuniform_plane_delta_t(make_ome_timeseries):
    path = make_ome_timeseries(nonuniform_plane_delta_t=True)
    with pytest.warns(UserWarning, match="DeltaT"):
        assert read_ome_time_spacing(path) == (None, "s")


def test_read_ome_time_spacing_absent(make_ome_timeseries):
    path = make_ome_timeseries()
    assert read_ome_time_spacing(path) == (None, "s")


def test_read_ome_time_spacing_non_ome_tiff(tmp_path):
    out = tmp_path / "out.tiff"
    create_tiff(out, np.empty((10, 20, 30), dtype="float32"), VoxelSize())
    assert read_ome_time_spacing(out) == (None, "s")


_OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _set_tiff_data_uuid(path, uuid_text, file_name=None):
    """Append a UUID child element to the first TiffData of the OME-XML.

    Mirrors the OME 2016-06 schema: UUID is a child of TiffData and FileName
    is an optional attribute of it, defaulting to the opened file.
    """
    with tifffile.TiffFile(path) as tiff:
        root = ElementTree.fromstring(tiff.ome_metadata)
    image = next(e for e in root if e.tag.endswith("Image"))
    pixels = next(e for e in image if e.tag.endswith("Pixels"))
    tiff_data = next(e for e in pixels if e.tag.endswith("TiffData"))
    uuid_el = ElementTree.SubElement(tiff_data, f"{{{_OME_NS}}}UUID")
    uuid_el.text = uuid_text
    if file_name is not None:
        uuid_el.set("FileName", file_name)
    ElementTree.register_namespace("", _OME_NS)
    xml = ElementTree.tostring(root, encoding="unicode")
    with tifffile.TiffFile(path, mode="r+") as tiff:
        tiff.pages[0].tags["ImageDescription"].overwrite(xml.encode("ascii"))


def test_check_ome_single_file_rejects_multifile(ome_timeseries_multifile):
    first, second, _ = ome_timeseries_multifile
    with pytest.raises(ValueError, match="Multi-file OME-TIFF"):
        check_ome_single_file(first)


def test_check_ome_single_file_allows_single_file(make_ome_timeseries):
    check_ome_single_file(make_ome_timeseries())


def test_check_ome_single_file_allows_self_referencing_uuid(
    make_ome_timeseries,
):
    path = make_ome_timeseries()
    _set_tiff_data_uuid(
        path, "urn:uuid:11111111-1111-4111-8111-111111111111", path.name
    )
    check_ome_single_file(path)


def test_check_ome_single_file_allows_uuid_without_file_name(
    make_ome_timeseries,
):
    path = make_ome_timeseries()
    _set_tiff_data_uuid(path, "urn:uuid:11111111-1111-4111-8111-111111111111")
    check_ome_single_file(path)


def test_check_ome_single_file_ignores_non_ome(tmp_path):
    out = tmp_path / "out.tiff"
    create_tiff(out, np.empty((10, 20, 30), dtype="float32"), VoxelSize())
    check_ome_single_file(out)


# --- Time-aware export: a time-bearing image is always written
# as OME-TIFF (the ImageJ branch stays time-less), T fills the T slot of the
# TZCYXS order, TimeIncrement is written only when the spacing is known. ---

# Layouts and shapes of the time-bearing exports (every layout writes a
# 4-timepoint file). The per-layout SizeT/SizeC/SizeZ attribute expectations
# live in TIMESERIES_EXPORT_PIXEL_SIZES and are consumed only by
# test_create_tiff_timeseries_ome_pixel_sizes.
TIMESERIES_EXPORT_CASES = [
    pytest.param("TYX", (4, 16, 16), id="TYX"),
    pytest.param("TCYX", (4, 2, 16, 16), id="TCYX"),
    pytest.param("TZYX", (4, 5, 16, 16), id="TZYX"),
    pytest.param("TCZYX", (4, 2, 5, 16, 16), id="TCZYX"),
]

TIMESERIES_EXPORT_PIXEL_SIZES = {
    "TYX": {"SizeT": "4", "SizeC": "1", "SizeZ": "1"},
    "TCYX": {"SizeT": "4", "SizeC": "2", "SizeZ": "1"},
    "TZYX": {"SizeT": "4", "SizeC": "1", "SizeZ": "5"},
    "TCZYX": {"SizeT": "4", "SizeC": "2", "SizeZ": "5"},
}


@pytest.mark.parametrize("layout,shape", TIMESERIES_EXPORT_CASES)
def test_create_tiff_timeseries_layouts_write_ome(tmp_path, layout, shape):
    data = (np.random.default_rng(0).random(shape) * 100).astype("uint16")
    out = tmp_path / "out.ome.tiff"
    create_tiff(out, data, VoxelSize(voxels_size=(1.0, 1.0, 1.0)), layout=layout)
    with tifffile.TiffFile(out) as tiff:
        assert tiff.series[0].axes == layout
        assert tiff.series[0].shape == shape
        assert tiff.imagej_metadata is None
        loaded = tiff.asarray()
    assert np.array_equal(loaded, data)


@pytest.mark.parametrize("layout,shape", TIMESERIES_EXPORT_CASES)
def test_create_tiff_timeseries_ome_pixel_sizes(tmp_path, layout, shape):
    data = (np.random.default_rng(0).random(shape) * 100).astype("uint16")
    out = tmp_path / "out.ome.tiff"
    voxel_size = VoxelSize(voxels_size=(0.235, 0.15, 0.2))
    create_tiff(out, data, voxel_size, layout=layout)
    with tifffile.TiffFile(out) as tiff:
        pixels = _ome_pixels(ElementTree.fromstring(tiff.ome_metadata))
    for key, value in TIMESERIES_EXPORT_PIXEL_SIZES[layout].items():
        assert pixels.get(key) == value
    assert _assert_no_warnings(read_tiff_voxel_size, out) == voxel_size


@pytest.mark.parametrize("layout,shape", TIMESERIES_EXPORT_CASES)
def test_create_tiff_timeseries_time_increment(tmp_path, layout, shape):
    data = (np.random.default_rng(0).random(shape) * 100).astype("uint16")
    out = tmp_path / "out.ome.tiff"
    create_tiff(
        out,
        data,
        VoxelSize(voxels_size=(1.0, 1.0, 1.0)),
        layout=layout,
        t_spacing=10.5,
    )
    with tifffile.TiffFile(out) as tiff:
        pixels = _ome_pixels(ElementTree.fromstring(tiff.ome_metadata))
    assert pixels.get("SizeT") == "4"
    assert pixels.get("TimeIncrement") == "10.5"
    assert pixels.get("TimeIncrementUnit") == "s"
    assert read_ome_time_spacing(out) == (10.5, "s")


@pytest.mark.parametrize("layout,shape", TIMESERIES_EXPORT_CASES)
def test_create_tiff_timeseries_unknown_t_spacing_no_time_metadata(
    tmp_path, layout, shape
):
    data = (np.random.default_rng(0).random(shape) * 100).astype("uint16")
    out = tmp_path / "out.ome.tiff"
    create_tiff(out, data, VoxelSize(voxels_size=(1.0, 1.0, 1.0)), layout=layout)
    with tifffile.TiffFile(out) as tiff:
        pixels = _ome_pixels(ElementTree.fromstring(tiff.ome_metadata))
    assert pixels.get("SizeT") == "4"
    assert pixels.get("TimeIncrement") is None
    assert pixels.get("TimeIncrementUnit") is None
    assert not [e for e in pixels if e.tag.endswith("Plane")]
    assert read_ome_time_spacing(out) == (None, "s")


def test_create_tiff_timeseries_bigtiff_behavior_unchanged(tmp_path):
    # forced bigtiff stays available for T layouts; BigTIFF is still only
    # chosen when forced or above the 4 GiB boundary
    data = (np.random.default_rng(0).random((4, 5, 16, 16)) * 100).astype("uint16")
    out = tmp_path / "out.tiff"
    written = tmp_path / "out.ome.tiff"
    create_tiff(
        out,
        data,
        VoxelSize(voxels_size=(1.0, 1.0, 1.0)),
        layout="TZYX",
        t_spacing=2.0,
        force_bigtiff=True,
    )
    with tifffile.TiffFile(written) as tiff:
        assert tiff.is_bigtiff
        assert tiff.series[0].axes == "TZYX"
        assert np.array_equal(tiff.asarray(), data)
    assert read_ome_time_spacing(written) == (2.0, "s")


def test_create_tiff_imagej_branch_stays_timeless(tmp_path):
    # a non-T layout keeps the ImageJ writer, even when a t_spacing is passed
    out = tmp_path / "out.tiff"
    create_tiff(
        out,
        np.zeros((5, 16, 16), dtype="uint16"),
        VoxelSize(voxels_size=(1.0, 1.0, 1.0)),
        layout="ZYX",
        t_spacing=10.0,
    )
    with tifffile.TiffFile(out) as tiff:
        assert tiff.imagej_metadata is not None
        assert tiff.ome_metadata is None


def test_ome_timeseries_multifile_chain(ome_timeseries_multifile):
    first, second, data = ome_timeseries_multifile
    assert first.exists()
    assert second.exists()
    with tifffile.TiffFile(first) as tiff:
        series = tiff.series[0]
        root_first = ElementTree.fromstring(tiff.ome_metadata)
        loaded = tiff.asarray()
    with tifffile.TiffFile(second) as tiff:
        root_second = ElementTree.fromstring(tiff.ome_metadata)
    pixels_first = _ome_pixels(root_first)
    assert pixels_first.get("SizeT") == "4"
    tiff_data = [e for e in pixels_first if e.tag.endswith("TiffData")]
    assert len(tiff_data) == 2
    uuid_pairs = {
        (u.text, u.get("FileName"))
        for td in tiff_data
        for u in td
        if u.tag.endswith("UUID")
    }
    assert len(uuid_pairs) == 2
    assert {name for _, name in uuid_pairs} == {first.name, second.name}
    assert _ome_pixels(root_second).get("SizeT") == "2"
    assert series.axes == "TZYX"
    assert series.shape == data.shape == (4, 2, 16, 16)
    np.testing.assert_array_equal(loaded, data)
