"""OME-Zarr (OME-NGFF) IO tests (#622).

The module skips when clearscale is not installed: it is not yet a declared
PanSeg dependency (pending its first PyPI release, #622).
"""

import numpy as np
import pytest
import zarr

from panseg.io.io import guess_stack_layout
from panseg.io.ome_zarr import (
    create_ome_zarr,
    is_ome_zarr,
    list_ome_zarr_keys,
    load_ome_zarr,
    ome_zarr_slot_occupied,
    read_ome_zarr_axis_order,
    read_ome_zarr_shape,
    read_ome_zarr_time_spacing,
    read_ome_zarr_voxel_size,
)
from panseg.io.voxelsize import VoxelSize
from tests.conftest import requires_clearscale

pytestmark = requires_clearscale

SHAPES = (2, 3, 4)


def _write_ome_zarr(
    path,
    layout,
    shape,
    pixel_size=None,
    unit=None,
    version="0.5",
    zarr_format=None,
):
    """Write an OME-Zarr image with clearscale (the library under test)."""
    from clearscale import Multiscale, OmeZarrGroup, Scale

    if zarr_format is None:
        zarr_format = 2 if version == "0.4" else 3
    group = zarr.open_group(store=str(path), mode="w", zarr_format=zarr_format)
    group.create_array("s0", data=np.zeros(shape, dtype="float32"))
    scale = Scale.from_lists(keys=layout, shape=shape, pixel_size=pixel_size, unit=unit)
    group.attrs.update(
        OmeZarrGroup.from_single(
            Multiscale.from_single(scale, scale_key="s0")
        ).to_attrs(version=version)
    )
    return group


def _write_collection(path, keys, layout="zyx", shape=SHAPES):
    """Write a collection store: one image group per key under the root."""
    from clearscale import Multiscale, OmeZarrGroup, Scale

    root = zarr.open_group(store=str(path), mode="w", zarr_format=3)
    for key in keys:
        image_group = root.create_group(key)
        image_group.create_array("s0", data=np.zeros(shape, dtype="float32"))
        scale = Scale.from_lists(layout, shape=shape)
        image_group.attrs.update(
            OmeZarrGroup.from_single(
                Multiscale.from_single(scale, scale_key="s0")
            ).to_attrs(version="0.5")
        )


def test_is_ome_zarr_detection(tmp_path):
    native = tmp_path / "native.zarr"
    group = zarr.open_group(store=str(native), mode="w")
    array = group.create_array(name="raw", shape=SHAPES, dtype="float32")
    array[:] = 0
    assert not is_ome_zarr(native)

    ome = tmp_path / "ome.zarr"
    _write_ome_zarr(ome, "zyx", SHAPES)
    assert is_ome_zarr(ome)

    collection = tmp_path / "collection.zarr"
    _write_collection(collection, ["raw", "segmentation"])
    assert is_ome_zarr(collection)


def test_create_load_roundtrip(tmp_path):
    from clearscale import OmeZarrGroup

    path = tmp_path / "out.ome.zarr"
    data = np.random.random(SHAPES).astype("float32")
    voxel_size = VoxelSize(voxels_size=(0.235, 0.15, 0.2))
    create_ome_zarr(path, data, "raw", voxel_size, axis_order="ZYX")

    assert np.array_equal(load_ome_zarr(path, "raw"), data)
    assert read_ome_zarr_shape(path, "raw") == SHAPES
    assert read_ome_zarr_axis_order(path, "raw") == "ZYX"
    assert read_ome_zarr_voxel_size(path, "raw") == voxel_size
    # the image group carries the NGFF metadata
    group = zarr.open_group(store=str(path), mode="r")["raw"]
    assert "ome" in dict(group.attrs)
    meta = OmeZarrGroup.from_group(group)
    assert meta.version == "0.5"


def test_create_versions(tmp_path):
    path_04 = tmp_path / "v04.ome.zarr"
    create_ome_zarr(
        path_04,
        np.zeros(SHAPES, dtype="float32"),
        "raw",
        VoxelSize(),
        axis_order="ZYX",
        version="0.4",
    )
    attrs = dict(zarr.open_group(store=str(path_04), mode="r")["raw"].attrs)
    assert "multiscales" in attrs
    assert attrs["multiscales"][0]["version"] == "0.4"

    path_05 = tmp_path / "v05.ome.zarr"
    create_ome_zarr(
        path_05,
        np.zeros(SHAPES, dtype="float32"),
        "raw",
        VoxelSize(),
        axis_order="ZYX",
        version="0.5",
    )
    attrs = dict(zarr.open_group(store=str(path_05), mode="r")["raw"].attrs)
    assert attrs["ome"]["version"] == "0.5"


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_create_label_image_group(tmp_path, version):
    # a segmentation is written as a label image below the parent image
    # group's labels container, registered and linked to its source image
    path = tmp_path / "out.ome.zarr"
    labels = np.ones(SHAPES, dtype="uint16")
    create_ome_zarr(
        path,
        labels,
        "raw",
        VoxelSize(),
        axis_order="ZYX",
        version=version,
        label_name="segmentation",
    )

    assert list_ome_zarr_keys(path) == ["raw/labels/segmentation"]
    assert np.array_equal(load_ome_zarr(path, "raw/labels/segmentation"), labels)

    root = zarr.open_group(store=str(path), mode="r")
    # v0.4 keeps the NGFF metadata at the attrs root, v0.5+ under `ome`
    container_meta = dict(root["raw/labels"].attrs).get(
        "ome", dict(root["raw/labels"].attrs)
    )
    assert container_meta["labels"] == ["segmentation"]
    label_meta = dict(root["raw/labels/segmentation"].attrs).get(
        "ome", dict(root["raw/labels/segmentation"].attrs)
    )
    assert label_meta["image-label"]["source"]["image"] == "../.."


def test_label_registration_keeps_existing_names(tmp_path):
    # writing a label into a store that already registers another one adds
    # to the registration instead of replacing it
    path = tmp_path / "out.ome.zarr"
    for name in ("other", "segmentation"):
        create_ome_zarr(
            path,
            np.ones(SHAPES, dtype="uint16"),
            "raw",
            VoxelSize(),
            axis_order="ZYX",
            label_name=name,
        )
    meta = dict(zarr.open_group(store=str(path), mode="r")["raw/labels"].attrs)
    assert meta.get("ome", meta)["labels"] == ["other", "segmentation"]


def test_export_is_order_independent(tmp_path):
    # writing the segmentation before the raw image lands in the same
    # layout as the other way round
    raw = np.random.random(SHAPES).astype("float32")
    labels = np.ones(SHAPES, dtype="uint16")
    voxel_size = VoxelSize(voxels_size=(0.235, 0.15, 0.2))

    raw_first = tmp_path / "raw_first.ome.zarr"
    create_ome_zarr(raw_first, raw, "raw", voxel_size, axis_order="ZYX")
    create_ome_zarr(
        raw_first,
        labels,
        "raw",
        voxel_size,
        axis_order="ZYX",
        label_name="segmentation",
    )

    seg_first = tmp_path / "seg_first.ome.zarr"
    create_ome_zarr(
        seg_first,
        labels,
        "raw",
        voxel_size,
        axis_order="ZYX",
        label_name="segmentation",
    )
    create_ome_zarr(seg_first, raw, "raw", voxel_size, axis_order="ZYX")

    for path in (raw_first, seg_first):
        assert list_ome_zarr_keys(path) == ["raw", "raw/labels/segmentation"]
        assert np.array_equal(load_ome_zarr(path, "raw"), raw)
        assert np.array_equal(load_ome_zarr(path, "raw/labels/segmentation"), labels)
        assert read_ome_zarr_voxel_size(path, "raw") == voxel_size
        assert read_ome_zarr_voxel_size(path, "raw/labels/segmentation") == voxel_size


def test_create_requires_key(tmp_path):
    path = tmp_path / "out.ome.zarr"
    for key in (None, ""):
        with pytest.raises(ValueError, match="Key is required for zarr format"):
            create_ome_zarr(
                path,
                np.zeros(SHAPES, dtype="float32"),
                key,
                VoxelSize(),
                axis_order="ZYX",
            )


@pytest.mark.parametrize("label_name", [None, "segmentation"])
def test_overwrite_requires_flag_and_replaces_data(tmp_path, label_name):
    path = tmp_path / "out.ome.zarr"
    dtype = "uint16" if label_name else "float32"
    first, second = np.zeros(SHAPES, dtype=dtype), np.ones(SHAPES, dtype=dtype)
    write = lambda data, **kw: create_ome_zarr(  # noqa: E731
        path, data, "raw", VoxelSize(), axis_order="ZYX", label_name=label_name, **kw
    )
    read_key = f"raw/labels/{label_name}" if label_name else "raw"

    write(first)
    assert ome_zarr_slot_occupied(path, "raw", label_name)
    with pytest.raises(ValueError, match="already holds"):
        write(second)
    write(second, overwrite=True)
    assert np.array_equal(load_ome_zarr(path, read_key), second)


def test_image_write_with_existing_labels_is_additive(tmp_path):
    # a segmentation alone leaves the parent image slot free
    path = tmp_path / "out.ome.zarr"
    create_ome_zarr(
        path,
        np.ones(SHAPES, dtype="uint16"),
        "raw",
        VoxelSize(),
        axis_order="ZYX",
        label_name="segmentation",
    )
    assert not ome_zarr_slot_occupied(path, "raw", None)
    raw = np.zeros(SHAPES, dtype="float32")
    create_ome_zarr(path, raw, "raw", VoxelSize(), axis_order="ZYX")
    assert np.array_equal(load_ome_zarr(path, "raw"), raw)
    assert ome_zarr_slot_occupied(path, "raw", None)
    assert ome_zarr_slot_occupied(path, "raw", "segmentation")


def test_label_image_requires_integer_dtype(tmp_path):
    with pytest.raises(ValueError, match="integer dtype"):
        create_ome_zarr(
            tmp_path / "out.ome.zarr",
            np.zeros(SHAPES, dtype="float32"),
            "raw",
            VoxelSize(),
            axis_order="ZYX",
            label_name="segmentation",
        )


def test_label_and_image_shape_mismatch_warns(tmp_path):
    # whichever export completes the image/label pair warns about the
    # mismatching shapes
    raw = np.zeros(SHAPES, dtype="float32")
    mismatched = np.ones((2, 3, 5), dtype="uint16")

    label_second = tmp_path / "label_second.ome.zarr"
    create_ome_zarr(label_second, raw, "raw", VoxelSize(), axis_order="ZYX")
    with pytest.warns(UserWarning, match="do not match"):
        create_ome_zarr(
            label_second,
            mismatched,
            "raw",
            VoxelSize(),
            axis_order="ZYX",
            label_name="segmentation",
        )

    image_second = tmp_path / "image_second.ome.zarr"
    create_ome_zarr(
        image_second,
        mismatched,
        "raw",
        VoxelSize(),
        axis_order="ZYX",
        label_name="segmentation",
    )
    with pytest.warns(UserWarning, match="do not match"):
        create_ome_zarr(image_second, raw, "raw", VoxelSize(), axis_order="ZYX")


def test_create_key_writes_nested_image_group(tmp_path):
    path = tmp_path / "out.ome.zarr"
    data = np.zeros(SHAPES, dtype="float32")
    create_ome_zarr(path, data, "a/b", VoxelSize(), axis_order="ZYX")
    assert list_ome_zarr_keys(path) == ["a/b"]
    assert np.array_equal(load_ome_zarr(path, "a/b"), data)
    # the container group itself is not an image
    with pytest.raises(KeyError):
        load_ome_zarr(path, "a")


def test_create_rejects_invalid_axis_order(tmp_path):
    path = tmp_path / "out.ome.zarr"
    with pytest.raises(ValueError, match="Axis order is required"):
        create_ome_zarr(path, np.zeros(SHAPES, dtype="float32"), "raw", VoxelSize())
    with pytest.raises(ValueError, match="not a valid OME-Zarr layout"):
        create_ome_zarr(
            path,
            np.zeros(SHAPES, dtype="float32"),
            "raw",
            VoxelSize(),
            axis_order="TCZYX",
        )
    for axis_order in ("xyz", "zzz"):
        with pytest.raises(ValueError, match="not a valid OME-Zarr layout"):
            create_ome_zarr(
                path,
                np.zeros(SHAPES, dtype="float32"),
                "raw",
                VoxelSize(),
                axis_order=axis_order,
            )


def test_read_axis_order_non_standard_axes_warns(tmp_path):
    path = tmp_path / "weird.ome.zarr"
    _write_ome_zarr(path, "byx", SHAPES)
    with pytest.warns(UserWarning, match="Unsupported NGFF axes"):
        assert read_ome_zarr_axis_order(path) is None


def test_read_voxel_size_converts_units(tmp_path):
    # nanometer voxels read back as micrometers
    path = tmp_path / "nm.ome.zarr"
    _write_ome_zarr(
        path,
        "zyx",
        SHAPES,
        pixel_size=[235.0, 150.0, 200.0],
        unit=["nanometer", "nanometer", "nanometer"],
    )
    assert read_ome_zarr_voxel_size(path, None).voxels_size == pytest.approx(
        (0.235, 0.15, 0.2)
    )


def test_read_voxel_size_missing_units(tmp_path):
    path = tmp_path / "unitless.ome.zarr"
    _write_ome_zarr(path, "zyx", SHAPES)
    with pytest.warns(UserWarning, match="Voxel size not found"):
        assert read_ome_zarr_voxel_size(path, None) == VoxelSize()


def test_read_voxel_size_2d_pads_z(tmp_path):
    path = tmp_path / "two_d.ome.zarr"
    _write_ome_zarr(
        path,
        "yx",
        (3, 4),
        pixel_size=[0.15, 0.2],
        unit=["micrometer", "micrometer"],
    )
    assert read_ome_zarr_voxel_size(path, None) == VoxelSize(
        voxels_size=(1.0, 0.15, 0.2)
    )


def test_read_time_spacing_converts_units(tmp_path):
    path = tmp_path / "minutes.ome.zarr"
    _write_ome_zarr(
        path,
        "tzyx",
        (3, 2, 3, 4),
        pixel_size=[60.0, 0.5, 0.5, 0.5],
        unit=["minute", "micrometer", "micrometer", "micrometer"],
    )
    assert read_ome_zarr_time_spacing(path) == (3600.0, "s")


def test_read_time_spacing_no_time_axis(tmp_path):
    path = tmp_path / "still.ome.zarr"
    _write_ome_zarr(path, "zyx", SHAPES)
    assert read_ome_zarr_time_spacing(path) == (None, "s")


def test_read_time_spacing_unknown_unit_warns(tmp_path):
    path = tmp_path / "weird_time.ome.zarr"
    _write_ome_zarr(
        path,
        "tyx",
        (3, 3, 4),
        pixel_size=[5.0, 0.5, 0.5],
        unit=["fortnight", "micrometer", "micrometer"],
    )
    with pytest.warns(UserWarning, match="Unknown time unit"):
        assert read_ome_zarr_time_spacing(path) == (5.0, "s")


def test_read_ome_zarr_v04(tmp_path):
    path = tmp_path / "v04.ome.zarr"
    _write_ome_zarr(
        path,
        "tzyx",
        (3, 2, 3, 4),
        pixel_size=[10.0, 0.5, 0.5, 0.5],
        unit=["second", "micrometer", "micrometer", "micrometer"],
        version="0.4",
    )
    assert read_ome_zarr_axis_order(path) == "TZYX"
    assert read_ome_zarr_voxel_size(path, None) == VoxelSize(
        voxels_size=(0.5, 0.5, 0.5)
    )
    assert read_ome_zarr_time_spacing(path) == (10.0, "s")


def test_reads_base_level_of_foreign_pyramid(tmp_path):
    # a hand-written ilastik/napari-style v0.4 pyramid with per-level scales
    # and translations: PanSeg reads level 0 only (GLOSSARY: Base level)
    path = tmp_path / "pyramid.ome.zarr"
    group = zarr.open_group(store=str(path), mode="w", zarr_format=2)
    level_0 = np.arange(24, dtype="float32").reshape(2, 3, 4)
    group.create_array("s0", data=level_0)
    group.create_array("s1", data=np.zeros((2, 2, 3), dtype="float32"))
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [0.5, 0.2, 0.2]}
                    ],
                },
                {
                    "path": "s1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 0.4, 0.4]},
                        {"type": "translation", "translation": [1.0, 2.0, 3.0]},
                    ],
                },
            ],
        }
    ]
    assert np.array_equal(load_ome_zarr(path, None), level_0)
    assert read_ome_zarr_shape(path) == (2, 3, 4)
    assert read_ome_zarr_voxel_size(path, None) == VoxelSize(
        voxels_size=(0.5, 0.2, 0.2)
    )


def test_list_keys_root_and_collection(tmp_path):
    root_image = tmp_path / "root.ome.zarr"
    _write_ome_zarr(root_image, "zyx", SHAPES)
    assert list_ome_zarr_keys(root_image) == []
    assert load_ome_zarr(root_image, None).shape == SHAPES

    collection = tmp_path / "collection.ome.zarr"
    _write_collection(collection, ["raw", "segmentation"])
    assert sorted(list_ome_zarr_keys(collection)) == ["raw", "segmentation"]
    assert load_ome_zarr(collection, "segmentation").shape == SHAPES
    with pytest.raises(RuntimeError, match="Ambiguous image groups"):
        load_ome_zarr(collection, None)


def _write_image_with_labels(path, key="raw", version="0.5"):
    """Build the official NGFF image + labels layout with clearscale.

    The layout is what other NGFF tools (napari, ilastik, ome-zarr) produce:
    the image group carries `labels/`, a labels container, and the label
    image with its image-label link.
    """
    from clearscale import ChildRef, FileRef, GroupKind, Multiscale, OmeZarrGroup, Scale
    from clearscale.ome_zarr import ImageLabel, MultiscaleProperties

    root = zarr.open_group(
        store=str(path), mode="w", zarr_format=2 if version == "0.4" else 3
    )
    scale = Scale.from_lists("zyx", SHAPES)
    image = root.create_group(key)
    image.create_array("s0", data=np.zeros(SHAPES, dtype="float32"))
    image.attrs.update(
        OmeZarrGroup.from_single(
            Multiscale.from_single(scale, scale_key="s0")
        ).to_attrs(version=version)
    )
    labels = image.create_group("labels")
    labels.attrs.update(
        OmeZarrGroup(
            kind=GroupKind.LABELS,
            children=(ChildRef("label", FileRef("segmentation")),),
        ).to_attrs(version=version)
    )
    label = labels.create_group("segmentation")
    label.create_array("s0", data=np.ones(SHAPES, dtype="uint16"))
    label.attrs.update(
        OmeZarrGroup.from_single(
            Multiscale(
                {"s0": scale},
                ome=MultiscaleProperties(
                    image_label=ImageLabel(source=FileRef("../.."))
                ),
            )
        ).to_attrs(version=version)
    )


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_reads_official_image_labels_layout(tmp_path, version):
    # the labels container is not a key, the label image is; key=None
    # resolves to the image, not to its label
    path = tmp_path / "official.ome.zarr"
    _write_image_with_labels(path, version=version)

    assert list_ome_zarr_keys(path) == ["raw", "raw/labels/segmentation"]
    assert load_ome_zarr(path, None).dtype == "float32"
    assert load_ome_zarr(path, "raw/labels/segmentation").dtype == "uint16"


def test_labels_only_store_resolves_to_label(tmp_path):
    # the seg-first export layout: a label under a group without an image
    path = tmp_path / "seg_first.ome.zarr"
    labels = np.ones(SHAPES, dtype="uint16")
    create_ome_zarr(
        path, labels, "raw", VoxelSize(), axis_order="ZYX", label_name="segmentation"
    )

    assert list_ome_zarr_keys(path) == ["raw/labels/segmentation"]
    assert np.array_equal(load_ome_zarr(path, None), labels)


def test_ambiguous_message_names_store_root(tmp_path):
    from clearscale import Multiscale, OmeZarrGroup, Scale

    path = tmp_path / "root_and_raw.ome.zarr"
    _write_image_with_labels(path)
    root = zarr.open_group(store=str(path), mode="a")
    root.create_array("s0", data=np.zeros(SHAPES, dtype="float32"))
    root.attrs.update(
        OmeZarrGroup.from_single(
            Multiscale.from_single(Scale.from_lists("zyx", SHAPES), scale_key="s0")
        ).to_attrs(version="0.5")
    )
    with pytest.raises(RuntimeError) as excinfo:
        load_ome_zarr(path, None)
    assert "Ambiguous image groups" in str(excinfo.value)
    assert "(store root)" in str(excinfo.value)
    assert "raw" in str(excinfo.value)


def test_guess_stack_layout_from_ngff_metadata(tmp_path):
    path = tmp_path / "out.ome.zarr"
    _write_ome_zarr(
        path,
        "tzyx",
        (4, 5, 16, 16),
        pixel_size=[10.0, 1.0, 1.0, 1.0],
        unit=["second", "micrometer", "micrometer", "micrometer"],
    )
    assert guess_stack_layout(path, key=None) == "TZYX"
