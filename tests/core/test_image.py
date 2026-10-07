import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from napari.layers import Image

from panseg.core.image import (
    ImageDimensionality,
    ImageLayout,
    ImageProperties,
    ImageType,
    PanSegImage,
    SemanticType,
    crop_to_stack_layout,
    import_image,
    save_image,
    split_stack_layout,
    stack_sort,
)
from panseg.io.h5 import create_h5, read_h5_axis_order, read_h5_time_spacing
from panseg.io.io import guess_stack_layout
from panseg.io.voxelsize import VoxelSize
from tests.conftest import (
    TIMESERIES_PROPS_KNOWN_T_SPACING,
    TIMESERIES_PROPS_UNKNOWN_T_SPACING,
)


# Tests for Enum classes
def test_semantic_type_enum():
    assert SemanticType.RAW.value == "raw"
    assert SemanticType.SEGMENTATION.value == "segmentation"
    assert SemanticType.PREDICTION.value == "prediction"


def test_image_type_enum():
    assert ImageType.IMAGE.value == "image"
    assert ImageType.LABEL.value == "labels"
    assert ImageType.to_choices() == ["image", "labels"]


def test_image_dimensionality_enum():
    assert ImageDimensionality.TWO.value == "2D"
    assert ImageDimensionality.THREE.value == "3D"


def test_image_layout_enum():
    assert ImageLayout.YX.value == "YX"
    assert ImageLayout.CYX.value == "CYX"
    assert ImageLayout.ZYX.value == "ZYX"
    assert ImageLayout.CZYX.value == "CZYX"
    assert ImageLayout.ZCYX.value == "ZCYX"
    assert ImageLayout.TYX.value == "TYX"
    assert ImageLayout.TCYX.value == "TCYX"
    assert ImageLayout.TZYX.value == "TZYX"
    assert ImageLayout.TCZYX.value == "TCZYX"
    assert ImageLayout.to_choices() == [
        "YX",
        "CYX",
        "ZYX",
        "CZYX",
        "ZCYX",
        "TYX",
        "TCYX",
        "TZYX",
        "TCZYX",
    ]


# Derived layout properties across all nine layouts: the layout string alone
# carries every axis, nothing about the axes is stored.
LAYOUT_DERIVED_PROPS = [
    pytest.param(
        ImageLayout.YX, None, None, ImageDimensionality.TWO, False, (0, 1), id="YX"
    ),
    pytest.param(
        ImageLayout.CYX, 0, None, ImageDimensionality.TWO, False, (1, 2), id="CYX"
    ),
    pytest.param(
        ImageLayout.ZYX,
        None,
        None,
        ImageDimensionality.THREE,
        False,
        (0, 1, 2),
        id="ZYX",
    ),
    pytest.param(
        ImageLayout.CZYX,
        0,
        None,
        ImageDimensionality.THREE,
        False,
        (1, 2, 3),
        id="CZYX",
    ),
    pytest.param(
        ImageLayout.ZCYX,
        1,
        None,
        ImageDimensionality.THREE,
        False,
        (0, 2, 3),
        id="ZCYX",
    ),
    pytest.param(
        ImageLayout.TYX, None, 0, ImageDimensionality.TWO, True, (1, 2), id="TYX"
    ),
    pytest.param(
        ImageLayout.TCYX, 1, 0, ImageDimensionality.TWO, True, (2, 3), id="TCYX"
    ),
    pytest.param(
        ImageLayout.TZYX,
        None,
        0,
        ImageDimensionality.THREE,
        True,
        (1, 2, 3),
        id="TZYX",
    ),
    pytest.param(
        ImageLayout.TCZYX,
        1,
        0,
        ImageDimensionality.THREE,
        True,
        (2, 3, 4),
        id="TCZYX",
    ),
]


@pytest.mark.parametrize(
    (
        "layout",
        "channel_axis",
        "time_axis",
        "dimensionality",
        "is_timeseries",
        "spatial_axis_indices",
    ),
    LAYOUT_DERIVED_PROPS,
)
def test_image_properties_derived_layout_props(
    layout,
    channel_axis,
    time_axis,
    dimensionality,
    is_timeseries,
    spatial_axis_indices,
):
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=layout,
        original_voxel_size=voxel_size,
    )
    assert props.channel_axis == channel_axis
    assert props.time_axis == time_axis
    assert props.dimensionality == dimensionality
    assert props.is_timeseries is is_timeseries
    assert layout.spatial_axis_indices == spatial_axis_indices


@pytest.mark.parametrize(
    ("layout", "axis", "expected"),
    [
        pytest.param(ImageLayout.YX, "X", 1, id="YX-X"),
        pytest.param(ImageLayout.CYX, "C", 0, id="CYX-C"),
        pytest.param(ImageLayout.ZCYX, "Z", 0, id="ZCYX-Z"),
        pytest.param(ImageLayout.TCZYX, "T", 0, id="TCZYX-T"),
        pytest.param(ImageLayout.TYX, "Y", 1, id="TYX-Y"),
        pytest.param(ImageLayout.ZYX, "T", None, id="ZYX-T-absent"),
        pytest.param(ImageLayout.YX, "C", None, id="YX-C-absent"),
    ],
)
def test_image_layout_axis_index(layout, axis, expected):
    assert layout.axis_index(axis) == expected


# Tests for ImageProperties class
def test_image_properties_initialization():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert image_props.name == "test_image"
    assert image_props.semantic_type == SemanticType.RAW
    assert image_props.voxel_size == voxel_size
    assert image_props.image_layout == ImageLayout.ZYX
    assert image_props.original_voxel_size == voxel_size


def test_image_properties_dimensionality():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")

    image_props_2d = ImageProperties(
        name="2D_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.YX,
        original_voxel_size=voxel_size,
    )
    assert image_props_2d.dimensionality == ImageDimensionality.TWO

    image_props_3d = ImageProperties(
        name="3D_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert image_props_3d.dimensionality == ImageDimensionality.THREE


def test_image_properties_image_type():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")

    raw_image_props = ImageProperties(
        name="raw_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert raw_image_props.image_type == ImageType.IMAGE

    label_image_props = ImageProperties(
        name="label_image",
        semantic_type=SemanticType.SEGMENTATION,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert label_image_props.image_type == ImageType.LABEL


def test_image_properties_t_spacing_default_unknown():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.TZYX,
        original_voxel_size=voxel_size,
    )
    assert props.t_spacing is None
    assert props.t_unit == "s"


def test_image_properties_t_spacing_seconds():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.TZYX,
        original_voxel_size=voxel_size,
        t_spacing=2.0,
    )
    assert props.t_spacing == 2.0
    assert props.t_unit == "s"


def test_image_properties_t_spacing_unit_normalization():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    kwargs = dict(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.TYX,
        original_voxel_size=voxel_size,
    )
    ms_props = ImageProperties(t_spacing=500.0, t_unit="ms", **kwargs)
    assert ms_props.t_spacing == 0.5
    assert ms_props.t_unit == "s"

    min_props = ImageProperties(t_spacing=2.0, t_unit="min", **kwargs)
    assert min_props.t_spacing == 120.0
    assert min_props.t_unit == "s"

    h_props = ImageProperties(t_spacing=1.5, t_unit="h", **kwargs)
    assert h_props.t_spacing == 5400.0
    assert h_props.t_unit == "s"


def test_image_properties_t_spacing_invalid():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    kwargs = dict(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.TYX,
        original_voxel_size=voxel_size,
    )
    with pytest.raises(ValueError):
        ImageProperties(t_spacing=0.0, **kwargs)
    with pytest.raises(ValueError):
        ImageProperties(t_spacing=5.0, t_unit="lightyears", **kwargs)


def test_image_properties_channel_axis():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")

    cyx_image_props = ImageProperties(
        name="cyx_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.CYX,
        original_voxel_size=voxel_size,
    )
    assert cyx_image_props.channel_axis == 0

    zcyx_image_props = ImageProperties(
        name="zcyx_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZCYX,
        original_voxel_size=voxel_size,
    )
    assert zcyx_image_props.channel_axis == 1


def test_image_properties_interpolation_order():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")

    label_image_props = ImageProperties(
        name="label_image",
        semantic_type=SemanticType.SEGMENTATION,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert label_image_props.interpolation_order() == 0

    raw_image_props = ImageProperties(
        name="raw_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert raw_image_props.interpolation_order() == 1


# Singleton squeeze rule: drop every length-1 axis except Y and X, the
# layout is the projection onto what remains.
SQUEEZE_CASES = [
    pytest.param(
        ImageLayout.TZYX,
        (1, 5, 16, 16),
        ImageLayout.ZYX,
        (5, 16, 16),
        id="T=1 TZYX->ZYX",
    ),
    pytest.param(
        ImageLayout.TCZYX,
        (7, 3, 1, 16, 16),
        ImageLayout.TCYX,
        (7, 3, 16, 16),
        id="Z=1 TCZYX->TCYX",
    ),
    pytest.param(
        ImageLayout.TCZYX,
        (1, 3, 5, 16, 16),
        ImageLayout.CZYX,
        (3, 5, 16, 16),
        id="T=1 TCZYX->CZYX",
    ),
    pytest.param(
        ImageLayout.TCZYX,
        (7, 1, 5, 16, 16),
        ImageLayout.TZYX,
        (7, 5, 16, 16),
        id="C=1 TCZYX->TZYX",
    ),
    pytest.param(
        ImageLayout.TYX, (1, 16, 16), ImageLayout.YX, (16, 16), id="T=1 TYX->YX"
    ),
    pytest.param(
        ImageLayout.ZYX, (1, 16, 16), ImageLayout.YX, (16, 16), id="Z=1 ZYX->YX"
    ),
    pytest.param(
        ImageLayout.CZYX,
        (1, 1, 16, 16),
        ImageLayout.YX,
        (16, 16),
        id="C=1,Z=1 CZYX->YX",
    ),
    pytest.param(
        ImageLayout.TCZYX,
        (1, 1, 5, 16, 16),
        ImageLayout.ZYX,
        (5, 16, 16),
        id="T=1,C=1 TCZYX->ZYX",
    ),
]


@pytest.mark.parametrize(
    "layout, shape, expected_layout, expected_shape", SQUEEZE_CASES
)
def test_construction_squeeze_rule(layout, shape, expected_layout, expected_shape):
    data = np.random.rand(*shape)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=layout,
        original_voxel_size=voxel_size,
    )
    image = PanSegImage(data, props)
    assert image.image_layout == expected_layout
    assert image.shape == expected_shape


def test_construction_squeeze_dropping_t_clears_t_spacing():
    data = np.random.rand(1, 5, 16, 16)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.TZYX,
        original_voxel_size=voxel_size,
        t_spacing=5.0,
    )
    image = PanSegImage(data, props)
    assert image.image_layout == ImageLayout.ZYX
    assert image.is_timeseries is False
    assert image.properties.t_spacing is None


def test_construction_squeeze_data_content():
    data = np.random.rand(1, 5, 16, 16)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.TZYX,
        original_voxel_size=voxel_size,
    )
    image = PanSegImage(data, props)
    np.testing.assert_array_equal(image.get_data(normalize_01=False), data[0])


# Tests for PanSegImage class
def test_panseg_image_initialization():
    data = np.random.rand(10, 10, 10)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PanSegImage(data, image_props)

    assert ps_image.shape == (10, 10, 10)
    assert ps_image.voxel_size == voxel_size
    assert ps_image.name == "test_image"


def test_panseg_image_derive_new():
    data = np.random.rand(10, 10, 10)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PanSegImage(data, image_props)

    new_data = np.random.rand(10, 10, 10)
    new_image = ps_image.derive_new(new_data, name="new_image")

    assert new_image.name == "new_image"
    assert new_image.shape == (10, 10, 10)
    assert new_image.voxel_size == voxel_size
    assert new_image.original_voxel_size == voxel_size


def _make_timeseries_image(t_spacing: float | None = None) -> PanSegImage:
    data = np.random.rand(2, 10, 10)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.TYX,
        original_voxel_size=voxel_size,
        t_spacing=t_spacing,
    )
    return PanSegImage(data, image_props)


def test_panseg_image_set_t_spacing_normalizes_unit():
    ps_image = _make_timeseries_image()

    ps_image.set_t_spacing(500.0, t_unit="ms")

    assert ps_image.properties.t_spacing == 0.5
    assert ps_image.properties.t_unit == "s"


def test_panseg_image_set_t_spacing_preserves_other_properties():
    ps_image = _make_timeseries_image()
    voxel_size = ps_image.voxel_size

    ps_image.set_t_spacing(10.0)

    assert ps_image.name == "test_image"
    assert ps_image.image_layout == ImageLayout.TYX
    assert ps_image.voxel_size == voxel_size
    assert ps_image.semantic_type == SemanticType.RAW
    assert ps_image.shape == (2, 10, 10)


def test_panseg_image_set_t_spacing_clears_known_value():
    ps_image = _make_timeseries_image(t_spacing=10.0)

    ps_image.set_t_spacing(None)

    assert ps_image.properties.t_spacing is None
    assert ps_image.properties.t_unit == "s"


def test_panseg_image_set_t_spacing_rejects_invalid_values():
    ps_image = _make_timeseries_image()

    with pytest.raises(ValueError, match="Time spacing must be positive"):
        ps_image.set_t_spacing(0.0)
    with pytest.raises(ValueError, match="not recognized"):
        ps_image.set_t_spacing(5.0, t_unit="lightyears")
    # a rejected value leaves the previous spacing untouched
    assert ps_image.properties.t_spacing is None


def test_panseg_image_get_data():
    data = np.random.rand(10, 10, 10)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PanSegImage(data, image_props)

    # Test without normalization
    normalised_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-12)
    retrieved_data = ps_image.get_data(normalize_01=True)
    assert normalised_data.dtype == retrieved_data.dtype
    np.testing.assert_allclose(retrieved_data, normalised_data)

    # Test with normalization
    retrieved_data = ps_image.get_data(normalize_01=False)
    assert normalised_data.dtype == retrieved_data.dtype
    np.testing.assert_allclose(retrieved_data, data)


def test_panseg_image_from_napari_layer():
    data = np.random.rand(10, 10, 10)
    voxel_size = (1.0, 1.0, 1.0)
    metadata = {
        "semantic_type": "raw",
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": "ZYX",
        "id": uuid4(),
    }
    napari_layer = Image(data, metadata=metadata, name="test_image")

    ps_image = PanSegImage.from_napari_layer(napari_layer)
    assert ps_image.name == "test_image"
    assert ps_image.shape == (10, 10, 10)
    assert ps_image.voxel_size.voxels_size == voxel_size
    assert tuple(ps_image.voxel_size) == voxel_size


def test_panseg_image_from_napari_layer_timeseries_defaults():
    """A layer created before the time dimension carries no t_spacing/t_unit
    keys in its metadata: it loads as a timeseries with unknown spacing."""
    data = np.random.rand(4, 5, 16, 16).astype("float32")
    voxel_size = (1.0, 1.0, 1.0)
    metadata = {
        "semantic_type": "raw",
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": "TZYX",
        "id": uuid4(),
    }
    napari_layer = Image(data, metadata=metadata, name="old_timeseries")

    ps_image = PanSegImage.from_napari_layer(napari_layer)
    assert ps_image.is_timeseries
    assert ps_image.properties.t_spacing is None
    assert ps_image.properties.t_unit == "s"


def test_panseg_image_to_napari_layer_tuple():
    data = np.random.rand(2, 2, 2)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PanSegImage(data, image_props)
    layer_tuple = ps_image.to_napari_layer_tuple()

    assert isinstance(layer_tuple, tuple)
    layer_tuple = tuple(layer_tuple)
    np.testing.assert_allclose(layer_tuple[0], ps_image.get_data(normalize_01=False))
    assert "metadata" in layer_tuple[1]
    assert layer_tuple[2] == ps_image.image_type.value


@pytest.mark.parametrize(
    "image_layout, shape, expected_axis_labels",
    [
        (ImageLayout.YX, (4, 5), ["y", "x"]),
        (ImageLayout.CYX, (2, 4, 5), ["c", "y", "x"]),
        (ImageLayout.ZYX, (3, 4, 5), ["z", "y", "x"]),
        (ImageLayout.CZYX, (2, 3, 4, 5), ["c", "z", "y", "x"]),
        (ImageLayout.TYX, (7, 4, 5), ["t", "y", "x"]),
        (ImageLayout.TCYX, (7, 2, 4, 5), ["t", "c", "y", "x"]),
        (ImageLayout.TZYX, (7, 3, 4, 5), ["t", "z", "y", "x"]),
        (ImageLayout.TCZYX, (7, 2, 3, 4, 5), ["t", "c", "z", "y", "x"]),
    ],
)
def test_to_napari_layer_tuple_axis_labels(image_layout, shape, expected_axis_labels):
    data = np.random.rand(*shape)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=image_layout,
        original_voxel_size=voxel_size,
    )
    ps_image = PanSegImage(data, image_props)
    layer_tuple = tuple(ps_image.to_napari_layer_tuple())

    assert layer_tuple[1]["axis_labels"] == expected_axis_labels

    # napari accepts the layer tuple with the axis labels
    layer = Image(layer_tuple[0], **layer_tuple[1])
    assert layer.data.shape == shape


def test_panseg_image_scale_property():
    data = np.random.rand(10, 10, 10)
    voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="scaled_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PanSegImage(data, image_props)
    assert ps_image.scale == (0.5, 1.0, 1.0)


@pytest.mark.parametrize("t_spacing", [0.5, None])
def test_scale_tzyx_t_axis_is_timepoint_indices(t_spacing):
    """The t axis shows the existing timepoints, not elapsed time: the t
    scale stays 1.0 whether the spacing is known or not."""
    data = np.random.rand(7, 5, 16, 16)
    voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 2.0), unit="um")
    image_props = ImageProperties(
        name="timeseries",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.TZYX,
        original_voxel_size=voxel_size,
        t_spacing=t_spacing,
    )
    ps_image = PanSegImage(data, image_props)
    assert ps_image.scale == (1.0, 0.5, 1.0, 2.0)


def test_scale_tcyx():
    data = np.random.rand(7, 3, 16, 16)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 2.0), unit="um")
    image_props = ImageProperties(
        name="timeseries",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.TCYX,
        original_voxel_size=voxel_size,
        t_spacing=2.0,
    )
    ps_image = PanSegImage(data, image_props)
    assert ps_image.scale == (1.0, 1.0, 1.0, 2.0)


def test_requires_scaling():
    data = np.random.rand(10, 10, 10)

    voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    same_voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    original_voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    assert same_voxel_size == voxel_size
    assert original_voxel_size != voxel_size

    image_props = ImageProperties(
        name="scaled_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=original_voxel_size,
    )
    ps_image = PanSegImage(data, image_props)
    assert ps_image.requires_scaling is True

    image_props = ImageProperties(
        name="scaled_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=same_voxel_size,
    )
    ps_image = PanSegImage(data, image_props)
    assert ps_image.requires_scaling is False

    assert same_voxel_size == voxel_size
    assert original_voxel_size != voxel_size


@pytest.fixture
def test_h5_dir():
    return Path(__file__).parent.parent / "resources" / "training_sources"


def test_import_image_YX(test_h5_dir):
    file = test_h5_dir / "train_2D_2D.h5"
    image = import_image(
        path=file,
        key="raw",
        stack_layout="YX",
    )
    assert isinstance(image, PanSegImage)
    assert image.semantic_type == SemanticType.RAW
    assert image.image_layout == ImageLayout.YX


def test_import_image_CYX(test_h5_dir):
    file = test_h5_dir / "train_2Dc_2D.h5"
    images = import_image(
        path=file,
        key="raw",
        stack_layout="CYX",
    )
    assert isinstance(images, list)
    assert all([isinstance(i, PanSegImage) for i in images])
    assert len(images) == 2
    assert images[0].semantic_type == SemanticType.RAW
    assert images[0].image_layout == ImageLayout.YX


def test_import_image_CYX_warning(mocker, test_h5_dir):
    mocker.patch("panseg.core.image.last_warning", new=0.0)
    mock_loader = mocker.patch("panseg.core.image.smart_load_with_vs")
    mock_loader.return_value = (
        np.random.rand(3, 2, 11),
        VoxelSize(voxels_size=(1.0, 1.0, 1.0)),
    )
    file = test_h5_dir / "train_3Dc_3D.h5"
    with pytest.raises(ValueError):
        import_image(
            path=file,
            key="raw",
            stack_layout="CYX",
        )


def test_import_image_YX_error(test_h5_dir):
    file = test_h5_dir / "train_2D_2D.h5"
    with pytest.raises(ValueError):
        import_image(
            path=file,
            key="raw",
            stack_layout="ZYX",
        )


def test_import_image_ZYX(test_h5_dir):
    file = test_h5_dir / "train_3D_3D.h5"
    image = import_image(
        path=file,
        key="raw",
        stack_layout="ZYX",
    )
    assert image.semantic_type == SemanticType.RAW
    assert image.image_layout == ImageLayout.ZYX


def test_import_image_ZYX_inv(test_h5_dir):
    file = test_h5_dir / "train_3D_3D.h5"
    image = import_image(
        path=file,
        key="raw",
        stack_layout="Z-YX",
    )
    assert image.semantic_type == SemanticType.RAW
    assert image.image_layout == ImageLayout.ZYX


def test_import_image_CZYX(test_h5_dir):
    file = test_h5_dir / "train_3Dc_3D.h5"
    images = import_image(
        path=file,
        key="raw",
        stack_layout="CZYX",
    )
    assert isinstance(images, list)
    assert all([isinstance(i, PanSegImage) for i in images])
    assert len(images) == 2
    assert images[0].semantic_type == SemanticType.RAW
    assert images[0].image_layout == ImageLayout.ZYX


def test_import_image_CZYX_warning(mocker, test_h5_dir):
    mocker.patch("panseg.core.image.last_warning", new=0.0)
    mock_loader = mocker.patch("panseg.core.image.smart_load_with_vs")
    mock_loader.return_value = (
        np.random.rand(3, 2, 10, 11),
        VoxelSize(voxels_size=(1.0, 1.0, 1.0)),
    )
    file = test_h5_dir / "train_3Dc_3D.h5"
    with pytest.raises(ValueError):
        import_image(
            path=file,
            key="raw",
            stack_layout="CZYX",
        )


def test_import_image_ZCYX(mocker, test_h5_dir):
    mocker.patch("panseg.core.image.last_warning", new=time.time())
    file = test_h5_dir / "train_3Dc_3D.h5"
    images = import_image(
        path=file,
        key="raw",
        stack_layout="ZCYX",
    )
    assert isinstance(images, list)
    assert all([isinstance(i, PanSegImage) for i in images])
    assert len(images) == 75
    assert images[0].semantic_type == SemanticType.RAW
    assert images[0].image_layout == ImageLayout.ZYX


def test_import_image_ZCYX_warning(mocker, test_h5_dir):
    mocker.patch("panseg.core.image.last_warning", new=0.0)
    file = test_h5_dir / "train_3Dc_3D.h5"
    with pytest.raises(ValueError):
        import_image(
            path=file,
            key="raw",
            stack_layout="ZCYX",
        )


OME_EXAMPLES = (
    Path(__file__).resolve().parent.parent / "resources" / "ome_tiff_examples"
)


@pytest.mark.parametrize(
    "file_name, layout, shape",
    [
        ("time-series.ome.tif", ImageLayout.TYX, (4, 32, 32)),
        ("4D-series.ome.tif", ImageLayout.TZYX, (4, 2, 32, 32)),
    ],
)
def test_import_image_ome_anchor_single(file_name, layout, shape):
    image = import_image(path=OME_EXAMPLES / file_name, stack_layout=layout.name)
    assert isinstance(image, PanSegImage)
    assert image.image_layout == layout
    assert image.shape == shape
    assert image.is_timeseries
    assert image.properties.t_spacing is None


def test_import_image_ome_anchor_multichannel(mocker):
    # C=3 > Z=2 trips the channel heuristic; pin the throttle like
    # test_import_image_ZCYX so the metadata-driven import passes
    mocker.patch("panseg.core.image.last_warning", new=time.time())
    images = import_image(
        path=OME_EXAMPLES / "multi-channel-4D-series.ome.tif", stack_layout="TCZYX"
    )
    assert isinstance(images, list)
    assert len(images) == 3
    for ch, image in enumerate(images):
        assert image.image_layout == ImageLayout.TZYX
        assert image.shape == (4, 2, 32, 32)
        assert image.properties.t_spacing is None
        assert image.name == f"image_{ch}"


@pytest.mark.parametrize(
    "unit, expected_t_spacing",
    [
        ("s", 500.0),
        ("ms", 0.5),
        ("min", 30000.0),
    ],
)
def test_import_image_ome_time_increment_units(
    make_ome_timeseries, unit, expected_t_spacing
):
    path = make_ome_timeseries(t_increment=500, t_increment_unit=unit)
    image = import_image(path=path, stack_layout="TZYX")
    assert image.image_layout == ImageLayout.TZYX
    assert image.properties.t_spacing == expected_t_spacing
    assert image.properties.t_unit == "s"


def test_import_image_ome_uniform_plane_delta_t(make_ome_timeseries):
    # DeltaT holds absolute times 0, 1000, 2000, 3000 ms; the uniform 1000 ms
    # difference is the spacing, converted to 1.0 s by ImageProperties
    path = make_ome_timeseries(plane_delta_t=1000, plane_delta_t_unit="ms")
    image = import_image(path=path, stack_layout="TZYX")
    assert image.properties.t_spacing == 1.0
    assert image.properties.t_unit == "s"


def test_import_image_ome_constant_zero_delta_t_imports_unknown(make_ome_timeseries):
    # regression: a constant-0.0 DeltaT must not reach the positivity guard
    # as spacing 0.0; the file imports with unknown t_spacing instead
    path = make_ome_timeseries(plane_delta_t=0.0, plane_delta_t_unit="ms")
    with pytest.warns(UserWarning, match="DeltaT"):
        image = import_image(path=path, stack_layout="TZYX")
    assert image.image_layout == ImageLayout.TZYX
    assert image.properties.t_spacing is None


def test_import_image_ome_nonuniform_plane_delta_t_warns_unknown(make_ome_timeseries):
    path = make_ome_timeseries(nonuniform_plane_delta_t=True)
    with pytest.warns(UserWarning, match="DeltaT"):
        image = import_image(path=path, stack_layout="TZYX")
    assert image.image_layout == ImageLayout.TZYX
    assert image.properties.t_spacing is None


def test_import_image_ome_absent_timing_metadata_unknown(make_ome_timeseries):
    path = make_ome_timeseries()
    image = import_image(path=path, stack_layout="TZYX")
    assert image.image_layout == ImageLayout.TZYX
    assert image.properties.t_spacing is None


def test_import_image_ome_multichannel_keeps_t_spacing(make_ome_timeseries):
    path = make_ome_timeseries(axes="TCZYX", t_increment=500, t_increment_unit="ms")
    images = import_image(path=path, stack_layout="TCZYX")
    assert isinstance(images, list)
    assert len(images) == 2
    for image in images:
        assert image.image_layout == ImageLayout.TZYX
        assert image.properties.t_spacing == 0.5


def test_import_image_ome_t1_imports_squeezed(make_ome_timeseries):
    path = make_ome_timeseries(axes="TYX", shape=(1, 16, 16))
    image = import_image(path=path, stack_layout="YX")
    assert image.image_layout == ImageLayout.YX
    assert image.shape == (16, 16)
    assert not image.is_timeseries


def test_import_image_ome_multifile_rejected(ome_timeseries_multifile):
    first, second, _ = ome_timeseries_multifile
    with pytest.raises(ValueError, match="Multi-file OME-TIFF"):
        import_image(path=first, stack_layout="TZYX")


def test_import_image_tzyx_single(make_ome_timeseries):
    path = make_ome_timeseries()
    image = import_image(path=path, stack_layout="TZYX")
    assert isinstance(image, PanSegImage)
    assert image.image_layout == ImageLayout.TZYX
    assert image.shape == (4, 5, 16, 16)


def test_import_image_tyx_single(make_ome_timeseries):
    path = make_ome_timeseries(axes="TYX")
    image = import_image(path=path, stack_layout="TYX")
    assert isinstance(image, PanSegImage)
    assert image.image_layout == ImageLayout.TYX
    assert image.shape == (4, 16, 16)


def test_import_image_tczyx_splits_channels(make_ome_timeseries):
    path = make_ome_timeseries(axes="TCZYX")
    images = import_image(path=path, stack_layout="TCZYX")
    assert isinstance(images, list)
    assert len(images) == 2
    for ch, image in enumerate(images):
        assert image.image_layout == ImageLayout.TZYX
        assert image.shape == (4, 5, 16, 16)
        assert image.name == f"image_{ch}"


def test_import_image_tcyx_splits_channels(make_ome_timeseries):
    path = make_ome_timeseries(axes="TCYX")
    images = import_image(path=path, stack_layout="TCYX")
    assert isinstance(images, list)
    assert len(images) == 2
    for image in images:
        assert image.image_layout == ImageLayout.TYX
        assert image.shape == (4, 16, 16)


# The T-bearing multichannel branches carry the same mislabel guard as
# CYX/CZYX: a "C" axis longer than 9 means the layout string does not match
# the data. The module-global throttle (last_warning) must be reset or the
# raise fires only once per 120s across tests.
def test_import_image_tcyx_warning(mocker, make_ome_timeseries):
    mocker.patch("panseg.core.image.last_warning", new=0.0)
    mock_loader = mocker.patch("panseg.core.image.smart_load_with_vs")
    mock_loader.return_value = (
        np.random.rand(4, 10, 16, 16),
        VoxelSize(voxels_size=(1.0, 1.0, 1.0)),
    )
    file = make_ome_timeseries(axes="TCYX", shape=(4, 10, 16, 16))
    with pytest.raises(ValueError, match="Double check the stack layout"):
        import_image(path=file, stack_layout="TCYX")


def test_import_image_tczyx_warning(mocker, make_ome_timeseries):
    mocker.patch("panseg.core.image.last_warning", new=0.0)
    mock_loader = mocker.patch("panseg.core.image.smart_load_with_vs")
    mock_loader.return_value = (
        np.random.rand(4, 10, 5, 16, 16),
        VoxelSize(voxels_size=(1.0, 1.0, 1.0)),
    )
    file = make_ome_timeseries(axes="TCZYX", shape=(4, 10, 5, 16, 16))
    with pytest.raises(ValueError, match="Double check the stack layout"):
        import_image(path=file, stack_layout="TCZYX")


def test_import_image_tzyx_slicing_t_first(make_ome_timeseries):
    path = make_ome_timeseries(shape=(4, 5, 20, 60))
    image = import_image(path=path, stack_layout="TZYX[:3,:,:,:50]")
    assert image.image_layout == ImageLayout.TZYX
    assert image.shape == (3, 5, 20, 50)


def test_import_image_tzyx_spatial_slicing(make_ome_timeseries):
    path = make_ome_timeseries(shape=(4, 5, 20, 60))
    image = import_image(path=path, stack_layout="TZYX[:,1:3,10:,10:20]")
    assert image.image_layout == ImageLayout.TZYX
    assert image.shape == (4, 2, 10, 10)


def test_import_image_tzyx_length_one_t_slice_squeezes(make_ome_timeseries):
    path = make_ome_timeseries()
    image = import_image(path=path, stack_layout="TZYX[:1,:,:,:]")
    assert image.image_layout == ImageLayout.ZYX
    assert image.shape == (5, 16, 16)
    assert image.properties.t_spacing is None


def test_split_stack_layout_letters_and_slice():
    axes, slicing = split_stack_layout("txyz[:3,:,:]")
    assert axes == "txyz"
    assert slicing is not None
    assert slicing.split(",") == [":3", ":", ":"]

    assert split_stack_layout("zyx") == ("zyx", None)
    assert split_stack_layout("-TZYX[0]") == ("-TZYX", "0")


def test_split_stack_layout_rejects_malformed_spec():
    with pytest.raises(ValueError, match="is not understood"):
        split_stack_layout("t zyx")
    with pytest.raises(ValueError, match="is not understood"):
        split_stack_layout("tzyx[:2,:, :")


def test_split_stack_layout_rejects_unknown_axis_letter():
    with pytest.raises(ValueError, match="unknown axis letter"):
        split_stack_layout("ZBA")


def test_import_image_inline_slice_uses_user_layout_order(tmp_path):
    """Entry 0 belongs to the first letter of the layout as written (z here),
    not to the first letter of the canonical TZYX it is sorted into."""
    path = tmp_path / "non_canonical_timeseries.h5"
    create_h5(path, np.zeros((6, 3, 8, 8), dtype="float32"), "raw", VoxelSize())

    image = import_image(path=path, key="raw", stack_layout="ztyx[:2,:,:,:]")
    assert image.image_layout == ImageLayout.TZYX
    assert image.shape == (3, 2, 8, 8)


def test_import_image_inline_slice_leaves_unlisted_axes_whole(tmp_path):
    """Fewer entries than axes: only t, x and y are indexed, z stays whole."""
    path = tmp_path / "timeseries.h5"
    create_h5(path, np.zeros((3, 64, 64, 5), dtype="float32"), "raw", VoxelSize())

    image = import_image(path=path, key="raw", stack_layout="txyz[:2,:,:]")
    assert image.image_layout == ImageLayout.TZYX
    assert image.shape == (2, 5, 64, 64)


def test_import_image_inline_slice_integer_drops_axis(tmp_path):
    path = tmp_path / "timeseries.h5"
    create_h5(path, np.zeros((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())

    image = import_image(path=path, key="raw", stack_layout="tzyx[0,:,:,:]")
    assert image.image_layout == ImageLayout.ZYX
    assert image.shape == (5, 16, 16)
    assert image.properties.t_spacing is None
    assert not image.is_timeseries


def test_import_image_inline_slice_with_step(tmp_path):
    path = tmp_path / "timeseries.h5"
    create_h5(path, np.zeros((6, 5, 16, 16), dtype="float32"), "raw", VoxelSize())

    image = import_image(path=path, key="raw", stack_layout="tzyx[0:6:2,:,:,:]")
    assert image.image_layout == ImageLayout.TZYX
    assert image.shape == (3, 5, 16, 16)


def test_import_image_rejects_more_slice_entries_than_axes(tmp_path):
    path = tmp_path / "timeseries.h5"
    create_h5(path, np.zeros((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())

    with pytest.raises(ValueError, match="entries but the stack layout"):
        import_image(path=path, key="raw", stack_layout="tzyx[:2,:,:,:,:]")


def test_crop_to_stack_layout_drops_axis_with_inversion_marker():
    data = np.zeros((3, 4, 5, 6))

    layout, cropped = crop_to_stack_layout(data, "tx-yz", ":,:,1,:")
    assert layout == "txz"
    assert cropped.shape == (3, 4, 6)


def test_import_image_t_layout_non_ome_stays_unknown(tmp_path):
    """T layouts are accepted for every format: the layout is the user's
    explicit assertion about their data; the spacing stays unknown for
    formats that cannot carry it."""
    path = tmp_path / "timeseries.h5"
    create_h5(path, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())

    image = import_image(path=path, key="raw", stack_layout="TZYX")
    assert isinstance(image, PanSegImage)
    assert image.image_layout == ImageLayout.TZYX
    assert image.shape == (4, 5, 16, 16)
    assert image.properties.t_spacing is None
    assert image.properties.t_unit == "s"


def test_import_image_tczyx_non_ome_splits_channels(tmp_path):
    path = tmp_path / "multichannel_timeseries.h5"
    create_h5(path, np.empty((4, 2, 5, 16, 16), dtype="float32"), "raw", VoxelSize())

    images = import_image(path=path, key="raw", stack_layout="TCZYX")
    assert isinstance(images, list)
    assert len(images) == 2
    for ch, image in enumerate(images):
        assert image.image_layout == ImageLayout.TZYX
        assert image.shape == (4, 5, 16, 16)
        assert image.name == f"image_{ch}"
        assert image.properties.t_spacing is None


def _timeseries_ps_image(
    data: np.ndarray,
    layout: str,
    t_spacing: float | None = None,
    semantic_type: SemanticType = SemanticType.RAW,
    name: str = "image",
) -> PanSegImage:
    """A PanSegImage on the shared timeseries skeleton: unit-ish voxel size
    and the given layout/t_spacing; split/merge and export tests build on
    it instead of repeating the ImageProperties boilerplate."""
    voxel_size = VoxelSize(voxels_size=(0.235, 0.15, 0.15), unit="um")
    return PanSegImage(
        data=data,
        properties=ImageProperties(
            name=name,
            semantic_type=semantic_type,
            voxel_size=voxel_size,
            image_layout=ImageLayout(layout),
            original_voxel_size=voxel_size,
            t_spacing=t_spacing,
        ),
    )


def test_split_image_CZYX():
    data = np.random.rand(3, 9, 10, 11)
    voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.CZYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PanSegImage(data, image_props)
    splits = ps_image.split_channels()

    assert len(splits) == 3
    assert all([s.image_layout == ImageLayout.ZYX for s in splits])
    assert all([s.semantic_type == SemanticType.RAW for s in splits])
    assert all([s.voxel_size == voxel_size for s in splits])
    assert all([s.shape == (9, 10, 11) for s in splits])


def test_split_image_ZCYX():
    data = np.random.rand(9, 4, 10, 11)
    voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZCYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PanSegImage(data, image_props)
    splits = ps_image.split_channels()

    assert len(splits) == 4
    assert all([s.image_layout == ImageLayout.ZYX for s in splits])
    assert all([s.semantic_type == SemanticType.RAW for s in splits])
    assert all([s.voxel_size == voxel_size for s in splits])
    assert all([s.shape == (9, 10, 11) for s in splits])


def test_split_image_CYX():
    data = np.random.rand(4, 10, 11)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.CYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PanSegImage(data, image_props)
    splits = ps_image.split_channels()

    assert len(splits) == 4
    assert all([s.image_layout == ImageLayout.YX for s in splits])
    assert all([s.semantic_type == SemanticType.RAW for s in splits])
    assert all([s.voxel_size == voxel_size for s in splits])
    assert all([s.shape == (10, 11) for s in splits])


def test_split_image_TCZYX():
    ps_image = _timeseries_ps_image(
        np.random.rand(7, 3, 5, 16, 16), "TCZYX", t_spacing=0.5
    )
    splits = ps_image.split_channels()

    assert len(splits) == 3
    assert [s.name for s in splits] == ["image_0", "image_1", "image_2"]
    assert all([s.image_layout == ImageLayout.TZYX for s in splits])
    assert all([s.is_timeseries for s in splits])
    assert all([s.shape == (7, 5, 16, 16) for s in splits])
    assert all([s.properties.t_spacing == 0.5 for s in splits])


def test_split_image_TCYX():
    ps_image = _timeseries_ps_image(np.random.rand(7, 4, 16, 16), "TCYX", t_spacing=0.5)
    splits = ps_image.split_channels()

    assert len(splits) == 4
    assert [s.name for s in splits] == ["image_0", "image_1", "image_2", "image_3"]
    assert all([s.image_layout == ImageLayout.TYX for s in splits])
    assert all([s.is_timeseries for s in splits])
    assert all([s.shape == (7, 16, 16) for s in splits])
    assert all([s.properties.t_spacing == 0.5 for s in splits])


def test_split_image_TZYX_not_split():
    ps_image = _timeseries_ps_image(np.random.rand(7, 5, 16, 16), "TZYX")
    assert ps_image.split_channels() == [ps_image]

    ps_image = _timeseries_ps_image(np.random.rand(7, 16, 16), "TYX")
    assert ps_image.split_channels() == [ps_image]


def test_merge_images_2d():
    data = np.random.rand(10, 11)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.YX,
        original_voxel_size=voxel_size,
    )
    ps_image_1 = PanSegImage(data, image_props)
    ps_image_2 = PanSegImage(data, image_props)

    merged = ps_image_1.merge_with(ps_image_2)
    assert merged.dimensionality == ImageDimensionality.TWO
    assert merged.is_multichannel
    assert merged.shape == (2, 10, 11)


def test_merge_images_3d():
    data = np.random.rand(9, 10, 11)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image_1 = PanSegImage(data, image_props)
    ps_image_2 = PanSegImage(data, image_props)

    merged = ps_image_1.merge_with(ps_image_2)
    assert merged.dimensionality == ImageDimensionality.THREE
    assert merged.is_multichannel
    assert merged.shape == (2, 9, 10, 11)


def test_merge_images_3dc():
    data = np.random.rand(2, 9, 10, 11)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.CZYX,
        original_voxel_size=voxel_size,
    )
    ps_image_1 = PanSegImage(data, image_props)
    data = np.random.rand(2, 9, 10, 11)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.CZYX,
        original_voxel_size=voxel_size,
    )
    ps_image_2 = PanSegImage(data, image_props)

    merged = ps_image_1.merge_with(ps_image_2)
    assert merged.dimensionality == ImageDimensionality.THREE
    assert merged.is_multichannel
    assert merged.shape == (4, 9, 10, 11)

    data = np.random.rand(9, 10, 11)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image_3 = PanSegImage(data, image_props)
    merged = merged.merge_with(ps_image_3)
    assert merged.dimensionality == ImageDimensionality.THREE
    assert merged.is_multichannel
    assert merged.shape == (5, 9, 10, 11)


def test_merge_timeseries_matching_t_spacing():
    ps_image_1 = _timeseries_ps_image(
        np.random.rand(7, 5, 16, 16), "TZYX", t_spacing=10.0
    )
    ps_image_2 = _timeseries_ps_image(
        np.random.rand(7, 5, 16, 16), "TZYX", t_spacing=10.0
    )

    merged = ps_image_1.merge_with(ps_image_2)
    assert merged.image_layout == ImageLayout.TCZYX
    assert merged.is_timeseries
    # The channel axis sits at index 1, after T.
    assert merged.shape == (7, 2, 5, 16, 16)
    assert merged.properties.t_spacing == 10.0
    splits = merged.split_channels()
    assert len(splits) == 2
    assert all([s.image_layout == ImageLayout.TZYX for s in splits])
    assert all([s.shape == (7, 5, 16, 16) for s in splits])


def test_merge_timeseries_2d():
    ps_image_1 = _timeseries_ps_image(np.random.rand(7, 16, 16), "TYX", t_spacing=5.0)
    ps_image_2 = _timeseries_ps_image(np.random.rand(7, 16, 16), "TYX", t_spacing=5.0)

    merged = ps_image_1.merge_with(ps_image_2)
    assert merged.image_layout == ImageLayout.TCYX
    assert merged.is_timeseries
    assert merged.shape == (7, 2, 16, 16)
    assert merged.properties.t_spacing == 5.0


def test_merge_timeseries_mismatched_t_spacing():
    ps_image_1 = _timeseries_ps_image(
        np.random.rand(7, 5, 16, 16), "TZYX", t_spacing=10.0
    )
    ps_image_2 = _timeseries_ps_image(
        np.random.rand(7, 5, 16, 16), "TZYX", t_spacing=20.0
    )

    with pytest.raises(ValueError):
        ps_image_1.merge_with(ps_image_2)


def test_merge_timeseries_set_vs_unknown_t_spacing():
    ps_image_1 = _timeseries_ps_image(
        np.random.rand(7, 5, 16, 16), "TZYX", t_spacing=10.0
    )
    ps_image_2 = _timeseries_ps_image(np.random.rand(7, 5, 16, 16), "TZYX")

    with pytest.raises(ValueError):
        ps_image_1.merge_with(ps_image_2)


def test_merge_timeseries_both_unknown_t_spacing():
    ps_image_1 = _timeseries_ps_image(np.random.rand(7, 5, 16, 16), "TZYX")
    ps_image_2 = _timeseries_ps_image(np.random.rand(7, 5, 16, 16), "TZYX")

    merged = ps_image_1.merge_with(ps_image_2)
    assert merged.image_layout == ImageLayout.TCZYX
    assert merged.shape == (7, 2, 5, 16, 16)
    assert merged.properties.t_spacing is None


def test_merge_timeseries_vs_still():
    ps_timeseries = _timeseries_ps_image(np.random.rand(7, 5, 16, 16), "TZYX")
    ps_still = _timeseries_ps_image(np.random.rand(5, 16, 16), "ZYX")

    with pytest.raises(ValueError):
        ps_timeseries.merge_with(ps_still)


def test_merge_images_wrong_semantic():
    data = np.random.rand(9, 10, 11)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image_1 = PanSegImage(data, image_props)
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.PREDICTION,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image_2 = PanSegImage(data, image_props)

    with pytest.raises(ValueError):
        ps_image_1.merge_with(ps_image_2)


def test_merge_images_2d_3d():
    data = np.random.rand(9, 10, 11)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image_1 = PanSegImage(data, image_props)
    data = np.random.rand(10, 11)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.YX,
        original_voxel_size=voxel_size,
    )
    ps_image_2 = PanSegImage(data, image_props)

    with pytest.raises(ValueError):
        ps_image_1.merge_with(ps_image_2)


def test_image_properties_json_roundtrip():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    props = ImageProperties(
        name="image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.TCZYX,
        original_voxel_size=voxel_size,
        t_spacing=0.5,
    )
    json_str = props.model_dump_json()
    loaded = ImageProperties.model_validate_json(json_str)
    assert loaded.image_layout == ImageLayout.TCZYX
    assert loaded.t_spacing == 0.5
    assert loaded.t_unit == "s"


def test_image_properties_json_old_format():
    old_json = (
        '{"name": "image", "semantic_type": "raw",'
        ' "voxel_size": {"voxels_size": [1.0, 1.0, 1.0], "unit": "um"},'
        ' "image_layout": "ZYX",'
        ' "original_voxel_size": {"voxels_size": [1.0, 1.0, 1.0], "unit": "um"},'
        ' "source_file_name": null}'
    )
    loaded = ImageProperties.model_validate_json(old_json)
    assert loaded.image_layout == ImageLayout.ZYX
    assert loaded.t_spacing is None
    assert loaded.t_unit == "s"


def test_napari_layer_roundtrip_timeseries():
    data = np.random.rand(7, 3, 5, 16, 16)
    voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 2.0), unit="um")
    image_props = ImageProperties(
        name="timeseries",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.TCZYX,
        original_voxel_size=voxel_size,
        t_spacing=2.0,
    )
    ps_image = PanSegImage(data, image_props)

    layer_tuple = ps_image.to_napari_layer_tuple()
    layer = Image(layer_tuple[0], **layer_tuple[1])
    loaded = PanSegImage.from_napari_layer(layer)

    assert loaded.image_layout == ImageLayout.TCZYX
    assert loaded.properties.t_spacing == 2.0
    # the t axis stays in timepoint indices: the scale is not t_spacing
    assert loaded.scale == (1.0, 1.0, 0.5, 1.0, 2.0)


def test_stack_sort_noop():
    stack_layout = "CZYX"
    data = np.arange(120).reshape((2, 3, 4, 5))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "CZYX"
    assert n_data.shape == (2, 3, 4, 5)
    assert n_voxel_size.voxels_size == (3, 4, 5)
    assert np.all(n_data == data)


def test_stack_sort_3dc():
    stack_layout = "ZCXY"
    data = np.empty((2, 3, 4, 5))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "CZYX"
    assert n_data.shape == (3, 2, 5, 4)
    assert n_voxel_size.voxels_size == (3, 5, 4)


def test_stack_sort_3d():
    stack_layout = "ZXY"
    data = np.empty((3, 4, 5))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "ZYX"
    assert n_data.shape == (3, 5, 4)
    assert n_voxel_size.voxels_size == (3, 5, 4)


def test_stack_sort_2dc():
    stack_layout = "XYC"
    data = np.empty((3, 4, 5))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "CYX"
    assert n_data.shape == (5, 4, 3)
    assert n_voxel_size.voxels_size == (3, 5, 4)


def test_stack_sort_2dc2():
    stack_layout = "YCX"
    data = np.empty((3, 4, 5))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "CYX"
    assert n_data.shape == (4, 3, 5)
    assert n_voxel_size.voxels_size == (3, 4, 5)


def test_stack_sort_2d():
    stack_layout = "YX"
    data = np.empty((3, 4))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "YX"
    assert n_data.shape == (3, 4)
    assert n_voxel_size.voxels_size == (3, 4, 5)


def test_stack_sort_3dc_invZ():
    stack_layout = "CYX-Z"
    data = np.arange(120).reshape((2, 3, 4, 5))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "CZYX"
    assert n_data.shape == (2, 5, 3, 4)
    assert n_voxel_size.voxels_size == (5, 3, 4)
    assert np.all(n_data[:, ::-1, :, :] == np.transpose(data, axes=[0, 3, 1, 2]))


def test_stack_sort_3dc_invC():
    stack_layout = "-CZYX"
    data = np.arange(120).reshape((2, 3, 4, 5))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "CZYX"
    assert n_data.shape == (2, 3, 4, 5)
    assert n_voxel_size.voxels_size == (3, 4, 5)
    assert np.all(n_data[::-1, :, :, :] == data)


def test_stack_sort_3dc_invCX():
    stack_layout = "-CZX-Y"
    data = np.arange(120).reshape((2, 3, 4, 5))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "CZYX"
    assert n_data.shape == (2, 3, 5, 4)
    assert n_voxel_size.voxels_size == (3, 5, 4)
    assert np.all(n_data[::-1, :, ::-1, :] == np.transpose(data, axes=[0, 1, 3, 2]))


def test_stack_sort_4d_noop():
    stack_layout = "TZYX"
    data = np.arange(120).reshape((2, 3, 4, 5))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "TZYX"
    assert n_data.shape == (2, 3, 4, 5)
    assert n_voxel_size.voxels_size == (3, 4, 5)
    assert np.all(n_data == data)


def test_stack_sort_5dc_noop():
    stack_layout = "TCZYX"
    data = np.arange(720).reshape((2, 3, 4, 5, 6))
    voxel_size = VoxelSize(voxels_size=(4, 5, 6))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "TCZYX"
    assert n_data.shape == (2, 3, 4, 5, 6)
    assert n_voxel_size.voxels_size == (4, 5, 6)
    assert np.all(n_data == data)


def test_stack_sort_4dc_reorder():
    stack_layout = "ZTCYX"
    data = np.arange(720).reshape((3, 2, 4, 5, 6))
    voxel_size = VoxelSize(voxels_size=(3, 5, 6))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "TCZYX"
    assert n_data.shape == (2, 4, 3, 5, 6)
    assert n_voxel_size.voxels_size == (3, 5, 6)
    assert np.all(n_data == np.transpose(data, axes=[1, 2, 0, 3, 4]))


def test_stack_sort_4d_invT():
    stack_layout = "-TZYX"
    data = np.arange(120).reshape((2, 3, 4, 5))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "TZYX"
    assert n_data.shape == (2, 3, 4, 5)
    assert n_voxel_size.voxels_size == (3, 4, 5)
    assert np.all(n_data[::-1, :, :, :] == data)


def test_stack_sort_2dc_invX():
    stack_layout = "CY-X"
    data = np.arange(24).reshape((2, 3, 4))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "CYX"
    assert n_data.shape == (2, 3, 4)
    assert n_voxel_size.voxels_size == (3, 4, 5)
    assert np.all(n_data[:, :, ::-1] == data)


# Shared timeseries fixtures: one raw float32 array per T layout on the
# documented shape skeleton, plus a uint16 segmentation whose label IDs are
# independent across timepoints by construction.
TIMESERIES_RAW_FIXTURES = [
    pytest.param("timeseries_tyx", ImageLayout.TYX, (4, 16, 16), id="TYX"),
    pytest.param("timeseries_tcyx", ImageLayout.TCYX, (4, 2, 16, 16), id="TCYX"),
    pytest.param("timeseries_tzyx", ImageLayout.TZYX, (4, 5, 16, 16), id="TZYX"),
    pytest.param("timeseries_tczyx", ImageLayout.TCZYX, (4, 2, 5, 16, 16), id="TCZYX"),
]


@pytest.mark.parametrize("fixture_name, layout, shape", TIMESERIES_RAW_FIXTURES)
def test_timeseries_raw_fixture(request, fixture_name, layout, shape):
    data = request.getfixturevalue(fixture_name)
    assert data.shape == shape
    assert data.dtype == np.float32
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    for t_props, expected_t_spacing in (
        (TIMESERIES_PROPS_KNOWN_T_SPACING, 10.0),
        (TIMESERIES_PROPS_UNKNOWN_T_SPACING, None),
    ):
        props = ImageProperties(
            name=fixture_name,
            semantic_type=SemanticType.RAW,
            voxel_size=voxel_size,
            image_layout=layout,
            original_voxel_size=voxel_size,
            **t_props,
        )
        image = PanSegImage(data, props)
        assert image.image_layout == layout
        assert image.is_timeseries
        assert image.shape == shape
        assert image.properties.t_spacing == expected_t_spacing


def test_timeseries_segmentation_fixture(timeseries_segmentation):
    seg = timeseries_segmentation
    assert seg.shape == (4, 5, 16, 16)
    assert seg.dtype == np.uint16
    # Label IDs are independent across timepoints: no ID shared by two t.
    label_sets = [set(np.unique(seg[t]).tolist()) - {0} for t in range(seg.shape[0])]
    assert all(label_sets)
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            assert not label_sets[i] & label_sets[j]
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    for t_props, expected_t_spacing in (
        (TIMESERIES_PROPS_KNOWN_T_SPACING, 10.0),
        (TIMESERIES_PROPS_UNKNOWN_T_SPACING, None),
    ):
        props = ImageProperties(
            name="timeseries_segmentation",
            semantic_type=SemanticType.SEGMENTATION,
            voxel_size=voxel_size,
            image_layout=ImageLayout.TZYX,
            original_voxel_size=voxel_size,
            **t_props,
        )
        image = PanSegImage(seg, props)
        assert image.image_layout == ImageLayout.TZYX
        assert image.image_type == ImageType.LABEL
        assert image.properties.t_spacing == expected_t_spacing


# --- Time-aware export: a time-bearing image roundtrips through
# save_image/import_image with layout and t_spacing preserved; ---


@pytest.mark.parametrize(
    "layout, t_spacing",
    [
        ("TYX", 10.0),
        ("TYX", None),
        ("TZYX", 10.0),
        ("TZYX", None),
    ],
)
@pytest.mark.parametrize("export_format", ["tiff", "h5", "zarr"])
def test_save_image_timeseries_roundtrip(
    tmp_path, timeseries_segmentation, layout, t_spacing, export_format
):
    if layout == "TYX":
        data = timeseries_segmentation[:, 0]
    else:
        data = timeseries_segmentation
    image = _timeseries_ps_image(
        data, layout, t_spacing=t_spacing, semantic_type=SemanticType.SEGMENTATION
    )
    save_image(
        image,
        tmp_path,
        "seg",
        key="segmentation",
        export_format=export_format,
        data_type="uint16",
    )

    if export_format == "tiff":
        path, key = tmp_path / "seg.ome.tiff", None
    elif export_format == "h5":
        path, key = tmp_path / "seg.h5", "segmentation"
    else:
        path, key = tmp_path / "seg.zarr", "segmentation"

    # the format metadata prefills the original layout
    assert guess_stack_layout(path, key) == layout

    imported = import_image(
        path=path,
        key=key,
        image_name="reimported",
        semantic_type="segmentation",
        stack_layout=layout,
    )
    assert isinstance(imported, PanSegImage)
    assert imported.image_layout == ImageLayout(layout)
    assert imported.properties.t_spacing == t_spacing
    assert imported.properties.t_unit == "s"
    assert np.array_equal(imported.get_data(), data)

    if export_format == "h5":
        # save_image delegates to create_h5 (attr writing covered in
        # tests/io/test_h5.py); here the layout and time attrs survive on
        # the core-layer export path
        assert read_h5_axis_order(path, key="segmentation") == layout
        assert read_h5_time_spacing(path, key="segmentation") == (t_spacing, "s")


@pytest.mark.parametrize("export_format", ["jpg", "png"])
def test_save_image_timeseries_rejects_pil_formats(
    tmp_path, timeseries_segmentation, export_format
):
    image = _timeseries_ps_image(
        timeseries_segmentation,
        "TZYX",
        t_spacing=10.0,
        semantic_type=SemanticType.SEGMENTATION,
    )
    with pytest.raises(
        ValueError, match=f"Export format {export_format} not recognized"
    ):
        save_image(
            image,
            tmp_path,
            "seg",
            export_format=export_format,
            data_type="uint16",
        )
    assert list(tmp_path.iterdir()) == []
