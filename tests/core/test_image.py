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
    import_image,
    stack_sort,
)
from panseg.io import voxelsize
from panseg.io.voxelsize import VoxelSize


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
    assert ImageLayout.to_choices() == ["YX", "CYX", "ZYX", "CZYX", "ZCYX"]


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


def test_stack_sort_2dc_invX():
    stack_layout = "CY-X"
    data = np.arange(24).reshape((2, 3, 4))
    voxel_size = VoxelSize(voxels_size=(3, 4, 5))

    n_stack_layout, n_data, n_voxel_size = stack_sort(stack_layout, data, voxel_size)
    assert n_stack_layout == "CYX"
    assert n_data.shape == (2, 3, 4)
    assert n_voxel_size.voxels_size == (3, 4, 5)
    assert np.all(n_data[:, :, ::-1] == data)
