import numpy as np
import pytest

from panseg.core.image import (
    ImageLayout,
    ImageProperties,
    PanSegImage,
    SemanticType,
)
from panseg.io.h5 import load_h5
from panseg.io.voxelsize import VoxelSize
from panseg.tasks import segmentation_tasks
from panseg.tasks.segmentation_tasks import (
    aio_watershed_task,
    clustering_segmentation_task,
    dt_watershed_task,
    lmc_segmentation_task,
)


@pytest.mark.parametrize(
    "shape, layout, stacked, is_nuclei, clustering, mode",
    [
        ((32, 64, 64), ImageLayout.ZYX, False, False, False, "-"),
        ((32, 64, 64), ImageLayout.ZYX, False, True, False, "-"),
        ((32, 64, 64), ImageLayout.ZYX, False, False, True, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, True, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, True, "multicut"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, True, "mutex_ws"),
        ((64, 64), ImageLayout.YX, False, False, False, "-"),
        ((64, 64), ImageLayout.YX, False, True, False, "-"),
        ((64, 64), ImageLayout.YX, False, False, True, "gasp"),
        ((64, 64), ImageLayout.YX, True, False, True, "gasp"),
        ((64, 64), ImageLayout.YX, False, False, True, "multicut"),
        ((64, 64), ImageLayout.YX, False, False, True, "mutex_ws"),
    ],
)
def test_dt_watershed_and_clustering(
    shape, layout, stacked, is_nuclei, clustering, mode
):
    mock_data = np.random.rand(*shape).astype("float32")

    property_ = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.PREDICTION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property_)

    result = dt_watershed_task(image=image, stacked=stacked, is_nuclei_image=is_nuclei)

    assert result.semantic_type == SemanticType.SEGMENTATION
    assert result.image_layout == property_.image_layout
    assert result.voxel_size == property_.voxel_size
    assert result.shape == mock_data.shape

    if not clustering:
        return

    result_clustering = clustering_segmentation_task(
        image=image, over_segmentation=result, mode=mode
    )
    assert result_clustering.semantic_type == SemanticType.SEGMENTATION
    assert result_clustering.image_layout == property_.image_layout
    assert result_clustering.voxel_size == property_.voxel_size
    assert result_clustering.shape == mock_data.shape


def test_mutex():
    mock_data = np.random.rand(32, 64, 64).astype("float32")

    property_ = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.PREDICTION,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property_)

    result_clustering = clustering_segmentation_task(
        image=image, over_segmentation=None, mode="mutex_ws"
    )
    assert result_clustering.semantic_type == SemanticType.SEGMENTATION
    assert result_clustering.image_layout == property_.image_layout
    assert result_clustering.voxel_size == property_.voxel_size
    assert result_clustering.shape == mock_data.shape


def test_lmc_segmentation_pred(mocker):
    lmc_seg = mocker.spy(
        segmentation_tasks,
        "lifted_multicut_from_nuclei_segmentation",
    )
    lmc_pmaps = mocker.spy(
        segmentation_tasks,
        "lifted_multicut_from_nuclei_pmaps",
    )
    mock_data = np.random.rand(32, 64, 64).astype("float32")
    mock_seg = (np.random.rand(32, 64, 64) * 32).astype("int8")
    layout = ImageLayout.ZYX

    pred_property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.PREDICTION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )

    raw_property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    seg_property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.SEGMENTATION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )

    pred_image = PanSegImage(data=mock_data, properties=pred_property)
    PanSegImage(data=mock_data, properties=raw_property)
    seg_image = PanSegImage(data=mock_seg, properties=seg_property)

    result = lmc_segmentation_task(
        boundary_pmap=pred_image,
        superpixels=seg_image,
        nuclei=pred_image,
    )

    lmc_pmaps.assert_called_once()
    lmc_seg.assert_not_called()
    assert result.semantic_type == SemanticType.SEGMENTATION
    assert result.image_layout == raw_property.image_layout
    assert result.voxel_size == raw_property.voxel_size
    assert result.shape == mock_data.shape


def test_lmc_segmentation_seg(mocker, napari_prediction, napari_segmentation, h5_file):
    lmc_seg = mocker.spy(
        segmentation_tasks,
        "lifted_multicut_from_nuclei_segmentation",
    )
    lmc_pmaps = mocker.spy(
        segmentation_tasks,
        "lifted_multicut_from_nuclei_pmaps",
    )

    raw_data = load_h5(h5_file, "raw")
    pred_image = PanSegImage.from_napari_layer(napari_prediction)
    seg_image = PanSegImage.from_napari_layer(napari_segmentation)
    pred2_image = PanSegImage.derive_new(pred_image, raw_data, name="pred2")
    seg2_image = PanSegImage.derive_new(seg_image, raw_data, name="seg2")

    result = lmc_segmentation_task(
        boundary_pmap=pred2_image,
        superpixels=seg2_image,
        nuclei=seg2_image,
    )

    lmc_pmaps.assert_not_called()
    lmc_seg.assert_called_once()
    assert result.semantic_type == SemanticType.SEGMENTATION
    assert result.image_layout == pred2_image.image_layout
    assert result.voxel_size == pred2_image.voxel_size
    assert result.shape == pred2_image.shape


@pytest.mark.parametrize(
    "shape, layout, stacked, is_nuclei, mode",
    [
        ((32, 64, 64), ImageLayout.ZYX, False, False, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, False, True, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, "multicut"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, "lmc"),
        ((64, 64), ImageLayout.YX, False, False, "gasp"),
        ((64, 64), ImageLayout.YX, False, True, "gasp"),
        ((64, 64), ImageLayout.YX, True, False, "gasp"),
        ((64, 64), ImageLayout.YX, False, False, "multicut"),
        ((64, 64), ImageLayout.YX, False, False, "mutex_ws"),
        ((64, 64), ImageLayout.YX, False, False, "lmc"),
    ],
)
def test_aio_watershed_and_clustering(shape, layout, stacked, is_nuclei, mode):
    mock_data = np.random.rand(*shape).astype("float32")

    raw_property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.PREDICTION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=raw_property)
    if mode == "lmc":
        nuc_property = ImageProperties(
            name="test",
            voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
            semantic_type=SemanticType.RAW,
            image_layout=layout,
            original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        )
        nuclei = PanSegImage(data=mock_data, properties=nuc_property)
    else:
        nuclei = None

    segmentation = aio_watershed_task(
        image=image,
        nuclei=nuclei,
        stacked=stacked,
        is_nuclei_image=is_nuclei,
        mode=mode,
    )

    assert segmentation.semantic_type == SemanticType.SEGMENTATION
    assert segmentation.image_layout == raw_property.image_layout
    assert segmentation.voxel_size == raw_property.voxel_size
    assert segmentation.shape == mock_data.shape
