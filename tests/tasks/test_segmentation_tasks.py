import logging

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
    "shape, layout, stacked, blockwise, is_nuclei, clustering, mode",
    [
        ((32, 64, 64), ImageLayout.ZYX, False, False, False, False, "-"),
        ((32, 64, 64), ImageLayout.ZYX, False, False, True, False, "-"),
        ((32, 64, 64), ImageLayout.ZYX, False, False, False, True, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, False, True, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, False, True, "multicut"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, False, True, "mutex_ws"),
        ((64, 64), ImageLayout.YX, False, False, False, False, "-"),
        ((64, 64), ImageLayout.YX, False, False, True, False, "-"),
        ((64, 64), ImageLayout.YX, False, False, False, True, "gasp"),
        ((64, 64), ImageLayout.YX, True, False, False, True, "gasp"),
        ((64, 64), ImageLayout.YX, False, False, False, True, "multicut"),
        ((64, 64), ImageLayout.YX, False, False, False, True, "mutex_ws"),
        # blockwise=True on small 3D volumes falls back to the single-pass
        # watershed in the functional; these rows verify the task plumbing
        ((32, 64, 64), ImageLayout.ZYX, False, True, False, False, "-"),
        ((32, 64, 64), ImageLayout.ZYX, False, True, True, False, "-"),
        ((32, 64, 64), ImageLayout.ZYX, False, True, False, True, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, False, True, True, True, "gasp"),
    ],
)
def test_dt_watershed_and_clustering(
    shape, layout, stacked, blockwise, is_nuclei, clustering, mode
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

    result = dt_watershed_task(
        image=image,
        stacked=stacked,
        blockwise=blockwise,
        is_nuclei_image=is_nuclei,
    )

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


def test_dt_watershed_task_blockwise_large_volume(caplog):
    # large enough volume (>= 2M voxels) so the functional takes the real
    # blockwise path instead of falling back to the single-pass watershed
    mock_data = np.random.rand(128, 128, 128).astype("float32")

    property_ = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.PREDICTION,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property_)

    with caplog.at_level(
        logging.WARNING, logger="panseg.functionals.segmentation.segmentation"
    ):
        result = dt_watershed_task(image=image, blockwise=True)

    fallback_warnings = [
        record
        for record in caplog.records
        if "blockwise mode not applicable" in record.getMessage()
    ]
    assert not fallback_warnings, [r.getMessage() for r in fallback_warnings]

    segmentation = result.get_data()
    assert result.semantic_type == SemanticType.SEGMENTATION
    assert result.shape == mock_data.shape
    assert segmentation.dtype == np.uint64
    assert segmentation.max() > segmentation.min() >= 0
    assert len(np.unique(segmentation)) > 1


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
    # raw intensities are outside [0, 1]; rescale them so they form a valid
    # boundary probability map (the new elf backend rejects non-finite costs)
    pmaps = (raw_data.astype("float32") - raw_data.min()) / (
        raw_data.max() - raw_data.min()
    )
    pred_image = PanSegImage.from_napari_layer(napari_prediction)
    seg_image = PanSegImage.from_napari_layer(napari_segmentation)
    pred2_image = PanSegImage.derive_new(pred_image, pmaps, name="pred2")
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
    "shape, layout, stacked, blockwise, is_nuclei, mode",
    [
        ((32, 64, 64), ImageLayout.ZYX, False, False, False, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, False, False, True, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, False, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, False, "multicut"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, False, "lmc"),
        ((64, 64), ImageLayout.YX, False, False, False, "gasp"),
        ((64, 64), ImageLayout.YX, False, False, True, "gasp"),
        ((64, 64), ImageLayout.YX, True, False, False, "gasp"),
        ((64, 64), ImageLayout.YX, False, False, False, "multicut"),
        ((64, 64), ImageLayout.YX, False, False, False, "mutex_ws"),
        ((64, 64), ImageLayout.YX, False, False, False, "lmc"),
        # blockwise=True on small 3D volumes falls back to the single-pass
        # watershed in the functional; these rows verify the task plumbing
        ((32, 64, 64), ImageLayout.ZYX, False, True, False, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, False, True, False, "multicut"),
        ((32, 64, 64), ImageLayout.ZYX, False, True, False, "mutex_ws"),
        ((32, 64, 64), ImageLayout.ZYX, False, True, False, "lmc"),
    ],
)
def test_aio_watershed_and_clustering(
    shape, layout, stacked, blockwise, is_nuclei, mode
):
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
        blockwise=blockwise,
        is_nuclei_image=is_nuclei,
        mode=mode,
    )

    assert segmentation.semantic_type == SemanticType.SEGMENTATION
    assert segmentation.image_layout == raw_property.image_layout
    assert segmentation.voxel_size == raw_property.voxel_size
    assert segmentation.shape == mock_data.shape
