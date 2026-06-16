import numpy as np
import pytest

from panseg.core.image import (
    ImageLayout,
    ImageProperties,
    PanSegImage,
    SemanticType,
)
from panseg.io.voxelsize import VoxelSize
from panseg.tasks.prediction_tasks import biio_prediction_task, unet_prediction_task


@pytest.mark.parametrize(
    "shape, layout, model_name",
    [
        ((8, 64, 64), ImageLayout.ZYX, "generic_confocal_3D_unet"),
        ((64, 64), ImageLayout.YX, "confocal_2D_unet_ovules_ds2x"),
    ],
)
def test_unet_prediction_task(shape, layout, model_name):
    mock_data = np.random.rand(*shape).astype("float32")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property)

    result = unet_prediction_task(
        image=image,
        model_name=model_name,
        model_id=None,
        device="cpu",
    )

    assert len(result) == 1
    result = result[0]

    assert result.semantic_type == SemanticType.PREDICTION
    assert result.image_layout == property.image_layout
    assert result.voxel_size == property.voxel_size
    assert result.shape == mock_data.shape


@pytest.mark.parametrize(
    "raw_fixture_name, input_layout, model_id",
    (
        ("raw_zcyx_96x2x96x96", "ZCYX", "philosophical-panda"),
        ("raw_cell_3d_100x128x128", "ZYX", "emotional-cricket"),
        ("raw_cell_2d_96x96", "YX", "pioneering-rhino"),
    ),
)
def test_biio_prediction_task(raw_fixture_name, input_layout, model_id, request):
    image = PanSegImage(
        data=request.getfixturevalue(raw_fixture_name),
        properties=ImageProperties(
            name="test",
            voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
            semantic_type=SemanticType.RAW,
            image_layout=input_layout,
            original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        ),
    )
    result = biio_prediction_task(
        image=image, model_id=model_id, suffix="_biio_prediction", device="cpu"
    )
    for new_image in result:
        assert new_image.semantic_type == SemanticType.PREDICTION
        assert "_biio_prediction" in new_image.name
