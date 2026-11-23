from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
import torch

from panseg.core.image import (
    ImageLayout,
    ImageProperties,
    PanSegImage,
    SemanticType,
)
from panseg.functionals.model_geometry import ModelGeometry
from panseg.functionals.training.biio import make_model_description
from panseg.functionals.training.model import UNet3D
from panseg.io.voxelsize import VoxelSize
from panseg.tasks.prediction_tasks import biio_prediction_task, unet_prediction_task
from tests.conftest import MOCK_UNET_KWARGS


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


@pytest.mark.slow
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


def _mock_raw_image(data: np.ndarray) -> PanSegImage:
    return PanSegImage(
        data=data,
        properties=ImageProperties(
            name="test",
            voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
            semantic_type=SemanticType.RAW,
            image_layout=ImageLayout.ZYX,
            original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        ),
    )


def _assert_biio_prediction_result(result, shape: tuple[int, ...]) -> None:
    assert len(result) == 1
    new_image = result[0]
    assert new_image.semantic_type == SemanticType.PREDICTION
    assert new_image.name == "test__biio_prediction_output"
    # single-channel CZYX output is cast to ZYX by PanSegImage
    assert new_image.image_layout == ImageLayout.ZYX
    assert new_image.shape == shape
    data = new_image.get_data()
    assert data.dtype == np.float32
    assert np.all(np.isfinite(data))
    assert np.all((data >= 0.0) & (data <= 1.0))


@pytest.mark.parametrize("by_id", [False, True], ids=["path", "bioimageio_id"])
def test_biio_prediction_task_mock_model_fixed_size(mock_biio_package, by_id, mocker):
    """Task end-to-end on a local model package with fixed-size input axes.

    Covers model-id resolution (a local zip, or a bioimage.io ID resolved to
    it), real bioimageio.core inference, and the fixed input_block_shape
    branch of biio_prediction.
    """
    model_id = str(mock_biio_package)
    if by_id:
        resolve = mocker.patch(
            "panseg.core.zoo.bioimageio_rdf_url", return_value=model_id
        )
        model_id = "mock-unet3d"

    result = biio_prediction_task(
        image=_mock_raw_image(np.zeros((8, 32, 32), dtype="float32")),
        model_id=model_id,
        suffix="_biio_prediction",
        device="cpu",
    )
    _assert_biio_prediction_result(result, (8, 32, 32))
    if by_id:
        resolve.assert_called_once_with("mock-unet3d")


def test_biio_prediction_task_mock_model_parameterized(tmp_path):
    """Task end-to-end on a PanSeg-packaged model with parameterized axes.

    Covers the blocksize_parameter tiling branch of biio_prediction and the
    file-based architecture loading of PanSeg's own model packaging.
    """
    torch.manual_seed(0)
    model = UNet3D(**MOCK_UNET_KWARGS)
    torch.save(model.state_dict(), tmp_path / "weights.pytorch")
    np.save(
        tmp_path / "test_in.npy", np.random.rand(1, 1, 16, 32, 64).astype("float32")
    )
    np.save(
        tmp_path / "test_out.npy", np.random.rand(1, 1, 16, 32, 64).astype("float32")
    )

    with chdir(tmp_path):
        desc = make_model_description(
            weights=Path("weights.pytorch"),
            model_name="mock-unet3d-parameterized",
            in_channels=1,
            out_channels=1,
            feature_maps=[8, 16],
            final_sigmoid=MOCK_UNET_KWARGS["final_sigmoid"],
            geometry=ModelGeometry(
                min_input=(16, 32, 64), halo=(4, 8, 12), is_2d=False
            ),
            layer_order="bcr",
            modality="confocal",
            output_type="boundaries",
            description="Tiny UNet3D built at test time; no network access.",
            resolution=(1.0, 1.0, 1.0),
            test_in=tmp_path / "test_in.npy",
            test_out=tmp_path / "test_out.npy",
            panseg_config=Path("weights.pytorch"),
        )
        package = tmp_path / "mock_model.zip"
        desc.package(package)

    result = biio_prediction_task(
        image=_mock_raw_image(np.zeros((96, 96, 96), dtype="float32")),
        model_id=str(package),
        suffix="_biio_prediction",
        device="cpu",
    )
    _assert_biio_prediction_result(result, (96, 96, 96))
