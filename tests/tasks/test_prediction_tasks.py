from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
import torch
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromLibraryDescr,
    AxisId,
    BatchAxis,
    ChannelAxis,
    Identifier,
    InputTensorDescr,
    IntervalOrRatioDataDescr,
    ModelDescr,
    OutputTensorDescr,
    PytorchStateDictWeightsDescr,
    SizeReference,
    SpaceInputAxis,
    SpaceOutputAxis,
    TensorId,
    Version,
    WeightsDescr,
)

from panseg.core.image import (
    ImageLayout,
    ImageProperties,
    PanSegImage,
    SemanticType,
)
from panseg.functionals.training.biio import make_model_description
from panseg.functionals.training.model import UNet3D
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


# Small UNet3D used to build local BioImage.IO model packages at test time,
# so the biio_prediction task can be tested end-to-end without any network access.
MOCK_UNET_KWARGS = {
    "in_channels": 1,
    "out_channels": 1,
    "final_sigmoid": True,
    "f_maps": [8, 16],
    "layer_order": "bcr",
}


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


def test_biio_prediction_task_mock_model_fixed_size(tmp_path):
    """Task end-to-end on a local model package with fixed-size input axes.

    Covers model-id resolution from a local zip, real bioimageio.core
    inference, and the fixed input_block_shape branch of biio_prediction.
    """
    torch.manual_seed(0)
    model = UNet3D(**MOCK_UNET_KWARGS)
    torch.save(model.state_dict(), tmp_path / "weights.pytorch")

    def space_input(axis_id: str, size: int) -> SpaceInputAxis:
        return SpaceInputAxis(
            id=AxisId(axis_id), size=size, scale=1.0, unit="micrometer"
        )

    def space_output(axis_id: str) -> SpaceOutputAxis:
        return SpaceOutputAxis(
            id=AxisId(axis_id),
            size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId(axis_id)),
            scale=1.0,
            unit="micrometer",
        )

    with chdir(tmp_path):
        desc = ModelDescr(
            name="mock-unet3d-fixed-size",
            description="Tiny UNet3D built at test time; no network access.",
            inputs=[
                InputTensorDescr(
                    id=TensorId("input"),
                    axes=[
                        BatchAxis(),
                        ChannelAxis(channel_names=[Identifier("ch0")]),
                        space_input("z", 8),
                        space_input("y", 32),
                        space_input("x", 32),
                    ],
                    data=IntervalOrRatioDataDescr(type="float32"),
                )
            ],
            outputs=[
                OutputTensorDescr(
                    id=TensorId("output"),
                    axes=[
                        BatchAxis(),
                        ChannelAxis(channel_names=[Identifier("out_ch0")]),
                        space_output("z"),
                        space_output("y"),
                        space_output("x"),
                    ],
                    data=IntervalOrRatioDataDescr(type="float32"),
                )
            ],
            weights=WeightsDescr(
                pytorch_state_dict=PytorchStateDictWeightsDescr(
                    source=Path("weights.pytorch"),
                    architecture=ArchitectureFromLibraryDescr(
                        import_from="panseg.functionals.training.model",
                        callable="UNet3D",
                        kwargs=MOCK_UNET_KWARGS,
                    ),
                    pytorch_version=Version(torch.__version__),
                )
            ),
        )
        package = tmp_path / "mock_model.zip"
        desc.package(package)

    result = biio_prediction_task(
        image=_mock_raw_image(np.zeros((8, 32, 32), dtype="float32")),
        model_id=str(package),
        suffix="_biio_prediction",
        device="cpu",
    )
    _assert_biio_prediction_result(result, (8, 32, 32))


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
            axis_min_sizes=(16, 32, 64),
            dimensionality="3D",
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
