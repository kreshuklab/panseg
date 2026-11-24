import numpy as np
import pytest

from panseg.core.image import (
    ImageLayout,
    ImageProperties,
    PanSegImage,
    SemanticType,
)
from panseg.io.h5 import load_h5, read_h5_voxel_size
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


# Tolerance on the per-voxel difference between predicted z slices of a raw
# volume that is one slice repeated along z: the slices are identical going in,
# so a 3D model must predict identical slices going out (float32 noise only;
# the pre-fix deviation was 2e-4).
Z_SLICE_ATOL = 1e-7


@pytest.mark.slow
def test_unet_prediction_task_z_slice_invariance(h5_file):
    """A raw volume of one z slice repeated must predict identical z slices.

    `generic_confocal_3D_unet` is a 3D model: in a direction along which the
    raw data does not change, its prediction must not change either. The first
    z slice of the sample ovule is repeated along z, so every predicted z slice
    is expected to match. Deviations mean that patch/halo border handling leaks
    into the prediction (the first and last slices get the least context).
    """
    n_z = 64
    raw = load_h5(h5_file, key="raw")
    voxel_size = read_h5_voxel_size(h5_file, key="raw")
    data = np.repeat(raw[:1].astype("float32"), n_z, axis=0)

    image = PanSegImage(
        data=data,
        properties=ImageProperties(
            name="z_stack",
            voxel_size=voxel_size,
            semantic_type=SemanticType.RAW,
            image_layout=ImageLayout.ZYX,
            original_voxel_size=voxel_size,
        ),
    )

    result = unet_prediction_task(
        image=image,
        model_name="generic_confocal_3D_unet",
        model_id=None,
        device="cpu",
        disable_tqdm=True,
    )

    assert len(result) == 1
    pmap = result[0].get_data()
    assert pmap.shape == data.shape

    reference_z = n_z // 2
    deviation = np.abs(pmap - pmap[reference_z]).reshape(n_z, -1).max(axis=1)
    deviating = np.flatnonzero(deviation > Z_SLICE_ATOL)
    assert deviating.size == 0, (
        f"{deviating.size} of {n_z} predicted z slices differ from slice "
        f"{reference_z} by more than {Z_SLICE_ATOL}.\n"
        f"Deviating z slices: {deviating.tolist()}\n"
        f"Max abs deviation per z slice: {np.round(deviation, 8).tolist()}"
    )


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
@pytest.mark.parametrize("shape", [(8, 32, 32), (16, 64, 64)], ids=["one", "blocks"])
def test_biio_prediction_task_mock_model_fixed_size(
    mock_biio_package, by_id, shape, mocker
):
    """Task end-to-end on a local model package with fixed (8, 32, 32) input axes.

    Covers model-id resolution (a local zip, or a bioimage.io ID resolved to
    it), real bioimageio.core inference, and the fixed-size branch of
    biio_prediction, for an image of the model's size and a larger one that
    is predicted in blocks.
    """
    model_id = str(mock_biio_package)
    if by_id:
        resolve = mocker.patch(
            "panseg.core.zoo.bioimageio_rdf_url", return_value=model_id
        )
        model_id = "mock-unet3d"

    result = biio_prediction_task(
        image=_mock_raw_image(np.zeros(shape, dtype="float32")),
        model_id=model_id,
        suffix="_biio_prediction",
        device="cpu",
    )
    _assert_biio_prediction_result(result, shape)
    if by_id:
        resolve.assert_called_once_with("mock-unet3d")


@pytest.mark.parametrize(
    "shape",
    [(96, 96, 96), (8, 16, 32), (120, 40, 70)],
    ids=["one_pass", "below_minimum", "blocks"],
)
def test_biio_prediction_task_mock_model_parameterized(
    mock_biio_package_parameterized, shape
):
    """Task end-to-end on a PanSeg-packaged model with parameterized axes.

    Covers the parameterized branch of biio_prediction, for an image that fits
    in one block, one below the model's minimum input, and one tiled into
    blocks, as well as the file-based architecture loading of PanSeg's own
    model packaging.
    """
    result = biio_prediction_task(
        image=_mock_raw_image(np.zeros(shape, dtype="float32")),
        model_id=str(mock_biio_package_parameterized),
        suffix="_biio_prediction",
        device="cpu",
    )
    _assert_biio_prediction_result(result, shape)
