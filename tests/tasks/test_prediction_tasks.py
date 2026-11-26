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
from tests.conftest import wall_contrast

# Half the smallest contrast measured with these models on their data
MIN_WALL_CONTRAST = 0.15
MIN_NUCLEI_LOGIT_CONTRAST = 3.0


def _raw_image(data: np.ndarray, layout, voxel_size) -> PanSegImage:
    size = VoxelSize(voxels_size=voxel_size, unit="um")
    return PanSegImage(
        data=data,
        properties=ImageProperties(
            name="test",
            voxel_size=size,
            semantic_type=SemanticType.RAW,
            image_layout=layout,
            original_voxel_size=size,
        ),
    )


@pytest.mark.parametrize(
    "model_name, layout, crop",
    [
        ("generic_confocal_3D_unet", ImageLayout.ZYX, np.s_[8:24]),
        ("confocal_2D_unet_ovules_ds2x", ImageLayout.YX, np.s_[16]),
    ],
)
def test_unet_prediction_task(model_name, layout, crop, ovule_cells):
    """PanSeg's confocal ovule models find the cell walls of an ovule image."""
    raw = ovule_cells.raw[crop].astype("float32")
    image = _raw_image(raw, layout, ovule_cells.voxel_size)

    result = unet_prediction_task(
        image=image,
        model_name=model_name,
        model_id=None,
        device="cpu",
    )

    assert len(result) == 1
    result = result[0]

    assert result.semantic_type == SemanticType.PREDICTION
    assert result.image_layout == image.image_layout
    assert result.voxel_size == image.voxel_size
    assert result.shape == raw.shape
    assert wall_contrast(result.get_data(), ovule_cells.label[crop]) > MIN_WALL_CONTRAST


def _cellpose_input(nuclei: np.ndarray) -> np.ndarray:
    """Cellpose takes nuclei scaled to their 1st-99th percentile in the first
    channel and nothing in the second; ZCYX."""
    low, high = np.percentile(nuclei, (1, 99))
    nuclei = ((nuclei - low) / (high - low)).astype("float32")
    return np.stack([nuclei, np.zeros_like(nuclei)], axis=1)


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
    "model_id, data_name, layout, crop",
    [
        ("pioneering-rhino", "ovule_cells", "YX", np.s_[16]),
        # the model's fixed input size
        ("emotional-cricket", "shoot_meristem_cells", "ZYX", np.s_[:]),
        ("philosophical-panda", "ovule_nuclei", "ZCYX", np.s_[34:66, 2:98, 2:98]),
    ],
)
def test_biio_prediction_task(model_id, data_name, layout, crop, request):
    """Real BioImage.IO models, run through PanSeg on the data they were made
    for, find what they were trained to find."""
    data = request.getfixturevalue(data_name)
    raw, label = data.raw[crop].astype("float32"), data.label[crop]
    if model_id == "philosophical-panda":
        raw = _cellpose_input(raw)

    result = biio_prediction_task(
        image=_raw_image(raw, layout, data.voxel_size),
        model_id=model_id,
        suffix="_biio_prediction",
        device="cpu",
    )

    assert result
    for new_image in result:
        assert new_image.semantic_type == SemanticType.PREDICTION
        assert "_biio_prediction" in new_image.name
        assert np.isfinite(new_image.get_data()).all()
    if model_id == "philosophical-panda":
        # Cellpose's flows are dY, dX and the cell probability, as logits
        flow = next(i for i in result if i.name.endswith("_flow")).get_data()
        nuclei, background = flow[2][label > 0], flow[2][label == 0]
        assert nuclei.mean() - background.mean() > MIN_NUCLEI_LOGIT_CONTRAST
    else:
        (prediction,) = result
        prediction = np.squeeze(prediction.get_data())
        assert prediction.shape == label.shape
        assert wall_contrast(prediction, label) > MIN_WALL_CONTRAST


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
@pytest.mark.parametrize(
    "crop",
    [np.s_[12:20, 48:80, 48:80], np.s_[8:24, 32:96, 32:96]],
    ids=["one", "blocks"],
)
def test_biio_prediction_task_mock_model_fixed_size(
    mock_biio_package, by_id, crop, ovule_cells, mocker
):
    """Task end-to-end on a local model package with fixed (8, 32, 32) input axes.

    Covers model-id resolution (a local zip, or a bioimage.io ID resolved to
    it), real bioimageio.core inference, and the fixed-size branch of
    biio_prediction, for an ovule crop of the model's size and a larger one that
    is predicted in blocks.
    """
    model_id = str(mock_biio_package)
    if by_id:
        resolve = mocker.patch(
            "panseg.core.zoo.bioimageio_rdf_url", return_value=model_id
        )
        model_id = "mock-unet3d"
    data = ovule_cells.raw[crop].astype("float32")

    result = biio_prediction_task(
        image=_mock_raw_image(data),
        model_id=model_id,
        suffix="_biio_prediction",
        device="cpu",
    )
    _assert_biio_prediction_result(result, data.shape)
    if by_id:
        resolve.assert_called_once_with("mock-unet3d")


@pytest.mark.parametrize(
    "crop",
    [np.s_[:, 16:112, 16:112], np.s_[12:20, 56:72, 48:80], np.s_[:]],
    ids=["one_pass", "below_minimum", "blocks"],
)
def test_biio_prediction_task_mock_model_parameterized(
    mock_biio_package_parameterized, crop, ovule_cells
):
    """Task end-to-end on a PanSeg-packaged model with parameterized axes.

    Covers the parameterized branch of biio_prediction, for an ovule crop that
    fits in one block, one below the model's minimum input, and the whole
    (32, 128, 128) ovule image, which is tiled into blocks; as well as the
    file-based architecture loading of PanSeg's own model packaging.
    """
    data = ovule_cells.raw[crop].astype("float32")

    result = biio_prediction_task(
        image=_mock_raw_image(data),
        model_id=str(mock_biio_package_parameterized),
        suffix="_biio_prediction",
        device="cpu",
    )
    _assert_biio_prediction_result(result, data.shape)
