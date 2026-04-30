import numpy as np
import pytest

from panseg.core.image import (
    ImageLayout,
    ImageProperties,
    PanSegImage,
    SemanticType,
)
from panseg.io.voxelsize import VoxelSize
from panseg.tasks.io_tasks import (
    export_image_task,
    import_image_task,
    merge_channels_task,
)
from panseg.tasks.workflow_handler import Task_message


@pytest.mark.parametrize(
    "shape,layout,export_format",
    [
        ((2, 64, 64), ImageLayout.CYX, "tiff"),
        ((2, 64, 64), ImageLayout.CYX, "h5"),
        # ((32, 64, 64), ImageLayout.ZYX, "zarr"),
        # ((64, 64), ImageLayout.YX, "tiff"),
        # ((64, 64), ImageLayout.YX, "h5"),
        # ((64, 64), ImageLayout.YX, "zarr"),
    ],
)
def test_image_io_round_trip_multichannel(tmp_path, shape, layout, export_format):
    mock_data = np.random.rand(*shape).astype("float32")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        source_file_name="test",
    )
    image = PanSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test",
        key="raw",
        export_format=export_format,
        data_type="float32",
    )

    if export_format == "tiff":
        file_path = tmp_path / "test.tiff"
        key = None
    elif export_format == "h5":
        file_path = tmp_path / "test.h5"
        key = "raw"
    else:
        file_path = tmp_path / "test.zarr"
        key = "raw"

    imported_image = import_image_task(
        input_path=file_path,
        key=key,
        image_name="tesi_import",
        semantic_type="raw",
        stack_layout=layout.name,
        m_slicing=None,
    )
    assert isinstance(imported_image, list)

    original_data = image.get_data()[0]
    imported_data = imported_image[0].get_data()

    assert np.allclose(original_data, imported_data)
    assert original_data.max() <= 1.0  # check if the normalization is applied
    assert imported_data.max() <= 1.0

    assert image.voxel_size == imported_image.voxel_size
    assert image.semantic_type == imported_image.semantic_type
    assert image.image_layout == imported_image.image_layout


@pytest.mark.parametrize(
    "shape,layout,export_format",
    [
        ((32, 64, 64), ImageLayout.ZYX, "tiff"),
        ((32, 64, 64), ImageLayout.ZYX, "h5"),
        ((32, 64, 64), ImageLayout.ZYX, "zarr"),
        ((64, 64), ImageLayout.YX, "tiff"),
        ((64, 64), ImageLayout.YX, "h5"),
        ((64, 64), ImageLayout.YX, "zarr"),
    ],
)
def test_image_io_round_trip(tmp_path, shape, layout, export_format):
    mock_data = np.random.rand(*shape).astype("float32")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        source_file_name="test",
    )
    image = PanSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test",
        key="raw",
        export_format=export_format,
        data_type="float32",
    )

    if export_format == "tiff":
        file_path = tmp_path / "test.tiff"
        key = None
    elif export_format == "h5":
        file_path = tmp_path / "test.h5"
        key = "raw"
    else:
        file_path = tmp_path / "test.zarr"
        key = "raw"

    imported_image = import_image_task(
        input_path=file_path,
        key=key,
        image_name="tesi_import",
        semantic_type="raw",
        stack_layout=layout.name,
        m_slicing=None,
    )
    assert isinstance(imported_image, PanSegImage)

    original_data = image.get_data(normalize_01=True)
    imported_data = imported_image.get_data()

    assert np.allclose(original_data, imported_data)
    assert original_data.max() <= 1.0  # check if the normalization is applied
    assert imported_data.max() <= 1.0

    assert image.voxel_size == imported_image.voxel_size
    assert image.semantic_type == imported_image.semantic_type
    assert image.image_layout == imported_image.image_layout


@pytest.mark.parametrize(
    "shape, layout, export_format",
    [
        ((32, 64, 64), ImageLayout.ZYX, "tiff"),
        ((32, 64, 64), ImageLayout.ZYX, "h5"),
        ((32, 64, 64), ImageLayout.ZYX, "zarr"),
        ((64, 64), ImageLayout.YX, "tiff"),
        ((64, 64), ImageLayout.YX, "h5"),
        ((64, 64), ImageLayout.YX, "zarr"),
    ],
)
def test_label_io_round_trip(tmp_path, shape, layout, export_format):
    mock_data = np.random.randint(0, 10, size=[4] * len(shape))
    repeats = np.array(shape) // 4
    for i, rep in enumerate(repeats):
        mock_data = mock_data.repeat(rep, axis=i)
    mock_data = mock_data.astype("uint16")
    assert np.all(mock_data.shape == shape)

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.SEGMENTATION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test",
        key="raw",
        export_format=export_format,
        data_type="uint16",
        export_mesh="glb" if len(shape) == 3 else "No",
        close_mesh=False,
    )

    if export_format == "tiff":
        file_path = tmp_path / "test.tiff"
        key = None
    elif export_format == "h5":
        file_path = tmp_path / "test.h5"
        key = "raw"
    else:
        file_path = tmp_path / "test.zarr"
        key = "raw"

    imported_image = import_image_task(
        input_path=file_path,
        key=key,
        image_name="tesi_import",
        semantic_type="segmentation",
        stack_layout=layout.name,
        m_slicing=None,
    )
    assert isinstance(imported_image, PanSegImage)

    original_data = image.get_data()[0]
    imported_data = imported_image.get_data()

    assert np.allclose(original_data, imported_data)
    assert original_data.max() > 1.0  # check if the normalization is not applied
    assert imported_data.max() > 1.0

    assert image.voxel_size == imported_image.voxel_size
    assert image.semantic_type == imported_image.semantic_type
    assert image.image_layout == imported_image.image_layout


def test_label_import_image_task_error_message(tmp_path):
    shape = (64, 64)
    layout = ImageLayout.YX
    export_format = "h5"
    mock_data = np.random.randint(0, 10, size=shape).astype("uint16")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.SEGMENTATION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test",
        key="raw",
        export_format=export_format,
        data_type="uint16",
    )

    file_path = tmp_path / "test.h5"
    key = "raw"

    result = import_image_task(
        input_path=file_path,
        key=key,
        image_name="tesi_import",
        semantic_type="segmentation",
        stack_layout="CYX",
        m_slicing=None,
    )
    assert isinstance(result, Task_message)
    assert "Data to import has shape (64, 64)" in result.message


def test_label_io_mesh_error(tmp_path):
    shape = (64, 64)
    export_format = "tiff"

    mock_data = np.random.randint(0, 10, size=[4] * len(shape))
    repeats = np.array(shape) // 4
    for i, rep in enumerate(repeats):
        mock_data = mock_data.repeat(rep, axis=i)
    mock_data = mock_data.astype("uint16")
    assert np.all(mock_data.shape == shape)

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.SEGMENTATION,
        image_layout=ImageLayout.YX,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property)

    with pytest.raises(ValueError):
        export_image_task(
            image=image,
            export_directory=tmp_path,
            name_pattern="test",
            key="raw",
            export_format=export_format,
            data_type="uint16",
            export_mesh="glb",
            close_mesh=True,
        )


def test_io_slicing_trip(tmp_path):
    shape = (32, 64, 64)
    layout = ImageLayout.ZYX
    export_format = "tiff"
    mock_data = np.random.randint(0, 10, size=shape).astype("uint16")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.SEGMENTATION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PanSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test_raw",
        key="raw",
        export_format=export_format,
        data_type="uint16",
    )

    file_path = tmp_path / "test_raw.tiff"
    key = None

    imported_image = import_image_task(
        input_path=file_path,
        key=key,
        image_name="tesi_import",
        semantic_type="segmentation",
        stack_layout="zyx",
        m_slicing="5:10,:, :50",
    )
    assert isinstance(imported_image, PanSegImage)

    imported_data = imported_image.get_data()

    assert imported_data.shape == (5, 64, 50)


def test_merge_channels():
    shape = (32, 64, 64)
    layout = ImageLayout.ZYX
    mock_data = np.random.randint(0, 10, size=shape).astype("uint16")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image_1 = PanSegImage(data=mock_data, properties=property)
    image_2 = PanSegImage(data=mock_data, properties=property)
    image_3 = PanSegImage(data=mock_data, properties=property)

    merged = merge_channels_task(image_1=image_1, image_2=image_2, image_3=image_3)
    assert isinstance(merged, PanSegImage)
    assert merged.semantic_type == SemanticType.RAW
    assert merged.image_layout == ImageLayout.CZYX
    assert merged.shape == (3, 32, 64, 64)


def test_merge_channels_one():
    shape = (32, 64, 64)
    layout = ImageLayout.ZYX
    mock_data = np.random.randint(0, 10, size=shape).astype("uint16")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image_1 = PanSegImage(data=mock_data, properties=property)

    merged = merge_channels_task(images=image_1)
    assert isinstance(merged, PanSegImage)
    assert merged.semantic_type == SemanticType.RAW
    assert merged.image_layout == ImageLayout.ZYX
    assert merged.shape == (32, 64, 64)
