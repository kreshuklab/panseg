import numpy as np
import pytest
from napari.layers import Image

from panseg.core.image import ImageLayout, PanSegImage
from panseg.functionals.dataprocessing.dataprocessing import ImagePairOperation
from panseg.io.voxelsize import VoxelSize
from panseg.tasks import dataprocessing_tasks as dpt
from panseg.tasks.workflow_handler import Task_message


def test_compute_slices_3d_no_rectangle(raw_cell_3d_100x128x128):
    """Test slicing with no rectangle provided."""
    shape = raw_cell_3d_100x128x128.shape
    z_slice, x_slice, y_slice = dpt._compute_slices_3d(None, (10, 50), shape)

    assert z_slice == slice(10, 50)
    assert x_slice == slice(0, shape[1])
    assert y_slice == slice(0, shape[2])


def test_compute_slices_3d_with_broken_rectangle(raw_cell_3d_100x128x128):
    """Test slicing with a valid rectangle."""
    shape = raw_cell_3d_100x128x128.shape
    rectangle = np.array([[10, 20, 30], [10, 20, 30], [50, 80, 90], [50, 80, 90]])
    with pytest.raises(ValueError):
        z_slice, x_slice, y_slice = dpt._compute_slices_3d(rectangle, (0, 30), shape)


def test_compute_slices_3d_with_rectangle(raw_cell_3d_100x128x128):
    """Test slicing with a valid rectangle."""
    shape = raw_cell_3d_100x128x128.shape
    rectangle = np.array([[10, 10, 10], [10, 10, 30], [10, 80, 30], [10, 80, 10]])

    z_slice, x_slice, y_slice = dpt._compute_slices_3d(rectangle, (0, 30), shape)

    assert z_slice == slice(0, 30)
    assert x_slice == slice(10, 80)
    assert y_slice == slice(10, 30)


def test_compute_slices_2d_no_rectangle():
    """Test slicing with None rectangle returns full slices."""
    shape = (100, 100)
    x_slice, y_slice = dpt._compute_slices_2d(None, shape)
    assert x_slice == slice(0, 100)
    assert y_slice == slice(0, 100)


def test_compute_slices_2d_with_rectangle():
    """Test slicing with valid rectangle coordinates."""
    rectangle = np.array(
        [[10, 20], [0, 0], [50, 60]]
    )  # [x_start, y_start], [0, 0], [x_end, y_end]
    shape = (100, 100)
    x_slice, y_slice = dpt._compute_slices_2d(rectangle, shape)
    assert x_slice == slice(10, 50)
    assert y_slice == slice(20, 60)


def test_compute_slices_2d_clamping():
    """Test that coordinates are clamped to image boundaries."""
    rectangle = np.array([[-10, -5], [0, 0], [150, 120]])  # Out of bounds coordinates
    shape = (100, 100)
    x_slice, y_slice = dpt._compute_slices_2d(rectangle, shape)
    assert x_slice == slice(0, 100)
    assert y_slice == slice(0, 100)


def test_cropping():
    np.testing.assert_equal(dpt._cropping(np.zeros((3, 3)), np.array([0, 0])), 0)


def test_image_croppig(napari_raw):
    ps_image = PanSegImage.from_napari_layer(napari_raw)
    dpt.image_cropping_task(image=ps_image)


def test_set_voxel_size_task(napari_raw):
    ps_image = PanSegImage.from_napari_layer(napari_raw)
    out = dpt.set_voxel_size_task(image=ps_image, voxel_size=(0.1, 0.2, 0.3))
    assert out.voxel_size == VoxelSize(voxels_size=(0.1, 0.2, 0.3))


def test_image_rescale_to_shape_task_yx(napari_raw_2d):
    ps_image = PanSegImage.from_napari_layer(napari_raw_2d)
    out = dpt.image_rescale_to_shape_task(image=ps_image, new_shape=(1, 100, 200))
    assert out.shape == (100, 200)


def test_image_rescale_to_shape_task_zyx(napari_raw):
    ps_image = PanSegImage.from_napari_layer(napari_raw)
    out = dpt.image_rescale_to_shape_task(image=ps_image, new_shape=(12, 20, 30))
    assert out.shape == (12, 20, 30)


def test_image_rescale_to_shape_task_cyx(napari_raw):
    napari_raw.metadata["image_layout"] = "CYX"
    ps_image = PanSegImage.from_napari_layer(napari_raw)
    out = dpt.image_rescale_to_shape_task(image=ps_image, new_shape=(15, 20, 30))
    assert out.shape == (10, 20, 30)


def test_image_rescale_to_shape_task_czyx(napari_raw_4d):
    napari_raw_4d.metadata["image_layout"] = "CZYX"
    ps_image = PanSegImage.from_napari_layer(napari_raw_4d)
    out = dpt.image_rescale_to_shape_task(image=ps_image, new_shape=(15, 20, 30))
    assert out.shape == (10, 15, 20, 30)
    assert out.image_layout == ImageLayout.CZYX


def test_image_rescale_to_shape_task_zcyx(napari_raw_4d):
    napari_raw_4d.metadata["image_layout"] = "ZCYX"
    ps_image = PanSegImage.from_napari_layer(napari_raw_4d)
    out = dpt.image_rescale_to_shape_task(image=ps_image, new_shape=(15, 20, 30))
    assert out.shape == (10, 15, 20, 30)
    assert out.image_layout == ImageLayout.CZYX


def test_image_rescale_to_voxel_size_task_yxz(napari_raw):
    ps_image = PanSegImage.from_napari_layer(napari_raw)
    old_voxels_size = (8, 4.5, 6.0)
    new_voxels_size = (4, 9, 12)
    ps_image.properties.voxel_size = VoxelSize(voxels_size=old_voxels_size)
    out = dpt.image_rescale_to_voxel_size_task(
        image=ps_image,
        new_voxels_size=new_voxels_size,
        new_unit="um",
    )
    # ps_image.shape * (old_voxels_size / new_voxels_size)
    assert out.shape == (20, 5, 5)
    assert out.voxel_size.voxels_size == new_voxels_size


def test_image_rescale_to_voxel_size_task_yx(napari_raw_2d):
    ps_image = PanSegImage.from_napari_layer(napari_raw_2d)
    old_voxels_size = (1, 8, 4.5)
    new_voxels_size = (1, 4, 9)
    ps_image.properties.voxel_size = VoxelSize(voxels_size=old_voxels_size)
    out = dpt.image_rescale_to_voxel_size_task(
        image=ps_image,
        new_voxels_size=new_voxels_size,
        new_unit="um",
    )
    # ps_image.shape * (old_voxels_size / new_voxels_size)
    assert out.shape == (20, 5)
    assert out.voxel_size.voxels_size == new_voxels_size


def test_image_rescale_to_voxel_size_task_cyx(napari_raw):
    napari_raw.metadata["image_layout"] = "CYX"
    ps_image = PanSegImage.from_napari_layer(napari_raw)
    old_voxels_size = (99, 8, 4.5)
    new_voxels_size = (5, 4, 9)
    ps_image.properties.voxel_size = VoxelSize(voxels_size=old_voxels_size)
    out = dpt.image_rescale_to_voxel_size_task(
        image=ps_image,
        new_voxels_size=new_voxels_size,
        new_unit="um",
    )
    # ps_image.shape * (old_voxels_size / new_voxels_size)
    assert out.shape == (10, 20, 5)
    assert out.voxel_size.voxels_size == new_voxels_size


def test_image_rescale_to_voxel_size_task_czyx(napari_raw_4d):
    napari_raw_4d.metadata["image_layout"] = "CZYX"
    ps_image = PanSegImage.from_napari_layer(napari_raw_4d)
    old_voxels_size = (8, 4.5, 6.0)
    new_voxels_size = (4, 9, 12)
    ps_image.properties.voxel_size = VoxelSize(voxels_size=old_voxels_size)
    out = dpt.image_rescale_to_voxel_size_task(
        image=ps_image,
        new_voxels_size=new_voxels_size,
        new_unit="um",
    )
    # ps_image.shape * (old_voxels_size / new_voxels_size)
    assert out.shape == (10, 20, 5, 5)
    assert out.voxel_size.voxels_size == new_voxels_size


def test_image_rescale_to_voxel_size_task_zcyx(napari_raw_4d):
    napari_raw_4d.metadata["image_layout"] = "ZCYX"
    ps_image = PanSegImage.from_napari_layer(napari_raw_4d)
    old_voxels_size = (8, 4.5, 6.0)
    new_voxels_size = (4, 9, 12)
    ps_image.properties.voxel_size = VoxelSize(voxels_size=old_voxels_size)
    out = dpt.image_rescale_to_voxel_size_task(
        image=ps_image,
        new_voxels_size=new_voxels_size,
        new_unit="um",
    )
    # ps_image.shape * (old_voxels_size / new_voxels_size)
    assert out.shape == (10, 20, 5, 5)
    assert out.voxel_size.voxels_size == new_voxels_size


def test_remove_false_positives_by_foreground_probability_task(mocker, napari_raw):
    mock_remove = mocker.patch(
        "panseg.tasks.dataprocessing_tasks."
        "remove_false_positives_by_foreground_probability"
    )
    kept = np.random.rand(10, 10, 10)
    removed = np.random.rand(10, 10, 10)
    mock_remove.return_value = [kept, removed]
    ps_image = PanSegImage.from_napari_layer(napari_raw)

    out = dpt.remove_false_positives_by_foreground_probability_task(
        segmentation=ps_image, foreground=ps_image, threshold=0.5
    )
    assert len(out) == 2
    assert np.all(out[0]._data == kept)
    assert np.all(out[1]._data == removed)


def test_set_biggest_instance_to_zero_task(napari_segmentation):
    ps_image = PanSegImage.from_napari_layer(napari_segmentation)
    ps_image._data[:3, :3, :3] = 42
    ps_image._data[6:, 6:, 6:] = 0
    out = dpt.set_biggest_instance_to_zero_task(image=ps_image)
    assert np.all(out._data[:3, :3, :3] == 0)
    assert np.all(out._data[6:, 6:, 6:] == 0)


def test_set_biggest_instance_to_zero_task_bg_is_instance(napari_segmentation):
    ps_image = PanSegImage.from_napari_layer(napari_segmentation)
    ps_image._data[:3, :3, :3] = 42
    ps_image._data[6:, 6:, 6:] = 0
    out = dpt.set_biggest_instance_to_zero_task(
        image=ps_image,
        instance_could_be_zero=True,
    )
    assert np.all(out._data[:3, :3, :3] == 42)
    assert np.all(out._data[6:, 6:, 6:] == 0)
    assert np.all(out._data == ps_image._data)


def test_relabel_segmentation_task_wrong_layer(mocker, napari_raw):
    ps_image = PanSegImage.from_napari_layer(napari_raw)
    mock_relabel = mocker.patch(
        "panseg.tasks.dataprocessing_tasks.relabel_segmentation"
    )
    out = dpt.relabel_segmentation_task(image=ps_image, background=0)
    assert isinstance(out, Task_message)
    assert "must be a segmentation" in out.message


def test_relabel_segmentation_task(mocker, napari_segmentation):
    ps_image = PanSegImage.from_napari_layer(napari_segmentation)
    mock_relabel = mocker.patch(
        "panseg.tasks.dataprocessing_tasks.relabel_segmentation"
    )
    mock_relabel.return_value = np.random.rand(10, 10, 10)
    out = dpt.relabel_segmentation_task(image=ps_image, background=0)
    assert isinstance(out, PanSegImage)


def test_image_pair_operation_task(mocker, napari_raw):
    ps_image = PanSegImage.from_napari_layer(napari_raw)
    mock_process = mocker.patch(
        "panseg.tasks.dataprocessing_tasks.relabel_segmentation"
    )
    mock_process.return_value = np.random.rand(10, 10, 10)
    out = dpt.image_pair_operation_task(
        image1=ps_image,
        image2=ps_image,
        operation="add",
    )
    assert isinstance(out, PanSegImage)
