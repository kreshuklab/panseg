import pytest
from napari.layers import Shapes

from panseg.tasks.dataprocessing_tasks import (
    image_rescale_to_shape_task,
    image_rescale_to_voxel_size_task,
    set_voxel_size_task,
)
from panseg.viewer_napari.widgets.preprocessing import (
    Preprocessing_Tab,
    RescaleModes,
)


@pytest.fixture
def preprocessing_tab():
    return Preprocessing_Tab()


def test_preprocessing_tab_initialization(preprocessing_tab):
    container = preprocessing_tab.get_container()
    assert len(container) == 15


def test_on_layer_select_raw(preprocessing_tab, napari_raw, mocker):
    mocked_hide = mocker.patch.object(
        preprocessing_tab.widget_rescaling.rescaling_factor[0], "hide"
    )
    mocked_show = mocker.patch.object(
        preprocessing_tab.widget_rescaling.rescaling_factor[0], "show"
    )

    preprocessing_tab._on_layer_selection(napari_raw)

    mocked_show.assert_called_once()
    mocked_hide.assert_not_called()


def test_on_layer_select_label(preprocessing_tab, napari_segmentation):
    assert preprocessing_tab.widget_rescaling.order.value == 0
    preprocessing_tab.widget_rescaling.order.value = 1

    preprocessing_tab._on_layer_selection(napari_segmentation)
    assert preprocessing_tab.widget_rescaling.order.value == 0


def test_on_layer_select_shapes(preprocessing_tab, napari_shapes):
    with pytest.raises(ValueError):
        preprocessing_tab._on_layer_selection(napari_shapes)

    with pytest.raises(ValueError):
        preprocessing_tab._on_layer_selection(None)


def test_on_rescale_order_change_no_image(preprocessing_tab):
    assert preprocessing_tab.widget_rescaling.order.value == 0

    preprocessing_tab.widget_layer_select.layer.choices = (None,)
    preprocessing_tab.widget_layer_select.layer.value = None
    assert preprocessing_tab._on_rescale_order_changed(0) is None


def test_on_rescale_order_change_labels(preprocessing_tab, napari_segmentation):
    assert preprocessing_tab.widget_rescaling.order.value == 0

    preprocessing_tab.widget_layer_select.layer.choices = (napari_segmentation,)
    preprocessing_tab.widget_layer_select.layer.value = napari_segmentation
    preprocessing_tab._on_rescale_order_changed(1)
    assert preprocessing_tab.widget_rescaling.order.value == 0


def test_toggle_visibility_1(preprocessing_tab, mocker):
    mocked_switch_2 = mocker.patch.object(preprocessing_tab, "toggle_visibility_2")

    preprocessing_tab.toggle_visibility_1(True)
    mocked_switch_2.assert_called_with(False)

    mocked_switch_2.reset_mock()
    preprocessing_tab.toggle_visibility_1(False)
    mocked_switch_2.assert_not_called()


def test_toggle_visibility_2(preprocessing_tab, mocker):
    mocked_switch_1 = mocker.patch.object(preprocessing_tab, "toggle_visibility_1")

    preprocessing_tab.toggle_visibility_2(True)
    mocked_switch_1.assert_called_with(False)

    mocked_switch_1.reset_mock()
    preprocessing_tab.toggle_visibility_2(False)
    mocked_switch_1.assert_not_called()


def test_toggle_visibility_3(preprocessing_tab, mocker):
    mocked_switch_1 = mocker.patch.object(preprocessing_tab, "toggle_visibility_1")
    preprocessing_tab.toggle_visibility_3(False)
    assert preprocessing_tab.hidden
    preprocessing_tab.toggle_visibility_3(True)
    mocked_switch_1.assert_called_with(True)
    assert not preprocessing_tab.hidden


def test_gaussian_smoothing_no_layer(preprocessing_tab, mocker):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    preprocessing_tab.widget_layer_select.layer.choices = (None,)
    preprocessing_tab.widget_layer_select.layer.value = None
    assert preprocessing_tab.widget_gaussian_smoothing() is None
    mocked_scheduler.assert_not_called()


def test_gaussian_smoothing_image(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    preprocessing_tab.widget_layer_select.layer.choices = (None,)
    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)
    preprocessing_tab.widget_gaussian_smoothing()
    mocked_scheduler.assert_called_once()


def test_cropping_no_image(preprocessing_tab, mocker):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.log",
        autospec=True,
    )

    preprocessing_tab.widget_layer_select.layer.choices = (None,)
    preprocessing_tab.widget_layer_select.layer.value = None
    preprocessing_tab.widget_cropping(crop_roi=None)
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Please select a layer to crop first!",
        thread="Preprocessing",
        level="WARNING",
    )


def test_cropping_no_shapes(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.log",
        autospec=True,
    )

    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)

    assert preprocessing_tab.widget_cropping(crop_roi=None) is None
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Please create a Shapes layer first!",
        thread="Preprocessing",
        level="WARNING",
    )


def test_cropping_too_many_rectangels(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.log",
        autospec=True,
    )

    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)
    shape = Shapes(
        data=[
            [[0, 0], [0, 1], [1, 1], [1, 0]],
            [[10, 10], [10, 11], [11, 11], [11, 10]],
        ]
    )
    preprocessing_tab.widget_cropping(crop_roi=shape)
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Can't use more than one rectangle for cropping!",
        thread="Preprocessing",
        level="WARNING",
    )


def test_cropping_ellipse(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.log",
        autospec=True,
    )

    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)
    shape = Shapes(
        data=[
            [[0, 0], [0, 1], [1, 1], [1, 0]],
        ]
    )
    shape.shape_type = ["ellipse"]
    preprocessing_tab.widget_cropping(crop_roi=shape)
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "A rectangle shape must be used for cropping",
        thread="Preprocessing",
        level="WARNING",
    )


def test_cropping_one_rectangle(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    shape = Shapes(
        data=[
            [[0, 0], [0, 1], [1, 1], [1, 0]],
        ]
    )
    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)
    preprocessing_tab.widget_cropping(crop_roi=shape)
    mocked_scheduler.assert_called_once()


def test_cropping_z_range(preprocessing_tab, mocker, napari_raw):
    mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    # is uninitialized
    assert preprocessing_tab.widget_cropping.crop_z.value == (0, 100)

    preprocessing_tab.initialised_widget_cropping = True
    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)

    assert preprocessing_tab.widget_cropping.crop_z.value == (0, 10)


def test_cropping_activation(
    preprocessing_tab, mocker, napari_raw, make_napari_viewer_proxy
):
    viewer = make_napari_viewer_proxy()
    mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )

    shape = Shapes(
        data=[
            [[0, 0], [0, 1], [1, 1], [1, 0]],
        ]
    )
    viewer.add_layer(shape)
    assert preprocessing_tab.widget_cropping.crop_roi.value is None
    preprocessing_tab.widget_cropping.crop_roi.choices = (shape,)
    assert preprocessing_tab.widget_cropping.crop_roi.value == shape

    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)
    assert not preprocessing_tab.initialised_widget_cropping

    mock_event = mocker.Mock()
    mock_event.value = shape
    mock_event.type = "inserted"

    preprocessing_tab.update_layer_selection(mock_event)
    assert preprocessing_tab.initialised_widget_cropping


def test_cropping_no_rectangle(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    shape = Shapes()
    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)
    preprocessing_tab.widget_cropping(crop_roi=shape)
    mocked_scheduler.assert_called_once()


def test_rescaling_no_voxelsize(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.log",
        autospec=True,
    )
    napari_raw._metadata["original_voxel_size"]["voxels_size"] = None
    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)

    preprocessing_tab.widget_rescaling()
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Original voxel size is missing, please set the voxel size manually.",
        thread="Preprocessing",
        level="WARNING",
    )


def test_rescaling_set_voxel_size(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)

    preprocessing_tab.widget_rescaling(mode=RescaleModes.SET_VOXEL_SIZE)
    assert set_voxel_size_task == mocked_scheduler.call_args[0][0]


def test_rescaling_to_shape_none(preprocessing_tab, napari_raw):
    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)

    with pytest.raises(ValueError):
        preprocessing_tab.widget_rescaling(mode=RescaleModes.TO_SHAPE)


def test_rescaling_to_shape(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)

    preprocessing_tab.widget_rescaling(
        mode=RescaleModes.TO_SHAPE, reference_layer=napari_raw
    )
    assert image_rescale_to_shape_task == mocked_scheduler.call_args[0][0]


def test_rescaling_from_factor(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)

    preprocessing_tab.widget_rescaling(mode=RescaleModes.FROM_FACTOR)
    assert image_rescale_to_voxel_size_task == mocked_scheduler.call_args[0][0]


def test_rescaling_to_voxel_size(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)

    preprocessing_tab.widget_rescaling(mode=RescaleModes.TO_VOXEL_SIZE)
    assert image_rescale_to_voxel_size_task == mocked_scheduler.call_args[0][0]


def test_rescaling_to_model_voxel_size(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    preprocessing_tab.widget_layer_select.layer.choices = (napari_raw,)

    preprocessing_tab.widget_rescaling(mode=RescaleModes.TO_MODEL_VOXEL_SIZE)
    assert image_rescale_to_voxel_size_task == mocked_scheduler.call_args[0][0]


def test_image_pair_operations_missing(preprocessing_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.preprocessing.log",
        autospec=True,
    )
    assert preprocessing_tab.widget_image_pair_operations(napari_raw, None) is None
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Please select two layers first!",
        thread="Preprocessing",
        level="WARNING",
    )


def test_update_layer_selection(
    preprocessing_tab,
    mocker,
    napari_raw,
    napari_prediction,
    napari_segmentation,
    napari_shapes,
    make_napari_viewer_proxy,
):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)
    viewer.add_layer(napari_prediction)
    viewer.add_layer(napari_segmentation)
    viewer.add_layer(napari_shapes)

    assert preprocessing_tab.widget_layer_select.layer.choices == ()

    sentinel = mocker.sentinel
    sentinel.value = napari_prediction
    sentinel.type = "active"
    preprocessing_tab.update_layer_selection(sentinel)

    assert napari_raw in preprocessing_tab.widget_layer_select.layer.choices
    assert napari_prediction in preprocessing_tab.widget_layer_select.layer.choices
    assert napari_segmentation in preprocessing_tab.widget_layer_select.layer.choices
    assert napari_shapes not in preprocessing_tab.widget_layer_select.layer.choices

    sentinel.type = "inserted"
    preprocessing_tab.update_layer_selection(sentinel)
    assert preprocessing_tab.widget_layer_select.layer.value == napari_prediction

    sentinel.value = napari_shapes
    preprocessing_tab.widget_cropping.crop_roi.choices = (napari_shapes,)
    preprocessing_tab.update_layer_selection(sentinel)


def test_on_cropping_image_changed_shape(preprocessing_tab, napari_shapes):
    preprocessing_tab.initialised_widget_cropping = True
    with pytest.raises(AssertionError):
        preprocessing_tab._on_cropping_image_changed(napari_shapes)


def test_on_cropping_image_changed_none(preprocessing_tab):
    preprocessing_tab.initialised_widget_cropping = True
    assert preprocessing_tab._on_cropping_image_changed(None) is None


def test_on_cropping_image_changed_2d(preprocessing_tab, mocker, napari_raw_2d):
    preprocessing_tab.initialised_widget_cropping = True
    mocked_hide = mocker.patch.object(preprocessing_tab.widget_cropping.crop_z, "hide")
    assert preprocessing_tab._on_cropping_image_changed(napari_raw_2d) is None
    mocked_hide.assert_called_once()


def test_on_cropping_image_changed_raw(preprocessing_tab, mocker, napari_raw):
    preprocessing_tab.initialised_widget_cropping = True
    mocked_show = mocker.patch.object(preprocessing_tab.widget_cropping.crop_z, "show")
    assert preprocessing_tab._on_cropping_image_changed(napari_raw) is None
    mocked_show.assert_called_once()
