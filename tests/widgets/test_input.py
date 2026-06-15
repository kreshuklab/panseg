from pathlib import Path

import pytest

from panseg.viewer_napari.widgets.input import (
    Docs_Container,
    Input_Tab,
    InputType,
    PathMode,
)


@pytest.fixture
def input_tab():
    """Fixture to create an Input_Tab instance for testing"""
    yield Input_Tab()


def test_input_tab_initialization(input_tab):
    container = input_tab.get_container()

    assert len(container) == 8


def test_input_tab_open_file(input_tab, mocker):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.input.schedule_task",
        autospec=True,
    )
    kwargs = {
        "path_mode": True,
        "path": Path(),
        "stack_layout": "",
        "layer_type": InputType.RAW.value,
        "new_layer_name": "",
    }
    input_tab.widget_open_file(**kwargs)
    mocked_scheduler.assert_not_called()

    input_tab.path_changed_once = True
    input_tab.widget_open_file(**kwargs)
    mocked_scheduler.assert_called_once()


def test_open_file_widget_path_handling(input_tab):
    assert input_tab.widget_open_file.path.mode.value == "r"
    assert input_tab.widget_open_file.path.label == "File path"

    input_tab._on_path_mode_changed(PathMode.FILE.value)
    assert input_tab.widget_open_file.path.mode.value == "r"
    assert input_tab.widget_open_file.path.label == "File path"

    input_tab._on_path_mode_changed(PathMode.DIR.value)
    assert input_tab.widget_open_file.path.mode.value == "d"
    assert input_tab.widget_open_file.path.label == "Zarr path\n(.zarr)"


def test_look_up_dataset_keys_empty(input_tab, zarr_file_empty, mocker):
    mock_gen_name = mocker.patch(
        "panseg.viewer_napari.widgets.input.Input_Tab.generate_layer_name"
    )
    input_tab.look_up_dataset_keys(zarr_file_empty)
    mock_gen_name.assert_not_called()


def test_look_up_dataset_keys_3d(input_tab, zarr_file_3d, mocker):
    mock_gen_name = mocker.patch(
        "panseg.viewer_napari.widgets.input.Input_Tab.generate_layer_name"
    )

    input_tab.look_up_dataset_keys(zarr_file_3d)
    assert input_tab.dataset_key.value in ("raw", "raw_2")
    assert all(n in ("raw", "raw_2") for n in input_tab.dataset_key.choices)
    assert mock_gen_name.call_count == 2


def test_look_up_dataset_keys_h5(input_tab, h5_file, mocker):
    mock_gen_name = mocker.patch(
        "panseg.viewer_napari.widgets.input.Input_Tab.generate_layer_name"
    )

    input_tab.look_up_dataset_keys(h5_file)
    assert input_tab.dataset_key.value in ("/label", "/raw")
    assert all(n in ("/label", "/raw") for n in input_tab.dataset_key.choices)
    assert mock_gen_name.call_count == 2


def test_set_voxel_size(input_tab, napari_raw, mocker):
    input_tab.widget_details_layer_select.layer.choices = [napari_raw]
    input_tab.widget_details_layer_select.layer.value = napari_raw

    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.input.schedule_task",
        autospec=True,
    )

    input_tab.widget_set_voxel_size(input_tab, (1.0, 1.0, 1.0))
    mocked_scheduler.assert_called_once()


def test_on_path_changed(input_tab, mocker):
    mocked_lookup = mocker.patch.object(input_tab, "look_up_dataset_keys")
    assert not input_tab.path_changed_once
    mock_path = mocker.Mock()
    input_tab._on_path_changed(mock_path)
    assert input_tab.path_changed_once
    mocked_lookup.assert_called_with(mock_path)


def test_on_refresh_keys_button(input_tab, mocker):
    mocked_lookup = mocker.patch.object(input_tab, "look_up_dataset_keys")
    input_tab.button_key_refresh.native.click()
    mocked_lookup.assert_called_once()


def test_update_layer_selection(
    input_tab,
    mocker,
    napari_raw,
    napari_segmentation,
    napari_prediction,
    napari_shapes,
    make_napari_viewer_proxy,
):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)
    viewer.add_layer(napari_segmentation)
    viewer.add_layer(napari_prediction)
    viewer.add_layer(napari_shapes)

    assert input_tab.widget_details_layer_select.layer.choices == (None,)

    sentinel = mocker.sentinel
    sentinel.value = napari_prediction
    sentinel.type = "active"
    input_tab.update_layer_selection(sentinel)

    assert napari_raw in input_tab.widget_details_layer_select.layer.choices
    assert napari_prediction in input_tab.widget_details_layer_select.layer.choices
    assert napari_segmentation in input_tab.widget_details_layer_select.layer.choices
    assert napari_shapes not in input_tab.widget_details_layer_select.layer.choices

    assert napari_prediction == input_tab.widget_details_layer_select.layer.value

    sentinel.value = napari_shapes
    input_tab.update_layer_selection(sentinel)

    assert napari_prediction == input_tab.widget_details_layer_select.layer.value


def test_open_docs(mocker):
    docs = Docs_Container()
    mocked_open = mocker.patch(
        "panseg.viewer_napari.widgets.input.webbrowser.open", autospec=True
    )
    docs.open_docs(mocker.sentinel)
    mocked_open.assert_called_once()
