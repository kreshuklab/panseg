from pathlib import Path

import h5py
import numpy as np
import pytest

from panseg.core.image import TIME_UNIT_CHOICES
from panseg.io.h5 import create_h5
from panseg.io.tiff import create_tiff
from panseg.io.voxelsize import VoxelSize
from panseg.tasks.dataprocessing_tasks import set_t_spacing_task
from panseg.viewer_napari.widgets.input import (
    Docs_Container,
    Input_Tab,
    InputType,
    PathMode,
)

OME_EXAMPLES = (
    Path(__file__).resolve().parent.parent / "resources" / "ome_tiff_examples"
)


@pytest.fixture
def input_tab():
    """Fixture to create an Input_Tab instance for testing"""
    yield Input_Tab()


def test_input_tab_initialization(input_tab):
    container = input_tab.get_container()

    assert len(container) == 10


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


@pytest.mark.parametrize(
    "t_layout",
    ["TYX", "TCYX", "TZYX", "TCZYX"],
)
def test_open_file_accepts_t_layouts(input_tab, mocker, tmp_path, t_layout):
    """The stack layout is the user's explicit assertion: every T layout
    string is passed through to the import task for every format."""
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.input.schedule_task",
        autospec=True,
    )
    path = tmp_path / "timeseries.h5"
    create_h5(path, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())
    input_tab.path_changed_once = True
    kwargs = {
        "path_mode": True,
        "path": path,
        "stack_layout": t_layout,
        "layer_type": InputType.RAW.value,
        "new_layer_name": "layer",
    }
    input_tab.widget_open_file(**kwargs)

    task_kwargs = mocked_scheduler.call_args.kwargs["task_kwargs"]
    assert task_kwargs["stack_layout"] == t_layout
    assert task_kwargs["input_path"] == path


def test_stack_layout_tooltip_mentions_t(input_tab):
    tooltip = input_tab.widget_open_file.stack_layout.tooltip
    assert "t for time" in tooltip


def test_stack_layout_tooltip_mentions_slice(input_tab):
    tooltip = input_tab.widget_open_file.stack_layout.tooltip
    assert "Truncate the data before importing" in tooltip


def test_open_file_passes_stack_layout_with_slice(input_tab, mocker, tmp_path):
    """The stack layout field hands the inline slice to the import task
    untouched: parsing and cropping happen in import_image."""
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.input.schedule_task",
        autospec=True,
    )
    path = tmp_path / "timeseries.h5"
    create_h5(path, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())
    input_tab.path_changed_once = True
    layout = "tzyx[:3,:,:,:]"
    kwargs = {
        "path_mode": True,
        "path": path,
        "stack_layout": layout,
        "layer_type": InputType.RAW.value,
        "new_layer_name": "layer",
    }
    input_tab.widget_open_file(**kwargs)

    task_kwargs = mocked_scheduler.call_args.kwargs["task_kwargs"]
    assert task_kwargs["stack_layout"] == layout
    assert task_kwargs["input_path"] == path


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


def test_set_t_spacing_hidden_for_still_image(input_tab, napari_raw):
    input_tab.widget_details_layer_select.layer.choices = [napari_raw]
    input_tab.widget_details_layer_select.layer.value = napari_raw
    assert not input_tab.widget_set_t_spacing.visible


def test_set_t_spacing_shown_for_timeseries(input_tab, napari_timeseries):
    input_tab.widget_details_layer_select.layer.choices = [napari_timeseries]
    input_tab.widget_details_layer_select.layer.value = napari_timeseries
    assert input_tab.widget_set_t_spacing.visible


def test_set_t_spacing_schedules_task(input_tab, napari_timeseries, mocker):
    input_tab.widget_details_layer_select.layer.choices = [napari_timeseries]
    input_tab.widget_details_layer_select.layer.value = napari_timeseries

    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.input.schedule_task",
        autospec=True,
    )

    input_tab.t_spacing.value = "30"
    input_tab.widget_set_t_spacing()

    mocked_scheduler.assert_called_once()
    args, kwargs = mocked_scheduler.call_args
    assert args[0] is set_t_spacing_task
    assert kwargs["task_kwargs"]["t_spacing"] == 30.0
    assert kwargs["task_kwargs"]["t_unit"] == "s"


def test_set_t_spacing_unit_choices(input_tab):
    assert list(input_tab.t_unit.choices) == list(TIME_UNIT_CHOICES)


def test_set_t_spacing_number_and_unit_in_horizontal_row(input_tab):
    """The number and its unit are grouped in one horizontal container."""
    combo = input_tab.widget_set_t_spacing.t_spacing_combo
    assert [w.name for w in combo] == ["_t_spacing", "_t_unit"]
    assert combo.layout == "horizontal"


@pytest.mark.parametrize(
    "value, unit, expected",
    [
        ("500", "ms", 500.0),
        ("1000", "µs", 1000.0),
        ("2", "min", 2.0),
        ("1.5", "h", 1.5),
    ],
)
def test_set_t_spacing_passes_value_and_unit(
    input_tab, napari_timeseries, mocker, value, unit, expected
):
    """The widget hands the value and the unit to the task untouched; the
    conversion to seconds happens in the task."""
    input_tab.widget_details_layer_select.layer.choices = [napari_timeseries]
    input_tab.widget_details_layer_select.layer.value = napari_timeseries

    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.input.schedule_task",
        autospec=True,
    )

    input_tab.t_spacing.value = value
    input_tab.t_unit.value = unit
    input_tab.widget_set_t_spacing()

    mocked_scheduler.assert_called_once()
    task_kwargs = mocked_scheduler.call_args.kwargs["task_kwargs"]
    assert task_kwargs["t_spacing"] == expected
    assert task_kwargs["t_unit"] == unit


def test_set_t_spacing_empty_field_is_unknown(input_tab, napari_timeseries, mocker):
    input_tab.widget_details_layer_select.layer.choices = [napari_timeseries]
    input_tab.widget_details_layer_select.layer.value = napari_timeseries

    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.input.schedule_task",
        autospec=True,
    )

    input_tab.t_spacing.value = ""
    input_tab.widget_set_t_spacing()

    mocked_scheduler.assert_called_once()
    task_kwargs = mocked_scheduler.call_args.kwargs["task_kwargs"]
    assert task_kwargs["t_spacing"] is None


def test_set_t_spacing_invalid_field_ignored(
    input_tab, napari_timeseries, mocker, caplog
):
    input_tab.widget_details_layer_select.layer.choices = [napari_timeseries]
    input_tab.widget_details_layer_select.layer.value = napari_timeseries

    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.input.schedule_task",
        autospec=True,
    )

    input_tab.t_spacing.value = "abc"
    input_tab.widget_set_t_spacing()

    mocked_scheduler.assert_not_called()
    assert "abc" in caplog.text


@pytest.mark.parametrize("bad_value", ["0", "-5"])
def test_set_t_spacing_nonpositive_field_ignored(
    input_tab, napari_timeseries, mocker, bad_value
):
    input_tab.widget_details_layer_select.layer.choices = [napari_timeseries]
    input_tab.widget_details_layer_select.layer.value = napari_timeseries

    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.input.schedule_task",
        autospec=True,
    )

    input_tab.t_spacing.value = bad_value
    input_tab.widget_set_t_spacing()

    mocked_scheduler.assert_not_called()


def test_details_info_shows_time_spacing_when_known(input_tab, napari_timeseries):
    input_tab.widget_details_layer_select.layer.choices = [napari_timeseries]
    input_tab.widget_details_layer_select.layer.value = napari_timeseries
    assert "Time spacing: 10 s" in input_tab.widget_info.value


def test_details_info_shows_unknown_time_spacing(
    input_tab, napari_timeseries_unknown_t_spacing
):
    input_tab.widget_details_layer_select.layer.choices = [
        napari_timeseries_unknown_t_spacing
    ]
    input_tab.widget_details_layer_select.layer.value = (
        napari_timeseries_unknown_t_spacing
    )
    assert "Time spacing: None" in input_tab.widget_info.value


def test_details_info_omits_time_spacing_for_still_image(input_tab, napari_raw):
    input_tab.widget_details_layer_select.layer.choices = [napari_raw]
    input_tab.widget_details_layer_select.layer.value = napari_raw
    assert "Time spacing" not in input_tab.widget_info.value


# The stack-layout prefill is pure wiring: a path change copies the io
# layer's guess into the field. The guessing itself (reader axes, file
# attrs, shape heuristics) is covered in tests/io; this pins the wiring
# over one representative source per guess path.
@pytest.mark.parametrize(
    "source, expected",
    [
        pytest.param("ome_anchor", "TZYX", id="ome_reader_axes"),
        pytest.param("h5_axis_order", "TZYX", id="h5_axis_order"),
        pytest.param("tiff_shape", "ZYX", id="tiff_shape_heuristic"),
        pytest.param("h5_old", "", id="h5_no_guess"),
    ],
)
def test_path_change_copies_io_guess_into_stack_layout(
    input_tab, tmp_path, source, expected
):
    if source == "ome_anchor":
        path = OME_EXAMPLES / "4D-series.ome.tif"
    elif source == "h5_axis_order":
        path = tmp_path / "timeseries.h5"
        create_h5(path, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())
        with h5py.File(path, "a") as f:
            f["raw"].attrs["axis_order"] = "TZYX"
    elif source == "tiff_shape":
        path = tmp_path / "out.tiff"
        create_tiff(path, np.empty((10, 20, 30), dtype="float32"), VoxelSize())
    else:
        path = tmp_path / "old.h5"
        create_h5(path, np.empty((4, 5, 16, 16), dtype="float32"), "raw", VoxelSize())

    input_tab.widget_open_file.path.value = path
    assert input_tab.widget_open_file.stack_layout.value == expected


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
