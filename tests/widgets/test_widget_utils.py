import pytest
from magicgui.widgets import Label, Widget
from napari.layers import Image, Labels

from panseg.core.image import PanSegImage
from panseg.tasks.workflow_handler import task_tracker
from panseg.viewer_napari.widgets import utils


def test_return_value_if_widget_is_widget(mocker):
    mock = mocker.Mock(spec=Widget)
    mock.value = "smth"
    assert utils._return_value_if_widget(mock) == "smth"


def test_return_value_if_widget_no_widget():
    assert utils._return_value_if_widget("smth") == "smth"


def test_div_text():
    w = utils.div("smth")
    assert isinstance(w, Label)


def test_div_text_long():
    w = utils.div("smth" * 20)
    assert isinstance(w, Label)


def test_div_no_text():
    w = utils.div()
    assert isinstance(w, Label)


def test_add_ps_image_to_viewer(make_napari_viewer, napari_raw):
    viewer = make_napari_viewer()
    ps_image = PanSegImage.from_napari_layer(napari_raw)
    utils.add_ps_image_to_viewer(ps_image, False)
    assert ps_image.name in viewer.layers
    assert isinstance(viewer.layers[0], Image)


def test_add_ps_image_to_viewer_label(make_napari_viewer, napari_segmentation):
    viewer = make_napari_viewer()
    ps_image = PanSegImage.from_napari_layer(napari_segmentation)
    utils.add_ps_image_to_viewer(ps_image, False)
    assert ps_image.name in viewer.layers
    assert isinstance(viewer.layers[0], Labels)


def test_add_ps_image_to_viewer_replace(make_napari_viewer, napari_raw):
    viewer = make_napari_viewer()
    ps_image = PanSegImage.from_napari_layer(napari_raw)
    utils.add_ps_image_to_viewer(ps_image, True)
    assert ps_image.name in viewer.layers

    utils.add_ps_image_to_viewer(ps_image, True)
    assert ps_image.name in viewer.layers
    assert len(viewer.layers) == 1


def test_schedule_task(qtbot):
    def mock_task(inp):
        return None

    utils.schedule_task(task_tracker(mock_task), {"inp": "smth"})
    qtbot.wait(50)


def test_schedule_not_task():
    def mock_task(inp):
        return None

    with pytest.raises(ValueError):
        utils.schedule_task(mock_task, {"inp": "smth"})


def test_schedule_wrong_return(qtbot):
    def mock_task(inp):
        return "smth"

    with qtbot.captureExceptions() as exceptions:
        utils.schedule_task(task_tracker(mock_task), {"inp": "smth"})
        qtbot.wait(50)

    assert len(exceptions) == 1


def test_schedule_task_pbar(qtbot, mocker):
    def mock_task(inp, _tracker):
        return None

    mock = mocker.Mock()

    utils.schedule_task(task_tracker(mock_task), {"inp": "smth", "_pbar": mock})
    qtbot.wait(50)
    assert not mock.visible


def test_schedule_task_hide(qtbot, mocker):
    def mock_task(inp):
        return None

    mock = mocker.Mock()

    utils.schedule_task(task_tracker(mock_task), {"inp": "smth", "_to_hide": [mock]})
    qtbot.wait(50)
    mock.hide.assert_called_once()
    mock.show.assert_called_once()


def test_increase_font_size():
    utils.increase_font_size()


def test_decrease_font_size():
    utils.decrease_font_size()


def test_open_docs(mocker):
    mock_browser = mocker.patch("panseg.viewer_napari.widgets.utils.webbrowser")
    help = utils.Help_text()
    assert mocker.sentinel == help.open_docs(mocker.sentinel)
    mock_browser.open.assert_called_once()
