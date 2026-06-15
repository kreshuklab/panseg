import pytest

from panseg.viewer_napari.widgets.output import Output_Tab


@pytest.fixture
def output_tab():
    return Output_Tab()


def test_output_initialization(output_tab):
    container = output_tab.get_container()
    assert len(container) == 3


def test_export_shape(output_tab, mocker, napari_shapes):
    mocked_export = mocker.patch(
        "panseg.viewer_napari.widgets.output.export_image_task"
    )
    output_tab.widget_export_image(image=None)
    mocked_export.assert_not_called()
    output_tab.widget_export_image(image=napari_shapes)
    mocked_export.assert_not_called()


def test_export_image(output_tab, mocker, napari_raw):
    mocked_export = mocker.patch(
        "panseg.viewer_napari.widgets.output.export_image_task"
    )
    output_tab.widget_export_image(image=napari_raw)
    mocked_export.assert_called_once()


def test_toggle_export_details_widget(output_tab, mocker):
    for w in output_tab.export_details:
        mocked_w_show = mocker.patch.object(w, "show")
        mocked_w_hide = mocker.patch.object(w, "hide")

    output_tab._toggle_export_details_widgets(True)
    mocked_w_show.assert_called_once()

    output_tab._toggle_export_details_widgets(False)
    mocked_w_hide.assert_called_once()


def test_toggle_key(output_tab, mocker):
    mocked_show = mocker.patch.object(output_tab.widget_export_image.key, "show")
    mocked_hide = mocker.patch.object(output_tab.widget_export_image.key, "hide")

    output_tab._toggle_key(False)
    mocked_hide.assert_called_once()
    mocked_show.assert_not_called()
    output_tab.widget_export_image.export_format.value = "h5"
    mocked_show.assert_called_once()


def test_on_images_changed_none(output_tab, mocker):
    mocked_toggle = mocker.patch.object(output_tab, "_toggle_key")
    output_tab._on_images_changed(image=None)
    mocked_toggle.assert_called_with(False)


def test_on_images_changed(output_tab, mocker, napari_raw):
    mocked_toggle = mocker.patch.object(output_tab, "_toggle_key")
    output_tab._on_images_changed(image=napari_raw)
    mocked_toggle.assert_called_with(True)


def test_on_images_changed_labels(output_tab, mocker, napari_segmentation):
    mocked_toggle = mocker.patch.object(output_tab, "_toggle_key")
    output_tab.widget_export_image.data_type.value = "float32"
    output_tab._on_images_changed(image=napari_segmentation)
    mocked_toggle.assert_called_with(True)
    assert output_tab.widget_export_image.data_type.value == "uint16"
