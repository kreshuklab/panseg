from uuid import uuid4

import numpy as np
import pytest
from napari.layers import Image

from panseg.core.image import ImageLayout, SemanticType
from panseg.viewer_napari.widgets.output import Output_Tab


@pytest.fixture(scope="module")
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


def _timeseries_layer(name: str, seed: int, layout: str = "TZYX") -> Image:
    voxel_size = (1.0, 1.0, 1.0)
    shape = (4, 5, 16, 16) if layout == "TZYX" else (4, 16, 16)
    data = np.random.default_rng(seed).random(shape).astype("float32")
    metadata = {
        "semantic_type": SemanticType.RAW,
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": layout,
        "t_spacing": 10.0,
        "t_unit": "s",
        "id": uuid4(),
    }
    return Image(data, metadata=metadata, name=name)


@pytest.mark.parametrize(
    "layout, expected_layout, expected_shape",
    [
        ("TZYX", ImageLayout.TCZYX, (4, 2, 5, 16, 16)),
        ("TYX", ImageLayout.TCYX, (4, 2, 16, 16)),
    ],
)
def test_export_image_merges_timeseries_channels(
    output_tab, mocker, layout, expected_layout, expected_shape
):
    """Merging two single-channel timeseries layers through the
    additional-channels widget exports a TCZYX/TCYX image."""
    mocked_export = mocker.patch(
        "panseg.viewer_napari.widgets.output.export_image_task"
    )
    first = _timeseries_layer("channel_1", seed=1, layout=layout)
    second = _timeseries_layer("channel_2", seed=2, layout=layout)

    output_tab.widget_export_image.image.choices = [first, second]
    output_tab.widget_export_image.image.value = first
    output_tab.widget_export_image.n_channels.value = 1
    output_tab.widget_export_image.n_channels.value = 2
    output_tab.additional_layers[0].value = second

    output_tab.widget_export_image(image=first, n_channels=2)

    exported = mocked_export.call_args.kwargs["image"]
    assert exported.image_layout == expected_layout
    assert exported.shape == expected_shape
    assert exported.properties.t_spacing == 10.0
