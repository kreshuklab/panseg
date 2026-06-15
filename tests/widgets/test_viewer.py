import pytest

from panseg.viewer_napari.viewer import Panseg_viewer


@pytest.fixture
def panseg_viewer(make_napari_viewer):
    pv = Panseg_viewer(make_napari_viewer())
    pv.init_tabs()
    pv.setup_layer_updates()
    yield pv

    pv.viewer.window._qt_window.close(True, False)


@pytest.fixture
def all_fields(panseg_viewer):
    pv = panseg_viewer
    return {
        pv.input_tab.widget_details_layer_select.layer: "input_tab",  # all
        pv.preprocessing_tab.widget_layer_select.layer: "preprocessing_tab",  # all
        pv.preprocessing_tab.widget_image_pair_operations.layer1: "preprocessing_tab",  # all
        pv.preprocessing_tab.widget_image_pair_operations.layer2: "preprocessing_tab",  # all
        pv.segmentation_tab.widget_layer_select.layer: "segmentation_tab",  # raw
        pv.segmentation_tab.widget_layer_select.prediction: "segmentation_tab",  # pred
        pv.segmentation_tab.widget_layer_select.nuclei: "segmentation_tab",  # all
        pv.segmentation_tab.widget_layer_select.superpixels: "segmentation_tab",  # segm
        pv.postprocessing_tab.widget_layer_select.layer: "postprocessing_tab",  # segm
        pv.postprocessing_tab.widget_fix_segmentation_by_nuclei.segmentation_cells: "postprocessing_tab",  # segm
        pv.postprocessing_tab.widget_fix_segmentation_by_nuclei.segmentation_nuclei: "postprocessing_tab",  # segm
        pv.postprocessing_tab.widget_fix_segmentation_by_nuclei.boundary_pmaps: "postprocessing_tab",  # pred
        pv.output_tab.widget_export_image.image: "output_tab",  # all
        pv.training_tab.widget_unet_training.image: "training_tab",  # raw
        pv.training_tab.widget_unet_training.segmentation: "training_tab",  # segm
    }


@pytest.fixture
def all_raw_fields(panseg_viewer):
    pv = panseg_viewer
    return {
        pv.input_tab.widget_details_layer_select.layer: "input_tab",  # all
        pv.preprocessing_tab.widget_layer_select.layer: "preprocessing_tab",  # all
        pv.preprocessing_tab.widget_image_pair_operations.layer1: "preprocessing_tab",  # all
        pv.preprocessing_tab.widget_image_pair_operations.layer2: "preprocessing_tab",  # all
        pv.segmentation_tab.widget_layer_select.layer: "segmentation_tab",  # raw
        pv.segmentation_tab.widget_layer_select.nuclei: "segmentation_tab",  # all
        pv.output_tab.widget_export_image.image: "output_tab",  # all
        pv.training_tab.widget_unet_training.image: "training_tab",  # raw
    }


@pytest.fixture
def all_pred_fields(panseg_viewer):
    pv = panseg_viewer
    return {
        pv.input_tab.widget_details_layer_select.layer: "input_tab",  # all
        pv.preprocessing_tab.widget_layer_select.layer: "preprocessing_tab",  # all
        pv.preprocessing_tab.widget_image_pair_operations.layer1: "preprocessing_tab",  # all
        pv.preprocessing_tab.widget_image_pair_operations.layer2: "preprocessing_tab",  # all
        pv.segmentation_tab.widget_layer_select.prediction: "segmentation_tab",  # pred
        pv.postprocessing_tab.widget_fix_segmentation_by_nuclei.boundary_pmaps: "postprocessing_tab",  # pred
        pv.output_tab.widget_export_image.image: "output_tab",  # all
    }


@pytest.fixture
def all_segm_fields(panseg_viewer):
    pv = panseg_viewer
    return {
        pv.input_tab.widget_details_layer_select.layer: "input_tab",  # all
        pv.preprocessing_tab.widget_layer_select.layer: "preprocessing_tab",  # all
        pv.preprocessing_tab.widget_image_pair_operations.layer1: "preprocessing_tab",  # all
        pv.preprocessing_tab.widget_image_pair_operations.layer2: "preprocessing_tab",  # all
        pv.segmentation_tab.widget_layer_select.nuclei: "segmentation_tab",  # all
        pv.segmentation_tab.widget_layer_select.superpixels: "segmentation_tab",  # segm
        pv.postprocessing_tab.widget_layer_select.layer: "postprocessing_tab",  # segm
        pv.postprocessing_tab.widget_fix_segmentation_by_nuclei.segmentation_cells: "postprocessing_tab",  # segm
        pv.postprocessing_tab.widget_fix_segmentation_by_nuclei.segmentation_nuclei: "postprocessing_tab",  # segm
        pv.output_tab.widget_export_image.image: "output_tab",  # all
        pv.training_tab.widget_unet_training.segmentation: "training_tab",  # segm
    }


def test_layer_empty(
    all_fields,
):
    for field, name in all_fields.items():
        assert field.choices in ((), (None,)), f"{name}: {field.choices}"


def test_layer_segmentation(
    panseg_viewer,
    all_segm_fields,
    napari_segmentation,
):
    panseg_viewer.viewer.add_layer(napari_segmentation)

    for field, name in all_segm_fields.items():
        assert napari_segmentation in field.choices, f"{name}: {field.choices}"


def test_layer_raw(
    panseg_viewer,
    all_raw_fields,
    napari_raw,
):
    panseg_viewer.viewer.add_layer(napari_raw)

    for field, name in all_raw_fields.items():
        assert napari_raw in field.choices, f"{name}: {field.choices}"


def test_layer_prediction(
    panseg_viewer,
    all_pred_fields,
    napari_prediction,
):
    panseg_viewer.viewer.add_layer(napari_prediction)

    for field, name in all_pred_fields.items():
        assert napari_prediction in field.choices, f"{name}: {field.choices}"


def test_viewer_setup(panseg_viewer, mocker):
    mock_run = mocker.patch("panseg.viewer_napari.viewer.napari.run")
    panseg_viewer.start_viewer()
    tabs = list(panseg_viewer.viewer.window.dock_widgets.keys())
    assert tabs == [
        "Input",
        "Preprocessing",
        "Segmentation",
        "Postprocessing",
        "Proofreading",
        "Output",
        "Train",
    ]
    mock_run.assert_called_once()
