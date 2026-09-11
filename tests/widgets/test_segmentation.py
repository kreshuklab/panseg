import pytest

from panseg.tasks.segmentation_tasks import (
    clustering_segmentation_task,
    lmc_segmentation_task,
)
from panseg.viewer_napari.widgets.segmentation import (
    AGGLOMERATION_MODES,
    Segmentation_Tab,
)


@pytest.fixture
def segmentation_tab():
    return Segmentation_Tab()


def test_segmentation_initialization(segmentation_tab):
    containers = segmentation_tab.get_container()
    assert len(containers) == 10


def test_set_separate_steps_list(segmentation_tab):
    segmentation_tab.set_separate_steps_list()
    containers = segmentation_tab.get_container()
    assert len(containers) == 13


def test_watershed_default_mode(segmentation_tab):
    assert segmentation_tab.widget_dt_ws.stacked.value is False
    assert segmentation_tab.widget_aio_ws.stacked.value is False
    assert segmentation_tab.widget_dt_ws.stacked.choices == (True, False)
    assert segmentation_tab.widget_aio_ws.stacked.choices == (True, False)


def test_toggle_visibility_1(segmentation_tab, mocker):
    mocked_switch_2 = mocker.patch.object(segmentation_tab, "toggle_visibility_2")
    mocked_switch_3 = mocker.patch.object(segmentation_tab, "toggle_visibility_3")

    segmentation_tab.toggle_visibility_1(True)
    mocked_switch_2.assert_called_with(False)
    mocked_switch_3.assert_called_with(False)

    mocked_switch_2.reset_mock()
    mocked_switch_3.reset_mock()
    segmentation_tab.toggle_visibility_1(False)
    mocked_switch_2.assert_not_called()
    mocked_switch_3.assert_not_called()


def test_toggle_visibility_2(segmentation_tab, mocker):
    mocked_switch_1 = mocker.patch.object(segmentation_tab, "toggle_visibility_1")
    mocked_switch_3 = mocker.patch.object(segmentation_tab, "toggle_visibility_3")

    segmentation_tab.toggle_visibility_2(True)
    mocked_switch_1.assert_called_with(False)
    mocked_switch_3.assert_called_with(False)

    mocked_switch_1.reset_mock()
    mocked_switch_3.reset_mock()
    segmentation_tab.toggle_visibility_2(False)
    mocked_switch_1.assert_not_called()
    mocked_switch_3.assert_not_called()


def test_toggle_visibility_3(segmentation_tab, mocker):
    mocked_switch_1 = mocker.patch.object(segmentation_tab, "toggle_visibility_1")
    mocked_switch_2 = mocker.patch.object(segmentation_tab, "toggle_visibility_2")

    segmentation_tab.toggle_visibility_3(True)
    mocked_switch_1.assert_called_with(False)
    mocked_switch_2.assert_called_with(False)

    mocked_switch_1.reset_mock()
    mocked_switch_2.reset_mock()
    segmentation_tab.toggle_visibility_3(False)
    mocked_switch_1.assert_not_called()
    mocked_switch_2.assert_not_called()


def test_toggle_visibility_aio_1(segmentation_tab, mocker):
    mocked_switch_2 = mocker.patch.object(segmentation_tab, "toggle_visibility_aio_2")

    segmentation_tab.toggle_visibility_aio_1(False)
    mocked_switch_2.assert_not_called()

    segmentation_tab.toggle_visibility_aio_1(True)
    mocked_switch_2.assert_called_with(False)


def test_toggle_visibility_aio_2(segmentation_tab, mocker):
    mocked_switch_1 = mocker.patch.object(segmentation_tab, "toggle_visibility_aio_1")

    segmentation_tab.toggle_visibility_aio_2(False)
    mocked_switch_1.assert_not_called()

    segmentation_tab.toggle_visibility_aio_2(True)
    mocked_switch_1.assert_called_with(False)


def test_dt_watershed_no_prediction(segmentation_tab, mocker):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (None,)
    segmentation_tab.widget_layer_select.prediction.value = None
    segmentation_tab.widget_dt_ws(stacked=False)
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Please run a boundary prediction first!",
        thread="Segmentation",
        level="WARNING",
    )


def test_dt_watershed(segmentation_tab, mocker, napari_prediction):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_dt_ws(stacked=False)
    mocked_scheduler.assert_called_once()
    mocked_log.assert_not_called()


def test_dt_watershed_modes(segmentation_tab, mocker, napari_prediction):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction

    expected_params = {
        True: {"stacked": True, "blockwise": False},
        False: {"stacked": False, "blockwise": True},
    }
    for mode, expected in expected_params.items():
        mocked_scheduler.reset_mock()
        segmentation_tab.widget_dt_ws(stacked=mode)
        mocked_scheduler.assert_called_once()
        task_kwargs = mocked_scheduler.call_args.kwargs["task_kwargs"]
        assert task_kwargs["stacked"] is expected["stacked"]
        assert task_kwargs["blockwise"] is expected["blockwise"]
    mocked_log.assert_not_called()


def test_agglomeration_no_layer(segmentation_tab, mocker):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.layer.choices = (None,)
    segmentation_tab.widget_layer_select.layer.value = None

    segmentation_tab.widget_agglomeration()
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Please run a prediction first!",
        thread="Segmentation",
        level="WARNING",
    )


def test_agglomeration_no_superpixels(segmentation_tab, mocker, napari_prediction):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_layer_select.superpixels.choices = (None,)
    segmentation_tab.widget_layer_select.superpixels.value = None

    segmentation_tab.widget_agglomeration()
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Please run `Boundary to Superpixels` first!",
        thread="Segmentation",
        level="WARNING",
    )


def test_agglomeration(
    segmentation_tab, mocker, napari_prediction, napari_segmentation
):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_layer_select.superpixels.choices = (napari_segmentation,)
    segmentation_tab.widget_layer_select.superpixels.value = napari_segmentation

    segmentation_tab.widget_agglomeration(mode=AGGLOMERATION_MODES[0][1])
    mocked_scheduler.assert_called_once()
    assert clustering_segmentation_task == mocked_scheduler.call_args[0][0]
    mocked_log.assert_not_called()

    mocked_log.reset_mock()
    mocked_scheduler.reset_mock()

    segmentation_tab.widget_agglomeration(mode=AGGLOMERATION_MODES[1][1])
    mocked_scheduler.assert_called_once()
    assert clustering_segmentation_task == mocked_scheduler.call_args[0][0]
    mocked_log.assert_not_called()

    mocked_log.reset_mock()
    mocked_scheduler.reset_mock()

    segmentation_tab.widget_agglomeration(mode=AGGLOMERATION_MODES[2][1])
    mocked_scheduler.assert_called_once()
    assert clustering_segmentation_task == mocked_scheduler.call_args[0][0]
    mocked_log.assert_not_called()


def test_agglomeration_lmc(
    segmentation_tab, mocker, napari_raw, napari_prediction, napari_segmentation
):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_layer_select.superpixels.choices = (napari_segmentation,)
    segmentation_tab.widget_layer_select.superpixels.value = napari_segmentation
    segmentation_tab.widget_layer_select.nuclei.choices = (napari_raw,)
    segmentation_tab.widget_layer_select.nuclei.value = napari_raw
    segmentation_tab.widget_agglomeration(mode=AGGLOMERATION_MODES[3][1])
    mocked_scheduler.assert_called_once()
    assert lmc_segmentation_task == mocked_scheduler.call_args[0][0]
    mocked_log.assert_not_called()


def test_agglomeration_lmc_missing(
    segmentation_tab, mocker, napari_raw, napari_prediction, napari_segmentation
):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_layer_select.superpixels.choices = (napari_segmentation,)
    segmentation_tab.widget_layer_select.superpixels.value = napari_segmentation
    segmentation_tab.widget_layer_select.nuclei.choices = (None,)
    segmentation_tab.widget_layer_select.nuclei.value = None
    segmentation_tab.widget_agglomeration(mode=AGGLOMERATION_MODES[3][1])
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Nuclei must be an Image or Labels layer",
        thread="Segmentation",
        level="WARNING",
    )


def test_on_mode_changed(segmentation_tab, mocker):
    mocked_show = mocker.patch.object(
        segmentation_tab.widget_layer_select.nuclei, "show"
    )
    mocked_hide = mocker.patch.object(
        segmentation_tab.widget_layer_select.nuclei, "hide"
    )

    segmentation_tab.widget_agglomeration.mode.value = AGGLOMERATION_MODES[0][1]
    mocked_show.assert_not_called()
    mocked_hide.assert_not_called()
    mocked_show.reset_mock()
    mocked_hide.reset_mock()

    segmentation_tab.widget_agglomeration.mode.value = AGGLOMERATION_MODES[1][1]
    mocked_show.assert_not_called()
    mocked_hide.assert_called_once()
    mocked_show.reset_mock()
    mocked_hide.reset_mock()
    segmentation_tab.widget_agglomeration.mode.value = AGGLOMERATION_MODES[2][1]
    mocked_show.assert_not_called()
    mocked_hide.assert_called_once()
    mocked_show.reset_mock()
    mocked_hide.reset_mock()
    segmentation_tab.widget_agglomeration.mode.value = AGGLOMERATION_MODES[3][1]
    mocked_show.assert_called_once()
    mocked_hide.assert_not_called()
    mocked_show.reset_mock()
    mocked_hide.reset_mock()


def test_on_show_advanced_changed(segmentation_tab):
    segmentation_tab.widget_dt_ws.show_advanced.value = True
    segmentation_tab.widget_dt_ws.show_advanced.value = False


def test_update_layer_selection(
    segmentation_tab,
    mocker,
    napari_raw,
    napari_prediction,
    napari_segmentation,
    make_napari_viewer_proxy,
):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)

    sentinel = mocker.sentinel
    sentinel.value = napari_raw
    sentinel.type = "inserted"
    assert segmentation_tab.widget_layer_select.layer.value is None
    assert segmentation_tab.widget_layer_select.nuclei.value is None
    assert segmentation_tab.widget_layer_select.prediction.value is None
    assert segmentation_tab.widget_layer_select.superpixels.value is None

    segmentation_tab.update_layer_selection(sentinel)
    assert segmentation_tab.widget_layer_select.layer.value == sentinel.value
    assert segmentation_tab.widget_layer_select.nuclei.value == sentinel.value

    viewer.add_layer(napari_prediction)
    sentinel.value = napari_prediction
    sentinel.type = "inserted"
    segmentation_tab.update_layer_selection(sentinel)
    assert segmentation_tab.widget_layer_select.prediction.value == sentinel.value

    viewer.add_layer(napari_segmentation)
    sentinel.value = napari_segmentation
    sentinel.type = "inserted"
    segmentation_tab.update_layer_selection(sentinel)
    assert segmentation_tab.widget_layer_select.superpixels.value == sentinel.value


def test_show_button(segmentation_tab, mocker):
    mocked_switch_1 = mocker.patch.object(segmentation_tab, "toggle_visibility_aio_1")
    segmentation_tab.widget_show_prediction_aio()
    mocked_switch_1.assert_called_with(visible=True)


def test_aio_ws_no_boundary(segmentation_tab, mocker):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )

    segmentation_tab.widget_layer_select.prediction.choices = (None,)
    segmentation_tab.widget_layer_select.prediction.value = None
    segmentation_tab.widget_aio_ws(
        stacked=False,
    )
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Please run a boundary prediction first!",
        thread="Segmentation",
        level="WARNING",
    )


def test_aio_ws_missing_lmc(segmentation_tab, mocker, napari_prediction):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )

    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_layer_select.nuclei.choices = (None,)
    segmentation_tab.widget_layer_select.nuclei.value = None
    segmentation_tab.widget_aio_ws(mode=AGGLOMERATION_MODES[3][1])
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Nuclei must be an Image or Labels layer",
        thread="Segmentation",
        level="WARNING",
    )


def test_aio_ws_lmc(segmentation_tab, mocker, napari_prediction, napari_raw):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )

    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_layer_select.nuclei.choices = (napari_raw,)
    segmentation_tab.widget_layer_select.nuclei.value = napari_raw

    segmentation_tab.widget_aio_ws(mode=AGGLOMERATION_MODES[3][1])
    mocked_scheduler.assert_called_once()
    mocked_log.assert_not_called()


def test_aio_ws_modes(segmentation_tab, mocker, napari_prediction):
    mocked_scheduler = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="panseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction

    expected_params = {
        True: {"stacked": True, "blockwise": False},
        False: {"stacked": False, "blockwise": True},
    }
    for mode, expected in expected_params.items():
        mocked_scheduler.reset_mock()
        segmentation_tab.widget_aio_ws(stacked=mode)
        mocked_scheduler.assert_called_once()
        task_kwargs = mocked_scheduler.call_args.kwargs["task_kwargs"]
        assert task_kwargs["stacked"] is expected["stacked"]
        assert task_kwargs["blockwise"] is expected["blockwise"]
        assert task_kwargs["mode"] == AGGLOMERATION_MODES[0][1]
    mocked_log.assert_not_called()
