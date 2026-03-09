from collections import deque
from pathlib import Path
from time import sleep, time

import h5py
import numpy as np
import pytest
from napari.qt.threading import WorkerBase, thread_worker

from panseg.core.image import PanSegImage
from panseg.viewer_napari.widgets import proofreading


def test_copy_if_not_none(mocker):
    mock = mocker.patch.object(mocker.sentinel, "copy")
    assert proofreading.copy_if_not_none(mocker.sentinel)
    mock.assert_called_once()

    assert proofreading.copy_if_not_none(None) is None


def test_get_current_viewer(make_napari_viewer_proxy):
    with pytest.raises(RuntimeError):
        proofreading.get_current_viewer_wrapper()
    viewer = make_napari_viewer_proxy()
    assert proofreading.get_current_viewer_wrapper() == viewer


def test_update_layer_empty(make_napari_viewer_proxy, napari_segmentation):
    viewer = make_napari_viewer_proxy()
    proofreading.update_layer(
        napari_segmentation.data, layer_name="test", scale=napari_segmentation.scale
    )
    np.testing.assert_array_equal(viewer.layers[0].data, napari_segmentation.data)
    np.testing.assert_array_equal(viewer.layers[0].scale, napari_segmentation.scale)
    assert viewer.layers[0].name == "test"


def test_update_layer(make_napari_viewer_proxy, napari_segmentation):
    viewer = make_napari_viewer_proxy()
    viewer.add_labels(np.zeros((5, 5, 5), dtype=int), name="test", scale=[1, 1, 1])
    proofreading.update_layer(
        napari_segmentation.data, layer_name="test", scale=napari_segmentation.scale
    )
    np.testing.assert_array_equal(viewer.layers[0].data, napari_segmentation.data)
    np.testing.assert_array_equal(viewer.layers[0].scale, napari_segmentation.scale)
    assert viewer.layers[0].name == "test"


def test_update_corrected_cells_mask_layer(mocker):
    mock = mocker.patch("panseg.viewer_napari.widgets.proofreading.update_layer")
    proofreading.update_corrected_cells_mask_layer(
        data=mocker.sentinel, scale=mocker.sentinel
    )

    mock.assert_called_with(
        mocker.sentinel,
        proofreading.CORRECTED_CELLS_LAYER_NAME,
        scale=mocker.sentinel,
        colormap=proofreading.correct_cells_cmap,
        opacity=1,
    )


def test_update_scribbles_layer(mocker):
    mock = mocker.patch("panseg.viewer_napari.widgets.proofreading.update_layer")
    proofreading.update_scribbles_layer(data=mocker.sentinel, scale=mocker.sentinel)
    mock.assert_called_with(
        mocker.sentinel, proofreading.SCRIBBLES_LAYER_NAME, scale=mocker.sentinel
    )


def test_update_region_empty(mocker, make_napari_viewer_proxy, napari_raw):
    viewer = make_napari_viewer_proxy()
    with pytest.raises(ValueError):
        proofreading.update_region(
            napari_raw.data,
            layer_name="test",
            region_slice=(slice(0, 5), slice(0, 5), slice(0, 5)),
            scale=napari_raw.scale,
        )


def test_update_region(make_napari_viewer_proxy, napari_segmentation, napari_raw):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(napari_raw.data, name="test", scale=[1, 1, 1])
    sl = (slice(0, 2), slice(0, 2), slice(0, 2))
    proofreading.update_region(
        napari_segmentation.data[sl],
        layer_name="test",
        region_slice=sl,
        scale=napari_segmentation.scale,
    )
    np.testing.assert_array_equal(
        viewer.layers[0].data[sl], napari_segmentation.data[sl]
    )
    np.testing.assert_array_equal(
        viewer.layers[0].data[(slice(2, -1), slice(2, -1), slice(2, -1))],
        napari_raw.data[
            (
                slice(2, -1),
                slice(2, -1),
                slice(2, -1),
            )
        ],
    )
    np.testing.assert_array_equal(viewer.layers[0].scale, napari_segmentation.scale)
    assert viewer.layers[0].name == "test"


def test_get_layer_data_empty(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    with pytest.raises(ValueError):
        proofreading.get_layer_data("test")


def test_get_layer_data(make_napari_viewer_proxy, napari_raw):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)
    assert np.all(proofreading.get_layer_data("test_image_3D") == napari_raw.data)


def test_get_layer_region_data_empty(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    with pytest.raises(ValueError):
        proofreading.get_layer_region_data("test", (slice(0, 1),))


def test_get_layer_region_data(make_napari_viewer_proxy, napari_raw):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)
    sl = (slice(0, 2), slice(0, 2), slice(0, 2))
    assert np.all(
        proofreading.get_layer_region_data("test_image_3D", sl) == napari_raw.data[sl]
    )


def test_preserve_labels(make_napari_viewer_proxy, napari_raw):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)
    proofreading.preserve_labels("test_image_3D")
    assert viewer.layers["test_image_3D"].preserve_labels


@pytest.fixture
def proof():
    return proofreading.ProofreadingHandler()


class TestProofreadingHandler:
    def test_init(self, proof):
        proof

    def test_reset_scribbles(self, mocker, proof):
        mock = mocker.patch("panseg.viewer_napari.widgets.proofreading.update_layer")
        proof.reset_scribbles()
        mock.assert_not_called()

        proof._state.active = True
        mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            segmentation=mocker.DEFAULT,
            scale=mocker.DEFAULT,
        )
        proof.reset_scribbles()
        mock.assert_called_once()

    def test_reset_corrected(self, mocker, proof):
        mock = mocker.patch("panseg.viewer_napari.widgets.proofreading.update_layer")
        proof.reset_corrected()
        mock.assert_not_called()

        proof._state.active = True
        mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            segmentation=mocker.DEFAULT,
            scale=mocker.DEFAULT,
        )
        proof.reset_corrected()
        mock.assert_called_once()

    def test_bboxes(self, mocker, proof):
        mock = mocker.patch.object(proof, "reset_bboxes")
        with pytest.raises(AssertionError):
            proof.bboxes
        mock.assert_called_once()

    def test_reset_bboxes(self, mocker, proof):
        mock = mocker.patch("panseg.viewer_napari.widgets.proofreading.get_bboxes")
        with pytest.raises(ValueError):
            proof.reset_bboxes()
        mock.assert_not_called()

        proof._state.active = True
        mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler.segmentation",
            new_callable=mocker.PropertyMock,
        )
        proof.reset_bboxes()
        mock.assert_called_once()

    def test_reset(self, proof):
        proof.reset()

    def test_setup(self, proof, mocker, napari_segmentation):
        mock_reset = mocker.patch.object(proof, "reset")
        mock_reset_bboxes = mocker.patch.object(proof, "reset_bboxes")
        mock_reset_corrected = mocker.patch.object(proof, "reset_corrected")
        mock_reset_scribbles = mocker.patch.object(proof, "reset_scribbles")
        proof.setup(PanSegImage.from_napari_layer(napari_segmentation))
        mock_reset.assert_called_once()
        mock_reset_bboxes.assert_called_once()
        mock_reset_corrected.assert_called_once()
        mock_reset_scribbles.assert_called_once()

    def test_capture_state(self, proof, mocker):
        mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            segmentation=mocker.DEFAULT,
            corrected_cells=mocker.DEFAULT,
            corrected_cells_mask=mocker.DEFAULT,
            bboxes=mocker.DEFAULT,
        )
        proof._capture_state()

    def test_save_to_history(self, proof, mocker):
        mock = mocker.patch.object(proof, "_capture_state")
        mock.return_value = mocker.sentinel
        proof.save_to_history()
        assert proof._state.history_undo[-1] == mocker.sentinel

    def test_restore_state(self, proof, mocker):
        mock_update_layer = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.update_layer",
        )
        mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            reset_scribbles=mocker.DEFAULT,
            seg_layer_name=mocker.DEFAULT,
            scale=mocker.DEFAULT,
        )
        proof._restore_state(mocker.sentinel)

        assert mock_update_layer.call_count == 2

    def test__perform_undo_redo(self, proof, mocker):
        mock = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            _capture_state=mocker.DEFAULT,
            _restore_state=mocker.DEFAULT,
        )
        proof._perform_undo_redo(deque(), deque(), "smth")
        [m.assert_not_called() for m in mock.values()]

        mock["_capture_state"].return_value = "current"
        pop = deque(("history",))
        append = deque()
        proof._perform_undo_redo(pop, append, "smth")
        mock["_restore_state"].assert_called_with("history")
        assert append == deque(("current",))

    def test_undo(self, proof, mocker):
        mock = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler._perform_undo_redo"
        )
        proof.undo()
        mock.assert_called_once()

    def test_redo(self, proof, mocker):
        mock = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler._perform_undo_redo"
        )
        proof.redo()
        mock.assert_called_once()

    def test_save_state_to_disk_suffix(self, proof, mocker):
        mock = mocker.patch("panseg.viewer_napari.widgets.proofreading.log")
        proof.save_state_to_disk(Path("not_h5.file"), raw=None, pmap=None)
        mock.assert_called_once()

    def test_save_state_to_disk_load(
        self,
        proof,
        mocker,
        tmp_path,
        napari_raw,
        napari_prediction,
        napari_segmentation,
        make_napari_viewer_proxy,
    ):
        h5_path = tmp_path / "valid.h5"
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_segmentation)
        proof._state.current_seg_layer_name = "test_segmentation_3D"

        mocks = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            # corrected_cells_mask=napari_segmentation.data,
            corrected_cells_mask=[4],
            corrected_cells=set((1, 2, 3)),
            scale=mocker.DEFAULT,
        )
        mock_update_layer = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.update_layer",
        )

        proof.save_state_to_disk(
            filepath=h5_path, raw=napari_raw, pmap=napari_prediction
        )

        with h5py.File(h5_path, "r") as f:
            assert all([k in f.keys() for k in ("label", "mask", "pmap", "raw")])

        proof.load_state_from_disk(h5_path)

        mock_update_layer.assert_called_with(
            [4],
            proofreading.CORRECTED_CELLS_LAYER_NAME,
            scale=mocks["scale"],
            colormap=proofreading.correct_cells_cmap,
            opacity=1,
        )

    def test_toggle_corrected_cell(self, mocker, proof):
        mock = mocker.patch.object(proof._state, "corrected_cells")
        proof._toggle_corrected_cell(0)
        mock.add.assert_called_with(0)

        mock.__contains__ = lambda a, b: True

        proof._toggle_corrected_cell(0)
        mock.remove.assert_called_with(0)

    def test_update_masks(self, proof, mocker):
        mocks = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading",
            update_corrected_cells_mask_layer=mocker.DEFAULT,
            get_layer_data=mocker.DEFAULT,
        )
        mocks_handler = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            scale=mocker.DEFAULT,
            segmentation=mocker.DEFAULT,
        )
        proof._update_masks(0)
        mocks["update_corrected_cells_mask_layer"].assert_called_once()

    def test_toggle_corrected_cell_(self, mocker, proof):
        mocks = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            _toggle_corrected_cell=mocker.DEFAULT,
            _update_masks=mocker.DEFAULT,
        )
        proof.toggle_corrected_cell(mocker.sentinel)
        [m.assert_called_with(mocker.sentinel) for m in mocks.values()]

    def test_update_corrected_cells_mask_slice_to_viewer(self, mocker, proof):
        mocks = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading",
            update_region=mocker.DEFAULT,
            preserve_labels=mocker.DEFAULT,
        )
        mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler.scale",
        )

        proof.update_corrected_cells_mask_slice_to_viewer(
            mocker.sentinel, mocker.sentinel
        )
        [m.assert_called_once() for m in mocks.values()]

    def test_update_after_proofreading(self, mocker, proof):
        mocks = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            create=True,
            _state=mocker.DEFAULT,
            seg_layer_name=mocker.DEFAULT,
            scale=mocker.DEFAULT,
            bboxes=mocker.DEFAULT,
        )
        mock_update = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.update_region",
        )

        proof.update_after_proofreading(
            mocker.sentinel, mocker.sentinel, mocker.sentinel
        )
        mocks["bboxes"].update.assert_called_once()
        mock_update.assert_called_once()


def test_widget_clean_scribble(mocker, make_napari_viewer_proxy, napari_raw):
    mock = mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.segmentation_handler.reset_scribbles",
    )
    mock_log = mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.log",
    )
    viewer = make_napari_viewer_proxy()

    proofreading.widget_clean_scribble(viewer)
    mock_log.assert_called_with(
        "Proofreading widget not initialized. Run the proofreading widget tool once first",
        thread="Clean scribble",
    )
    mock_log.reset_mock()

    proofreading.segmentation_handler._state.active = True
    proofreading.widget_clean_scribble(viewer)
    mock_log.assert_called_with(
        "Scribble Layer not defined. Run the proofreading widget tool once first",
        thread="Clean scribble",
    )
    mock_log.reset_mock()

    viewer.add_layer(napari_raw)
    viewer.layers[0].name = "Scribbles"
    proofreading.widget_clean_scribble(viewer)
    mock.assert_called_once()


def test_widget_add_label_to_corrected(
    mocker, make_napari_viewer_proxy, napari_segmentation
):
    viewer = make_napari_viewer_proxy()
    with pytest.raises(ValueError):
        proofreading.widget_add_label_to_corrected(viewer, (0, 0, 0))

    mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.segmentation_handler._scale",
        (1, 1, 1),
    )
    mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.segmentation_handler._state.current_seg_layer_name",
        proofreading.CORRECTED_CELLS_LAYER_NAME,
    )
    mock = mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.segmentation_handler.toggle_corrected_cell",
    )

    napari_segmentation.name = proofreading.CORRECTED_CELLS_LAYER_NAME
    viewer.add_layer(napari_segmentation)
    proofreading.widget_add_label_to_corrected(viewer, (0, 0, 0))
    mock.assert_called_once()


def test_initialize_from_layer_wrong_layer(mocker, napari_segmentation):
    napari_segmentation.name = proofreading.SCRIBBLES_LAYER_NAME
    mock = mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.log",
    )
    proofreading.initialize_from_layer(napari_segmentation, False)
    mock.assert_called_with(
        "Scribble or corrected cells layer is not intended to be proofread, choose a segmentation",
        thread="Proofreading tool",
        level="error",
    )


def test_initialize_from_layer_not_sure(mocker, napari_segmentation):
    mock = mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.log",
    )
    proofreading.segmentation_handler._state.active = True
    proofreading.initialize_from_layer(napari_segmentation, False)
    mock.assert_called_with(
        "Proofreading is already initialized. Are you sure you want to reset everything?",
        thread="Proofreading tool",
        level="warning",
    )


def test_initialize_from_layer(mocker, napari_segmentation, make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_segmentation)
    mocks = mocker.patch.multiple(
        "panseg.viewer_napari.widgets.proofreading",
        widget_proofreading_initialisation=mocker.DEFAULT,
    )
    proofreading.segmentation_handler._state.active = True
    proofreading.initialize_from_layer(napari_segmentation, True)
    mocks["widget_proofreading_initialisation"].are_you_sure.hide.assert_called_once()


def test_initialize_from_file_not_sure(mocker):
    mock = mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.log",
    )
    proofreading.segmentation_handler._state.active = True
    proofreading.initialize_from_file(Path(), False)
    mock.assert_called_with(
        "Proofreading is already initialized. Are you sure you want to reset everything?",
        thread="Proofreading tool",
        level="warning",
    )


def test_initialize_from_file(mocker):
    mocks = mocker.patch.multiple(
        "panseg.viewer_napari.widgets.proofreading",
        segmentation_handler=mocker.DEFAULT,
        setup_proofreading_widget=mocker.DEFAULT,
    )
    proofreading.segmentation_handler._state.active = True
    proofreading.initialize_from_file(mocker.sentinel, True)
    mocks["segmentation_handler"].load_state_from_disk.assert_called_with(
        mocker.sentinel
    )
    mocks["setup_proofreading_widget"].assert_called_once()


def test_widget_proofreading_initialisation(mocker, napari_segmentation):
    mocks = mocker.patch.multiple(
        "panseg.viewer_napari.widgets.proofreading",
        log=mocker.DEFAULT,
        initialize_from_layer=mocker.DEFAULT,
        initialize_from_file=mocker.DEFAULT,
    )
    proofreading.widget_proofreading_initialisation("New", None, None)
    mocks["log"].assert_called_with(
        "No segmentation layer selected",
        thread="Proofreading tool",
        level="error",
    )
    proofreading.widget_proofreading_initialisation("Load from file", None, None)
    mocks["log"].assert_called_with(
        "No state file selected", thread="Proofreading tool", level="error"
    )

    mocks["initialize_from_layer"].assert_not_called()
    mocks["initialize_from_file"].assert_not_called()

    proofreading.widget_proofreading_initialisation(
        "New", napari_segmentation, None, mocker.sentinel
    )
    mocks["initialize_from_layer"].assert_called_with(
        napari_segmentation, are_you_sure=mocker.sentinel
    )
    mocks["initialize_from_file"].assert_not_called()
    mocks["initialize_from_layer"].reset_mock()

    proofreading.widget_proofreading_initialisation(
        "Load from file", None, Path(), mocker.sentinel
    )
    mocks["initialize_from_file"].assert_called_with(
        Path(), are_you_sure=mocker.sentinel
    )
    mocks["initialize_from_layer"].assert_not_called()


def test_on_mode_change(mocker):
    mock = mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.widget_proofreading_initialisation"
    )
    proofreading._on_mode_changed("New")
    mock.segmentation.show.assert_called_once()
    mock.filepath.hide.assert_called_once()
    proofreading._on_mode_changed("Load from file")
    mock.segmentation.hide.assert_called_once()
    mock.filepath.show.assert_called_once()


def test_widget_split_and_merge_from_scribbles_log(mocker, napari_raw):
    mocks = mocker.patch.multiple(
        "panseg.viewer_napari.widgets.proofreading",
        log=mocker.DEFAULT,
        segmentation_handler=mocker.DEFAULT,
    )
    mocks["segmentation_handler"].active = False
    proofreading.widget_split_and_merge_from_scribbles(mocker.sentinel, napari_raw)
    mocks["log"].assert_called_with(
        "Proofreading is not initialized. Run the initialization widget first.",
        thread="Proofreading tool",
    )
    mocks["segmentation_handler"].active = True
    proofreading.widget_split_and_merge_from_scribbles(mocker.sentinel, None)
    mocks["log"].assert_called_with(
        "Please select a boundary image first!",
        thread="Proofreading tool",
    )


@pytest.mark.parametrize("run_id", range(10))
def test_widget_split_and_merge_from_scribbles(mocker, napari_raw, run_id, qtbot):
    WorkerBase.await_workers(500)
    mock_split_merge = mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.split_merge_from_seeds",
    )
    mock_scribble = mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler.scribbles",
        new_callable=mocker.PropertyMock,
    )
    scribble = mocker.Mock()
    mock_scribble.return_value = scribble
    scribble.sum.return_value = 0

    handler = mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.segmentation_handler",
        new=proofreading.ProofreadingHandler(),
    )
    handler._state.active = True

    worker = proofreading.widget_split_and_merge_from_scribbles(
        mocker.sentinel, napari_raw
    )
    worker.await_workers()
    mock_split_merge.assert_not_called()

    mock_save = mocker.patch.object(handler, "save_to_history")
    mocker.patch.multiple(
        "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
        segmentation=mocker.DEFAULT,
        seg_layer_name=mocker.DEFAULT,
        scale=mocker.DEFAULT,
        bboxes=mocker.DEFAULT,
        max_label=mocker.DEFAULT,
        corrected_cells=mocker.DEFAULT,
    )
    mocker.patch(
        "panseg.viewer_napari.widgets.proofreading.update_region",
    )
    scribble.sum.return_value = 5
    mock_split_merge.return_value = [mocker.sentinel] * 3

    worker = proofreading.widget_split_and_merge_from_scribbles(
        mocker.sentinel, napari_raw
    )
    worker.await_workers()
    mock_save.assert_called_once()
    mock_split_merge.assert_called_once()


def test_locking():
    handler = proofreading.ProofreadingHandler()

    with handler.lock_manager():
        assert handler.is_locked()

    assert not handler.is_locked()

    with handler.lock_manager():
        with pytest.raises(TimeoutError):
            with handler.lock_manager(timeout=0):
                pass

    assert not handler.is_locked()


def test_corrected_cells():
    handler = proofreading.ProofreadingHandler()
    handler.corrected_cells


def test_corrected_cells_mask(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_labels(
        np.zeros((5, 5, 5), dtype=int),
        name=proofreading.CORRECTED_CELLS_LAYER_NAME,
        scale=[1, 1, 1],
    )
    handler = proofreading.ProofreadingHandler()
    handler.corrected_cells_mask


def test_max_label(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_labels(
        np.zeros((5, 5, 5), dtype=int),
        name="test_seg",
        scale=[1, 1, 1],
    )
    handler = proofreading.ProofreadingHandler()
    handler._state.current_seg_layer_name = "test_seg"

    assert handler.max_label == 0


@pytest.mark.parametrize("i", range(100))
def test_locking_threaded_simple(i):
    WorkerBase.await_workers(500)
    handler = proofreading.ProofreadingHandler()

    @thread_worker
    def threaded(freeze: bool):
        with handler.lock_manager(10 if not freeze else 300):
            if freeze:
                t0 = time()
                while time() - t0 < 10:
                    sleep(0.001)
                    yield
            else:
                return
        sleep(0.001)

    worker = threaded(True)
    worker.start()
    sleep(0.001)
    assert handler.is_locked()
    worker.await_workers()


@pytest.mark.parametrize("i", range(100))
def test_locking_threaded_timeout(qtbot, i):
    WorkerBase.await_workers(500)

    handler = proofreading.ProofreadingHandler()

    @thread_worker
    def threaded(freeze: bool):
        with handler.lock_manager(10 if not freeze else 10000):
            if freeze:
                t0 = time()
                while time() - t0 < 10000:
                    qtbot.wait(10)
                    yield
            else:
                return
        sleep(0.001)

    worker1 = threaded(True)
    worker1.start()

    while not worker1.is_running:
        sleep(0.0001)

    assert handler.is_locked()
    assert worker1.is_running

    worker2 = threaded(False)
    with qtbot.capture_exceptions() as exceptions:
        with qtbot.waitSignal(worker2.errored, raising=True):
            worker2.start()
    assert len(exceptions) == 1
    assert exceptions[0][0] is TimeoutError
    worker1.await_workers()
    worker2.await_workers()


def test_widget_filter_segmentation_log(mocker):
    mocks = mocker.patch.multiple(
        "panseg.viewer_napari.widgets.proofreading",
        log=mocker.DEFAULT,
        segmentation_handler=mocker.DEFAULT,
    )
    mocks["segmentation_handler"].active = False
    with pytest.raises(ValueError):
        proofreading.widget_filter_segmentation()
    mocks["log"].assert_called_with(
        "Proofreading widget not initialized. Run the proofreading widget tool once first",
        thread="Export correct labels",
        level="error",
    )


def test_widget_filter_segmentation(mocker, make_napari_viewer_proxy):
    mocks = mocker.patch.multiple(
        "panseg.viewer_napari.widgets.proofreading",
        segmentation_handler=mocker.DEFAULT,
        split_merge_from_seeds=mocker.DEFAULT,
        ImageProperties=mocker.DEFAULT,
        PanSegImage=mocker.DEFAULT,
        napari=mocker.DEFAULT,
    )
    mocker.sentinel._add_layer_from_data = mocker.Mock()
    mocks["napari"].current_viewer.return_value = mocker.sentinel

    proofreading.widget_filter_segmentation()
    sleep(0.1)
    mocks["ImageProperties"].assert_called_once()
    mocks["PanSegImage"].assert_called_once()


def test_widget_undo(mocker):
    mocks = mocker.patch.multiple(
        "panseg.viewer_napari.widgets.proofreading",
        segmentation_handler=mocker.DEFAULT,
        log=mocker.DEFAULT,
    )
    mocks["segmentation_handler"].active = False
    proofreading.widget_undo()
    mocks["log"].assert_called_with(
        "Proofreading widget not initialized. Nothing to undo.", thread="Undo"
    )
    mocks["segmentation_handler"].active = True
    proofreading.widget_undo()
    mocks["segmentation_handler"].undo.assert_called_once()


def test_widget_redo(mocker):
    mocks = mocker.patch.multiple(
        "panseg.viewer_napari.widgets.proofreading",
        segmentation_handler=mocker.DEFAULT,
        log=mocker.DEFAULT,
    )
    mocks["segmentation_handler"].active = False
    proofreading.widget_redo()
    mocks["log"].assert_called_with(
        "Proofreading widget not initialized. Nothing to redo.", thread="Redo"
    )
    mocks["segmentation_handler"].active = True
    proofreading.widget_redo()
    mocks["segmentation_handler"].redo.assert_called_once()


def test_widget_save_state(mocker):
    mocks = mocker.patch.multiple(
        "panseg.viewer_napari.widgets.proofreading",
        segmentation_handler=mocker.DEFAULT,
    )
    proofreading.widget_save_state(mocker.sentinel)
    mocks["segmentation_handler"].save_state_to_disk.assert_called_with(
        mocker.sentinel,
        raw=None,
        pmap=None,
    )


def test_setup_keybindings(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    proofreading.setup_proofreading_keybindings()


def test_setup_proofreading_widget():
    proofreading.setup_proofreading_widget()


def test_update_layer_selection(
    mocker, napari_raw, napari_segmentation, make_napari_viewer_proxy
):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)

    sentinel = mocker.sentinel
    sentinel.value = napari_raw
    sentinel.type = "inserted"

    assert proofreading.widget_proofreading_initialisation.segmentation.value is None
    assert proofreading.widget_save_state.raw.value is None
    assert proofreading.widget_save_state.pmap.value is None
    assert proofreading.widget_split_and_merge_from_scribbles.image.value is None

    proofreading.update_layer_selection(sentinel)

    assert proofreading.widget_proofreading_initialisation.segmentation.value is None
    assert proofreading.widget_save_state.raw.value is None
    assert proofreading.widget_save_state.pmap.value is None
    assert proofreading.widget_split_and_merge_from_scribbles.image.value is None

    viewer.add_layer(napari_segmentation)
    sentinel.value = napari_segmentation
    proofreading.update_layer_selection(sentinel)

    assert (
        proofreading.widget_proofreading_initialisation.segmentation.value
        is napari_segmentation
    )
    assert proofreading.widget_save_state.raw.value is None
    assert proofreading.widget_save_state.pmap.value is None
    assert proofreading.widget_split_and_merge_from_scribbles.image.value is None
