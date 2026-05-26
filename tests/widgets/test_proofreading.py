from collections import deque
from pathlib import Path

import h5py
import numpy as np
import pytest
from magicgui.widgets import Container
from napari.qt import get_qapp

from panseg.core.image import PanSegImage
from panseg.viewer_napari.widgets.proofreading import (
    CORRECTED_CELLS_LAYER_NAME,
    SCRIBBLES_LAYER_NAME,
    Proofreading_Tab,
    ProofreadingHandler,
    correct_cells_cmap,
)


@pytest.fixture
def proof():
    return ProofreadingHandler()


@pytest.fixture(scope="function")
def tab() -> Proofreading_Tab:
    return Proofreading_Tab()


class TestProofreadingHandler:
    def test_init(self, proof):
        assert not proof.active

    def test_get_layer_data_empty(self, make_napari_viewer_proxy, proof):
        viewer = make_napari_viewer_proxy()
        with pytest.raises(ValueError):
            proof.get_layer_data("test")

    def test_get_layer_data(self, make_napari_viewer_proxy, napari_raw, proof):
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_raw)
        assert np.all(proof.get_layer_data("test_image_3D") == napari_raw.data)

    def test_get_layer_data_no_viewer(self, proof):
        with pytest.raises(RuntimeError):
            proof.get_layer_data("some_layer")

    def test_update_layer(self, make_napari_viewer_proxy, napari_segmentation, proof):
        with pytest.raises(RuntimeError):
            proof.update_layer(
                napari_segmentation.data,
                layer_name="test",
                scale=napari_segmentation.scale,
            )

        viewer = make_napari_viewer_proxy()
        viewer.add_labels(np.zeros((5, 5, 5), dtype=int), name="test", scale=[1, 1, 1])

        proof.update_layer(
            napari_segmentation.data, layer_name="test", scale=napari_segmentation.scale
        )
        np.testing.assert_array_equal(viewer.layers[0].data, napari_segmentation.data)
        np.testing.assert_array_equal(viewer.layers[0].scale, napari_segmentation.scale)
        assert viewer.layers[0].name == "test"

        proof.update_layer(
            napari_segmentation.data,
            layer_name="new_layer",
            scale=napari_segmentation.scale,
        )

        np.testing.assert_array_equal(viewer.layers[1].data, napari_segmentation.data)
        np.testing.assert_array_equal(viewer.layers[1].scale, napari_segmentation.scale)
        assert viewer.layers[1].name == "new_layer"

    def test_reset_scribbles(self, mocker, proof):
        mock = mocker.patch.object(proof, "update_layer")
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
        mock = mocker.patch.object(proof, "update_layer")
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
        mock = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            reset_scribbles=mocker.DEFAULT,
            seg_layer_name=mocker.DEFAULT,
            scale=mocker.DEFAULT,
            update_layer=mocker.DEFAULT,
        )
        proof._restore_state(mocker.sentinel)

        assert mock["update_layer"].call_count == 2

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
        mock = mocker.patch.object(proof, "_perform_undo_redo")
        proof.undo()
        mock.assert_called_once()

    def test_redo(self, proof, mocker):
        mock = mocker.patch.object(proof, "_perform_undo_redo")
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
        mock_update_layer = mocker.patch.object(proof, "update_layer")

        proof.save_state_to_disk(
            filepath=h5_path, raw=napari_raw, pmap=napari_prediction
        )

        with h5py.File(h5_path, "r") as f:
            assert all([k in f.keys() for k in ("label", "mask", "pmap", "raw")])

        proof.load_state_from_disk(h5_path)

        mock_update_layer.assert_called_with(
            [4],
            CORRECTED_CELLS_LAYER_NAME,
            scale=mocks["scale"],
            colormap=correct_cells_cmap,
            opacity=1,
        )

    def test_load_state_from_disk_no_file(self, proof):
        with pytest.raises(ValueError):
            proof.load_state_from_disk(Path("wrong_path"))

    def test_toggle_corrected_cell(self, mocker, proof):
        mock = mocker.patch.object(proof._state, "corrected_cells")
        proof._toggle_corrected_cell(0)
        mock.add.assert_called_with(0)

        mock.__contains__ = lambda a, b: True

        proof._toggle_corrected_cell(0)
        mock.remove.assert_called_with(0)

    def test_update_masks(self, proof, mocker):
        mocks = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            scale=mocker.DEFAULT,
            segmentation=mocker.DEFAULT,
            get_layer_data=mocker.DEFAULT,
            update_layer=mocker.DEFAULT,
        )
        proof._update_masks(0)
        mocks["update_layer"].assert_called_once()

    def test_toggle_corrected_cell_(self, mocker, proof):
        mocks = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            _toggle_corrected_cell=mocker.DEFAULT,
            _update_masks=mocker.DEFAULT,
        )
        proof.toggle_corrected_cell(mocker.sentinel)
        [m.assert_called_with(mocker.sentinel) for m in mocks.values()]

    def test_update_after_proofreading(
        self, mocker, proof, make_napari_viewer_proxy, napari_segmentation
    ):
        mocks = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            create=True,
            _state=mocker.DEFAULT,
            seg_layer_name="test_segmentation_3D",
            scale=mocker.DEFAULT,
            bboxes=mocker.DEFAULT,
        )
        viewer = make_napari_viewer_proxy()

        with pytest.raises(ValueError):
            proof.update_after_proofreading(
                mocker.sentinel, mocker.sentinel, mocker.sentinel
            )

        viewer.add_layer(napari_segmentation)
        mocks["bboxes"].update.assert_called_once()

    def test_corrected_cells(self, proof):
        proof.corrected_cells

    def test_corrected_cells_mask(self, proof, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        viewer.add_labels(
            np.zeros((5, 5, 5), dtype=int),
            name=CORRECTED_CELLS_LAYER_NAME,
            scale=[1, 1, 1],
        )
        proof.corrected_cells_mask

    def test_max_label(self, proof, make_napari_viewer_proxy):
        viewer = make_napari_viewer_proxy()
        viewer.add_labels(
            np.zeros((5, 5, 5), dtype=int),
            name="test_seg",
            scale=[1, 1, 1],
        )
        proof._state.current_seg_layer_name = "test_seg"

        assert proof.max_label == 0


class TestProofreadingTab:
    def test_init(self, tab):
        assert not tab.busy
        assert len(tab.container) == 11

    def test_hide_all(self, tab):
        app = get_qapp()
        tab.container.show()
        tab._hide_all_widgets()
        assert all([not w.visible for w in tab.container[3:]])
        tab.container.hide()
        app.quit()

    def test_show_all(self, tab):
        app = get_qapp()
        tab.container.show()
        tab._show_all_widgets()
        assert all([w.visible for w in tab.container[3:]])
        tab.container.hide()
        app.quit()

    def test_get_container(self, tab):
        assert isinstance(tab.get_container(), Container)

    def test_widget_init_from_layer(
        self, tab, make_napari_viewer_proxy, napari_segmentation, mocker
    ):
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_segmentation)
        mock = mocker.patch.object(tab, "_initialize_from_layer")
        mock_keys = mocker.patch.object(tab, "_setup_proofreading_keybindings")
        tab.widget_proofreading_initialisation(
            mode="New",
            segmentation=napari_segmentation,
            filepath=None,
            are_you_sure=False,
        )
        mock.assert_called_once()
        mock_keys.assert_called_once()

    def test_widget_init_from_file(self, tab, mocker):
        mock = mocker.patch.object(tab, "_initialize_from_file")
        mock_keys = mocker.patch.object(tab, "_setup_proofreading_keybindings")
        tab.widget_proofreading_initialisation(
            mode="Load from file",
            segmentation=None,
            filepath="some/test/path",
            are_you_sure=False,
        )
        mock.assert_called_once()
        mock_keys.assert_called_once()

    def test_initialize_from_file(self, tab, mocker):
        mocks = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            load_state_from_disk=mocker.DEFAULT,
        )
        tab.handler._state.active = True
        tab._initialize_from_file(file=mocker.sentinel, are_you_sure=True)
        mocks["load_state_from_disk"].assert_called_with(mocker.sentinel)

    def test_initialize_from_file_not_sure(self, tab, mocker):
        mock = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.log",
        )
        tab.handler._state.active = True
        tab._initialize_from_file(Path(), False)
        mock.assert_called_with(
            "Proofreading is already initialized. Are you sure you want to reset everything?",
            thread="Proofreading tool",
            level="warning",
        )

    def test_initialize_from_layer_not_sure(self, tab, mocker, napari_segmentation):
        mock = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.log",
        )
        tab.handler._state.active = True
        tab._initialize_from_layer(napari_segmentation, False)
        mock.assert_called_with(
            "Proofreading is already initialized. Are you sure you want to reset everything?",
            thread="Proofreading tool",
            level="warning",
        )

    def test_initialize_from_layer_wrong_layer(self, tab, mocker, napari_segmentation):
        napari_segmentation.name = SCRIBBLES_LAYER_NAME
        mock = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.log",
        )
        tab._initialize_from_layer(napari_segmentation, False)
        mock.assert_called_with(
            "Scribble or corrected cells layer is not intended to be proofread, choose a segmentation",
            thread="Proofreading tool",
            level="error",
        )

    def test_widget_proofreading_initialisation_bad_mode(self, tab):
        with pytest.raises(ValueError):
            tab.widget_proofreading_initialisation(mode="bad")

    def test_widget_proofreading_initialisation_wrong(
        self, tab, mocker, napari_segmentation
    ):
        mock_log = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.log",
        )
        mocks = mocker.patch.multiple(
            tab,
            _initialize_from_layer=mocker.DEFAULT,
            _initialize_from_file=mocker.DEFAULT,
        )
        tab.widget_proofreading_initialisation(
            mode="New", segmentation=None, filepath=None
        )
        mock_log.assert_called_with(
            "No segmentation layer selected",
            thread="Proofreading tool",
            level="error",
        )
        tab.widget_proofreading_initialisation(
            mode="Load from file", segmentation=None, filepath=None
        )
        mock_log.assert_called_with(
            "No state file selected", thread="Proofreading tool", level="error"
        )

        mocks["_initialize_from_layer"].assert_not_called()
        mocks["_initialize_from_file"].assert_not_called()

        tab.widget_proofreading_initialisation(
            mode="New",
            segmentation=napari_segmentation,
            filepath=None,
            are_you_sure=mocker.sentinel,
        )
        mocks["_initialize_from_layer"].assert_called_with(
            napari_segmentation, are_you_sure=mocker.sentinel
        )
        mocks["_initialize_from_file"].assert_not_called()
        mocks["_initialize_from_layer"].reset_mock()

        tab.widget_proofreading_initialisation(
            mode="Load from file",
            segmentation=None,
            filepath=Path(),
            are_you_sure=mocker.sentinel,
        )
        mocks["_initialize_from_file"].assert_called_with(
            Path(), are_you_sure=mocker.sentinel
        )
        mocks["_initialize_from_layer"].assert_not_called()

    def test_widget_clean_scribble(
        self, tab, mocker, make_napari_viewer_proxy, napari_raw
    ):
        mock = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler.reset_scribbles",
        )
        mock_log = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.log",
        )
        viewer = make_napari_viewer_proxy()

        tab.widget_clean_scribble(viewer=viewer)
        mock_log.assert_called_with(
            "Proofreading widget not initialized. Run the proofreading widget tool once first",
            thread="Clean scribble",
        )
        mock_log.reset_mock()

        tab.handler._state.active = True
        tab.widget_clean_scribble(viewer=viewer)
        mock_log.assert_called_with(
            "Scribble Layer not defined. Run the proofreading widget tool once first",
            thread="Clean scribble",
        )
        mock_log.reset_mock()

        viewer.add_layer(napari_raw)
        viewer.layers[0].name = "Scribbles"
        tab.widget_clean_scribble(viewer=viewer)
        mock.assert_called_once()

    def test_widget_split_and_merge_from_scribbles_log(self, tab, mocker, napari_raw):
        mock_log = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.log",
        )
        tab.handler._state.active = False
        tab.widget_split_and_merge_from_scribbles(
            viewer=mocker.sentinel, image=napari_raw
        )
        mock_log.assert_called_with(
            "Proofreading is not initialized. Run the initialization widget first.",
            thread="Proofreading tool",
        )
        tab.handler._state.active = True
        tab.widget_split_and_merge_from_scribbles(viewer=mocker.sentinel, image=None)
        mock_log.assert_called_with(
            "Please select a boundary image first!",
            thread="Proofreading tool",
        )

    @pytest.mark.parametrize("rep", range(5))
    def test_widget_split_and_merge_from_scribbles(
        self, tab, mocker, napari_raw, rep, qtbot
    ):
        assert not tab.busy
        app = get_qapp()
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

        tab.handler._state.active = True
        worker = tab.widget_split_and_merge_from_scribbles(
            viewer=mocker.sentinel, image=napari_raw
        )
        assert worker

        qtbot.waitUntil(lambda: not tab.busy)
        assert not worker.is_running
        assert not tab.busy

        mock_split_merge.assert_not_called()

        mocks = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            segmentation=mocker.DEFAULT,
            seg_layer_name=mocker.DEFAULT,
            scale=mocker.DEFAULT,
            bboxes=mocker.DEFAULT,
            max_label=mocker.DEFAULT,
            corrected_cells=mocker.DEFAULT,
            save_to_history=mocker.DEFAULT,
            update_after_proofreading=mocker.DEFAULT,
        )
        scribble.sum.return_value = 5
        mock_split_merge.return_value = [mocker.sentinel] * 3

        worker = tab.widget_split_and_merge_from_scribbles(
            viewer=mocker.sentinel, image=napari_raw
        )
        qtbot.waitUntil(lambda: not tab.busy)
        assert not worker.is_running

        mocks["update_after_proofreading"].assert_called_once()

        tab.busy = True
        worker = tab.widget_split_and_merge_from_scribbles(
            viewer=mocker.sentinel, image=napari_raw
        )
        assert worker is None
        app.quit()

    def test_widget_add_label_to_corrected(
        self, tab, mocker, make_napari_viewer_proxy, napari_segmentation
    ):
        viewer = make_napari_viewer_proxy()
        with pytest.raises(ValueError):
            tab._widget_add_label_to_corrected(viewer, (0, 0, 0))
        mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler.segmentation",
            new_callable=mocker.PropertyMock,
        )
        mock_scale = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler.scale",
            new_callable=lambda: [1, 1, 1],
        )
        mock_state = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler._state",
            create=True,
        )
        mock_toggle = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler.toggle_corrected_cell",
        )

        mock_state.current_seg_layer_name = CORRECTED_CELLS_LAYER_NAME

        napari_segmentation.name = CORRECTED_CELLS_LAYER_NAME
        viewer.add_layer(napari_segmentation)
        tab._widget_add_label_to_corrected(viewer, (0, 0, 0))
        mock_toggle.assert_called_once()

    def test_on_mode_change(self, tab, mocker):
        mock = mocker.patch.object(tab, "widget_proofreading_initialisation")
        tab._on_mode_changed("New")
        mock.segmentation.show.assert_called_once()
        mock.filepath.hide.assert_called_once()
        tab._on_mode_changed("Load from file")
        mock.segmentation.hide.assert_called_once()
        mock.filepath.show.assert_called_once()

    def test_widget_filter_segmentation_log(self, tab, mocker):
        mock_log = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.log",
        )
        tab.handler._state.active = False
        with pytest.raises(ValueError):
            tab.widget_filter_segmentation()
        mock_log.assert_called_with(
            "Proofreading widget not initialized. Run the proofreading widget tool once first",
            thread="Export correct labels",
            level="error",
        )

    def test_widget_filter_segmentation(self, tab, mocker, qtbot, napari_segmentation):
        pan_seg = PanSegImage.from_napari_layer(napari_segmentation)
        mock_get_layer = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler.get_layer_data",
        )
        mock_get_layer.return_value = napari_segmentation.data
        mocks = mocker.patch.multiple(
            "panseg.viewer_napari.widgets.proofreading",
            ImageProperties=mocker.DEFAULT,
            PanSegImage=mocker.DEFAULT,
            napari=mocker.DEFAULT,
            log=mocker.DEFAULT,
        )
        # mocker.sentinel._add_layer_from_data = mocker.Mock()
        # mocks["napari"].current_viewer.return_value = mocker.sentinel

        tab.handler._state.active = True
        tab.handler._state.current_seg_layer_name = "test"
        tab.handler._state.seg_properties = pan_seg.properties
        worker = tab.widget_filter_segmentation()
        assert worker
        qtbot.waitUntil(lambda: not tab.busy)
        assert not worker._running

        mocks["log"].assert_called_with(
            "Done extracting corrected labels",
            thread="filter_segmentation",
            level="INFO",
        )
        mocks["PanSegImage"].assert_called_once()
        mocks["ImageProperties"].assert_called_once()

    def test_widget_filter_segmentation_busy(self, tab, mocker):
        tab.handler._state.active = True
        tab.busy = True

        mock_log = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.log",
        )
        tab.widget_filter_segmentation()
        mock_log.assert_called_with(
            "Busy! Try again later!", thread="filter_segmentation", level="Warning"
        )

    def test_widget_undo(self, tab, mocker):
        mock_log = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.log",
        )
        mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler.undo",
        )

        tab.handler._state.active = False
        tab.widget_undo()
        mock_log.assert_called_with(
            "Proofreading widget not initialized. Nothing to undo.", thread="Undo"
        )
        tab.handler._state.active = True
        tab.widget_undo()
        tab.handler.undo.assert_called_once()

    def test_widget_redo(self, tab, mocker):
        mock_log = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.log",
        )
        mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.ProofreadingHandler.redo",
        )
        tab.handler._state.active = False
        tab.widget_redo()
        mock_log.assert_called_with(
            "Proofreading widget not initialized. Nothing to redo.", thread="Redo"
        )
        tab.handler._state.active = True
        tab.widget_redo()
        tab.handler.redo.assert_called_once()

    def test_widget_save_state(self, tab, mocker):
        mock = mocker.patch.object(tab.handler, "save_state_to_disk")

        tab.widget_save_state(filepath=mocker.sentinel)
        mock.assert_called_with(
            mocker.sentinel,
            raw=None,
            pmap=None,
        )

    def test_setup_keybindings(self, tab, make_napari_viewer_proxy):
        make_napari_viewer_proxy()
        tab._setup_proofreading_keybindings()

    def test_update_layer_selection(
        self, tab, mocker, napari_raw, napari_segmentation, make_napari_viewer_proxy
    ):
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_raw)

        sentinel = mocker.sentinel
        sentinel.value = napari_raw
        sentinel.type = "inserted"

        assert tab.widget_proofreading_initialisation.segmentation.value is None
        assert tab.widget_save_state.raw.value is None
        assert tab.widget_save_state.pmap.value is None
        assert tab.widget_split_and_merge_from_scribbles.image.value is None

        tab.update_layer_selection(sentinel)

        assert tab.widget_proofreading_initialisation.segmentation.value is None
        assert tab.widget_save_state.raw.value is None
        assert tab.widget_save_state.pmap.value is None
        assert tab.widget_split_and_merge_from_scribbles.image.value is None

        viewer.add_layer(napari_segmentation)
        sentinel.value = napari_segmentation
        tab.update_layer_selection(sentinel)

        assert (
            tab.widget_proofreading_initialisation.segmentation.value
            is napari_segmentation
        )
        assert tab.widget_save_state.raw.value is None
        assert tab.widget_save_state.pmap.value is None
        assert tab.widget_split_and_merge_from_scribbles.image.value is None
