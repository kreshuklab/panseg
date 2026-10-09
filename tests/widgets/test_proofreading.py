import contextlib
import threading
import time
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import pytest
from magicgui.widgets import Container
from napari.qt import get_qapp

from panseg.core.image import PanSegImage
from panseg.functionals.proofreading.split_merge_tools import split_merge_from_seeds
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


def _pump_until(stop_func, timeout: float = 10.0) -> None:
    """Polls `stop_func` while processing posted Qt events.

    Unlike `qtbot.waitUntil`, this does not enter a nested `QEventLoop`:
    on the macOS CI runner. A nested event loop in the presence of a live
    viewer stops processing timer events on macos in gh ci, so `waitUntil`
    freezes without ever firing its own timeout.
    `processEvents` still delivers queued cross-thread signals
    (e.g. `worker.returned` -> `on_done`),
    and the timeout is wall-clock based, so it always fires.
    """
    deadline = time.monotonic() + timeout
    while not stop_func():
        if time.monotonic() > deadline:
            raise TimeoutError(f"condition not met within {timeout:.0f}s")
        get_qapp().processEvents()
        time.sleep(0.01)


class TestProofreadingHandler:
    def test_init(self, proof):
        assert not proof.active

    def test_get_layer_data_empty(self, make_napari_viewer_proxy, proof):
        make_napari_viewer_proxy()
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
            _write_segmentation=mocker.DEFAULT,
        )
        proof._restore_state(mocker.sentinel)

        mock["_write_segmentation"].assert_called_once()
        mock["update_layer"].assert_called_once()

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
            corrected_cells={1, 2, 3},
            scale=mocker.DEFAULT,
        )
        mock_update_layer = mocker.patch.object(proof, "update_layer")

        proof.save_state_to_disk(
            filepath=h5_path, raw=napari_raw, pmap=napari_prediction
        )

        with h5py.File(h5_path, "r") as f:
            assert all(k in f for k in ("label", "mask", "pmap", "raw"))

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


def _full_region_slice(shape: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(0, s) for s in shape)


class TestProofreadingHandlerTimeSeries:
    @pytest.fixture
    def bound_handler(
        self, proof, make_napari_viewer_proxy, napari_timeseries_segmentation
    ):
        """A handler bound to timepoint 1 of the deterministic timeseries layer."""
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_timeseries_segmentation)
        proof.setup(
            PanSegImage.from_napari_layer(napari_timeseries_segmentation), timepoint=1
        )
        return proof, viewer, napari_timeseries_segmentation

    def test_setup_binds_slice_world(self, bound_handler):
        proof, viewer, layer = bound_handler

        assert proof.active
        assert proof.is_timeseries
        assert proof.timepoint == 1
        assert proof.n_timepoints == 3
        assert "Scribbles (t=1)" in viewer.layers
        assert "Correct Labels (t=1)" in viewer.layers
        assert "Scribbles" not in viewer.layers
        assert "Correct Labels" not in viewer.layers
        for layer_name in ("Scribbles (t=1)", "Correct Labels (t=1)"):
            canvas = viewer.layers[layer_name]
            assert canvas.data.shape == layer.data.shape[1:]
            np.testing.assert_array_equal(canvas.scale, (1.0, 1.0, 1.0))
            np.testing.assert_array_equal(canvas.data, 0)

    def test_scribbles_and_corrected_mask_read_dynamic_layers(self, bound_handler):
        proof, viewer, _ = bound_handler
        viewer.layers["Scribbles (t=1)"].data[0, 0, 0] = 7
        viewer.layers["Correct Labels (t=1)"].data[0, 0, 0] = 7

        np.testing.assert_array_equal(
            proof.scribbles, viewer.layers["Scribbles (t=1)"].data
        )
        np.testing.assert_array_equal(
            proof.corrected_cells_mask, viewer.layers["Correct Labels (t=1)"].data
        )

    def test_segmentation_property_reads_slice(self, bound_handler):
        proof, _, timeseries = bound_handler
        np.testing.assert_array_equal(proof.segmentation, timeseries.data[1])
        assert proof.segmentation.shape == timeseries.data.shape[1:]

    def test_max_label_on_slice(self, bound_handler):
        proof, _, _ = bound_handler
        # t=1 carries labels 3 and 4; other timepoints carry 1, 2 and 5.
        assert proof.max_label == 4

    def test_bboxes_on_slice(self, bound_handler):
        proof, _, _ = bound_handler
        assert set(proof.bboxes) == {0, 3, 4}

    def test_reset_scribbles_targets_current_timepoint(self, bound_handler):
        proof, viewer, _ = bound_handler
        viewer.layers["Scribbles (t=1)"].data[0, 0, 0] = 7

        proof.reset_scribbles()

        np.testing.assert_array_equal(viewer.layers["Scribbles (t=1)"].data, 0)

    def test_reset_corrected_targets_current_timepoint(self, bound_handler):
        proof, viewer, _ = bound_handler
        proof.toggle_corrected_cell(3)

        proof.reset_corrected()

        np.testing.assert_array_equal(viewer.layers["Correct Labels (t=1)"].data, 0)
        assert proof.corrected_cells == set()

    def test_toggle_corrected_cell_marks_slice(self, bound_handler):
        proof, viewer, _ = bound_handler

        proof.toggle_corrected_cell(3)

        assert proof.corrected_cells == {3}
        mask = viewer.layers["Correct Labels (t=1)"].data
        assert mask.shape == (4, 10, 10)
        np.testing.assert_array_equal(mask[0:2, 0:2, 0:2], 1)
        assert mask.sum() == 8

    def test_undo_snapshot_is_slice_sized(self, bound_handler):
        proof, _, _ = bound_handler

        proof.save_to_history()

        snapshot = proof._state.history_undo[0]
        assert snapshot.segmentation.shape == (4, 10, 10)
        assert snapshot.corrected_cells_mask.shape == (4, 10, 10)

    def test_undo_restores_segmentation_at_timepoint(self, bound_handler):
        proof, _viewer, layer = bound_handler
        before = layer.data.copy()
        proof.save_to_history()
        edited = proof.segmentation.copy()
        edited[0:2, 0:2, 0:2] = 99
        layer.data[1] = edited

        proof.undo()

        np.testing.assert_array_equal(layer.data[1], before[1])
        assert (layer.data[1][0:2, 0:2, 0:2] == 3).all()

    def test_update_after_proofreading_writes_only_selected_timepoint(
        self, bound_handler
    ):
        proof, _, layer = bound_handler
        before = layer.data.copy()

        edited = proof.segmentation.copy()
        edited[0:2, 0:2, 0:2] = 42
        proof.update_after_proofreading(edited, _full_region_slice(edited.shape), {})

        after = layer.data
        np.testing.assert_array_equal(after[0], before[0])
        np.testing.assert_array_equal(after[2], before[2])
        np.testing.assert_array_equal(after[1], edited)

    def test_update_after_proofreading_2d_tyx(
        self, proof, make_napari_viewer_proxy, napari_timeseries_segmentation_2d
    ):
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_timeseries_segmentation_2d)
        layer = viewer.layers["test_segmentation_timeseries_2d"]
        proof.setup(
            PanSegImage.from_napari_layer(napari_timeseries_segmentation_2d),
            timepoint=2,
        )
        before = layer.data.copy()

        edited = proof.segmentation.copy()
        edited[5:7, 5:7] = 6
        proof.update_after_proofreading(edited, _full_region_slice(edited.shape), {})

        np.testing.assert_array_equal(layer.data[0], before[0])
        np.testing.assert_array_equal(layer.data[1], before[1])
        np.testing.assert_array_equal(layer.data[2], edited)

    def test_rebind_renames_canvases_and_persists_content(self, bound_handler):
        proof, viewer, _ = bound_handler
        viewer.layers["Scribbles (t=1)"].data[0, 0, 0] = 7
        viewer.layers["Correct Labels (t=1)"].data[1, 1, 1] = 7
        scribbles_before = viewer.layers["Scribbles (t=1)"].data.copy()
        corrected_before = viewer.layers["Correct Labels (t=1)"].data.copy()

        proof.rebind(2)

        assert proof.timepoint == 2
        assert "Scribbles (t=1)" not in viewer.layers
        assert "Correct Labels (t=1)" not in viewer.layers
        assert "Scribbles (t=2)" in viewer.layers
        assert "Correct Labels (t=2)" in viewer.layers
        np.testing.assert_array_equal(
            viewer.layers["Scribbles (t=2)"].data, scribbles_before
        )
        np.testing.assert_array_equal(
            viewer.layers["Correct Labels (t=2)"].data, corrected_before
        )

    def test_rebind_recomputes_derived_state(self, bound_handler):
        proof, _, _ = bound_handler
        proof.toggle_corrected_cell(3)
        proof.save_to_history()
        assert proof._state.history_undo

        proof.rebind(2)

        # The mask marked the t=0 cube; at t=2 that cube is label 5.
        assert proof.corrected_cells == {5}
        assert set(proof.bboxes) == {0, 5}
        assert len(proof._state.history_undo) == 0
        assert len(proof._state.history_redo) == 0

    def test_rebind_rejects_out_of_range_timepoint(self, bound_handler):
        proof, _, _ = bound_handler

        with pytest.raises(ValueError):
            proof.rebind(3)
        assert proof.timepoint == 1

    def test_save_and_load_roundtrip_timeseries(
        self, bound_handler, tmp_path, make_napari_viewer_proxy
    ):
        proof, viewer, layer = bound_handler
        proof.toggle_corrected_cell(3)
        h5_path = tmp_path / "timeseries_state.h5"

        proof.save_state_to_disk(h5_path, raw=None, pmap=None)

        with h5py.File(h5_path, "r") as f:
            assert f["label"].shape == layer.data.shape
            assert f["mask"].shape == (4, 10, 10)
            assert f["mask"].attrs["timepoint"] == 1
            assert set(f["mask"].attrs["corrected_cells"]) == {3}

        fresh_handler = ProofreadingHandler()
        fresh_handler.load_state_from_disk(h5_path)

        assert fresh_handler.is_timeseries
        assert fresh_handler.timepoint == 1
        assert fresh_handler.corrected_cells == {3}
        assert "Scribbles (t=1)" in viewer.layers
        assert "Correct Labels (t=1)" in viewer.layers
        # the loaded state restores the corrected mask into the viewer layer
        mask = viewer.layers["Correct Labels (t=1)"].data
        assert mask[0:2, 0:2, 0:2].sum() == 8

    def test_save_static_has_no_timepoint_attribute(
        self, proof, make_napari_viewer_proxy, napari_segmentation, tmp_path
    ):
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_segmentation)
        proof.setup(PanSegImage.from_napari_layer(napari_segmentation))
        h5_path = tmp_path / "static_state.h5"

        proof.save_state_to_disk(h5_path, raw=None, pmap=None)

        with h5py.File(h5_path, "r") as f:
            assert "timepoint" not in f["mask"].attrs

    def test_n_timepoints_raises_for_static(self, proof):
        with pytest.raises(ValueError):
            _ = proof.n_timepoints

    def test_timepoint_raises_for_static(self, proof):
        with pytest.raises(ValueError):
            _ = proof.timepoint

    def test_setup_requires_timepoint_for_timeseries(
        self, proof, napari_timeseries_segmentation
    ):
        with pytest.raises(ValueError, match="must be bound to a timepoint"):
            proof.setup(PanSegImage.from_napari_layer(napari_timeseries_segmentation))

    def test_corrected_cells_from_mask_excludes_background(self, bound_handler):
        proof, viewer, _ = bound_handler
        # A stale mark over background at t=1 (z=3, y=0, x=0 is outside every cell).
        viewer.layers["Correct Labels (t=1)"].data[3, 0, 0] = 1

        proof.rebind(1)

        assert 0 not in proof.corrected_cells
        assert proof.corrected_cells == set()

    def test_load_legacy_timeseries_file_without_mask(
        self, bound_handler, napari_timeseries_segmentation, tmp_path
    ):
        proof, viewer, _ = bound_handler
        proof.toggle_corrected_cell(3)
        h5_path = tmp_path / "legacy_timeseries_state.h5"
        proof.save_state_to_disk(h5_path, raw=None, pmap=None)
        with h5py.File(h5_path, "a") as f:
            del f["mask"]

        fresh_handler = ProofreadingHandler()
        fresh_handler.load_state_from_disk(h5_path)

        # No timepoint attribute: the session falls back to the displayed
        # timepoint (the T slider centers on t=1) with an empty slice mask.
        assert fresh_handler.active
        assert fresh_handler.timepoint == 1
        assert fresh_handler.corrected_cells == set()
        mask = viewer.layers["Correct Labels (t=1)"].data
        assert mask.shape == napari_timeseries_segmentation.data.shape[1:]
        np.testing.assert_array_equal(mask, 0)


class TestProofreadingTab:
    def test_init(self, tab):
        assert not tab.busy
        assert len(tab.container) == 12

    def test_timepoint_select_has_label_and_tooltip(self, tab):
        assert tab.widget_timepoint_select.label == "Timepoint"
        assert "timepoint" in tab.widget_timepoint_select.tooltip
        assert "Clean scribbles" in tab.widget_timepoint_select.tooltip

    def test_hide_all(self, tab):
        app = get_qapp()
        tab.container.show()
        tab._show_all_widgets()
        tab.widget_timepoint_container.show()
        tab.widget_timepoint_select.show()
        tab._hide_all_widgets()
        assert all(not w.visible for w in tab._session_widgets())
        assert not tab.widget_timepoint_container.visible
        assert not tab.widget_timepoint_select.visible
        tab.container.hide()
        app.quit()

    def test_show_all(self, tab):
        app = get_qapp()
        tab.container.show()
        tab._show_all_widgets()
        assert all(w.visible for w in tab._session_widgets())
        # The timeseries-only widgets are managed per session type.
        assert not tab.widget_timepoint_container.visible
        assert not tab.widget_timepoint_select.visible
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
    def test_widget_split_and_merge_from_scribbles(self, tab, mocker, napari_raw, rep):
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

        _pump_until(lambda: not tab.busy)
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
        _pump_until(lambda: not tab.busy)
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
        mocker.patch(
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

    def test_widget_filter_segmentation(self, tab, mocker, napari_segmentation):
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
        _pump_until(lambda: not tab.busy)
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


class TestProofreadingTabTimeSeries:
    @pytest.fixture
    def timeseries_tab(
        self, tab, make_napari_viewer_proxy, napari_timeseries_segmentation
    ):
        """A tab initialized on the deterministic timeseries layer at timepoint 0."""
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_timeseries_segmentation)
        viewer.dims.set_current_step(0, 0)
        tab.container.show()
        tab._initialize_from_layer(napari_timeseries_segmentation)
        return tab, viewer

    def test_init_defaults_to_slider_position(
        self, tab, make_napari_viewer_proxy, napari_timeseries_segmentation
    ):
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_timeseries_segmentation)
        tab.container.show()
        viewer.dims.set_current_step(0, 2)

        tab._initialize_from_layer(napari_timeseries_segmentation)

        assert tab.handler.timepoint == 2
        assert tab.widget_timepoint_container.visible
        assert tab.widget_timepoint_select.visible
        assert tab.widget_timepoint_select.value == 2
        assert tab.widget_timepoint_select.max == 2
        assert "Scribbles (t=2)" in viewer.layers

    def test_init_static_hides_timepoint_widgets(
        self, tab, make_napari_viewer_proxy, napari_segmentation
    ):
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_segmentation)
        tab.container.show()

        tab._initialize_from_layer(napari_segmentation)

        assert not tab.widget_timepoint_container.visible
        assert not tab.widget_timepoint_select.visible
        assert not tab.handler.is_timeseries

    def test_reinit_removes_stale_helper_layers(
        self, timeseries_tab, napari_segmentation, mocker
    ):
        tab, viewer = timeseries_tab
        assert "Scribbles (t=0)" in viewer.layers
        viewer.add_layer(napari_segmentation)

        tab._initialize_from_layer(napari_segmentation, are_you_sure=True)

        assert "Scribbles (t=0)" not in viewer.layers
        assert "Correct Labels (t=0)" not in viewer.layers
        assert "Scribbles" in viewer.layers
        assert "Correct Labels" in viewer.layers
        assert not tab.handler.is_timeseries

    def test_timepoint_input_change_rebinds_and_moves_slider(self, timeseries_tab):
        tab, viewer = timeseries_tab
        assert tab.handler.timepoint == 0
        scribbles = viewer.layers["Scribbles (t=0)"]
        scribbles.data[0, 0, 0] = 7
        marks = scribbles.data.copy()

        tab.widget_timepoint_select.value = 2

        assert tab.handler.timepoint == 2
        assert viewer.dims.current_step[0] == 2
        assert "Scribbles (t=0)" not in viewer.layers
        assert "Scribbles (t=2)" in viewer.layers
        np.testing.assert_array_equal(viewer.layers["Scribbles (t=2)"].data, marks)

    def test_slider_move_does_not_update_timepoint_input(self, timeseries_tab):
        tab, viewer = timeseries_tab
        assert tab.handler.timepoint == 0

        viewer.dims.set_current_step(0, 2)

        assert tab.widget_timepoint_select.value == 0
        assert tab.handler.timepoint == 0

    def test_init_from_layer_rejects_timeseries_helper_layer_by_prefix(
        self, tab, make_napari_viewer_proxy, napari_timeseries_segmentation, mocker
    ):
        mock_log = mocker.patch("panseg.viewer_napari.widgets.proofreading.log")
        layer = napari_timeseries_segmentation
        layer.name = "Scribbles (t=1)"

        tab._initialize_from_layer(layer)

        mock_log.assert_called_with(
            "Scribble or corrected cells layer is not intended to be proofread, choose a segmentation",
            thread="Proofreading tool",
            level="error",
        )
        assert not tab.handler.active

    def test_init_from_layer_choices_exclude_helper_layers_by_prefix(
        self, tab, make_napari_viewer_proxy, napari_timeseries_segmentation
    ):
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_timeseries_segmentation)
        viewer.add_labels(
            np.zeros((3, 4, 10, 10), dtype="uint16"), name="Scribbles (t=1)"
        )

        tab._initialize_from_layer(napari_timeseries_segmentation)

        choices = tab.widget_proofreading_initialisation.segmentation.choices
        assert napari_timeseries_segmentation in choices
        assert viewer.layers["Scribbles (t=1)"] not in choices

    def test_split_merge_refused_when_slider_diverges(
        self, timeseries_tab, mocker, napari_timeseries_prediction
    ):
        tab, viewer = timeseries_tab
        viewer.dims.set_current_step(0, 2)
        mock_log = mocker.patch("panseg.viewer_napari.widgets.proofreading.log")
        mock_split_merge = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.split_merge_from_seeds"
        )

        tab.widget_split_and_merge_from_scribbles(
            viewer=viewer, image=napari_timeseries_prediction
        )

        mock_split_merge.assert_not_called()
        assert mock_log.call_args_list[-1].args[0] == (
            "The displayed timepoint (2) does not match the session timepoint (0). "
            "Split/Merge was not applied: set the Timepoint field to 2 or move the time "
            "slider back to 0."
        )
        assert not tab.busy

    def test_set_busy_locks_timepoint_field(self, timeseries_tab):
        tab, _viewer = timeseries_tab
        assert tab.widget_timepoint_select.enabled

        tab._set_busy(True)

        assert tab.busy
        assert not tab.widget_timepoint_select.enabled

        tab._set_busy(False)

        assert not tab.busy
        assert tab.widget_timepoint_select.enabled

    def test_split_merge_writeback_isolated_from_midflight_timepoint_change(
        self,
        timeseries_tab,
        napari_timeseries_prediction,
        mocker,
    ):
        """A Timepoint change while a split/merge worker runs is refused.

        The in-flight write-back must land in the session timepoint's slice
        only, and the undo snapshot the worker pushed must survive.
        """
        tab, viewer = timeseries_tab
        layer = viewer.layers["test_segmentation_timeseries"]
        before = layer.data.copy()
        scribbles = viewer.layers["Scribbles (t=0)"]
        # One scribble color over the two t=0 cells: they merge.
        scribbles.data[0:2, 0:2, 0:2] = 1
        scribbles.data[0:2, 5:7, 5:7] = 1

        # Hold the worker inside split_merge_from_seeds — after it pushed the
        # undo snapshot, before it writes back — to make the race window
        # deterministic.
        reached, release = threading.Event(), threading.Event()
        real_split_merge = split_merge_from_seeds

        def blocked_split_merge(*args, **kwargs):
            reached.set()
            assert release.wait(timeout=30)
            return real_split_merge(*args, **kwargs)

        mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.split_merge_from_seeds",
            side_effect=blocked_split_merge,
        )
        mock_log = mocker.patch("panseg.viewer_napari.widgets.proofreading.log")

        worker = tab.widget_split_and_merge_from_scribbles(
            viewer=viewer, image=napari_timeseries_prediction
        )
        assert worker is not None
        assert tab.busy
        try:
            # The Timepoint field is locked while the worker runs.
            assert not tab.widget_timepoint_select.enabled
            assert reached.wait(timeout=10)
            assert len(tab.handler._state.history_undo) == 1

            tab.widget_timepoint_select.value = 2

            # The rebind is refused: session, slider and field all stay on t=0.
            assert mock_log.call_args_list[-1].args[0] == (
                "The proofreading tool is busy. The timepoint change to 2 was not "
                "applied: wait for the running worker to finish before switching "
                "timepoints."
            )
            assert tab.handler.timepoint == 0
            assert tab.widget_timepoint_select.value == 0
            assert viewer.dims.current_step[0] == 0
            assert "Scribbles (t=0)" in viewer.layers

            release.set()
            _pump_until(lambda: not tab.busy)
            assert tab.widget_timepoint_select.enabled

            # The merged result landed in t=0 only.
            after = layer.data
            np.testing.assert_array_equal(after[1], before[1])
            np.testing.assert_array_equal(after[2], before[2])
            merged = after[0]
            assert (merged[0:2, 0:2, 0:2] == 1).all()
            assert (merged[0:2, 5:7, 5:7] == 1).all()
            assert (merged[4:, 8:, 8:] == 0).all()

            # The worker's undo snapshot survived the attempted rebind.
            assert len(tab.handler._state.history_undo) == 1
            assert not tab.handler._state.history_redo
            np.testing.assert_array_equal(
                tab.handler._state.history_undo[0].segmentation, before[0]
            )
        finally:
            # Never leave the worker blocked in a failed test run.
            release.set()
            with contextlib.suppress(Exception):
                _pump_until(lambda: not worker.is_running)

    def test_split_merge_applies_to_session_timepoint_only(
        self, timeseries_tab, napari_timeseries_prediction
    ):
        tab, viewer = timeseries_tab
        layer = viewer.layers["test_segmentation_timeseries"]
        before = layer.data.copy()
        scribbles = viewer.layers["Scribbles (t=0)"]
        # One scribble color over the two t=0 cells: they merge.
        scribbles.data[0:2, 0:2, 0:2] = 1
        scribbles.data[0:2, 5:7, 5:7] = 1

        worker = tab.widget_split_and_merge_from_scribbles(
            viewer=viewer, image=napari_timeseries_prediction
        )
        assert worker is not None
        _pump_until(lambda: not tab.busy)

        after = layer.data
        np.testing.assert_array_equal(after[1], before[1])
        np.testing.assert_array_equal(after[2], before[2])
        merged = after[0]
        assert (merged[0:2, 0:2, 0:2] == 1).all()
        assert (merged[0:2, 5:7, 5:7] == 1).all()
        assert (merged[4:, 8:, 8:] == 0).all()

    def test_split_merge_slices_boundary_image_at_session_timepoint(
        self,
        timeseries_tab,
        napari_timeseries_prediction,
        mocker,
    ):
        tab, viewer = timeseries_tab
        mock_split_merge = mocker.patch(
            "panseg.viewer_napari.widgets.proofreading.split_merge_from_seeds"
        )
        mock_split_merge.return_value = (
            tab.handler.segmentation.copy(),
            _full_region_slice((4, 10, 10)),
            {},
        )
        scribbles = viewer.layers["Scribbles (t=0)"]
        scribbles.data[0, 0, 0] = 1

        tab.widget_split_and_merge_from_scribbles(
            viewer=viewer, image=napari_timeseries_prediction
        )
        _pump_until(lambda: not tab.busy)

        image_data = mock_split_merge.call_args.kwargs["image"]
        np.testing.assert_array_equal(image_data, napari_timeseries_prediction.data[0])

    def test_double_click_refused_when_slider_diverges(
        self, timeseries_tab, make_napari_viewer_proxy, mocker
    ):
        tab, viewer = timeseries_tab
        viewer.dims.set_current_step(0, 2)
        mock_log = mocker.patch("panseg.viewer_napari.widgets.proofreading.log")
        mock_toggle = mocker.patch.object(tab.handler, "toggle_corrected_cell")

        tab._widget_add_label_to_corrected(viewer, (2, 1, 1, 1))

        mock_toggle.assert_not_called()
        assert mock_log.call_args_list[-1].args[0] == (
            "The displayed timepoint (2) does not match the session timepoint (0). "
            "The cell marking was not applied: set the Timepoint field to 2 or move "
            "the time slider back to 0."
        )

    def test_widget_add_label_to_corrected_rasterizes_slice(self, timeseries_tab):
        tab, viewer = timeseries_tab
        # Event position at t=0, z=1, y=1, x=1; scale (1, 1, 1, 1): the t
        # axis is in timepoint indices, so the world position is the index.
        position = (0.0, 1.0, 1.0, 1.0)

        tab._widget_add_label_to_corrected(viewer, position)

        # (z, y, x) = (1, 1, 1) is inside the t=0 cube, label 1.
        assert tab.handler.corrected_cells == {1}

    def test_extract_corrected_labels_single_timepoint_layer(
        self,
        timeseries_tab,
    ):
        tab, viewer = timeseries_tab
        tab.widget_timepoint_select.value = 1
        tab.handler.toggle_corrected_cell(3)

        worker = tab.widget_filter_segmentation()
        assert worker is not None
        # The Timepoint field is locked while the extraction runs.
        assert not tab.widget_timepoint_select.enabled
        _pump_until(lambda: not tab.busy)
        assert tab.widget_timepoint_select.enabled

        extracted = viewer.layers["test_segmentation_timeseries_corrected_t001"]
        assert extracted.data.shape == (4, 10, 10)
        expected = tab.handler.segmentation.copy()
        expected[tab.handler.corrected_cells_mask == 0] = 0
        np.testing.assert_array_equal(extracted.data, expected)

    def test_clean_scribble_targets_current_timepoint(self, timeseries_tab, mocker):
        tab, viewer = timeseries_tab
        viewer.layers["Scribbles (t=0)"].data[0, 0, 0] = 7
        mock_reset = mocker.patch.object(tab.handler, "reset_scribbles")

        tab.widget_clean_scribble(viewer=viewer)

        mock_reset.assert_called_once()

    def test_init_from_file_binds_saved_timepoint(
        self, tab, make_napari_viewer_proxy, napari_timeseries_segmentation, tmp_path
    ):
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_timeseries_segmentation)
        tab.container.show()
        tab._initialize_from_layer(napari_timeseries_segmentation)
        tab.handler.toggle_corrected_cell(3)
        h5_path = tmp_path / "timeseries_state.h5"
        tab.handler.save_state_to_disk(h5_path, raw=None, pmap=None)
        viewer.layers.remove("test_segmentation_timeseries")
        viewer.dims.set_current_step(0, 0)

        tab._initialize_from_file(h5_path, are_you_sure=True)

        assert tab.handler.timepoint == 1
        assert tab.widget_timepoint_select.visible
        assert tab.widget_timepoint_select.value == 1
        assert viewer.dims.current_step[0] == 1
        assert tab.handler.corrected_cells == {3}
        assert "Correct Labels (t=1)" in viewer.layers
