"""Probes for the macOS CI freeze: layer.refresh() on canvas-bound layers.

PR #613's tests are the first in the suite to call ``layer.refresh()`` on a
layer already bound to a (never shown) QtViewer canvas, and the macOS CI job
freezes deterministically on the first such test. These probes bisect the
refresh into its sub-steps (napari 0.9.1 ``Layer.refresh`` kwarg surface) to
identify which sub-step wedges the runner.

Each probe is an independent test; execution order is definition order, so
the pytest progress line stops exactly at the freezing variant.
"""

import numpy as np
import pytest


@pytest.fixture
def labels4d(make_napari_viewer):
    viewer = make_napari_viewer()
    data = np.zeros((3, 4, 10, 10), dtype="uint16")
    data[1, 0:2, 0:2, 0:2] = 3
    layer = viewer.add_labels(data, name="probe_labels")
    return viewer, layer


def test_01_add_layer_only(labels4d):
    """Control: add a canvas-bound layer without ever refreshing it."""
    viewer, _ = labels4d
    assert "probe_labels" in viewer.layers


def test_02_refresh_no_events(labels4d):
    """Refresh with every sub-step disabled: pure re-slicing, no events."""
    _, layer = labels4d
    layer.refresh(thumbnail=False, data_displayed=False, highlight=False, extent=False)


def test_03_refresh_data_displayed_only(labels4d):
    """Only the vispy data update: events.set_data -> node.set_data + update."""
    _, layer = labels4d
    layer.data[1, 0, 0, 0] = 9
    layer.refresh(thumbnail=False, data_displayed=True, highlight=False, extent=False)


def test_04_refresh_thumbnail_only(labels4d):
    _, layer = labels4d
    layer.refresh(thumbnail=True, data_displayed=False, highlight=False, extent=False)


def test_05_refresh_highlight_only(labels4d):
    _, layer = labels4d
    layer.refresh(thumbnail=False, data_displayed=False, highlight=True, extent=False)


def test_06_refresh_extent_only(labels4d):
    _, layer = labels4d
    layer.refresh(thumbnail=False, data_displayed=False, highlight=False, extent=True)


def test_07_refresh_default(labels4d):
    """The full default refresh, as used by update_after_proofreading."""
    _, layer = labels4d
    layer.data[1, 0, 0, 0] = 9
    layer.refresh()


def test_08_refresh_default_twice(labels4d):
    _, layer = labels4d
    layer.refresh()
    layer.refresh()


def test_09_full_context_default_refresh(make_napari_viewer_proxy):
    """Closest replica of the frozen test's viewer state.

    Proxy-wrapped viewer, TZYX labels layer plus the two slice-sized helper
    canvases, in-place write at the session timepoint, default refresh.
    """
    viewer = make_napari_viewer_proxy()
    data = np.zeros((3, 4, 10, 10), dtype="uint16")
    data[1, 0:2, 0:2, 0:2] = 3
    layer = viewer.add_labels(data, name="test_segmentation_timeseries")
    viewer.add_labels(np.zeros((4, 10, 10), dtype="uint16"), name="Scribbles (t=1)")
    viewer.add_labels(
        np.zeros((4, 10, 10), dtype="uint16"), name="Correct Labels (t=1)"
    )

    edited = layer.data[1].copy()
    edited[0:2, 0:2, 0:2] = 42
    layer.data[(1, slice(0, 4), slice(0, 10), slice(0, 10))] = edited
    layer.refresh()
