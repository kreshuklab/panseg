"""Shared session fixtures for the napari widget tests."""

import gc
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def _absorb_widget_one_timers(qapp):
    """Pay the session one-time napari startup costs before the first widget test.

    The first ``QtViewer`` (vispy canvas, GL context) and the first napari
    settings write are expensive one-timers that would otherwise land on the
    durations line of whichever test creates the first viewer. Creating and
    destroying one viewer up front moves them off the per-test lines. The
    ``QtViewer._instances`` check mirrors the leak assertion napari's viewer
    fixture makes at setup, so later viewer tests see a clean registry.
    """
    from napari import Viewer
    from napari._qt.qt_viewer import QtViewer
    from napari.settings import get_settings

    get_settings().reset()

    viewer = Viewer(show=False, show_welcome_screen=False)
    with patch.object(viewer.window._qt_window, "_save_current_window_settings"):
        viewer.close()
    del viewer
    gc.collect()
    assert len(QtViewer._instances) == 0, "absorber leaked a QtViewer"
