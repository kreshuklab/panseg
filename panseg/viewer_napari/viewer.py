from pathlib import Path
import warnings

import napari
from napari._qt.qt_event_loop import _svg_path_to_icon
from napari.components._viewer_constants import CanvasPosition
from qtpy import QtCore, QtWidgets

from panseg import logger
from panseg.__version__ import __version__
from panseg.utils import check_version
from panseg.viewer_napari.containers import (
    get_proofreading_tab,
)
from panseg.viewer_napari.updater import update
from panseg.viewer_napari.widgets import proofreading
from panseg.viewer_napari.widgets.input import Input_Tab
from panseg.viewer_napari.widgets.output import Output_Tab
from panseg.viewer_napari.widgets.postprocessing import Postprocessing_Tab
from panseg.viewer_napari.widgets.preprocessing import Preprocessing_Tab
from panseg.viewer_napari.widgets.segmentation import Segmentation_Tab
from panseg.viewer_napari.widgets.training import Training_Tab
from panseg.viewer_napari.widgets.utils import decrease_font_size, increase_font_size


def scroll_wrap(w):
    scrollArea = QtWidgets.QScrollArea()
    scrollArea.setWidget(w.native)
    scrollArea.setWidgetResizable(True)
    pol = QtWidgets.QSizePolicy()
    pol.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Minimum)
    pol.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding)
    scrollArea.setSizePolicy(pol)
    return scrollArea


class Panseg_viewer:
    def __init__(self, viewer=None):
        if viewer is None:
            viewer = napari.Viewer(title="PanSeg v2", show_welcome_screen=False)
        self.viewer = viewer

        self._container_list = []
        self._tabs = []

        self.viewer.window.plugins_menu.addAction("Update PanSeg", update)
        self.viewer.window.view_menu.addAction(
            "Increase font size", increase_font_size, "Ctrl+i"
        )
        self.viewer.window.view_menu.addAction(
            "Decrease font size", decrease_font_size, "Ctrl+o"
        )

    def init_tabs(self):
        self.input_tab = Input_Tab()
        self.output_tab = Output_Tab()
        self.preprocessing_tab = Preprocessing_Tab()
        self.segmentation_tab = Segmentation_Tab()
        self.postprocessing_tab = Postprocessing_Tab()
        self.training_tab = Training_Tab(self.segmentation_tab.prediction_widgets)

        self._tabs = [
            self.input_tab,
            self.output_tab,
            self.preprocessing_tab,
            self.segmentation_tab,
            self.postprocessing_tab,
            self.training_tab,
        ]

    @property
    def container_list(self):
        assert self._tabs, "Tabs not initialized!"
        if self._container_list == []:
            self._container_list = [
                (self.input_tab.get_container(), "Input"),
                (self.preprocessing_tab.get_container(), "Preprocessing"),
                (self.segmentation_tab.get_container(), "Segmentation"),
                (self.postprocessing_tab.get_container(), "Postprocessing"),
                (get_proofreading_tab(), "Proofreading"),
                (self.output_tab.get_container(), "Output"),
                (self.training_tab.get_container(), "Train"),
            ]
        return self._container_list

    def add_containers_to_dock(self):
        for _containers, name in self.container_list:
            # _containers.native.setMaximumHeight(600)
            # allow content to float to top of dock
            _containers.native.layout().addStretch()
            _containers.native.setMinimumWidth(400)
            _containers = scroll_wrap(_containers)
            _containers.setMinimumWidth(350)
            self.viewer.window.add_dock_widget(
                _containers,
                name=name,
                tabify=True,
            )

    def setup_layer_updates(self):
        assert self._tabs, "Tabs not initialized!"
        # Drop-down update for new layers
        self.viewer.layers.events.inserted.connect(
            self.preprocessing_tab.update_layer_selection
        )
        self.viewer.layers.events.inserted.connect(
            self.segmentation_tab.update_layer_selection
        )
        self.viewer.layers.events.inserted.connect(
            self.postprocessing_tab.update_layer_selection
        )
        self.viewer.layers.events.inserted.connect(
            self.training_tab.update_layer_selection
        )
        self.viewer.layers.events.inserted.connect(proofreading.update_layer_selection)
        self.viewer.layers.events.inserted.connect(
            self.output_tab.update_layer_selection
        )

        # Drop-down update for renaming of layers
        self.viewer.layers.selection.events.active.connect(
            self.input_tab.update_layer_selection
        )
        self.viewer.layers.selection.events.active.connect(
            self.preprocessing_tab.update_layer_selection
        )
        self.viewer.layers.selection.events.active.connect(
            self.segmentation_tab.update_layer_selection
        )
        self.viewer.layers.selection.events.active.connect(
            self.postprocessing_tab.update_layer_selection
        )
        self.viewer.layers.selection.events.active.connect(
            self.training_tab.update_layer_selection
        )
        self.viewer.layers.selection.events.active.connect(
            proofreading.update_layer_selection
        )
        self.viewer.layers.selection.events.active.connect(
            self.output_tab.update_layer_selection
        )

        # welcome overlay
        self.viewer.layers.events.removed.connect(self._on_layerlist_change)
        self.viewer.layers.events.inserted.connect(self._on_layerlist_change)

    def setup_welcome_page(self):
        v_short, v_features = check_version(current_version=__version__, silent=True)
        text = (
            "\n \n \n"
            + "Welcome to PanSeg!\n \n \n"
            + "To load an image use the menu on the right\n\n"
            + v_short
            + "\n\n"
            + v_features
        )
        self.viewer.text_overlay.text = text
        self.viewer.text_overlay.visible = True
        # self.viewer.text_overlay.order = 1000000
        self.viewer.welcome_screen.visible = False

        self.viewer.text_overlay.position = CanvasPosition.TOP_CENTER

    def _on_layerlist_change(self):
        """Hide the welcome screen when layers get created."""
        if len(self.viewer.layers) > 0:
            self.viewer.text_overlay.visible = False
        else:
            self.viewer.text_overlay.visible = True

    def setup_logo(self):
        """Set the icon of the window title bar."""
        logo_path = Path(__file__).parent.parent / "resources" / "logo.svg"
        self.viewer.window._qt_window._window_icon = logo_path
        self.viewer.window._update_logo()

    def finalize_viewer(self):
        # Show input tab by default
        self.viewer.window.dock_widgets["Input"].parent().show()
        self.viewer.window.dock_widgets["Input"].parent().raise_()

        self.viewer.window.file_menu.menuAction().setVisible(False)
        self.viewer.window.layers_menu.menuAction().setVisible(False)

        # By hiding the menu, also closing shortcuts are removed
        @self.viewer.bind_key("Ctrl+w")
        def _close_window(viewer):
            self.viewer.window._qt_window.close(False, True)

        @self.viewer.bind_key("Ctrl+q")
        def _close_app(viewer):
            viewer.window._qt_window.close(True, True)

    def start_viewer(self):
        self.init_tabs()
        self.add_containers_to_dock()
        self.setup_layer_updates()
        self.setup_welcome_page()
        self.setup_logo()
        self.finalize_viewer()

        napari.run()
