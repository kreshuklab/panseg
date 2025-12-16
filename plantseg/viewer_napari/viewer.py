import warnings
import webbrowser
from pathlib import Path

import napari
from qtpy import QtCore, QtWidgets
from qtpy.QtGui import QPicture, QPixmap

from plantseg import logger
from plantseg.__version__ import __version__
from plantseg.utils import check_version
from plantseg.viewer_napari.containers import (
    get_proofreading_tab,
)
from plantseg.viewer_napari.updater import update
from plantseg.viewer_napari.widgets import proofreading
from plantseg.viewer_napari.widgets.input import Input_Tab
from plantseg.viewer_napari.widgets.output import Output_Tab
from plantseg.viewer_napari.widgets.postprocessing import Postprocessing_Tab
from plantseg.viewer_napari.widgets.preprocessing import Preprocessing_Tab
from plantseg.viewer_napari.widgets.segmentation import Segmentation_Tab
from plantseg.viewer_napari.widgets.training import Training_Tab
from plantseg.viewer_napari.widgets.utils import decrease_font_size, increase_font_size


def scroll_wrap(w):
    scrollArea = QtWidgets.QScrollArea()
    scrollArea.setWidget(w.native)
    scrollArea.setWidgetResizable(True)
    pol = QtWidgets.QSizePolicy()
    pol.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Minimum)
    pol.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding)
    scrollArea.setSizePolicy(pol)
    return scrollArea


class Plantseg_viewer:
    def __init__(self, viewer=None):
        if viewer is None:
            viewer = napari.Viewer(title="PlantSeg v2")
        self.viewer = viewer

        self._container_list = []
        self._tabs = []

        self.viewer.window.plugins_menu.addAction("Update PlantSeg", update)
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

    def setup_welcome_page(self):
        welcome_widget = self.viewer.window._qt_viewer._welcome_widget
        v_short, v_features = check_version(current_version=__version__, silent=True)

        for i, child in enumerate(welcome_widget.findChildren(QtWidgets.QWidget)):
            if isinstance(child, QtWidgets.QLabel):
                if i == 3:
                    child.setText(
                        "Welcome to PlantSeg!\n\nTo load an image use the menu on the right\n\n"
                        + v_short
                        + "\n\n"
                        + v_features
                    )
                else:
                    child.setText("")
                child.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

    def finalize_viewer(self):
        # Show data tab by default
        if logger.level > 10:  # 10 = DEBUG level
            # Suppress FutureWarning about `dock_widgets`` being private
            # We need to use `_dock_widgets` (not `dock_widgets`) because we need access
            # to the `QtViewerDockWidget` objects which have `.show()` and `.raise_()` methods
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.viewer.window._dock_widgets["Input"].show()
                self.viewer.window._dock_widgets["Input"].raise_()
        else:
            self.viewer.window._dock_widgets["Input"].show()
            self.viewer.window._dock_widgets["Input"].raise_()

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
        self.finalize_viewer()

        self.wd = WelcomeDialog(self.viewer.window._qt_viewer)
        self.wd.show()

        napari.run()


class WelcomeDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(WelcomeDialog, self).__init__(parent)
        self.setWindowTitle("PanSeg 2.0 released")
        # layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        p = str((Path(__file__) / "../../resources/logo_change.png").resolve())
        image = QPixmap(p).scaledToWidth(500)
        im1 = QtWidgets.QLabel()
        im1.setPixmap(image)

        self.text = QtWidgets.QLabel(
            "<h1>PlantSeg is now PanSeg!</h1>"
            "<p>With the 2.0 release we rename PlantSeg to PanSeg to highlight "
            "its capabilities beyond plant tissue segmentation.</p>"
            "<p><b>To get the new version run the update, or if not possible, follow the download link below!</b><br>"
            "(Update only possible for versions installed using the executable.)</p>"
            "<p>This new release is acompanied by a new publication you can also find below.</p>",
            parent,
        )
        self.text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.publication_button = QtWidgets.QPushButton("Open Publication")
        self.publication_button.clicked.connect(self.open_publication)
        self.publication_button.setStyleSheet("background-color: green")
        self.download_button = QtWidgets.QPushButton("Download PanSeg 2.0")
        self.download_button.clicked.connect(self.open_download)
        self.download_button.setStyleSheet("background-color: green")
        self.update_button = QtWidgets.QPushButton("Run update")
        self.update_button.setStyleSheet("background-color: green")
        self.update_button.clicked.connect(update)

        layout.addWidget(im1, 0, 0, 1, 3, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.text, 1, 0, 1, 3)
        layout.addWidget(
            self.download_button,
            3,
            0,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        layout.addWidget(
            self.update_button,
            3,
            1,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        layout.addWidget(
            self.publication_button,
            3,
            2,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )

    def open_download(self):
        # TODO: Change URL
        webbrowser.open(
            "https://kreshuklab.github.io/plant-seg/", new=0, autoraise=True
        )

    def open_publication(self):
        # TODO: Change URL
        webbrowser.open(
            "https://kreshuklab.github.io/plant-seg/", new=0, autoraise=True
        )
