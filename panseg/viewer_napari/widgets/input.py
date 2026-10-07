import webbrowser
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from magicgui import magic_factory
from magicgui.widgets import Container, Label, PushButton, create_widget
from magicgui.widgets.bases import ButtonWidget, CategoricalWidget, ValueWidget
from napari.layers import Image, Labels, Layer
from qtpy import QtGui

from panseg import logger
from panseg.core.image import TIME_UNIT_CHOICES, PanSegImage, SemanticType
from panseg.io import H5_EXTENSIONS, ZARR_EXTENSIONS
from panseg.io.h5 import list_h5_keys
from panseg.io.io import guess_stack_layout
from panseg.io.pil import PIL_EXTENSIONS
from panseg.io.tiff import TIFF_EXTENSIONS
from panseg.io.zarr import list_zarr_keys
from panseg.tasks.dataprocessing_tasks import set_t_spacing_task, set_voxel_size_task
from panseg.tasks.io_tasks import import_image_task
from panseg.viewer_napari import log
from panseg.viewer_napari.widgets.utils import (
    _return_value_if_widget,
    div,
    get_layers,
    schedule_task,
)


class InputType(Enum):
    RAW = "Image"
    PREDICTION = "Boundary"
    SEGMENTATION = "Segmentation"

    @classmethod
    def to_choices(cls):
        return [member.value for member in cls]


class PathMode(Enum):
    FILE = "tiff, h5, png, jpg"
    DIR = "zarr"

    @classmethod
    def to_choices(cls):
        return [member.value for member in cls]


class Input_Tab:
    def __init__(self):
        self.current_dataset_keys: Sequence[Optional[str]] = [None]

        # @@@@@ Open File @@@@@
        self.widget_open_file = self.factory_open_file()
        self._wrap_key_refresh()
        self.widget_open_file.self.bind(self)

        self.widget_open_file.path_mode.changed.connect(self._on_path_mode_changed)
        self.widget_open_file.path.changed.connect(self._on_path_changed)
        self.button_key_refresh.changed.connect(self._on_refresh_keys_button)
        self.path_changed_once = False

        self.dataset_key._default_choices = lambda _: self.current_dataset_keys
        self.dataset_key.reset_choices()
        self.dataset_key.changed.connect(self._on_dataset_key_changed)
        self.widget_open_file.called.connect(self._on_done)

        # @@@@@@ Set Voxel size @@@@@
        self.widget_set_voxel_size = self.factory_set_voxel_size()
        self.widget_set_voxel_size.self.bind(self)
        self.widget_set_voxel_size.hide()

        # self.widget_set_voxel_size.called.connect(self._on_set_voxel_size_layer_done)

        # @@@@@ Set time spacing @@@@@
        self.widget_set_t_spacing = self.factory_set_t_spacing()
        self._wrap_t_spacing()
        self.widget_set_t_spacing.self.bind(self)
        self.widget_set_t_spacing.hide()

        # @@@@@ Show info @@@@@
        self.widget_info = Label(
            value="Select layer to show information here...",
            label="Infos",
            tooltip="Information about the selected layer.",
        )
        self.widget_info.hide()

        # @@@@@ Show details @@@@@
        self.widget_details_layer_select = self.factory_details_layer_select()
        self.widget_details_layer_select.self.bind(self)
        self.widget_details_layer_select.layer.changed.connect(
            self._on_details_layer_select_changed
        )

        self.docs = Docs_Container()

        # self.widget_show_info.layer.changed.connect(self._on_info_layer_changed)

    def get_container(self):
        return Container(
            widgets=[
                self.docs.get_doc_container(),
                div("Open File"),
                self.widget_open_file,
                div("Details"),
                self.widget_details_layer_select,
                self.widget_info,
                div("Set voxel size"),
                self.widget_set_voxel_size,
                div("Set time spacing"),
                self.widget_set_t_spacing,
            ],
            labels=False,
        )

    @magic_factory(
        call_button="Open File",
        path_mode={
            "label": "File type",
            "choices": PathMode.to_choices(),
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
            "value": PathMode.FILE.value,
        },
        path={
            "value": Path.home(),
            "label": "File path",
            "mode": "r",
            "tooltip": "Select a file to be imported, the file can be a tiff, h5, png, jpg.",
        },
        new_layer_name={
            "value": "",
            "label": "Layer name",
            "tooltip": "Define the name of the output layer, default is either image or label.",
        },
        layer_type={
            "value": InputType.RAW.value,
            "label": "Layer type",
            "tooltip": "Select the type of input you are loading.",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
            "choices": InputType.to_choices(),
        },
        stack_layout={
            "value": "",
            "label": "Stack layout",
            "tooltip": "t for time, c for channel, xyz for dimensions, e.g.:\ntzyxc will be reshaped to [T][C][Z]YX.\nInvert an axis by adding `-` infront of the letter.\nTruncate the data before importing with a slice after the letters, e.g. txyz[:3,:,:]:\nthe entries follow the layout as written, before reordering, an integer drops its axis.",
            "widget_type": "LineEdit",
        },
    )
    def factory_open_file(
        self,
        path_mode: bool,
        path: Path,
        stack_layout: str,
        layer_type: str,
        new_layer_name: str,
    ) -> None:
        """Open a file and return a napari layer."""

        # the function argument is always None because of the magicfactory
        dataset_key = self.dataset_key.value

        if not self.path_changed_once:
            log("Please select a file to load!", thread="Input")
            return
        if layer_type == InputType.RAW.value:
            semantic_type = SemanticType.RAW
        elif layer_type == InputType.SEGMENTATION.value:
            semantic_type = SemanticType.SEGMENTATION
        elif layer_type == InputType.PREDICTION.value:
            semantic_type = SemanticType.PREDICTION
        else:
            raise ValueError(f"Unknown layer type {layer_type}")

        schedule_task(
            import_image_task,
            task_kwargs={
                "input_path": path,
                "key": dataset_key,
                "image_name": new_layer_name,
                "semantic_type": semantic_type,
                "stack_layout": stack_layout,
            },
        )

    def _wrap_key_refresh(self):
        w = self.widget_open_file

        dataset_key_d = {
            "label": "Key (h5/zarr only)",
            "widget_type": "ComboBox",
            "options": {
                "tooltip": "Key to be loaded from h5",
            },
            "annotation": str,
            "name": "_dataset_key",
        }
        refresh_d = {
            "label": "Refresh",
            "widget_type": "PushButton",
            "annotation": bool,
            "name": "_button_key_refresh",
        }

        key: CategoricalWidget = create_widget(**dataset_key_d)
        refresh: ButtonWidget = create_widget(**refresh_d)
        refresh.max_width = 80

        combo = Container(
            widgets=[key, refresh],
            layout="horizontal",
            labels=False,
            name="key_combo",
            label="Dataset key",
            gui_only=True,
        )

        self.dataset_key = key
        self.button_key_refresh = refresh
        combo.hide()
        w.insert(3, combo)

    def generate_layer_name(self, path: Path, dataset_key: str) -> str:
        dataset_key = dataset_key.replace("/", "_")

        if self.widget_open_file.key_combo.visible:
            if "label" in self.dataset_key.value:
                self.widget_open_file.layer_type.value = InputType.SEGMENTATION.value
            elif "raw" in self.dataset_key.value:
                self.widget_open_file.layer_type.value = InputType.RAW.value

        return path.stem + dataset_key

    def look_up_dataset_keys(self, path: Path):
        path = _return_value_if_widget(path)
        if not path.exists():
            return
        ext = path.suffix.lower()

        if ext in H5_EXTENSIONS:
            self.widget_open_file.key_combo.show()
            dataset_keys = list_h5_keys(path)

        elif ext in ZARR_EXTENSIONS:
            self.widget_open_file.key_combo.show()
            dataset_keys = list_zarr_keys(path)

        else:
            self.widget_open_file.new_layer_name.value = self.generate_layer_name(
                path, ""
            )
            self.widget_open_file.key_combo.hide()
            return

        self.current_dataset_keys = dataset_keys.copy()
        self.dataset_key.choices = dataset_keys

        # handle empth zarr/h5 files
        if dataset_keys == [None] or not dataset_keys:
            self.widget_open_file.key_combo.hide()
            return

        # change selected key only if old key is unavailable
        if self.dataset_key.value not in dataset_keys:
            self.dataset_key.value = dataset_keys[0]

        self.widget_open_file.new_layer_name.value = self.generate_layer_name(
            path, self.dataset_key.value
        )

    def _on_path_mode_changed(self, path_mode: str):
        logger.debug("_on_path_mode_changed called!")
        path_mode = _return_value_if_widget(path_mode)
        if path_mode == PathMode.FILE.value:  # file
            self.widget_open_file.path.mode = "r"
            self.widget_open_file.path.label = "File path"
        elif path_mode == PathMode.DIR.value:  # directory case
            self.widget_open_file.path.mode = "d"
            self.widget_open_file.path.label = "Zarr path\n(.zarr)"

    def _on_path_changed(self, path: Path):
        logger.debug("_on_path_changed called!")
        self.path_changed_once = True
        if path.exists():
            self.look_up_dataset_keys(path)
            self.update_stack_layout()

    def _on_refresh_keys_button(self, press: bool):
        logger.debug("_on_refresh_keys_button called!")
        self.look_up_dataset_keys(self.widget_open_file.path.value)

    def _on_dataset_key_changed(self, dataset_key: str):
        logger.debug("_on_dataset_key_changed called!")
        dataset_key = _return_value_if_widget(dataset_key)
        if dataset_key:
            self.widget_open_file.new_layer_name.value = self.generate_layer_name(
                self.widget_open_file.path.value, dataset_key
            )
        self.update_stack_layout()

    def update_stack_layout(self):
        path = self.widget_open_file.path.value
        ext = path.suffix.lower()

        if ext in H5_EXTENSIONS:
            key = self.dataset_key.value
            self.widget_open_file.stack_layout.value = guess_stack_layout(path, key)

        elif ext in ZARR_EXTENSIONS:
            key = self.dataset_key.value
            self.widget_open_file.stack_layout.value = guess_stack_layout(path, key)

        elif ext in TIFF_EXTENSIONS:
            self.widget_open_file.stack_layout.value = guess_stack_layout(path)

        elif ext in PIL_EXTENSIONS:
            self.widget_open_file.stack_layout.value = guess_stack_layout(path)

    def _on_done(self):
        logger.debug("_on_done called!")
        self.look_up_dataset_keys(self.widget_open_file.path.value)

    def _selected_panseg_image(self) -> PanSegImage:
        """Return the PanSegImage of the layer selected in the Details widget."""
        layer = self.widget_details_layer_select.layer.value
        if layer is None:
            raise ValueError("No layer selected.")

        assert isinstance(layer, (Image, Labels)), (
            "Only Image and Labels layers are supported for PanSeg."
            f" Layer was {layer}, type: {type(layer)}"
        )
        return PanSegImage.from_napari_layer(layer)

    @magic_factory(
        call_button="Set Voxel Size",
        voxel_size={
            "label": "Voxel size [um]",
            "tooltip": "Set the voxel size in micrometers.",
        },
    )
    def factory_set_voxel_size(
        self,
        voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Set the voxel size of the selected layer."""
        ps_image = self._selected_panseg_image()
        return schedule_task(
            set_voxel_size_task,
            task_kwargs={
                "image": ps_image,
                "voxel_size": voxel_size,
            },
        )

    @magic_factory(
        call_button="Set Time Spacing",
    )
    def factory_set_t_spacing(self) -> None:
        """Set the time spacing of the selected timeseries layer."""
        ps_image = self._selected_panseg_image()
        if not ps_image.is_timeseries:
            raise ValueError(
                f"Layer {ps_image.name} is not a timeseries, no time spacing to set."
            )

        value = self.t_spacing.value.strip()
        if value == "":
            t_spacing_value: float | None = None
        else:
            try:
                t_spacing_value = float(value)
            except ValueError:
                logger.warning(f"Invalid time spacing {value!r}.")
                return
            if t_spacing_value <= 0:
                logger.warning(f"Time spacing must be positive, got {t_spacing_value}.")
                return

        return schedule_task(
            set_t_spacing_task,
            task_kwargs={
                "image": ps_image,
                "t_spacing": t_spacing_value,
                "t_unit": self.t_unit.value,
            },
        )

    def _wrap_t_spacing(self):
        """Group the time spacing value and its unit in one horizontal row."""
        w = self.widget_set_t_spacing

        t_spacing_d = {
            "label": "Time spacing",
            "widget_type": "LineEdit",
            "options": {
                "tooltip": "Set the time spacing between timepoints in the selected unit.\n"
                "Leave empty to mark the time spacing as unknown.",
            },
            "annotation": str,
            "name": "_t_spacing",
        }
        t_unit_d = {
            "label": "Unit",
            "widget_type": "ComboBox",
            "value": "s",
            "options": {
                "choices": list(TIME_UNIT_CHOICES),
                "tooltip": "Unit of the time spacing.",
            },
            "annotation": str,
            "name": "_t_unit",
        }

        t_spacing: ValueWidget = create_widget(**t_spacing_d)
        t_unit: CategoricalWidget = create_widget(**t_unit_d)
        t_unit.max_width = 80

        combo = Container(
            widgets=[t_spacing, t_unit],
            layout="horizontal",
            labels=False,
            name="t_spacing_combo",
            label="Time spacing",
            gui_only=True,
        )

        self.t_spacing = t_spacing
        self.t_unit = t_unit
        w.insert(0, combo)  # pyright: ignore

    def _on_details_layer_select_changed(self, layer: Optional[Layer]):
        logger.debug(f"_on_details_layer_select_changed called for layer {layer}!")

        if layer is None:
            self.widget_set_voxel_size.hide()
            self.widget_set_t_spacing.hide()
            self.widget_info.hide()
            return

        if (
            not (isinstance(layer, Labels) or isinstance(layer, Image))
            and layer._metadata
        ):
            logger.debug(f"Can't show info for {layer}")
            return

        self.widget_details_layer_select.show()
        self.widget_set_voxel_size.show()
        self.widget_info.show()

        ps_image = PanSegImage.from_napari_layer(layer)
        if ps_image.is_timeseries:
            self.widget_set_t_spacing.show()
        else:
            self.widget_set_t_spacing.hide()

        if ps_image.has_valid_voxel_size():
            voxel_size_formatted = "("
            for vs in ps_image.voxel_size:
                voxel_size_formatted += f"{vs:.2f}, "

            voxel_size_formatted = (
                voxel_size_formatted[:-2] + f") {ps_image.voxel_size.unit}"
            )
        else:
            voxel_size_formatted = "None"

        parts = {
            "shape": f"Shape: {ps_image.shape}",
            "voxels": f"Voxel size: {voxel_size_formatted}",
            "type": f"Type: {ps_image.semantic_type.value}",
            "layout": f"Layout: {ps_image.image_layout.value}",
        }
        str_info = (
            f"{parts['shape']:<30} {parts['voxels']:<30}\n"
            f"{parts['type']:<30} {parts['layout']:<30}"
        )
        if ps_image.is_timeseries:
            t_spacing = ps_image.properties.t_spacing
            t_spacing_formatted = (
                f"{t_spacing:.3g} {ps_image.properties.t_unit}"
                if t_spacing is not None
                else "None"
            )
            str_info += f"\n{f'Time spacing: {t_spacing_formatted}':<30}"

        font = QtGui.QFont("Monospace")
        font.setStyleHint(QtGui.QFont.TypeWriter)

        self.widget_info.value = str_info
        self.widget_info.native.setFont(font)

    def _on_set_voxel_size_layer_done(self):
        logger.debug("_on_set_voxel_size_layer_done called!")

    def update_layer_selection(self, event):
        logger.debug(f"Updating input layer selection: {event.value}, {event.type}")
        layer = event.value
        if layer is None or not layer._metadata:
            return
        all_layers = get_layers(
            [
                SemanticType.RAW,
                SemanticType.PREDICTION,
                SemanticType.SEGMENTATION,
            ]
        )
        self.widget_details_layer_select.layer.choices = all_layers

        if layer in all_layers:
            # trigger refresh even if name has not changed
            self.widget_details_layer_select.layer.value = None
            self.widget_details_layer_select.layer.value = layer
        else:
            logger.debug(f"Can't show info for {layer}")

    @magic_factory(
        call_button=False,
        layer={
            "label": "Layer",
            "tooltip": "Select layer to show its details, and change its voxel size.",
        },
    )
    def factory_details_layer_select(
        self,
        layer: Layer | None = None,
    ):
        pass


class Docs_Container:
    def __init__(self):
        logger.debug("Docs init")
        self.logo_path = (
            Path(__file__).resolve().parent.parent.parent
            / "resources"
            / "header_h_white.png"
        )
        assert self.logo_path.exists(), "Logo not found!"
        self.docs_url = "https://kreshuklab.github.io/panseg/"

    def get_doc_container(self) -> Container:
        logger.debug("get_doc_container called!")
        """Creates a container with a documentation button and a logo."""

        button = PushButton(text="Open Documentation")
        button.changed.connect(self.open_docs)
        hover_text = Label(value="Hover over any field for more details!")
        container = Container(
            widgets=[button, hover_text],
            label=f'<img src="{self.logo_path}" width=250>',
            layout="vertical",
            labels=False,
        )
        container[0].show()
        return Container(widgets=[container], labels=True, layout="horizontal")

    def open_docs(self, button):
        logger.debug("open_docs called!")
        """Open the documentation URL in the default web browser when the button is clicked."""
        webbrowser.open(self.docs_url, new=0, autoraise=True)
        logger.info(f"Docs webpage opened: {self.docs_url}")
        return button
