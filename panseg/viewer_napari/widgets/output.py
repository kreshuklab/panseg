import time
from pathlib import Path
from typing import Optional

from magicgui import magic_factory
from magicgui.widgets import CheckBox, ComboBox, Container, Label
from napari.layers import Image, Labels, Layer
from psygnal import Signal

from panseg import logger
from panseg.core.image import PanSegImage
from panseg.tasks.io_tasks import export_image_task, merge_channels_task
from panseg.tasks.workflow_handler import workflow_handler
from panseg.viewer_napari import log
from panseg.viewer_napari.widgets.utils import add_ps_image_to_viewer, div, get_layers
from panseg.workflow_gui.editor import Workflow_gui


class Batch_Tab:
    def __init__(self, output_tab: Optional["Output_Tab"] = None):
        self.widget_export_workflow = self.factory_export_headless_workflow()
        self.widget_export_workflow.self.bind(self)
        self.widget_export_workflow.hide()

        self.widget_edit_workflow = self.factory_edit_workflow()
        self.widget_edit_workflow.self.bind(self)

        self.widget_export_placeholder = Label(
            value="Export an image before saving the workflow\nfor batch execution!"
        )

        if output_tab:
            output_tab.successful_export.connect(self.toggle_export_vis)

    def get_container(self):
        return Container(
            widgets=[
                div("Export Batch Workflow"),
                self.widget_export_placeholder,
                self.widget_export_workflow,
                div("Edit Batch Workflow"),
                self.widget_edit_workflow,
            ],
            labels=False,
        )

    @magic_factory(
        call_button="Export Workflow",
        directory={
            "label": "Export directory",
            "mode": "d",
            "tooltip": "Select the directory where the workflow will be exported",
        },
        workflow_name={
            "label": "Workflow name",
            "tooltip": "Name of the exported workflow file.",
        },
    )
    def factory_export_headless_workflow(
        self,
        directory: Path = Path.home(),
        workflow_name: str = "headless_workflow.yaml",
    ) -> None:
        """Save the workflow as a yaml file"""

        if not workflow_name.endswith(".yaml"):
            workflow_name = f"{workflow_name}.yaml"

        workflow_path = directory / workflow_name
        workflow_handler.save_to_yaml(path=workflow_path)

        log(f"Workflow saved to {workflow_path}", thread="Export stacks", level="info")

    @magic_factory(
        call_button="Edit a Workflow",
    )
    def factory_edit_workflow(self) -> None:
        log("Starting workflow editor", thread="Workflow")
        Workflow_gui()

    def toggle_export_vis(self):
        logger.debug("toggle export called!")
        self.widget_export_placeholder.hide()
        self.widget_export_workflow.show()


class Output_Tab:
    successful_export = Signal()

    def __init__(self):
        self.widget_export_image = self.factory_export_image()
        self.widget_export_image.self.bind(self)

        self.export_details = [
            self.widget_export_image.directory,
            self.widget_export_image.name_pattern,
            self.widget_export_image.n_channels,
            self.widget_export_image.export_format,
            self.widget_export_image.data_type,
            self.widget_export_image.scale_to_origin,
            self.widget_export_image.key,
            self.widget_export_image.export_mesh,
            self.widget_export_image.close_mesh,
        ]
        self._toggle_export_details_widgets(False)
        self._toggle_key(False)
        self.widget_export_image.image.changed.connect(self._on_images_changed)
        self.widget_export_image.export_format.changed.connect(
            self._on_export_format_changed
        )

        self.additional_layers = Container(
            widgets=[], visible=False, label="Additional channels", labels=True
        )
        self.widget_export_image.insert(4, self.additional_layers)
        self.widget_export_image.n_channels.changed.connect(self._on_n_channels_change)

        self.batch_tab = Batch_Tab(self)

    def get_container(self):
        return Container(
            widgets=[
                div("Output"),
                self.widget_export_image,
                self.batch_tab.get_container(),
            ],
            labels=False,
        )

    @magic_factory(
        call_button="Export Layer",
        image={
            "label": "Layer to export",
            "tooltip": "Select all layer to be exported, and (optional) set a custom file name suffix that will be "
            "appended at end of the layer name.",
        },
        directory={
            "label": "Export directory",
            "mode": "d",
            "tooltip": "Select the directory where the files will be exported",
        },
        n_channels={
            "label": "Number of channels",
            "tooltip": "Add additional layers to the output image",
            "widget_type": "SpinBox",
            "value": 1,
            "min": 1,
        },
        name_pattern={
            "label": "Export name pattern",
            "tooltip": "Pattern for the exported file name. Use {image_name} to include the layer name, "
            "or {file_name} to include the original file name.",
        },
        export_format={
            "label": "Export format",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
            "choices": ["tiff", "h5", "zarr"],
            "tooltip": "Export format, if tiff is selected, each layer will be exported as a separate file. "
            "If h5 is selected, all layers will be exported in a single file.",
        },
        key={
            "label": "Key",
            "tooltip": "Key to be used in the h5 or zarr file.",
        },
        export_mesh={
            "label": "Export Mesh",
            "tooltip": "Convert segmentation to mesh.\nglb is encuraged, "
            "other formats might have missing color, or merge meshes.",
            "choices": ["No", "glb", "obj", "ply"],
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
            "value": "No",
        },
        close_mesh={
            "label": "Close mesh",
            "tooltip": "Whether to close the mesh at the edges of the image.",
        },
        scale_to_origin={
            "label": "Rescale to original resolution",
            "tooltip": "Rescale the image to the original voxel size.",
        },
        data_type={
            "label": "Data Type",
            "choices": ["float32", "uint8", "uint16"],
            "tooltip": "Export datatype (uint16 for segmentation) and all others for images.",
        },
    )
    def factory_export_image(
        self,
        image: Layer | None = None,
        directory: Path = Path.home(),
        n_channels: int = 1,
        name_pattern: str = "{file_name}_export",
        export_format: str = "tiff",
        key: str = "raw",
        export_mesh: str = "No",
        close_mesh: bool = False,
        scale_to_origin: bool = True,
        data_type: str = "uint16",
    ) -> None | bool:
        """Export layers in various formats."""

        if not isinstance(image, (Image, Labels)):
            log(
                "Please select an Image or Labels layers to export!",
                thread="Output",
                level="WARNING",
            )
            return

        ps_image = PanSegImage.from_napari_layer(image)

        # merge channels if necessary
        if n_channels > 1:
            images = [ps_image]
            for selection in self.additional_layers:
                if not isinstance(selection.value, Layer):
                    log(
                        "Choose all channels or adjust the number of channels!",
                        thread="Output",
                        level="WARNING",
                    )
                    return
                images.append(PanSegImage.from_napari_layer(selection.value))
            ps_image = merge_channels_task(
                **{f"image_{i}": image for i, image in enumerate(images)}
            )
        if export_mesh == "No":
            export_mesh = None

        timer = time.time()
        log("export_image_task started", thread="Output", level="info")
        export_image_task(
            image=ps_image,
            export_directory=directory,
            name_pattern=name_pattern,
            key=key,
            scale_to_origin=scale_to_origin,
            export_format=export_format,
            data_type=data_type,
            export_mesh=export_mesh,
            close_mesh=close_mesh,
        )
        timer = time.time() - timer
        log(
            f"export_image_task finished in {timer:.2f} seconds",
            thread="Output",
            level="info",
        )
        self.successful_export.emit(True)

        return

    def _toggle_export_details_widgets(self, show: bool):
        for widget in self.export_details:
            if show:
                widget.show()
            else:
                widget.hide()

    def _toggle_key(self, show: bool):
        if not show or self.widget_export_image.export_format.value == "tiff":
            self.widget_export_image.key.hide()
            return
        self.widget_export_image.key.show()

    def _on_images_changed(self, image: Image | Labels | None):
        if image is None:
            self._toggle_export_details_widgets(False)
            self._toggle_key(False)
            return

        # Multichannel not allowed for segmentations
        if isinstance(image, Labels):
            self.widget_export_image.n_channels.value = 1
            self.widget_export_image.n_channels.enabled = False
            self.widget_export_image.export_mesh.enabled = True
            self.widget_export_image.close_mesh.enabled = True
        else:
            self.widget_export_image.n_channels.enabled = True
            self.widget_export_image.export_mesh.value = "No"
            self.widget_export_image.export_mesh.enabled = False
            self.widget_export_image.close_mesh.enabled = False

        if isinstance(image, Labels) or isinstance(image, Image):
            self._toggle_export_details_widgets(True)
            self._toggle_key(True)
            self.widget_export_image.key.value = (
                "raw" if isinstance(image, Image) else "segmentation"
            )

            if (
                isinstance(image, Labels)
                and self.widget_export_image.data_type.value == "float32"
            ):
                log(
                    "Data type float32 is not supported for Labels layers, changing to uint16",
                    thread="Export stacks",
                    level="warning",
                )
                self.widget_export_image.data_type.value = "uint16"
            return

        raise ValueError(
            "Only Image and Labels layers are supported for PanSeg export."
        )

    def _on_export_format_changed(self, export_format: str):
        self._toggle_key(export_format != "tiff")

    def _on_n_channels_change(self, n_channels: int):
        self.additional_layers.clear()
        for ch in range(n_channels - 1):
            self.additional_layers.append(
                ComboBox(
                    choices=self.widget_export_image.image.choices,
                    label=f"{ch + 1}",
                    tooltip="Additional layer to merge with main output "
                    "into separate channel.",
                )
            )
        if len(self.additional_layers) > 0:
            self.additional_layers.show()
        else:
            self.additional_layers.hide()

    def update_layer_selection(self, event):
        """Updates layer drop-down menus"""
        logger.debug(f"Updating output layer selection: {event.value}, {event.type}")
        all_layers = get_layers()

        self.widget_export_image.image.choices = all_layers
