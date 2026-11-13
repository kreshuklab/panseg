from pathlib import Path
from typing import Literal

import torch
from magicgui import magic_factory, widgets
from magicgui.types import Undefined
from magicgui.widgets import (
    Container,
    FileEdit,
    Label,
    ProgressBar,
    PushButton,
)
from napari.layers import Image, Labels

from panseg import PATH_PANSEG_MODELS, logger
from panseg.core.image import ImageLayout, PanSegImage, SemanticType
from panseg.core.zoo import model_zoo
from panseg.functionals.training.biio import (
    PANSEG_CITATION,
    parse_authors,
    parse_citations,
)
from panseg.functionals.training.model import UNet2D, UNet3D
from panseg.functionals.training.train import find_h5_files
from panseg.io.h5 import read_h5_shape, read_h5_voxel_size
from panseg.tasks.training_tasks import unet_training_task
from panseg.viewer_napari import log
from panseg.viewer_napari.widgets.prediction import Prediction_Widgets
from panseg.viewer_napari.widgets.utils import (
    Help_text,
    div,
    get_layers,
    schedule_task,
)

NONE_LICENSE = "(none)"
LICENSE_CHOICES = [
    NONE_LICENSE,
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Apache-2.0",
    "GPL-3.0-only",
    "LGPL-3.0-only",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-NC-4.0",
    "Unlicense",
]


class Training_Tab:
    def __init__(self, prediction_tab: Prediction_Widgets | None):
        self.prediction_tab = prediction_tab
        self.in_shape = None
        self.out_shape = None

        # Constants
        self.ALL_CUDA_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.MPS = ["mps"] if torch.backends.mps.is_available() else []
        self.ALL_DEVICES = self.ALL_CUDA_DEVICES + self.MPS + ["cpu"]
        self.CUSTOM = "Custom"

        self.previous_z_patch_size = 16
        # Dimensionality required by the selected pretrained model, if any
        self.pretrained_dimensionality: Literal["2D", "3D"] | None = None

        # Section visibility state
        self.train_data_open = True
        self.meta_data_open = False

        # @@@@@ Training Data section @@@@@
        self.widget_unet_training_data = self.factory_unet_training_data()
        self.widget_unet_training_data.self.bind(self)

        # @@@@@ Model section @@@@@
        self.widget_unet_model = self.factory_unet_model()
        self.widget_unet_model.self.bind(self)

        # @@@@@ Meta Data section @@@@@
        self.widget_unet_metadata = self.factory_unet_metadata()
        self.widget_unet_metadata.self.bind(self)

        # @@@@@ Start Training @@@@@
        self.widget_unet_training = self.factory_unet_training()
        self.widget_unet_training.self.bind(self)

        # section dividers (kept at container level so they span full width)
        self.div_train_data = div("Training Data")
        self.div_model = div("Model")
        self.div_meta_data = div("Meta Data")
        self.div_start_training = div()

        # multi-channel container
        self.widget_unet_training_data.channels[1].enabled = False
        self.additional_inputs = Container(
            widgets=[], visible=False, labels=True, label="Additional Inputs"
        )
        self.widget_unet_training_data.insert(
            self.widget_unet_training_data.index(
                self.widget_unet_training_data.segmentation
            ),
            self.additional_inputs,
        )
        self.widget_unet_training_data.channels.changed.connect(
            self.update_additional_inputs
        )

        self.widget_unet_training_data.from_disk.changed.connect(
            self._on_from_disk_change
        )

        self.widget_unet_model.device._default_choices = self.ALL_DEVICES
        self.widget_unet_model.device.reset_choices()
        self.widget_unet_model.device.value = self.ALL_DEVICES[0]

        self.widget_unet_model.pretrained._default_choices = lambda _: (
            model_zoo.list_models()
        )
        self.widget_unet_model.pretrained.changed.connect(self._on_pretrained_changed)
        self.widget_unet_model.pretrained.reset_choices()

        self.widget_unet_model.dimensionality.changed.connect(
            self._on_dimensionality_change
        )

        self.widget_unet_metadata.modality.changed.connect(
            self._on_custom_modality_change
        )
        self.widget_unet_metadata.output_type.changed.connect(
            self._on_custom_output_type_change
        )
        self.widget_unet_metadata.modality._default_choices = (
            model_zoo.get_unique_modalities() + [self.CUSTOM]
        )
        self.widget_unet_metadata.modality.reset_choices()
        self.widget_unet_metadata.output_type._default_choices = (
            model_zoo.get_unique_output_types() + [self.CUSTOM]
        )
        self.widget_unet_metadata.output_type.reset_choices()
        self.widget_unet_training_data.dataset.changed.connect(self._on_dataset_change)
        self.widget_unet_training_data.image.changed.connect(self._on_image_change)
        self.widget_unet_training_data.segmentation.changed.connect(
            self._on_segmentation_change
        )

        # @@@@@ Hide/Show buttons for the collapsible sections @@@@@
        self.widget_show_train_data = PushButton(label="Show")
        self.widget_show_train_data.clicked.connect(
            lambda *args: self.toggle_visibility_train_data(True)
        )

        self.widget_show_metadata = PushButton(label="Show")
        self.widget_show_metadata.clicked.connect(
            lambda *args: self.toggle_visibility_metadata(True)
        )

        # @@@@@ Help Text @@@@@
        help_text = "<strong>UNet Training:</strong> Train a custom segmentation model on your own data."
        self.help_text_container = Help_text()
        self.tab_help = self.help_text_container.get_doc_container(
            help_text,
            sub_url="chapters/panseg_interactive_napari/unet_training/",
        )

        self.widget_show_train_data.hide()
        self.toggle_visibility_metadata(False)

        self.widget_info = Label(value=f"Model dir: {PATH_PANSEG_MODELS}")
        self._automatic_channel_change = False

        # magicgui only unifies label widths on post-init widget insertions or
        # label changes; sections without one are left with unaligned labels
        self.widget_unet_training_data._unify_label_widths()
        self.widget_unet_model._unify_label_widths()
        self.widget_unet_metadata._unify_label_widths()

    def get_container(self):
        return Container(
            widgets=[
                self.tab_help,
                self.div_train_data,
                self.widget_show_train_data,
                self.widget_unet_training_data,
                self.div_model,
                self.widget_unet_model,
                self.div_meta_data,
                self.widget_show_metadata,
                self.widget_unet_metadata,
                self.div_start_training,
                self.widget_unet_training,
                self.widget_info,
            ],
            labels=False,
        )

    def toggle_visibility_train_data(self, visible: bool):
        """Toggles visibility of the training data section"""
        logger.debug(f"toggle_visibility_train_data called with {visible}")
        self.train_data_open = visible
        if visible:
            self.widget_show_train_data.hide()
            self.toggle_visibility_metadata(False)
            self.div_model.show()
            self.widget_unet_model.show()
            self.widget_unet_training_data.show()
            self._on_from_disk_change(self.widget_unet_training_data.from_disk.value)
            self.widget_unet_training_data.channels.show()
            self.widget_unet_training_data.resolution.show()
        else:
            self.widget_unet_training_data.hide()
            self.widget_show_train_data.show()

    def toggle_visibility_metadata(self, visible: bool):
        """Toggles visibility of the meta data section"""
        logger.debug(f"toggle_visibility_metadata called with {visible}")
        self.meta_data_open = visible
        if visible:
            self.widget_show_metadata.hide()
            self.toggle_visibility_train_data(False)
            self.div_model.hide()
            self.widget_unet_model.hide()
            self.widget_unet_metadata.show()
            self._on_custom_output_type_change(
                self.widget_unet_metadata.output_type.value
            )
            self._on_custom_modality_change(self.widget_unet_metadata.modality.value)
        else:
            self.widget_unet_metadata.hide()
            self.widget_show_metadata.show()

    @magic_factory(
        call_button=False,
        from_disk={
            "label": "Train from",
            "value": "Disk",
            "widget_type": "RadioButtons",
            "choices": ["Disk", "Current Data"],
            "orientation": "horizontal",
            "tooltip": "Whether to train from a dataset on disk or from loaded layers in Napari.",
        },
        dataset={
            "label": "Dataset",
            "value": Path.home(),
            "mode": "d",
            "tooltip": "Dataset directory. It must contain a `train` and a `val`\n"
            "directory, each containing h5 files.\n"
            "Input/Output keys must be `raw` and `label` respectively.",
        },
        image={
            "label": "Training input",
            "tooltip": "e.g. raw microscopy image",
            "visible": False,
        },
        segmentation={
            "label": "Training segmentation",
            "tooltip": "The to the training input corresponding segmentation.",
            "visible": False,
        },
        channels={
            "label": "In and Out Channels",
            "value": (1, 1),
            "tooltip": "Number of input and output channels",
            "widget_type": "TupleEdit",
            "enabled": True,
        },
        resolution={
            "label": "Data Resolution",
            "value": [1.0, 1.0, 1.0],
            "widget_type": "TupleEdit",
            "tooltip": "Voxel size in um of the training data.\n"
            "Is initialized correctly from the chosen data if possible.",
            "enabled": False,
        },
    )
    def factory_unet_training_data(
        self,
        from_disk: str,
        dataset: Path | None,
        image: Image | None,
        segmentation: Labels | None,
        channels,
        resolution,
    ) -> None:
        pass

    @magic_factory(
        call_button=False,
        dimensionality={
            "label": "Dimensionality",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
            "value": "3D",
            "choices": ["3D", "2D"],
            "tooltip": "Train a 3D unet or a 2D unet.\n"
            "A 2D unet is trained per z-slice on 3D (zyx/czyx) data,\n"
            "with the patch size in z fixed to 1.\n"
            "Fixed to the architecture of the selected pretrained model.",
            "enabled": False,
            # only shown when 2D per-slice vs 3D training is a real choice,
            # i.e. for volumetric (zyx/czyx) data (see update_dimensionality)
            "visible": False,
        },
        pretrained={
            "label": "Pretrained model",
            "tooltip": "Optionally select an existing model to retrain.\n"
            "Hover over the name to show the model description.\n"
            "Leave empty to create a new model.",
            "choices": [None],
            "value": None,
        },
        feature_maps={
            "label": "Feature dimensions",
            "widget_type": "ListEdit",
            "value": [16],
            "tooltip": "Number of feature maps at each level of the encoder.\n"
            "If it's one number, the number of feature maps is given by\n"
            "the geometric progression: f_maps ^ k, k=1,2,3,4\n"
            "Can't be modified for pretrained models.",
        },
        patch_size={
            "label": "Patch size",
            "value": [16, 64, 64],
            "widget_type": "TupleEdit",
            "tooltip": "",
        },
        max_num_iters={
            "label": "Max iterations",
            "value": 100,
            "max": 100000000,
            "min": 1,
            "tooltip": "Maximum number of iterations after which the training\n"
            "will be stopped. Stops earlier if the accuracy converges.",
        },
        device={
            "label": "Device",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
        },
    )
    def factory_unet_model(
        self,
        dimensionality,
        pretrained: str | None,
        feature_maps,
        patch_size,
        max_num_iters: int,
        device,
    ) -> None:
        pass

    @magic_factory(
        call_button=False,
        model_name={
            "label": "Model name",
            "value": "",
            "tooltip": "How your new model should be called.\n"
            "Can't be the name of an existing model.",
        },
        description={
            "label": "Description",
            "value": "A model trained by the user.",
            "tooltip": "Model description will be saved alongside the model.",
        },
        modality={
            "label": "Microscopy modality",
            "tooltip": "Modality of the model (e.g. confocal, light-sheet ...).",
            "widget_type": "ComboBox",
            "value": None,
        },
        custom_modality={
            "label": "Custom modality",
            "value": Undefined,
            "visible": False,
        },
        output_type={
            "label": "Prediction type",
            "widget_type": "ComboBox",
            "tooltip": "Type of prediction (e.g. cell boundaries prediction or nuclei...).",
            "value": None,
        },
        custom_output_type={
            "label": "Custom type",
            "value": Undefined,
            "visible": False,
        },
        authors={
            "label": "Authors",
            "widget_type": "TextEdit",
            "tooltip": "One author per line, either 'Name' or 'Name <email>'.",
        },
        additional_citations={
            "label": "Additional citations",
            "widget_type": "TextEdit",
            "tooltip": "One citation per line: '<DOI or URL> [free text]'.\n"
            "Example: 10.1234/abc.def Smith, J. et al. Some result.\n"
            f"The PanSeg citation ({PANSEG_CITATION.doi}) is always included.",
        },
        license={
            "label": "License",
            "widget_type": "ComboBox",
            "choices": LICENSE_CHOICES,
            "value": NONE_LICENSE,
        },
        documentation={
            "label": "Documentation",
            "widget_type": "TextEdit",
            "tooltip": "Markdown documentation for the model.\n"
            "Saved as README.md next to the model.",
        },
    )
    def factory_unet_metadata(
        self,
        model_name: str,
        description: str,
        modality: str | None,
        custom_modality: str,
        output_type: str | None,
        custom_output_type: str,
        authors: str,
        additional_citations: str,
        license: str,
        documentation: str,
    ) -> None:
        pass

    @magic_factory(
        call_button="Start Training",
        pbar={"label": "Training in progress", "max": 0, "min": 0, "visible": False},
    )
    def factory_unet_training(self, pbar: ProgressBar | None) -> None:
        """Train a boundary prediction unet from the current section values."""
        self._run_training(
            from_disk=self.widget_unet_training_data.from_disk.value,
            dataset=self.widget_unet_training_data.dataset.value,
            image=self.widget_unet_training_data.image.value,
            segmentation=self.widget_unet_training_data.segmentation.value,
            channels=self.widget_unet_training_data.channels.value,
            resolution=self.widget_unet_training_data.resolution.value,
            dimensionality=self.widget_unet_model.dimensionality.value,
            pretrained=self.widget_unet_model.pretrained.value,
            feature_maps=self.widget_unet_model.feature_maps.value,
            patch_size=self.widget_unet_model.patch_size.value,
            max_num_iters=self.widget_unet_model.max_num_iters.value,
            device=self.widget_unet_model.device.value,
            model_name=self.widget_unet_metadata.model_name.value,
            description=self.widget_unet_metadata.description.value,
            modality=self.widget_unet_metadata.modality.value,
            custom_modality=self.widget_unet_metadata.custom_modality.value,
            output_type=self.widget_unet_metadata.output_type.value,
            custom_output_type=self.widget_unet_metadata.custom_output_type.value,
            authors=self.widget_unet_metadata.authors.value,
            additional_citations=self.widget_unet_metadata.additional_citations.value,
            license=self.widget_unet_metadata.license.value,
            documentation=self.widget_unet_metadata.documentation.value,
            pbar=pbar,
        )

    def _run_training(
        self,
        # data
        from_disk: str,
        dataset: Path | None,
        image: Image | None,
        segmentation: Labels | None,
        channels,
        resolution,
        dimensionality,
        # model
        pretrained: str | None,
        feature_maps,
        patch_size,
        max_num_iters: int,
        device,
        # metadata
        model_name: str,
        description: str,
        modality: str | None,
        custom_modality: str,
        output_type: str | None,
        custom_output_type: str,
        # fair metadata
        authors: str = "",
        additional_citations: str = "",
        license: str = NONE_LICENSE,
        documentation: str = "",
        pbar: ProgressBar | None = None,
    ) -> None:
        """Train a boundary prediction unet"""
        if from_disk == "Disk":
            image = None
            segmentation = None
            if dataset is None:
                log("Please choose a dataset to load!", thread="train_gui")
                return

        else:
            dataset = None
            if image is None or segmentation is None:
                log(
                    "Please choose a raw image and a segmentation to train!",
                    thread="train_gui",
                )
                return
            if len(self.additional_inputs) > 0:
                image = [image] + [i.value for i in self.additional_inputs]

        if modality is None or output_type is None:
            log("Choose a modality and a prediction type!", thread="train_gui")
            return
        if modality == self.CUSTOM:
            modality = custom_modality
            if len(modality) == 0:
                log("Custom modality can't be empty!", thread="train_gui")
                return
        if output_type == self.CUSTOM:
            output_type = custom_output_type
            if len(output_type) == 0:
                log("Custom prediction type can't be empty!", thread="train_gui")
                return
        if len(model_name) == 0:
            log("Please choose a model name!", thread="train_gui")
            return
        if len(model_name) < 5:
            log("Please choose a longer model name!", thread="train_gui")
            return

        try:
            parse_authors(authors)
            parse_citations(additional_citations)
        except ValueError as e:
            log(f"Invalid model metadata: {e}", thread="train_gui", level="ERROR")
            return

        # Enable geometric progression by setting type to int
        if len(feature_maps) == 1:
            feature_maps = feature_maps[0]

        checkpoint_dir = PATH_PANSEG_MODELS / model_name
        if checkpoint_dir.exists():
            log(
                "Model exists already! Please choose another name.",
                thread="train_gui",
                level="WARNING",
            )
            return

        layer_order = "bcr"
        pre_model_path = None
        if pretrained is not None:
            model, model_config, pre_model_path = model_zoo.get_model_by_name(
                pretrained
            )
            pretrained_lo = model_config["layer_order"]
            if pretrained_lo != layer_order:
                logger.debug(
                    f"Pretrained model has non-default layer_order, changing from {layer_order}"
                )
                layer_order = pretrained_lo
        logger.info(f"Model architecture: {layer_order}")

        widgets_to_reset = [
            self.widget_unet_model.pretrained,
        ]
        if self.prediction_tab:
            widgets_to_reset.extend(
                [
                    self.prediction_tab.widget_unet_prediction.model_name,
                    self.prediction_tab.model_filters[1],
                    self.prediction_tab.model_filters[2],
                ]
            )

        log("Starting training task", thread="train_gui")
        schedule_task(
            task=unet_training_task,
            task_kwargs={
                "dataset_dir": dataset,
                "image": image,
                "segmentation": segmentation,
                "model_name": model_name,
                "in_channels": channels[0],
                "out_channels": channels[1],
                "feature_maps": feature_maps,
                "patch_size": patch_size,
                "max_num_iters": max_num_iters,
                "dimensionality": dimensionality,
                "device": device,
                "modality": modality,
                "output_type": output_type,
                "description": description,
                "resolution": resolution,
                "pre_trained": pre_model_path,
                "layer_order": layer_order,
                "authors": authors,
                "additional_citations": additional_citations,
                "license": None if license == NONE_LICENSE else license,
                "documentation": documentation,
                "widgets_to_reset": widgets_to_reset,
                "_pbar": pbar,
                "_to_hide": [self.widget_unet_training.call_button],
            },
        )

    def _on_from_disk_change(self, from_disk: str):
        logger.debug(f"_on_from_disk_change called: {from_disk}")
        if not self.train_data_open:
            return
        if from_disk == "Disk":
            self.widget_unet_training_data.image.hide()
            self.widget_unet_training_data.segmentation.hide()
            self.widget_unet_training_data.dataset.show()
            self._on_dataset_change(self.widget_unet_training_data.dataset.value)
        else:
            self.widget_unet_training_data.dataset.hide()
            self.widget_unet_training_data.image.show()
            self.widget_unet_training_data.segmentation.show()
            self.widget_unet_training_data.channels[0].enabled = True
            self.update_additional_inputs(self.widget_unet_training_data.channels.value)

    def update_layer_selection(self, event):
        """Updates layer drop-down menus"""
        logger.debug(
            f"Updating segmentation layer selection: {event.value}, {event.type}"
        )
        raws = get_layers(SemanticType.RAW)
        segmentations = get_layers(SemanticType.SEGMENTATION)

        self.widget_unet_training_data.image.choices = raws
        self.widget_unet_training_data.segmentation.choices = segmentations

        if raws and segmentations:
            self.widget_unet_training_data.from_disk.value = "Current Data"

    def _on_dimensionality_change(self, dimensionality: Literal["3D", "2D"]):
        """Update patch size according to chosen dimensionality.

        2D models are trained per z-slice on zyx/czyx data, so the patch size
        in z is frozen to 1 for "2D" and restored for "3D".
        """
        logger.debug(f"_on_dimensionality_change called with {dimensionality}")

        # Patch size:
        z_patch = self.widget_unet_model.patch_size[0]
        if dimensionality == "2D":
            if z_patch.value != 1:  # guard multiple executions
                self.previous_z_patch_size = z_patch.value
            z_patch.value = 1
            z_patch.enabled = False
        else:
            z_patch.value = self.previous_z_patch_size
            z_patch.enabled = True
        if self.widget_unet_training_data.from_disk.value == "Disk":
            self.update_channels()

    def update_channels(self):
        """Updates the number of input and output channels

        Determination based on self.in_shape and self.out_shape, which are set
        on data change.
        This function would support output channels, but currently the training
        does not, therefore the output must not have a channel axis.
        The channels do not depend on the chosen dimensionality: zyx/czyx data
        is spatial in z for "3D" training as well as for "2D" training, where a
        2D model is trained per z-slice.
        """
        self._automatic_channel_change = True
        try:
            logger.debug("update_channels called")
            ch = self.widget_unet_training_data.channels
            self.widget_unet_training_data.channels[0].enabled = False
            self.widget_unet_training_data.channels[1].enabled = False

            if self.in_shape is None or self.out_shape is None:
                return
            if len(self.in_shape) == 2:
                if len(self.out_shape) == 2:
                    ch.value = (1, 1)
                    self.widget_unet_training_data.channels[0].enabled = True
                elif len(self.out_shape) == 3:
                    ch.value = (1, self.out_shape[0])
                    raise ValueError("No channels in output supported!")
                elif len(self.out_shape) == 4:
                    raise ValueError("For 2D input only 2D output is supported!")
                else:
                    raise ValueError(f"Output shape {self.out_shape} not supported!")

            elif len(self.in_shape) == 3:
                if len(self.out_shape) == 2:
                    # must be 2D!
                    ch.value = (self.in_shape[0], 1)

                elif len(self.out_shape) == 3:
                    # ZYX input and ZYX output: z is spatial for "3D" training
                    # as well as for "2D" training per z-slice
                    ch.value = (1, 1)
                    self.widget_unet_training_data.channels[0].enabled = True

                elif len(self.out_shape) == 4:
                    # must be 3D!
                    ch.value = (1, self.out_shape[0])
                    raise ValueError("No channels in output supported!")
                else:
                    raise ValueError(f"Output shape {self.out_shape} not supported!")

            elif len(self.in_shape) == 4:
                if len(self.out_shape) == 2:
                    raise ValueError("For 3D input only 3D output is supported!")
                elif len(self.out_shape) == 3:
                    ch.value = (self.in_shape[0], 1)
                elif len(self.out_shape) == 4:
                    ch.value = (self.in_shape[0], self.out_shape[0])
                    raise ValueError("No channels in output supported!")
                else:
                    raise ValueError(f"Output shape {self.out_shape} not supported!")
            else:
                raise ValueError(f"Input shape {self.in_shape} not supported!")
            logger.debug(f"Determined channels: {ch.value}")
        except ValueError as e:
            log(f"Error: {e}", thread="training", level="ERROR")
        finally:
            self._automatic_channel_change = False

    def update_additional_inputs(self, channels: tuple[int, int]):
        logger.debug(
            f"update_additional_inputs called: {channels}, automatic: {self._automatic_channel_change}"
        )
        # If the channel change was automatic, do nothing
        if self._automatic_channel_change:
            return
        if not self.train_data_open:
            return
        self.additional_inputs.clear()
        if channels[0] <= 1:
            self.additional_inputs.hide()
        else:
            self.additional_inputs.show()

        for i in range(channels[0] - 1):
            if self.widget_unet_training_data.from_disk.value == "Disk":
                self.additional_inputs.hide()
                self.additional_inputs.append(
                    FileEdit(mode="d", tooltip=f"Additional input channel {i + 1}")
                )
            else:
                self.additional_inputs.append(
                    widgets.create_widget(
                        annotation=Image | None,
                    )
                )

    def _on_custom_modality_change(self, modality: str | None):
        logger.debug(f"_on_custom_modality_change called: {modality}")
        if not self.meta_data_open:
            return
        if modality == self.CUSTOM:
            self.widget_unet_metadata.custom_modality.show()
        else:
            self.widget_unet_metadata.custom_modality.hide()

    def _on_custom_output_type_change(self, output_type: str | None):
        logger.debug(f"_on_custom_output_type_change called: {output_type}")
        if not self.meta_data_open:
            return
        if output_type == self.CUSTOM:
            self.widget_unet_metadata.custom_output_type.show()
        else:
            self.widget_unet_metadata.custom_output_type.hide()

    def _on_dataset_change(self, dataset_dir: Path | None):
        """Updates resolution, dimensionality and channels"""
        if dataset_dir is None or not dataset_dir.exists() or not dataset_dir.is_dir():
            self.in_shape, self.out_shape = None, None
            self.update_dimensionality()
            return
        if not all((dataset_dir / d).exists() for d in ["train", "val"]):
            return

        logger.debug(f"_on_dataset_change called: {dataset_dir}")

        h5s = find_h5_files(dataset_dir / "train")
        if len(h5s) < 1:
            logger.debug("_on_dataset_change: no h5 files found")
            return

        # get resolution
        voxel_size = read_h5_voxel_size(h5s[0], "raw").voxels_size
        if voxel_size is None:
            voxel_size = (1.0, 1.0, 1.0)
            self.widget_unet_training_data.resolution.enabled = True
        else:
            self.widget_unet_training_data.resolution.enabled = False

        self.widget_unet_training_data.resolution.value = voxel_size
        logger.debug(f"Resolution of training data: {voxel_size}")

        # get channels
        self.in_shape = read_h5_shape(h5s[0], key="raw")
        self.out_shape = read_h5_shape(h5s[0], key="label")
        logger.debug(
            f"In/Out shape determined from dataset: {self.in_shape}, {self.out_shape}"
        )
        self.update_dimensionality()

    def _on_image_change(self, image: Image | None):
        """Update resolution, dimensionality and input channels on image change"""
        self.in_shape = None
        if image is None:
            return
        pl_image = PanSegImage.from_napari_layer(image)
        voxel_size = pl_image.voxel_size.voxels_size
        if voxel_size is None:
            voxel_size = (1.0, 1.0, 1.0)
            self.widget_unet_training_data.resolution.enabled = True
        else:
            self.widget_unet_training_data.resolution.enabled = False
        self.widget_unet_training_data.resolution.value = voxel_size

        ch_dim = pl_image.channel_axis
        if ch_dim and pl_image.image_layout == ImageLayout.ZCYX:
            raise ValueError("Only CZYX image layout supported for 4D training.")
        self.in_shape = pl_image.shape
        self.update_dimensionality()

    def _on_segmentation_change(self, seg: Labels):
        """Update resolution, dimensionality and output channels on image change"""
        self.out_shape = None
        if seg is None:
            return
        pl_image = PanSegImage.from_napari_layer(seg)
        ch_dim = pl_image.channel_axis
        if ch_dim:
            raise ValueError("No channels in output supported.")
        self.out_shape = pl_image.shape
        self.update_dimensionality()

    def update_dimensionality(self):
        """Updates and enables/disables the dimensionality input

        This should be called whenever the input is changed, after
        setting self.in_shape and self.out_shape.

        For 2D (yx/cyx) data only "2D" training is possible. For volumetric
        (zyx/czyx) data the user can choose between "3D" training and "2D"
        training of a 2D model per z-slice (patch z-extent 1), so the choice
        is shown and enabled in that case (and if the data shape is unknown),
        with "3D" as default. Otherwise it is hidden and set automatically.

        A selected pretrained model fixes the dimensionality.
        Triggers _on_dimensionality_change, which calls update_channels.
        """
        logger.debug(
            f"update_dimensionality called, in/out: {self.in_shape}, {self.out_shape}"
        )
        dimensionality = self.widget_unet_model.dimensionality
        model_dimensionality = self.pretrained_dimensionality

        if self.in_shape is None or self.out_shape is None:
            # Data shape unknown, e.g. training from a dataset directory whose
            # files could not be read (yet) -> dimensionality not derivable
            dimensionality.visible = True
            if model_dimensionality is None:
                dimensionality.enabled = True
            else:
                dimensionality.value = model_dimensionality
                dimensionality.enabled = False
            dimensionality.changed.emit(dimensionality.value)
            return
        if len(self.in_shape) == 2:
            # 2D (yx/cyx) data: only 2D training possible
            dimensionality.visible = False
            dimensionality.enabled = False
            dimensionality.value = "2D"
            if len(self.out_shape) > 3:
                raise ValueError(
                    f"Input/Output shapes not supported: {self.in_shape}, {self.out_shape}"
                )
            if model_dimensionality == "3D":
                self._log_dimensionality_mismatch()

        elif len(self.in_shape) == 3:
            if len(self.out_shape) == 2:
                # CYX raw + YX label: 2D data, only 2D training possible
                dimensionality.visible = False
                dimensionality.enabled = False
                dimensionality.value = "2D"
                if model_dimensionality == "3D":
                    self._log_dimensionality_mismatch()
            elif len(self.out_shape) == 3:
                # ZYX data: 3D training or 2D training per z-slice
                dimensionality.visible = True
                if model_dimensionality is None:
                    dimensionality.value = "3D"
                    dimensionality.enabled = True
                else:
                    dimensionality.value = model_dimensionality
                    dimensionality.enabled = False
            elif len(self.out_shape) == 4:
                # ZYX raw + CZYX label: output channels not supported
                dimensionality.visible = False
                dimensionality.enabled = False
                dimensionality.value = "3D"

        elif len(self.in_shape) == 4:
            if len(self.out_shape) < 3:
                raise ValueError(
                    f"Input/Output shapes not supported: {self.in_shape}, {self.out_shape}"
                )
            if len(self.out_shape) == 3:
                # CZYX data: 3D training or 2D training per z-slice
                dimensionality.visible = True
                if model_dimensionality is None:
                    dimensionality.value = "3D"
                    dimensionality.enabled = True
                else:
                    dimensionality.value = model_dimensionality
                    dimensionality.enabled = False
            else:
                # CZYX raw + CZYX label: output channels not supported
                dimensionality.visible = False
                dimensionality.enabled = False
                dimensionality.value = "3D"

        if (not 1 < len(self.in_shape) < 5) or (not 1 < len(self.out_shape) < 5):
            raise ValueError(
                f"Input/Output shapes not supported: {self.in_shape}, {self.out_shape}"
            )
        # update_channels called from setting dimensionality
        dimensionality.changed.emit(dimensionality.value)

    def _log_dimensionality_mismatch(self) -> None:
        """Log that the pretrained model does not fit the chosen data."""
        log(
            "Model incompatible with chosen data!\n"
            f"Model is a {self.pretrained_dimensionality} model, but the data "
            "only supports 2D training",
            thread="training",
            level="ERROR",
        )

    def _on_pretrained_changed(self, model_name: str | None):
        logger.debug(f"_on_model_name_changed called: {model_name}")

        if model_name is None:
            self.description = "No description available for this model."
            self.widget_unet_model.feature_maps.enabled = True
            self.pretrained_dimensionality = None
            self.update_dimensionality()
        else:
            self.widget_unet_model.feature_maps.enabled = False
            model, model_config, pre_model_path = model_zoo.get_model_by_name(
                model_name
            )
            logger.info(f"Selected model config: {model_config}")
            if isinstance(model_config["f_maps"], list):
                self.widget_unet_model.feature_maps.value = model_config["f_maps"]
            else:
                self.widget_unet_model.feature_maps.value = [model_config["f_maps"]]

            self.description = model_zoo.get_model_description(model_name)

            if isinstance(model, UNet2D):
                self.pretrained_dimensionality = "2D"
            elif isinstance(model, UNet3D):
                self.pretrained_dimensionality = "3D"
            else:
                self.pretrained_dimensionality = None
            # Fix the dimensionality to the model architecture; this only logs
            # an incompatibility if the chosen data cannot support it.
            self.update_dimensionality()

            model_channels = (model_config["in_channels"], model_config["out_channels"])
            if model_channels != self.widget_unet_training_data.channels.value:
                log(
                    "Model incompatible chosen data!\nModel channels: "
                    f"{model_channels}\nData channels: "
                    f"{self.widget_unet_training_data.channels.value}",
                    thread="training",
                    level="ERROR",
                )
        self.widget_unet_model.pretrained.tooltip = (
            "Select an existing model to retrain. Current model description:"
            f"\n\n{self.description}"
        )
