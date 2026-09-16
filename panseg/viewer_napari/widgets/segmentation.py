from typing import Optional

from magicgui import magic_factory
from magicgui.widgets import Container, ProgressBar, PushButton
from napari.layers import Image, Labels, Layer

from panseg import logger
from panseg.core.image import ImageLayout, PanSegImage, SemanticType
from panseg.tasks.segmentation_tasks import (
    aio_watershed_task,
    clustering_segmentation_task,
    dt_watershed_task,
    lmc_segmentation_task,
)
from panseg.viewer_napari import log
from panseg.viewer_napari.widgets.prediction import Prediction_Widgets
from panseg.viewer_napari.widgets.utils import div, get_layers, schedule_task

AGGLOMERATION_MODES = [
    ("GASP", "gasp"),
    ("MutexWS", "mutex_ws"),
    ("MultiCut", "multicut"),
    ("LiftedMultiCut", "lmc"),
]


class Segmentation_Tab:
    def __init__(self):
        self.STACKED = [
            ("2D Watershed", True),
            ("3D Watershed", False),
        ]
        # @@@@@ Layer selector @@@@@
        self.widget_layer_select = self.factory_layer_select()
        self.widget_layer_select.self.bind(self)
        self.widget_layer_select.prediction.changed.connect(self._on_prediction_change)
        self.widget_layer_select.append(
            Container(widgets=[], visible=False, labels=True, label="Additional Inputs")
        )
        self.widget_layer_select[-1].margins = (0, 0, 0, 0)

        # @@@@@ agglomeration @@@@@
        self.widget_agglomeration = self.factory_agglomeration()
        self.widget_agglomeration.self.bind(self)
        self.widget_agglomeration.mode._default_choices = AGGLOMERATION_MODES
        self.widget_agglomeration.mode.reset_choices()
        self.widget_agglomeration.mode.value = AGGLOMERATION_MODES[0][1]
        self.widget_agglomeration.mode.changed.connect(self._on_mode_changed)

        # @@@@@ dt watershed @@@@@
        self.widget_dt_ws = self.factory_dt_ws()
        self.widget_dt_ws.self.bind(self)
        self.widget_dt_ws.stacked._default_choices = self.STACKED
        self.widget_dt_ws.stacked.reset_choices()
        self.widget_dt_ws.stacked.value = False

        self.advanced_dt_ws = [
            self.widget_dt_ws.sigma_seeds,
            self.widget_dt_ws.sigma_weights,
            self.widget_dt_ws.alpha,
            self.widget_dt_ws.use_pixel_pitch,
            self.widget_dt_ws.pixel_pitch,
            self.widget_dt_ws.apply_nonmax_suppression,
            self.widget_dt_ws.is_nuclei_image,
        ]
        for widget in self.advanced_dt_ws:
            widget.hide()
        self.widget_dt_ws.show_advanced.changed.connect(self._on_show_advanced_changed)

        # Avoid throwing an error when the first image is loaded but
        # its layout is not supported
        self.initialised_widget_dt_ws: bool = False

        # @@@@@ Prediction widgets @@@@@
        self.prediction_widgets = Prediction_Widgets(self.widget_layer_select)

        # @@@@@ Hide/Show buttons @@@@@
        self.widget_show_prediction = self.factory_show_button()
        self.widget_show_prediction.name += "_prediction"
        self.widget_show_prediction.self.bind(self)
        self.widget_show_prediction.toggle.bind(lambda _: self.toggle_visibility_1)
        self.prediction_widgets.widget_unet_prediction.hide()

        self.widget_show_watershed = self.factory_show_button()
        self.widget_show_watershed.name += "_watershed"
        self.widget_show_watershed.self.bind(self)
        self.widget_show_watershed.toggle.bind(lambda _: self.toggle_visibility_2)

        self.widget_show_agglomeration = self.factory_show_button()
        self.widget_show_agglomeration.name += "_agglomeration"
        self.widget_show_agglomeration.self.bind(self)
        self.widget_show_agglomeration.toggle.bind(lambda _: self.toggle_visibility_3)

        # @@@@@ AllInOne Watershed & agglomeration @@@@@
        self.widget_aio_ws = self.factory_aio_ws()
        self.widget_aio_ws.self.bind(self)

        self.widget_aio_ws.mode._default_choices = AGGLOMERATION_MODES
        self.widget_aio_ws.mode.reset_choices()
        self.widget_aio_ws.mode.value = AGGLOMERATION_MODES[0][1]
        self.widget_aio_ws.mode.changed.connect(self._on_mode_changed)

        self.widget_aio_ws.stacked._default_choices = self.STACKED
        self.widget_aio_ws.stacked.reset_choices()
        self.widget_aio_ws.stacked.value = False
        self.widget_aio_ws._unify_label_widths()

        self.widget_show_prediction_aio = self.factory_show_button()
        self.widget_show_prediction_aio.name += "_prediction_aio"
        self.widget_show_prediction_aio.self.bind(self)
        self.widget_show_prediction_aio.toggle.bind(
            lambda _: self.toggle_visibility_aio_1
        )

        self.widget_show_aio_ws = self.factory_show_button()
        self.widget_show_aio_ws.name += "_aio_ws"
        self.widget_show_aio_ws.self.bind(self)
        self.widget_show_aio_ws.toggle.bind(lambda _: self.toggle_visibility_aio_2)

        self.container_aio_buttons = self.factory_aio_buttons()
        self._aio_advanced_button.clicked.connect(
            lambda _: self.set_separate_steps_list()
        )
        self._aio_call_button.clicked.connect(lambda _: self.widget_aio_ws())

        self.container_list = []
        self.container = Container(
            widgets=self.container_list,
            labels=False,
        )
        self.container.native.layout().addStretch()
        self.set_merged_steps_list()

    def set_separate_steps_list(self):
        self.aio = False
        new_container_list = [
            (div("Layer Selection"), True),
            (self.widget_layer_select, True),
            (div("1. Boundary Prediction"), True),
            (self.widget_show_prediction, True),
            (self.prediction_widgets.widget_unet_prediction, False),
            (self.prediction_widgets.widget_add_custom_model, False),
            (div("2. Boundary to Superpixel"), True),
            (self.widget_show_watershed, True),
            (self.widget_dt_ws, False),
            (div("3. Superpixel to Segmentation"), True),
            (self.widget_show_agglomeration, True),
            (self.widget_agglomeration, False),
            (self.container_aio_buttons, False),
        ]
        self.update_container(new_container_list)

    def set_merged_steps_list(self):
        self.aio = True
        new_container_list = [
            (div("Layer Selection"), True),
            (self.widget_layer_select, True),
            (div("1. Boundary Prediction"), True),
            (self.widget_show_prediction_aio, True),
            (self.prediction_widgets.widget_unet_prediction, False),
            (self.prediction_widgets.widget_add_custom_model, False),
            (div("2. Boundary to Segmentation"), True),
            (self.widget_show_aio_ws, True),
            (self.widget_aio_ws, False),
            (self.container_aio_buttons, False),
        ]
        self.update_container(new_container_list)

    def update_container(self, new_container_list):
        for w, v in self.container_list:
            w.hide()
        for _ in range(len(self.container)):
            self.container.pop(0)
        for w, v in new_container_list:
            self.container.append(w)
            if v:
                w.show()
            else:
                w.hide()
        self.container_list = new_container_list
        self.container.native.layout().addStretch()

    def get_container(self):
        return self.container

    @magic_factory(
        call_button=False,
        layer={
            "label": "Input image",
            "tooltip": "Select a layer to operate on.",
        },
        prediction={
            "label": "Boundary prediction",
            "tooltip": "Select the boundary prediction.",
        },
        nuclei={
            "label": "Nuclei image",
            "tooltip": "Nuclei foreground prediction or segmentation.",
            "visible": False,
        },
        superpixels={
            "label": "Superpixels",
            "tooltip": "Over-segmentation labels layer (superpixels) to use as input for clustering.",
        },
    )
    def factory_layer_select(
        self,
        layer: Image,
        prediction: Image,
        nuclei: Layer,
        superpixels: Labels,
    ):
        pass

    @magic_factory(
        call_button="Show",
    )
    def factory_show_button(self, toggle):
        toggle(visible=True)

    def toggle_visibility_1(self, visible: bool):
        """Toggles visibility of the prediction section"""
        if visible:
            self.widget_show_prediction.hide()
            self.toggle_visibility_2(False)
            self.toggle_visibility_3(False)
            self.prediction_widgets.widget_unet_prediction.show()
        else:
            self.prediction_widgets.widget_unet_prediction.hide()
            self.prediction_widgets.widget_add_custom_model.hide()
            self.widget_show_prediction.show()

    def toggle_visibility_2(self, visible: bool):
        """Toggles visibility of the watershed section"""
        if visible:
            self.widget_show_watershed.hide()
            self.toggle_visibility_1(False)
            self.toggle_visibility_3(False)
            self.widget_dt_ws.show()
        else:
            self.widget_dt_ws.hide()
            self.widget_show_watershed.show()

    def toggle_visibility_3(self, visible: bool):
        """Toggles visibility of the agglomeration section"""
        if visible:
            self.widget_show_agglomeration.hide()
            self.toggle_visibility_1(False)
            self.toggle_visibility_2(False)
            self.widget_agglomeration.show()
        else:
            self.widget_agglomeration.hide()
            self.widget_show_agglomeration.show()

    def toggle_visibility_aio_1(self, visible: bool):
        """Toggles visibility of the prediction section"""
        logger.debug(f"toggle_visibility_aio_1 {visible} called!")
        if visible:
            self.widget_show_prediction_aio.hide()
            self.toggle_visibility_aio_2(False)
            self.prediction_widgets.widget_unet_prediction.show()
        else:
            self.prediction_widgets.widget_unet_prediction.hide()
            self.prediction_widgets.widget_add_custom_model.hide()
            self.widget_show_prediction_aio.show()

    def toggle_visibility_aio_2(self, visible: bool):
        """Toggles visibility of the aio watershed section"""
        logger.debug(f"toggle_visibility_aio_2 {visible} called!")
        if visible:
            self.widget_show_aio_ws.hide()
            self.toggle_visibility_aio_1(False)
            self.widget_aio_ws.show()
            self.container_aio_buttons.show()
        else:
            self.widget_aio_ws.hide()
            self.container_aio_buttons.hide()
            self.widget_show_aio_ws.show()

    @magic_factory(
        call_button="Boundary to Superpixels",
        stacked={
            "label": "Mode",
            "tooltip": "Define if the Watershed will run slice by slice (faster) "
            "or in 3D (more accurate)",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
        },
        threshold={
            "label": "Boundary threshold",
            "tooltip": "A low value will increase over-segmentation tendency "
            "and a large value increase under-segmentation tendency.",
            "widget_type": "FloatSlider",
            "max": 1.0,
            "min": 0.0,
        },
        min_size={
            "label": "Minimum superpixel size",
            "tooltip": "Minimum superpixel size allowed in voxels.",
        },
        # Advanced parameters
        show_advanced={
            "label": "Show advanced parameters",
            "tooltip": "Show advanced parameters for the Watershed algorithm.",
            "widget_type": "CheckBox",
        },
        sigma_seeds={
            "label": "Sigma seeds",
            "tooltip": "Standard deviation for Gaussian smoothing applied to "
            "the seed map. Defaults to 1.0.",
        },
        sigma_weights={
            "label": "Sigma weights",
            "tooltip": "Standard deviation for Gaussian smoothing applied to "
            "theweight map. Defaults to 2.0.",
        },
        alpha={
            "label": "Alpha",
            "tooltip": "Blending factor between the input image and the "
            "distance transform when computing the weight map. Defaults to 1.0.",
        },
        use_pixel_pitch={
            "label": "Use pixel pitch",
            "tooltip": "Anisotropy factors for the distance transform. "
            "If None, isotropic distances are assumed. Defaults to None.",
        },
        pixel_pitch={
            "label": "Pixel pitch",
            "tooltip": "Anisotropy factors for the distance transform. "
            "If None, isotropic distances are assumed. Defaults to None.",
        },
        apply_nonmax_suppression={
            "label": "Apply nonmax suppression",
            "tooltip": "applies non-maximum suppression to the detected seeds, "
            "reducing seed redundancy. This requires the Nifty library.",
        },
        is_nuclei_image={
            "label": "Is nuclei image",
            "tooltip": "If True, indicates that the input image is a nuclei "
            "image, and preprocessing is applied accordingly.",
        },
        pbar={"label": "Watershed is running", "max": 0, "min": 0, "visible": False},
    )
    def factory_dt_ws(
        self,
        stacked: bool,
        threshold: float = 0.5,
        min_size: int = 100,
        show_advanced: bool = False,
        sigma_seeds: float = 0.2,
        sigma_weights: float = 2.0,
        alpha: float = 1.0,
        use_pixel_pitch: bool = False,
        pixel_pitch: tuple[int, int, int] = (1, 1, 1),
        apply_nonmax_suppression: bool = False,
        is_nuclei_image: bool = False,
        pbar: Optional[ProgressBar] = None,
    ) -> None:
        if self.widget_layer_select.prediction.value is None:
            log(
                "Please run a boundary prediction first!",
                thread="Segmentation",
                level="WARNING",
            )
            return
        ps_image = PanSegImage.from_napari_layer(
            self.widget_layer_select.prediction.value
        )

        return schedule_task(
            dt_watershed_task,
            task_kwargs={
                "image": ps_image,
                "threshold": threshold,
                "sigma_seeds": sigma_seeds,
                "stacked": stacked,
                "blockwise": not stacked,
                "sigma_weights": sigma_weights,
                "min_size": min_size,
                "alpha": alpha,
                "pixel_pitch": pixel_pitch if use_pixel_pitch else None,
                "apply_nonmax_suppression": apply_nonmax_suppression,
                "is_nuclei_image": is_nuclei_image,
                "_pbar": pbar,
                "_to_hide": [self.widget_dt_ws.call_button],
            },
        )

    @magic_factory(
        call_button="Superpixels to Segmentation",
        mode={
            "label": "Agglomeration mode",
            "tooltip": "Select which agglomeration algorithm to use.",
            "widget_type": "ComboBox",
        },
        beta={
            "label": "Under/Over segmentation factor",
            "tooltip": "A low value will increase under-segmentation tendency "
            "and a large value increase over-segmentation tendency.",
            "widget_type": "FloatSlider",
            "max": 1.0,
            "min": 0.0,
        },
        minsize={
            "label": "Minimum segment size",
            "tooltip": "Minimum segment size allowed in voxels.",
        },
    )
    def factory_agglomeration(
        self,
        mode: str,
        beta: float = 0.6,
        minsize: int = 100,
    ) -> None:
        if self.widget_layer_select.prediction.value is None:
            log(
                "Please run a prediction first!",
                thread="Segmentation",
                level="WARNING",
            )
            return
        if self.widget_layer_select.superpixels.value is None:
            log(
                "Please run `Boundary to Superpixels` first!",
                thread="Segmentation",
                level="WARNING",
            )
            return
        ps_image = PanSegImage.from_napari_layer(
            self.widget_layer_select.prediction.value
        )
        ps_labels = PanSegImage.from_napari_layer(
            self.widget_layer_select.superpixels.value
        )
        nuclei = self.widget_layer_select.nuclei.value

        # Hide the superpixels layer to avoid overlapping with the new segmentation
        self.widget_layer_select.superpixels.value.visible = False

        if mode == "lmc":
            if not isinstance(nuclei, (Image, Labels)):
                log(
                    "Nuclei must be an Image or Labels layer",
                    thread="Segmentation",
                    level="WARNING",
                )
                return

            ps_nuclei = PanSegImage.from_napari_layer(nuclei)
            return schedule_task(
                lmc_segmentation_task,
                task_kwargs={
                    "boundary_pmap": ps_image,
                    "superpixels": ps_labels,
                    "nuclei": ps_nuclei,
                    "beta": beta,
                    "post_min_size": minsize,
                },
            )

        return schedule_task(
            clustering_segmentation_task,
            task_kwargs={
                "image": ps_image,
                "over_segmentation": ps_labels,
                "mode": mode.lower(),
                "beta": beta,
                "post_min_size": minsize,
            },
        )

    def _on_mode_changed(self, mode: str):
        if mode == "lmc":
            self.widget_layer_select.nuclei.show()
            log("Check nuclei layer selection!", thread="Segmentation", level="INFO")
        else:
            self.widget_layer_select.nuclei.hide()

    def _on_show_advanced_changed(self, state: bool):
        if state:
            self.prediction_widgets.widget_unet_prediction.advanced.value = False
            for widget in self.advanced_dt_ws:
                widget.show()
        else:
            for widget in self.advanced_dt_ws:
                widget.hide()

    def _on_prediction_change(self, image: Optional[Image]):
        if image is None:
            return
        ps_image = PanSegImage.from_napari_layer(image)

        if ps_image.image_layout == ImageLayout.ZYX:
            self.widget_dt_ws.stacked.show()
        else:
            self.widget_dt_ws.stacked.hide()
            self.widget_dt_ws.stacked.value = False
            if ps_image.image_layout != ImageLayout.YX:
                log(
                    f"Unsupported image layout: {ps_image.image_layout}",
                    thread="DT Watershed",
                    level="error",
                )

    def update_layer_selection(self, event):
        """Updates layer drop-down menus"""
        logger.debug(
            f"Updating segmentation layer selection: {event.value}, {event.type}"
        )
        raws = get_layers(SemanticType.RAW)
        predictions = get_layers(SemanticType.PREDICTION)
        segmentations = get_layers(SemanticType.SEGMENTATION)

        self.widget_layer_select.layer.choices = raws
        self.widget_layer_select.prediction.choices = predictions
        self.widget_layer_select.nuclei.choices = raws + predictions + segmentations
        self.widget_layer_select.superpixels.choices = segmentations

        # Hide empty choices
        if self.widget_layer_select.prediction.choices == ():
            self.widget_layer_select.prediction.hide()
        else:
            self.widget_layer_select.prediction.show()

        if self.widget_layer_select.superpixels.choices == ():
            self.widget_layer_select.superpixels.hide()
        else:
            self.widget_layer_select.superpixels.show()
        if self.aio:
            self.widget_layer_select.superpixels.hide()

        # Set values to inserted
        if event.type == "inserted":
            if event.value._metadata.get("semantic_type", None) == SemanticType.RAW:
                self.widget_layer_select.layer.value = event.value
                self.widget_layer_select.nuclei.value = event.value
            elif (
                event.value._metadata.get("semantic_type", None)
                == SemanticType.PREDICTION
            ):
                self.widget_layer_select.prediction.value = event.value
            elif (
                event.value._metadata.get("semantic_type", None)
                == SemanticType.SEGMENTATION
            ):
                self.widget_layer_select.superpixels.value = event.value

        # update additional layer selection for multi channel models
        for combo_box in self.widget_layer_select[-1]:
            combo_box.choices = raws

    @magic_factory(
        call_button=False,
        stacked={
            "label": "Mode",
            "tooltip": "Define if the Watershed will run slice by slice (faster) "
            "or on the full volume across all CPUs (fastest for large volumes).",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
        },
        threshold={
            "label": "Boundary threshold",
            "tooltip": "A low value will increase over-segmentation tendency "
            "and a large value increase under-segmentation tendency.",
            "widget_type": "FloatSlider",
            "max": 1.0,
            "min": 0.0,
        },
        min_size={
            "label": "Minimum superpixel size",
            "tooltip": "Minimum superpixel size allowed in voxels.",
        },
        # Advanced parameters
        sigma_seeds={
            "label": "Sigma seeds",
            "tooltip": "Standard deviation for Gaussian smoothing applied to "
            "the seed map. Defaults to 1.0.",
            "visible": False,
        },
        sigma_weights={
            "label": "Sigma weights",
            "tooltip": "Standard deviation for Gaussian smoothing applied to "
            "theweight map. Defaults to 2.0.",
            "visible": False,
        },
        alpha={
            "label": "Alpha",
            "tooltip": "Blending factor between the input image and the "
            "distance transform when computing the weight map. Defaults to 1.0.",
            "visible": False,
        },
        use_pixel_pitch={
            "label": "Use pixel pitch",
            "tooltip": "Anisotropy factors for the distance transform. "
            "If None, isotropic distances are assumed. Defaults to None.",
            "visible": False,
        },
        pixel_pitch={
            "label": "Pixel pitch",
            "tooltip": "Anisotropy factors for the distance transform. "
            "If None, isotropic distances are assumed. Defaults to None.",
            "visible": False,
        },
        apply_nonmax_suppression={
            "label": "Apply nonmax suppression",
            "tooltip": "applies non-maximum suppression to the detected seeds, "
            "reducing seed redundancy. This requires the Nifty library.",
            "visible": False,
        },
        is_nuclei_image={
            "label": "Is nuclei image",
            "tooltip": "If True, indicates that the input image is a nuclei "
            "image, and preprocessing is applied accordingly.",
            "visible": False,
        },
        # Agglomeration settings
        mode={
            "label": "Agglomeration mode",
            "tooltip": "Select which agglomeration algorithm to use.",
            "widget_type": "ComboBox",
            "choices": AGGLOMERATION_MODES,
        },
        beta={
            "label": "Under/Over segmentation factor",
            "tooltip": "A low value will increase under-segmentation tendency "
            "and a large value increase over-segmentation tendency.",
            "widget_type": "FloatSlider",
            "max": 1.0,
            "min": 0.0,
        },
        pbar={"label": "Segmentation is running", "max": 0, "min": 0, "visible": False},
    )
    def factory_aio_ws(
        self,
        stacked: bool,
        threshold: float = 0.5,
        min_size: int = 100,
        sigma_seeds: float = 0.2,
        sigma_weights: float = 2.0,
        alpha: float = 1.0,
        use_pixel_pitch: bool = False,
        pixel_pitch: tuple[int, int, int] = (1, 1, 1),
        apply_nonmax_suppression: bool = False,
        is_nuclei_image: bool = False,
        mode: str = AGGLOMERATION_MODES[0][1],
        beta: float = 0.6,
        pbar: Optional[ProgressBar] = None,
    ) -> None:
        if self.widget_layer_select.prediction.value is None:
            log(
                "Please run a boundary prediction first!",
                thread="Segmentation",
                level="WARNING",
            )
            return
        ps_image = PanSegImage.from_napari_layer(
            self.widget_layer_select.prediction.value
        )
        nuclei = None
        if mode == "lmc":
            if not isinstance(self.widget_layer_select.nuclei.value, (Image, Labels)):
                log(
                    "Nuclei must be an Image or Labels layer",
                    thread="Segmentation",
                    level="WARNING",
                )
                return
            nuclei = PanSegImage.from_napari_layer(
                self.widget_layer_select.nuclei.value
            )

        return schedule_task(
            aio_watershed_task,
            task_kwargs={
                "image": ps_image,
                "nuclei": nuclei,
                "threshold": threshold,
                "sigma_seeds": sigma_seeds,
                "stacked": stacked,
                "blockwise": not stacked,
                "sigma_weights": sigma_weights,
                "min_size": min_size,
                "alpha": alpha,
                "pixel_pitch": pixel_pitch if use_pixel_pitch else None,
                "apply_nonmax_suppression": apply_nonmax_suppression,
                "is_nuclei_image": is_nuclei_image,
                "mode": mode,
                "beta": beta,
                "_pbar": pbar,
                "_to_hide": [self.widget_dt_ws.call_button],
            },
        )

    def factory_aio_buttons(self):
        call_button = PushButton(text="Boundary to Segmentation")
        advanced_button = PushButton(text="Advanced")
        advanced_button.max_width = 80

        widget_aio_combo = Container(
            widgets=[call_button, advanced_button],
            layout="horizontal",
            labels=False,
            label="",
            name="_button_combo",
            gui_only=True,
        )
        widget_aio_combo.margins = (9, 0, 9, 9)

        self._aio_call_button = call_button
        self._aio_advanced_button = advanced_button
        widget_aio_combo.show()
        return widget_aio_combo
