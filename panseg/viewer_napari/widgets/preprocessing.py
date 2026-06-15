from enum import Enum
from typing import Optional

import napari
from magicgui import magic_factory
from magicgui.widgets import Container, Label, PushButton
from napari.layers import Image, Labels, Layer, Shapes

from panseg import logger
from panseg.core.image import ImageDimensionality, PanSegImage, SemanticType
from panseg.core.zoo import model_zoo
from panseg.io.voxelsize import VoxelSize
from panseg.tasks.dataprocessing_tasks import (
    ImagePairOperation,
    gaussian_smoothing_task,
    image_cropping_task,
    image_pair_operation_task,
    image_rescale_to_shape_task,
    image_rescale_to_voxel_size_task,
    set_voxel_size_task,
)
from panseg.viewer_napari import log
from panseg.viewer_napari.widgets.utils import (
    Help_text,
    div,
    get_layers,
    schedule_task,
)


class RescaleType(Enum):
    NEAREST = (0, "Nearest")
    LINEAR = (1, "Linear")
    BILINEAR = (2, "Bilinear")

    def __init__(self, int_val, str_val):
        self.int_val = int_val
        self.str_val = str_val

    @classmethod
    def to_choices(cls):
        return [(mode.str_val, mode.int_val) for mode in cls]


class RescaleModes(Enum):
    FROM_FACTOR = "From factor"
    TO_LAYER_VOXEL_SIZE = "To layer voxel size"
    TO_LAYER_SHAPE = "To layer shape"
    TO_MODEL_VOXEL_SIZE = "To model voxel size"
    TO_VOXEL_SIZE = "To voxel size"
    TO_SHAPE = "To shape"
    SET_VOXEL_SIZE = "Set voxel size"

    @classmethod
    def to_choices(cls):
        return [(mode.value, mode) for mode in RescaleModes]


class Preprocessing_Tab:
    def __init__(self):
        # @@@@@ Layer selector @@@@@
        self.widget_layer_select = self.factory_layer_select()
        self.widget_layer_select.self.bind(self)
        self.widget_layer_select.layer.changed.connect(self._on_cropping_image_changed)
        self.widget_layer_select.layer.changed.connect(self._on_layer_selection)

        # @@@@@ Smoothing @@@@@
        self.widget_gaussian_smoothing = self.factory_gaussian_smoothing()
        self.widget_gaussian_smoothing.self.bind(self)

        # @@@@@ Cropping @@@@@
        self.widget_cropping = self.factory_cropping()
        self.widget_cropping.self.bind(self)

        self.initialised_widget_cropping: bool = False

        addShape_button = PushButton(
            value=False,
            text="To start cropping, add a shapes layer by clicking here,\nthen draw a single rectangle",
        )
        viewer = napari.current_viewer()
        if viewer is not None:
            addShape_button.clicked.connect(
                lambda _: viewer.add_shapes(ndim=3, scale=viewer.layers.extent.step)
            )
        self.widget_cropping_placeholder = Container(widgets=[addShape_button])
        self.widget_cropping.hide()

        # @@@@@ Rescaling @@@@@
        self.widget_rescaling = self.factory_rescaling()
        self.widget_rescaling.self.bind(self)

        self.list_widget_rescaling_all = [
            self.widget_rescaling.out_voxel_size,
            self.widget_rescaling.reference_layer,
            self.widget_rescaling.reference_model,
            self.widget_rescaling.rescaling_factor,
            self.widget_rescaling.reference_shape,
        ]

        for widget in self.list_widget_rescaling_all:
            widget.hide()
        self.widget_rescaling.rescaling_factor.show()
        self.widget_rescaling.reference_shape[0].max = 20000
        self.widget_rescaling.reference_shape[1].max = 20000
        self.widget_rescaling.reference_shape[2].max = 20000

        self.widget_rescaling.order.changed.connect(self._on_rescale_order_changed)
        self.widget_rescaling.mode.changed.connect(self._rescale_update_visibility)

        self.list_widget_rescaling_3d = [
            self.widget_rescaling.rescaling_factor[0],
            self.widget_rescaling.reference_shape[0],
            self.widget_rescaling.out_voxel_size[0],
        ]

        # @@@@@ Image Math @@@@@
        self.widget_image_pair_operations = self.factory_image_pair_operations()
        self.widget_image_pair_operations.self.bind(self)

        # @@@@@ Placeholder @@@@@
        self.hidden_label = Label(
            value="Preprocessing only supported for 2D and 3D images"
        )
        self.hidden_label.hide()
        self.to_restore = []
        self.hidden = False

        # @@@@@ Toggle buttons @@@@@
        self.widget_show_rescaling = self.factory_show_button()
        self.widget_show_rescaling.self.bind(self)
        self.widget_show_rescaling.toggle.bind(lambda _: self.toggle_visibility_1)

        self.widget_show_image_operations = self.factory_show_button()
        self.widget_show_image_operations.self.bind(self)
        self.widget_show_image_operations.toggle.bind(
            lambda _: self.toggle_visibility_2
        )

        self.toggle_visibility_1(True)
        self.toggle_visibility_2(False)

        help_text = "<strong>Preprocessing:</strong> Optional image operations."
        self.help_text_container = Help_text()
        self.tab_help = self.help_text_container.get_doc_container(
            help_text,
            sub_url="chapters/panseg_interactive_napari/preprocessing/",
        )

        self.container = Container(
            widgets=[
                self.tab_help,
                self.hidden_label,
                div("Layer Selection"),
                self.widget_layer_select,
                div("Crop"),
                self.widget_cropping_placeholder,
                self.widget_cropping,
                div("Rescale"),
                self.widget_show_rescaling,
                self.widget_rescaling,
                div("Gaussian Smoothing"),
                self.widget_gaussian_smoothing,
                div("Image pair operations"),
                self.widget_show_image_operations,
                self.widget_image_pair_operations,
            ],
            labels=False,
        )

    def get_container(self):
        # getter to keep consistency to other tabs
        # only needed to hide hole tab (4d workaround)
        return self.container

    @magic_factory(
        call_button=False,
        layer={
            "label": "Layer",
            "tooltip": "Select a layer to operate on.",
        },
    )
    def factory_layer_select(self, layer: Image):
        pass

    @magic_factory(
        call_button="Show",
    )
    def factory_show_button(self, toggle):
        toggle(visible=True)

    def toggle_visibility_1(self, visible: bool):
        """Toggle visibility of rescaling section"""
        if visible:
            self.widget_show_rescaling.hide()
            self.toggle_visibility_2(False)
            self.widget_rescaling.show()

        else:
            self.widget_rescaling.hide()
            self.widget_show_rescaling.show()

    def toggle_visibility_2(self, visible: bool):
        """Toggle visibility of image pair operations section"""
        if visible:
            self.widget_show_image_operations.hide()
            self.toggle_visibility_1(False)
            self.widget_image_pair_operations.show()

        else:
            self.widget_image_pair_operations.hide()
            self.widget_show_image_operations.show()

    def toggle_visibility_3(self, visible: bool):
        """Toggle visibility of everything in preprocessing"""
        logger.debug(f"toggle_visibility_3 called with {visible}")
        if visible and self.hidden:
            self.hidden_label.hide()
            for w in self.to_restore:
                w.show()
            self.to_restore = []
            self.toggle_visibility_1(True)
            self.hidden = False
        elif not visible:
            self.hidden = True
            for w in self.container:
                if w.visible:
                    self.to_restore.append(w)
                    w.hide()
            self.hidden_label.show()

    @magic_factory(
        call_button="Run Gaussian Smoothing",
        sigma={
            "label": "Sigma",
            "widget_type": "FloatSlider",
            "tooltip": "Define the size of the gaussian smoothing kernel. "
            "The larger the more blurred will be the output image.",
            "max": 10.0,
            "min": 0.1,
        },
    )
    def factory_gaussian_smoothing(self, sigma: float = 1.0) -> None:
        """Apply Gaussian smoothing to an image layer."""

        if self.widget_layer_select.layer.value is None:
            log(
                "Please select a layers first!",
                thread="Preprocessing",
                level="WARNING",
            )
            return
        ps_image = PanSegImage.from_napari_layer(self.widget_layer_select.layer.value)

        return schedule_task(
            gaussian_smoothing_task,
            task_kwargs={
                "image": ps_image,
                "sigma": sigma,
            },
        )

    @magic_factory(
        call_button="Crop Image",
        crop_roi={
            "label": "Crop shapes layer",
            "tooltip": "This must be a shape layer with a rectangle overlaying the area to crop.",
        },
        crop_z={
            "label": "Z slices [start, end)",
            "tooltip": "Number of z slices to take next to the current selection.\nSTART is included, END is not.",
            "widget_type": "RangeSlider",
            "max": 100,
            "min": 0,
            "step": 1,
        },
        update_other_widgets={
            "visible": False,
            "tooltip": "To allow toggle the update of other widgets in unit tests; invisible to users.",
        },
    )
    def factory_cropping(
        self,
        crop_roi: Shapes,
        crop_z: tuple[int, int] = (0, 100),
        update_other_widgets: bool = True,
    ) -> None:
        layer = self.widget_layer_select.layer.value
        if layer is None:
            log(
                "Please select a layer to crop first!",
                thread="Preprocessing",
                level="WARNING",
            )
            return

        if crop_roi is None:
            log(
                "Please create a Shapes layer first!",
                thread="Preprocessing",
                level="WARNING",
            )
            return
        if len(crop_roi.shape_type) > 1:
            log(
                "Can't use more than one rectangle for cropping!",
                thread="Preprocessing",
                level="WARNING",
            )
            return
        if crop_roi.shape_type and crop_roi.shape_type[0] != "rectangle":
            log(
                "A rectangle shape must be used for cropping",
                thread="Preprocessing",
                level="WARNING",
            )
            return
        if not isinstance(layer, (Image, Labels)):
            log(
                f"{type(layer)} cannot be cropped, please use Image layers or Labels layers",
                thread="Preprocessing",
                level="WARNING",
            )
            return

        if len(crop_roi.data) > 0:
            rectangle = crop_roi.data[0].astype("int64")
        else:
            rectangle = None

        ps_image = PanSegImage.from_napari_layer(layer)
        crop_roi.visible = False
        layer.visible = False

        return schedule_task(
            image_cropping_task,
            task_kwargs={
                "image": ps_image,
                "rectangle": rectangle,
                "crop_z": crop_z,
            },
        )

    def update_layer_selection(self, event):
        """To be called when the layer list changes."""
        logger.debug(
            f"Updating preprocessing layer selection: {event.value}, {event.type}"
        )

        all_layers = get_layers(
            [
                SemanticType.RAW,
                SemanticType.PREDICTION,
                SemanticType.SEGMENTATION,
            ]
        )
        self.widget_layer_select.layer.choices = all_layers
        self.widget_image_pair_operations.layer1.choices = all_layers
        self.widget_image_pair_operations.layer2.choices = all_layers

        self.widget_cropping.crop_roi.choices = list(
            filter(lambda l: isinstance(l, Shapes), napari.current_viewer().layers)
        )

        if event.type != "inserted":
            return
        if event.value in self.widget_layer_select.layer.choices:
            self.widget_layer_select.layer.value = event.value
        if isinstance(event.value, Shapes):
            logger.debug("Shapes layer added, updating cropping.")
            self.initialised_widget_cropping = True
            self.widget_cropping.crop_roi.value = event.value
            self._on_cropping_image_changed(self.widget_layer_select.layer.value)
            self.widget_cropping_placeholder.hide()
            self.widget_cropping.show()

    def _on_cropping_image_changed(self, image: Optional[Layer]):
        """Called upon changing widget_layer_select"""
        logger.debug("_on_cropping_image_changed called!")
        if not self.initialised_widget_cropping:
            return
        if image is None:
            return

        assert isinstance(image, (Image, Labels)), (
            f"{type(image)} cannot be cropped, please use Image layers or Labels layers"
        )
        ps_image = PanSegImage.from_napari_layer(image)

        if ps_image.dimensionality == ImageDimensionality.TWO:
            self.widget_cropping.crop_z.hide()
            return None

        if ps_image.is_multichannel:
            raise ValueError("Multichannel images are not supported for cropping.")

        self.widget_cropping.crop_z.show()
        image_shape_z = ps_image.shape[0]

        self.widget_cropping.crop_z.step = 1

        if self.widget_cropping.crop_z.value[1] > image_shape_z:
            self.widget_cropping.crop_z.value = (0, image_shape_z)
        self.widget_cropping.crop_z.max = image_shape_z

    @magic_factory(
        call_button="Rescale Image",
        mode={
            "label": "Rescale mode",
            "choices": RescaleModes.to_choices(),
        },
        rescaling_factor={
            "label": "Rescaling factor",
            "tooltip": "Define the scaling factor to use for resizing the input image.",
            "options": {"step": 0.0001},
        },
        out_voxel_size={
            "label": "Out voxel size",
            "tooltip": "Define the output voxel size. Units are same as imported, "
            '(if units are missing default is "um").',
            "options": {"step": 0.0001},
        },
        reference_layer={
            "label": "Reference layer",
            "tooltip": "Rescale to same voxel size as selected layer.",
        },
        reference_model={
            "label": "Reference model",
            "tooltip": "Rescale to same voxel size as selected model.",
            "choices": model_zoo.list_models(),
        },
        reference_shape={
            "label": "Out shape",
            "tooltip": "Rescale to a manually selected shape.",
        },
        order={
            "label": "Interpolation order",
            "choices": RescaleType.to_choices(),
            "tooltip": "0 for nearest neighbours (default for labels), 1 for linear, 2 for bilinear.",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
        },
    )
    def factory_rescaling(
        self,
        mode: RescaleModes = RescaleModes.FROM_FACTOR,
        rescaling_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        out_voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        reference_layer: Layer | None = None,
        reference_model: str = model_zoo.list_models()[0],
        reference_shape: tuple[int, int, int] = (1, 1, 1),
        order: int = 0,
    ) -> None:
        """Rescale an image or label layer."""

        layer = self.widget_layer_select.layer.value
        if isinstance(layer, Image) or isinstance(layer, Labels):
            ps_image = PanSegImage.from_napari_layer(layer)
        else:
            raise ValueError("Image must be an Image or Label layer.")
        # Cover set voxel size case
        if not ps_image.has_valid_original_voxel_size():
            if mode not in [
                RescaleModes.SET_VOXEL_SIZE,
                RescaleModes.TO_LAYER_SHAPE,
                RescaleModes.TO_SHAPE,
            ]:
                log(
                    "Original voxel size is missing, please set the voxel "
                    "size manually.",
                    thread="Preprocessing",
                    level="WARNING",
                )
                return

        if mode == RescaleModes.SET_VOXEL_SIZE:
            # Run set voxel size task
            return schedule_task(
                set_voxel_size_task,
                task_kwargs={
                    "image": ps_image,
                    "voxel_size": out_voxel_size,
                },
            )

        if mode in [RescaleModes.TO_LAYER_SHAPE, RescaleModes.TO_SHAPE]:
            if reference_layer is None:
                raise ValueError("Reference layer can not be None!")
            if mode == RescaleModes.TO_LAYER_SHAPE:
                output_shape = reference_layer.data.shape
            else:
                output_shape = reference_shape

            return schedule_task(
                image_rescale_to_shape_task,
                task_kwargs={
                    "image": ps_image,
                    "new_shape": output_shape,
                    "order": order,
                },
            )

        # Cover rescale that requires a valid voxel size
        current_voxel_size = ps_image.voxel_size
        if mode == RescaleModes.FROM_FACTOR:
            out_voxel_size = current_voxel_size.voxelsize_from_factor(rescaling_factor)

        if mode == RescaleModes.TO_VOXEL_SIZE:
            out_voxel_size = VoxelSize(
                voxels_size=out_voxel_size, unit=current_voxel_size.unit
            )

        if mode == RescaleModes.TO_LAYER_VOXEL_SIZE:
            if not (
                isinstance(reference_layer, Image)
                or isinstance(reference_layer, Labels)
            ):
                raise ValueError("Reference layer must be an Image or Label layer.")
            reference_ps_image = PanSegImage.from_napari_layer(reference_layer)
            out_voxel_size = reference_ps_image.voxel_size

        if mode == RescaleModes.TO_MODEL_VOXEL_SIZE:
            model_voxel_size = model_zoo.get_model_resolution(reference_model)
            if model_voxel_size is None:
                raise ValueError(
                    f"Model {reference_model} does not have a resolution defined."
                )
            out_voxel_size = VoxelSize(
                voxels_size=model_voxel_size, unit=current_voxel_size.unit
            )

        return schedule_task(
            image_rescale_to_voxel_size_task,
            task_kwargs={
                "image": ps_image,
                "new_voxels_size": out_voxel_size.voxels_size,
                "new_unit": out_voxel_size.unit,
                "order": order,
            },
        )

    def _rescale_update_visibility(self, mode: RescaleModes):
        logger.debug("_rescale_update_visibility called!")
        for widget in self.list_widget_rescaling_all:
            widget.hide()

        match mode:
            case RescaleModes.FROM_FACTOR:
                self.widget_rescaling.rescaling_factor.show()

            case RescaleModes.TO_LAYER_VOXEL_SIZE:
                self.widget_rescaling.reference_layer.show()

            case RescaleModes.TO_MODEL_VOXEL_SIZE:
                self.widget_rescaling.reference_model.show()

            case RescaleModes.TO_VOXEL_SIZE:
                self.widget_rescaling.out_voxel_size.show()

            case RescaleModes.TO_LAYER_SHAPE:
                self.widget_rescaling.reference_layer.show()

            case RescaleModes.TO_SHAPE:
                self.widget_rescaling.reference_shape.show()

            case RescaleModes.SET_VOXEL_SIZE:
                self.widget_rescaling.out_voxel_size.show()

            case _:
                raise ValueError(f"{mode} is not implemented yet.")

    def _on_layer_selection(self, image: Layer):
        """Called upon changing widget_layer_select"""
        logger.debug(f"_on_layer_selection called: {image}")
        if not (isinstance(image, Image) or isinstance(image, Labels)):
            raise ValueError("Image must be an Image or Label layer.")

        if len(image.data.shape) > 3:
            logger.warning(
                "Preprocessing not supported for 4d images",
            )
            self.toggle_visibility_3(False)
            return
        else:
            self.toggle_visibility_3(True)

        if image.data.ndim == 2 or (image.data.ndim == 3 and image.data.shape[0] == 1):
            for widget in self.list_widget_rescaling_3d:
                widget.hide()
        else:
            for widget in self.list_widget_rescaling_3d:
                widget.show()

        offset = 1 if image.data.ndim == 2 else 0
        for i, (shape, scale) in enumerate(zip(image.data.shape, image.scale)):
            # TODO: fix for 4d images
            self.widget_rescaling.out_voxel_size[i + offset].value = scale
            self.widget_rescaling.reference_shape[i + offset].value = shape

        if isinstance(image, Labels):
            self.widget_rescaling.order.value = RescaleType.NEAREST.int_val

    def _on_rescale_order_changed(self, order):
        logger.debug("_on_rescale_order_changed called!")
        current_image = self.widget_layer_select.layer.value

        if current_image is None:
            return None

        if isinstance(current_image, Labels) and order != RescaleType.NEAREST.int_val:
            log(
                "Labels can only be rescaled with nearest interpolation",
                thread="Rescaling",
                level="warning",
            )
            self.widget_rescaling.order.value = RescaleType.NEAREST.int_val

    @magic_factory(
        call_button="Run Operation",
        layer1={
            "label": "Layer 1",
            "tooltip": "First image to apply the operation.",
        },
        layer2={
            "label": "Layer 2",
            "tooltip": "Second image to apply the operation.",
        },
        operation={
            "label": "Operation",
            "choices": ImagePairOperation,
        },
        normalize_input={
            "label": "Normalize input",
            "tooltip": "Normalize the input images to the range [0, 1].",
        },
        clip_output={
            "label": "Clip output",
            "tooltip": "Clip the output to the range [0, 1].",
        },
        normalize_output={
            "label": "Normalize output",
            "tooltip": "Normalize the output image to the range [0, 1].",
        },
    )
    def factory_image_pair_operations(
        self,
        layer1: Image,
        layer2: Image,
        operation: ImagePairOperation = "add",
        normalize_input: bool = False,
        clip_output: bool = False,
        normalize_output: bool = False,
    ) -> None:
        """Apply an operation to two image layers."""

        if layer1 is None or layer2 is None:
            log(
                "Please select two layers first!",
                thread="Preprocessing",
                level="WARNING",
            )
            return
        ps_layer1 = PanSegImage.from_napari_layer(layer1)
        ps_layer2 = PanSegImage.from_napari_layer(layer2)

        return schedule_task(
            image_pair_operation_task,
            task_kwargs={
                "image1": ps_layer1,
                "image2": ps_layer2,
                "operation": operation,
                "normalize_input": normalize_input,
                "clip_output": clip_output,
                "normalize_output": normalize_output,
            },
        )
