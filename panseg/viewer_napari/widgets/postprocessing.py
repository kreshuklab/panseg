from typing import Optional

from magicgui import magic_factory
from magicgui.widgets import Container
from napari.layers import Image, Labels

from panseg import logger
from panseg.core.image import PanSegImage, SemanticType
from panseg.tasks.dataprocessing_tasks import (
    fix_over_under_segmentation_from_nuclei_task,
    relabel_segmentation_task,
    remove_false_positives_by_foreground_probability_task,
    set_biggest_instance_to_zero_task,
)
from panseg.viewer_napari import log
from panseg.viewer_napari.widgets.utils import (
    Help_text,
    div,
    get_layers,
    schedule_task,
)


class Postprocessing_Tab:
    def __init__(self):
        # @@@@@ Layer selector @@@@@
        self.widget_layer_select = self.factory_layer_select()
        self.widget_layer_select.self.bind(self)
        # self.widget_layer_select.layer._default_choices = lambda _: get_layers(
        #     SemanticType.SEGMENTATION
        # )
        self.widget_layer_select.layer.changed.connect(self._on_layer_changed)

        # @@@@@ Relable @@@@@
        self.widget_relabel = self.factory_relabel()
        self.widget_relabel.self.bind(self)

        # @@@@@ Set biggest instance to zero @@@@@
        self.widget_set_biggest_instance_zero = self.factory_set_biggest_instance_zero()
        self.widget_set_biggest_instance_zero.self.bind(self)

        # @@@@@ remove false positive @@@@@
        self.widget_remove_false_positives_by_foreground = (
            self.factory_remove_false_positives_by_foreground()
        )
        self.widget_remove_false_positives_by_foreground.self.bind(self)

        # @@@@@ fix segmentation by nuclei @@@@@
        self.widget_fix_segmentation_by_nuclei = (
            self.factory_fix_segmentation_by_nuclei()
        )
        self.widget_fix_segmentation_by_nuclei.self.bind(self)

        # @@@@@ Toggle buttons @@@@@
        self.widget_show_remove_false_positive = self.factory_show_button()
        self.widget_show_remove_false_positive.self.bind(self)
        self.widget_show_remove_false_positive.toggle.bind(
            lambda _: self.toggle_visibility_1
        )

        self.widget_show_fix_segmentation = self.factory_show_button()
        self.widget_show_fix_segmentation.self.bind(self)
        self.widget_show_fix_segmentation.toggle.bind(
            lambda _: self.toggle_visibility_2
        )

        help_text = (
            "<strong>Postprocessing:</strong> Optional steps to improve a segmentation."
        )
        self.help_text_container = Help_text()
        self.tab_help = self.help_text_container.get_doc_container(
            help_text,
            sub_url="chapters/panseg_interactive_napari/postprocessing/",
        )

        self.toggle_visibility_1(False)
        self.toggle_visibility_2(False)

    def get_container(self):
        return Container(
            widgets=[
                self.tab_help,
                div("Layer Selection"),
                self.widget_layer_select,
                div("Relabel Instances"),
                self.widget_relabel,
                div("Set biggest Instance to Zero"),
                self.widget_set_biggest_instance_zero,
                div("Remove False-Positives by Foreground"),
                self.widget_show_remove_false_positive,
                self.widget_remove_false_positives_by_foreground,
                div("Split/Merge Instances by Nuclei"),
                self.widget_show_fix_segmentation,
                self.widget_fix_segmentation_by_nuclei,
            ],
            labels=False,
        )

    @magic_factory(
        call_button="Show",
    )
    def factory_show_button(self, toggle):
        toggle(visible=True)

    def toggle_visibility_1(self, visible: bool):
        "Toggle visibility of remove false positives"
        if visible:
            self.widget_show_remove_false_positive.hide()
            self.toggle_visibility_2(False)
            self.widget_remove_false_positives_by_foreground.show()
        else:
            self.widget_remove_false_positives_by_foreground.hide()
            self.widget_show_remove_false_positive.show()

    def toggle_visibility_2(self, visible: bool):
        "Toggle visibility of fix segmentation by nuclei"
        if visible:
            self.widget_show_fix_segmentation.hide()
            self.toggle_visibility_1(False)
            self.widget_fix_segmentation_by_nuclei.show()
        else:
            self.widget_fix_segmentation_by_nuclei.hide()
            self.widget_show_fix_segmentation.show()

    @magic_factory(
        call_button=False,
        layer={
            "label": "Segmentation Layer",
            "tooltip": "Apply background corrections to this layer",
        },
    )
    def factory_layer_select(self, layer: Labels):
        pass

    @magic_factory(
        call_button="Relabel Instances",
        background={
            "label": "Background Label",
            "tooltip": "Background label will be set to 0. Default is None.",
            "max": 1000,
            "min": 0,
        },
    )
    def factory_relabel(
        self,
        background: int | None = None,
    ) -> None:
        """Relabel an image layer."""
        if self.widget_layer_select.layer.value is None:
            log(
                "No Segmentation layer found!", thread="Postprocessing", level="WARNING"
            )
            return

        ps_image = PanSegImage.from_napari_layer(self.widget_layer_select.layer.value)
        self.widget_layer_select.layer.value.visible = False

        schedule_task(
            relabel_segmentation_task,
            task_kwargs={
                "image": ps_image,
                "background": background,
            },
        )

    def _on_layer_changed(self, segmentation: Optional[Labels]):
        if segmentation is None:
            # self.widget_relabel.background.hide()
            return
        self.widget_relabel.background.max = int(segmentation.data.max())

    @magic_factory(
        call_button="Set Biggest Instance to Zero",
        instance_could_be_zero={
            "label": "Treat 0 as Instance",
            "tooltip": "If ticked, a proper instance segmentation with 0 as background will not be modified.",
        },
    )
    def factory_set_biggest_instance_zero(
        self,
        instance_could_be_zero: bool = False,
    ) -> None:
        """Set the biggest instance to zero in a label layer."""

        if self.widget_layer_select.layer.value is None:
            log(
                "No Segmentation layer found!", thread="Postprocessing", level="WARNING"
            )
            return

        ps_image = PanSegImage.from_napari_layer(self.widget_layer_select.layer.value)
        self.widget_layer_select.layer.value.visible = False

        schedule_task(
            set_biggest_instance_to_zero_task,
            task_kwargs={
                "image": ps_image,
                "instance_could_be_zero": instance_could_be_zero,
            },
        )

    @magic_factory(
        call_button="Remove Objects with Low Foreground Probability",
        foreground={
            "label": "Foreground",
            "tooltip": "Foreground probability layer.",
        },
        threshold={
            "label": "Threshold",
            "tooltip": "Threshold value to remove false positives.",
            "widget_type": "FloatSlider",
            "max": 1.0,
            "min": 0.0,
            "step": 0.01,
        },
    )
    def factory_remove_false_positives_by_foreground(
        self, foreground: Image, threshold: float = 0.5
    ) -> None:
        """Remove false positives from a segmentation layer using a foreground probability layer."""

        if self.widget_layer_select.layer.value is None:
            log(
                "No Segmentation layer found!", thread="Postprocessing", level="WARNING"
            )
            return

        ps_segmentation = PanSegImage.from_napari_layer(
            self.widget_layer_select.layer.value
        )
        ps_foreground = PanSegImage.from_napari_layer(foreground)
        self.widget_layer_select.layer.value.visible = False

        schedule_task(
            remove_false_positives_by_foreground_probability_task,
            task_kwargs={
                "segmentation": ps_segmentation,
                "foreground": ps_foreground,
                "threshold": threshold,
            },
        )

    @magic_factory(
        call_button="Split/Merge Instances by Nuclei",
        segmentation_cells={"label": "Cell Segmentation"},
        segmentation_nuclei={"label": "Nuclear Segmentation"},
        boundary_pmaps={"label": "Boundary image"},
        threshold={
            "label": "Boundary Threshold (%)",
            "tooltip": "Set the percentage range for merging (first value) and splitting (second value) cells. "
            'For example, "33" means cells with less than 33% overlap with nuclei are merged, and '
            '"66" means cells with more than 66% overlap are split.',
            "widget_type": "FloatRangeSlider",
            "max": 100,
            "min": 0,
            "step": 0.1,
        },
        quantile={
            "label": "Nuclei Size Filter (%)",
            "tooltip": "Set the size range to filter nuclei, represented as percentages. "
            'For example, "0.3" excludes the smallest 30%, and "99.9" excludes the largest 0.1% of nuclei.',
            "widget_type": "FloatRangeSlider",
            "max": 100,
            "min": 0,
            "step": 0.1,
        },
    )
    def factory_fix_segmentation_by_nuclei(
        self,
        segmentation_cells: Labels,
        segmentation_nuclei: Labels,
        boundary_pmaps: Image | None = None,
        threshold=(33, 66),
        quantile=(0.3, 99.9),
    ) -> None:
        """
        Widget for correcting over- and under-segmentation of cells based on nuclei segmentation.

        This widget allows users to adjust cell segmentation by leveraging nuclei segmentation. It supports
        merging over-segmented cells and splitting under-segmented cells, with optional boundary refinement.

        Args:
            segmentation_cells (Labels): Input layer representing segmented cell instances.
            segmentation_nuclei (Labels): Input layer representing segmented nuclei instances.
            boundary_pmaps (Image | None, optional): Optional boundary probability map (same shape as input layers).
                Higher values indicate probable cell boundaries, used to refine segmentation.
            threshold (tuple[float, float], optional): Merge and split thresholds as percentages (0-100).
                - The first value is the merge threshold: cells with nuclei overlap below this value are merged.
                - The second value is the split threshold: cells with nuclei overlap above this value are split.
                Default is (33, 66).
            quantile (tuple[float, float], optional): Minimum and maximum quantile values for filtering nuclei sizes (0-100).
                - The first value excludes the smallest nuclei (e.g., "0.3" excludes the smallest 0.3%).
                - The second value excludes the largest nuclei (e.g., "99.9" excludes the largest 0.1%).
                Default is (0.3, 99.9).

        Returns:
            None
        """
        ps_seg_cel = PanSegImage.from_napari_layer(segmentation_cells)
        ps_seg_nuc = PanSegImage.from_napari_layer(segmentation_nuclei)
        ps_pmap_cell_boundary = (
            PanSegImage.from_napari_layer(boundary_pmaps) if boundary_pmaps else None
        )

        # Normalize percentages to fractions
        threshold_merge = threshold[0] / 100
        threshold_split = threshold[1] / 100
        quantile_min = quantile[0] / 100
        quantile_max = quantile[1] / 100
        self.widget_layer_select.layer.value.visible = False

        return schedule_task(
            fix_over_under_segmentation_from_nuclei_task,
            task_kwargs={
                "cell_seg": ps_seg_cel,
                "nuclei_seg": ps_seg_nuc,
                "threshold_merge": threshold_merge,
                "threshold_split": threshold_split,
                "quantile_min": quantile_min,
                "quantile_max": quantile_max,
                "boundary": ps_pmap_cell_boundary,
            },
        )

    def _on_layer_insertion(self, layer):
        self.widget_layer_select.layer.choices = get_layers(SemanticType.SEGMENTATION)
        self.widget_remove_false_positives_by_foreground.foreground.choices = (
            get_layers(SemanticType.RAW)
        )
        self.widget_fix_segmentation_by_nuclei.segmentation_cells.choices = get_layers(
            SemanticType.SEGMENTATION
        )
        self.widget_fix_segmentation_by_nuclei.segmentation_nuclei.choices = get_layers(
            SemanticType.SEGMENTATION
        )
        self.widget_fix_segmentation_by_nuclei.boundary_pmaps.choices = get_layers(
            SemanticType.PREDICTION
        )

        if layer not in self.widget_layer_select.layer.choices:
            logger.debug("")
        else:
            self.widget_layer_select.layer.value = layer

    def update_layer_selection(self, event):
        logger.debug(
            f"Updating postprocessing layer selection: {event.value}, {event.type}"
        )
        segmentations = get_layers(SemanticType.SEGMENTATION)
        self.widget_layer_select.layer.choices = segmentations
        self.widget_remove_false_positives_by_foreground.foreground.choices = (
            get_layers(SemanticType.RAW)
        )
        self.widget_fix_segmentation_by_nuclei.segmentation_cells.choices = (
            segmentations
        )
        self.widget_fix_segmentation_by_nuclei.segmentation_nuclei.choices = (
            segmentations
        )
        self.widget_fix_segmentation_by_nuclei.boundary_pmaps.choices = get_layers(
            SemanticType.PREDICTION
        )

        if event.type == "inserted":
            if event.value._metadata.get("semantic_type", None) == SemanticType.RAW:
                self.widget_remove_false_positives_by_foreground.foreground.value = (
                    event.value
                )
            elif (
                event.value._metadata.get("semantic_type", None)
                == SemanticType.SEGMENTATION
            ):
                self.widget_layer_select.layer.value = event.value
                self.widget_fix_segmentation_by_nuclei.segmentation_cells.value = (
                    event.value
                )
                self.widget_fix_segmentation_by_nuclei.segmentation_nuclei.value = (
                    event.value
                )
            elif (
                event.value._metadata.get("semantic_type", None)
                == SemanticType.PREDICTION
            ):
                self.widget_fix_segmentation_by_nuclei.boundary_pmaps.value = (
                    event.value
                )
