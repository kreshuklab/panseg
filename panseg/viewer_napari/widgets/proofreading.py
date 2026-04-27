import os
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time

import h5py
import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Label
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils import CyclicLabelColormap
from pydantic import BaseModel, Field
from qtpy.QtCore import QMutex, Qt

from panseg import logger
from panseg.core.image import ImageProperties, PanSegImage, SemanticType
from panseg.functionals.proofreading.split_merge_tools import split_merge_from_seeds
from panseg.functionals.proofreading.utils import get_bboxes
from panseg.io import H5_EXTENSIONS
from panseg.viewer_napari import log
from panseg.viewer_napari.widgets.utils import Help_text, div, get_layers

DEFAULT_KEY_BINDING_PROOFREAD = "n"
DEFAULT_KEY_BINDING_CLEAN = "j"
SCRIBBLES_LAYER_NAME = "Scribbles"
CORRECTED_CELLS_LAYER_NAME = "Correct Labels"
MAX_UNDO_ACTIONS = 10

correct_cells_cmap = CyclicLabelColormap(
    colors=[
        (0.76388469, 0.02003777, 0.61156412, 1.0),
        (0.76388469, 0.02003777, 0.61156412, 1.0),
    ],
    name=CORRECTED_CELLS_LAYER_NAME,
)

try:
    MAX_UNDO_ACTIONS = int(os.getenv("PANSEG_MAX_UNDO_ACTIONS", str(MAX_UNDO_ACTIONS)))
except ValueError:
    log(
        "Invalid value for PANSEG_MAX_UNDO_ACTIONS, using default: 10",
        thread="Proofreading",
        level="warning",
    )


def copy_if_not_none(obj):
    """Returns a copy of the object if it's not None."""
    return None if obj is None else obj.copy()


def get_current_viewer_wrapper() -> napari.Viewer:
    """Returns the current Napari viewer instance."""
    viewer = napari.current_viewer()
    if viewer is None:
        log(
            "No viewer found. Please open a viewer and try again.",
            thread="Get Current Viewer",
            level="error",
        )
        raise RuntimeError("No viewer found. Please open a viewer and try again.")
    return viewer


def update_layer(
    data: np.ndarray, layer_name: str, scale: tuple[float, ...], **kwargs
) -> None:
    """Updates a layer in the viewer with new data.

    Args:
        data (np.ndarray): The new data to update the layer with.
        layer_name (str): The name of the layer to update.
    """
    viewer = get_current_viewer_wrapper()
    if layer_name in viewer.layers:
        viewer.layers[layer_name].data = data
        viewer.layers[layer_name].scale = scale  # type: ignore
        viewer.layers[layer_name].refresh()

    else:
        viewer.add_labels(data, name=layer_name, scale=scale, **kwargs)


def update_corrected_cells_mask_layer(
    data: np.ndarray, scale: tuple[float, ...]
) -> None:
    """Updates the corrected cells mask layer in the viewer with new data.

    Args:
        data (np.ndarray): The new data to update the layer with.
    """
    update_layer(
        data,
        CORRECTED_CELLS_LAYER_NAME,
        scale=scale,
        colormap=correct_cells_cmap,
        opacity=1,
    )


def update_scribbles_layer(data: np.ndarray, scale: tuple[float, ...]) -> None:
    """Updates the scribbles layer in the viewer with new data.

    Args:
        data (np.ndarray): The new data to update the layer with.
    """
    update_layer(data, SCRIBBLES_LAYER_NAME, scale=scale)


def update_region(
    data: np.ndarray,
    layer_name: str,
    region_slice: tuple[slice, ...],
    scale: tuple[float, ...],
) -> None:
    """Updates a region of a layer in the viewer with new data.

    Args:
        data (np.ndarray): The new data to update the layer with.
        layer_name (str): The name of the layer to update.
        region_slice (tuple[slice, ...]): The region slice to update.
    """
    viewer = get_current_viewer_wrapper()
    if layer_name in viewer.layers:
        viewer.layers[layer_name].data[region_slice] = data
        viewer.layers[layer_name].scale = scale  # type: ignore
        viewer.layers[layer_name].refresh()
    else:
        raise ValueError(f"Layer {layer_name} not found in viewer")


def get_layer_data(layer_name: str) -> np.ndarray:
    """Returns the data of a layer in the viewer.

    Args:
        layer_name (str): The name of the layer to get the data from.

    Returns:
        np.ndarray: The data of the layer.
    """
    viewer = get_current_viewer_wrapper()
    if layer_name not in viewer.layers:
        log(
            f"Layer {layer_name} not found in viewer",
            thread="Get Layer Data",
            level="error",
        )
        raise ValueError(f"Layer {layer_name} not found in viewer")
    return viewer.layers[layer_name].data


def get_layer_region_data(
    layer_name: str, region_slice: tuple[slice, ...]
) -> np.ndarray:
    """Returns a region of the data of a layer in the viewer.

    Args:
        layer_name (str): The name of the layer to get the data from.
        region_slice (tuple[slice, ...]): The region slice to get the data from.

    Returns:
        np.ndarray: The data of the region.
    """
    viewer = get_current_viewer_wrapper()
    if layer_name not in viewer.layers:
        log(
            f"Layer {layer_name} not found in viewer",
            thread="Get Layer Region Data",
            level="error",
        )
        raise ValueError(f"Layer {layer_name} not found in viewer")
    return viewer.layers[layer_name].data[region_slice]


def preserve_labels(layer_name: str) -> None:
    """Preserves labels on a layer in the viewer.

    Args:
        layer_name (str): The name of the layer to preserve.
    """
    viewer = get_current_viewer_wrapper()
    if layer_name not in viewer.layers:
        log(
            f"Layer {layer_name} not found in viewer",
            thread="preserve_labels",
            level="error",
        )
        raise ValueError(f"Layer {layer_name} not found in viewer")
    viewer.layers[layer_name].preserve_labels = True  # type: ignore
    viewer.layers[layer_name].refresh()


class ProofreadingState(BaseModel):
    """Model for storing proofreading state."""

    active: bool = False
    lock: bool = False
    current_seg_layer_name: str | None = None
    corrected_cells: set = Field(default_factory=set)
    bboxes: dict[int, list[list[int]]] | None = None
    seg_properties: ImageProperties | None = None
    history_undo: deque = deque(maxlen=MAX_UNDO_ACTIONS)
    history_redo: deque = deque(maxlen=MAX_UNDO_ACTIONS)


# We need to use the dataclass decorator to avoid issues with the BaseModel serialization of numpy arrays
@dataclass()
class ProofreadingData:
    """Model for storing proofreading data."""

    segmentation: np.ndarray
    corrected_cells: set
    corrected_cells_mask: np.ndarray
    bboxes: dict[int, list[list[int]]]


class ProofreadingHandler:
    """Handler for managing segmentation proofreading and corrections.

    This class handles the state of the segmentation, corrected cells, scribbles,
    and bounding boxes, while allowing for interactions such as undoing changes.
    """

    def __init__(self):
        """Initializes the ProofreadingHandler with an inactive state."""
        self._state = ProofreadingState()
        self._scale = None
        self.mutex = QMutex()

    @contextmanager
    def lock_manager(self, timeout: int = 30):
        """Blocking context manager for locking and unlocking proofreading handler."""
        success = self.mutex.tryLock(timeout)
        if not success:
            raise TimeoutError("Failed to acquire lock for ProofreadingHandler")

        try:
            yield
        finally:
            self.mutex.unlock()

    def is_locked(self) -> bool:
        """Checks if the proofreading handler is locked."""
        was_unlocked = self.mutex.tryLock(0)
        if was_unlocked:
            self.mutex.unlock()
        return not was_unlocked

    # Proofreading state properties
    @property
    def active(self) -> bool:
        """Returns the proofreading status."""
        return self._state.active

    @property
    def scale(self) -> tuple[float, ...]:
        """Returns the current scale of the segmentation."""
        if self._scale is None:
            raise ValueError("Scale not found")
        return self._scale

    @property
    def seg_layer_name(self) -> str:
        """Returns the current segmentation layer name."""
        if self._state.current_seg_layer_name is None:
            raise ValueError("Segmentation layer not found")
        return self._state.current_seg_layer_name

    @property
    def seg_properties(self) -> ImageProperties:
        """Returns the properties of the current segmentation."""

        if self._state.seg_properties is None:
            raise ValueError("Segmentation properties not found")
        return self._state.seg_properties

    @property
    def segmentation(self) -> np.ndarray:
        """Returns the current segmentation data."""
        if self._state.current_seg_layer_name is None:
            raise ValueError("Segmentation layer not found")
        return get_layer_data(self._state.current_seg_layer_name)

    @property
    def scribbles(self) -> np.ndarray:
        """Returns the current scribbles."""
        return get_layer_data(SCRIBBLES_LAYER_NAME)

    def reset_scribbles(self) -> None:
        """Resets the scribble data to an empty state."""
        if not self.active:
            log(
                "Proofreading widget not initialized. Run the proofreading widget tool once first",
                thread="Reset Scribbles",
            )
            return
        update_layer(
            np.zeros_like(self.segmentation), SCRIBBLES_LAYER_NAME, scale=self.scale
        )

    @property
    def corrected_cells(self) -> set:
        """Returns the set of corrected cells."""
        return self._state.corrected_cells

    @property
    def corrected_cells_mask(self) -> np.ndarray:
        """Returns the mask for corrected cells."""
        return get_layer_data(CORRECTED_CELLS_LAYER_NAME)

    def reset_corrected(self) -> None:
        """Resets the corrected cells mask to an empty state."""
        if not self.active:
            log(
                "Proofreading widget not initialized. Run the proofreading widget tool once first",
                thread="Reset Corrected Cells Mask",
            )
            return None

        self._state.corrected_cells = set()
        update_layer(
            np.zeros_like(self.segmentation),
            CORRECTED_CELLS_LAYER_NAME,
            scale=self.scale,
            colormap=correct_cells_cmap,
            opacity=1,
        )

    @property
    def bboxes(self) -> dict[int, list[list[int]]]:
        """Returns the bounding boxes (bboxes) for the segmentation."""
        if self._state.bboxes is None:
            self.reset_bboxes()

        assert self._state.bboxes is not None
        return self._state.bboxes

    def reset_bboxes(self) -> None:
        """Resets the bounding boxes (bboxes) for the segmentation."""
        if not self.active:
            log(
                "Proofreading widget not initialized. Run the proofreading widget tool once first",
                thread="Reset Bboxes",
            )
            raise ValueError(
                "Proofreading widget not initialized. Run the proofreading widget tool once first"
            )
        self._state.bboxes = get_bboxes(self.segmentation, slack=(0, 0, 0))

    @property
    def max_label(self) -> int:
        """Returns the maximum label value in the segmentation."""
        return self.segmentation.max()

    # Global properties
    def reset(self) -> None:
        """Resets the proofreading handler to its initial state."""
        self._state = ProofreadingState()

    def setup(self, segmentation: PanSegImage):
        """Initializes the proofreading handler with a new segmentation.

        Args:
            segmentation (PanSegImage): The segmentation image to set up.
        """
        self.reset()
        self._scale = segmentation.scale
        self._state = ProofreadingState(
            active=True,
            current_seg_layer_name=segmentation.name,
            seg_properties=segmentation.properties,
        )
        self.reset_bboxes()
        self.reset_corrected()
        self.reset_scribbles()

    ## Undo/Redo actions
    def _capture_state(self) -> ProofreadingData:
        """Captures the current state of the handler."""
        return ProofreadingData(
            segmentation=self.segmentation.copy(),
            corrected_cells=self.corrected_cells.copy(),
            corrected_cells_mask=self.corrected_cells_mask.copy(),
            bboxes=self.bboxes.copy(),
        )

    def save_to_history(self) -> None:
        """Saves the current state to the undo history and clears the redo history."""
        self._state.history_undo.append(self._capture_state())
        self._state.history_redo.clear()  # Clear the redo stack when new actions are made

    def _restore_state(self, state: ProofreadingData) -> None:
        """Restores a given state."""
        update_layer(
            data=state.segmentation, layer_name=self.seg_layer_name, scale=self.scale
        )
        update_layer(
            data=state.corrected_cells_mask,
            layer_name=CORRECTED_CELLS_LAYER_NAME,
            scale=self.scale,
        )

        self.reset_scribbles()
        self._state.corrected_cells = state.corrected_cells
        self._state.bboxes = state.bboxes

    def _perform_undo_redo(
        self,
        history_pop: deque,
        history_append: deque,
        action_name: str,
    ):
        """Generalized function to handle undo and redo actions."""
        if not history_pop:
            log(f"No more actions to {action_name}.", thread=action_name.capitalize())
            return

        current_state = self._capture_state()
        last_state = history_pop.pop()

        history_append.append(current_state)
        self._restore_state(last_state)
        log(f"{action_name.capitalize()} completed", thread=action_name.capitalize())

    def undo(self):
        """Restores the previous state from the history stack."""
        self._perform_undo_redo(
            history_pop=self._state.history_undo,
            history_append=self._state.history_redo,
            action_name="undo",
        )

    def redo(self):
        """Restores the next state from the redo history."""
        self._perform_undo_redo(
            history_pop=self._state.history_redo,
            history_append=self._state.history_undo,
            action_name="redo",
        )

    def save_state_to_disk(
        self, filepath: Path, raw: Image | None, pmap: Image | None = None
    ):
        """Saves the current state to disk as an HDF5 file."""

        if filepath.suffix.lower() not in H5_EXTENSIONS:
            log(
                f"Invalid file extension: {filepath.suffix}. Please use a valid HDF5 file extensions: {H5_EXTENSIONS}",
                thread="Save State",
            )
            return

        viewer = get_current_viewer_wrapper()

        segmentation_layer = viewer.layers[self.seg_layer_name]
        assert isinstance(segmentation_layer, Labels)
        ps_segmentation = PanSegImage.from_napari_layer(segmentation_layer)
        ps_segmentation.to_h5(filepath, key="label", mode="w")

        mask_layer = self.corrected_cells_mask

        with h5py.File(filepath, "a") as f:
            f.create_dataset(name="mask", data=mask_layer)
            f["mask"].attrs["corrected_cells"] = list(self.corrected_cells)

        for name, image in [("raw", raw), ("pmap", pmap)]:
            if image is not None:
                ps_image = PanSegImage.from_napari_layer(image)
                ps_image.to_h5(filepath, key=name)

        log(f"State saved to {filepath}", thread="Save State")

    def load_state_from_disk(self, filepath: Path):
        """Loads a saved state from disk."""

        if not filepath.exists():
            log(f"File not found: {filepath}", thread="Load State")
            return None

        viewer = get_current_viewer_wrapper()
        ps_segmentation = PanSegImage.from_h5(filepath, key="label")

        with h5py.File(filepath, "r") as f:
            if "mask" not in f:
                log("Corrected cells mask not found in file", thread="Load State")
                corrected_cells = set()
                mask = np.zeros_like(ps_segmentation._data)
            else:
                corrected_cells = set(f["mask"].attrs["corrected_cells"])  # type: ignore
                mask: np.ndarray = f["mask"][...]  # type: ignore

            for name in ["raw", "pmap"]:
                if name in f:
                    ps_image = PanSegImage.from_h5(filepath, key=name)
                    if ps_image.name not in [layer.name for layer in viewer.layers]:
                        ps_image_layer_tuple = ps_image.to_napari_layer_tuple()
                        viewer._add_layer_from_data(*ps_image_layer_tuple)
                    else:
                        log(
                            f"Layer {ps_image.name} already exists in viewer",
                            thread="Load State",
                        )

        # Create the segmentation layer
        if ps_segmentation.name in [layer.name for layer in viewer.layers]:
            viewer.layers.remove(ps_segmentation.name)  # pyright: ignore

        ps_image_layer_tuple = ps_segmentation.to_napari_layer_tuple()
        viewer._add_layer_from_data(*ps_image_layer_tuple)
        self.setup(ps_segmentation)

        update_layer(
            mask,
            CORRECTED_CELLS_LAYER_NAME,
            scale=self.scale,
            colormap=correct_cells_cmap,
            opacity=1,
        )
        self._state.corrected_cells = corrected_cells
        log(f"State loaded from {filepath}", thread="Load State")

    # Corrected cells Operations
    def _toggle_corrected_cell(self, cell_id: int):
        """Adds or removes the cell from the corrected set.

        Args:
            cell_id (int): The ID of the cell to toggle.
        """
        if cell_id in self._state.corrected_cells:
            self._state.corrected_cells.remove(cell_id)
        else:
            self._state.corrected_cells.add(cell_id)

    def _update_masks(self, cell_id: int):
        """Updates the corrected cells mask with the toggled cell.

        Args:
            cell_id (int): The ID of the cell to update.
        """
        id_mask = self.segmentation == cell_id

        corrected_mask = get_layer_data(CORRECTED_CELLS_LAYER_NAME)
        corrected_mask[id_mask] += 1
        corrected_mask[id_mask] %= 2
        update_corrected_cells_mask_layer(corrected_mask, scale=self.scale)

    def toggle_corrected_cell(self, cell_id: int):
        """Toggles a cell as corrected or not.

        Args:
            cell_id (int): The ID of the cell to toggle.
        """
        self._toggle_corrected_cell(cell_id)
        self._update_masks(cell_id)

    def update_corrected_cells_mask_slice_to_viewer(
        self, slice_data: np.ndarray, region_slice: tuple[slice, ...]
    ):
        """Updates a slice of the corrected cells mask in the viewer.

        Args:
            slice_data (np.ndarray): The data to update the slice with.
            region_slice (tuple[slice, ...]): The region slice to update.
        """
        update_region(
            slice_data, CORRECTED_CELLS_LAYER_NAME, region_slice, scale=self.scale
        )
        preserve_labels(CORRECTED_CELLS_LAYER_NAME)

    def update_after_proofreading(
        self,
        seg_slice: np.ndarray,
        region_slice: tuple[slice, ...],
        bbox: dict[int, list[list[int]]],
    ):
        """Updates the viewer after proofreading is completed.

        Args:
            seg_slice (np.ndarray): The segmentation slice to update.
            region_slice (tuple[slice, ...]): The region slice to update in the viewer.
            bbox (dict): The bounding box to update.
        """
        self.bboxes.update(bbox)
        update_region(
            data=seg_slice,
            layer_name=self.seg_layer_name,
            region_slice=region_slice,
            scale=self.scale,
        )


segmentation_handler = ProofreadingHandler()


@magicgui(call_button=f"Clean scribbles - < {DEFAULT_KEY_BINDING_CLEAN} >")
def widget_clean_scribble(viewer: napari.Viewer):
    """Cleans the scribbles layer in the Napari viewer."""
    if not segmentation_handler.active:
        log(
            "Proofreading widget not initialized. Run the proofreading widget tool once first",
            thread="Clean scribble",
        )
        return

    if "Scribbles" not in viewer.layers:
        log(
            "Scribble Layer not defined. Run the proofreading widget tool once first",
            thread="Clean scribble",
        )
        return

    segmentation_handler.reset_scribbles()


def widget_add_label_to_corrected(viewer: napari.Viewer, position: tuple[int, ...]):
    """Adds or removes a label at a given position to/from the corrected cells.

    Args:
        position (tuple[int, ...]): The position of the cell in the viewer.
    """
    if CORRECTED_CELLS_LAYER_NAME not in viewer.layers:
        raise ValueError("Corrected cells layer not found in viewer")

    raster_position = [
        int(p / s) for p, s in zip(position, segmentation_handler.scale, strict=True)
    ]
    cell_id = segmentation_handler.segmentation[*raster_position]
    segmentation_handler.toggle_corrected_cell(cell_id)


help_text_container = Help_text()


def get_widget_tab_help_text(help_text_container):
    help_text = (
        "<strong>Proofreading:</strong><br>"
        "Correct segmentation by interactively merging/splitting labels."
    )

    tab_help = help_text_container.get_doc_container(
        help_text,
        sub_url="chapters/panseg_interactive_napari/proofreading/",
    )
    return tab_help


# before initialization
widget_tab_help_text = get_widget_tab_help_text(help_text_container)

# after initialization
widget_label_split_merge = help_text_container.get_doc_container(
    text="<strong>INSTRUCTIONS:</strong><br>Mark labels by drawing onto the `Scribbles` layer"
    " in different colors.<br>Labels marked with <strong>the same color</strong>"
    " will be merged<br>Labels marked with <strong>different colors</strong> will be split.",
)


def initialize_from_layer(segmentation: Labels, are_you_sure: bool = False) -> None:
    if segmentation.name in [
        SCRIBBLES_LAYER_NAME,
        CORRECTED_CELLS_LAYER_NAME,
    ]:  # Avoid re-initializing with proofreading helper layers
        log(
            "Scribble or corrected cells layer is not intended to be proofread, choose a segmentation",
            thread="Proofreading tool",
            level="error",
        )
        return

    if segmentation_handler.active and not are_you_sure:
        log(
            "Proofreading is already initialized. Are you sure you want to reset everything?",
            thread="Proofreading tool",
            level="warning",
        )
        widget_proofreading_initialisation.are_you_sure.show()
        widget_proofreading_initialisation.call_button.text = (
            "I understand, please re-initialise!!"  # type: ignore
        )
        return

    ps_segmentation = PanSegImage.from_napari_layer(segmentation)
    segmentation_handler.setup(ps_segmentation)

    widget_tab_help_text.hide()
    widget_proofreading_initialisation.are_you_sure.value = False
    widget_proofreading_initialisation.are_you_sure.hide()
    widget_proofreading_initialisation.call_button.text = "Re-initialize Proofreading"  # type: ignore
    widget_proofreading_initialisation.call_button.text = "Re-initialize Proofreading"  # type: ignore
    setup_proofreading_widget()
    log("Proofreading initialized", thread="Proofreading tool")

    viewer = get_current_viewer_wrapper()
    # Avoid re-initializing with proofreading helper layers
    widget_proofreading_initialisation.segmentation.choices = [
        layer
        for layer in viewer.layers
        if layer.name not in [SCRIBBLES_LAYER_NAME, CORRECTED_CELLS_LAYER_NAME]
    ]


def initialize_from_file(state: Path, are_you_sure: bool = False) -> None:
    if segmentation_handler.active and not are_you_sure:
        log(
            "Proofreading is already initialized. Are you sure you want to reset everything?",
            thread="Proofreading tool",
            level="warning",
        )
        widget_proofreading_initialisation.are_you_sure.show()
        widget_proofreading_initialisation.call_button.text = (
            "I understand, please re-initialise!!"  # type: ignore
        )
        return

    segmentation_handler.load_state_from_disk(state)

    widget_tab_help_text.hide()
    widget_proofreading_initialisation.are_you_sure.value = False
    widget_proofreading_initialisation.are_you_sure.hide()
    widget_proofreading_initialisation.call_button.text = "Re-initialize Proofreading"  # type: ignore
    setup_proofreading_widget()
    log("Proofreading initialized", thread="Proofreading tool")


@magicgui(
    call_button="Initialize Proofreading",
    mode={
        "label": "Mode",
        "choices": ["New", "Load from file"],
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
    },
    segmentation={
        "label": "Segmentation",
        "tooltip": "The segmentation layer to proofread",
    },
    filepath={
        "label": "Resume from file",
        "mode": "r",
        "filter": "*.h5",
        "tooltip": "Load a previous proofreading state from a h5 file",
    },
    are_you_sure={"label": "I understand this resets everything", "visible": False},
)
def widget_proofreading_initialisation(
    mode: str = "New",
    segmentation: Labels | None = None,
    filepath: Path | None = None,
    are_you_sure: bool = False,
) -> None:
    """Initializes the proofreading widget.

    Args:
        segmentation (Labels): The segmentation layer.
        state (Path | None): Path to a previous state file (optional).
    """
    if mode == "New":
        if segmentation is None:
            log(
                "No segmentation layer selected",
                thread="Proofreading tool",
                level="error",
            )
            return
        initialize_from_layer(segmentation, are_you_sure=are_you_sure)
    elif mode == "Load from file":
        if filepath is None:
            log("No state file selected", thread="Proofreading tool", level="error")
            return
        initialize_from_file(filepath, are_you_sure=are_you_sure)
        widget_save_state.filepath.value = filepath
    else:
        raise ValueError("Unknown mode")

    setup_proofreading_keybindings()


widget_proofreading_initialisation.are_you_sure.hide()
widget_proofreading_initialisation.filepath.hide()


@widget_proofreading_initialisation.mode.changed.connect
def _on_mode_changed(mode: str):
    if mode == "New":
        widget_proofreading_initialisation.segmentation.show()
        widget_proofreading_initialisation.filepath.hide()
    elif mode == "Load from file":
        widget_proofreading_initialisation.segmentation.hide()
        widget_proofreading_initialisation.filepath.show()


@magicgui(
    call_button=f"Split / Merge - < {DEFAULT_KEY_BINDING_PROOFREAD} >",
    image={
        "label": "Boundary image",
        "tooltip": "Probability map (prediction) or raw image of boundaries as reference",
    },
)
def widget_split_and_merge_from_scribbles(
    viewer: napari.Viewer,
    image: Image | None,
):
    """Splits or merges segments using scribbles as seeds for corrections.

    Args:
        image (Image): The probability map or raw image layer.
    """
    if not segmentation_handler.active:
        log(
            "Proofreading is not initialized. Run the initialization widget first.",
            thread="Proofreading tool",
        )
        return

    if image is None:
        log(
            "Please select a boundary image first!",
            thread="Proofreading tool",
        )
        return

    ps_image = PanSegImage.from_napari_layer(image)

    if ps_image.semantic_type == SemanticType.RAW:
        log(
            "Pmap/Image layer appears to be a raw image and not a boundary "
            "probability map. For the best proofreading results, try to use a "
            "boundaries probability layer (e.g. from the Run Prediction widget)",
            thread="Proofreading tool",
            level="warning",
        )

    if ps_image.is_multichannel:
        log(
            "Pmap/Image layer appears to be a multichannel image. "
            "Proofreading does not support multichannel images. ",
            thread="Proofreading tool",
            level="error",
        )

    @thread_worker(progress=True)
    def func():
        with segmentation_handler.lock_manager():
            if segmentation_handler.scribbles.sum() == 0:
                log("No scribbles found", thread="Proofreading tool")
                return None
            segmentation_handler.save_to_history()

            new_seg, region_slice, bboxes = split_merge_from_seeds(
                segmentation_handler.scribbles,
                segmentation_handler.segmentation,
                image=ps_image.get_data(),
                bboxes=segmentation_handler.bboxes,
                max_label=segmentation_handler.max_label,
                correct_labels=segmentation_handler.corrected_cells,
            )

            segmentation_handler.update_after_proofreading(
                new_seg, region_slice, bboxes
            )

    worker = func()  # type: ignore
    worker.start()
    return worker


widget_label_extraction = Label(
    value="Double click in move mode to select labels.\n"
    "Selected labels will be extracted to a new layer.",
)


@magicgui(call_button="Extract Corrected labels")
def widget_filter_segmentation() -> None:
    """Extracts corrected labels from the segmentation.

    Returns:
        Future[LayerDataTuple]: A future that will return the extracted segmentation layer.
    """
    if not segmentation_handler.active:
        log(
            "Proofreading widget not initialized. Run the proofreading widget tool once first",
            thread="Export correct labels",
            level="error",
        )
        raise ValueError(
            "Proofreading widget not initialized. Run the proofreading widget tool once first"
        )

    @thread_worker(progress=True)
    def func():
        with segmentation_handler.lock_manager():
            filtered_seg = segmentation_handler.segmentation.copy()
            filtered_seg[segmentation_handler.corrected_cells_mask == 0] = 0

            properties = segmentation_handler.seg_properties

            new_seg_properties = ImageProperties(
                name=f"{properties.name}_corrected",
                semantic_type=SemanticType.SEGMENTATION,
                voxel_size=properties.voxel_size,
                image_layout=properties.image_layout,
                original_voxel_size=properties.original_voxel_size,
            )
            new_ps_seg = PanSegImage(filtered_seg, new_seg_properties)
            new_seg_layer_tuple = new_ps_seg.to_napari_layer_tuple()

        return new_seg_layer_tuple

    def on_done(result):
        viewer = napari.current_viewer()
        if result is not None and viewer is not None:
            viewer._add_layer_from_data(*result)

    worker = func()  # type: ignore
    worker.returned.connect(on_done)
    worker.start()


widget_filter_segmentation.tooltip = (
    "Double right click to select a label and add it to the mask.\n"
    "Extract saves the masked labels on a new layer"
)


@magicgui(call_button="Undo Last Action")
def widget_undo():
    """Undo the last proofreading action."""
    if not segmentation_handler.active:
        log("Proofreading widget not initialized. Nothing to undo.", thread="Undo")
        return
    segmentation_handler.undo()


@magicgui(call_button="Redo Last Action")
def widget_redo():
    """Redo the last undone action."""
    if not segmentation_handler.active:
        log("Proofreading widget not initialized. Nothing to redo.", thread="Redo")
        return
    segmentation_handler.redo()


@magicgui(
    call_button="Save current proofreading snapshot",
    filepath={
        "label": "File path",
        "mode": "w",
        "filter": "*.h5",
        "tooltip": "Save as h5 file",
    },
    raw={
        "label": "Raw image",
        "tooltip": "Optional raw image for reference",
    },
    pmap={
        "label": "Probability map",
        "tooltip": "Optional probability map for reference",
    },
)
def widget_save_state(
    filepath: Path = Path.home(), raw: Image | None = None, pmap: Image | None = None
):
    """Saves the current proofreading state to disk.

    Args:
        filepath (str): The filepath to save the state to.

    """
    segmentation_handler.save_state_to_disk(filepath, raw=raw, pmap=pmap)


widget_save_div = div("Save proofreading")


def setup_proofreading_keybindings():
    """Sets up keybindings for the proofreading tool in Napari."""
    viewer = napari.current_viewer()
    if viewer is None:
        return

    @viewer.bind_key(DEFAULT_KEY_BINDING_PROOFREAD, overwrite=True)
    def _widget_split_and_merge_from_scribbles(_viewer: napari.Viewer):
        widget_split_and_merge_from_scribbles(viewer=_viewer)  # type: ignore

    @viewer.bind_key(DEFAULT_KEY_BINDING_CLEAN, overwrite=True)
    def _widget_clean_scribble(_viewer: napari.Viewer):
        widget_clean_scribble(viewer=_viewer)

    def _add_label_to_corrected(_viewer: napari.Viewer, event):
        # Maybe it would be better to run this callback only if the layer is active
        # if _viewer.layers.selection.active.name == CORRECTED_CELLS_LAYER_NAME:
        if CORRECTED_CELLS_LAYER_NAME in _viewer.layers:
            widget_add_label_to_corrected(viewer=viewer, position=event.position)

    viewer.mouse_double_click_callbacks.pop()
    viewer.mouse_double_click_callbacks.append(_add_label_to_corrected)


activation_list_proofreading = [
    widget_label_split_merge,
    widget_split_and_merge_from_scribbles,
    widget_clean_scribble,
    widget_label_extraction,
    widget_filter_segmentation,
    widget_undo,
    widget_redo,
    widget_save_div,
    widget_save_state,
]

for widget in activation_list_proofreading:
    widget.hide()


def setup_proofreading_widget():
    for widget in activation_list_proofreading:
        widget.show()


def update_layer_selection(event):
    """Updates layer drop-down menus"""
    logger.debug(f"Updating segmentation layer selection: {event.value}, {event.type}")
    raws = get_layers(SemanticType.RAW)
    predictions = get_layers(SemanticType.PREDICTION)
    segmentations = get_layers(SemanticType.SEGMENTATION)

    widget_proofreading_initialisation.segmentation.choices = segmentations
    widget_save_state.raw.choices = raws
    widget_save_state.pmap.choices = raws + predictions
    widget_split_and_merge_from_scribbles.image.choices = raws + predictions

    # Set values to inserted
    if event.type == "inserted":
        if (
            event.value._metadata.get("semantic_type", None)
            == SemanticType.SEGMENTATION
        ):
            widget_proofreading_initialisation.segmentation.value = event.value
