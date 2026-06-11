import timeit
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import napari
import napari.settings
from magicgui.widgets import Container, Label, ProgressBar, PushButton, Widget
from napari.layers import Layer
from napari.qt.threading import create_worker
from psygnal import evented
from psygnal.qt import start_emitting_from_queue
from qtpy import QtGui, QtWidgets
from rich.console import Console

from panseg import logger
from panseg.core.image import PanSegImage, SemanticType
from panseg.tasks.workflow_handler import Task_message
from panseg.viewer_napari import log

console = Console()


def _return_value_if_widget(x):
    if isinstance(x, Widget):
        return x.value
    return x


def div(text: str = "", divider=True):
    """Returns a divider widget
    Can put up to 54 chars headline into it.
    """
    resources = Path(__file__).resolve().parent.parent.parent / "resources"
    space = "\u00a0"
    if text:
        if len(text) > 54:
            logger.warning(
                "Divider text too long, might not be displayed correctly!\n"
                f"Text: {text}"
            )
        ql = QtWidgets.QLabel()
        text_len = ql.fontMetrics().boundingRect(text).width()
        # 490px target length, 172px div png
        needed_ws = (490 - 172) - (text_len)
        # length of white space ~3.5px, char length less, needs to be balanced
        n_ws = int((needed_ws / 4.1) + (text_len * 0.13))

        centered_text = text.center(n_ws, space)  # wraps text in non-breaking spaces
        if divider:
            w = Label(
                value=(
                    f"<img src={resources / 'div1.png'}>"
                    f"{centered_text}"
                    f"<img src={resources / 'div2.png'}>"
                )
            )
        else:
            w = Label(value=(f"{text.ljust(n_ws, space)}"))
    else:
        w = Label(value=f"<img src={resources / 'div.png'}>")

    font = QtGui.QFont()
    font.setBold(True)
    w.native.setFont(font)

    return w


def get_layers(
    s_type: Optional[SemanticType | Iterable[SemanticType]] = None,
) -> list[Layer]:
    """Get layers of specific type, e.g. raw, lable, prediction, segmentation"""
    log(f"get_layers called with filter: {s_type}", "utils", level="DEBUG")
    viewer = napari.current_viewer()
    if viewer is None:
        return []

    ll = viewer.layers

    relevant_layers = []
    if s_type is not None:
        if not isinstance(s_type, Iterable):
            s_type = (s_type,)
        for layer in ll:
            if layer._metadata.get("semantic_type", False) in s_type:
                relevant_layers.append(layer)
    else:
        relevant_layers = ll
    return relevant_layers


def add_ps_image_to_viewer(layer: PanSegImage, replace: bool = False) -> None:
    """Add a PanSegImage to the viewer.

    Args:
        layer (PanSegImage): PanSegImage to be added to the viewer.
        replace (bool): If True, the layer with the same name will be replaced. Defaults to False.
    """
    viewer = napari.current_viewer()
    data, meta, layer_type = layer.to_napari_layer_tuple()

    name = meta.get("name", None)
    if replace and name in viewer.layers:
        viewer.layers.remove(name)

    if layer_type == "image":
        viewer.add_image(data, **meta)
    elif layer_type == "labels":
        viewer.add_labels(data, **meta)


@evented
@dataclass()
class PBar_Tracker:
    """Keeps track of a progress bar

    Attributes:
        progress: must change every step to trigger update
        total: number of steps the progress bar should have.
            Set to 0 to have an indetermined progress bar.
    """

    progress: int = 0
    total: int = 0


def update_progressbar(pbar: ProgressBar, tracker: PBar_Tracker):
    """Increment progress bar. Creates a callback to be attached to a tracker.
    Args:
        pbar: progress bar to update
        tracker: Tracker instance of this progress bar.

    Returns:
        Update callback function
    """
    return lambda: (setattr(pbar, "max", tracker.total), pbar.increment(1))


def schedule_task(
    task: Callable,
    task_kwargs: dict,
) -> None:
    """Schedule a task to be executed in a separate thread and update the widgets with the result.

    Args:
        task (Callable): Function to be executed, the function should be a workflow task,
            and return a PanSegImage or a tuple/list of PanSegImage, or None.
        task_kwargs (dict): Keyword arguments for the function.
    """

    if hasattr(task, "__panseg_task__"):
        task_name = task.__panseg_task__
    else:
        raise ValueError(f"Function {task.__name__} is not a PanSeg task.")

    timer_start = timeit.default_timer()

    def on_done(task_result: PanSegImage | list[PanSegImage] | Task_message | None):
        timer = timeit.default_timer() - timer_start
        log(f"{task_name} complete in {timer:.2f}s", thread="Task")

        if isinstance(task_result, PanSegImage):
            add_ps_image_to_viewer(task_result, replace=True)

        elif isinstance(task_result, (tuple, list)):
            for ps_im in task_result:
                if not isinstance(ps_im, PanSegImage):
                    raise ValueError(
                        f"Task {task_name} returned an unexpected value {task_result}"
                    )

            for ps_im in task_result:
                add_ps_image_to_viewer(ps_im, replace=True)

        elif isinstance(task_result, Task_message):
            console.print(task_result.trace)
            log(task_result.message, thread=task_result.name, level=task_result.level)
            return None
        elif task_result is None:
            return None

        else:
            raise ValueError(
                f"Task {task_name} returned an unexpected value {task_result}"
            )

    # Setup progress bar
    pbar = None
    if "_pbar" in task_kwargs:
        tracker = PBar_Tracker()
        pbar = task_kwargs.pop("_pbar")
        pbar.max = tracker.total
        pbar.visible = True
        tracker.events.progress.connect(
            update_progressbar(pbar, tracker), thread="main"
        )
        task_kwargs["_tracker"] = tracker
        start_emitting_from_queue(msec=100)
    if hide_list := task_kwargs.pop("_to_hide", None):
        for to_hide in hide_list:
            to_hide.hide()

    # _progress displays spinner for all tasks, not necessary for magicgui progress bar.
    worker = create_worker(task, **task_kwargs, _progress=True)
    worker.returned.connect(on_done)

    # Hide progress bar after task
    if pbar is not None:
        worker.returned.connect(lambda _: setattr(pbar, "visible", False))
        worker.errored.connect(lambda _: setattr(pbar, "visible", False))
    if hide_list is not None:
        worker.returned.connect(lambda _: [w.show() for w in hide_list])
        worker.errored.connect(lambda _: [w.show() for w in hide_list])

    worker.start()
    log(f"{task_name} started", thread="Task")
    return None


def increase_font_size():
    try:
        settings = napari.settings.get_settings()
        settings.appearance.font_size += 1
    except ValueError:
        log("Font size can't be increased further!", thread="Font", level="Warning")


def decrease_font_size():
    try:
        settings = napari.settings.get_settings()
        settings.appearance.font_size -= 1
    except ValueError:
        log("Font size can't be reduced further!", thread="Font", level="Warning")


class Help_text:
    def __init__(self):
        logger.debug("Help text init")
        self.docs_url = "https://kreshuklab.github.io/panseg/latest/"

    def get_doc_container(self, text="", sub_url="") -> Container:
        """Creates a container with a documentation button and a logo."""
        logger.debug("get_doc_container called!")

        self.docs_url += sub_url
        button = PushButton(text="Help")
        button.max_width = 80
        button.max_height = 24
        button.changed.connect(self.open_docs)
        container = Container(
            widgets=[button],
            label=text,
            layout="horizontal",
            labels=False,
        )
        container[0].show()
        container.margins = [25, 0, 0, 0]
        return Container(widgets=[container], labels=True, layout="horizontal")

    def open_docs(self, button):
        logger.debug("open_docs called!")
        """Open the documentation URL in the default web browser when the button is clicked."""
        webbrowser.open(self.docs_url, new=0, autoraise=True)
        logger.info(f"Docs webpage opened: {self.docs_url}")
        return button
