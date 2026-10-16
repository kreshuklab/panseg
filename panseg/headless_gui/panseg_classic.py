from enum import Enum
from pathlib import Path

from magicgui import magicgui
from magicgui.widgets import Container

########################################################################################################################
#
# Input Setup Widget
#
########################################################################################################################


class FilePickMode(Enum):
    File = "File"
    Directory = "Directory"

    @classmethod
    def to_choices(cls) -> list[str]:
        return [mode.value for mode in FilePickMode]


@magicgui(
    call_button=False,
    file_pick_mode={
        "label": "Input Mode",
        "tooltip": "Select the workflow to run",
        "choices": FilePickMode.to_choices(),
    },
    file={
        "label": "File",
        "mode": "r",
        "layout": "vertical",
        "tooltip": "Select the file to process one by one",
    },
    directory={
        "label": "Directory",
        "mode": "d",
        "tooltip": "Process all files in the directory; if it holds only subdirectories, each subdirectory's files are processed as one time series",
    },
)
def widget_input_model(
    file_pick_mode: str = FilePickMode.File.value,
    file: Path = Path(".").absolute(),
    directory: Path = Path(".").absolute(),
):
    pass


@widget_input_model.file_pick_mode.changed.connect
def _on_mode_change(file_pick_mode):
    if file_pick_mode == FilePickMode.File.value:
        widget_input_model.file.show()
        widget_input_model.directory.hide()
    else:
        widget_input_model.file.hide()
        widget_input_model.directory.show()


widget_input_model.directory.hide()


########################################################################################################################
#
# PanSeg Classic Workflow
#
########################################################################################################################
@magicgui(
    call_button="Run - PanSeg Classic",
)
def widget_setup_workflow_config():
    pass


widget_panseg_classic = Container(
    widgets=[widget_input_model, widget_setup_workflow_config], labels=False
)
