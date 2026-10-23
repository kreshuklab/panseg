import signal
from pathlib import Path
from typing import Optional

import psygnal
import rich.traceback
from qtpy import QtCore, QtGui, QtWidgets

import panseg
from panseg.workflow_gui.widgets import Workflow_widgets, logger

rich.traceback.install(
    show_locals=True,
    suppress=[psygnal],
)
signal.signal(signal.SIGINT, signal.SIG_DFL)


class Workflow_gui(Workflow_widgets):
    """Editor for workflow yaml files.

    Args:
        config_path:
            Optional Path to yaml input file.

    """

    def __init__(self, config_path: Optional[Path] = None, run=True):
        super().__init__()
        self.config_path = config_path
        self.config: Optional[dict] = None
        self.advanced = False

        # setup initial state
        if not isinstance(self.config_path, Path):
            logger.debug("No config, showing loader..")
            self.show_loader()
        else:
            logger.debug("Config provided")
            self.loader_w(config_path=self.config_path)

        iconpath = (Path(panseg.__path__[0]).parent) / "Menu" / "icon.png"
        QtWidgets.QApplication.instance().setWindowIcon(
            QtGui.QIcon(str(iconpath)),
        )

        self.main_window.native.setWindowTitle("PanSeg Workflow Editor")
        self.main_window.show(run=run)

    def show_loader(self):
        logger.debug("Changing to loader view")
        self.config = None
        self.content.clear()
        self.content.append(self.loader_w)  # pyright: ignore

        self.bottom_buttons.clear()
        self.bottom_buttons.append(self.exit)  # pyright: ignore

        [w.show() for w in self.content]
        [w.show() for w in self.bottom_buttons]

        mwn = self.main_window.native
        c_size = self.content.native.sizeHint()
        b_size = self.bottom_buttons.native.sizeHint()
        mwn.resize(c_size + b_size + QtCore.QSize(0, 80))

    def show_config(self):
        try:
            self.validate_config()
        except ValueError as e:
            logger.error(f"Workflow not valid, please choose a valid workflow!\n{e}")
            self.show_loader()
            return

        self.bottom_buttons.clear()
        self.bottom_buttons.extend((self.change_config, self.exit))  # pyright: ignore

        logger.debug("Changing to config view")
        self.content.clear()
        self.fill_config_c()

        [w.show() for w in self.content]
        [w.show() for w in self.bottom_buttons]

        self.switch_advanced_view(False)

        mwn = self.main_window.native
        c_size = self.content.native.sizeHint()
        b_size = self.bottom_buttons.native.sizeHint()
        mwn.resize(c_size + b_size + QtCore.QSize(0, 80))

    def validate_config(self):
        if not isinstance(self.config, dict):
            raise ValueError("Config was not parsed to dict")
        if "inputs" not in self.config:
            raise ValueError(
                "The workflow configuration does not contain an 'inputs' section."
            )
        if isinstance(self.config["inputs"], list):
            raise ValueError(
                "The 'inputs' section is a list of multiple input dicts: a "
                "headless batch configuration, not a single workflow. Run it "
                "with `panseg --config` instead of the workflow editor."
            )
        if "infos" not in self.config:
            raise ValueError(
                "The workflow configuration does not contain an 'infos' section."
            )
        if "list_tasks" not in self.config:
            raise ValueError(
                "The workflow configuration does not contain an 'list_tasks' section."
            )
        if "runner" not in self.config:
            logger.warning(
                "The workflow configuration does not contain a 'runner' section."
                " Using the default serial runner."
            )
            self.config["runner"] = "serial"


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    config = Path("examples/headless_workflow.yaml")
    # config = Path("examples/multi.yaml")
    # config = Path("examples/long.yaml")
    # config = None
    Workflow_gui(config)
