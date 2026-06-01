# pylint: disable=import-outside-toplevel

import argparse
import sys
from pathlib import Path
from typing import Optional

from panseg import logger
from panseg.__version__ import __version__
from panseg.utils import check_version, clean_models, load_config


def create_parser():
    """Create and return the argument parser for the CLI."""
    arg_parser = argparse.ArgumentParser(
        description="PanSeg: Cell/nuclei instance segmentation software"
    )
    arg_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Launch CLI from CONFIG (path to the YAML config file)",
    )
    arg_parser.add_argument(
        "--napari",
        "-n",
        action="store_true",
        help="Launch Napari GUI",
    )
    arg_parser.add_argument(
        "--train",
        type=Path,
        help="Launch training from CONFIG (path to the YAML config file)",
    )
    arg_parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Print PanSeg version",
    )
    arg_parser.add_argument(
        "--clean",
        action="store_true",
        help='Remove all models from "~/.panseg_models"',
    )
    arg_parser.add_argument(
        "--edit",
        "-e",
        type=Path,
        nargs="?",
        default=False,
        const=None,
        help="Launch GUI to edit a workflow yaml file. Optionally provide a path.",
        metavar="yaml",
    )
    arg_parser.add_argument(
        "--loglevel",
        choices=["ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set the level of the logger. DEBUG level will also show all warnings including compatibility warnings.",
    )
    return arg_parser


def launch_napari(loglevel: str = "INFO"):
    """Launch the Napari viewer."""
    import warnings

    import rich.traceback

    from panseg.viewer_napari.viewer import Panseg_viewer

    # Suppress FutureWarning about functools.partial from Python 3.13
    # This is a known issue with magicgui library until it's updated
    # Only suppress when not in DEBUG mode, show in DEBUG mode for developers
    if loglevel != "DEBUG":
        warnings.filterwarnings(
            "ignore",
            message=".*functools.partial will be a method descriptor.*",
            category=FutureWarning,
        )

    rich.traceback.install(
        show_locals=True,
        suppress=[],
    )

    Panseg_viewer().start_viewer()


def launch_workflow_headless(path: Path):
    """Run a workflow in headless mode."""
    from panseg.headless.headless import run_headless_workflow

    run_headless_workflow(path=path)


def launch_training(path: Path):
    """Launch the training"""
    config = load_config(path)

    from panseg.functionals.training.train import unet_training

    unet_training(*config)


def launch_editor(path: Optional[Path]):
    """Launch the workflow editor"""
    from panseg.workflow_gui.editor import Workflow_gui

    Workflow_gui(path)


def main():
    """Main function to parse arguments and call corresponding functionality."""
    parser = create_parser()
    args = parser.parse_args()
    check_version(__version__)

    if args.version:
        print(__version__)
        sys.exit(0)

    if args.loglevel:
        logger.setLevel(args.loglevel)
        logger.info(f"Setting loglevel to {args.loglevel}.")

    if args.clean:
        clean_models()
    elif args.napari:
        launch_napari(loglevel=args.loglevel or "INFO")
    elif args.config:
        launch_workflow_headless(args.config)
    elif args.train:
        launch_training(args.train)
    elif args.edit or args.edit is None:
        launch_editor(args.edit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
