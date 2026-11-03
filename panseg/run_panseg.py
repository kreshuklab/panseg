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


def _translate_loaders(config: dict, path: Path) -> None:
    """Fold the YAML `loaders` section into `unet_training` arguments.

    `loaders` is the section `create_model_config` writes into every
    checkpoint's ``config_train.yml``: the written record of the effective
    loader setup. It is not an argument of `unet_training` (the loaders are
    derived from ``patch_size``/``device``), so a config that carries it — e.g. a
    model's own ``config_train.yml`` used as a starting point — must not be
    rejected as having unknown keys.

    Exactly one knob in that section has a live counterpart: the validation
    patch grid's ``stride_shape``, which maps to the ``val_slice_builder``
    argument. The rest of the section is a record; the keys of the val slice
    builder that would silently do nothing (a filter ``name`` or a
    ``threshold``: val patches are never filtered) are rejected instead, as is a
    val ``patch_shape`` that differs from ``patch_size`` (the val loader is
    always built from the training patch size).
    """
    loaders = config.pop("loaders", None)
    if loaders is None:
        return
    if not isinstance(loaders, dict):
        raise ValueError(f"`loaders` must be a mapping in training config {path}.")

    val = loaders.get("val")
    if val is None:
        return
    if not isinstance(val, dict):
        raise ValueError(f"`loaders.val` must be a mapping in training config {path}.")
    slice_builder = val.get("slice_builder")
    if slice_builder is None:
        return
    if not isinstance(slice_builder, dict):
        raise ValueError(
            f"`loaders.val.slice_builder` must be a mapping in training config {path}."
        )

    unknown = sorted(set(slice_builder) - {"name", "patch_shape", "stride_shape"})
    if unknown:
        raise ValueError(
            f"Unsupported keys in `loaders.val.slice_builder` of {path}: {unknown}. "
            "Validation patches are built unfiltered on the training patch shape, so "
            "only `stride_shape` changes the grid (a `threshold` would be a no-op)."
        )
    name = slice_builder.get("name", "SliceBuilder")
    if name != "SliceBuilder":
        raise ValueError(
            f"`loaders.val.slice_builder.name` must be 'SliceBuilder' in {path}, "
            f"got {name!r}: validation patches are never threshold-filtered."
        )
    patch_size = config.get("patch_size")
    configured_patch_shape = slice_builder.get("patch_shape")
    if patch_size is not None and configured_patch_shape is not None:
        if list(configured_patch_shape) != list(patch_size):
            raise ValueError(
                f"`loaders.val.slice_builder.patch_shape` ({configured_patch_shape}) "
                f"must match `patch_size` ({patch_size}) in {path}: the validation "
                "loader is built from the training patch size."
            )

    stride_shape = slice_builder.get("stride_shape")
    if stride_shape is None:
        return
    if "val_slice_builder" in config:
        raise ValueError(
            f"Configure the validation patch stride once in {path}: either "
            "`val_slice_builder` or `loaders.val.slice_builder.stride_shape`."
        )
    config["val_slice_builder"] = {"stride_shape": list(stride_shape)}


def launch_training(path: Path):
    """Launch the training"""
    config = load_config(path)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid training config: {path}")

    # The example config nests the parameters under a `training:` key,
    # but a flat top-level config is accepted as well.
    config = dict(config.get("training", config))
    _translate_loaders(config, path)

    import inspect

    from panseg.functionals.training.train import unet_training

    valid_keys = set(inspect.signature(unet_training).parameters)
    unknown_keys = sorted(set(config) - valid_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in training config {path}: {unknown_keys}. "
            f"Valid keys are: {sorted(valid_keys)}"
        )

    unet_training(**config)


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
