"""Shared helpers for the training test modules."""

from pathlib import Path

import h5py
import numpy as np


def write_h5(file_path: Path, raw, label, raw_dtype=None, label_dtype=None) -> Path:
    """Write a training H5 file: a `raw` and a `label` dataset.

    Args:
        file_path: destination; parent directories are created.
        raw: the raw volume (ZYX or CZYX).
        label: the channel-less instance label volume (ZYX).
        raw_dtype (optional): dtype to cast `raw` to on the way in.
        label_dtype (optional): dtype to cast `label` to on the way in.

    Returns:
        The written path.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_dtype is not None:
        raw = np.asarray(raw, dtype=raw_dtype)
    if label_dtype is not None:
        label = np.asarray(label, dtype=label_dtype)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("raw", data=raw)
        f.create_dataset("label", data=label)
    return file_path


def model_dir_patches(root: Path) -> list[tuple[object, str, object]]:
    """``(target, attribute, value)`` triples redirecting model-zoo writes to `root`.

    Every module that holds its own reference to a model path must be
    redirected: `unet_training` writes the checkpoints to the training module's
    `PATH_PANSEG_MODELS`, while `model_zoo.add_custom_model` copies the packaged
    files into its own `PATH_PANSEG_MODELS` and appends the record to the custom
    zoo file. Missing one of them writes into the user's `~/.panseg_models`.
    """
    import panseg.core.zoo as zoo_module
    from panseg import FILE_MODEL_ZOO_CUSTOM
    from panseg.core.zoo import model_zoo
    from panseg.functionals.training import train as train_module

    custom_zoo = root / FILE_MODEL_ZOO_CUSTOM
    return [
        (train_module, "PATH_PANSEG_MODELS", root),
        (zoo_module, "PATH_PANSEG_MODELS", root),
        (zoo_module, "PATH_MODEL_ZOO_CUSTOM", custom_zoo),
        (model_zoo, "path_zoo", root),
        (model_zoo, "path_zoo_custom", custom_zoo),
    ]
