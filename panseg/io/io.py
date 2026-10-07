import logging
from pathlib import Path

import numpy as np

from panseg.io.h5 import (
    H5_EXTENSIONS,
    load_h5,
    read_h5_axis_order,
    read_h5_shape,
    read_h5_voxel_size,
)
from panseg.io.pil import PIL_EXTENSIONS, load_pil, read_pil_shape
from panseg.io.tiff import (
    TIFF_EXTENSIONS,
    load_tiff,
    read_ome_axes,
    read_tiff_shape,
    read_tiff_voxel_size,
)
from panseg.io.voxelsize import VoxelSize
from panseg.io.zarr import (
    ZARR_EXTENSIONS,
    load_zarr,
    read_zarr_axis_order,
    read_zarr_shape,
    read_zarr_voxel_size,
)

logger = logging.getLogger(__name__)

allowed_data_format = TIFF_EXTENSIONS + H5_EXTENSIONS + PIL_EXTENSIONS + ZARR_EXTENSIONS


def smart_load(path: Path, key: str | None = None, default=load_tiff) -> np.ndarray:
    """
    Load a dataset from a file. The loader is chosen based on the file extension.
    Supported formats are: tiff, h5, zarr, and PIL images.
    If the format is not supported, a default loader can be provided (default: load_tiff).

    Args:
        path (Path): path to the file to load.
        key (str): key of the dataset to load (if h5 or zarr).
        default (callable): default loader if the type is not understood.

    Returns:
        stack (np.ndarray): numpy array with the image data.

    Examples:
        >>> data = smart_load('path/to/file.tif')
        >>> data = smart_load('path/to/file.h5', key='raw')

    """
    ext = (path.suffix).lower()
    if key == "":
        key = None

    if ext in H5_EXTENSIONS:
        return load_h5(path, key)

    elif ext in TIFF_EXTENSIONS:
        return load_tiff(path)

    elif ext in PIL_EXTENSIONS:
        return load_pil(path)

    elif ext in ZARR_EXTENSIONS:
        return load_zarr(path, key)

    else:
        logger.warning(f"No default found for {ext}, reverting to default loader.")
        return default(path)


def smart_load_with_vs(path: Path, key: str | None = None, default=load_tiff) -> tuple:
    """
    Load a dataset from a file and returns some meta info about it. The loader is chosen based on the file extension.
    Supported formats are: tiff, h5, zarr, and PIL images.
    If the format is not supported, a default loader can be provided (default: load_tiff).

    Args:
        path (Path): path to the file to load.
        key (str): key of the dataset to load (if h5 or zarr).
        default (callable): default loader if the type is not understood.

    Returns:
        stack (np.ndarray): numpy array with the image data.

    Examples:
        >>> data = smart_load('path/to/file.tif')
        >>> data = smart_load('path/to/file.h5', key='raw')

    """
    ext = (path.suffix).lower()
    if key == "":
        key = None

    if ext in H5_EXTENSIONS:
        return load_h5(path, key), read_h5_voxel_size(path, key)

    if ext in TIFF_EXTENSIONS:
        return load_tiff(path), read_tiff_voxel_size(path)

    if ext in PIL_EXTENSIONS:
        return load_pil(path), VoxelSize()

    if ext in ZARR_EXTENSIONS:
        return load_zarr(path, key), read_zarr_voxel_size(path, key)

    else:
        logger.warning(
            f"No default found for {ext}, reverting to default loader with no voxel size reader."
        )
        return default(path), VoxelSize()


def shape_to_stack_layout(shape) -> str:
    """Guess the stack layout of an image based on its shape

    Might return empty string if shape could not be guessed, in particular
    for 4D shapes from which no channel axis can be distinguished.
    """
    if shape is None:
        return ""
    if len(shape) > 4:
        return ""

    d_to_put = ["Z", "Y", "X"]
    stack = []
    d_small = [i for i, d in enumerate(shape) if d < 10]
    channel = None
    if len(d_small) == 1:
        channel = d_small[0]

    n_spatial = len(shape) - (1 if channel is not None else 0)
    if n_spatial > len(d_to_put):
        return ""

    for i in range(len(shape))[::-1]:
        if i == channel:
            stack.insert(0, "C")
        else:
            stack.insert(0, d_to_put.pop())
    return "".join(stack)


def guess_stack_layout(path: Path, key: str | None = None) -> str:
    """Guess the stack layout of a file, for prefilling the stack layout input.

    OME-TIFF files use the reader's axis string (singleton axes already
    dropped by the reader), PanSeg-owned h5/zarr datasets their axis_order
    attribute, everything else the shape heuristic.

    Might return empty string if the layout could not be guessed.
    """
    ext = path.suffix.lower()

    if ext in TIFF_EXTENSIONS:
        axes = read_ome_axes(path)
        if axes is not None:
            return axes
        return shape_to_stack_layout(read_tiff_shape(path))

    if ext in H5_EXTENSIONS:
        axis_order = read_h5_axis_order(path, key)
        if axis_order is not None:
            return axis_order
        return shape_to_stack_layout(read_h5_shape(path, key))

    if ext in ZARR_EXTENSIONS:
        axis_order = read_zarr_axis_order(path, key)
        if axis_order is not None:
            return axis_order
        return shape_to_stack_layout(read_zarr_shape(path, key))

    if ext in PIL_EXTENSIONS:
        return shape_to_stack_layout(read_pil_shape(path))

    return ""
