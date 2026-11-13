"""
Reading and writing Zarr files. This is created in the same format as h5.py.

Notes:
1. Although the function is called "open", there is no need to close an array: data is automatically flushed to disk, and files are automatically closed whenever an array is modified. [Ref](https://zarr.readthedocs.io/en/stable/tutorial.html#persistent-arrays).
2. An open file either gives a `zarr.Array` or `zarr.Group` object. In PanSeg, `zarr.open` is expected to return a `zarr.Group` object.
3. Zarr works with string paths, not `pathlib.Path` objects. No need to upgrade to `pathlib`.
"""

import warnings
from pathlib import Path

import numpy as np
import zarr
from packaging.version import parse as parse_version

from panseg.io.ome_zarr import (
    create_ome_zarr,
    is_ome_zarr,
    list_ome_zarr_keys,
    load_ome_zarr,
    read_ome_zarr_axis_order,
    read_ome_zarr_shape,
    read_ome_zarr_time_spacing,
    read_ome_zarr_voxel_size,
)
from panseg.io.voxelsize import VoxelSize

ZARR_VERSION = parse_version(zarr.__version__)
IS_ZARR_V3 = ZARR_VERSION >= parse_version("3.0.0")

ZARR_EXTENSIONS = [".zarr"]
ZARR_KEYS = ["raw", "prediction", "segmentation"]


def _validate_zarr_file(path: Path) -> None:
    """Check if a file is a Zarr file."""
    assert path.exists(), f"File not found: {path}"
    assert str(path).find(".zarr") != -1, f"File is not a Zarr file: {path}"


def _find_input_key(zarr_file: zarr.Group) -> str:
    """Return the first matching key in `ZARR_KEYS` or the key to the only dataset if just one is found.

    Args:
        zarr_file (zarr.Group): The Zarr file (or group) containing datasets.

    Returns:
        str: The key of the matching dataset.

    Raises:
        RuntimeError: If no datasets are found or the dataset is ambiguous.
    """
    found_datasets = []

    def collect_datasets(group, prefix: str = "") -> None:
        for key, node in group.members():
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(node, zarr.Array):
                found_datasets.append(name)
            else:
                collect_datasets(node, name)

    collect_datasets(zarr_file)

    if not found_datasets:
        raise RuntimeError(f"No datasets found - verify '{zarr_file.tree()}'.")

    if len(found_datasets) == 1:
        return found_datasets[0]
    else:
        for zarr_key in ZARR_KEYS:
            if zarr_key in found_datasets:
                return zarr_key

        raise RuntimeError(
            f"Ambiguous datasets '{found_datasets}' in {zarr_file}. "
            f"PanSeg expects only one dataset in the input Zarr."
        )


def _get_zarr_dataset(zarr_path: Path, key: str | None = None) -> zarr.Array:
    zarr_file = zarr.open_group(zarr_path, mode="r")
    if key is None:
        key = _find_input_key(zarr_file)

    ds = zarr_file[key]
    if not isinstance(ds, zarr.Array):
        raise ValueError(f"'{key}' is not a zarr.Array.")

    return zarr.open_array(zarr_path, mode="r", path=key)


def load_zarr(
    path: Path,
    key: str | None,
    slices: slice | None = None,
) -> np.ndarray:
    """Load a dataset from a Zarr file and return it or its meta-information.

    OME-Zarr stores resolve their base level through the NGFF
    metadata. PanSeg-native stores use the pre-NGFF flat-dataset readers.

    Args:
        path (Path): The path to the Zarr file.
        key (str | None): The internal key of the desired dataset. For an
            OME-Zarr store, the path of the image group.
        slices (slice | None, optional): Slice to load. Defaults to None.

    Returns:
        np.ndarray: The dataset as a NumPy array.
    """
    if is_ome_zarr(path):
        return load_ome_zarr(path, key, slices)

    _validate_zarr_file(path)
    data = _get_zarr_dataset(path, key)
    data = data[...] if slices is None else data[slices]
    return data


def read_zarr_shape(path: Path, key: str | None = None) -> tuple[int, ...]:
    """Read the shape of a dataset in a Zarr file.

    Args:
        path (Path): The path to the Zarr file.
        key (str | None, optional): The internal key of the desired dataset. Defaults to None.

    Returns:
        tuple[int, ...]: The shape of the dataset.
    """
    if is_ome_zarr(path):
        return read_ome_zarr_shape(path, key)

    _validate_zarr_file(path)
    data = _get_zarr_dataset(path, key)
    return data.shape


def read_zarr_axis_order(path: Path, key: str | None = None) -> str | None:
    """Read the axis_order attribute of a dataset written by PanSeg, e.g. "TZYX".

    Args:
        path (Path): The path to the Zarr file.
        key (str | None, optional): The internal key of the desired dataset. Defaults to None.

    Returns:
        str | None: The axis order string, or None when the attribute is
        absent (older files).
    """
    if is_ome_zarr(path):
        return read_ome_zarr_axis_order(path, key)

    _validate_zarr_file(path)
    data = _get_zarr_dataset(path, key)
    axis_order = data.attrs.get("axis_order", None)
    return str(axis_order) if axis_order is not None else None


def read_zarr_voxel_size(path: Path, key: str | None) -> VoxelSize:
    """Read the voxel size of a dataset in a Zarr file.

    Args:
        path (Path): The path to the Zarr file.
        key (str | None): The internal key of the desired dataset.

    Returns:
        VoxelSize: The voxel size of the dataset.
    """
    if is_ome_zarr(path):
        return read_ome_zarr_voxel_size(path, key)

    _validate_zarr_file(path)
    data = _get_zarr_dataset(path, key)

    if "element_size_um" in data.attrs:
        return VoxelSize(voxels_size=data.attrs["element_size_um"])
    elif "resolution" in data.attrs:
        return VoxelSize(voxels_size=data.attrs["resolution"])

    warnings.warn(f"Voxel size not found in {path}.")
    return VoxelSize()


def read_zarr_time_spacing(
    path: Path, key: str | None = None
) -> tuple[float | None, str]:
    """Read the time spacing attrs of a dataset written by PanSeg.

    The attrs are written when the exported axis order carries a time axis
    and the spacing is known. Returns (None, "s") when the attrs are absent
    (older files, unknown spacing, or no time axis).

    Args:
        path (Path): The path to the Zarr file.
        key (str | None, optional): The internal key of the desired dataset. Defaults to None.

    Returns:
        tuple: (time spacing value or None, unit string)
    """
    if is_ome_zarr(path):
        return read_ome_zarr_time_spacing(path, key)

    _validate_zarr_file(path)
    data = _get_zarr_dataset(path, key)
    t_spacing = data.attrs.get("t_spacing", None)
    if t_spacing is None:
        return None, "s"
    return float(t_spacing), str(data.attrs.get("t_spacing_unit", "s"))


def create_zarr(
    path: Path,
    stack: np.ndarray,
    key: str | None,
    voxel_size: VoxelSize,
    mode: str = "a",
    axis_order: str | None = None,
    t_spacing: float | None = None,
    t_spacing_unit: str = "s",
    label_name: str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Create a Zarr array from a NumPy array.

    Zarr export writes an OME-NGFF image: the store
    carries an image group with clearscale-built multiscale metadata instead
    of PanSeg dataset attributes. The image group is written to the path
    given by ``key`` (required, e.g. "raw"); with ``label_name`` the stack is
    written as a label image below the parent image group's labels container
    (``<key>/labels/<label_name>``), which is how segmentations are exported.

    Args:
        path (Path): The path to the Zarr file.
        stack (np.ndarray): The NumPy array to save as a dataset.
        key (str | None): The path of the image group; for a label image,
            the path of its parent image group. Required.
        voxel_size (VoxelSize): The voxel size of the dataset.
        mode (str): The mode to open the Zarr file ['w', 'a'].
        axis_order (str | None): Layout string of the stack (e.g. "TZYX"),
            defining the NGFF axes of the image.
        t_spacing (float | None): Time spacing between timepoints in
            ``t_spacing_unit``, written as the NGFF time-axis scale and unit. None
            (unknown) writes no time unit.
        t_spacing_unit (str): Unit of the time spacing.
        label_name (str | None): Name of the label image below
            ``<key>/labels/`` (e.g. "segmentation"); None writes an image
            group at ``key``.
        overwrite (bool): Replace an image or label already present at the
            target slot; without it, writing into an occupied slot raises a
            ValueError.
    """
    create_ome_zarr(
        path=path,
        stack=stack,
        key=key if key else None,
        voxel_size=voxel_size,
        mode=mode,
        axis_order=axis_order,
        t_spacing=t_spacing,
        t_spacing_unit=t_spacing_unit,
        label_name=label_name,
        overwrite=overwrite,
    )


def list_zarr_keys(path: Path) -> list[str]:
    """
    List all keys in a Zarr file.

    For an OME-Zarr store, the keys are the paths of the image groups; a
    store whose root is an image group returns an empty list (the image is
    addressed without a key).

    Args:
        path (Path): The path to the Zarr file.

    Returns:
        list[str]: A list of keys in the Zarr file.
    """
    if is_ome_zarr(path):
        return list_ome_zarr_keys(path)

    def _recursive_find_keys(
        zarr_group: zarr.Group, base: Path = Path("")
    ) -> list[str]:
        _list_keys = []
        for key, dataset in zarr_group.members():
            if isinstance(dataset, zarr.Group):
                new_base = base / key
                _list_keys.extend(_recursive_find_keys(dataset, new_base))
            elif isinstance(dataset, zarr.Array):
                _list_keys.append(str(base / key))
        return _list_keys

    zarr_file = zarr.open_group(store=path, mode="r")
    return _recursive_find_keys(zarr_file)


def del_zarr_key(path: Path, key: str, mode: str = "a") -> None:
    """
    Delete a dataset from a Zarr file.

    Args:
        path (Path): The path to the Zarr file.
        key (str): The internal key of the dataset to be deleted.
        mode (str): The mode to open the Zarr file ['w', 'a'].

    """
    zarr_file = zarr.open_group(store=path, mode=mode)
    if key in zarr_file:
        del zarr_file[key]


def rename_zarr_key(path: Path, old_key: str, new_key: str, mode: str = "r+") -> None:
    """
    Rename a dataset in a Zarr file.

    Args:
        path (Path): The path to the Zarr file.
        old_key (str): The current key of the dataset.
        new_key (str): The new key for the dataset.
        mode (str): The mode to open the Zarr file ['r+'].

    """
    zarr_file = zarr.open_group(store=path, mode=mode)
    if old_key in zarr_file:
        old_array = zarr_file[old_key]
        new_array = zarr_file.create_array(
            name=new_key, shape=old_array.shape, dtype=old_array.dtype
        )
        new_array[:] = old_array[:]
        new_array.attrs.update(dict(old_array.attrs))
        del zarr_file[old_key]
