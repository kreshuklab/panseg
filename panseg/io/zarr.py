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

    Args:
        path (Path): The path to the Zarr file.
        key (str | None): The internal key of the desired dataset.
        slices (slice | None, optional): Slice to load. Defaults to None.

    Returns:
        np.ndarray: The dataset as a NumPy array.
    """
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
    _validate_zarr_file(path)
    data = _get_zarr_dataset(path, key)
    return data.shape


def read_zarr_voxel_size(path: Path, key: str | None) -> VoxelSize:
    """Read the voxel size of a dataset in a Zarr file.

    Args:
        path (Path): The path to the Zarr file.
        key (str | None): The internal key of the desired dataset.

    Returns:
        VoxelSize: The voxel size of the dataset.
    """
    _validate_zarr_file(path)
    data = _get_zarr_dataset(path, key)

    if "element_size_um" in data.attrs:
        return VoxelSize(voxels_size=data.attrs["element_size_um"])
    elif "resolution" in data.attrs:
        return VoxelSize(voxels_size=data.attrs["resolution"])

    warnings.warn(f"Voxel size not found in {path}.")
    return VoxelSize()


def create_zarr(
    path: Path,
    stack: np.ndarray,
    key: str,
    voxel_size: VoxelSize,
    mode: str = "a",
) -> None:
    """
    Create a Zarr array from a NumPy array.

    Args:
        path (Path): The path to the Zarr file.
        stack (np.ndarray): The NumPy array to save as a dataset.
        key (str): The internal key of the desired dataset.
        voxel_size (VoxelSize): The voxel size of the dataset.
        mode (str): The mode to open the Zarr file ['w', 'a'].
    """
    if not key:
        raise ValueError("Key cannot be None or empty.")

    zarr_file = zarr.open_group(store=str(path), mode=mode)

    if IS_ZARR_V3:
        # v3 requires explicit `create_array`, and write manually
        assert isinstance(zarr_file, zarr.Group), f"Invalid Zarr file: {path}"
        zarr_file.create_array(name=key, shape=stack.shape, dtype=stack.dtype)
        zarr_file[key][:] = stack
    else:
        # v2 allows `create_dataset` with `overwrite`
        zarr_file.create_dataset(key, data=stack, compression="gzip", overwrite=True)

    zarr_file[key].attrs["element_size_um"] = voxel_size.voxels_size


def list_zarr_keys(path: Path) -> list[str]:
    """
    List all keys in a Zarr file.

    Args:
        path (Path): The path to the Zarr file.

    Returns:
        list[str]: A list of keys in the Zarr file.
    """

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
