import warnings
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from panseg.io.voxelsize import VoxelSize

# allowed h5 keys
H5_EXTENSIONS = [".hdf", ".h5", ".hd5", "hdf5"]
H5_KEYS = ["raw", "prediction", "segmentation", "labels"]


def _validate_h5_file(path: Path) -> None:
    assert path.suffix.lower() in H5_EXTENSIONS, (
        f"File extension not supported. Supported extensions: {H5_EXTENSIONS}"
    )
    assert path.exists(), f"File not found: {path}"


def _find_input_key(h5_file) -> str:
    """
    returns the first matching key in H5_KEYS or if only one dataset is found
    the key to that dataset
    """
    found_datasets = []

    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            found_datasets.append(name)

    h5_file.visititems(visitor_func)

    if not found_datasets:
        raise RuntimeError(f"No datasets found in '{h5_file.filename}'")

    if len(found_datasets) == 1:
        return found_datasets[0]
    else:
        for h5_key in H5_KEYS:
            if h5_key in found_datasets:
                return h5_key

        raise RuntimeError(
            f"Ambiguous datasets '{found_datasets}' in {h5_file.filename}. "
            f"panseg expects only one dataset to be present in input H5."
        )


def _get_h5_dataset(f: h5py.File, key: Optional[str] = None) -> h5py.Dataset:
    if key is None:
        key = _find_input_key(f)

    ds = f[key]
    if not isinstance(ds, h5py.Dataset):
        raise ValueError(f"'{key}' is not a h5py.Dataset.")

    return ds


def load_h5(
    path: Path,
    key: Optional[str] = None,
    slices: Optional[slice] = None,
) -> np.ndarray:
    """
    Load a dataset from a h5 file and returns some meta info about it.

    Args:
        path (Path): Path to the h5file
        key (Optional[str], optional): Optional, key of the dataset in the h5 file. Defaults to None.
        slices (Optional[slice], optional): Optional, slice to load. Defaults to None.

    Returns:
        np.ndarray: dataset as numpy array
    """

    assert path.suffix.lower() in H5_EXTENSIONS, (
        f"File extension not supported. Supported extensions: {H5_EXTENSIONS}"
    )
    assert path.exists(), f"File not found: {path}"

    with h5py.File(path, "r") as f:
        data = _get_h5_dataset(f, key)
        data = data[...] if slices is None else data[slices]

    return data


def read_h5_shape(path: Path, key: Optional[str] = None) -> tuple[int, ...]:
    """
    Load a dataset from a h5 file and returns some meta info about it.

    Args:
        path (Path): Path to the h5file
        key (Optional[str], optional): Optional, key of the dataset in the h5 file. Defaults to None.

    Returns:
        tuple[int]: shape of the dataset
    """
    with h5py.File(path, "r") as f:
        data = _get_h5_dataset(f, key)
        shape = data.shape

    return shape


def read_h5_voxel_size(
    path: Path,
    key: Optional[str] = None,
) -> VoxelSize:
    """
    Load the voxel size from a h5 file.

    Args:
        path (Path): path to the h5 file
        key (Optional[str], optional): key of the dataset in the h5 file. Defaults to None.

    Returns:
        VoxelSize: voxel size of the dataset
    """
    with h5py.File(path, "r") as f:
        data = _get_h5_dataset(f, key)
        voxel_size = data.attrs.get("element_size_um", None)

        if voxel_size is None:
            warnings.warn(f"Voxel size not found in {path}.")
            return VoxelSize()

        voxel_size = VoxelSize(voxels_size=voxel_size.tolist())

    return voxel_size


def create_h5(
    path: Path,
    stack: np.ndarray,
    key: str,
    voxel_size: Optional[VoxelSize] = None,
    mode: str = "a",
) -> None:
    """
    Create a dataset inside a h5 file from a numpy array.

    Args:
        path (Path): path to the h5 file
        stack (np.ndarray): numpy array to save as dataset in the h5 file.
        key (str): key of the dataset in the h5 file.
        voxel_size (VoxelSize): voxel size of the dataset.
        mode (str): mode to open the h5 file ['w', 'a'].

    """

    if key is None:
        raise ValueError("Key is required to create a dataset in a h5 file.")

    if key == "":
        raise ValueError("Key cannot be empty to create a dataset in a h5 file.")

    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]
        f.create_dataset(key, data=stack, compression="gzip")
        # save voxel_size
        if voxel_size is not None and voxel_size.voxels_size is not None:
            f[key].attrs["element_size_um"] = voxel_size.voxels_size


def list_h5_keys(path: Path) -> list[str]:
    """
    List all keys in a h5 file

    Args:
        path (Path): path to the h5 file (Path object)

    Returns:
        keys (list[str]): A list of keys in the h5 file.

    """
    _validate_h5_file(path)

    def _recursive_find_keys(f, base="/"):
        _list_keys = []
        for key, dataset in f.items():
            if isinstance(dataset, h5py.Group):
                new_base = f"{base}{key}/"
                _list_keys += _recursive_find_keys(dataset, new_base)

            elif isinstance(dataset, h5py.Dataset):
                _list_keys.append(f"{base}{key}")
        return _list_keys

    with h5py.File(path, "r") as h5_f:
        return _recursive_find_keys(h5_f)


def del_h5_key(path: Path, key: str, mode: str = "a") -> None:
    """
    helper function to delete a dataset from a h5file

    Args:
        path (Path): path to the h5 file (Path object)
        key (str): key of the dataset to delete
        mode (str): mode to open the h5 file ['r', 'r+']

    """
    _validate_h5_file(path)
    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]
            f.close()


def rename_h5_key(path: Path, old_key: str, new_key: str, mode="r+") -> None:
    """
    Rename the 'old_key' dataset to 'new_key'

    Args:
        path (Path): path to the h5 file (Path object)
        old_key (str): old key name
        new_key (str): new key name
        mode (str): mode to open the h5 file ['r', 'r+']

    """
    _validate_h5_file(path)
    with h5py.File(path, mode) as f:
        if old_key in f:
            f[new_key] = f[old_key]
            del f[old_key]
            f.close()
