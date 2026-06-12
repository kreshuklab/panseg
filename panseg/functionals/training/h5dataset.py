import logging
from typing import Literal

import h5py
import numpy as np
from torch.utils.data import Dataset

from panseg.functionals.prediction.utils.slice_builder import FilterSliceBuilder

logger = logging.getLogger(__name__)


class HDF5Dataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.

    Unifies dataset shapes by adding the z dimension to datasets. Inputs also get the channel dimension added.
    The training input has therefor shape czyx, the label zyx.

    Args:
        file_path (str): path to H5 file containing raw data as well as labels and per pixel weights (optional)
        augmenter (transforms.Augmenter): list of augmentations to be applied to the raw and label data sets
        patch_shape (tuple): shape of the patch to be extracted from the raw data set
        raw_internal_path (str or list): H5 internal path to the raw dataset
        label_internal_path (str or list): H5 internal path to the label dataset
        global_normalization (bool): if True, the mean and std of the raw data will be calculated over the whole dataset
    """

    def __init__(
        self,
        file_path,
        augmenter,
        patch_shape,
        dimensionality: Literal["2D", "3D"],
        raw_internal_path="raw",
        label_internal_path="label",
        global_normalization=True,
    ):
        self.file_path = file_path
        self.dimensionality = dimensionality

        with h5py.File(file_path, "r") as f:
            self.raw = self.load_dataset(f, raw_internal_path, ensure_channel=True)
            stats = calculate_stats(self.raw, global_normalization)
            self.augmenter = augmenter
            self.raw_transform = self.augmenter.raw_transform(stats)

            # create label/weight transform only in train/val phase
            self.label_transform = self.augmenter.label_transform()
            self.label = self.load_dataset(f, label_internal_path, ensure_channel=False)
            self._check_volume_sizes()

            # build slice indices for raw and label data sets
            slice_builder = FilterSliceBuilder(
                self.raw,
                self.label,
                patch_shape=patch_shape,
            )
            self.raw_slices = slice_builder.raw_slices
            self.label_slices = slice_builder.label_slices

            self.patch_count = len(self.raw_slices)
            logger.info(f"{self.patch_count} patches found in {file_path}")

    def load_dataset(self, input_file, internal_path, ensure_channel: bool):
        ds = input_file[internal_path][:]
        # Add z dimension for 2d arrays, the augmenter only supports 3d/4d
        if self.dimensionality == "2D":
            if ds.ndim not in [2, 3]:
                raise ValueError(
                    f"Dimensionality = 2D, but tried loading {ds.ndim}D dataset"
                )
            if ensure_channel and ds.ndim == 2:  # add channel and z
                return np.expand_dims(ds, [0, 1])  # -> czyx
            elif ensure_channel and ds.ndim == 3:  # add z only
                return np.expand_dims(ds, 1)  # -> czyx
            elif ds.ndim == 2:  # add z only
                return np.expand_dims(ds, 0)  # ->  zyx
            elif ds.ndim == 3:  # add nothing
                return ds  # -> cyx

        elif self.dimensionality == "3D":
            if ds.ndim not in [3, 4]:
                raise ValueError(
                    f"Dimensionality = 3D, but tried loading {ds.ndim}D dataset"
                )
            if ensure_channel and ds.ndim == 3:
                return np.expand_dims(ds, 0)
            return ds
        raise ValueError(f"Unknown dimensionality {self.dimensionality}")

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self.raw_transform(self.raw[raw_idx])

        # get the slice for a given index 'idx'
        label_idx = self.label_slices[idx]
        label_patch_transformed = self.label_transform(self.label[label_idx])
        # return the transformed raw and label patches
        return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count

    @staticmethod
    def create_h5_file(file_path):
        raise NotImplementedError

    def _check_volume_sizes(self):
        def _volume_shape(volume, dim):
            if dim == "3D":
                if volume.ndim == 3:  # ZYX
                    return volume.shape
                elif volume.ndim == 4:  # CYZX
                    return volume.shape[1:]
            elif dim == "2D":
                if volume.ndim == 3:  # ZYX
                    return volume.shape[1:]
                elif volume.ndim == 4:  # CZYX
                    return volume.shape[2:]
            raise ValueError(
                f"Volume of shape {volume.shape} does not fit to reported dimensionality {dim}"
            )

        assert self.raw.ndim in [3, 4], "Raw dataset must be YX, CYX, ZYX, CZYX"
        assert self.label.ndim in [3, 4], "Label dataset must be 2D (YX) or 3D (ZYX)"

        assert _volume_shape(self.raw, self.dimensionality) == _volume_shape(
            self.label, self.dimensionality
        ), "Raw and label image data has to be of the same size"


def calculate_stats(images, global_normalization=True):
    """
    Calculates min, max, mean, std given a list of nd-arrays
    """
    if global_normalization:
        # flatten first since the images might not be the same size
        flat = np.concatenate([img.ravel() for img in images])
        pmin, pmax, mean, std = (
            np.percentile(flat, 1),
            np.percentile(flat, 99.6),
            np.mean(flat),
            np.std(flat),
        )
    else:
        pmin, pmax, mean, std = None, None, None, None

    return {"pmin": pmin, "pmax": pmax, "mean": mean, "std": std}
