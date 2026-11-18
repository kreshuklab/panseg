import collections
import logging
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from panseg.functionals.prediction.utils.slice_builder import SliceBuilder

logger = logging.getLogger(__name__)


def pad_to_shape(image: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Reflect-pad the high end of the spatial (last three) axes up to target_shape."""
    pads = [target - size for target, size in zip(target_shape, image.shape[-3:])]
    if not any(pads):
        return image
    if any(pad < 0 for pad in pads):
        raise ValueError(
            f"target_shape {target_shape} is smaller than {image.shape[-3:]}"
        )
    widths = [(0, 0)] * (image.ndim - 3) + [(0, pad) for pad in pads]
    return np.pad(image, widths, mode="reflect")


def mirror_pad(
    image: np.ndarray, padding_shape: tuple[int, int, int], multichannel: bool
) -> np.ndarray:
    """
    Pad the image with a mirror reflection of itself.

    Ideally, the padding shape should correspond to the spatial dimensions of the image, i.e.
    - If the image is 2D   (YX), the padding_shape should be a tuple of two integers.
    - If the image is 3D  (CYX), the padding_shape should be a tuple of two integers.
    - If the image is 3D  (ZYX), the padding_shape should be a tuple of three integers.
    - If the image is 4D (CZYX), the padding_shape should be a tuple of three integers.

    In other parts of this repo, the padding_shape is assumed to be 3D (ZYX/CYX).
    But one cannot know if the first dimension is channel for 3D images. Thus,
    `multichannel` is used to determine if the first dimension is channel or not.

    This function is used on data in its original/full shape before it is patchified.
    Halo should be the real surrounding of the patch, not the mirror padding of itself.

    Args:
        image (np.ndarray): The input image array to be padded.
        padding_shape (tuple of int): Specifies the amount of padding for each dimension, should be YX or ZYX.
        multichannel (bool, optional): Whether the image is multichannel or not.

    Returns:
        np.ndarray: The mirror-padded image.

    Raises:
        ValueError: If any element of padding_shape is negative.
    """
    if any(p < 0 for p in padding_shape):
        raise ValueError("padding_shape must be non-negative")

    if all(p == 0 for p in padding_shape):
        return image

    if (
        image.shape[0] == 1 and padding_shape[0] != 0
    ):  # handle 1YX cases where padding is ZYX
        raise ValueError(
            "Cannot pad a single-channel image (1YX) along the channel dimension"
        )

    pad_width = [(p, p) for p in padding_shape]

    if multichannel:
        if len(image.shape) == 3:  # CYX image
            if padding_shape[0] != 0:  # given 3D (CYX) padding shape, C has to be 0
                raise ValueError("Cannot pad a 2D multichannel image (CYX) along C")
        if len(image.shape) == 4:  # CZYX image
            pad_width = [
                (0, 0)
            ] + pad_width  # given 3D (ZYX) padding shape, has to add 0 C padding

    return np.pad(image, pad_width, mode="reflect")


def remove_padding(m, padding_shape):
    """
    Removes padding from the margins of a multi-dimensional array.

    Args:
        m (np.ndarray): The input array to be unpadded.
        padding_shape (tuple of int, optional): The amount of padding to remove from each dimension.
            Assumes the tuple length matches the array dimensions.

    Returns:
        np.ndarray: The unpadded array.
    """
    if padding_shape is None:
        return m

    # Correctly construct slice objects for each dimension in padding_shape and apply them to m.
    return m[(..., *(slice(p, -p or None) for p in padding_shape))]


class ArrayDataset(Dataset):
    """
    Based on pytorch-3dunet  AbstractHDF5Dataset
    https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/datasets/hdf5.py

    Inference only implementation of torch.utils.data.Dataset
    """

    def __init__(
        self,
        raw: np.ndarray,
        slice_builder: SliceBuilder,
        augs: Callable[[np.ndarray], torch.Tensor],
        halo_shape: Optional[tuple[int, int, int]] = None,
        multichannel: bool = False,
        verbose_logging: bool = True,
    ):
        """
        Args:
            raw (np.ndarray): raw data
            slice_builder (SliceBuilder): slice builder
            augs (Callable): data augmentation pipeline
            verbose_logging (bool): if True, log info messages
        """
        self.raw = raw
        self.augs = augs
        self.raw_slices = slice_builder.raw_slices

        if halo_shape is None:
            halo_shape = (0, 0, 0)
        self.halo_shape = halo_shape
        self.raw_padded = mirror_pad(self.raw, self.halo_shape, multichannel)

        if verbose_logging:
            logger.info(f"Number of patches: {len(self.raw_slices)}")

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]
        if len(raw_idx) == 4:
            halo_shape = (0,) + self.halo_shape
        else:
            halo_shape = self.halo_shape
        assert len(raw_idx) == len(halo_shape), (
            f"raw_idx {len(raw_idx)} and halo_shape {len(halo_shape)} must have the same length."
        )

        raw_idx_padded = tuple(
            slice(index.start, index.stop + 2 * halo, None)
            for index, halo in zip(raw_idx, halo_shape)
        )
        raw_patch = self.raw_padded[raw_idx_padded]
        raw_patch_transformed = self.augs(raw_patch)

        # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
        if len(raw_idx) == 4:
            raw_idx = raw_idx[1:]
        return raw_patch_transformed, raw_idx

    def __len__(self):
        return len(self.raw_slices)


def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))
