import logging
from typing import Literal

import h5py
import numpy as np
from torch.utils.data import Dataset

from panseg.functionals.prediction.utils.slice_builder import (
    FilterSliceBuilder,
    SliceBuilder,
)

logger = logging.getLogger(__name__)

_DIMENSIONALITY_2D = ("2D", "2d", "2")
_DIMENSIONALITY_3D = ("3D", "3d", "3")

_SUPPORTED_AXIS_ORDERS = ("YX", "CYX", "ZYX", "CZYX")

_LAYOUT_PAIRINGS_2D = {
    (2, 2): ("YX", "YX"),
    (3, 2): ("CYX", "YX"),
    (3, 3): ("ZYX", "ZYX"),
    (4, 3): ("CZYX", "ZYX"),
}
_LAYOUT_PAIRINGS_3D = {
    (3, 3): ("ZYX", "ZYX"),
    (4, 3): ("CZYX", "ZYX"),
}


def _parse_axis_order(axis_order, ndim: int, dataset_name: str) -> str:
    """Validate a dataset's ``axis_order`` attribute against its ndim.

    Args:
        axis_order: the ``axis_order`` dataset attribute (e.g. "CZYX")
        ndim (int): ndim of the dataset the attribute was read from
        dataset_name (str): name of the dataset for error messages

    Returns:
        str: the validated canonical layout string ("YX", "CYX", "ZYX" or "CZYX")

    Raises:
        ValueError: if the attribute declares a time axis, does not match the
            dataset's ndim, or is not one of the supported layouts.
    """
    if isinstance(axis_order, bytes):
        axis_order = axis_order.decode()
    if not isinstance(axis_order, str):
        raise ValueError(
            f"The axis_order attribute of the {dataset_name} dataset must be a "
            f"string, got {type(axis_order).__name__}."
        )
    axis_order = axis_order.upper()
    if "T" in axis_order:
        raise ValueError(
            f"Time axis not supported for training, but the {dataset_name} dataset "
            f"declares axis_order '{axis_order}'."
        )
    if len(axis_order) != ndim:
        raise ValueError(
            f"axis_order '{axis_order}' of the {dataset_name} dataset is "
            f"inconsistent with its {ndim}D shape."
        )
    if axis_order not in _SUPPORTED_AXIS_ORDERS:
        raise ValueError(
            f"Unsupported axis_order '{axis_order}' on the {dataset_name} dataset; "
            f"expected one of {', '.join(_SUPPORTED_AXIS_ORDERS)}."
        )
    return axis_order


def resolve_dataset_layouts(
    dimensionality: str,
    raw_ndim: int,
    label_ndim: int,
    raw_axis_order: str | None = None,
    label_axis_order: str | None = None,
) -> tuple[str, str]:
    """Resolve the on-disk layouts of the raw and label datasets.

    Both datasets are resolved together because the ``(raw_ndim, label_ndim)``
    pairing disambiguates e.g. a CYX raw from a ZYX raw (volumetric
    data trained per z-slice): both are 3D on disk.

    The ``axis_order`` dataset attribute (written by PanSeg on export, e.g.
    "ZYX" or "CZYX") wins when present. Files without the attribute are resolved
    by the pairing rule:

    - 2D training: ``(2, 2)`` -> YX, ``(3, 2)`` -> CYX (legacy 2D data, a
      singleton z is fabricated for the augmenter); ``(3, 3)`` -> ZYX and
      ``(4, 3)`` -> CZYX (volumetric data, trained per z-slice).
    - 3D training: ``(3, 3)`` -> ZYX, ``(4, 3)`` -> CZYX.

    Args:
        dimensionality (str): "2D" or "3D" model architecture
        raw_ndim (int): ndim of the raw dataset
        label_ndim (int): ndim of the label dataset
        raw_axis_order (str | None): optional ``axis_order`` attr of the raw dataset
        label_axis_order (str | None): optional ``axis_order`` attr of the label dataset

    Returns:
        tuple[str, str]: canonical layouts ``(raw_layout, label_layout)``, each
        one of "YX", "CYX", "ZYX" or "CZYX"

    Raises:
        ValueError: if the dimensionality is unknown, an ``axis_order`` attribute
            is invalid, or the (raw, label) dimensionalities do not fit together.
    """
    if dimensionality in _DIMENSIONALITY_2D:
        dim, pairings = "2D", _LAYOUT_PAIRINGS_2D
    elif dimensionality in _DIMENSIONALITY_3D:
        dim, pairings = "3D", _LAYOUT_PAIRINGS_3D
    else:
        raise ValueError(f"Unknown dimensionality {dimensionality}")

    raw_layout = (
        _parse_axis_order(raw_axis_order, raw_ndim, "raw")
        if raw_axis_order is not None
        else None
    )
    label_layout = (
        _parse_axis_order(label_axis_order, label_ndim, "label")
        if label_axis_order is not None
        else None
    )

    if raw_layout is not None and label_layout is not None:
        # The attributes win over the ndim-based fallback
        resolved = (raw_layout, label_layout)
    else:
        # Fallback pairing rule: the (raw ndim, label ndim) pairing disambiguates,
        # known attributes constrain the candidates
        candidates = [
            layouts
            for (rn, ln), layouts in pairings.items()
            if rn == raw_ndim
            and ln == label_ndim
            and (raw_layout is None or layouts[0] == raw_layout)
            and (label_layout is None or layouts[1] == label_layout)
        ]
        if len(candidates) != 1:
            accepted = ", ".join(f"({rn}, {ln})" for rn, ln in pairings)
            raise ValueError(
                f"Cannot resolve the data layout for {dim} training: raw is "
                f"{raw_ndim}D (axis_order={raw_axis_order!r}), label is "
                f"{label_ndim}D (axis_order={label_axis_order!r}). Accepted "
                f"(raw ndim, label ndim) combinations for {dim} training: {accepted}."
            )
        resolved = candidates[0]

    if resolved[1] not in ("YX", "ZYX"):
        raise ValueError(
            f"The label dataset must be channel-less (axis_order 'YX' or 'ZYX'), "
            f"but got '{resolved[1]}'."
        )
    if dim == "3D" and (resolved[0] not in ("ZYX", "CZYX") or resolved[1] != "ZYX"):
        # Guards the attributes-win path: 2D-layout data with valid axis_order
        # attrs must not silently get a fabricated z axis for 3D training
        raise ValueError(
            f"3D training requires volumetric data (raw zyx/czyx, label zyx), but "
            f"resolved raw '{resolved[0]}' and label '{resolved[1]}'. "
            "For 2D data (yx/cyx) use dimensionality '2D'."
        )
    return resolved


def _apply_layout(data: np.ndarray, layout: str, ensure_channel: bool) -> np.ndarray:
    """Bring ``data`` into the training tensor contract.

    Raw is loaded as czyx and label as zyx; missing channel and z axes are added
    as singleton axes. The augmenter is 3D/4D-written, so even pure 2D data gets
    a (fake) z axis.
    """
    if ensure_channel:
        if layout == "YX":
            return np.expand_dims(data, (0, 1))  # -> czyx
        elif layout == "CYX":
            return np.expand_dims(data, 1)  # -> czyx
        elif layout == "ZYX":
            return np.expand_dims(data, 0)  # -> czyx
        return data  # CZYX -> czyx
    if layout == "YX":
        return np.expand_dims(data, 0)  # -> zyx
    return data  # ZYX -> zyx


class HDF5Dataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.

    Unifies dataset shapes: raw is loaded as czyx, label as zyx. The on-disk layout of each dataset is resolved from its
    `axis_order` attribute when present (written by PanSeg on export, e.g. "ZYX" or "CZYX") and otherwise from the
    (raw ndim, label ndim) pairing, see `resolve_dataset_layouts`. For 2D training this distinguishes legacy 2D on-disk
    data (YX/CYX, a singleton z is fabricated for the augmenter and squeezed before the UNet2D forward pass) from
    volumetric data (ZYX/CZYX), which is trained per z-slice: each z-slice is an independent 2D sample with patch
    z-extent 1, so downstream code (augmenter, trainer, export) sees exactly the same patch shapes as for legacy 2D data.

    Args:
        file_path (str): path to H5 file containing raw data as well as labels and per pixel weights (optional)
        augmenter (transforms.Augmenter): list of augmentations to be applied to the raw and label data sets
        patch_shape (tuple): shape of the patch to be extracted from the raw data set
        dimensionality (str): "2D" or "3D" model to train (also accepts "2d"/"2" and "3d"/"3")
        raw_internal_path (str or list): H5 internal path to the raw dataset
        label_internal_path (str or list): H5 internal path to the label dataset
        global_normalization (bool): if True, the mean and std of the raw data will be calculated over the whole dataset
        filter_patches (bool): if True (train), build the patch grid with
            `FilterSliceBuilder`: stride max(p // 2, 1) and patches below the
            foreground threshold dropped. If False (val), build a plain
            `SliceBuilder` grid: no threshold filtering and (by default)
            non-overlapping, so the val loss covers the whole volume and is not
            biased towards foreground-rich patches.
        stride_shape (tuple): patch stride for the unfiltered grid
            (`filter_patches=False`); defaults to `patch_shape`, i.e.
            non-overlapping. Must be None for the filtered train grid (which
            always uses max(p // 2, 1) as stride).
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
        filter_patches: bool = True,
        stride_shape: tuple[int, int, int] | None = None,
    ):
        self.file_path = file_path
        self.dimensionality = dimensionality

        with h5py.File(file_path, "r") as f:
            raw_ds = f[raw_internal_path]
            label_ds = f[label_internal_path]
            raw_layout, label_layout = resolve_dataset_layouts(
                dimensionality,
                raw_ds.ndim,
                label_ds.ndim,
                raw_axis_order=raw_ds.attrs.get("axis_order"),
                label_axis_order=label_ds.attrs.get("axis_order"),
            )
            if dimensionality in _DIMENSIONALITY_2D and "Z" in raw_layout:
                logger.info(
                    f"2D training on volumetric data in {file_path}: "
                    "training per z-slice."
                )
            self.raw = _apply_layout(raw_ds[:], raw_layout, ensure_channel=True)
            stats = calculate_stats(self.raw, global_normalization)
            self.augmenter = augmenter
            self.raw_transform = self.augmenter.raw_transform(stats)

            # create label/weight transform only in train/val phase
            self.label_transform = self.augmenter.label_transform()
            self.label = _apply_layout(label_ds[:], label_layout, ensure_channel=False)
            self._check_volume_sizes()

            # build slice indices for raw and label data sets
            if filter_patches:
                # train: overlapping, foreground-filtered grid
                if stride_shape is not None:
                    raise ValueError(
                        "stride_shape is only configurable for the unfiltered "
                        "(validation) grid; the filtered train grid always uses "
                        "max(patch_shape // 2, 1) as stride."
                    )
                slice_builder = FilterSliceBuilder(
                    self.raw,
                    self.label,
                    patch_shape=patch_shape,
                )
            else:
                # val: unfiltered grid, non-overlapping unless configured otherwise
                slice_builder = SliceBuilder(
                    self.raw,
                    self.label,
                    patch_shape=patch_shape,
                    stride_shape=stride_shape
                    if stride_shape is not None
                    else patch_shape,
                )
            self.raw_slices = slice_builder.raw_slices
            label_slices = slice_builder.label_slices
            assert label_slices is not None, "label slices exist when a label is given"
            self.label_slices = label_slices

            self.patch_count = len(self.raw_slices)
            logger.info(f"{self.patch_count} patches found in {file_path}")

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
        assert self.raw.ndim == 4, (
            f"Raw dataset must be loaded as czyx, got shape {self.raw.shape}"
        )
        assert self.label.ndim == 3, (
            f"Label dataset must be loaded as zyx, got shape {self.label.shape}"
        )

        assert self.raw.shape[1:] == self.label.shape, (
            f"Raw and label image data has to be of the same size, "
            f"got raw zyx shape {self.raw.shape[1:]} and "
            f"label zyx shape {self.label.shape} in {self.file_path}"
        )


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
