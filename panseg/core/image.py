import logging
import re
import time
from enum import Enum
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID, uuid4

import h5py
import numpy as np
from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from pydantic import BaseModel, model_validator

import panseg.functionals.dataprocessing as dp
from panseg.io.h5 import H5_EXTENSIONS, create_h5, read_h5_time_spacing
from panseg.io.io import smart_load_with_vs
from panseg.io.mesh import create_mesh
from panseg.io.tiff import (
    TIFF_EXTENSIONS,
    check_ome_single_file,
    create_tiff,
    read_ome_time_spacing,
)
from panseg.io.voxelsize import VoxelSize
from panseg.io.zarr import ZARR_EXTENSIONS, create_zarr, read_zarr_time_spacing

logger = logging.getLogger(__name__)
last_warning = 0.0

# Conversion factors from source time units to the canonical unit, seconds.
_TIME_UNITS_TO_SECONDS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "us": 1e-6,
    "\u00b5s": 1e-6,  # µs (micro sign)
    "\u03bcs": 1e-6,  # µs (Greek small letter mu)
    "microsecond": 1e-6,
    "microseconds": 1e-6,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}

# Canonical time units offered in the UIs, ordered from the shortest to the
# longest duration. A curated subset of the keys of _TIME_UNITS_TO_SECONDS:
# the aliases stay for parsing, the UIs only offer one spelling per unit.
TIME_UNIT_CHOICES = ("µs", "ms", "s", "min", "h")


class SemanticType(Enum):
    """
    Enum class for image types.

    Attributes:
        RAW (str): Reserved for raw images (e.g. microscopy images)
        PREDICTION (str): Reserved for model prediction
        SEGMENTATION (str): Reserved for segmentation masks
    """

    RAW = "raw"
    PREDICTION = "prediction"
    SEGMENTATION = "segmentation"


class ImageType(Enum):
    """
    Enum class for image types.

    Attributes:
        IMAGE (str): Image data
        LABEL (str): Label data
    """

    IMAGE = "image"
    LABEL = "labels"

    @classmethod
    def to_choices(cls) -> list[str]:
        return [member.value for member in cls]


class ImageDimensionality(Enum):
    """
    Enum class for the spatial dimensionality of an image.

    Dimensionality is spatial only: it never counts the time or channel
    axes. Time presence is the orthogonal ``is_timeseries`` property.

    Attributes:
        TWO (str): 2D images
        THREE (str): 3D images
    """

    TWO = "2D"
    THREE = "3D"


class ImageLayout(Enum):
    """
    Enum class for image layout.
    Axis available are:
    - T: Time
    - C: Channel
    - Z: Depth
    - Y: Height
    - X: Width

    Every layout is a projection of the canonical order T-C-Z-Y-X: only the
    present axes, in that relative order. The layout string alone carries the
    axes; axis indices, spatial dimensionality, and time presence are derived
    from it by projection and never stored.

    Attributes:
        YX (str): 2D image with X and Y axis
        CYX (str): 2D image with Channel, X and Y axis
        ZYX (str): 3D image with Z, X and Y axis
        CZYX (str): 3D image with Channel, Z, X and Y axis
        ZCYX (str): 3D image with Z, Channel, X and
        TYX (str): 2D timeseries with Time, X and Y axis
        TCYX (str): 2D timeseries with Time, Channel, X and Y axis
        TZYX (str): 3D timeseries with Time, Z, X and Y axis
        TCZYX (str): 3D timeseries with Time, Channel, Z, X and Y axis
    """

    YX = "YX"
    CYX = "CYX"
    ZYX = "ZYX"
    CZYX = "CZYX"
    ZCYX = "ZCYX"  # This is not supported, should be converted to CZYX during import
    TYX = "TYX"
    TCYX = "TCYX"
    TZYX = "TZYX"
    TCZYX = "TCZYX"

    @classmethod
    def to_choices(cls) -> list[str]:
        return [il.value for il in cls]

    @property
    def spatial_axes(self) -> str:
        """The spatial projection of the layout, i.e. the axes without T and C."""
        return self.value.replace("T", "").replace("C", "")

    @property
    def spatial_axis_indices(self) -> tuple[int, ...]:
        """Indices of the spatial axes in the layout, in Z, Y, X order."""
        return tuple(self.value.index(axis) for axis in self.spatial_axes)

    def axis_index(self, axis: str) -> int | None:
        """Index of the given axis in the layout, or None if absent."""
        if axis in self.value:
            return self.value.index(axis)
        return None

    @property
    def is_timeseries(self) -> bool:
        """True if the layout carries a time axis."""
        return "T" in self.value

    @property
    def dimensionality(self) -> ImageDimensionality:
        """Spatial dimensionality, independent of the time and channel axes."""
        if "Z" in self.value:
            return ImageDimensionality.THREE
        return ImageDimensionality.TWO

    @property
    def channel_axis(self) -> int | None:
        """Index of the channel axis in the layout, or None if absent."""
        if "C" in self.value:
            return self.value.index("C")
        return None

    @property
    def time_axis(self) -> int | None:
        """Index of the time axis in the layout, or None if absent."""
        if "T" in self.value:
            return self.value.index("T")
        return None


class ImageProperties(BaseModel):
    """
    Basic properties of an image.

    Attributes:
        name (str): Name of the image
        semantic_type (SemanticType): Semantic type of the image
        voxel_size (VoxelSize): Voxel size of the image
        image_layout (ImageLayout): Image layout of the image
        original_voxel_size (VoxelSize): Original voxel size of the image
        source_file_name (str | None): Name of the source file
        t_spacing (float | None): Time spacing between timepoints.
            None means the source carried no timing metadata.
        t_unit (str): Unit of the time spacing. Normalized to "s" at
            construction; other units (ms, µs, min, h) are converted to
            seconds.
    """

    name: str
    semantic_type: SemanticType
    voxel_size: VoxelSize
    image_layout: ImageLayout
    original_voxel_size: VoxelSize
    source_file_name: str | None = None
    t_spacing: float | None = None
    t_unit: str = "s"

    @model_validator(mode="after")
    def _normalize_time_spacing(self) -> "ImageProperties":
        factor = _TIME_UNITS_TO_SECONDS.get(self.t_unit.lower())
        if factor is None:
            raise ValueError(
                f"Time unit {self.t_unit!r} not recognized, should be one of "
                "s, ms, µs (us), min or h (hour)"
            )
        if self.t_spacing is not None:
            if self.t_spacing <= 0:
                raise ValueError("Time spacing must be positive")
            self.t_spacing = self.t_spacing * factor
        self.t_unit = "s"
        return self

    @property
    def dimensionality(self) -> ImageDimensionality:
        return self.image_layout.dimensionality

    @property
    def image_type(self) -> ImageType:
        if self.semantic_type in (SemanticType.RAW, SemanticType.PREDICTION):
            return ImageType.IMAGE
        elif self.semantic_type == SemanticType.SEGMENTATION:
            return ImageType.LABEL
        else:
            raise ValueError(f"Semantic type {self.semantic_type} not recognized")

    @property
    def channel_axis(self) -> int | None:
        return self.image_layout.channel_axis

    @property
    def time_axis(self) -> int | None:
        return self.image_layout.time_axis

    @property
    def is_timeseries(self) -> bool:
        return self.image_layout.is_timeseries

    def interpolation_order(self, image_default: int = 1) -> int:
        if self.image_type == ImageType.LABEL:
            return 0
        elif self.image_type == ImageType.IMAGE:
            return image_default
        else:
            raise ValueError(f"Image type {self.image_type} not recognized")


class PanSegImage:
    """Image class represents an image with its metadata and data."""

    _data: np.ndarray
    _properties: ImageProperties

    def __init__(self, data: np.ndarray, properties: ImageProperties):
        self._properties = properties
        data, properties = self._check_shape(data, properties)
        data = self._check_ndim(data)

        self._data = data
        self._properties = properties

        self._check_labels_have_no_channels()
        self._id = uuid4()

    def derive_new(self, data: np.ndarray, name: str, **kwargs) -> "PanSegImage":
        """
        Derive a new image from the current image.

        The new image will have the same properties as the original image, except for the name and the properties passed as kwargs.

        args:
            data (np.ndarray): New data
            name (str): New name
            **kwargs: other properties to change

        Returns:
            PanSegImage: New image
        """
        property_dict = self._properties.model_dump()

        if name == self.name:
            raise ValueError("New derived name should be different from the original")

        property_dict["name"] = name

        for key, value in kwargs.items():
            if key in property_dict:
                property_dict[key] = value
            else:
                raise ValueError(
                    f"Property {key} not recognized, should be one of {property_dict.keys()}"
                )

        new_properties = ImageProperties(**property_dict)
        return PanSegImage(data, new_properties)

    def set_t_spacing(self, t_spacing: float | None, t_unit: str = "s") -> None:
        """Replace the time spacing of this image, in place.

        Rebuilds the properties through the validated ImageProperties
        constructor, so the unit is normalized to seconds and the
        spacing checked for positivity - unlike a raw attribute
        assignment on the properties. The frame-by-frame loop uses it
        to stamp the inputs' shared spacing onto the split timepoints
        (a timepoint is not a timeseries and carries no spacing of its
        own; see split_timepoints), so task bodies preserve it through
        derive_new and property tasks can override it.

        Args:
            t_spacing (float | None): new time spacing in the given
                unit, or None to mark it unknown.
            t_unit (str): unit of t_spacing (s, ms, µs/us, min or h).
        """
        properties = self._properties.model_dump()
        properties["t_spacing"] = t_spacing
        properties["t_unit"] = t_unit
        self._properties = ImageProperties(**properties)

    @classmethod
    def from_napari_layer(cls, layer: Image | Labels) -> "PanSegImage":
        """
        Load a PanSegImage from a napari layer.

        Args:
            layer (Image | Labels): Napari layer to load
        """

        metadata = layer.metadata

        if "semantic_type" not in metadata:
            raise ValueError("Semantic type not found in metadata")

        semantic_type = SemanticType(metadata["semantic_type"])

        if isinstance(layer, Image):
            image_type = ImageType.IMAGE
        elif isinstance(layer, Labels):
            image_type = ImageType.LABEL

        if "original_voxel_size" not in metadata:
            raise ValueError("Original voxel size not found in metadata")

        original_voxel_size = VoxelSize(**metadata["original_voxel_size"])

        if "image_layout" not in metadata:
            raise ValueError("Image layout not found in metadata")

        image_layout = ImageLayout(metadata["image_layout"])

        if "voxel_size" not in metadata:
            raise ValueError("Voxel size not found in metadata")
        new_voxel_size = VoxelSize(**metadata["voxel_size"])

        source_file_name = metadata.get("source_file_name", None)

        # Old layers lack the time spacing metadata: it defaults to unknown.
        t_spacing = metadata.get("t_spacing", None)
        t_unit = metadata.get("t_unit", "s")

        # Loading from napari layer, the id needs to be present in the metadata
        # If not present, the layer is corrupted
        if "id" in metadata:
            id = metadata["id"]
        else:
            raise ValueError("ID not found in metadata")

        properties = ImageProperties(
            name=layer.name,
            semantic_type=semantic_type,
            voxel_size=new_voxel_size,
            image_layout=image_layout,
            original_voxel_size=original_voxel_size,
            source_file_name=source_file_name,
            t_spacing=t_spacing,
            t_unit=t_unit,
        )

        if image_type != properties.image_type:
            raise ValueError(
                f"Image type {image_type} does not match semantic type {properties.semantic_type}"
            )

        ps_image = cls(layer.data, properties)  # type: ignore
        ps_image._id = id
        return ps_image

    def split_channels(self) -> list["PanSegImage"]:
        if not self.is_multichannel:
            return [self]
        assert self.channel_axis is not None, "No channel axis known!"

        # The split image keeps every axis except C.
        prefix = "T" if self.is_timeseries else ""
        new_image_layout = ImageLayout(prefix + self.image_layout.spatial_axes)

        images = []
        for ch in range(self.shape[self.channel_axis]):
            images.append(
                self.derive_new(
                    data=self.get_data(channel=ch),
                    name=self.name + f"_{ch}",
                    image_layout=new_image_layout,
                )
            )
        return images

    def split_timepoints(self) -> list["PanSegImage"]:
        """Split a timeseries into single-timepoint images, one per timepoint.

        Mirrors split_channels: each timepoint is a derive_new of this image
        with the T axis dropped from the layout (TZYX->ZYX, TYX->YX,
        TCZYX->CZYX, TCYX->CYX), the time slice as data and the name
        f"{name}_t{i}" (the channel naming convention). A timepoint is not a
        timeseries: it carries no time spacing (t_spacing None, t_unit the
        canonical "s"), so the caller restacks with restack_timepoints to
        stamp the parent spacing back on.

        A still image returns itself as the only timepoint.
        """
        if not self.is_timeseries:
            return [self]
        assert self.time_axis is not None, "No time axis known!"

        # The split image keeps every axis except T.
        new_image_layout = ImageLayout(self.image_layout.value.replace("T", ""))

        images = []
        for t_index in range(self.shape[self.time_axis]):
            images.append(
                self.derive_new(
                    data=np.take(self._data, t_index, axis=self.time_axis),
                    name=self.name + f"_t{t_index}",
                    image_layout=new_image_layout,
                    t_spacing=None,
                )
            )
        return images

    def merge_with(self, image: "PanSegImage"):
        """Merge two images along the channel dimension."""
        if not all(
            (
                self.semantic_type == image.semantic_type,
                self.voxel_size == image.voxel_size,
                self.dimensionality == image.dimensionality,
                self.is_timeseries == image.is_timeseries,
                self.properties.t_spacing == image.properties.t_spacing,
            )
        ):
            raise ValueError("Images can't be merged, not compatible!")

        images = self.split_channels()
        images.extend(image.split_channels())

        prefix = "T" if self.is_timeseries else ""
        new_image_layout = ImageLayout(prefix + "C" + self.image_layout.spatial_axes)

        new_props = ImageProperties(
            name=self.name + "_merged",
            semantic_type=self.semantic_type,
            voxel_size=self.voxel_size,
            image_layout=new_image_layout,
            original_voxel_size=self.original_voxel_size,
            source_file_name=self.source_file_name,
            t_spacing=self.properties.t_spacing,
        )

        stack_axis = 1 if self.is_timeseries else 0
        data = np.stack([im.get_data() for im in images], axis=stack_axis)
        return PanSegImage(data, new_props)

    def to_napari_layer_tuple(self) -> LayerDataTuple:
        """
        Prepare and normalise the image to be loaded as a napari layer.

        All the metadata will be stored in the metadata of the layer.

        Returns:
            LayerDataTuple: Tuple containing the 0-1-normalised data, metadata, and type of the image.
        """
        # Dump the model properties to a dictionary
        metadata = self._properties.model_dump()

        # Preserve the ID in the metadata
        metadata["id"] = self.id

        # Create the LayerDataTuple
        layer_data_tuple = (
            self.get_data(),
            {
                "name": self.name,
                "scale": self.scale,
                "axis_labels": [ax.lower() for ax in self.image_layout.value],
                "metadata": metadata,
            },
            self.image_type.value,
        )

        return LayerDataTuple(layer_data_tuple)

    def to_h5(
        self, path: Path | str, key: str | None, mode: Literal["a", "w", "w-"] = "a"
    ) -> None:
        """Save the image with all metadata to an h5 file.

        Args:
            path (Path): Path to the h5 file
            key (str): Key to save the data in the h5 file
            mode (str): Mode to open the h5 file ['a', 'w', 'w-']
        """

        if isinstance(path, str):
            path = Path(path)

        if path.suffix.lower() not in H5_EXTENSIONS:
            raise ValueError(
                f"File format {path.suffix} not supported, should be one of {H5_EXTENSIONS}"
            )

        key = key if key is not None else self.name

        data = self._data
        voxel_size = self.voxel_size
        metadata = self.properties.model_dump_json()

        with h5py.File(path, mode=mode) as f:
            f.create_dataset(key, data=data)
            if voxel_size.voxels_size is not None:
                f[key].attrs["element_size_um"] = voxel_size.voxels_size
            f[key].attrs["panseg_image_metadata_json"] = metadata
            f[key].attrs["axis_order"] = self.image_layout.value
            if self.is_timeseries and self.properties.t_spacing is not None:
                f[key].attrs["t_spacing"] = self.properties.t_spacing
                f[key].attrs["t_spacing_unit"] = self.properties.t_unit

    @classmethod
    def from_h5(cls, path: Path | str, key: str) -> "PanSegImage":
        """Build an instance of PanSegImage from an h5 file."""

        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise ValueError(f"File {path} not found")

        with h5py.File(path, "r") as f:
            if key not in f:
                raise ValueError(f"Key {key} not found in the h5 file")

            data: np.ndarray = f[key][...]  # type: ignore
            metadata = f[key].attrs.get("panseg_image_metadata_json", None)

        if metadata is None:
            raise ValueError("PanSeg metadata not found in the h5 file")

        properties = ImageProperties.model_validate_json(metadata)
        return cls(data, properties)

    def _check_ndim(self, data: np.ndarray) -> np.ndarray:
        expected_ndim = len(self.image_layout.value)
        if data.ndim != expected_ndim:
            raise ValueError(
                f"Data has shape {data.shape} but should have {expected_ndim} dimensions for layout {self.image_layout}"
            )

        return data

    def _check_shape(
        self, data: np.ndarray, properties: ImageProperties
    ) -> tuple[np.ndarray, ImageProperties]:
        if self.image_layout == ImageLayout.ZCYX:
            logger.warning(
                "Image layout is ZCYX but should have been converted to CZYX. PanSeg is doing this now."
            )
            properties.image_layout = ImageLayout.CZYX
            data = np.moveaxis(data, 0, 1)
            return self._check_shape(data, properties)

        # Singleton squeeze rule: drop every length-1 axis except Y and X,
        # the layout is the projection onto what remains.
        layout = self.image_layout
        axes = layout.value
        drop_idx = [
            i
            for i, (ax, n) in enumerate(zip(axes, data.shape))
            if ax not in "YX" and n == 1
        ]
        if not drop_idx:
            return data, properties

        dropped_axes = [ax for i, ax in enumerate(axes) if i in drop_idx]
        remaining_axes = "".join(ax for i, ax in enumerate(axes) if i not in drop_idx)
        dropped_words = {
            "T": "timepoint",
            "C": "channel",
            "Z": "z slice",
        }
        dropped = " and ".join(dropped_words[ax] for ax in dropped_axes)
        logger.warning(
            f"Image layout is {layout.value} but data has only one {dropped}, casting to {remaining_axes}"
        )
        for i in reversed(drop_idx):
            data = np.take(data, 0, axis=i)
        properties.image_layout = ImageLayout(remaining_axes)
        if "T" in dropped_axes:
            properties.t_spacing = None
        return data, properties

    def _check_labels_have_no_channels(self) -> None:
        if self.image_type == ImageType.LABEL:
            if self.channel_axis is not None:
                raise ValueError(
                    f"Label images should not have channel axis, but found layout {self.image_layout}"
                )

    @property
    def requires_scaling(self) -> bool:
        """Returns True if the image has a different voxel size."""
        if self.voxel_size != self.original_voxel_size:
            return True
        return False

    def _get_data_channel_layout(
        self, channel: int | None = None, normalize_01: bool = True
    ) -> np.ndarray:
        """Get the data if the layout is multichannel."""
        if channel is None:
            data = self._data
            if normalize_01:
                assert self.channel_axis is not None
                data = dp.normalize_01_channel_wise(data, self.channel_axis)
            return data

        if channel < 0:
            raise ValueError(f"Channel should be a positive integer, but got {channel}")

        assert self.channel_axis is not None
        if channel > self._data.shape[self.channel_axis]:
            raise ValueError(
                f"Channel {channel} is out of bounds, the image has {self._data.shape[self.channel_axis]} channels"
            )

        data = dp.select_channel(self._data, channel, self.channel_axis)
        if normalize_01:
            data = dp.normalize_01(data)
        return data

    def get_data(
        self, channel: int | None = None, normalize_01: bool = False
    ) -> np.ndarray:
        """Returns the data of the image.

        Args:
            channel (int): Channel to load from the image (if the image is
                multichannel). If None, all channels are loaded.
            normalize_01 (bool): Normalize the data between 0 and 1, if the
                image is a Label image, the data is not normalized.
        """
        data = self._data
        if self.image_type == ImageType.LABEL:
            return data

        if self.channel_axis is not None:
            data = self._get_data_channel_layout(channel, normalize_01)
        elif normalize_01:
            data = dp.normalize_01(data)
        return data

    @property
    def scale(self) -> tuple[float, ...]:
        """Returns the scale of the image.

        The scale is equal to the voxel size in each spatial dimension, 1.0
        in the time dimension and 1 in the channel dimension. The time axis
        stays in timepoint indices (the napari t axis shows the existing
        timepoints, not elapsed time): the spacing is carried in
        properties.t_spacing and surfaced by the export metadata and the
        input tab info panel.
        """
        if self.image_layout == ImageLayout.ZCYX:
            raise ValueError(
                f"Image layout {self.image_layout} not supported, should have been converted to CZYX"
            )

        axis_scales = {
            "T": 1.0,
            "C": 1.0,
            "Z": self.voxel_size.z,
            "Y": self.voxel_size.y,
            "X": self.voxel_size.x,
        }
        return tuple(axis_scales[ax] for ax in self.image_layout.value)

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the image."""
        return self._data.shape

    @property
    def properties(self) -> ImageProperties:
        """Returns the properties of the image."""
        return self._properties

    @property
    def voxel_size(self) -> VoxelSize:
        """Returns the voxel size of the image."""
        return self._properties.voxel_size

    @property
    def original_voxel_size(self) -> VoxelSize:
        return self._properties.original_voxel_size

    @property
    def source_file_name(self) -> str | None:
        return self._properties.source_file_name

    @property
    def name(self) -> str:
        return self._properties.name

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def unique_name(self) -> str:
        return f"{self.name}_{self.id}"

    @property
    def image_type(self) -> ImageType:
        return self._properties.image_type

    @property
    def semantic_type(self) -> SemanticType:
        return self._properties.semantic_type

    @property
    def image_layout(self) -> ImageLayout:
        return self._properties.image_layout

    @property
    def dimensionality(self) -> ImageDimensionality:
        return self._properties.dimensionality

    @property
    def channel_axis(self) -> int | None:
        return self._properties.channel_axis

    @property
    def time_axis(self) -> int | None:
        return self._properties.time_axis

    @property
    def is_multichannel(self) -> bool:
        """Returns True if the image is multichannel, False otherwise."""
        return self.channel_axis is not None

    @property
    def is_timeseries(self) -> bool:
        """Returns True if the image carries a time axis, False otherwise."""
        return self._properties.is_timeseries

    def interpolation_order(self, image_default: int = 1) -> int:
        """Returns the default interpolation order used for the image."""
        return self._properties.interpolation_order(image_default)

    def has_valid_voxel_size(self) -> bool:
        """Returns True if the voxel size is valid (not None), False otherwise."""
        return self.voxel_size.voxels_size is not None

    def has_valid_original_voxel_size(self) -> bool:
        """Returns True if the original voxel size is valid (not None), False otherwise."""
        return self.original_voxel_size.voxels_size is not None


def restack_timepoints(
    timepoints: list[PanSegImage],
    t_spacing: float | None,
    t_unit: str = "s",
    name: str | None = None,
) -> PanSegImage:
    """Restack single-timepoint images into a timeseries along a new outer T axis.

    The inverse of split_timepoints: the timepoints are stacked with
    np.stack along a new leading axis and the layout gains a T prefix
    (ZYX->TZYX, YX->TYX). t_spacing/t_unit are stamped by the caller (they
    are lost in the split) - usually the parent timeseries's spacing.

    Args:
        timepoints (list[PanSegImage]): single-timepoint images; every
            timepoint must be a still image and agree in shape, layout,
            voxel_size, semantic_type and t_spacing (the same checks as
            merge_with).
        t_spacing (float | None): time spacing between the restacked
            timepoints, None if unknown.
        t_unit (str): unit of the time spacing, normalized to seconds.
        name (str | None): name of the restacked image; defaults to the
            first timepoint's name plus "_restacked".
    """
    if not timepoints:
        raise ValueError("Restacking needs at least one timepoint")

    first = timepoints[0]
    for timepoint in timepoints:
        if timepoint.is_timeseries:
            raise ValueError(
                f"Image {timepoint.name} to restack is not a single timepoint "
                f"(layout {timepoint.image_layout})"
            )
    for timepoint in timepoints[1:]:
        if not all(
            (
                timepoint.semantic_type == first.semantic_type,
                timepoint.voxel_size == first.voxel_size,
                timepoint.dimensionality == first.dimensionality,
                timepoint.image_layout == first.image_layout,
                timepoint.shape == first.shape,
                timepoint.properties.t_spacing == first.properties.t_spacing,
            )
        ):
            raise ValueError(
                f"Timepoints {first.name} and {timepoint.name} can't be "
                "restacked, not compatible!"
            )

    if name is None:
        name = first.name + "_restacked"

    new_props = ImageProperties(
        name=name,
        semantic_type=first.semantic_type,
        voxel_size=first.voxel_size,
        image_layout=ImageLayout("T" + first.image_layout.value),
        original_voxel_size=first.original_voxel_size,
        source_file_name=first.source_file_name,
        t_spacing=t_spacing,
        t_unit=t_unit,
    )
    data = np.stack([timepoint.get_data() for timepoint in timepoints], axis=0)
    return PanSegImage(data, new_props)


def _layout_tokens(stack_layout: str) -> list[tuple[str, bool]]:
    """Tokenize a stack layout into (letter, inverted) pairs, in written order.

    A '-' inverts the letter that follows it.
    """
    tokens: list[tuple[str, bool]] = []
    invert_next = False
    for char in stack_layout:
        if char == "-":
            invert_next = True
            continue
        tokens.append((char, invert_next))
        invert_next = False
    return tokens


def stack_sort(stack_layout: str, data, voxel_size):
    """Sort the stack layout, data, and voxelsize

    Makes the image stack layout unique for any number of dimensions.
    """
    # Canonical rank order: T, C, Z, Y, X. T and C are non-spatial axes and
    # do not take part in the voxel size mapping.
    sort_order = [
        ("T", 0),
        ("C", 1),
        ("Z", 2),
        ("Y", 3),
        ("X", 4),
    ]

    tokens = _layout_tokens(stack_layout)
    # ZCXY -> [2,1,4,3]
    invert = [inverted for _, inverted in tokens]
    sort_idxs = []
    sort_idxs_wo_channel = []
    for c, _ in tokens:
        sort_idxs.extend([n for c_sorted, n in sort_order if c == c_sorted])
        sort_idxs_wo_channel.extend(
            [n for c_sorted, n in sort_order if c == c_sorted and c not in "CT"]
        )
    # fill in gaps, like missing channel dimension:
    sort_idxs = np.argsort(sort_idxs)

    data = np.flip(data, axis=np.nonzero(invert)[0])
    data = np.transpose(data, axes=sort_idxs)
    stack_layout = stack_layout.replace("-", "")
    stack_layout = "".join([stack_layout[i] for i in sort_idxs])

    # ZCXY -> [1,0,3,2] -> [1,3,2] -> [0,2,1]
    # layout->sort_idxs ->sort_w/o_ch->sorting for voxelsize
    sort_idxs_wo_channel = np.argsort(sort_idxs_wo_channel)
    if len(sort_idxs_wo_channel) == 2:
        sort_idxs_wo_channel = np.insert(sort_idxs_wo_channel + 1, 0, 0)
    if voxel_size.voxels_size is not None:
        new_voxels = np.array(voxel_size.voxels_size)[sort_idxs_wo_channel]
        voxel_size = VoxelSize(voxels_size=tuple(new_voxels))
    return stack_layout, data, voxel_size


# A stack layout optionally carries a slice in brackets after the letters,
# e.g. "txyz[:3,:,:]": the letters order the axes, the slice truncates the
# data before the axes are inverted and reordered.
_STACK_LAYOUT_SPEC_PATTERN = re.compile(
    r"(?P<axes>[-A-Za-z]*)(?:\[(?P<slicing>[^\[\]]*)\])?"
)


def split_stack_layout(stack_layout: str) -> tuple[str, str | None]:
    """Split a stack layout spec into its layout letters and an optional slice.

    The slice, when present, is what sits between the brackets without the
    brackets themselves, e.g. "txyz[:3,:,:]" -> ("txyz", ":3,:,:").

    Raises:
        ValueError: if the spec is not letters optionally followed by a
            bracketed slice, or carries letters outside t, c, z, y, x.
    """
    match = _STACK_LAYOUT_SPEC_PATTERN.fullmatch(stack_layout.strip())
    if match is None:
        raise ValueError(
            f"Stack layout {stack_layout!r} is not understood: expected layout "
            "letters (t, c, z, y, x, each optionally prefixed with '-') with an "
            "optional slice in brackets, e.g. 'txyz[:3,:,:]'."
        )
    axes, slicing = match.group("axes"), match.group("slicing")
    unknown = sorted(
        {char for char in axes if char != "-" and char.upper() not in "TCZYX"}
    )
    if unknown:
        raise ValueError(
            f"Stack layout {stack_layout!r} is not understood: unknown axis "
            f"letter(s) {''.join(unknown)!r}, expected t, c, z, y or x, each "
            "optionally prefixed with '-'."
        )
    return axes, slicing or None


def _parse_slicing(slicing: str) -> list[slice | int]:
    """Parse a slicing string like "0:3, :, :, :50" into numpy index entries."""
    entries: list[slice | int] = []
    for part in slicing.split(","):
        part = part.strip()
        try:
            if ":" in part:
                fields = [field.strip() for field in part.split(":")]
                entries.append(
                    slice(*(int(field) if field else None for field in fields))
                )
            elif part:
                entries.append(int(part))
            else:
                raise ValueError("empty entry")
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Slicing {slicing!r} is not understood: expected one entry per "
                f"axis like '0:3', ':', '10:20:2' or an integer, got {part!r}."
            ) from err
    return entries


def crop_to_stack_layout(
    data: np.ndarray, stack_layout: str, slicing: str
) -> tuple[str, np.ndarray]:
    """Crop data along the axes of stack_layout, in the order they are written.

    The entries follow the user-supplied layout, before any axis inversion or
    reordering: entry i applies to axis i of stack_layout. There may be fewer
    entries than axes, the remaining axes are then left untouched. An integer
    entry selects a single index and drops its axis: the letter, and its '-'
    inversion marker, leaves the layout too.

    Returns:
        The (possibly shortened) stack layout and the cropped data.
    """
    entries = _parse_slicing(slicing)
    n_axes = len(_layout_tokens(stack_layout))
    if len(entries) > n_axes:
        raise ValueError(
            f"Slicing {slicing!r} has {len(entries)} entries but the stack "
            f"layout {stack_layout!r} has {n_axes} axes."
        )
    dropped = {i for i, entry in enumerate(entries) if isinstance(entry, int)}
    data = data[tuple(entries)]
    if dropped:
        stack_layout = "".join(
            f"{'-' if inverted else ''}{char}"
            for i, (char, inverted) in enumerate(_layout_tokens(stack_layout))
            if i not in dropped
        )

    return stack_layout, data


def import_image(
    path: Path,
    key: str | None = None,
    image_name: str = "image",
    semantic_type: str = "raw",
    stack_layout: str = "YX",
) -> PanSegImage | list[PanSegImage]:
    """
    Open an image file and create a PanSegImage object.

    Args:
        path (Path): Path to the image file
        key (Optional[str]): Key to load data from h5 or zarr files
        image_name (str): Name of the image (a unique name to identify the image)
        semantic_type (str): Semantic type of the image, should be raw,
            segmentation, prediction or label.
        stack_layout (str): Layout of the image, any of the letters t, c, z, y, x
            Prepend any letter with a minus to invert it.
            A slice can follow the letters, e.g. "txyz[:3,:,:]", to slice the
            data before the axes are reordered.
    """
    global last_warning
    stack_layout, slicing = split_stack_layout(stack_layout.upper())
    is_tiff = path.suffix.lower() in TIFF_EXTENSIONS
    if is_tiff:
        # Multi-file OME-TIFF (UUID/FileName chain) is rejected at import.
        check_ome_single_file(path)

    data, voxel_size = smart_load_with_vs(path, key)
    if voxel_size is None:
        voxel_size = VoxelSize()

    # The time spacing is extracted from OME-TIFF metadata or the PanSeg
    # h5/zarr attrs; other formats stay T-unaware and import with it unknown.
    t_spacing: float | None = None
    t_unit = "s"
    if is_tiff:
        t_spacing, t_unit = read_ome_time_spacing(path)
    elif path.suffix.lower() in H5_EXTENSIONS:
        t_spacing, t_unit = read_h5_time_spacing(path, key)
    elif path.suffix.lower() in ZARR_EXTENSIONS:
        t_spacing, t_unit = read_zarr_time_spacing(path, key)

    if not len(stack_layout.replace("-", "")) == len(data.shape):
        raise ValueError(
            f"Data to import has shape {data.shape}, incompatible with chosen layout {stack_layout}"
        )

    original_data_shape = data.shape
    # The slicing follows the user-supplied layout order, so it happens before
    # stack_sort reorders (and inverts) the axes to the canonical T-C-Z-Y-X.
    if slicing is not None:
        stack_layout, data = crop_to_stack_layout(data, stack_layout, slicing)
    stack_layout, data, voxel_size = stack_sort(stack_layout, data, voxel_size)

    images = []
    image_layout = ImageLayout(stack_layout)

    if image_layout in [
        ImageLayout.ZYX,
        ImageLayout.YX,
        ImageLayout.TZYX,
        ImageLayout.TYX,
    ]:
        image_properties = ImageProperties(
            name=image_name,
            semantic_type=SemanticType(semantic_type),
            voxel_size=voxel_size,
            image_layout=image_layout,
            original_voxel_size=voxel_size,
            source_file_name=path.stem,
            t_spacing=t_spacing if image_layout.is_timeseries else None,
            t_unit=t_unit if image_layout.is_timeseries else "s",
        )
        if image_properties.image_type == ImageType.IMAGE:
            data = dp.normalize_01(data)

        return PanSegImage(data=data, properties=image_properties)

    elif image_layout is ImageLayout.CYX:
        if (data.shape[0] > min(data.shape) or data.shape[0] > 9) and (
            time.time() - last_warning
        ) > 120:
            last_warning = time.time()
            raise ValueError(
                f"Double check the stack layout and try again!\nData shape {original_data_shape}"
            )

        for ch in range(data.shape[0]):
            image_properties = ImageProperties(
                name=image_name + f"_{ch}",
                semantic_type=SemanticType(semantic_type),
                voxel_size=voxel_size,
                image_layout=ImageLayout.YX,
                original_voxel_size=voxel_size,
                source_file_name=path.stem,
            )
            images.append(PanSegImage(data=data[ch], properties=image_properties))

    elif image_layout is ImageLayout.CZYX:
        if (data.shape[0] > min(data.shape) or data.shape[0] > 9) and (
            time.time() - last_warning
        ) > 120:
            last_warning = time.time()
            raise ValueError(
                f"Double check the stack layout and try again!\nData shape {original_data_shape}"
            )

        for ch in range(data.shape[0]):
            image_properties = ImageProperties(
                name=image_name + f"_{ch}",
                semantic_type=SemanticType(semantic_type),
                voxel_size=voxel_size,
                image_layout=ImageLayout.ZYX,
                original_voxel_size=voxel_size,
                source_file_name=path.stem,
            )
            images.append(PanSegImage(data=data[ch], properties=image_properties))

    elif image_layout is ImageLayout.TCYX:
        if (data.shape[1] > min(data.shape[2:]) or data.shape[1] > 9) and (
            time.time() - last_warning
        ) > 120:
            last_warning = time.time()
            raise ValueError(
                f"Double check the stack layout and try again!\nData shape {original_data_shape}"
            )

        for ch in range(data.shape[1]):
            image_properties = ImageProperties(
                name=image_name + f"_{ch}",
                semantic_type=SemanticType(semantic_type),
                voxel_size=voxel_size,
                image_layout=ImageLayout.TYX,
                original_voxel_size=voxel_size,
                source_file_name=path.stem,
                t_spacing=t_spacing,
                t_unit=t_unit,
            )
            images.append(PanSegImage(data=data[:, ch], properties=image_properties))

    elif image_layout is ImageLayout.TCZYX:
        if (data.shape[1] > min(data.shape[2:]) or data.shape[1] > 9) and (
            time.time() - last_warning
        ) > 120:
            last_warning = time.time()
            raise ValueError(
                f"Double check the stack layout and try again!\nData shape {original_data_shape}"
            )

        for ch in range(data.shape[1]):
            image_properties = ImageProperties(
                name=image_name + f"_{ch}",
                semantic_type=SemanticType(semantic_type),
                voxel_size=voxel_size,
                image_layout=ImageLayout.TZYX,
                original_voxel_size=voxel_size,
                source_file_name=path.stem,
                t_spacing=t_spacing,
                t_unit=t_unit,
            )
            images.append(PanSegImage(data=data[:, ch], properties=image_properties))

    elif image_layout is ImageLayout.ZCYX:
        logger.warning("##### WARNING: Depricated image layout ZCYX used #####")
        if (data.shape[1] > min(data.shape) or data.shape[1] > 9) and (
            time.time() - last_warning
        ) > 120:
            last_warning = time.time()
            raise ValueError(
                f"Double check the stack layout and try again!\nData shape {original_data_shape}"
            )

        for ch in range(data.shape[1]):
            image_properties = ImageProperties(
                name=image_name + f"_{ch}",
                semantic_type=SemanticType(semantic_type),
                voxel_size=voxel_size,
                image_layout=ImageLayout.ZYX,
                original_voxel_size=voxel_size,
                source_file_name=path.stem,
            )
            images.append(PanSegImage(data=data[:, ch], properties=image_properties))

    return images


def _image_postprocessing(
    image: PanSegImage, scale_to_origin: bool, export_dtype: str
) -> tuple[np.ndarray, VoxelSize]:
    assert isinstance(image, PanSegImage), f"type: {type(image)}"
    if scale_to_origin and image.requires_scaling:
        data = dp.scale_image_to_voxelsize(
            image.get_data(),
            input_voxel_size=image.voxel_size.as_tuple(),
            output_voxel_size=image.original_voxel_size.as_tuple(),
            order=image.interpolation_order(),
        )
        new_voxel_size = image.original_voxel_size
    else:
        data = image.get_data()
        new_voxel_size = image.voxel_size

    if image.image_type == ImageType.IMAGE:
        data = dp.normalize_01(data)
        if export_dtype in ["uint8", "uint16"]:
            max_val = np.iinfo(export_dtype).max
            data = (data * max_val).astype(export_dtype)
        elif export_dtype in ["float32", "float64"]:
            data = data.astype(export_dtype)
        else:
            raise ValueError(
                f"Data type {export_dtype} not recognized, should be uint8, uint16, float32 or float64"
            )
    elif image.image_type == ImageType.LABEL:
        if export_dtype in ["float32", "float64"]:
            raise ValueError(
                f"Data type {export_dtype} not recognized for label image, should be uint8 or uint16"
            )
        data = data.astype(export_dtype)
    else:
        raise ValueError(
            f"Image type {image.image_type} not recognized, should be image or label"
        )

    return data, new_voxel_size


def save_image(
    image: PanSegImage,
    export_directory: Path,
    name_pattern: str,
    key: str | None = None,
    scale_to_origin: bool = True,
    export_format: str = "tiff",
    data_type: str = "uint16",
    export_mesh: Optional[str] = None,
    close_mesh: bool = False,
) -> None:
    """
    Write a PanSegImage object to disk.

    Args:
        image (PanSegImage): input image to be saved to disk
        export_directory (Path): output directory path where the image will be saved
        name_pattern (str): output file name pattern, can contain the {image_name} or {file_name} tokens
            to be replaced in the final file name.
        key (str | None): key for the image (used only for h5 and zarr formats).
        scale_to_origin (bool): scale the voxel size to the original one
        export_format (str): file format (tiff, h5, zarr)
        data_type (str): data type to save the image.
    """

    data, voxel_size = _image_postprocessing(
        image, scale_to_origin=scale_to_origin, export_dtype=data_type
    )

    directory = Path(export_directory)
    directory.mkdir(parents=True, exist_ok=True)

    name_pattern = name_pattern.replace("{image_name}", image.name)

    if image.source_file_name is not None:
        name_pattern = name_pattern.replace("{file_name}", image.source_file_name)

    if export_format == "tiff":
        file_path_name = directory / f"{name_pattern}.tiff"
        create_tiff(
            path=file_path_name,
            stack=data,
            voxel_size=voxel_size,
            layout=image.image_layout.value,
            t_spacing=image.properties.t_spacing,
        )

    elif export_format == "zarr":
        if key is None or key == "":
            raise ValueError("Key is required for zarr format")

        file_path_name = directory / f"{name_pattern}.zarr"
        create_zarr(
            path=file_path_name,
            stack=data,
            voxel_size=voxel_size,
            key=key,
            axis_order=image.image_layout.value,
            t_spacing=image.properties.t_spacing,
            t_spacing_unit=image.properties.t_unit,
        )

    elif export_format == "h5":
        if key is None or key == "":
            raise ValueError("Key is required for h5 format")

        file_path_name = directory / f"{name_pattern}.h5"
        create_h5(
            path=file_path_name,
            stack=data,
            voxel_size=voxel_size,
            key=key,
            axis_order=image.image_layout.value,
            t_spacing=image.properties.t_spacing,
            t_spacing_unit=image.properties.t_unit,
        )

    else:
        raise ValueError(
            f"Export format {export_format} not recognized, should be tiff, h5 or zarr"
        )

    if export_mesh is not None and export_mesh.lower() != "no":
        if image.image_type != ImageType.LABEL:
            raise ValueError(
                "Mesh export only supported for Segmentations, "
                f"received: {image.image_type}, {image.name}"
            )
        if image.dimensionality != ImageDimensionality.THREE:
            raise ValueError(
                "Mesh export only supported for 3D, "
                f"received: {image.dimensionality}, {image.name}"
            )
        if image.is_timeseries:
            # one mesh file per timepoint keeps the 1:1 timepoint-file
            # mapping; empty timepoints export their (empty) scene too
            time_axis = image.time_axis
            assert time_axis is not None, "No time axis known!"
            for t_index in range(data.shape[time_axis]):
                frame = np.take(data, t_index, axis=time_axis)
                file_path_name = (
                    directory / f"{name_pattern}_t{t_index:03d}.{export_mesh}"
                )
                create_mesh(
                    path=file_path_name,
                    stack=frame,
                    voxel_size=voxel_size,
                    close_mesh=close_mesh,
                )
        else:
            file_path_name = directory / f"{name_pattern}.{export_mesh}"
            create_mesh(
                path=file_path_name,
                stack=data,
                voxel_size=voxel_size,
                close_mesh=close_mesh,
            )
