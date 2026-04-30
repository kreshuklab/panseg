import logging
import time
from enum import Enum
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID, uuid4

import h5py
import numpy as np
from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from pydantic import BaseModel

import panseg.functionals.dataprocessing as dp
from panseg.io.h5 import H5_EXTENSIONS, create_h5
from panseg.io.io import smart_load_with_vs
from panseg.io.mesh import create_mesh
from panseg.io.tiff import create_tiff
from panseg.io.voxelsize import VoxelSize
from panseg.io.zarr import create_zarr

logger = logging.getLogger(__name__)
last_warning = 0.0


class SemanticType(Enum):
    """
    Enum class for image types.

    Attributes:
        RAW (str): Reserved for raw images (e.g. microscopy images)
        LABEL (str): Reserved for ground truth labels
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
    Enum class for image dimensionality.

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
    - X: Width
    - Y: Height
    - Z: Depth
    - C: Channel

    Attributes:
        YX (str): 2D image with X and Y axis
        CYX (str): 2D image with Channel, X and Y axis
        ZYX (str): 3D image with Z, X and Y axis
        CZYX (str): 3D image with Channel, Z, X and Y axis
        ZCYX (str): 3D image with Z, Channel, X and
    """

    YX = "YX"
    CYX = "CYX"
    ZYX = "ZYX"
    CZYX = "CZYX"
    ZCYX = "ZCYX"  # This is not supported, should be converted to CZYX during import

    @classmethod
    def to_choices(cls) -> list[str]:
        return [il.value for il in cls]


class ImageProperties(BaseModel):
    """
    Basic properties of an image.

    Attributes:
        name (str): Name of the image
        semantic_type (SemanticType): Semantic type of the image
        voxel_size (VoxelSize): Voxel size of the image
        image_layout (ImageLayout): Image layout of the image
        original_voxel_size (VoxelSize): Original voxel size of the image
    """

    name: str
    semantic_type: SemanticType
    voxel_size: VoxelSize
    image_layout: ImageLayout
    original_voxel_size: VoxelSize
    source_file_name: str | None = None

    @property
    def dimensionality(self) -> ImageDimensionality:
        if self.image_layout in (ImageLayout.YX, ImageLayout.CYX):
            return ImageDimensionality.TWO
        elif self.image_layout in (
            ImageLayout.ZYX,
            ImageLayout.CZYX,
            ImageLayout.ZCYX,
        ):
            return ImageDimensionality.THREE
        else:
            raise ValueError(f"Image layout {self.image_layout} not recognized")

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
        if self.image_layout in (ImageLayout.CYX, ImageLayout.CZYX):
            return 0
        elif self.image_layout == ImageLayout.ZCYX:
            return 1
        elif self.image_layout in (ImageLayout.YX, ImageLayout.ZYX):
            return None
        else:
            raise ValueError(f"Image layout {self.image_layout} not recognized")

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

        property_dict = self._properties.model_dump(exclude_none=True)

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

        if self.dimensionality == ImageDimensionality.TWO:
            new_image_layout = ImageLayout.YX
        else:
            new_image_layout = ImageLayout.ZYX

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

    def merge_with(self, image: "PanSegImage"):
        if not all(
            (
                self.semantic_type == image.semantic_type,
                self.voxel_size == image.voxel_size,
                self.dimensionality == image.dimensionality,
            )
        ):
            raise ValueError("Images can't be merged, not compatible!")

        images = self.split_channels()
        images.extend(image.split_channels())

        if self.dimensionality == ImageDimensionality.TWO:
            new_image_layout = ImageLayout.CYX
        else:
            new_image_layout = ImageLayout.CZYX

        new_props = ImageProperties(
            name=self.name + "_merged",
            semantic_type=self.semantic_type,
            voxel_size=self.voxel_size,
            image_layout=new_image_layout,
            original_voxel_size=self.original_voxel_size,
            source_file_name=self.source_file_name,
        )

        data = np.stack([im.get_data() for im in images])
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
        if self.image_layout in (ImageLayout.CYX, ImageLayout.ZYX):
            if data.ndim != 3:
                raise ValueError(
                    f"Data has shape {data.shape} but should have 3 dimensions for layout {self.image_layout}"
                )

        elif self.image_layout in (ImageLayout.CZYX, ImageLayout.ZCYX):
            if data.ndim != 4:
                raise ValueError(
                    f"Data has shape {data.shape} but should have 4 dimensions for layout {self.image_layout}"
                )

        elif self.image_layout in (ImageLayout.YX,):
            if data.ndim != 2:
                raise ValueError(
                    f"Data has shape {data.shape} but should have 2 dimensions for layout {self.image_layout}"
                )

        else:
            raise ValueError(f"Image layout {self.image_layout} not recognized")

        return data

    def _check_shape(
        self, data: np.ndarray, properties: ImageProperties
    ) -> tuple[np.ndarray, ImageProperties]:
        if self.image_layout == ImageLayout.ZYX:
            if data.shape[0] == 1:
                logger.warning(
                    "Image layout is ZYX but data has only one z slice, casting to YX"
                )
                properties.image_layout = ImageLayout.YX
                return data[0], properties

        elif self.image_layout == ImageLayout.CZYX:
            if data.shape[0] == 1 and data.shape[1] == 1:
                logger.warning(
                    "Image layout is CZYX but data has only one z slice and one channel, casting to YX"
                )
                properties.image_layout = ImageLayout.YX
                return data[0, 0], properties

            elif data.shape[0] == 1 and data.shape[1] > 1:
                logger.warning(
                    "Image layout is CZYX but data has only one channel, casting to ZYX"
                )
                properties.image_layout = ImageLayout.ZYX
                return data[0], properties

            elif data.shape[0] > 1 and data.shape[1] == 1:
                logger.warning(
                    "Image layout is CZYX but data has only one z slice, casting to CYX"
                )
                properties.image_layout = ImageLayout.CYX
                return data[:, 0], properties

        elif self.image_layout == ImageLayout.ZCYX:
            logger.warning(
                "Image layout is ZCYX but should have been converted to CZYX. PanSeg is doing this now."
            )
            properties.image_layout = ImageLayout.CZYX
            data = np.moveaxis(data, 0, 1)
            return self._check_shape(data, properties)

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

        The scale is equal to the voxel size in each spatial dimension and 1 in other channels.
        """
        if self.image_layout == ImageLayout.YX:
            return (self.voxel_size.y, self.voxel_size.x)
        elif self.image_layout == ImageLayout.ZYX:
            return (self.voxel_size.z, self.voxel_size.y, self.voxel_size.x)
        elif self.image_layout == ImageLayout.CYX:
            return (1.0, self.voxel_size.y, self.voxel_size.x)
        elif self.image_layout == ImageLayout.CZYX:
            return (1.0, self.voxel_size.z, self.voxel_size.y, self.voxel_size.x)
        elif self.image_layout == ImageLayout.ZCYX:
            raise ValueError(
                f"Image layout {self.image_layout} not supported, should have been converted to CZYX"
            )
        else:
            raise ValueError(f"Image layout {self.image_layout} not recognized")

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
    def is_multichannel(self) -> bool:
        """Returns True if the image is multichannel, False otherwise."""
        return self.channel_axis is not None

    def interpolation_order(self, image_default: int = 1) -> int:
        """Returns the default interpolation order used for the image."""
        return self._properties.interpolation_order(image_default)

    def has_valid_voxel_size(self) -> bool:
        """Returns True if the voxel size is valid (not None), False otherwise."""
        return self.voxel_size.voxels_size is not None

    def has_valid_original_voxel_size(self) -> bool:
        """Returns True if the original voxel size is valid (not None), False otherwise."""
        return self.original_voxel_size.voxels_size is not None


def stack_sort(stack_layout: str, data, voxel_size):
    """Sort the stack layout, data, and voxelsize

    Makes the image stack layout unique for any number of dimensions.
    """
    sort_order = [
        ("C", 0),
        ("Z", 1),
        ("Y", 2),
        ("X", 3),
    ]

    sort_idxs = []
    sort_idxs_wo_channel = []
    # ZCXY -> [1,0,3,2]
    invert = []
    invert_next = False
    for c in stack_layout:
        if c == "-":
            invert_next = True
            continue
        if invert_next:
            invert.append(True)
            invert_next = False
        else:
            invert.append(False)
        sort_idxs.extend([n for c_sorted, n in sort_order if c == c_sorted])
        sort_idxs_wo_channel.extend(
            [n for c_sorted, n in sort_order if c == c_sorted and c != "C"]
        )
    # fill in gaps, like missing channel dimension:
    sort_idxs = np.argsort(sort_idxs)

    data = np.flip(data, axis=np.nonzero(invert)[0])
    data = np.transpose(data, axes=sort_idxs)
    stack_layout = stack_layout.replace("-", "")
    stack_layout = "".join([stack_layout[i] for i in sort_idxs])

    # ZCXY -> [1,0,3,2] -> [1,3,2] -> [0,2,1]
    # layout..sort_idxs...sort_wo_ch..sorting for voxelsize
    sort_idxs_wo_channel = np.argsort(sort_idxs_wo_channel)
    if len(sort_idxs_wo_channel) == 2:
        sort_idxs_wo_channel = np.insert(sort_idxs_wo_channel + 1, 0, 0)
    if voxel_size.voxels_size is not None:
        new_voxels = np.array(voxel_size.voxels_size)[sort_idxs_wo_channel]
        voxel_size = VoxelSize(voxels_size=tuple(new_voxels))
    return stack_layout, data, voxel_size


def import_image(
    path: Path,
    key: str | None = None,
    image_name: str = "image",
    semantic_type: str = "raw",
    stack_layout: str = "YX",
    m_slicing: str | None = None,
) -> PanSegImage | list[PanSegImage]:
    """
    Open an image file and create a PanSegImage object.

    Args:
        path (Path): Path to the image file
        key (Optional[str]): Key to load data from h5 or zarr files
        image_name (str): Name of the image (a unique name to identify the image)
        semantic_type (str): Semantic type of the image, should be raw, segmentation,
            prediction or label
        stack_layout (str): Layout of the image, should be YX, CYX, ZYX, CZYX or ZCYX
        m_slicing (str): Slicing to apply to the image, should be a string
            with the format [start:stop, ...] for each dimension.
    """
    global last_warning
    stack_layout = stack_layout.upper()
    data, voxel_size = smart_load_with_vs(path, key)
    if voxel_size is None:
        voxel_size = VoxelSize()

    if not len(stack_layout.replace("-", "")) == len(data.shape):
        raise ValueError(
            f"Data to import has shape {data.shape}, incompatible with chosen layout {stack_layout}"
        )

    original_data_shape = data.shape
    stack_layout, data, voxel_size = stack_sort(stack_layout, data, voxel_size)

    images = []
    image_layout = ImageLayout(stack_layout)

    if image_layout in [ImageLayout.ZYX, ImageLayout.YX]:
        if m_slicing is not None:
            data = dp.image_crop(data, m_slicing)

        image_properties = ImageProperties(
            name=image_name,
            semantic_type=SemanticType(semantic_type),
            voxel_size=voxel_size,
            image_layout=image_layout,
            original_voxel_size=voxel_size,
            source_file_name=path.stem,
        )

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
        )

    elif export_format == "h5":
        if key is None or key == "":
            raise ValueError("Key is required for h5 format")

        file_path_name = directory / f"{name_pattern}.h5"
        create_h5(path=file_path_name, stack=data, voxel_size=voxel_size, key=key)

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
        file_path_name = directory / f"{name_pattern}.{export_mesh}"
        create_mesh(
            path=file_path_name,
            stack=data,
            voxel_size=voxel_size,
            close_mesh=close_mesh,
        )
