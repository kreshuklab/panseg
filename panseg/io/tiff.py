import logging
import warnings
from pathlib import Path
from typing import Optional
from xml.etree import cElementTree as ElementTree

import numpy as np
import tifffile

from panseg.io.voxelsize import VoxelSize

logger = logging.getLogger(__name__)

TIFF_EXTENSIONS = [".tiff", ".tif"]


def _read_imagej_meta(tiff) -> VoxelSize:
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    Returns the voxel size and the voxel units
    """

    def _xy_voxel_size(tags, key):
        assert key in ["XResolution", "YResolution"]
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return None

    image_metadata = tiff.imagej_metadata
    z = image_metadata.get("spacing", 1.0)
    voxel_size_unit = image_metadata.get("unit", "um")

    tags = tiff.pages[0].tags
    # parse X, Y resolution
    y = _xy_voxel_size(tags, "YResolution")
    x = _xy_voxel_size(tags, "XResolution")
    # return voxel size

    if x is None or y is None:
        logger.warning("Error parsing imagej tiff meta.")
        return VoxelSize()

    return VoxelSize(voxels_size=(z, y, x), unit=voxel_size_unit)


def _read_ome_meta(tiff) -> VoxelSize:
    """
    Returns the voxels size and the voxel units
    """
    xml_om = tiff.ome_metadata
    tree = ElementTree.fromstring(xml_om)

    image_element = [image for image in tree if image.tag.find("Image") != -1]
    if image_element:
        image_element = image_element[0]
    else:
        warnings.warn(
            "Error parsing omero tiff meta Image. Reverting to default voxel size (1., 1., 1.) um"
        )
        return VoxelSize()

    pixels_element = [
        pixels for pixels in image_element if pixels.tag.find("Pixels") != -1
    ]
    if pixels_element:
        pixels_element = pixels_element[0]
    else:
        warnings.warn(
            "Error parsing omero tiff meta Pixels. Reverting to default voxel size (1., 1., 1.) um"
        )
        return VoxelSize()

    units = []
    x, y, z, voxel_size_unit = None, None, None, "um"

    for key, value in pixels_element.items():
        if key == "PhysicalSizeX":
            x = float(value)

        elif key == "PhysicalSizeY":
            y = float(value)

        elif key == "PhysicalSizeZ":
            z = float(value)

        if key in ["PhysicalSizeXUnit", "PhysicalSizeYUnit", "PhysicalSizeZUnit"]:
            units.append(value)

    if units:
        voxel_size_unit = units[0]
        if not all(unit == voxel_size_unit for unit in units):
            warnings.warn("Units are not homogeneous: {units}")

    if x is None or y is None or z is None:
        warnings.warn("Error parsing omero tiff meta. ")
        return VoxelSize()

    return VoxelSize(voxels_size=(z, y, x), unit=voxel_size_unit)


def read_tiff_voxel_size(file_path: Path) -> VoxelSize:
    """
    Returns the voxels size and the voxel units for imagej and ome style tiff (if absent returns [1, 1, 1], um)

    Args:
        file_path (Path): path to the tiff file

    Returns:
        VoxelSize: voxel size and unit

    """
    with tifffile.TiffFile(file_path) as tiff:
        if tiff.imagej_metadata is not None:
            return _read_imagej_meta(tiff)

        elif tiff.ome_metadata is not None:
            return _read_ome_meta(tiff)

        warnings.warn("No metadata found.")
        return VoxelSize()


def read_tiff_shape(path: Path) -> Optional[tuple[int, ...]]:
    """Read the shape of a tiff file.

    Dimensions of lenght one will be ignored, as they will be
    squeezed during import.

    Args:
        path: The path to the tifffile

    Returns:
        tuple: the shape of the tifffile
    """
    shape = None
    with tifffile.TiffFile(path) as tiff:
        meta = tiff.shaped_metadata
        if meta is not None:
            shape = meta[0].get("shape")
        if shape is None:
            shape = tiff.asarray().shape
    return tuple([i for i in shape if i != 1])


def load_tiff(path: Path) -> np.ndarray:
    """
    Load a dataset from a tiff file and returns some meta info about it.
    Args:
        path (str): path to the tiff files to load

    Returns:
        np.ndarray: loaded data as numpy array
    """
    return tifffile.imread(path).squeeze()


def create_tiff(
    path: Path,
    stack: np.ndarray,
    voxel_size: VoxelSize,
    layout: str = "ZYX",
    force_bigtiff=False,
) -> None:
    """
    Create a tiff file from a numpy array

    Args:
        path (Path): path to save the tiff file
        stack (np.ndarray): numpy array to save as tiff
        voxel_size (list or tuple): tuple of the voxel size
        voxel_size_unit (str): units of the voxel size
        force_bigtiff (bool): forces the use of bigtiff. Used for testing.

    """
    # taken from: https://pypi.org/project/tifffile docs
    # dimensions in TZCYXS order
    if layout == "ZYX":
        assert stack.ndim == 3, "Stack dimensions must be in ZYX order"
        z, y, x = stack.shape
        stack = stack.reshape(1, z, 1, y, x, 1)

    elif layout == "YX":
        assert stack.ndim == 2, "Stack dimensions must be in YX order"
        y, x = stack.shape
        stack = stack.reshape(1, 1, 1, y, x, 1)

    elif layout == "CYX":
        assert stack.ndim == 3, "Stack dimensions must be in CYX order"
        c, y, x = stack.shape
        stack = stack.reshape(1, 1, c, y, x, 1)

    elif layout == "ZCYX":
        assert stack.ndim == 4, "Stack dimensions must be in ZCYX order"
        z, c, y, x = stack.shape
        stack = stack.reshape(1, z, c, y, x, 1)

    elif layout == "CZYX":
        assert stack.ndim == 4, "Stack dimensions must be in CZYX order"
        c, z, y, x = stack.shape
        stack = stack.reshape(1, z, c, y, x, 1)

    else:
        raise ValueError(f"Layout {layout} not supported")

    if voxel_size.voxels_size is not None:
        assert len(voxel_size.voxels_size) == 3, (
            "Voxel size must have 3 elements (z, y, x)"
        )
        spacing, y, x = voxel_size.voxels_size
    else:
        spacing, y, x = (1.0, 1.0, 1.0)

    resolution = (1.0 / x, 1.0 / y)
    # Save output results as tiff

    if stack.nbytes > 4294967295 or force_bigtiff:
        tifffile.imwrite(
            path,
            data=stack,
            dtype=stack.dtype,
            imagej=False,
            bigtiff=True,
            resolution=resolution,
            metadata={"axes": "TZCYXS", "spacing": spacing, "unit": voxel_size.unit},
        )
    else:
        tifffile.imwrite(
            path,
            data=stack,
            dtype=stack.dtype,
            imagej=True,
            resolution=resolution,
            metadata={"axes": "TZCYXS", "spacing": spacing, "unit": voxel_size.unit},
            compression="zlib",
        )
