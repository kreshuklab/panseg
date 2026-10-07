import logging
import warnings
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import numpy as np
import tifffile
from pydantic import ValidationError

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


def _first_ome_pixels(tiff) -> Optional[ElementTree.Element]:
    """
    Returns the Pixels element of the first OME Image, or None when the file
    is not an OME-TIFF or the OME-XML carries no Image or Pixels element.
    """
    if tiff.ome_metadata is None:
        return None
    tree = ElementTree.fromstring(tiff.ome_metadata)

    image_element = [image for image in tree if image.tag.find("Image") != -1]
    if not image_element:
        return None

    pixels_element = [
        pixels for pixels in image_element[0] if pixels.tag.find("Pixels") != -1
    ]
    if not pixels_element:
        return None
    return pixels_element[0]


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
            warnings.warn(f"Units are not homogeneous: {units}")

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
        try:
            if tiff.imagej_metadata is not None:
                return _read_imagej_meta(tiff)

            elif tiff.ome_metadata is not None:
                return _read_ome_meta(tiff)
        except ValidationError:
            warnings.warn(
                "Error parsing tiff meta unit. Reverting to default "
                "voxel size (1., 1., 1.) um"
            )
            return VoxelSize()

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


def read_ome_axes(path: Path) -> Optional[str]:
    """
    Return the axis string of the first OME-TIFF series, e.g. "TZYX".

    Singleton axes are already dropped by the reader, so a single-timepoint
    file reads as "YX" and a single-channel file without "C". Returns None
    when the file is not an OME-TIFF.

    Args:
        path (Path): path to the tiff file

    Returns:
        str | None: axis string of the first OME series, or None
    """
    with tifffile.TiffFile(path) as tiff:
        if tiff.ome_metadata is None or not tiff.series:
            return None
        return tiff.series[0].axes


def read_ome_time_spacing(path: Path) -> tuple[Optional[float], str]:
    """
    Return the time spacing of an OME-TIFF file as (value, unit).

    Pixels.TimeIncrement is used when present. Otherwise the spacing is
    recovered from Plane.DeltaT, which holds the ABSOLUTE time of a plane
    since the start of the acquisition rather than an increment between
    timepoints: the DeltaT of the first documented plane of each timepoint
    (document order is assumed to equal plane raster order) is collected
    and the spacing is the uniform consecutive difference of those values.
    A single timepoint, a timepoint documented without DeltaT, or
    non-uniform or non-positive differences warn and are treated as
    missing. The value is expressed in the file's unit; conversion to the
    canonical unit seconds happens at ImageProperties construction.
    Returns (None, "s") for non-OME files and for files without timing
    metadata.

    Args:
        path (Path): path to the tiff file

    Returns:
        tuple: (time spacing value or None, unit string)
    """
    with tifffile.TiffFile(path) as tiff:
        pixels = _first_ome_pixels(tiff)
    if pixels is None:
        return None, "s"

    time_increment = pixels.get("TimeIncrement")
    if time_increment is not None:
        return float(time_increment), pixels.get("TimeIncrementUnit", "s")

    # first plane (document order = plane raster order) of each timepoint
    first_planes = {}
    for plane in pixels:
        if plane.tag.find("Plane") == -1:
            continue
        if plane.get("TheT", "0") not in first_planes:
            first_planes[plane.get("TheT", "0")] = plane

    if not first_planes:
        return None, "s"

    unit = next(iter(first_planes.values())).get("DeltaTUnit", "s")

    # DeltaT is an absolute time since acquisition start, not an increment:
    # the spacing between timepoints is recovered by differencing the
    # first-plane values instead of reading them directly
    values, missing = [], []
    for the_t, plane in first_planes.items():
        delta_t = plane.get("DeltaT")
        if delta_t is None:
            missing.append(the_t)
        else:
            values.append(float(delta_t))

    if missing:
        warnings.warn(
            f"Plane.DeltaT missing for timepoint(s) {missing} "
            "(planes documented without DeltaT), "
            "treating the time spacing as missing"
        )
        return None, "s"

    if len(values) < 2:
        warnings.warn(
            f"Plane.DeltaT present for a single timepoint only ({values[0]}), "
            "a time spacing cannot be derived from it, "
            "treating the time spacing as missing"
        )
        return None, "s"

    diffs = np.diff(values)
    if np.allclose(diffs, diffs[0]) and diffs[0] > 0:
        return float(diffs[0]), unit

    warnings.warn(
        f"Non-uniform Plane.DeltaT across timepoints {values} "
        f"(first-plane differences {diffs.tolist()}), "
        "treating the time spacing as missing"
    )
    return None, "s"


def check_ome_single_file(path: Path) -> None:
    """
    Raise ValueError when the OME-TIFF describes a multi-file series.

    A series is multi-file when the TiffData of the first Image carry more
    than one distinct (UUID, FileName) pair, or a FileName that does not
    match the opened file. The FileName attribute of the UUID child element
    is optional: when absent it defaults to the opened file. Non-OME files
    are ignored.

    Args:
        path (Path): path to the tiff file

    Raises:
        ValueError: if the file is part of a multi-file OME-TIFF series
    """
    with tifffile.TiffFile(path) as tiff:
        pixels = _first_ome_pixels(tiff)
    if pixels is None:
        return

    pairs = set()
    for tiff_data in pixels:
        if tiff_data.tag.find("TiffData") == -1:
            continue
        for uuid in tiff_data:
            if uuid.tag.find("UUID") == -1:
                continue
            name = uuid.get("FileName") or path.name
            pairs.add((uuid.text, name))

    foreign = sorted({name for _, name in pairs if name.lower() != path.name.lower()})
    if len(pairs) > 1 or foreign:
        raise ValueError(
            f"Multi-file OME-TIFF (UUID/FileName chain) is not supported, "
            f"{path.name} references other file(s): {foreign}"
        )


def create_tiff(
    path: Path,
    stack: np.ndarray,
    voxel_size: VoxelSize,
    layout: str = "ZYX",
    t_spacing: Optional[float] = None,
    force_bigtiff=False,
) -> None:
    """
    Create a tiff file from a numpy array

    A time-bearing layout is always written as an OME-TIFF: the ImageJ
    format cannot represent the time axis. BigTIFF is still only used when
    the data exceeds 4 GiB or ``force_bigtiff`` is set.

    Args:
        path (Path): path to save the tiff file
        stack (np.ndarray): numpy array to save as tiff
        voxel_size (list or tuple): tuple of the voxel size
        voxel_size_unit (str): units of the voxel size
        t_spacing (float | None): time spacing between timepoints in seconds,
            written as OME-XML TimeIncrement when the layout carries a time
            axis. None (unknown) exports SizeT without time metadata.
        force_bigtiff (bool): forces the use of bigtiff. Used for testing.

    """
    # taken from: https://pypi.org/project/tifffile docs
    # dimensions in TZCYXS order
    if layout == "ZYX":
        assert stack.ndim == 3, "Stack dimensions must be in ZYX order"
        z, y, x = stack.shape
        logical_stack, logical_axes = stack, "ZYX"
        stack = stack.reshape(1, z, 1, y, x, 1)

    elif layout == "YX":
        assert stack.ndim == 2, "Stack dimensions must be in YX order"
        y, x = stack.shape
        logical_stack, logical_axes = stack, "YX"
        stack = stack.reshape(1, 1, 1, y, x, 1)

    elif layout == "CYX":
        assert stack.ndim == 3, "Stack dimensions must be in CYX order"
        c, y, x = stack.shape
        logical_stack, logical_axes = stack, "CYX"
        stack = stack.reshape(1, 1, c, y, x, 1)

    elif layout == "ZCYX":
        assert stack.ndim == 4, "Stack dimensions must be in ZCYX order"
        z, c, y, x = stack.shape
        logical_stack, logical_axes = stack, "ZCYX"
        stack = stack.reshape(1, z, c, y, x, 1)

    elif layout == "CZYX":
        assert stack.ndim == 4, "Stack dimensions must be in CZYX order"
        stack = np.transpose(stack, (1, 0, 2, 3))
        z, c, y, x = stack.shape
        logical_stack, logical_axes = stack, "ZCYX"
        stack = stack.reshape(1, z, c, y, x, 1)

    elif layout == "TYX":
        assert stack.ndim == 3, "Stack dimensions must be in TYX order"
        logical_stack, logical_axes = stack, "TYX"

    elif layout == "TCYX":
        assert stack.ndim == 4, "Stack dimensions must be in TCYX order"
        logical_stack, logical_axes = stack, "TCYX"

    elif layout == "TZYX":
        assert stack.ndim == 4, "Stack dimensions must be in TZYX order"
        logical_stack, logical_axes = stack, "TZYX"

    elif layout == "TCZYX":
        assert stack.ndim == 5, "Stack dimensions must be in TCZYX order"
        logical_stack, logical_axes = stack, "TCZYX"

    else:
        raise ValueError(f"Layout {layout} not supported")

    is_timeseries = "T" in layout

    if voxel_size.voxels_size is not None:
        assert len(voxel_size.voxels_size) == 3, (
            "Voxel size must have 3 elements (z, y, x)"
        )
        spacing, y, x = voxel_size.voxels_size
    else:
        spacing, y, x = (1.0, 1.0, 1.0)

    resolution = (1.0 / x, 1.0 / y)
    # Save output results as tiff

    use_bigtiff = stack.nbytes > 4294967295 or force_bigtiff
    if is_timeseries or use_bigtiff:
        # OME-XML (unlike the shaped-JSON format) does not read `spacing`/`unit`
        # metadata keys and rejects the 6-D TZCYXS reshape, so write the logical
        # stack with explicit PhysicalSize* attributes to keep the voxel size.
        ome_metadata = {
            "axes": logical_axes,
            "PhysicalSizeX": x,
            "PhysicalSizeXUnit": voxel_size.unit,
            "PhysicalSizeY": y,
            "PhysicalSizeYUnit": voxel_size.unit,
            "PhysicalSizeZ": spacing,
            "PhysicalSizeZUnit": voxel_size.unit,
        }
        if is_timeseries and t_spacing is not None:
            # the time spacing is canonical seconds
            ome_metadata["TimeIncrement"] = t_spacing
            ome_metadata["TimeIncrementUnit"] = "s"
        if len(suffs := path.suffixes[-2:]) in [1, 2] and suffs[0] != ".ome":
            path = path.with_name(path.stem + ".ome" + path.suffix)
        tifffile.imwrite(
            path,
            data=logical_stack,
            dtype=logical_stack.dtype,
            bigtiff=use_bigtiff,
            ome=True,
            photometric="minisblack",
            resolution=resolution,
            metadata=ome_metadata,
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
