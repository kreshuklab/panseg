import logging

import numpy as np

from panseg.core.image import (
    ImageDimensionality,
    ImageLayout,
    PanSegImage,
    SemanticType,
)
from panseg.functionals.dataprocessing import (
    ImagePairOperation,
    fix_over_under_segmentation_from_nuclei,
    image_gaussian_smoothing,
    image_rescale,
    process_images,
    relabel_segmentation,
    remove_false_positives_by_foreground_probability,
    set_biggest_instance_to_zero,
)
from panseg.io.voxelsize import VoxelSize
from panseg.tasks import task_tracker, timepoint_map

logger = logging.getLogger(__name__)


@task_tracker
@timepoint_map
def gaussian_smoothing_task(image: PanSegImage, sigma: float) -> PanSegImage:
    """
    Apply Gaussian smoothing to a PanSegImage object.

    Args:
        image (PanSegImage): input image
        sigma (float): standard deviation of the Gaussian kernel

    """
    if image.is_multichannel:
        raise ValueError("Gaussian smoothing is not supported for multichannel images.")

    data = image.get_data()
    smoothed_data = image_gaussian_smoothing(data, sigma=sigma)
    new_image = image.derive_new(smoothed_data, name=f"{image.name}_smoothed")
    return new_image


def _compute_slices_3d(rectangle, crop_z: tuple[int, int], shape):
    """
    Compute slices for cropping based on a given rectangle and z-slices.
    """
    z_slice = slice(*crop_z)
    if rectangle is None:
        return z_slice, slice(0, shape[1]), slice(0, shape[2])

    if len(np.unique(rectangle[:, 0])) != 1:
        raise ValueError("Invalid crop, the rextangle must be drawn in the XY plane.")

    x_start = max(rectangle[0, 1], 0)
    x_end = min(rectangle[2, 1], shape[1])
    x_slice = slice(x_start, x_end)

    y_start = max(rectangle[0, 2], 0)
    y_end = min(rectangle[2, 2], shape[2])
    y_slice = slice(y_start, y_end)
    return z_slice, x_slice, y_slice


def _compute_slices_2d(rectangle, shape):
    """
    Compute slices for cropping based on a given rectangle.
    """
    if rectangle is None:
        return slice(0, shape[0]), slice(0, shape[1])

    x_start = max(rectangle[0, 0], 0)
    x_end = min(rectangle[2, 0], shape[0])
    x_slice = slice(x_start, x_end)

    y_start = max(rectangle[0, 1], 0)
    y_end = min(rectangle[2, 1], shape[1])
    y_slice = slice(y_start, y_end)
    return x_slice, y_slice


def _cropping(data, crop_slices):
    """
    Apply cropping on the provided data based on the computed slices.
    """
    return data[crop_slices]


@task_tracker
@timepoint_map
def image_cropping_task(
    image: PanSegImage, rectangle=None, crop_z: tuple[int, int] = (0, 100)
) -> PanSegImage:
    """
    Crop the image based on the given rectangle and z-slices.

    Args:
        image (PanSegImage): The image to be cropped.
        rectangle (Optional): Rectangle defining the region to crop.
        crop_z (tuple[int, int]): Z-slice range for cropping.

    Returns:
        PanSegImage: The cropped image.
    """
    if image.is_multichannel:
        raise ValueError("Cropping is not supported for multichannel images.")

    data = image.get_data()

    # Compute crop slices
    if image.dimensionality == ImageDimensionality.TWO:
        crop_slices = _compute_slices_2d(rectangle, data.shape)
    else:
        crop_slices = _compute_slices_3d(rectangle, crop_z, data.shape)

    # Perform cropping on the data
    cropped_data = _cropping(data, crop_slices)

    # Create and return a new PanSegImage object from the cropped data
    cropped_image = image.derive_new(cropped_data, name=f"{image.name}_cropped")

    return cropped_image


@task_tracker
@timepoint_map
def set_voxel_size_task(
    image: PanSegImage, voxel_size: tuple[float, float, float]
) -> PanSegImage:
    """Set the voxel size of an image.

    Args:
        image (PanSegImage): input image
        voxel_size (tuple[float, float, float]): new voxel size
    """
    new_voxel_size = VoxelSize(voxels_size=voxel_size)
    new_image = image.derive_new(
        image._data,
        name=f"{image.name}_set_voxel_size",
        voxel_size=new_voxel_size,
        original_voxel_size=new_voxel_size,
    )
    return new_image


@task_tracker
@timepoint_map
def set_t_spacing_task(
    image: PanSegImage, t_spacing: float | None, t_unit: str = "s"
) -> PanSegImage:
    """Set the time spacing of a timeseries image.

    Note: on a timeseries input with a known spacing, t_spacing=None does
    not clear the restacked output's spacing - the restacked output takes
    the shared input spacing and the frame-by-frame loop warns about the
    fallback. It clears only when the input spacing is unknown or the
    input is a still image (which passes the loop through untouched).

    Args:
        image (PanSegImage): input image
        t_spacing (float | None): new time spacing in the given unit, or None
            to mark it unknown
        t_unit (str): unit of t_spacing (s, ms, µs/us, min or h); converted
            to the canonical unit seconds

    Returns:
        PanSegImage: new image with the new time spacing
    """
    new_image = image.derive_new(
        image._data,
        name=f"{image.name}_set_t_spacing",
        t_spacing=t_spacing,
        t_unit=t_unit,
    )
    return new_image


@task_tracker
@timepoint_map
def image_rescale_to_shape_task(
    image: PanSegImage, new_shape: tuple[int, ...], order: int = 0
) -> PanSegImage:
    """Rescale an image to a new shape.

    Args:
        image (PanSegImage): input image
        new_shape (tuple[int, ...]): new shape of the image
        order (int): order of the interpolation
    """
    if image.image_layout == ImageLayout.YX:
        scaling_factor = (new_shape[1] / image.shape[0], new_shape[2] / image.shape[1])
        spatial_scaling_factor = (1.0, scaling_factor[0], scaling_factor[1])
    elif image.image_layout == ImageLayout.ZYX:
        scaling_factor = (
            new_shape[0] / image.shape[0],
            new_shape[1] / image.shape[1],
            new_shape[2] / image.shape[2],
        )
        spatial_scaling_factor = scaling_factor
    elif image.image_layout == ImageLayout.CYX:
        scaling_factor = (
            1.0,
            new_shape[1] / image.shape[1],
            new_shape[2] / image.shape[2],
        )
        spatial_scaling_factor = (1.0, scaling_factor[1], scaling_factor[2])
    elif image.image_layout == ImageLayout.CZYX:
        scaling_factor = (
            1.0,
            new_shape[0] / image.shape[1],
            new_shape[1] / image.shape[2],
            new_shape[2] / image.shape[3],
        )
        spatial_scaling_factor = scaling_factor[1:]
    elif image.image_layout == ImageLayout.ZCYX:
        scaling_factor = (
            new_shape[0] / image.shape[0],
            1.0,
            new_shape[1] / image.shape[2],
            new_shape[2] / image.shape[3],
        )
        spatial_scaling_factor = (
            scaling_factor[0],
            scaling_factor[2],
            scaling_factor[3],
        )

    out_data = image_rescale(image.get_data(), scaling_factor, order=order)

    if image.has_valid_voxel_size():
        out_voxel_size = image.voxel_size.voxelsize_from_factor(spatial_scaling_factor)
    else:
        out_voxel_size = VoxelSize()

    new_image = image.derive_new(
        out_data, name=f"{image.name}_reshaped", voxel_size=out_voxel_size
    )
    return new_image


@task_tracker
@timepoint_map
def image_rescale_to_voxel_size_task(
    image: PanSegImage,
    new_voxels_size: tuple[float, float, float],
    new_unit: str,
    order: int = 0,
) -> PanSegImage:
    """Rescale an image to a new voxel size.

    If the voxel size is not defined in the input image, use the set voxel size task to set the voxel size.

    Args:
        image (PanSegImage): input image
        new_voxel_size (VoxelSize): new voxel size
        new_unit: (str): unit, e.g. `um`
        order (int): order of the interpolation

    """
    new_voxel_size = VoxelSize(voxels_size=new_voxels_size, unit=new_unit)
    spatial_scaling_factor = image.voxel_size.scalefactor_from_voxelsize(new_voxel_size)

    if image.image_layout == ImageLayout.YX:
        scaling_factor = (spatial_scaling_factor[1], spatial_scaling_factor[2])
    elif image.image_layout == ImageLayout.CYX:
        scaling_factor = (1.0, spatial_scaling_factor[1], spatial_scaling_factor[2])
    elif image.image_layout == ImageLayout.ZYX:
        scaling_factor = spatial_scaling_factor
    elif image.image_layout == ImageLayout.CZYX:
        scaling_factor = (1.0, *spatial_scaling_factor)
    elif image.image_layout == ImageLayout.ZCYX:
        scaling_factor = (spatial_scaling_factor[0], 1.0, *spatial_scaling_factor[1:])
    else:
        raise ValueError(f"Unknown image layout {image.image_layout}")

    out_data = image_rescale(image.get_data(), scaling_factor, order=order)
    new_image = image.derive_new(
        out_data, name=f"{image.name}_rescaled", voxel_size=new_voxel_size
    )
    return new_image


@task_tracker
@timepoint_map(broadcast=True)
def remove_false_positives_by_foreground_probability_task(
    segmentation: PanSegImage, foreground: PanSegImage, threshold: float
) -> list[PanSegImage]:
    """Remove false positives from a segmentation based on the foreground probability.

    On timeseries input the task runs per timepoint: label IDs are independent
    per timepoint, with no correspondence across timepoints. The foreground
    probability may be a still image, applied to every timepoint.

    Args:
        segmentation (PanSegImage): input segmentation
        foreground (PanSegImage): input foreground probability
        threshold (float): threshold value

    """
    if segmentation.shape != foreground.shape:
        raise ValueError(
            "Segmentation and foreground probability must have the same shape."
        )

    kept, removed = remove_false_positives_by_foreground_probability(
        segmentation.get_data(), foreground.get_data(), threshold
    )
    image_kept = segmentation.derive_new(
        kept,
        name=f"{segmentation.name}_fg_filtered",
    )
    image_removed = segmentation.derive_new(
        removed,
        name=f"{segmentation.name}_false_positives",
    )
    return [image_kept, image_removed]


@task_tracker
@timepoint_map
def fix_over_under_segmentation_from_nuclei_task(
    cell_seg: PanSegImage,
    nuclei_seg: PanSegImage,
    threshold_merge: float,
    threshold_split: float,
    quantile_min: float,
    quantile_max: float,
    boundary: PanSegImage | None = None,
) -> PanSegImage:
    """
    Task to fix over- and under-segmentation of cells based on nuclear segmentation.

    Args:
        cell_seg (PanSegImage): Input cell segmentation as a PanSegImage object.
        nuclei_seg (PanSegImage): Input nuclear segmentation as a PanSegImage object.
        threshold_merge (float): Threshold for merging cells, as a fraction (0-1).
        threshold_split (float): Threshold for splitting cells, as a fraction (0-1).
        quantile_min (float): Minimum quantile for filtering nuclei sizes, as a fraction (0-1).
        quantile_max (float): Maximum quantile for filtering nuclei sizes, as a fraction (0-1).
        boundary (PanSegImage | None, optional): Optional boundary probability map for segmentation refinement.

    Returns:
        PanSegImage: Corrected cell segmentation as a PanSegImage object.
    """
    corrected_data = fix_over_under_segmentation_from_nuclei(
        cell_seg.get_data(),
        nuclei_seg.get_data(),
        threshold_merge=threshold_merge,
        threshold_split=threshold_split,
        quantile_min=quantile_min,
        quantile_max=quantile_max,
        boundary=boundary.get_data() if boundary else None,
    )
    return cell_seg.derive_new(corrected_data, name=f"{cell_seg.name}_nuc_fixed")


@task_tracker
@timepoint_map
def set_biggest_instance_to_zero_task(
    image: PanSegImage, instance_could_be_zero: bool = False
) -> PanSegImage:
    """
    Task to set the largest segment in a segmentation image to zero.

    Args:
        image (PanSegImage): Segmentation image to process.
        instance_could_be_zero (bool): If True, 0 might be an instance label,
        add 1 to all labels before processing.

    Returns:
        PanSegImage: New segmentation image with largest instance set to 0.
    """
    if not image.semantic_type == SemanticType.SEGMENTATION:
        raise ValueError("Input image must be a segmentation or mask image.")
    data = image.get_data()
    logger.info(
        f"Processing {image.name} with shape {data.shape} and max {data.max()}, min {data.min()}."
    )
    new_data = set_biggest_instance_to_zero(
        data, instance_could_be_zero=instance_could_be_zero
    )
    new_image = image.derive_new(new_data, name=f"{image.name}_bg0")
    return new_image


@task_tracker
@timepoint_map
def relabel_segmentation_task(
    image: PanSegImage, background: int | None = None
) -> PanSegImage:
    """
    Task to relabel a segmentation image contiguously, ensuring non-touching
    segments with the same ID are relabeled.

    Args:
        image (PanSegImage): Segmentation image to process.

    Returns:
        PanSegImage: New segmentation image with relabeled instances.
    """
    if not image.semantic_type == SemanticType.SEGMENTATION:
        raise ValueError("Input image must be a segmentation or mask image.")
    data = image.get_data()
    new_data = relabel_segmentation(data, background=background)
    new_image = image.derive_new(new_data, name=f"{image.name}_relabeled")
    return new_image


@task_tracker
@timepoint_map(broadcast=True)
def image_pair_operation_task(
    image1: PanSegImage,
    image2: PanSegImage,
    operation: ImagePairOperation,
    normalize_input: bool = False,
    clip_output: bool = False,
    normalize_output: bool = False,
) -> PanSegImage:
    """
    Task to perform an operation on two images.

    Supports timepoint broadcasting for mixing time series and still images.

    Args:
        image1 (PanSegImage): First image to process.
        Image2 (PanSegImage): Second image to process.
        operation (str): Operation to perform on the images.
        normalize_input (bool): Normalize input images before processing.
        clip_output (bool): Clip output values to the range [0, 1].
        normalize_output (bool): Normalize output values to the range [0, 1].

    Returns:
        PanSegImage: New image resulting from the operation.
    """
    result = process_images(
        image1.get_data(),
        image2.get_data(),
        operation=operation,
        normalize_input=normalize_input,
        clip_output=clip_output,
        normalize_output=normalize_output,
    )
    new_image = image1.derive_new(
        result, name=f"{image1.name}_{operation}_{image2.name}"
    )
    return new_image
