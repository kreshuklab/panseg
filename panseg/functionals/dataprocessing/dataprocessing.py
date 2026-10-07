from typing import Literal

import numpy as np
from scipy.ndimage import zoom
from skimage.filters import median  # pylint: disable=no-name-in-module
from skimage.morphology import ball, disk
from vigra import gaussianSmoothing


def compute_scaling_factor(
    input_voxel_size: tuple[float, float, float],
    output_voxel_size: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Compute the scaling factor to rescale an image from input voxel size to output voxel size.
    """
    scaling = tuple(
        i_size / o_size for i_size, o_size in zip(input_voxel_size, output_voxel_size)
    )
    if len(scaling) != 3:
        raise ValueError(
            f"Expected scaling factor to be 3D, but got {len(scaling)}D input"
        )
    return scaling


def compute_scaling_voxelsize(
    input_voxel_size: tuple[float, float, float],
    scaling_factor: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Compute the output voxel size after scaling an image with a given scaling factor.
    """
    output_voxel_size = tuple(
        i_size / s_size for i_size, s_size in zip(input_voxel_size, scaling_factor)
    )
    if len(output_voxel_size) != 3:
        raise ValueError(
            f"Expected output voxel size to be 3D, but got {len(output_voxel_size)}D input"
        )
    return output_voxel_size


def scale_image_to_voxelsize(
    image: np.ndarray,
    input_voxel_size: tuple[float, float, float],
    output_voxel_size: tuple[float, float, float],
    order: int = 0,
) -> np.ndarray:
    """
    Scale an image from a given voxel size to another voxel size.

    Args:
        image (np.ndarray): Input image to scale
        input_voxel_size (tuple[float, float, float]): Input voxel size
        output_voxel_size (tuple[float, float, float]): Output voxel size
        order (int): Interpolation order, must be 0 for segmentation and 1, 2 for images

    Returns:
        scaled_image (np.ndarray): Scaled image as numpy array
    """
    factor = compute_scaling_factor(input_voxel_size, output_voxel_size)
    return image_rescale(image, factor, order=order)


def image_rescale(image: np.ndarray, factor: tuple, order: int) -> np.ndarray:
    """
    Scale an image by a given factor in each dimension.
    factor should have dimensionality of image.
    If image is 2d factor can be 3d and padded with a 1.0 in first place.

    Args:
        image (np.ndarray): Input image to scale
        factor (tuple): Scaling factor in each dimension
        order (int): Interpolation order, must be 0 for segmentation and 1, 2 for images

    Returns:
        scaled_image (np.ndarray): Scaled image as numpy array
    """
    if all([f == 1.0 for f in factor]):
        return image
    if len(factor) != image.ndim:
        if image.ndim == 2 and len(factor) == 3 and factor[0] == 1.0:
            factor = factor[1:]
        else:
            raise ValueError(
                f"Image and factor do not align: img shape {image.shape}, factor len {len(factor)}"
            )
    return zoom(image, zoom=factor, order=order)


def image_median(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Apply median smoothing on an image with a given radius.

    Args:
        image (np.ndarray): Input image to apply median smoothing.
        radius (int): Radius of the median filter.

    Returns:
        np.ndarray: Median smoothed image.
    """
    if radius <= 0:
        raise ValueError("Radius must be a positive integer.")

    if image.ndim == 2:
        # 2D image
        return median(image, disk(radius))
    elif image.ndim == 3:
        if image.shape[0] == 1:
            # Single slice (ZYX or YX) case
            return median(image[0], disk(radius)).reshape(image.shape)
        else:
            # 3D image
            return median(image, ball(radius))
    else:
        raise ValueError(
            "Unsupported image dimensionality. Image must be either 2D or 3D."
        )


def image_gaussian_smoothing(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply gaussian smoothing on an image with a given sigma.

    Args:
        image (np.ndarray): Input image to apply gaussian smoothing
        sigma (float): Sigma value for gaussian smoothing

    Returns:
        smoothed_image (np.ndarray): Gaussian smoothed image as numpy array
    """
    image = image.astype("float32")
    max_sigma = (np.array(image.shape) - 1) / 3
    sigma_array = np.minimum(max_sigma, np.ones(max_sigma.ndim) * sigma)
    return gaussianSmoothing(image, sigma_array)


ImageLayout = Literal["ZYX", "YX", "CZYX", "CYX"]


def fix_layout_to_YX(data: np.ndarray, input_layout: ImageLayout) -> np.ndarray:
    """
    Fix the layout of the input data from any supported layout to YX layout.

    Args:
        data (np.ndarray): Input data
        input_layout (ImageLayout): Input layout of the data

    Returns:
        np.ndarray: Data in YX layout
    """
    if input_layout == "ZYX":
        if data.shape[0] != 1:
            raise ValueError("Cannot convert multi-channel image ZYX to YX layout")
        _data = data[0]

    elif input_layout == "YX":
        _data = data

    elif input_layout == "CZYX":
        if data.shape[0] != 1 and data.shape[1] != 1:
            raise ValueError("Cannot convert multi-channel image CZYX to YX layout")
        _data = data[0, 0]

    elif input_layout == "CYX":
        if data.shape[0] != 1:
            raise ValueError("Cannot convert multi-channel image CYX to YX layout")
        _data = data[0]
    else:
        raise ValueError(f"Unsupported input layout {input_layout}")

    if _data.ndim != 2:
        raise ValueError(f"Expected 2D image, but got {_data.ndim}D image")

    return _data


def fix_layout_to_ZYX(data: np.ndarray, input_layout: ImageLayout) -> np.ndarray:
    """
    Fix the layout of the input data from any supported layout to ZYX layout.

    Args:
        data (np.ndarray): Input data
        input_layout (ImageLayout): Input layout of the data

    Returns:
        np.ndarray: Data in ZYX layout
    """
    if input_layout == "ZYX":
        _data = data

    elif input_layout == "YX":
        _data = data[None, ...]

    elif input_layout == "CZYX":
        if data.shape[0] != 1:
            raise ValueError("Cannot convert multi-channel image CZYX to ZYX layout")
        _data = data[0]

    elif input_layout == "CYX":
        _data = data

    else:
        raise ValueError(f"Unsupported input layout {input_layout}")

    if _data.ndim != 3:
        raise ValueError(f"Expected 3D image, but got {_data.ndim}D image")

    return _data


def fix_layout_to_CZYX(data: np.ndarray, input_layout: ImageLayout) -> np.ndarray:
    """
    Fix the layout of the input data from any supported layout to CZYX layout.

    Args:
        data (np.ndarray): Input data
        input_layout (ImageLayout): Input layout of the data
    """
    if input_layout == "ZYX":
        _data = data[None, ...]

    elif input_layout == "YX":
        _data = data[None, None, ...]

    elif input_layout == "CZYX":
        _data = data

    elif input_layout == "CYX":
        _data = data[:, None, ...]

    else:
        raise ValueError(f"Unsupported input layout {input_layout}")

    if _data.ndim != 4:
        raise ValueError(f"Expected 4D image, but got {_data.ndim}D image")

    return _data


def fix_layout_to_CYX(data: np.ndarray, input_layout: ImageLayout) -> np.ndarray:
    """
    Fix the layout of the input data from any supported layout to CYX layout.

    Args:
        data (np.ndarray): Input data
        input_layout (ImageLayout): Input layout of the data

    Returns:
        np.ndarray: Data in CYX layout
    """
    if input_layout == "ZYX":
        if data.shape[0] != 1:
            raise ValueError("Cannot convert multi-channel image ZYX to CYX layout")
        _data = data

    elif input_layout == "YX":
        if data.shape[0] != 1:
            raise ValueError("Cannot convert multi-channel image YX to CYX layout")
        _data = data[None, ...]

    elif input_layout == "CZYX":
        if data.shape[1] != 1:
            raise ValueError("Cannot convert multi-channel image CZYX to CYX layout")
        _data = data[:, 0]

    elif input_layout == "CYX":
        _data = data

    else:
        raise ValueError(f"Unsupported input layout {input_layout}")

    if _data.ndim != 3:
        raise ValueError(f"Expected 3D image, but got {_data.ndim}D image")

    return _data


def fix_layout(
    data: np.ndarray, input_layout: ImageLayout, output_layout: ImageLayout
) -> np.ndarray:
    """
    Fix the layout of the input data from any supported layout to the desired output layout.

    Args:
        data (np.ndarray): Input data
        input_layout (ImageLayout): Input layout of the data
        output_layout (ImageLayout): Desired output layout of the data

    Returns:
        np.ndarray: Data in the desired output layout
    """
    if output_layout == "ZYX":
        return fix_layout_to_ZYX(data, input_layout)

    elif output_layout == "YX":
        return fix_layout_to_YX(data, input_layout)

    elif output_layout == "CZYX":
        return fix_layout_to_CZYX(data, input_layout)

    elif output_layout == "CYX":
        return fix_layout_to_CYX(data, input_layout)

    else:
        raise ValueError(f"Unsupported output layout {output_layout}")


def normalize_01(data: np.ndarray, eps=1e-12) -> np.ndarray:
    """
    Normalize a numpy array between 0 and 1 and converts it to float32.

    Args:
        data (np.ndarray): Input numpy array
        eps (float): A small value added to the denominator for numerical stability

    Returns:
        normalized_data (np.ndarray): Normalized numpy array
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data) + eps).astype("float32")


def select_channel(data: np.ndarray, channel: int, channel_axis: int = 0) -> np.ndarray:
    """
    Select a channel from a numpy array with shape (C x Z x Y x X) or (C x Y x X) or (Z x C x Y x X)

    Args:
        data (np.ndarray): Input numpy array
        channel (int): Channel to select
        channel_axis (int): Channel axis

    Returns:
        selected_channel (np.ndarray): Selected channel as numpy array
    """
    return np.take(data, channel, axis=channel_axis)


def normalize_01_channel_wise(
    data: np.ndarray, channel_axis: int = 0, eps=1e-12
) -> np.ndarray:
    """
    Normalize each channel of a numpy array between 0 and 1 and converts it to float32.

    Args:
        data (np.ndarray): Input numpy array
        channel_axis (int): Channel axis
        eps (float): A small value added to the denominator for numerical stability

    Returns:
        np.ndarray: Normalized numpy array
    """
    # Move the channel axis to the first axis
    data = np.moveaxis(data, channel_axis, 0)

    # Normalize each channel independently
    normalized_channels = np.array([normalize_01(channel, eps=eps) for channel in data])

    # Move the axis back to its original position
    return np.moveaxis(normalized_channels, 0, channel_axis)


ImagePairOperation = Literal["add", "multiply", "subtract", "divide", "max"]


def process_images(
    image1: np.ndarray,
    image2: np.ndarray,
    operation: ImagePairOperation,
    normalize_input: bool = False,
    clip_output: bool = False,
    normalize_output: bool = True,
) -> np.ndarray:
    """
    General function for performing image operations with optional preprocessing and post-processing.

    Args:
        image1 (np.ndarray): First input image.
        image2 (np.ndarray): Second input image.
        operation (str): Operation to perform ('add', 'multiply', 'subtract', 'divide', 'max').
        normalize_input (bool): Whether to normalize the input images to the range [0, 1]. Default is False.
        clip_output (bool): Whether to clip the resulting image values to the range [0, 1]. Default is False.
        normalize_output (bool): Whether to normalize the output image to the range [0, 1]. Default is True.

    Returns:
        np.ndarray: The resulting image after performing the operation.
    """
    # Preprocessing: Normalize input images if specified
    if normalize_input:
        image1, image2 = normalize_01(image1), normalize_01(image2)

    # Perform the specified operation
    if operation == "add":
        result = image1 + image2
    elif operation == "multiply":
        result = image1 * image2
    elif operation == "subtract":
        result = image1 - image2
    elif operation == "divide":
        result = image1 / image2
    elif operation == "max":
        result = np.maximum(image1, image2)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    # Post-processing: Clip and/or normalize output if specified
    if clip_output:
        result = np.clip(result, 0, 1)
    if normalize_output:
        result = normalize_01(result)

    return result


def add_images(
    image1: np.ndarray,
    image2: np.ndarray,
    clip_output: bool = False,
    normalize_output: bool = True,
    normalize_input: bool = False,
) -> np.ndarray:
    """
    Adds two images with optional preprocessing and post-processing.
    """
    return process_images(
        image1,
        image2,
        operation="add",
        clip_output=clip_output,
        normalize_output=normalize_output,
        normalize_input=normalize_input,
    )


def multiply_images(
    image1: np.ndarray,
    image2: np.ndarray,
    clip_output: bool = False,
    normalize_output: bool = True,
    normalize_input: bool = False,
) -> np.ndarray:
    """
    Multiplies two images with optional preprocessing and post-processing.
    """
    return process_images(
        image1,
        image2,
        operation="multiply",
        clip_output=clip_output,
        normalize_output=normalize_output,
        normalize_input=normalize_input,
    )


def subtract_images(
    image1: np.ndarray,
    image2: np.ndarray,
    clip_output: bool = False,
    normalize_output: bool = True,
    normalize_input: bool = False,
) -> np.ndarray:
    """
    Subtracts the second image from the first with optional preprocessing and post-processing.
    """
    return process_images(
        image1,
        image2,
        operation="subtract",
        clip_output=clip_output,
        normalize_output=normalize_output,
        normalize_input=normalize_input,
    )


def divide_images(
    image1: np.ndarray,
    image2: np.ndarray,
    clip_output: bool = False,
    normalize_output: bool = True,
    normalize_input: bool = False,
) -> np.ndarray:
    """
    Divides the first image by the second with optional preprocessing and post-processing.
    """
    return process_images(
        image1,
        image2,
        operation="divide",
        clip_output=clip_output,
        normalize_output=normalize_output,
        normalize_input=normalize_input,
    )


def max_images(
    image1: np.ndarray,
    image2: np.ndarray,
    clip_output: bool = False,
    normalize_output: bool = True,
    normalize_input: bool = False,
) -> np.ndarray:
    """
    Computes the pixel-wise maximum of two images with optional preprocessing and post-processing.
    """
    return process_images(
        image1,
        image2,
        operation="max",
        clip_output=clip_output,
        normalize_output=normalize_output,
        normalize_input=normalize_input,
    )
