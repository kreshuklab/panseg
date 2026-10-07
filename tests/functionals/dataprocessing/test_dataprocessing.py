import numpy as np
import pytest

from panseg.functionals.dataprocessing.dataprocessing import (
    compute_scaling_factor,
    compute_scaling_voxelsize,
    image_gaussian_smoothing,
    image_median,
    image_rescale,
    normalize_01,
    normalize_01_channel_wise,
    scale_image_to_voxelsize,
    select_channel,
)


def test_compute_scaling_factor():
    input_voxel_size = (2.0, 2.0, 2.0)
    output_voxel_size = (1.0, 1.0, 1.0)
    scaling = compute_scaling_factor(input_voxel_size, output_voxel_size)
    assert scaling == (2.0, 2.0, 2.0)

    with pytest.raises(ValueError):
        compute_scaling_factor((2.0, 2.0), (1.0, 1.0, 1.0))


def test_compute_scaling_voxelsize():
    input_voxel_size = (2.0, 2.0, 2.0)
    scaling_factor = (2.0, 2.0, 2.0)
    output_voxel_size = compute_scaling_voxelsize(input_voxel_size, scaling_factor)
    assert output_voxel_size == (1.0, 1.0, 1.0)

    with pytest.raises(ValueError):
        compute_scaling_voxelsize((2.0, 2.0), (2.0, 2.0, 2.0))


def test_scale_image_to_voxelsize():
    image = np.random.rand(10, 10, 10)
    input_voxel_size = (2.0, 2.0, 2.0)
    output_voxel_size = (1.0, 1.0, 1.0)
    scaled_image = scale_image_to_voxelsize(image, input_voxel_size, output_voxel_size)
    assert scaled_image.shape == (20, 20, 20)


def test_image_rescale():
    image = np.random.rand(10, 10, 10)
    factor = (2.0, 2.0, 2.0)
    rescaled_image = image_rescale(image, factor, order=1)
    assert rescaled_image.shape == (20, 20, 20)

    rescaled_image = image_rescale(image, (1.0, 1.0, 1.0), order=1)
    assert np.array_equal(rescaled_image, image)


def test_image_rescale_2d():
    image = np.random.rand(10, 10)
    factor = (2.0, 2.0)
    rescaled_image = image_rescale(image, factor, order=1)
    assert rescaled_image.shape == (20, 20)

    rescaled_image = image_rescale(image, (1.0, 1.0, 1.0), order=1)
    assert np.array_equal(rescaled_image, image)

    factor = (1.0, 2.0, 2.0)
    rescaled_image = image_rescale(image, factor, order=1)
    assert rescaled_image.shape == (20, 20)

    factor = (2.0, 2.0, 2.0)
    with pytest.raises(ValueError):
        rescaled_image = image_rescale(image, factor, order=1)


def test_image_median():
    radius = 1

    image = np.random.rand(10, 10)
    median_image = image_median(image, radius)
    assert median_image.shape == (10, 10)

    image = np.random.rand(1, 10, 10)
    median_image = image_median(image, radius)
    assert median_image.shape == (1, 10, 10)

    image_3d = np.random.rand(10, 10, 10)
    median_image_3d = image_median(image_3d, radius)
    assert median_image_3d.shape == (10, 10, 10)


def test_image_gaussian_smoothing():
    image = np.random.rand(10, 10, 10)
    sigma = 2.0
    smoothed_image = image_gaussian_smoothing(image, sigma)
    assert smoothed_image.shape == (10, 10, 10)
    assert smoothed_image.dtype == np.float32


def test_image_gaussian_smoothing_2d():
    image = np.random.rand(10, 10)
    sigma = 2.0
    smoothed_image = image_gaussian_smoothing(image, sigma)
    assert smoothed_image.shape == (10, 10)
    assert smoothed_image.dtype == np.float32


def test_normalize_01():
    data = np.random.rand(10, 10, 10)
    normalized_data = normalize_01(data)
    assert np.allclose(np.min(normalized_data), 0)
    assert np.allclose(np.max(normalized_data), 1)
    assert normalized_data.min() >= 0.0
    assert normalized_data.max() <= 1.0 + 1e-6


def test_select_channel():
    data = np.random.rand(5, 10, 10, 10)

    channel_data = select_channel(data, 2, channel_axis=0)
    assert channel_data.shape == (10, 10, 10)

    channel_data = select_channel(data, 2, channel_axis=1)
    assert channel_data.shape == (5, 10, 10)


def test_normalize_01_channel_wise():
    data = np.random.rand(3, 10, 10, 10)
    normalized_data = normalize_01_channel_wise(data, channel_axis=0)
    assert normalized_data.shape == (3, 10, 10, 10)
    for i in range(3):
        assert np.allclose(np.min(normalized_data[i]), 0)
        assert np.allclose(np.max(normalized_data[i]), 1)
