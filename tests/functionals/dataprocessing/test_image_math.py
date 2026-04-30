import numpy as np
import pytest

from panseg.functionals.dataprocessing import (
    add_images,
    divide_images,
    max_images,
    multiply_images,
    process_images,
    subtract_images,
)
from panseg.functionals.dataprocessing.dataprocessing import normalize_01


@pytest.mark.parametrize(
    "operation, expected_result",
    [
        ("add", np.array([[2, 4], [6, 8]])),
        ("multiply", np.array([[1, 4], [9, 16]])),
        ("subtract", np.array([[0, 0], [0, 0]])),
        ("divide", np.array([[1, 1], [1, 1]])),
        ("max", np.array([[1, 2], [3, 4]])),
    ],
)
def test_process_images(operation, expected_result):
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    result = process_images(
        image1,
        image2,
        operation=operation,
        normalize_input=False,
        clip_output=False,
        normalize_output=False,
    )
    assert np.allclose(result, expected_result), f"Failed for operation: {operation}"


def test_process_images_clip_output():
    image1 = np.array([[0.5, 1.5], [2.5, 3.5]])
    image2 = np.array([[1.0, 1.0], [1.0, 1.0]])

    result = process_images(
        image1, image2, operation="add", clip_output=True, normalize_output=False
    )
    expected = np.clip(image1 + image2, 0, 1)
    assert np.allclose(result, expected), "Clipping failed"


def test_process_images_normalize_input_output():
    image1 = np.array([[10, 20], [30, 40]])
    image2 = np.array([[1, 2], [3, 4]])

    result = process_images(
        image1, image2, operation="add", normalize_input=True, normalize_output=True
    )
    assert np.allclose(result, normalize_01(result)), "Normalization failed"


# Test cases for specific operations
def test_add_images():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    result = add_images(
        image1, image2, clip_output=False, normalize_output=False, normalize_input=False
    )
    expected = image1 + image2
    assert np.allclose(result, expected), "Addition failed"


def test_multiply_images():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    result = multiply_images(
        image1, image2, clip_output=False, normalize_output=False, normalize_input=False
    )
    expected = image1 * image2
    assert np.allclose(result, expected), "Multiplication failed"


def test_subtract_images():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    result = subtract_images(
        image1, image2, clip_output=False, normalize_output=False, normalize_input=False
    )
    expected = image1 - image2
    assert np.allclose(result, expected), "Subtraction failed"


def test_divide_images():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    result = divide_images(
        image1, image2, clip_output=False, normalize_output=False, normalize_input=False
    )
    expected = image1 / image2
    assert np.allclose(result, expected), "Division failed"


def test_max_images():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[4, 3], [2, 1]])

    result = max_images(
        image1, image2, clip_output=False, normalize_output=False, normalize_input=False
    )
    expected = np.maximum(image1, image2)
    assert np.allclose(result, expected), "Max operation failed"


def test_process_images_invalid_operation():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="Unsupported operation: invalid"):
        process_images(
            image1,
            image2,
            operation="invalid",
            normalize_input=False,
            clip_output=False,
            normalize_output=False,
        )
