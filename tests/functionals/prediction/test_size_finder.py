import os

import numpy as np
import pytest
import torch

from panseg.core.zoo import model_zoo
from panseg.functionals.prediction.utils.size_finder import (
    find_a_max_patch_shape,
    find_batch_size,
    find_patch_and_halo_shapes,
)

IN_GITHUB_ACTIONS = (
    os.getenv("GITHUB_ACTIONS") == "true"
)  # set to true in GitHub Actions by default to skip CUDA tests
DOWNLOAD_MODELS = (
    os.getenv("DOWNLOAD_MODELS") == "true"
)  # set to false in locall testing to skip downloading models
LARGE_VRAM_GPUS = [
    "NVIDIA A100",
    "NVIDIA A40",
]  # these two are not full names because A100 has multiple models
ALL_TESTED_GPUS = [
    "NVIDIA GeForce RTX 2080 Ti",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA A100-PCIE-40GB",
    "NVIDIA A40",
    "NVIDIA GeForce RTX 4050 Laptop GPU",
]
MAX_PATCH_SHAPES = {
    "generic_confocal_3D_unet": {
        "NVIDIA GeForce RTX 2080 Ti": (208, 208, 208),
        "NVIDIA GeForce RTX 3090": (256, 256, 256),
        "NVIDIA A100-PCIE-40GB": (272, 272, 272),
        "NVIDIA A40": (272, 272, 272),
        "NVIDIA GeForce RTX 4050 Laptop GPU": (160, 160, 160),
    },
    "confocal_2D_unet_ovules_ds2x": {
        "NVIDIA GeForce RTX 2080 Ti": (
            1,
            1920,
            1920,
        ),  # (1, 2048, 2048) if search step is 1.
        "NVIDIA GeForce RTX 3090": (
            1,
            2880,
            2880,
        ),  # (1, 2960, 2960) if search step is 1.
        "NVIDIA A100-PCIE-40GB": (1, 3200, 3200),
        "NVIDIA A40": (1, 3200, 3200),
        "NVIDIA GeForce RTX 4050 Laptop GPU": (1, 1280, 1280),
    },
}

try:
    # This will raise an AssertionError if Pytorch is not installed with CUDA support
    GPU_DEVICE_NAME = torch.cuda.get_device_name(0) if not IN_GITHUB_ACTIONS else ""
except AssertionError:  # catch Pytorch not installed with CUDA support
    GPU_DEVICE_NAME = ""


@pytest.mark.parametrize(
    "full_volume_shape, max_patch_shape, min_halo_shape, expected",
    [  # Here halo is 1-sided
        (
            (1, 12000, 50000),
            (192, 192, 192),
            (44, 44, 44),
            ((1, 2572, 2572), (0, 44, 44)),
        ),
        ((1, 120, 500), (192, 192, 192), (44, 44, 44), ((1, 120, 500), (0, 0, 0))),
        ((95, 120, 500), (192, 192, 192), (44, 44, 44), ((95, 120, 500), (0, 0, 0))),
        ((95, 120, 5000), (192, 192, 192), (44, 44, 44), ((95, 120, 532), (0, 0, 44))),
        ((5000, 120, 95), (192, 192, 192), (44, 44, 44), ((532, 120, 95), (44, 0, 0))),
        (
            (100, 1000, 1000),
            (192, 192, 192),
            (44, 44, 44),
            ((100, 178, 178), (0, 44, 44)),
        ),
        (
            (1000, 1000, 1000),
            (192, 192, 192),
            (44, 44, 44),
            ((104, 104, 104), (44, 44, 44)),
        ),
        (
            (1000, 1000, 1000),
            (192, 192, 192),
            (100, 100, 100),
            ((192, 192, 192), (0, 0, 0)),
        ),
        ((16, 8, 8), (10, 10, 10), (4, 4, 4), ((7, 8, 8), (4, 0, 0))),
        ((16, 10, 10), (10, 10, 10), (4, 4, 4), ((2, 2, 2), (4, 4, 4))),
        ((12, 12, 8), (10, 10, 10), (4, 4, 4), ((3, 3, 8), (4, 4, 0))),
    ],
)
def test_find_patch_and_halo_shapes(
    full_volume_shape, max_patch_shape, min_halo_shape, expected
):
    result = find_patch_and_halo_shapes(
        full_volume_shape, max_patch_shape, min_halo_shape
    )
    assert result == expected

    double_halo_out = np.array(result[1]) * 2
    assert np.prod(result[0] + double_halo_out) <= np.prod(max_patch_shape)

    double_halo_shape = (
        min_halo_shape[0] * 2,
        min_halo_shape[1] * 2,
        min_halo_shape[2] * 2,
    )
    result = find_patch_and_halo_shapes(
        full_volume_shape, max_patch_shape, double_halo_shape, both_sides=True
    )
    assert result == expected


@pytest.mark.skipif(
    GPU_DEVICE_NAME not in ALL_TESTED_GPUS,
    reason="Measured devices are not available.",
)
@pytest.mark.parametrize("model_name", MAX_PATCH_SHAPES.keys())
def test_find_patch_shape(model_name):
    model, _, _ = model_zoo.get_model_by_name(model_name, model_update=DOWNLOAD_MODELS)
    found_patch_shape = find_a_max_patch_shape(model, 1, "cuda:0")
    expected_patch_shape = MAX_PATCH_SHAPES[model_name][GPU_DEVICE_NAME]
    assert found_patch_shape == expected_patch_shape


@pytest.mark.skipif(
    not any(gpu in GPU_DEVICE_NAME for gpu in LARGE_VRAM_GPUS),
    reason="Test requires a large VRAM device (e.g., NVIDIA A100 or NVIDIA A40).",
)
def test_find_batch_size_error_handling():
    model, _, _ = model_zoo.get_model_by_name(
        "confocal_3D_unet_ovules_ds3x", model_update=DOWNLOAD_MODELS
    )
    found_batch_size = find_batch_size(model, 1, (86, 395, 395), (0, 44, 44), "cuda:0")
    assert found_batch_size == 1


@pytest.mark.skipif(
    not any(gpu in GPU_DEVICE_NAME for gpu in LARGE_VRAM_GPUS),
    reason="Test requires a large VRAM device (e.g., NVIDIA A100 or NVIDIA A40).",
)
def test_find_patch_shape_error_handling():
    model, _, _ = model_zoo.get_model_by_name(
        "PanSeg_3Dnuc_platinum", model_update=DOWNLOAD_MODELS
    )
    found_patch_shape = find_a_max_patch_shape(model, 1, "cuda:0")
    if "NVIDIA A100-PCIE-40GB" == GPU_DEVICE_NAME:
        print("NVIDIA A100-PCIE-40GB tested")
        assert found_patch_shape == (352, 352, 352)
    if "NVIDIA A40" == GPU_DEVICE_NAME:
        print("NVIDIA A40 tested")
        assert found_patch_shape == (352, 352, 352)
