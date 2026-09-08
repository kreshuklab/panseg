import os

import numpy as np
import pytest
import torch
from torch import nn

from panseg.core.zoo import model_zoo
from panseg.functionals.prediction.utils import size_finder
from panseg.functionals.prediction.utils.size_finder import (
    derive_patch_and_halo_shapes,
    find_batch_size,
    find_feasible_patch_and_halo_shapes,
    probe_max_patch_shape,
    will_CUDA_OOM,
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
    "NVIDIA GeForce RTX 4090",
]
MAX_PATCH_SHAPES = {
    "generic_confocal_3D_unet": {
        "NVIDIA GeForce RTX 2080 Ti": (208, 208, 208),
        "NVIDIA GeForce RTX 3090": (256, 256, 256),
        "NVIDIA A100-PCIE-40GB": (272, 272, 272),
        "NVIDIA A40": (272, 272, 272),
        "NVIDIA GeForce RTX 4050 Laptop GPU": (160, 160, 160),
        "NVIDIA GeForce RTX 4090": (256, 256, 256),
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
        "NVIDIA GeForce RTX 4090": (1, 2560, 2560),
    },
}

try:
    # This will raise an AssertionError if Pytorch is not installed with CUDA support
    GPU_DEVICE_NAME = torch.cuda.get_device_name(0) if not IN_GITHUB_ACTIONS else ""
except AssertionError:  # catch Pytorch not installed with CUDA support
    GPU_DEVICE_NAME = ""

# Fixed scenario for the feasibility-loop unit tests (no GPU required, will_CUDA_OOM is faked)
VOL = (48, 1536, 1536)
HALO = (0, 44, 44)
PROBED_MAX = (256, 256, 256)


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
def test_derive_patch_and_halo_shapes(
    full_volume_shape, max_patch_shape, min_halo_shape, expected
):
    result = derive_patch_and_halo_shapes(
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
    result = derive_patch_and_halo_shapes(
        full_volume_shape, max_patch_shape, double_halo_shape, both_sides=True
    )
    assert result == expected


class FakeOomProbe:
    """Stand-in for will_CUDA_OOM: OOMs iff the actual forward shape exceeds `threshold` voxels."""

    def __init__(self):
        self.threshold = None  # None -> never OOM
        self.verified_shapes = []

    def __call__(self, model, in_channels, patch_shape, patch_halo, batch_size, device):
        actual = tuple(patch_shape[i] + 2 * patch_halo[i] for i in range(3))
        self.verified_shapes.append(actual)
        if self.threshold is None:
            return False
        return int(np.prod(actual)) > self.threshold


@pytest.fixture()
def fake_oom_probe(monkeypatch):
    """Deterministic stand-ins for the GPU-probing pieces of the feasibility loop."""
    probe = FakeOomProbe()
    monkeypatch.setattr(size_finder, "will_CUDA_OOM", probe)
    monkeypatch.setattr(
        size_finder,
        "probe_max_patch_shape",
        lambda model, in_channels, device: PROBED_MAX,
    )
    return probe


def test_find_feasible_patch_and_halo_shapes_drives_oom_probe(fake_oom_probe):
    expected = derive_patch_and_halo_shapes(VOL, PROBED_MAX, HALO)
    result = find_feasible_patch_and_halo_shapes(
        nn.Module(), 1, VOL, HALO, device="cuda"
    )
    assert result == expected
    assert len(fake_oom_probe.verified_shapes) == 1


def test_find_feasible_patch_and_halo_shapes_shrinks_until_verified(
    fake_oom_probe,
):
    fake_oom_probe.threshold = (
        15_000_000  # first attempt (16.8M voxels) OOMs, the shrunken one fits
    )

    result = find_feasible_patch_and_halo_shapes(
        nn.Module(), 1, VOL, HALO, device="cuda"
    )

    result_voxels = int(np.prod([result[0][i] + 2 * result[1][i] for i in range(3)]))
    assert result_voxels <= fake_oom_probe.threshold
    assert result_voxels < int(np.prod(fake_oom_probe.verified_shapes[0]))
    assert len(fake_oom_probe.verified_shapes) == 2


def test_find_feasible_patch_and_halo_shapes_raises_when_nothing_fits(
    fake_oom_probe,
):
    fake_oom_probe.threshold = 1  # every attempt OOMs

    with pytest.raises(
        RuntimeError, match="Could not determine a feasible patch shape"
    ):
        find_feasible_patch_and_halo_shapes(
            nn.Module(), 1, VOL, HALO, device="cuda", max_attempts=3
        )
    assert len(fake_oom_probe.verified_shapes) == 3


def test_find_feasible_patch_and_halo_shapes_skips_verification_on_cpu(
    fake_oom_probe,
):
    result = find_feasible_patch_and_halo_shapes(
        nn.Module(), 1, VOL, HALO, device="cpu"
    )

    assert result == derive_patch_and_halo_shapes(VOL, PROBED_MAX, HALO)
    assert fake_oom_probe.verified_shapes == []


@pytest.mark.skipif(
    GPU_DEVICE_NAME not in ALL_TESTED_GPUS,
    reason="Measured devices are not available.",
)
@pytest.mark.parametrize("model_name", MAX_PATCH_SHAPES.keys())
def test_find_patch_shape(model_name):
    model, _, _ = model_zoo.get_model_by_name(model_name, model_update=DOWNLOAD_MODELS)
    found_patch_shape = probe_max_patch_shape(model, 1, "cuda:0")
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
    found_patch_shape = probe_max_patch_shape(model, 1, "cuda:0")
    if "NVIDIA A100-PCIE-40GB" == GPU_DEVICE_NAME:
        print("NVIDIA A100-PCIE-40GB tested")
        assert found_patch_shape == (352, 352, 352)
    if "NVIDIA A40" == GPU_DEVICE_NAME:
        print("NVIDIA A40 tested")
        assert found_patch_shape == (352, 352, 352)


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or GPU_DEVICE_NAME == "",
    reason="CUDA device not available.",
)
def test_find_feasible_patch_and_halo_shapes_shrinks_on_oom(monkeypatch):
    """Regression for issue #578: when GPU memory becomes unavailable between the
    max-patch probe and the batch size determination (e.g. other processes on a shared
    GPU), the auto-determined patch shape must shrink until it fits instead of raising
    an OOM error at batch size 1."""
    model, _, _ = model_zoo.get_model_by_name(
        "confocal_3D_unet_ovules_ds3x", model_update=DOWNLOAD_MODELS
    )
    device = "cuda:0"
    full_volume_shape = (48, 1536, 1536)
    min_halo_shape = model_zoo.compute_3D_halo_for_pytorch3dunet(model)

    free, total = torch.cuda.mem_get_info()
    if free < 3 * 2**30:  # not enough spare VRAM to simulate competing allocations
        pytest.skip("Less than 3 GiB of free VRAM available.")
    reference = derive_patch_and_halo_shapes(
        full_volume_shape, probe_max_patch_shape(model, 1, device), min_halo_shape
    )

    real_find = size_finder.derive_patch_and_halo_shapes
    holder = []

    def reserving_find(*args, **kwargs):
        if not holder:  # emulate memory appearing after the max-patch probe
            holder.append(
                torch.empty(max(total // 10, 2**30), dtype=torch.uint8, device=device)
            )
        return real_find(*args, **kwargs)

    monkeypatch.setattr(size_finder, "derive_patch_and_halo_shapes", reserving_find)
    try:
        patch, patch_halo = find_feasible_patch_and_halo_shapes(
            model, 1, full_volume_shape, min_halo_shape, device
        )
    finally:
        holder.clear()
        torch.cuda.empty_cache()

    # the returned shape is verified to fit even with the competing allocation in place
    assert not will_CUDA_OOM(model, 1, patch, patch_halo, 1, device)
    # and the shrink loop actually engaged (the unshrunken shape no longer fit)
    assert int(np.prod(patch)) < int(np.prod(reference[0]))
