"""The GPU memory probes must size patches/batches in the execution precision.

`probe_max_patch_shape`, `will_CUDA_OOM`, and `find_batch_size` measure GPU
memory with real forward (and backward) passes. Under `amp: auto` those runs
must happen under the same autocast mode as the later training/inference runs,
otherwise the patch/batch is sized for fp32 memory and the bf16 headroom is
discarded. The feasibility search and `ArrayPredictor` must pass their
configured mode down to the probes.
"""

import numpy as np
import pytest
import torch
from torch import nn

from panseg.functionals.prediction.utils import array_predictor, size_finder
from panseg.functionals.prediction.utils.array_predictor import ArrayPredictor
from panseg.functionals.prediction.utils.size_finder import (
    find_feasible_patch_and_halo_shapes,
    probe_max_patch_shape,
    will_CUDA_OOM,
)
from panseg.functionals.training.model import UNet2D, UNet3D

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for AMP tests"
)

VOL = (48, 1536, 1536)
HALO = (0, 44, 44)


def _tiny_3d_model() -> UNet3D:
    return UNet3D(in_channels=1, out_channels=1, f_maps=8, num_levels=2)


def _tiny_2d_model() -> UNet2D:
    return UNet2D(in_channels=1, out_channels=1, f_maps=8, num_levels=2)


def _record_autocast(model: nn.Module) -> list[tuple[bool, torch.dtype]]:
    """Record the CUDA autocast state seen inside every model forward."""
    seen: list[tuple[bool, torch.dtype]] = []
    model.register_forward_hook(
        lambda module, inputs, output: seen.append(
            (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        )
    )
    return seen


@requires_cuda
@pytest.mark.parametrize("amp, enabled", [("auto", True), ("off", False)])
def test_find_batch_size_probes_in_the_resolved_precision_mode(amp, enabled):
    from panseg.functionals.prediction.utils.size_finder import find_batch_size

    model = _tiny_3d_model()
    seen = _record_autocast(model)

    find_batch_size(model, 1, (8, 8, 8), (0, 0, 0), "cuda", amp=amp)

    assert seen, "probe ran no forward pass"
    for autocast_on, autocast_dtype in seen:
        assert autocast_on is enabled
        if enabled:
            assert autocast_dtype == torch.bfloat16


@requires_cuda
@pytest.mark.parametrize("amp, enabled", [("auto", True), ("off", False)])
def test_find_batch_size_training_probe_in_the_resolved_precision_mode(
    amp, enabled
):
    from panseg.functionals.prediction.utils.size_finder import find_batch_size

    model = _tiny_3d_model()
    seen = _record_autocast(model)

    find_batch_size(model, 1, (8, 8, 8), (0, 0, 0), "cuda", for_training=True, amp=amp)

    assert seen, "probe ran no forward pass"
    for autocast_on, _ in seen:
        assert autocast_on is enabled


@requires_cuda
@pytest.mark.parametrize("amp, enabled", [("auto", True), ("off", False)])
def test_will_cuda_oom_probes_in_the_resolved_precision_mode(amp, enabled):
    model = _tiny_3d_model()
    seen = _record_autocast(model)

    will_CUDA_OOM(model, 1, (8, 8, 8), (0, 0, 0), 1, "cuda", amp=amp)

    assert seen, "probe ran no forward pass"
    for autocast_on, _ in seen:
        assert autocast_on is enabled


@requires_cuda
@pytest.mark.parametrize("amp, enabled", [("auto", True), ("off", False)])
def test_probe_max_patch_shape_runs_in_the_resolved_precision_mode(amp, enabled):
    # 2D models take the linear search, which starts right at the top and is
    # therefore the fastest probe path to observe on a real GPU
    model = _tiny_2d_model()
    seen = _record_autocast(model)

    probe_max_patch_shape(model, 1, "cuda", amp=amp)

    assert seen, "probe ran no forward pass"
    for autocast_on, _ in seen:
        assert autocast_on is enabled


class _RecordingProbe:
    """Stand-in for a size_finder probe: records the `amp` it was called with."""

    def __init__(self, result):
        self.result = result
        self.amps: list[str] = []

    def __call__(self, *args, amp: str = "auto", **kwargs):
        self.amps.append(amp)
        return self.result


def _patch_probes(monkeypatch):
    probe = _RecordingProbe((64, 64, 64))
    oom = _RecordingProbe(False)
    monkeypatch.setattr(size_finder, "probe_max_patch_shape", probe)
    monkeypatch.setattr(size_finder, "will_CUDA_OOM", oom)
    return probe, oom


def test_find_feasible_search_probes_in_the_configured_mode(monkeypatch):
    probe, oom = _patch_probes(monkeypatch)

    find_feasible_patch_and_halo_shapes(
        nn.Module(), 1, VOL, HALO, device="cuda", amp="off"
    )

    assert probe.amps == ["off"]
    assert oom.amps == ["off"]


def test_find_feasible_search_probes_default_to_auto(monkeypatch):
    probe, oom = _patch_probes(monkeypatch)

    find_feasible_patch_and_halo_shapes(nn.Module(), 1, VOL, HALO, device="cuda")

    assert probe.amps == ["auto"]
    assert oom.amps == ["auto"]


def test_array_predictor_probes_in_the_configured_mode(monkeypatch):
    """The predictor's batch-size probe shares its precision mode."""
    calls: dict[str, str] = {}

    def fake_find_batch(model, in_channels, patch_shape, patch_halo, device, **kwargs):
        calls["batch"] = kwargs.get("amp", "auto")
        return 2

    monkeypatch.setattr(array_predictor, "find_batch_size", fake_find_batch)

    ArrayPredictor(
        model=_tiny_3d_model(),
        in_channels=1,
        out_channels=1,
        device="cpu",
        patch=(8, 8, 8),
        patch_halo=(0, 0, 0),
        single_batch_mode=False,
        headless=False,
        amp="off",
    )

    assert calls == {"batch": "off"}
