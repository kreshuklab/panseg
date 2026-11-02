"""Tests for the shared `amp: auto|off` precision switch.

`panseg.functionals.precision.amp_context` is the single place that maps the
user-facing `amp` option to a torch context manager; training, inference, and
the GPU memory probes all resolve the precision mode through it, so sizing and
execution can never run in different precisions.
"""

import pytest
import torch

from panseg.functionals.precision import amp_context, cudnn_benchmark

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for AMP tests"
)


def test_amp_off_never_enables_autocast():
    with amp_context("off", "cuda"):
        assert not torch.is_autocast_enabled("cuda")
        assert not torch.is_autocast_enabled("cpu")


def test_amp_auto_is_silent_fp32_on_cpu():
    with amp_context("auto", "cpu"):
        assert not torch.is_autocast_enabled("cuda")
        assert not torch.is_autocast_enabled("cpu")


@requires_cuda
def test_amp_auto_is_bf16_autocast_on_cuda():
    with amp_context("auto", "cuda:0"):
        assert torch.is_autocast_enabled("cuda")
        assert torch.get_autocast_dtype("cuda") == torch.bfloat16


@requires_cuda
def test_amp_auto_accepts_plain_cuda_device_string():
    with amp_context("auto", "cuda"):
        assert torch.is_autocast_enabled("cuda")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_amp_rejects_unknown_modes(device):
    with pytest.raises(ValueError, match="amp"):
        amp_context("sometimes", device)


def test_cudnn_benchmark_enabled_only_within_context():
    previous = torch.backends.cudnn.benchmark

    with cudnn_benchmark():
        assert torch.backends.cudnn.benchmark is True

    assert torch.backends.cudnn.benchmark == previous


def test_cudnn_benchmark_restores_true_previous_value():
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        with cudnn_benchmark(enabled=False):
            assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.benchmark = previous
