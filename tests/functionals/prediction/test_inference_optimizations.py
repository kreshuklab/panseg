"""Measured inference optimizations: channels_last_3d and torch.compile.

`channels_last_3d` is a memory-layout change, not a math change: outputs must
match the plain layout within a tight tolerance and stay contiguous float32
numpy arrays. It is default-on on CUDA for 3D models and must stay off on CPU
and for 2D models. `torch.compile` is opt-in (one-time compile cost; triton
needs `C_INCLUDE_PATH=/usr/local/cuda/include` in some environments) and must
degrade gracefully to eager when compilation fails.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from panseg.functionals.prediction import prediction as prediction_module
from panseg.functionals.prediction import unet_prediction
from panseg.functionals.prediction.utils.array_dataset import ArrayDataset
from panseg.functionals.prediction.utils.array_predictor import ArrayPredictor
from panseg.functionals.training.model import UNet3D
from tests.functionals.prediction.helpers import (
    MODEL_CONFIG,
    make_dataset,
    raw_volume,
    requires_cuda,
    tiny_3d_model,
)


def _predict(model, dataset, device: str, in_channels: int = 1, **kwargs) -> np.ndarray:
    return _make_predictor(model, device, in_channels, **kwargs)(dataset)


def _make_predictor(
    model, device: str, in_channels: int = 1, **kwargs
) -> ArrayPredictor:
    return ArrayPredictor(
        model=model,
        in_channels=in_channels,
        out_channels=1,
        device=device,
        patch=(8, 64, 64),
        patch_halo=(2, 8, 8),
        single_batch_mode=True,
        headless=False,
        **kwargs,
    )


def _record_memory_format(model: torch.nn.Module) -> list[tuple[bool, bool]]:
    """Record (input is channels_last_3d, all 5D params are) inside each forward."""
    seen: list[tuple[bool, bool]] = []
    cl3d = torch.channels_last_3d

    def hook(module, inputs, output):
        input_cl = inputs[0].is_contiguous(memory_format=cl3d)
        params_cl = all(
            p.is_contiguous(memory_format=cl3d)
            for p in module.parameters()
            if p.dim() == 5
        )
        seen.append((input_cl, params_cl))

    model.register_forward_hook(hook)
    return seen


@requires_cuda
def test_channels_last_3d_outputs_match_plain_layout():
    """The layout change must not change the math (tight tolerance)."""
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))
    state = tiny_3d_model().state_dict()

    def _run(channels_last_3d: bool) -> np.ndarray:
        model = tiny_3d_model()
        model.load_state_dict(state)
        return _predict(
            model,
            dataset,
            "cuda",
            amp="off",  # isolate the layout change from the precision change
            channels_last_3d=channels_last_3d,
        )

    pmaps_plain = _run(channels_last_3d=False)
    pmaps_cl = _run(channels_last_3d=True)

    assert pmaps_cl.dtype == np.float32
    assert pmaps_cl.flags.c_contiguous
    assert pmaps_cl.shape == pmaps_plain.shape == (1, 16, 96, 96)
    # the layout change is not a math change, but it lets cuDNN pick different
    # kernels with different accumulation orders (observed max |diff| ~ 4e-4);
    # this tolerance is ~50x tighter than the bf16-on/off equivalence one
    np.testing.assert_allclose(pmaps_cl, pmaps_plain, rtol=1e-3, atol=1e-3)


def _multichannel_dataset() -> ArrayDataset:
    """Two-channel input: with C=1 the channel stride is layout-agnostic and a
    plain contiguous input would trivially count as channels_last_3d."""
    raw = np.random.default_rng(0).random((2, 16, 96, 96)).astype("float32")
    return make_dataset(raw, patch=(8, 64, 64), halo=(2, 8, 8), multichannel=True)


@requires_cuda
def test_channels_last_3d_engaged_on_cuda():
    model = tiny_3d_model(in_channels=2)
    seen = _record_memory_format(model)

    predictor = _make_predictor(model, "cuda", in_channels=2, amp="off")
    seen.clear()  # the sizing probe forwards run plain by design (conservative)
    predictor(_multichannel_dataset())

    assert seen, "no forward pass ran"
    assert all(input_cl for input_cl, _ in seen), "inputs not channels_last_3d"
    assert all(params_cl for _, params_cl in seen), "params not channels_last_3d"


@requires_cuda
def test_channels_last_3d_disabled_on_request():
    model = tiny_3d_model(in_channels=2)
    seen = _record_memory_format(model)

    _predict(
        model,
        _multichannel_dataset(),
        "cuda",
        in_channels=2,
        amp="off",
        channels_last_3d=False,
    )

    assert seen, "no forward pass ran"
    assert not any(input_cl for input_cl, _ in seen)


def test_channels_last_3d_stays_off_on_cpu():
    model = tiny_3d_model(in_channels=2)
    seen = _record_memory_format(model)

    # default channels_last_3d=True must not engage on CPU
    _predict(model, _multichannel_dataset(), "cpu", in_channels=2)

    assert seen, "no forward pass ran"
    assert not any(input_cl for input_cl, _ in seen)


def test_compile_defaults_to_off(monkeypatch):
    def fail_if_called(model, *args, **kwargs):
        raise AssertionError("torch.compile must not run when compile is off")

    monkeypatch.setattr(torch, "compile", fail_if_called)
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))

    pmaps = _predict(tiny_3d_model(), dataset, "cpu")

    assert pmaps.shape == (1, 16, 96, 96)


@requires_cuda
def test_compile_wraps_model_and_matches_eager(monkeypatch):
    calls: list = []
    real_compile = torch.compile

    def recording_compile(model, *args, **kwargs):
        calls.append(model)
        return real_compile(model, *args, **kwargs)

    monkeypatch.setattr(torch, "compile", recording_compile)
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))
    state = tiny_3d_model().state_dict()

    def _run(compile: bool) -> np.ndarray:
        model = tiny_3d_model()
        model.load_state_dict(state)
        return _predict(model, dataset, "cuda", amp="off", compile=compile)

    pmaps_eager = _run(compile=False)
    pmaps_compiled = _run(compile=True)

    assert len(calls) == 1, "torch.compile was not used for compile=True"
    assert pmaps_compiled.dtype == np.float32
    assert pmaps_compiled.flags.c_contiguous
    np.testing.assert_allclose(pmaps_compiled, pmaps_eager, rtol=1e-3, atol=1e-3)


def test_compile_degrades_gracefully_on_first_forward_failure(monkeypatch, caplog):
    """A failing compiled forward falls back to eager with a clear warning."""

    class FailingCompiled(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            raise RuntimeError("inductor: fatal error: cuda.h: No such file")

    monkeypatch.setattr(
        torch, "compile", lambda model, **kwargs: FailingCompiled(model)
    )
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))

    with caplog.at_level("WARNING"):
        pmaps = _predict(tiny_3d_model(), dataset, "cpu", compile=True)

    assert pmaps.shape == (1, 16, 96, 96)
    assert np.isfinite(pmaps).all()
    assert any("torch.compile" in record.getMessage() for record in caplog.records)


def test_unet_prediction_forwards_optimization_flags(tmp_path, monkeypatch):
    """`unet_prediction` passes the optimization flags to the predictor."""
    calls: dict = {}

    def fake_find_feasible(*args, **kwargs):
        return (8, 64, 64), (2, 8, 8)

    class FakePredictor:
        def __init__(self, **kwargs):
            calls["channels_last_3d"] = kwargs.get("channels_last_3d")
            calls["compile"] = kwargs.get("compile")

        def __call__(self, dataset):
            return np.zeros((1,) + dataset.raw.shape, dtype="float32")

    monkeypatch.setattr(
        prediction_module, "find_feasible_patch_and_halo_shapes", fake_find_feasible
    )
    monkeypatch.setattr(prediction_module, "ArrayPredictor", FakePredictor)

    torch.manual_seed(0)
    model = UNet3D(in_channels=1, out_channels=1, final_sigmoid=True, f_maps=[8, 16])
    weights_path = tmp_path / "best_checkpoint.pytorch"
    torch.save(model.state_dict(), weights_path)
    config_path = tmp_path / "config_train.yml"
    config_path.write_text(MODEL_CONFIG)

    unet_prediction(
        raw=raw_volume(),
        input_layout="ZYX",
        model_name=None,
        model_id=None,
        config_path=config_path,
        model_weights_path=weights_path,
        device="cpu",
        disable_tqdm=True,
        channels_last_3d=False,
        compile=True,
    )

    assert calls == {"channels_last_3d": False, "compile": True}


@requires_cuda
def test_array_predictor_h2d_is_non_blocking_on_cuda(tensor_to_spy):
    """The pinned batch transfers of the prediction loop copy to the GPU
    asynchronously (the constructor's memory probes are synchronous by
    design and excluded here)."""
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))
    model = tiny_3d_model()
    predictor = _make_predictor(model, "cuda", amp="off")

    del tensor_to_spy[:]  # drop the constructor's probe transfers
    predictor(dataset)

    # patch (8, 64, 64) + halo (2, 8, 8) both sides -> batches (b, 1, 12, 80, 80);
    # a positional device arg marks the H2D move (the channels_last_3d
    # conversion is kwargs-only)
    batch_calls = [
        kw for shape, args, kw in tensor_to_spy if shape == (1, 1, 12, 80, 80) and args
    ]
    assert batch_calls, "the batch transfer must be observed"
    assert all(kw.get("non_blocking") is True for kw in batch_calls)


@pytest.mark.parametrize("pinned", [True, False])
def test_array_predictor_h2d_follows_the_pinned_state(
    tensor_to_spy, monkeypatch, pinned
):
    """The flag follows the batch's pinned state, not the device type.

    On unpinned memory `non_blocking=True` is a silent no-op, so a loader
    without pinning (or a device where pinning is unavailable) must copy
    blocking. Run on CPU to exercise the derivation without CUDA.
    """
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: pinned)
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))
    predictor = _make_predictor(tiny_3d_model(), "cpu", amp="off")

    del tensor_to_spy[:]  # drop the constructor's probe transfers
    predictor(dataset)

    batch_calls = [
        kw for shape, args, kw in tensor_to_spy if shape == (1, 1, 12, 80, 80) and args
    ]
    assert batch_calls, "the batch transfer must be observed"
    assert all(kw.get("non_blocking") is pinned for kw in batch_calls)
