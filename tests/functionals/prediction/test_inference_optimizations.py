"""Measured inference optimizations: channels_last_3d and torch.compile.

`channels_last_3d` is a memory-layout change, not a math change: outputs must
match the plain layout within a tight tolerance and stay contiguous float32
numpy arrays. It is opt-in and defaults to off because the measured result is
stack-dependent (1.7x slower on torch 2.13/CUDA 13/cuDNN 9.26); when requested
it applies only to 3D models on CUDA and must stay off on CPU and for 2D
models.

cuDNN autotuning is not enabled anywhere: its per-shape tuning cost does not
amortize at typical prediction batch counts, and for training the first-step
autotune plus the training batch-size probe cost far more than the steady-state
win.

`torch.compile` is now automatic by default (`compile=None`): it engages only
on CUDA and only when the number of prediction batches amortizes the one-time
compile cost (`COMPILE_MIN_BATCHES`). An explicit `compile=True` / `False`
override still forces compilation / eager execution. Compiling on CPU is never
worth it, so auto must keep it off there even when the batch threshold is met.
A failed compile must degrade gracefully to eager (triton needs
`C_INCLUDE_PATH=/usr/local/cuda/include` in some environments).
"""

import inspect
import math

import numpy as np
import pytest
import torch

from panseg.functionals.prediction import prediction as prediction_module
from panseg.functionals.prediction import unet_prediction
from panseg.functionals.prediction.utils import array_predictor, size_finder
from panseg.functionals.prediction.utils.array_dataset import ArrayDataset
from panseg.functionals.prediction.utils.array_predictor import ArrayPredictor
from tests.functionals.prediction.helpers import (
    make_dataset,
    raw_volume,
    requires_cuda,
    save_model_files,
    tiny_2d_model,
    tiny_3d_model,
)


def _predict(model, dataset, device: str, in_channels: int = 1, **kwargs) -> np.ndarray:
    return _make_predictor(model, device, in_channels, **kwargs)(dataset)


def _make_predictor(
    model,
    device: str,
    in_channels: int = 1,
    patch: tuple[int, int, int] = (8, 64, 64),
    patch_halo: tuple[int, int, int] = (2, 8, 8),
    single_batch_mode: bool = True,
    **kwargs,
) -> ArrayPredictor:
    return ArrayPredictor(
        model=model,
        in_channels=in_channels,
        out_channels=1,
        device=device,
        patch=patch,
        patch_halo=patch_halo,
        single_batch_mode=single_batch_mode,
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

    predictor = _make_predictor(
        model, "cuda", in_channels=2, amp="off", channels_last_3d=True
    )
    # no clear: the constructor's sizing probes must use the same layout as the
    # prediction loop (see `test_memory_probes_use_the_prediction_memory_format`)
    predictor(_multichannel_dataset())

    assert seen, "no forward pass ran"
    assert all(input_cl for input_cl, _ in seen), "inputs not channels_last_3d"
    assert all(params_cl for _, params_cl in seen), "params not channels_last_3d"


@pytest.mark.parametrize("single_batch_mode", [True, False])
@requires_cuda
def test_memory_probes_use_the_prediction_memory_format(monkeypatch, single_batch_mode):
    """The probes must size in the prediction loop's memory format.

    Sizing in plain contiguous layout while the loop ran in channels_last_3d
    under-estimated the largest single allocation and prediction OOMed at
    batch size 1 (patch 168 + halo 44 = 256^3).
    """
    monkeypatch.setattr(size_finder, "MAX_PROBE_BATCH_SIZE", 4)
    model = tiny_3d_model(in_channels=2)
    seen = _record_memory_format(model)

    _make_predictor(
        model,
        "cuda",
        in_channels=2,
        amp="off",
        single_batch_mode=single_batch_mode,
        channels_last_3d=True,
    )

    assert seen, "no probe forward ran"
    assert all(input_cl for input_cl, _ in seen), "probe inputs not channels_last_3d"
    assert all(params_cl for _, params_cl in seen), "probe params not channels_last_3d"


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

    # an explicit channels_last_3d=True must not engage on CPU
    _predict(
        model,
        _multichannel_dataset(),
        "cpu",
        in_channels=2,
        channels_last_3d=True,
    )

    assert seen, "no forward pass ran"
    assert not any(input_cl for input_cl, _ in seen)


@requires_cuda
def test_channels_last_3d_defaults_off_on_cuda_for_3d_models():
    """Opt-in: the default must not engage the layout on CUDA either."""
    model = tiny_3d_model(in_channels=2)
    seen = _record_memory_format(model)

    predictor = _make_predictor(model, "cuda", in_channels=2, amp="off")
    assert predictor.channels_last_3d is False
    predictor(_multichannel_dataset())

    assert seen, "no forward pass ran"
    assert not any(input_cl for input_cl, _ in seen)


@requires_cuda
def test_channels_last_3d_stays_off_for_2d_models_on_cuda():
    """2D models run 4D tensors; the 3D memory format must not be engaged."""
    # 2D models predict one Z slice at a time: patch depth 1, no Z halo
    predictor = _make_predictor(
        tiny_2d_model(),
        "cuda",
        amp="off",
        patch=(1, 64, 64),
        patch_halo=(0, 8, 8),
        channels_last_3d=True,
    )
    assert predictor.channels_last_3d is False

    pmaps = predictor(make_dataset(raw_volume(), patch=(1, 64, 64), halo=(0, 8, 8)))
    assert pmaps.shape == (1, 16, 96, 96)


def test_compile_auto_stays_off_on_cpu_even_above_threshold(monkeypatch):
    """Auto compile is CUDA-only: meeting the batch threshold is not enough."""
    calls: list = []

    def recording_compile(model, *args, **kwargs):
        calls.append(model)
        return model  # passthrough: no real inductor compile

    monkeypatch.setattr(array_predictor, "COMPILE_MIN_BATCHES", 1)
    monkeypatch.setattr(torch, "compile", recording_compile)
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))

    pmaps = _predict(tiny_3d_model(), dataset, "cpu")

    assert pmaps.shape == (1, 16, 96, 96)
    assert calls == [], "auto compile must never run on CPU"


def test_compile_auto_threshold_is_cuda_only_and_batch_based():
    """Unit-check the threshold arithmetic without needing a GPU.

    `_should_compile` is pure: relabelling `predictor.device` exercises the CUDA
    branch without moving any tensors.
    """
    min_batches = array_predictor.COMPILE_MIN_BATCHES

    predictor = _make_predictor(tiny_3d_model(), "cpu", amp="off")
    # CPU never compiles automatically, even for an absurd number of batches
    assert predictor._should_compile(10**9) is False

    predictor.device = "cuda"
    assert predictor._should_compile(min_batches - 1) is False
    assert predictor._should_compile(min_batches) is True

    # explicit overrides win over device and threshold
    forced = _make_predictor(tiny_3d_model(), "cpu", amp="off", compile=True)
    assert forced._should_compile(0) is True

    eager = _make_predictor(tiny_3d_model(), "cpu", amp="off", compile=False)
    eager.device = "cuda"
    assert eager._should_compile(10**9) is False


@requires_cuda
def test_compile_auto_decision_counts_batches_not_patches(monkeypatch):
    """The threshold applies to `len(DataLoader)`, i.e. batches, not patches."""
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))
    n_patches = len(dataset)
    n_batches_at_bs2 = math.ceil(n_patches / 2)
    # Above what batch size 2 yields, but reachable with batch size 1.
    threshold = n_batches_at_bs2 + 1
    assert threshold <= n_patches, "test needs batch size 1 to cross the threshold"

    calls: list = []

    def recording_compile(model, *args, **kwargs):
        calls.append(model)
        return model  # passthrough: no real inductor compile

    monkeypatch.setattr(torch, "compile", recording_compile)
    monkeypatch.setattr(array_predictor, "COMPILE_MIN_BATCHES", threshold)

    def _run(batch_size: int) -> None:
        monkeypatch.setattr(
            array_predictor, "find_batch_size", lambda *a, **k: batch_size
        )
        predictor = _make_predictor(
            tiny_3d_model(), "cuda", amp="off", single_batch_mode=False
        )
        assert predictor.batch_size == batch_size
        predictor(dataset)

    _run(2)  # ceil(n_patches / 2) batches < threshold
    assert calls == [], "compile decision used patches instead of batches"

    _run(1)  # n_patches batches >= threshold
    assert len(calls) == 1, "compile was not enabled at batch size 1"


def test_compile_explicit_override_on_cpu(monkeypatch):
    """`compile=True` forces compilation on CPU; `compile=False` forces eager."""
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))

    forced_calls: list = []

    def recording_passthrough(model, *args, **kwargs):
        forced_calls.append(model)
        return model  # passthrough: no real inductor compile

    # threshold far above the dataset's batch count: only the override matters
    monkeypatch.setattr(array_predictor, "COMPILE_MIN_BATCHES", 10**6)
    monkeypatch.setattr(torch, "compile", recording_passthrough)
    pmaps = _predict(tiny_3d_model(), dataset, "cpu", compile=True)

    assert pmaps.shape == (1, 16, 96, 96)
    assert len(forced_calls) == 1, "compile=True must force a single compile call"

    eager_calls: list = []

    def eager_recording(model, *args, **kwargs):
        eager_calls.append(model)
        return model  # passthrough: no real inductor compile

    # threshold of 1 would trigger auto compilation on CUDA
    monkeypatch.setattr(array_predictor, "COMPILE_MIN_BATCHES", 1)
    monkeypatch.setattr(torch, "compile", eager_recording)
    pmaps = _predict(tiny_3d_model(), dataset, "cpu", compile=False)

    assert pmaps.shape == (1, 16, 96, 96)
    assert eager_calls == [], "compile=False must force eager execution"


@requires_cuda
def test_compile_wraps_model_and_matches_eager(monkeypatch, caplog):
    """`compile=True` is the explicit override; the math must match eager."""
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
    with caplog.at_level("WARNING"):
        pmaps_compiled = _run(compile=True)

    assert len(calls) == 1, "torch.compile was not used for compile=True"
    assert not any(
        "torch.compile failed" in record.getMessage() for record in caplog.records
    ), "torch.compile failed and silently fell back to eager"
    assert pmaps_compiled.dtype == np.float32
    assert pmaps_compiled.flags.c_contiguous
    np.testing.assert_allclose(pmaps_compiled, pmaps_eager, rtol=1e-3, atol=1e-3)


def test_compile_degrades_gracefully_on_first_forward_failure(monkeypatch, caplog):
    """`compile=True` forces compilation even on CPU; a failing compiled forward
    falls back to eager with a clear warning (no CUDA required)."""

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


def test_compile_is_idempotent_across_calls(monkeypatch):
    """Repeated `__call__` on one predictor must not retry `torch.compile`."""
    calls: list = []

    def recording_passthrough(model, *args, **kwargs):
        calls.append(model)
        return model  # passthrough: no real inductor compile

    monkeypatch.setattr(torch, "compile", recording_passthrough)
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))
    predictor = _make_predictor(tiny_3d_model(), "cpu", amp="off", compile=True)

    first = predictor(dataset)
    second = predictor(dataset)

    assert len(calls) == 1, "torch.compile must run exactly once per predictor"
    assert first.shape == second.shape == (1, 16, 96, 96)


def test_compile_failure_is_not_retried_on_later_calls(monkeypatch, caplog):
    """A graceful first-forward fallback is remembered: no compile retry."""
    calls: list = []

    class FailingCompiled(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            raise RuntimeError("inductor: fatal error: cuda.h: No such file")

    def failing_compile(model, *args, **kwargs):
        calls.append(model)
        return FailingCompiled(model)

    monkeypatch.setattr(torch, "compile", failing_compile)
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))
    predictor = _make_predictor(tiny_3d_model(), "cpu", amp="off", compile=True)

    with caplog.at_level("WARNING"):
        first = predictor(dataset)
        second = predictor(dataset)

    assert first.shape == second.shape == (1, 16, 96, 96)
    assert np.isfinite(second).all()
    assert len(calls) == 1, "a known-failing compile must not be retried"


def test_unet_prediction_forwards_tuning_options(tmp_path, monkeypatch):
    """`unet_prediction` shares the tuning options with the sizing probe and predictor."""
    calls: dict = {}

    def fake_find_feasible(*args, **kwargs):
        calls["probe_amp"] = kwargs.get("amp", "auto")
        calls["probe_channels_last_3d"] = kwargs.get("channels_last_3d")
        return (8, 64, 64), (2, 8, 8)

    class FakePredictor:
        def __init__(self, **kwargs):
            calls["amp"] = kwargs.get("amp")
            calls["channels_last_3d"] = kwargs.get("channels_last_3d")
            calls["has_compile_arg"] = "compile" in kwargs

        def __call__(self, dataset):
            return np.zeros((1,) + dataset.raw.shape, dtype="float32")

    monkeypatch.setattr(
        prediction_module, "find_feasible_patch_and_halo_shapes", fake_find_feasible
    )
    monkeypatch.setattr(prediction_module, "ArrayPredictor", FakePredictor)

    config_path, weights_path = save_model_files(tmp_path)

    pmaps = unet_prediction(
        raw=raw_volume(),
        input_layout="ZYX",
        model_name=None,
        model_id=None,
        config_path=config_path,
        model_weights_path=weights_path,
        device="cpu",
        disable_tqdm=True,
        amp="off",
        channels_last_3d=True,
    )

    assert calls == {
        "probe_amp": "off",
        "probe_channels_last_3d": True,
        "amp": "off",
        "channels_last_3d": True,
        "has_compile_arg": False,
    }
    assert pmaps.shape == (1, 16, 96, 96)


def test_unet_prediction_has_no_compile_argument():
    """Compile is decided by `ArrayPredictor`; callers cannot pass it."""
    assert "compile" not in inspect.signature(unet_prediction).parameters


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
