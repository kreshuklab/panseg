"""Automatic `torch.compile` for long bf16 training runs.

Compile is a measured 1.3-1.5x win on the default bf16 path with 27% lower
peak memory, but only after the cold compile cost amortizes and never on fp32
(TF32 numerics + a 5-8% gain). The trainer keeps `self.model` plain so
checkpoints, optimizer identities, DataParallel and `isinstance(UNet2D)` are
unchanged; only forward passes go through the compiled handle.
"""

from pathlib import Path

import pytest
import torch
from torch import nn

from panseg.functionals.training import trainer as trainer_module
from panseg.functionals.training.losses import DiceLoss
from panseg.functionals.training.model import UNet3D
from panseg.functionals.training.trainer import (
    TRAIN_COMPILE_MIN_ITERS,
    UNetTrainer,
)


class _DataParallelLike(nn.DataParallel):
    """nn.DataParallel stand-in that skips the GPU-dependent __init__."""

    def __init__(self, module):
        torch.nn.Module.__init__(self)
        self.module = module

    def forward(self, *inputs):
        return self.module(*inputs)


class _CountedHandle(nn.Module):
    """Compiled-handle stand-in that records calls and forwards to the model."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.module(x)


class _CompileRecorder:
    def __init__(self, factory=None):
        self.calls: list[nn.Module] = []
        self._factory = factory or (lambda module: module)

    def __call__(self, module, *args, **kwargs):
        self.calls.append(module)
        return self._factory(module)


def _make_trainer(
    tmp_path: Path,
    monkeypatch,
    *,
    device: str = "cuda",
    max_num_iterations: int = TRAIN_COMPILE_MIN_ITERS,
    amp: str = "auto",
    bf16: bool = True,
    model: nn.Module | None = None,
    recorder: _CompileRecorder | None = None,
):
    recorder = recorder or _CompileRecorder()
    monkeypatch.setattr(torch, "compile", recorder)
    monkeypatch.setattr(
        trainer_module, "bf16_autocast_enabled", lambda amp, device: bf16
    )
    model = model if model is not None else UNet3D(1, 1, f_maps=[4, 4])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=10
        ),
        loss_criterion=DiceLoss(),
        loaders={"train": [None], "val": []},
        checkpoint_dir=tmp_path,
        max_num_iterations=max_num_iterations,
        device=device,
        amp=amp,
    )
    return trainer, recorder, model


@pytest.mark.parametrize(
    ("device", "bf16", "max_num_iterations", "wrap_dp", "expected"),
    [
        ("cpu", True, 5000, False, False),  # CPU never compiles
        ("cuda", False, 5000, False, False),  # fp32 never compiles
        ("cuda", True, TRAIN_COMPILE_MIN_ITERS - 1, False, False),  # too short
        ("cuda", True, TRAIN_COMPILE_MIN_ITERS, False, True),  # gate met
        ("cuda", True, 5000, True, False),  # DataParallel is excluded
    ],
)
def test_compile_gate(
    tmp_path, monkeypatch, device, bf16, max_num_iterations, wrap_dp, expected
):
    model = UNet3D(1, 1, f_maps=[4, 4])
    if wrap_dp:
        model = _DataParallelLike(model)

    trainer, recorder, _ = _make_trainer(
        tmp_path,
        monkeypatch,
        device=device,
        max_num_iterations=max_num_iterations,
        bf16=bf16,
        model=model,
    )

    assert bool(recorder.calls) is expected
    assert (trainer._compiled_model is not None) is expected


def test_compile_is_off_for_amp_off_even_on_cuda(tmp_path, monkeypatch):
    """The real resolved-precision helper decides, not just the device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    recorder = _CompileRecorder()
    monkeypatch.setattr(torch, "compile", recorder)
    model = UNet3D(1, 1, f_maps=[4, 4])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=10
        ),
        loss_criterion=DiceLoss(),
        loaders={"train": [None], "val": []},
        checkpoint_dir=tmp_path,
        max_num_iterations=5000,
        device="cuda:0",
        amp="off",
    )

    assert trainer._compiled_model is None
    assert recorder.calls == []


def test_compiled_trainer_keeps_plain_model_and_checkpoint_keys(tmp_path, monkeypatch):
    handle = _CountedHandle
    trainer, recorder, model = _make_trainer(
        tmp_path, monkeypatch, recorder=_CompileRecorder(handle)
    )
    assert recorder.calls == [model]
    assert trainer.model is model
    assert trainer._compiled_model is not model

    keys_before = set(model.state_dict())
    parameter_before = next(iter(model.parameters()))

    # the structural checks do not need a GPU; run the forward on CPU
    trainer.device = "cpu"
    trainer.amp = "off"
    _input = torch.rand(1, 1, 8, 8, 8)
    output, loss = trainer._forward_pass(_input, _input)

    assert trainer._compiled_model.calls == 1
    assert output.shape == _input.shape
    assert torch.isfinite(loss)
    assert set(model.state_dict()) == keys_before
    assert next(iter(trainer.optimizer.param_groups[0]["params"])) is parameter_before


def test_validation_uses_the_compiled_handle(tmp_path, monkeypatch):
    trainer, _, model = _make_trainer(
        tmp_path, monkeypatch, recorder=_CompileRecorder(_CountedHandle)
    )
    trainer.device = "cpu"
    trainer.amp = "off"
    _input = torch.rand(1, 1, 8, 8, 8)
    trainer.loaders = {"train": [], "val": [(_input, _input)]}

    loss = trainer.validate()

    assert trainer._compiled_model.calls == 1
    assert torch.isfinite(torch.tensor(loss))


class _FailingHandle(nn.Module):
    def __init__(self, message):
        super().__init__()
        self.message = message
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        raise RuntimeError(self.message)


def _forward_trainer(model, compiled, device="cpu", amp="off"):
    trainer = UNetTrainer.__new__(UNetTrainer)
    trainer.model = model
    trainer.loss_criterion = DiceLoss()
    trainer.amp = amp
    trainer.device = device
    trainer._compiled_model = compiled
    return trainer


def test_first_forward_failure_falls_back_to_eager_permanently(caplog):
    model = UNet3D(1, 1, f_maps=[4, 4])
    handle = _FailingHandle("triton failed to compile")
    trainer = _forward_trainer(model, handle)
    _input = torch.rand(1, 1, 8, 8, 8)

    output, loss = trainer._forward_pass(_input, _input)

    assert handle.calls == 1
    assert trainer._compiled_model is None
    assert output.shape == _input.shape
    assert "falling back to eager" in caplog.text

    # eager from now on; the failed handle is never retried
    trainer._forward_pass(_input, _input)
    assert handle.calls == 1


def test_first_forward_oom_propagates_without_fallback():
    model = UNet3D(1, 1, f_maps=[4, 4])
    handle = _FailingHandle("CUDA out of memory. Tried to allocate 2.00 GiB.")
    trainer = _forward_trainer(model, handle, device="cuda:0")
    _input = torch.rand(1, 1, 8, 8, 8)

    with pytest.raises(RuntimeError, match="out of memory"):
        trainer._forward_pass(_input, _input)

    assert handle.calls == 1
    assert trainer._compiled_model is handle
