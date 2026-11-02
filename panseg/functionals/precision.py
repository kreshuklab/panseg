"""Precision-mode helpers shared by training, inference, and device probes.

The user-facing ``amp: auto|off`` switch is resolved to a torch context manager
in exactly one place — :func:`amp_context` — and every compute path (training,
inference, and the GPU memory probes in ``size_finder``) runs inside it. The
memory probes therefore always size patches/batches for the same precision the
model will actually run in: sizing and execution cannot diverge.
"""

import contextlib
from contextlib import AbstractContextManager, nullcontext
from typing import Literal, cast

import torch

AmpMode = Literal["auto", "off"]
AMP_MODES: tuple[str, ...] = ("auto", "off")


def validate_amp(amp: str | bool) -> AmpMode:
    """Check that `amp` is a supported mode and return it typed.

    YAML parses the bare mode names as booleans (``off`` -> False, ``on`` ->
    True); those spellings map to ``"off"`` and ``"auto"`` respectively.

    Args:
        amp: the configured precision mode.

    Returns:
        The mode, narrowed to `AmpMode`.

    Raises:
        ValueError: if `amp` is not one of `AMP_MODES`.
    """
    if amp is False:  # YAML reads a bare `off` as a boolean
        amp = "off"
    elif amp is True:  # ... and a bare `on` likewise
        amp = "auto"
    if amp not in AMP_MODES:
        raise ValueError(f"Unknown amp mode {amp!r}. Valid modes: {AMP_MODES}.")
    return cast(AmpMode, amp)


def _is_cuda_device(device: str | torch.device) -> bool:
    return torch.device(device).type == "cuda"


def amp_context(amp: str | bool, device: str | torch.device) -> AbstractContextManager:
    """Resolve ``amp: auto|off`` to the torch context manager for `device`.

    ``"auto"`` runs bf16 autocast on CUDA devices with bf16 support and
    silently falls back to plain fp32 everywhere else (CPU, MPS, older CUDA
    hardware). ``"off"`` always runs fp32. YAML booleans are accepted for the
    bare spellings (``off`` -> ``"off"``, ``on`` -> ``"auto"``, see
    `validate_amp`).

    Args:
        amp: the configured precision mode (``str | bool``: YAML parses bare
            ``off``/``on`` spellings as booleans).
        device: device the compute will run on.

    Returns:
        A context manager: bf16 `torch.autocast` or a no-op.

    Raises:
        ValueError: if `amp` is not a supported mode.
    """
    amp = validate_amp(amp)
    if (
        amp == "auto"
        and _is_cuda_device(device)
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@contextlib.contextmanager
def cudnn_benchmark(enabled: bool = True):
    """Temporarily set `torch.backends.cudnn.benchmark`, restoring it after.

    cuDNN autotuning pays off where input shapes are fixed (the training and
    inference patch loops); the flag is global, so the previous value is
    restored when the fixed-shape loop ends.
    """
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = enabled
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous
