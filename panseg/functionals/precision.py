"""Precision-mode helpers shared by training, inference, and device probes.

The user-facing ``amp: auto|off`` switch is resolved to a torch context manager
in exactly one place — :func:`amp_context` — and every compute path (training,
inference, and the GPU memory probes in ``size_finder``) runs inside it. The
memory probes therefore always size patches/batches for the same precision the
model will actually run in: sizing and execution cannot diverge.
"""

from contextlib import AbstractContextManager, nullcontext
from typing import Literal, cast

import torch

AmpMode = Literal["auto", "off"]
AmpSetting = AmpMode | bool
"""Accepted `amp` spellings at API boundaries: the modes `AmpMode` plus the
YAML-boolean aliases that `validate_amp` resolves (a bare ``on``/``true`` reads
as True -> ``"auto"``, a bare ``off``/``false`` as False -> ``"off"``)."""
AMP_MODES: tuple[str, ...] = ("auto", "off")


def validate_amp(amp: AmpSetting) -> AmpMode:
    """Check that `amp` is a supported mode and return it resolved.

    Accepted spellings: the strings ``"auto"`` and ``"off"``, plus the
    YAML-boolean aliases — YAML parses a bare ``on``/``true`` as True and a bare
    ``off``/``false`` as False, which resolve to ``"auto"`` and ``"off"``
    respectively.

    Args:
        amp: the configured precision mode (see the accepted spellings above).

    Returns:
        The resolved mode, narrowed to `AmpMode`.

    Raises:
        ValueError: if `amp` is not one of the accepted spellings.
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


def bf16_autocast_enabled(amp: AmpSetting, device: str | torch.device) -> bool:
    """Whether `amp_context(amp, device)` resolves to bf16 autocast.

    `amp_context` is built on this predicate; callers that branch on the
    resolved precision (e.g. the training ``torch.compile`` gate) use it too.

    Args:
        amp: precision mode (see `validate_amp`).
        device: device the compute will run on.

    Returns:
        True only for ``"auto"`` on a CUDA device where bf16 is supported.

    Raises:
        ValueError: if `amp` is not a supported mode.
    """
    return (
        validate_amp(amp) == "auto"
        and _is_cuda_device(device)
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )


def amp_context(amp: AmpSetting, device: str | torch.device) -> AbstractContextManager:
    """Resolve ``amp: auto|off`` to the torch context manager for `device`.

    ``"auto"`` runs bf16 autocast on CUDA devices with bf16 support and
    silently falls back to plain fp32 everywhere else (CPU, MPS, older CUDA
    hardware). ``"off"`` always runs fp32. All spellings accepted by
    `validate_amp` (including the YAML-boolean aliases) work here.

    Args:
        amp: the configured precision mode (see `validate_amp` for the
            accepted spellings).
        device: device the compute will run on.

    Returns:
        A context manager: bf16 `torch.autocast` or a no-op.

    Raises:
        ValueError: if `amp` is not a supported mode.
    """
    if bf16_autocast_enabled(amp, device):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()
