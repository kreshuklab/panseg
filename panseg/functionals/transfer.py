"""Host-to-device transfer of loader batches.

``tensor.to(device, non_blocking=True)`` only lets the CPU continue before the
copy completes for *pinned* (page-locked) host memory. On ordinary memory the
flag is a silent no-op — the CPU still blocks for the whole copy (measured:
0.396 ms vs 0.443 ms for a 7.8 MiB tensor, see ``.scratch/h2d-results.md``).
The flag is therefore derived from the tensor itself, never from the device
type: a loader that does not pin its batches (``pin_memory=False``, the CPU
loader, or a plain list of tensors) gets a blocking copy.
"""

import torch


def to_device(tensor: torch.Tensor, device) -> torch.Tensor:
    """Move a host tensor to `device`, copying asynchronously if it is pinned.

    A non-blocking copy is ordered before the CUDA operations that follow it on
    the same stream, so it is safe to use the result immediately.

    Args:
        tensor: the tensor to move (pinned or not).
        device: the destination device, e.g. ``"cuda:0"`` or ``"cpu"``.

    Returns:
        The tensor on `device` (the same object if it is already there).
    """
    return tensor.to(device, non_blocking=tensor.is_pinned())
