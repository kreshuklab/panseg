"""Host-to-device transfer contract for loader batches.

`non_blocking=True` only lets the CPU return before the copy completes for
*pinned* (page-locked) host memory; on ordinary memory the flag is a silent
no-op (measured: the CPU still blocks for the whole copy, see
``.scratch/h2d-results.md``). The flag must therefore follow the tensor's own
pinned state, never the device type: a loader that does not pin its batches
still has to get a blocking copy.
"""

import pytest
import torch

from panseg.functionals.transfer import to_device
from tests.conftest import requires_cuda


class _RecordingTensor:
    """Minimal `torch.Tensor` stand-in recording the transfer flag."""

    def __init__(self, pinned: bool):
        self._pinned = pinned
        self.transfers: list[tuple[str, bool]] = []

    def is_pinned(self) -> bool:
        return self._pinned

    def to(self, device, non_blocking: bool = False):
        self.transfers.append((device, non_blocking))
        return self


@pytest.mark.parametrize("pinned", [True, False])
def test_to_device_derives_non_blocking_from_the_pinned_state(pinned):
    tensor = _RecordingTensor(pinned)

    to_device(tensor, "cuda")

    # `non_blocking=True` on unpinned memory does nothing: it must not be claimed
    assert tensor.transfers == [("cuda", pinned)]


@requires_cuda
def test_pinned_tensors_are_detected_and_move_to_the_gpu():
    tensor = torch.zeros(1, 1, 2, 2, 2).pin_memory()

    assert tensor.is_pinned() is True
    moved = to_device(tensor, "cuda")

    assert moved.is_cuda
    torch.testing.assert_close(moved.cpu(), tensor)
