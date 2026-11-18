"""Regression tests for batch-size handling in `ArrayPredictor`.

Prediction used to pick a batch size from a single forward probe and then run
the whole loop with it. On large 2D full-slice patches the probe fits once, but
the repeated forwards of the prediction loop fragment the CUDA caching
allocator and a later iteration OOMs (issue #7). `ArrayPredictor` must recover
from that by falling back to a smaller batch instead of aborting the run.
"""

import numpy as np
import pytest
import torch
from torch import nn

from panseg.functionals.prediction.utils import array_predictor
from panseg.functionals.prediction.utils.array_dataset import ArrayDataset
from panseg.functionals.prediction.utils.array_predictor import ArrayPredictor
from panseg.functionals.prediction.utils.slice_builder import SliceBuilder
from panseg.functionals.training.model import UNet2D


class FailingForwardUNet2D(UNet2D):
    """Tiny 2D model whose forward raises a fake CUDA OOM on selected calls.

    Every other call returns a constant prediction, so the test never needs a
    real device or a specific amount of free GPU memory.
    """

    def __init__(self, oom_calls: set[int]):
        super().__init__(1, 1, f_maps=8, num_levels=2)
        self.oom_calls = oom_calls
        self.calls = 0
        self.seen_batch_sizes: list[int] = []

    def forward(self, x):
        call = self.calls
        self.calls += 1
        self.seen_batch_sizes.append(x.shape[0])
        if call in self.oom_calls:
            raise RuntimeError("CUDA out of memory. Tried to allocate 999.00 GiB.")
        return torch.ones(
            (x.shape[0], 1) + tuple(x.shape[-2:]), dtype=x.dtype, device=x.device
        )


def _dataset(n_slices: int = 32) -> ArrayDataset:
    raw = np.zeros((n_slices, 64, 64), dtype="float32")
    patch = (1, 64, 64)
    slice_builder = SliceBuilder(raw, None, patch, stride_shape=patch)
    return ArrayDataset(
        raw,
        slice_builder,
        augs=lambda patch: torch.from_numpy(patch)[None].float(),
        halo_shape=(0, 0, 0),
    )


def _predictor(model: nn.Module) -> ArrayPredictor:
    return ArrayPredictor(
        model=model,
        in_channels=1,
        out_channels=1,
        device="cpu",
        patch=(1, 64, 64),
        patch_halo=(0, 0, 0),
        single_batch_mode=False,
        headless=False,
        disable_tqdm=True,
    )


def test_array_predictor_falls_back_to_smaller_batch_on_oom(monkeypatch):
    # The probe accepted batch 16, the first batch succeeds, the second OOMs.
    # The partially accumulated first batch must not survive the restart.
    monkeypatch.setattr(array_predictor, "find_batch_size", lambda *a, **k: 16)
    model = FailingForwardUNet2D(oom_calls={1})
    predictor = _predictor(model)

    prediction = predictor(_dataset())

    assert predictor.batch_size == 8
    assert model.seen_batch_sizes == [16, 16, 8, 8, 8, 8]
    assert prediction.shape == (1, 32, 64, 64)
    # the model returns logits of 1, which the predictor activates (sigmoid)
    assert np.allclose(prediction, torch.sigmoid(torch.tensor(1.0)).item())


def test_array_predictor_names_patch_and_halo_when_batch_one_cannot_fit(monkeypatch):
    monkeypatch.setattr(array_predictor, "will_CUDA_OOM", lambda *a, **k: True)

    with pytest.raises(
        RuntimeError,
        match=r"patch shape \(1, 64, 64\) with halo \(0, 0, 0\).*batch size 1",
    ):
        ArrayPredictor(
            model=FailingForwardUNet2D(oom_calls=set()),
            in_channels=1,
            out_channels=1,
            device="cuda",
            patch=(1, 64, 64),
            patch_halo=(0, 0, 0),
            single_batch_mode=True,
            headless=False,
            disable_tqdm=True,
        )


def test_array_predictor_names_patch_halo_and_batch_when_nothing_fits(monkeypatch):
    # Every batch size OOMs, including 1: prediction must fail with a message
    # that identifies the shapes instead of leaking the raw CUDA error.
    monkeypatch.setattr(array_predictor, "find_batch_size", lambda *a, **k: 16)
    model = FailingForwardUNet2D(oom_calls=set(range(10)))
    predictor = _predictor(model)

    with pytest.raises(
        RuntimeError,
        match=r"patch\s+shape \(1, 64, 64\) with halo \(0, 0, 0\).*batch size 1",
    ):
        predictor(_dataset())

    assert model.seen_batch_sizes == [16, 8, 4, 2, 1]
