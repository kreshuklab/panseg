"""Shared helpers for the prediction test modules (probes, AMP, inference opts)."""

import numpy as np
import pytest
import torch

from panseg.functionals.prediction.utils.array_dataset import ArrayDataset
from panseg.functionals.prediction.utils.slice_builder import SliceBuilder
from panseg.functionals.training.augs import get_test_augmentations
from panseg.functionals.training.model import UNet2D, UNet3D

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for AMP tests"
)

MODEL_CONFIG = """model:
  name: UNet3D
  in_channels: 1
  out_channels: 1
  final_sigmoid: true
  layer_order: gcr
  f_maps: [8, 16]
  num_groups: 4
"""


def tiny_3d_model(in_channels: int = 1) -> UNet3D:
    """Small two-level UNet3D: fast probe/prediction forwards on any device."""
    return UNet3D(
        in_channels=in_channels,
        out_channels=1,
        final_sigmoid=True,
        f_maps=8,
        num_levels=2,
    )


def tiny_2d_model() -> UNet2D:
    """2D counterpart of `tiny_3d_model` for the 2D probe paths."""
    return UNet2D(in_channels=1, out_channels=1, f_maps=8, num_levels=2)


def make_dataset(
    raw: np.ndarray,
    patch: tuple[int, int, int],
    halo: tuple[int, int, int],
    multichannel: bool = False,
) -> ArrayDataset:
    """Raw ZYX/CZYX array -> ArrayDataset with overlapping patches (exercises stitching)."""
    stride = tuple(max(p // 2, 1) for p in patch)
    slice_builder = SliceBuilder(
        raw, label_dataset=None, patch_shape=patch, stride_shape=stride
    )
    return ArrayDataset(
        raw,
        slice_builder,
        get_test_augmentations(raw),
        halo_shape=halo,
        multichannel=multichannel,
        verbose_logging=False,
    )


def raw_volume() -> np.ndarray:
    """Deterministic small ZYX raw volume."""
    return np.random.default_rng(0).random((16, 96, 96)).astype("float32")
