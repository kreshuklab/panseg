"""Regression tests for issue #146: an AssertionError
("Sample size has to be bigger than the patch size") when the image is smaller
than the patch size in any dimension."""

import numpy as np
import pytest
import torch

from panseg.functionals.prediction import unet_prediction
from panseg.functionals.prediction.utils import size_finder
from panseg.functionals.training.model import UNet3D

MODEL_CONFIG = """model:
  name: UNet3D
  in_channels: 1
  out_channels: 1
  final_sigmoid: true
  layer_order: gcr
  f_maps: [8, 16]
  num_groups: 4
"""


@pytest.fixture()
def dummy_model(tmp_path):
    """Small local UNet3D loaded via config path, so no model zoo download is needed."""
    model = UNet3D(in_channels=1, out_channels=1, final_sigmoid=True, f_maps=[8, 16])
    weights_path = tmp_path / "best_checkpoint.pytorch"
    torch.save(model.state_dict(), weights_path)
    config_path = tmp_path / "config_train.yml"
    config_path.write_text(MODEL_CONFIG)
    return config_path, weights_path


def test_unet_prediction_auto_patch_never_exceeds_sample(dummy_model, monkeypatch):
    # Anisotropic ZYX volume: Z is smaller than the probed maximum patch and Y and X
    # differ strongly. The sqrt redistribution of the voxel budget used to push the
    # derived patch past the sample size in Y and crash SliceBuilder with an
    # AssertionError. The probed max patch (64, 64, 64) yields sqrt(262144 // 16) = 128,
    # which is bigger than the 96-voxel Y dimension without the fix.
    monkeypatch.setattr(
        size_finder,
        "probe_max_patch_shape",
        lambda model, in_channels, device: (64, 64, 64),
    )
    raw = np.random.rand(16, 96, 256).astype("float32")

    pmap = unet_prediction(
        raw=raw,
        input_layout="ZYX",
        model_name=None,
        model_id=None,
        patch=None,
        patch_halo=(0, 0, 0),
        config_path=dummy_model[0],
        model_weights_path=dummy_model[1],
        device="cpu",
        disable_tqdm=True,
    )

    assert pmap.shape[-3:] == raw.shape
    assert np.all(np.isfinite(pmap))


def test_unet_prediction_rejects_manual_patch_bigger_than_sample(dummy_model):
    raw = np.random.rand(8, 64, 64).astype("float32")

    with pytest.raises(ValueError, match="bigger than the sample shape"):
        unet_prediction(
            raw=raw,
            input_layout="ZYX",
            model_name=None,
            model_id=None,
            patch=(32, 64, 64),
            patch_halo=(0, 0, 0),
            config_path=dummy_model[0],
            model_weights_path=dummy_model[1],
            device="cpu",
            disable_tqdm=True,
        )


def test_unet_prediction_cpu_patch_follows_free_memory(dummy_model, monkeypatch):
    # Little free memory gives the smallest CPU patch, not the former fixed 256^3
    monkeypatch.setattr(size_finder, "_available_memory", lambda: 100_000_000)
    seen = []
    derive = size_finder.derive_patch_and_halo_shapes

    def spy(full_volume_shape, max_patch_shape, *args, **kwargs):
        seen.append(tuple(max_patch_shape))
        return derive(full_volume_shape, max_patch_shape, *args, **kwargs)

    monkeypatch.setattr(size_finder, "derive_patch_and_halo_shapes", spy)
    raw = np.random.rand(16, 256, 256).astype("float32")

    pmap = unet_prediction(
        raw=raw,
        input_layout="ZYX",
        model_name=None,
        model_id=None,
        patch=None,
        patch_halo=(0, 0, 0),
        config_path=dummy_model[0],
        model_weights_path=dummy_model[1],
        device="cpu",
        disable_tqdm=True,
    )

    assert seen == [(64, 64, 64)]
    assert pmap.shape[-3:] == raw.shape
    assert np.all(np.isfinite(pmap))
