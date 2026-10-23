"""Regression tests for `unet_prediction`.

- Issue #146: an AssertionError ("Sample size has to be bigger than the patch size")
  when the image is smaller than the patch size in any dimension.
- Issue #612: 2D models on 3D input got patches deeper than one Z slice, and 3D
  models on thin images failed inside the pooling layers.
"""

import numpy as np
import pytest
import torch

from panseg.functionals.prediction import unet_prediction
from panseg.functionals.prediction.utils import size_finder
from panseg.functionals.training.model import UNet2D, UNet3D

MODEL_CONFIG = """model:
  name: UNet3D
  in_channels: 1
  out_channels: 1
  final_sigmoid: true
  layer_order: gcr
  f_maps: [8, 16]
  num_groups: 4
"""


MODEL_CONFIG_2D = """model:
  name: UNet2D
  in_channels: 1
  out_channels: 2
  final_sigmoid: true
  layer_order: gcr
  f_maps: [8, 16]
  num_groups: 4
"""

MODEL_CONFIG_3D_DEEP = """model:
  name: UNet3D
  in_channels: 1
  out_channels: 1
  final_sigmoid: true
  layer_order: gcr
  f_maps: [8, 16, 32, 64]
  num_groups: 4
"""


def _save_model(tmp_path, model, config):
    weights_path = tmp_path / "best_checkpoint.pytorch"
    torch.save(model.state_dict(), weights_path)
    config_path = tmp_path / "config_train.yml"
    config_path.write_text(config)
    return config_path, weights_path


@pytest.fixture()
def dummy_model_2d(tmp_path):
    """Small local UNet2D with nuclei-style two-channel output."""
    model = UNet2D(in_channels=1, out_channels=2, final_sigmoid=True, f_maps=[8, 16])
    return _save_model(tmp_path, model, MODEL_CONFIG_2D)


@pytest.fixture()
def dummy_model_3d_deep(tmp_path):
    """Small local UNet3D with 4 levels, so it needs at least 8 Z slices."""
    model = UNet3D(
        in_channels=1, out_channels=1, final_sigmoid=True, f_maps=[8, 16, 32, 64]
    )
    return _save_model(tmp_path, model, MODEL_CONFIG_3D_DEEP)


def _predict(model_files, raw, layout, patch=None, patch_halo=None):
    return unet_prediction(
        raw=raw,
        input_layout=layout,
        model_name=None,
        model_id=None,
        patch=patch,
        patch_halo=patch_halo,
        config_path=model_files[0],
        model_weights_path=model_files[1],
        device="cpu",
        disable_tqdm=True,
    )


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


@pytest.mark.parametrize(
    ("raw_shape", "layout", "probed_max"),
    [
        ((8, 128, 128), "ZYX", None),  # real CPU probe, (1, 1024, 1024)
        ((20, 128, 128), "ZYX", (1, 256, 256)),
        ((600, 100), "YX", (1, 128, 128)),  # long image, used to give a 12 px strip
    ],
)
def test_unet_prediction_2d_model_on_3d_input(
    dummy_model_2d, monkeypatch, raw_shape, layout, probed_max
):
    if probed_max is not None:
        monkeypatch.setattr(
            size_finder,
            "probe_max_patch_shape",
            lambda model, in_channels, device: probed_max,
        )
    raw = np.random.rand(*raw_shape).astype("float32")

    pmap = _predict(dummy_model_2d, raw, layout)

    zyx = raw_shape if layout == "ZYX" else (1, *raw_shape)
    assert pmap.shape == (2, *zyx)
    assert np.all(np.isfinite(pmap))


def test_unet_prediction_2d_model_auto_patch_drops_z_halo(dummy_model_2d):
    raw = np.random.rand(4, 128, 128).astype("float32")

    pmap = _predict(dummy_model_2d, raw, "ZYX", patch_halo=(1, 8, 8))

    assert pmap.shape == (2, 4, 128, 128)


def test_unet_prediction_2d_model_manual_patch(dummy_model_2d):
    raw = np.random.rand(4, 128, 128).astype("float32")

    pmap = _predict(dummy_model_2d, raw, "ZYX", patch=(1, 64, 64), patch_halo=(0, 8, 8))

    assert pmap.shape == (2, 4, 128, 128)
    assert np.all(np.isfinite(pmap))


@pytest.mark.parametrize(
    ("patch", "patch_halo", "match"),
    [
        ((2, 64, 64), (0, 0, 0), "has 2 Z slices"),
        ((1, 64, 64), (1, 8, 8), "Set the Z halo to 0"),
    ],
)
def test_unet_prediction_2d_model_rejects_z_patch_or_halo(
    dummy_model_2d, patch, patch_halo, match
):
    raw = np.random.rand(4, 128, 128).astype("float32")

    with pytest.raises(ValueError, match=match):
        _predict(dummy_model_2d, raw, "ZYX", patch=patch, patch_halo=patch_halo)


@pytest.mark.parametrize(
    ("raw_shape", "layout", "patch", "patch_halo", "match"),
    [
        (
            (4, 64, 64),
            "ZYX",
            None,
            (0, 0, 0),
            "needs at least 8 Z slices, but the image has 4",
        ),
        ((64, 64), "YX", None, (0, 0, 0), "the image has 1"),
        ((16, 64, 64), "ZYX", (4, 64, 64), (1, 0, 0), "has 6"),
    ],
)
def test_unet_prediction_3d_model_rejects_thin_image(
    dummy_model_3d_deep, raw_shape, layout, patch, patch_halo, match
):
    raw = np.random.rand(*raw_shape).astype("float32")

    with pytest.raises(ValueError, match=match):
        _predict(dummy_model_3d_deep, raw, layout, patch=patch, patch_halo=patch_halo)


@pytest.mark.parametrize(
    ("raw_shape", "patch", "patch_halo"),
    [((8, 64, 64), None, (0, 0, 0)), ((16, 64, 64), (4, 64, 64), (2, 0, 0))],
)
def test_unet_prediction_3d_model_accepts_min_depth(
    dummy_model_3d_deep, raw_shape, patch, patch_halo
):
    raw = np.random.rand(*raw_shape).astype("float32")

    pmap = _predict(dummy_model_3d_deep, raw, "ZYX", patch=patch, patch_halo=patch_halo)

    assert pmap.shape == (1, *raw_shape)
    assert np.all(np.isfinite(pmap))
