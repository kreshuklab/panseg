"""End-to-end UNet prediction on Apple silicon MPS devices.

Only runs where torch.backends.mps.is_available() is True; guards the device
plumbing (model placement + OOM probe) that previously broke prediction on
Apple silicon, see https://github.com/kreshuklab/panseg/issues/385 and #552.
"""

import numpy as np
import pytest
import torch

from panseg.functionals.prediction.prediction import unet_prediction
from panseg.functionals.training.model import UNet3D

IS_MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


@pytest.mark.skipif(
    not IS_MPS_AVAILABLE, reason="Requires an Apple silicon GPU (MPS) device."
)
def test_unet_prediction_with_mps_device(tmp_path):
    model = UNet3D(in_channels=1, out_channels=1, final_sigmoid=True, f_maps=[8, 16])
    weights_path = tmp_path / "best_checkpoint.pytorch"
    torch.save(model.state_dict(), weights_path)

    config_path = tmp_path / "config_train.yml"
    config_path.write_text(
        """model:
  name: UNet3D
  in_channels: 1
  out_channels: 1
  final_sigmoid: true
  layer_order: gcr
  f_maps: [8, 16]
  num_groups: 4
"""
    )

    raw = np.random.rand(16, 64, 64).astype("float32")  # ZYX
    pmap = unet_prediction(
        raw=raw,
        input_layout="ZYX",
        model_name=None,
        model_id=None,
        patch=None,
        patch_halo=(0, 0, 0),
        config_path=config_path,
        model_weights_path=weights_path,
        device="mps",
        disable_tqdm=True,
    )

    assert pmap.shape[-3:] == raw.shape
    assert np.all(np.isfinite(pmap))
