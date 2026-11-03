"""AMP (`amp: auto|off`) behavior of inference: precision, outputs, stitching.

`unet_prediction`/`ArrayPredictor` must run in the configured precision mode
(same switch as training and the memory probes), return fp32 probability maps
whatever the compute precision, and produce equivalent outputs under
`amp: auto` and `amp: off` — including the halo removal and overlapping-patch
stitching, which must be unaffected by the precision change.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from panseg.functionals.precision import AmpMode
from panseg.functionals.prediction import prediction as prediction_module
from panseg.functionals.prediction import unet_prediction
from panseg.functionals.prediction.utils.array_predictor import ArrayPredictor
from panseg.functionals.training.model import UNet3D
from tests.functionals.prediction.helpers import (
    MODEL_CONFIG,
    make_dataset,
    raw_volume,
    requires_cuda,
    tiny_3d_model,
)


@requires_cuda
@pytest.mark.parametrize("amp, enabled", [("auto", True), ("off", False)])
def test_array_predictor_runs_in_the_resolved_precision_mode(amp, enabled):
    """Sizing probe and prediction loop run under the configured mode."""
    model = tiny_3d_model()
    seen: list[bool] = []
    model.register_forward_hook(
        lambda module, inputs, output: seen.append(torch.is_autocast_enabled("cuda"))
    )
    dataset = make_dataset(raw_volume(), patch=(8, 64, 64), halo=(2, 8, 8))

    predictor = ArrayPredictor(
        model=model,
        in_channels=1,
        out_channels=1,
        device="cuda",
        patch=(8, 64, 64),
        patch_halo=(2, 8, 8),
        single_batch_mode=True,
        headless=False,
        amp=amp,
    )
    pmaps = predictor(dataset)

    assert seen, "no forward pass ran"
    assert all(state is enabled for state in seen)
    assert pmaps.dtype == np.float32
    assert pmaps.shape == (1, 16, 96, 96)


@requires_cuda
def test_array_predictor_outputs_close_between_amp_modes():
    """bf16 inference must reproduce fp32 inference within bf16 tolerance.

    The prediction uses overlapping patches (stride = patch // 2) and a halo,
    so this also pins the halo removal and patch stitching under autocast.
    """
    raw = raw_volume()
    patch, halo = (8, 64, 64), (2, 8, 8)
    dataset = make_dataset(raw, patch=patch, halo=halo)
    state = tiny_3d_model().state_dict()

    def _predict(amp: AmpMode) -> np.ndarray:
        model = tiny_3d_model()
        model.load_state_dict(state)
        predictor = ArrayPredictor(
            model=model,
            in_channels=1,
            out_channels=1,
            device="cuda",
            patch=patch,
            patch_halo=halo,
            single_batch_mode=True,
            headless=False,
            amp=amp,
        )
        return predictor(dataset)

    pmaps_off = _predict("off")
    pmaps_auto = _predict("auto")

    assert pmaps_auto.dtype == np.float32
    assert pmaps_off.shape == pmaps_auto.shape == (1, 16, 96, 96)
    assert np.isfinite(pmaps_auto).all()
    np.testing.assert_allclose(pmaps_auto, pmaps_off, rtol=5e-2, atol=5e-2)


@pytest.fixture()
def dummy_model_files(tmp_path: Path):
    """Small local UNet3D saved like a zoo model (config + weights)."""
    torch.manual_seed(0)
    model = UNet3D(in_channels=1, out_channels=1, final_sigmoid=True, f_maps=[8, 16])
    weights_path = tmp_path / "best_checkpoint.pytorch"
    torch.save(model.state_dict(), weights_path)
    config_path = tmp_path / "config_train.yml"
    config_path.write_text(MODEL_CONFIG)
    return config_path, weights_path


def _predict_unet(model_files, raw, amp):
    return unet_prediction(
        raw=raw,
        input_layout="ZYX",
        model_name=None,
        model_id=None,
        patch=(8, 64, 64),
        patch_halo=(2, 8, 8),
        config_path=model_files[0],
        model_weights_path=model_files[1],
        device="cpu",
        disable_tqdm=True,
        amp=amp,
    )


def test_unet_prediction_amp_modes_agree_on_cpu(dummy_model_files):
    """On CPU both modes resolve to fp32 and must agree exactly; output is fp32."""
    raw = raw_volume()

    pmaps_off = _predict_unet(dummy_model_files, raw, amp="off")
    pmaps_auto = _predict_unet(dummy_model_files, raw, amp="auto")

    assert pmaps_auto.dtype == np.float32
    assert pmaps_auto.shape == (1, 16, 96, 96)
    np.testing.assert_array_equal(pmaps_auto, pmaps_off)


def test_unet_prediction_flows_amp_to_sizing_and_predictor(
    dummy_model_files, monkeypatch
):
    """The patch-sizing probe and the predictor share the configured mode."""
    calls: dict[str, str] = {}

    def fake_find_feasible(
        model,
        in_channels,
        full_volume_shape,
        min_halo_shape,
        device,
        max_attempts=5,
        both_sides=False,
        amp="auto",
    ):
        calls["probe"] = amp
        return (8, 64, 64), (2, 8, 8)

    class FakePredictor:
        def __init__(self, **kwargs):
            calls["predictor"] = kwargs.get("amp", "auto")

        def __call__(self, dataset):
            return np.zeros((1,) + dataset.raw.shape, dtype="float32")

    monkeypatch.setattr(
        prediction_module, "find_feasible_patch_and_halo_shapes", fake_find_feasible
    )
    monkeypatch.setattr(prediction_module, "ArrayPredictor", FakePredictor)

    pmaps = unet_prediction(
        raw=raw_volume(),
        input_layout="ZYX",
        model_name=None,
        model_id=None,
        config_path=dummy_model_files[0],
        model_weights_path=dummy_model_files[1],
        device="cpu",
        disable_tqdm=True,
        amp="off",
    )

    assert calls == {"probe": "off", "predictor": "off"}
    assert pmaps.shape == (1, 16, 96, 96)
