"""Golden fence for the logits/probabilities output contract (WP1 double sigmoid).

`tests/resources/output_contract_golden.pt` holds the externally visible
outputs captured from the code *before* `forward()` started returning logits
(generator: `/tmp/opencode/gen_golden_output_contract.py` at the time). They
pin the numerical fence of the output-contract change: whatever happens
internally, `unet_prediction` probability maps, the embedding path and the
model's probabilities output must reproduce those values, and the training
path must see exactly the logits it saw before.

A tolerance of 1e-5 relative guards against benign float drift across torch
builds; the double-sigmoid bug itself moves outputs by ~0.3 absolute.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from panseg.functionals.prediction import unet_prediction
from panseg.functionals.prediction.utils.array_predictor import ArrayPredictor
from panseg.functionals.training.model import UNet2D, UNet3D, UNet3DProbabilities
from tests.functionals.prediction.conftest import MODEL_CONFIG, make_dataset

GOLDEN_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "output_contract_golden.pt"
)
RTOL = 1e-5
ATOL = 1e-6


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN_PATH, map_location="cpu", weights_only=True)


def make_3d(final_sigmoid: bool, out_channels: int = 1) -> UNet3D:
    """The exact architecture the golden values were generated with."""
    return UNet3D(
        in_channels=1,
        out_channels=out_channels,
        final_sigmoid=final_sigmoid,
        f_maps=[8, 16],
        num_groups=4,
        layer_order="gcr",
    )


def make_2d() -> UNet2D:
    """The exact 2D architecture the golden values were generated with."""
    return UNet2D(
        in_channels=1,
        out_channels=2,
        final_sigmoid=True,
        f_maps=[8, 16],
        num_groups=4,
        layer_order="gcr",
    )


def test_forward_matches_golden_logits_in_train_and_eval(golden):
    """forward() returns the old train-mode logits in *every* mode.

    This is the flipped pin: pre-change, eval-mode forward returned the golden
    probabilities (the double-sigmoid contract).
    """
    for make, state_key, input_key, logits_key in [
        (lambda: make_3d(True), "state_unet3d", "input_3d", "gold_logits_3d"),
        (
            lambda: make_3d(False, out_channels=2),
            "state_unet3d_softmax",
            "input_3d",
            "gold_logits_softmax_3d",
        ),
        (make_2d, "state_unet2d", "input_2d", "gold_logits_2d"),
    ]:
        model = make()
        model.load_state_dict(golden[state_key])
        for mode in ("train", "eval"):
            model.train(mode == "train")
            with torch.no_grad():
                out = model(golden[input_key])
            np.testing.assert_allclose(
                out.numpy(), golden[logits_key].numpy(), rtol=RTOL, atol=ATOL
            )


def test_predict_matches_golden_probabilities(golden):
    """predict() must reproduce exactly the old eval-mode probabilities."""
    for make, state_key, input_key, probs_key in [
        (lambda: make_3d(True), "state_unet3d", "input_3d", "gold_probs_3d"),
        (
            lambda: make_3d(False, out_channels=2),
            "state_unet3d_softmax",
            "input_3d",
            "gold_softmax_3d",
        ),
        (make_2d, "state_unet2d", "input_2d", "gold_probs_2d"),
    ]:
        model = make()
        model.load_state_dict(golden[state_key])
        model.eval()
        with torch.no_grad():
            out = model.predict(golden[input_key])
        np.testing.assert_allclose(
            out.numpy(), golden[probs_key].numpy(), rtol=RTOL, atol=ATOL
        )


def test_unet_prediction_matches_golden(golden, tmp_path):
    """`unet_prediction` maps must be numerically unchanged and in [0, 1]."""
    torch.save(golden["state_unet3d"], tmp_path / "best_checkpoint.pytorch")
    (tmp_path / "config_train.yml").write_text(MODEL_CONFIG)
    patch = tuple(golden["patch"].tolist())
    halo = tuple(golden["halo"].tolist())

    pmaps = unet_prediction(
        raw=golden["raw_3d"].numpy(),
        input_layout="ZYX",
        model_name=None,
        model_id=None,
        patch=patch,
        patch_halo=halo,
        config_path=tmp_path / "config_train.yml",
        model_weights_path=tmp_path / "best_checkpoint.pytorch",
        device="cpu",
        disable_tqdm=True,
    )

    assert pmaps.dtype == np.float32
    assert pmaps.min() >= 0 and pmaps.max() <= 1
    np.testing.assert_allclose(
        pmaps, golden["gold_infer_3d"].numpy(), rtol=RTOL, atol=ATOL
    )


def test_embedding_path_matches_golden(golden):
    """The embedding path must keep today's numbers.

    The `is_embedding` branch historically consumes the model's *activated*
    output (Softmax for sparse models) rather than raw embeddings; whether that
    is the right semantics is undecided (KRiedmiller/panseg#5). Until that is
    settled the branch must produce exactly the values it produces today.
    """
    patch = tuple(golden["patch"].tolist())
    halo = tuple(golden["halo"].tolist())
    raw = golden["raw_3d"].numpy()
    model = make_3d(False, out_channels=2)
    model.load_state_dict(golden["state_unet3d_softmax"])

    dataset = make_dataset(raw, patch=patch, halo=halo)
    predictor = ArrayPredictor(
        model=model,
        in_channels=1,
        out_channels=1,
        device="cpu",
        patch=patch,
        patch_halo=halo,
        single_batch_mode=True,
        headless=False,
        is_embedding=True,
        disable_tqdm=True,
    )
    embeddings = predictor(dataset)

    np.testing.assert_allclose(
        embeddings, golden["gold_embed_3d"].numpy(), rtol=RTOL, atol=ATOL
    )


def test_state_dict_keys_are_unchanged(golden):
    """Old checkpoints stay loadable: no parameter or buffer renames/additions.

    The golden file captured the state-dict keys before the contract change.
    """
    assert set(make_3d(True).state_dict()) == set(golden["state_keys_unet3d"])
    assert set(make_2d().state_dict()) == set(golden["state_keys_unet2d"])


def test_export_subclass_matches_golden_probabilities(golden):
    """The BioImage.IO export callable must emit exactly the old probabilities.

    `biio.py` exports `UNet3DProbabilities`/`UNet2DProbabilities`, whose
    `forward` applies the final activation: published models keep their
    documented [0, 1] output contract. `predict()` must not double-activate.
    """
    model = UNet3DProbabilities(
        in_channels=1,
        out_channels=1,
        final_sigmoid=True,
        f_maps=[8, 16],
        num_groups=4,
        layer_order="gcr",
    )
    model.load_state_dict(golden["state_unet3d"])
    model.eval()
    with torch.no_grad():
        out = model(golden["input_3d"])
        pred = model.predict(golden["input_3d"])
    np.testing.assert_allclose(
        out.numpy(), golden["gold_probs_3d"].numpy(), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        pred.numpy(), golden["gold_probs_3d"].numpy(), rtol=RTOL, atol=ATOL
    )


def test_array_predictor_does_not_double_activate(golden):
    """Feeding the export class to `ArrayPredictor` must not activate twice
    (sigmoid(sigmoid(x)) collapses the probabilities to ~0.5).

    The predictor applies the model's output activation after `forward`; the
    export classes' `forward` is already activated, so their activation entry is
    the identity. Both models carry the same weights, so both must produce the
    same probability map.
    """
    patch = tuple(golden["patch"].tolist())
    halo = tuple(golden["halo"].tolist())
    raw = golden["raw_3d"].numpy()

    def predict(model) -> np.ndarray:
        model.load_state_dict(golden["state_unet3d"])
        return ArrayPredictor(
            model=model,
            in_channels=1,
            out_channels=1,
            device="cpu",
            patch=patch,
            patch_halo=halo,
            single_batch_mode=True,
            headless=False,
            disable_tqdm=True,
        )(make_dataset(raw, patch=patch, halo=halo))

    base_output = predict(make_3d(True))
    export_output = predict(
        UNet3DProbabilities(
            in_channels=1,
            out_channels=1,
            final_sigmoid=True,
            f_maps=[8, 16],
            num_groups=4,
            layer_order="gcr",
        )
    )

    np.testing.assert_allclose(export_output, base_output, rtol=RTOL, atol=ATOL)
    # a double sigmoid would pull every value towards 0.5
    assert not np.allclose(base_output, 0.5, atol=0.1)
