import pytest
import torch

from panseg.functionals.training.embeddings import embeddings_to_affinities
from panseg.functionals.training.model import SpocoNet, UNet2D
from tests.conftest import IS_CUDA_AVAILABLE


def _forward_with_final_conv_output(model, x) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(forward output, output of the final 1x1 conv)`` for `x`."""
    captured: dict[str, torch.Tensor] = {}
    handle = model.final_conv.register_forward_hook(
        lambda module, inputs, output: captured.setdefault("logits", output)
    )
    try:
        with torch.no_grad():
            y = model(x)
    finally:
        handle.remove()
    return y, captured["logits"]


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Cuda is not available")
class TestModelPrediction:
    def test_UNet2D(self):
        torch.manual_seed(0)
        model = UNet2D(in_channels=3, out_channels=1)
        model.eval()
        x = torch.randn(4, 3, 260, 260)
        y, logits = _forward_with_final_conv_output(model, x)
        assert y.shape == (4, 1, 260, 260)
        # forward() returns the final conv's logits unchanged in eval mode too:
        # nothing (in particular no activation) is applied after it
        assert torch.equal(y, logits)

    def test_SpocoNet(self):
        model = SpocoNet.from_unet_params(
            in_channels=1, out_channels=8, f_maps=[16, 32, 64, 128, 256]
        )
        model.eval()
        x1 = torch.randn(4, 1, 260, 260)
        x2 = torch.rand(4, 1, 260, 260)
        y1, y2 = model(x1, x2)
        assert y1.shape == (4, 8, 260, 260)
        assert y2.shape == (4, 8, 260, 260)

    def test_embeddings_to_affinities(self):
        x = torch.randn(4, 8, 128, 128)
        offsets = [[-1, 0], [0, -1]]
        delta = 0.5
        affs = embeddings_to_affinities(x, offsets, delta)
        assert affs.shape == (4, 2, 128, 128)
        assert torch.all(affs >= 0) and torch.all(affs <= 1)


class TestOutputContract:
    """`forward()` returns logits in every mode; `predict()` returns probabilities."""

    @staticmethod
    def _model(final_sigmoid: bool, out_channels: int = 1, **kwargs) -> UNet2D:
        # "gcr" (GroupNorm) has no mode-dependent state, unlike BatchNorm or
        # Dropout. The train/eval outputs of these fixtures are therefore equal
        # by construction; that equality only pins that the contract does not
        # depend on the mode, not a general train/eval equivalence.
        torch.manual_seed(0)
        return UNet2D(
            in_channels=3,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            f_maps=(8, 16),
            layer_order="gcr",
            **kwargs,
        )

    def test_forward_returns_logits_in_train_and_eval_mode(self):
        torch.manual_seed(1)
        x = torch.randn(2, 3, 32, 32)
        model = self._model(final_sigmoid=True)
        with torch.no_grad():
            model.train()
            train_out = model(x)
            train_probs = model.predict(x)
            model.eval()
            eval_out = model(x)
            eval_probs = model.predict(x)

        # the real contract, asserted in each mode: predict() is
        # activate(forward()), and forward() returns logits, not probabilities
        # (sigmoid outputs would be strictly positive)
        torch.testing.assert_close(train_probs, model.activate(train_out))
        torch.testing.assert_close(eval_probs, model.activate(eval_out))
        assert (train_out < 0).any()
        assert (eval_out < 0).any()

        # "gcr" (GroupNorm) carries no mode-dependent state, so the values
        # happen to be identical here; this is not a general claim that
        # forward() is mode-invariant for every architecture
        assert torch.equal(train_out, eval_out)

    def test_predict_returns_probabilities(self):
        torch.manual_seed(1)
        x = torch.randn(2, 3, 32, 32)
        model = self._model(final_sigmoid=True)
        with torch.no_grad():
            model.eval()
            logits = model(x)
            probs = model.predict(x)

            assert probs.shape == logits.shape
            assert (probs >= 0).all() and (probs <= 1).all()
            torch.testing.assert_close(probs, torch.sigmoid(logits))
            torch.testing.assert_close(probs, model.activate(logits))

            # the contract holds in train mode too; with "gcr" (GroupNorm) the
            # values themselves cannot change between modes
            model.train()
            assert torch.equal(model.predict(x), probs)

    def test_predict_applies_softmax_when_configured(self):
        torch.manual_seed(1)
        x = torch.randn(2, 3, 32, 32)
        model = self._model(final_sigmoid=False, out_channels=2)
        with torch.no_grad():
            model.eval()
            logits = model(x)
            probs = model.predict(x)

        torch.testing.assert_close(probs, torch.softmax(logits, dim=1))

    def test_predict_is_a_no_op_for_regression_models(self):
        torch.manual_seed(1)
        x = torch.randn(2, 3, 32, 32)
        model = self._model(final_sigmoid=True, is_segmentation=False)
        with torch.no_grad():
            model.eval()
            assert torch.equal(model.predict(x), model(x))
