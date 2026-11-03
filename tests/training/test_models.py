import pytest
import torch

from panseg.functionals.training.embeddings import embeddings_to_affinities
from panseg.functionals.training.model import (
    SpocoNet,
    UNet2D,
    UNet2DProbabilities,
    UNet3D,
    UNet3DProbabilities,
)
from tests.conftest import IS_CUDA_AVAILABLE


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Cuda is not available")
class TestModelPrediction:
    def test_UNet2D(self):
        torch.manual_seed(0)
        model = UNet2D(in_channels=3, out_channels=1)
        model.eval()
        x = torch.randn(4, 3, 260, 260)
        y = model(x)
        assert y.shape == (4, 1, 260, 260)
        # forward() returns logits in every mode; the activation is explicit
        assert (y < 0).any() or (y > 1).any()

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
        torch.manual_seed(0)
        return UNet2D(
            in_channels=3,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            f_maps=(8, 16),
            **kwargs,
        )

    def test_forward_returns_logits_in_train_and_eval_mode(self):
        torch.manual_seed(1)
        x = torch.randn(2, 3, 32, 32)
        model = self._model(final_sigmoid=True)
        with torch.no_grad():
            model.train()
            train_out = model(x)
            model.eval()
            eval_out = model(x)

        # mode-independent contract: the same logits in both modes
        assert torch.equal(train_out, eval_out)
        # logits, not probabilities: sigmoid outputs are strictly positive
        assert (eval_out < 0).any()

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

        # mode-independent contract: the same probabilities in both modes
        with torch.no_grad():
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

    @pytest.mark.parametrize(
        "base_cls, probabilities_cls, volume_shape",
        [
            (UNet2D, UNet2DProbabilities, (2, 3, 32, 32)),
            (UNet3D, UNet3DProbabilities, (2, 3, 16, 16, 16)),
        ],
    )
    def test_probabilities_subclass_is_state_dict_compatible(
        self, base_cls, probabilities_cls, volume_shape
    ):
        """Old weights must stay loadable: the export subclass adds no state.

        The subclass is what `biio.py` exports; it must have exactly the
        parameter/buffer names of the trained architecture (a wrapper would
        rename them and break `PytorchStateDictWeightsDescr` loading).
        """
        torch.manual_seed(0)
        kwargs = dict(
            in_channels=3, out_channels=1, final_sigmoid=True, f_maps=(8, 16)
        )
        base = base_cls(**kwargs)
        probabilities = probabilities_cls(**kwargs)

        state = base.state_dict()
        # exact key set (parameters *and* buffers), no renames, no additions
        assert set(state) == set(probabilities.state_dict())
        # strict load in both directions
        probabilities.load_state_dict(state)
        base.load_state_dict(probabilities.state_dict())

        x = torch.randn(*volume_shape)
        base.eval()
        probabilities.eval()
        with torch.no_grad():
            # the subclass's forward is exactly the base model's probability output
            torch.testing.assert_close(probabilities(x), base.predict(x))
