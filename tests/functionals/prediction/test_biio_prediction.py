import numpy as np
import pytest
from bioimageio.core import create_prediction_pipeline
from bioimageio.core.axis import AxisId
from bioimageio.core.prediction import predict
from bioimageio.core.sample import Sample
from bioimageio.core.tensor import Tensor
from bioimageio.spec import load_model_description
from bioimageio.spec.model.v0_5 import TensorId

from panseg.functionals.prediction.prediction import biio_prediction


def _predict_whole(package, raw: np.ndarray) -> np.ndarray:
    """The model's own prediction of the whole image, without blocks (CZYX)."""
    model = load_model_description(package, perform_io_checks=False)
    pipeline = create_prediction_pipeline(model, devices=["cpu"])
    dims = [AxisId(a) for a in ("batch", "channel", "z", "y", "x")]
    sample = Sample(
        members={TensorId("input"): Tensor(array=raw[None, None], dims=dims)},
        stat={},
        id="raw",
    )
    prediction = predict(model=pipeline, inputs=sample).members[TensorId("output")]
    pipeline.unload()
    return prediction.data.to_numpy()[0]


@pytest.mark.parametrize(
    ("package", "shape"),
    [
        ("mock_biio_package", (8, 32, 32)),  # the model's fixed size
        ("mock_biio_package_parameterized", (40, 48, 80)),  # fits in one block
    ],
    ids=["fixed", "parameterized"],
)
def test_image_that_fits_in_one_block_is_predicted_whole(package, shape, request):
    """An image that fits in one block is predicted whole, as the model itself
    does, rather than in padded pieces with the halo carved out."""
    package = request.getfixturevalue(package)
    raw = np.random.default_rng(0).random(shape, dtype=np.float32)

    prediction = biio_prediction(
        raw=raw, input_layout="ZYX", model_id=str(package), device="cpu"
    )

    np.testing.assert_allclose(
        prediction["output"], _predict_whole(package, raw), rtol=1e-5, atol=1e-6
    )
