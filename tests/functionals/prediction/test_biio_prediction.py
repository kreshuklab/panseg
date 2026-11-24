import numpy as np
from bioimageio.core import create_prediction_pipeline
from bioimageio.core.axis import AxisId
from bioimageio.core.prediction import predict
from bioimageio.core.sample import Sample
from bioimageio.core.tensor import Tensor
from bioimageio.spec import load_model_description
from bioimageio.spec.model.v0_5 import TensorId

from panseg.functionals.prediction.prediction import biio_prediction


def test_fixed_size_model_runs_a_full_size_image_in_one_pass(mock_biio_package):
    """An image of the model's fixed input size is predicted whole, as the model
    itself does, rather than in padded blocks."""
    raw = np.random.default_rng(0).random((8, 32, 32), dtype=np.float32)

    prediction = biio_prediction(
        raw=raw, input_layout="ZYX", model_id=str(mock_biio_package), device="cpu"
    )

    model = load_model_description(mock_biio_package, perform_io_checks=False)
    pipeline = create_prediction_pipeline(model, devices=["cpu"])
    dims = [AxisId(a) for a in ("batch", "channel", "z", "y", "x")]
    sample = Sample(
        members={TensorId("input"): Tensor(array=raw[None, None], dims=dims)},
        stat={},
        id="raw",
    )
    expected = predict(model=pipeline, inputs=sample).members[TensorId("output")]
    pipeline.unload()

    np.testing.assert_allclose(
        prediction["output"], expected.data.to_numpy()[0], rtol=1e-5, atol=1e-6
    )
