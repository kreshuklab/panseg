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
    ("package", "crop"),
    [
        # the model's fixed size, (8, 32, 32)
        ("mock_biio_package", np.s_[12:20, 48:80, 48:80]),
        # (32, 48, 80), which fits in one block
        ("mock_biio_package_parameterized", np.s_[:, 40:88, 24:104]),
    ],
    ids=["fixed", "parameterized"],
)
def test_image_that_fits_in_one_block_is_predicted_whole(
    package, crop, ovule_cells, request
):
    """An image that fits in one block is predicted whole, as the model itself
    does, rather than in padded pieces with the halo carved out."""
    package = request.getfixturevalue(package)
    raw = ovule_cells.raw[crop].astype(np.float32)

    prediction = biio_prediction(
        raw=raw, input_layout="ZYX", model_id=str(package), device="cpu"
    )

    np.testing.assert_allclose(
        prediction["output"], _predict_whole(package, raw), rtol=1e-5, atol=1e-6
    )
