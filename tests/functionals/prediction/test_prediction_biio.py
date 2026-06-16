import pytest

from panseg.functionals.prediction.prediction import biio_prediction


@pytest.mark.parametrize(
    "raw_fixture_name, input_layout, model_id",
    (
        ("raw_zcyx_96x2x96x96", "ZCYX", "philosophical-panda"),
        ("raw_cell_3d_100x128x128", "ZYX", "emotional-cricket"),
        ("raw_cell_2d_96x96", "YX", "pioneering-rhino"),
    ),
)
def test_biio_prediction(raw_fixture_name, input_layout, model_id, request):
    named_pmaps = biio_prediction(
        request.getfixturevalue(raw_fixture_name), input_layout, model_id, device="cpu"
    )
    for key, pmap in named_pmaps.items():
        assert pmap is not None, f"Prediction map for {key} is None"
        assert pmap.ndim == 4, f"Prediction map for {key} has {pmap.ndim} dimensions"
