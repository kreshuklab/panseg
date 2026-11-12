import shutil
import zipfile
from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
import torch
from bioimageio.core import test_model as biio_test_model
from bioimageio.spec.model.v0_5 import AxisId, TensorId

from panseg.functionals.training.biio import (
    PANSEG_CITATION,
    make_model_description,
)
from panseg.functionals.training.model import UNet3D
from panseg.functionals.training.train import zero_mean_unit_variance

WEIGHTS = (
    Path(__file__).parent.parent.parent
    / "resources"
    / "models"
    / "best_checkpoint.pytorch"
)


@pytest.fixture
def model_dir(tmp_path):
    shutil.copy(WEIGHTS, tmp_path)
    np.save(tmp_path / "inputs.npy", np.random.rand(1, 1, 16, 50, 64))
    np.save(tmp_path / "outputs.npy", np.random.rand(1, 1, 16, 50, 64))
    return tmp_path


def make_description(model_dir, **overrides):
    defaults = {
        "weights": Path("best_checkpoint.pytorch"),
        "model_name": "dummy_model",
        "in_channels": 1,
        "out_channels": 1,
        "feature_maps": 64,
        "final_sigmoid": True,
        "axis_min_sizes": (16, 32, 64),
        "dimensionality": "3D",
        "layer_order": "bcr",
        "modality": "mod",
        "output_type": "boundaries",
        "description": "dummy model",
        "resolution": (0.5, 0.02, 2),
        "test_in": model_dir / "inputs.npy",
        "test_out": model_dir / "outputs.npy",
        "panseg_config": Path("best_checkpoint.pytorch"),
    }
    with chdir(model_dir):
        return make_model_description(**{**defaults, **overrides})


def test_make_model_description_fair_fields(model_dir):
    desc = make_description(
        model_dir,
        authors=["Jane Doe", "John Smith <john@example.com>"],
        additional_citations=["10.1234/abc.def Smith, J. et al. Some result."],
        license="MIT",
        documentation="A very good model.",
    )

    assert [a.name for a in desc.authors] == ["Jane Doe", "John Smith"]
    assert desc.authors[1].email == "john@example.com"

    assert len(desc.cite) == 2
    assert desc.cite[0].text == PANSEG_CITATION.text
    assert desc.cite[0].doi == PANSEG_CITATION.doi
    assert desc.cite[1].doi == "10.1234/abc.def"
    assert desc.cite[1].text == "Smith, J. et al. Some result."

    assert desc.license == "MIT"
    assert str(desc.documentation) == "README.md"


def test_make_model_description_citation_url(model_dir):
    desc = make_description(
        model_dir,
        additional_citations=["https://example.com/paper.html Some result"],
    )
    assert desc.cite[1].url == "https://example.com/paper.html"
    assert desc.cite[1].text == "Some result"


def test_make_model_description_citation_identifier_only(model_dir):
    desc = make_description(
        model_dir,
        additional_citations=["doi:10.1234/abc.def", "https://example.com/paper.html"],
    )
    assert desc.cite[1].doi == "10.1234/abc.def"
    assert desc.cite[1].text == "10.1234/abc.def"
    assert desc.cite[2].url == "https://example.com/paper.html"
    assert desc.cite[2].text == "https://example.com/paper.html"


def test_make_model_description_defaults(model_dir):
    desc = make_description(model_dir)

    assert desc.authors == []
    assert len(desc.cite) == 1
    assert desc.cite[0].text == PANSEG_CITATION.text
    assert desc.cite[0].doi == PANSEG_CITATION.doi
    assert desc.license is None
    assert desc.documentation is None
    assert [str(c) for c in desc.covers] == ["cover.png"]
    assert (model_dir / "cover.png").exists()


def test_make_model_description_final_activation_plumbing(model_dir):
    """The exported architecture must reproduce the trained model's output contract.

    The exported callable is the probabilities subclass (`forward` applies the
    final activation), state-dict compatible with `UNet3D`, and `final_sigmoid`
    still selects *which* activation.
    """
    desc = make_description(model_dir, final_sigmoid=False)
    arch = desc.weights.pytorch_state_dict.architecture
    assert str(arch.callable) == "UNet3DProbabilities"
    assert arch.kwargs["final_sigmoid"] is False

    desc = make_description(model_dir, final_sigmoid=True)
    arch = desc.weights.pytorch_state_dict.architecture
    assert str(arch.callable) == "UNet3DProbabilities"
    assert arch.kwargs["final_sigmoid"] is True


def test_axes_use_exactly_representable_scales(model_dir):
    """Axis scales are exact for the size arithmetic of bioimageio.spec.

    Resolutions that are not exact as floats (0.24) are declared as integer
    counts of a finer unit (240 nanometer); exact resolutions (1.0) stay in
    micrometers. Input and output axes must agree.
    """
    desc = make_description(model_dir, resolution=(0.24, 0.22, 1.0))

    in_axes = {a.id: a for a in desc.inputs[0].axes if a.id in ("z", "y", "x")}
    out_axes = {a.id: a for a in desc.outputs[0].axes if a.id in ("z", "y", "x")}
    expected = {
        "z": (240.0, "nanometer"),
        "y": (220.0, "nanometer"),
        "x": (1.0, "micrometer"),
    }
    for axis_id, (scale, unit) in expected.items():
        assert (in_axes[axis_id].scale, in_axes[axis_id].unit) == (scale, unit)
        assert (out_axes[axis_id].scale, out_axes[axis_id].unit) == (scale, unit)


def test_output_sizes_match_input_sizes(model_dir):
    """Output sizes must equal input sizes for every size parameter n.

    Regression test for the float truncation in bioimageio.spec's
    `SizeReference.get_size`: for resolutions like 0.24 it computes
    `int(44 * 0.24 / 0.24)` as `int(43.999...) = 43`. The output scale
    workaround (`_size_preserving_scale`) keeps the resolved output sizes equal
    to the input sizes.
    """
    desc = make_description(
        model_dir,
        resolution=(0.24, 0.24, 0.24),
        axis_min_sizes=(16, 43, 44),
    )

    for n in (0, 1, 2):
        ns = {(TensorId("input"), AxisId(a)): n for a in ("z", "y", "x")}
        sizes = desc.get_axis_sizes(ns, batch_size=2)
        for a in ("z", "y", "x"):
            key = (TensorId("output"), AxisId(a))
            in_size = sizes.inputs[(TensorId("input"), AxisId(a))]
            assert sizes.outputs[key] == in_size, f"n={n}, axis={a}"


def test_make_model_description_invalid_author(model_dir):
    with pytest.raises(ValueError, match="Name <not-an-email>"):
        make_description(model_dir, authors=["Name <not-an-email>"])

    with pytest.raises(ValueError, match="<john@example.com>"):
        make_description(model_dir, authors=["<john@example.com>"])


def test_make_model_description_invalid_citation(model_dir):
    with pytest.raises(ValueError, match="just some text"):
        make_description(model_dir, additional_citations=["just some text"])

    # the DOI/URL must come first: it cannot be told apart from the text
    # if it is embedded in it
    with pytest.raises(ValueError, match="Some result. doi:10.1234/abc.def"):
        make_description(
            model_dir, additional_citations=["Some result. doi:10.1234/abc.def"]
        )

    # the error must explain why an identifier is required
    with pytest.raises(ValueError, match="requires a DOI or URL"):
        make_description(model_dir, additional_citations=["just some text"])


def test_make_model_description_writes_readme(model_dir):
    make_description(model_dir, documentation="# My notes\nTrained on blobs.")

    content = (model_dir / "README.md").read_text(encoding="utf-8")
    assert "# dummy_model" in content
    assert "# My notes" in content
    assert "Trained on blobs." in content
    assert "# Validation" in content
    assert "test_in.npy" in content
    assert "test_out.npy" in content


def test_package_includes_cover(model_dir):
    zip_path = model_dir / "model.zip"
    with chdir(model_dir):
        desc = make_description(model_dir)
        desc.package(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "cover.png" in names


def test_packaged_model_reproduces_probability_test_out(model_dir):
    """End-to-end export fence: the packaged callable emits the test_out tensor.

    `bioimageio.core.test_model` loads the frozen `model.py` from the package
    and calls the exported callable's `forward`, which must apply the final
    activation: published models keep their [0, 1] output contract and are
    validated against probability test tensors. Runs fully offline.
    """
    torch.manual_seed(0)
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        f_maps=(8, 16),
        layer_order="gcr",
        final_sigmoid=True,
    )
    torch.save(model.state_dict(), model_dir / "best_checkpoint.pytorch")
    model.eval()

    # the training loader stores float32 test tensors
    test_in = np.load(model_dir / "inputs.npy").astype("float32")
    np.save(model_dir / "inputs.npy", test_in)
    with torch.no_grad():
        logits = model(zero_mean_unit_variance(torch.from_numpy(test_in)))
        test_out = model.predict(zero_mean_unit_variance(torch.from_numpy(test_in)))
    assert (logits < 0).any(), "the fixture must produce a non-trivial logit map"
    np.save(model_dir / "outputs.npy", test_out.numpy())

    desc = make_description(
        model_dir,
        feature_maps=(8, 16),
        layer_order="gcr",
        axis_min_sizes=(16, 50, 64),
        resolution=(1.0, 1.0, 1.0),
    )
    summary = biio_test_model(desc, devices=["cpu"])

    assert summary.status == "passed"
