import shutil
import zipfile
from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
import torch
from bioimageio.core import test_model as biio_test_model
from bioimageio.spec.model.v0_5 import (
    AxisId,
    ParameterizedSize,
    SizeReference,
    SpaceOutputAxis,
    SpaceOutputAxisWithHalo,
    TensorId,
)

from panseg.functionals.model_geometry import ModelGeometry
from panseg.functionals.training.biio import (
    PANSEG_CITATION,
    make_model_description,
    pad_to_effective_minimum,
)
from panseg.functionals.training.model import UNet3D
from panseg.functionals.training.train import zero_mean_unit_variance

WEIGHTS = (
    Path(__file__).parent.parent.parent
    / "resources"
    / "models"
    / "best_checkpoint.pytorch"
)

GEOMETRY = ModelGeometry(
    min_input=(16, 32, 64), increment=(8, 8, 8), halo=(4, 8, 12), is_2d=False
)
"""A 3D geometry whose halo fits the effective minimum on every axis."""


@pytest.fixture
def model_dir(tmp_path):
    shutil.copy(WEIGHTS, tmp_path)
    np.save(tmp_path / "inputs.npy", np.random.rand(1, 1, 16, 56, 64))
    np.save(tmp_path / "outputs.npy", np.random.rand(1, 1, 16, 56, 64))
    return tmp_path


def make_description(model_dir, **overrides):
    defaults = {
        "weights": Path("best_checkpoint.pytorch"),
        "model_name": "dummy_model",
        "in_channels": 1,
        "out_channels": 1,
        "feature_maps": 64,
        "final_sigmoid": True,
        "geometry": GEOMETRY,
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


def test_input_minima_include_the_halo_floor(model_dir):
    """A halo axis declares at least `2 * halo + 1` input pixels.

    That is the smallest tile with a non-empty, artifact-free interior after
    cropping. `min_input` alone (4 px) would leave no interior for a 20 px
    halo.
    """
    geometry = ModelGeometry(
        min_input=(4, 4, 4), increment=(1, 1, 1), halo=(20, 20, 20), is_2d=False
    )
    np.save(model_dir / "inputs.npy", np.random.rand(1, 1, 41, 50, 64))
    np.save(model_dir / "outputs.npy", np.random.rand(1, 1, 41, 50, 64))

    desc = make_description(model_dir, geometry=geometry)

    in_axes = {a.id: a for a in desc.inputs[0].axes if a.id in ("z", "y", "x")}
    for axis_id in ("z", "y", "x"):
        size = in_axes[axis_id].size
        assert isinstance(size, ParameterizedSize)
        assert (size.min, size.step) == (41, 1)


def test_input_minimum_stays_min_input_above_the_halo_floor(model_dir):
    """`min_input` wins when it already exceeds `2 * halo + 1`."""
    desc = make_description(model_dir)  # GEOMETRY: min_input (16,32,64), halo (4,8,12)

    in_axes = {a.id: a for a in desc.inputs[0].axes if a.id in ("z", "y", "x")}
    for axis_id, minimum in zip(("z", "y", "x"), GEOMETRY.min_input):
        size = in_axes[axis_id].size
        assert isinstance(size, ParameterizedSize)
        # every input axis steps by the phase period and its minimum is valid
        assert size.step == GEOMETRY.increment[("z", "y", "x").index(axis_id)]
        assert size.min == minimum
        assert size.min % size.step == 0


def test_output_axes_declare_the_true_geometry_halo(model_dir):
    desc = make_description(model_dir)

    out_axes = {a.id: a for a in desc.outputs[0].axes if a.id in ("z", "y", "x")}
    for axis_id, halo in zip(("z", "y", "x"), GEOMETRY.halo):
        axis = out_axes[axis_id]
        assert isinstance(axis, SpaceOutputAxisWithHalo)
        assert axis.halo == halo
        assert axis.size == SizeReference(
            tensor_id=TensorId("input"), axis_id=AxisId(axis_id)
        )


def test_output_axis_without_halo_stays_plain(model_dir):
    """`halo` must be >= 1, so a zero-halo axis keeps the plain output axis."""
    geometry = ModelGeometry(
        min_input=(8, 16, 16), increment=(8, 8, 8), halo=(0, 4, 4), is_2d=False
    )

    desc = make_description(model_dir, geometry=geometry)

    in_axes = {a.id: a for a in desc.inputs[0].axes if a.id in ("z", "y", "x")}
    out_axes = {a.id: a for a in desc.outputs[0].axes if a.id in ("z", "y", "x")}
    assert type(out_axes["z"]) is SpaceOutputAxis
    assert in_axes["z"].size.min == 8
    assert isinstance(out_axes["y"], SpaceOutputAxisWithHalo)
    assert out_axes["y"].halo == 4


def test_true_halo_is_not_capped(model_dir):
    """A UNet3D runs on 8 px but needs 44 px of context per side.

    The halo is the whole point of `SpaceOutputAxisWithHalo`: external
    consumers must overlap tiles by 44 px. The input minimum therefore moves up
    to `2 * 48 + 1 = 97`, rounded up to 104, the smallest artifact-free,
    step-valid tile.
    """
    geometry = ModelGeometry(
        min_input=(8, 8, 8), increment=(8, 8, 8), halo=(48, 48, 48), is_2d=False
    )
    np.save(model_dir / "inputs.npy", np.random.rand(1, 1, 104, 104, 104))
    np.save(model_dir / "outputs.npy", np.random.rand(1, 1, 104, 104, 104))

    desc = make_description(model_dir, geometry=geometry)

    out_axes = {a.id: a for a in desc.outputs[0].axes if a.id in ("z", "y", "x")}
    for axis in out_axes.values():
        assert isinstance(axis, SpaceOutputAxisWithHalo)
        assert axis.halo == 48

    in_axes = {a.id: a for a in desc.inputs[0].axes if a.id in ("z", "y", "x")}
    assert all(axis.size.min == 104 for axis in in_axes.values())
    assert all(axis.size.step == 8 for axis in in_axes.values())


def test_2d_description_uses_the_yx_geometry(model_dir, tmp_path):
    geometry = ModelGeometry(
        min_input=(1, 4, 4), increment=(1, 8, 8), halo=(0, 20, 20), is_2d=True
    )
    np.save(tmp_path / "inputs_2d.npy", np.random.rand(1, 1, 56, 64))
    np.save(tmp_path / "outputs_2d.npy", np.random.rand(1, 1, 56, 64))

    desc = make_description(
        model_dir,
        geometry=geometry,
        test_in=tmp_path / "inputs_2d.npy",
        test_out=tmp_path / "outputs_2d.npy",
    )

    in_axes = {a.id: a for a in desc.inputs[0].axes if a.id in ("y", "x")}
    assert {a.id: a.size.min for a in in_axes.values()} == {"y": 48, "x": 48}
    assert {a.id: a.size.step for a in in_axes.values()} == {"y": 8, "x": 8}

    out_axes = {a.id: a for a in desc.outputs[0].axes if a.id in ("y", "x")}
    assert {a.id: a.halo for a in out_axes.values()} == {"y": 20, "x": 20}
    assert all(isinstance(a, SpaceOutputAxisWithHalo) for a in out_axes.values())

    # the exported architecture follows the geometry, not a separate flag
    arch = desc.weights.pytorch_state_dict.architecture
    assert str(arch.callable) == "UNet2DProbabilities"


def test_pad_to_effective_minimum_pads_every_spatial_axis():
    geometry = ModelGeometry(
        min_input=(8, 8, 8), increment=(8, 8, 8), halo=(48, 48, 48), is_2d=False
    )
    test_in = torch.rand(1, 1, 16, 50, 64)

    padded = pad_to_effective_minimum(test_in, geometry)

    assert tuple(padded.shape) == (1, 1, 104, 104, 104)
    # symmetric padding with the odd pixel on the right; the data is unchanged
    assert padded[0, 0, 44:60, 27:77, 20:84].equal(test_in[0, 0])
    assert padded[0, 0, :44].count_nonzero() == 0


def test_pad_to_effective_minimum_pads_off_step_axes_above_the_minimum():
    """Sizes above the minimum must still satisfy `min + n * step`.

    The description declares a `ParameterizedSize`, so a 150 px axis is
    invalid for `min = 104, step = 8` even though it exceeds the minimum.
    Leaving it untouched fails BioImage.IO validation after training; the
    axis must be rounded up to the next valid size (152).
    """
    geometry = ModelGeometry(
        min_input=(8, 8, 8), increment=(8, 8, 8), halo=(48, 48, 48), is_2d=False
    )
    test_in = torch.rand(1, 1, 80, 150, 150)

    padded = pad_to_effective_minimum(test_in, geometry)

    assert tuple(padded.shape) == (1, 1, 104, 152, 152)
    # symmetric padding with the odd pixel on the right; the data is unchanged
    assert padded[0, 0, 12:92, 1:151, 1:151].equal(test_in[0, 0])


def test_pad_to_effective_minimum_rounds_2d_axes_to_the_step():
    geometry = ModelGeometry(
        min_input=(1, 4, 4), increment=(1, 8, 8), halo=(0, 20, 20), is_2d=True
    )
    test_in = torch.rand(1, 1, 7, 52)

    padded = pad_to_effective_minimum(test_in, geometry)

    assert tuple(padded.shape) == (1, 1, 48, 56)
    assert padded[0, 0, 20:27, 2:54].equal(test_in[0, 0])


def test_pad_to_effective_minimum_is_a_no_op_above_the_minimum():
    geometry = ModelGeometry(
        min_input=(8, 8, 8), increment=(8, 8, 8), halo=(48, 48, 48), is_2d=False
    )
    test_in = torch.rand(1, 1, 104, 104, 104)

    assert pad_to_effective_minimum(test_in, geometry) is test_in


def test_pad_to_effective_minimum_pads_yx_only_for_2d():
    geometry = ModelGeometry(
        min_input=(1, 4, 4), increment=(1, 8, 8), halo=(0, 20, 20), is_2d=True
    )
    test_in = torch.rand(1, 1, 7, 20)

    padded = pad_to_effective_minimum(test_in, geometry)

    assert tuple(padded.shape) == (1, 1, 48, 48)
    assert padded[0, 0, 20:27, 14:34].equal(test_in[0, 0])


def test_padded_small_patch_produces_a_valid_description(model_dir):
    """A training patch below `2 * halo + 1` still exports.

    The packaged test input is padded up to the effective minimum instead of
    shrinking the halo or the minimum. The description validates because the
    stored test tensor satisfies the effective input minima.
    """
    geometry = ModelGeometry(
        min_input=(8, 8, 8), increment=(8, 8, 8), halo=(48, 48, 48), is_2d=False
    )
    test_in = pad_to_effective_minimum(torch.rand(1, 1, 16, 50, 64), geometry)
    np.save(model_dir / "inputs.npy", test_in.numpy())
    np.save(model_dir / "outputs.npy", test_in.numpy())

    desc = make_description(model_dir, geometry=geometry)

    minima = {a.id: a.size.min for a in desc.inputs[0].axes if a.id in ("z", "y", "x")}
    assert minima == {"z": 104, "y": 104, "x": 104}
    assert (
        np.load(model_dir / "inputs.npy").shape
        == np.load(model_dir / "outputs.npy").shape
        == tuple(test_in.shape)
    )


def test_padded_off_step_patch_produces_a_valid_description(model_dir):
    """A patch above the minimum but off the step still exports.

    Regression test for the training patch (80, 150, 150): padding only up to
    the effective minimum left Y/X at 150, which `bioimageio.spec` rejects
    because 150 is not 104 + n * 8.
    """
    geometry = ModelGeometry(
        min_input=(8, 8, 8), increment=(8, 8, 8), halo=(48, 48, 48), is_2d=False
    )
    test_in = pad_to_effective_minimum(torch.rand(1, 1, 80, 150, 150), geometry)
    np.save(model_dir / "inputs.npy", test_in.numpy())
    np.save(model_dir / "outputs.npy", test_in.numpy())

    desc = make_description(model_dir, geometry=geometry)

    assert tuple(test_in.shape) == (1, 1, 104, 152, 152)
    minima = {a.id: a.size.min for a in desc.inputs[0].axes if a.id in ("z", "y", "x")}
    steps = {a.id: a.size.step for a in desc.inputs[0].axes if a.id in ("z", "y", "x")}
    assert minima == {"z": 104, "y": 104, "x": 104}
    assert steps == {"z": 8, "y": 8, "x": 8}
    assert np.load(model_dir / "inputs.npy").shape == tuple(test_in.shape)


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
        geometry=ModelGeometry(
            min_input=(16, 43, 44), increment=(8, 8, 8), halo=(4, 5, 6), is_2d=False
        ),
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
        geometry=ModelGeometry(
            min_input=(16, 50, 64), increment=(1, 1, 1), halo=(2, 4, 4), is_2d=False
        ),
        resolution=(1.0, 1.0, 1.0),
    )
    summary = biio_test_model(desc, devices=["cpu"])

    assert summary.status == "passed"
