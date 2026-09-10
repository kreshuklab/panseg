import shutil
import zipfile
from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from panseg.functionals.training.biio import (
    PANSEG_CITATION,
    make_model_description,
)

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
    defaults = dict(
        weights=Path("best_checkpoint.pytorch"),
        model_name="dummy_model",
        in_channels=1,
        out_channels=1,
        feature_maps=64,
        patch_size=(16, 32, 64),
        dimensionality="3D",
        layer_order="bcr",
        modality="mod",
        output_type="boundaries",
        description="dummy model",
        resolution=(0.5, 0.02, 2),
        test_in=model_dir / "inputs.npy",
        test_out=model_dir / "outputs.npy",
        panseg_config=Path("best_checkpoint.pytorch"),
    )
    with chdir(model_dir):
        return make_model_description(**{**defaults, **overrides})


def test_make_model_description(model_dir):
    make_description(model_dir)


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


def test_make_model_description_invalid_author(model_dir):
    with pytest.raises(ValueError, match="Name <not-an-email>"):
        make_description(model_dir, authors=["Name <not-an-email>"])

    with pytest.raises(ValueError, match="<john@example.com>"):
        make_description(model_dir, authors=["<john@example.com>"])

    with pytest.raises(ValueError, match="Invalid author line"):
        make_description(model_dir, authors=["Inva/lid Name"])


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


def test_cover_shows_input_and_output(model_dir):
    # Realistic z-scored data: mostly dark background with a bright
    # structure in one quadrant. Regression test for the cover rendering
    # black on the input side (min-max normalization + diagonal split).
    rng = np.random.default_rng(0)
    raw = np.zeros((16, 64, 64), dtype="float32")
    raw[:, 32:, :32] = 5.0
    raw += rng.normal(0, 0.1, size=(16, 64, 64)).astype("float32")
    data = ((raw - raw.mean()) / raw.std()).astype("float32")
    out = (1 / (1 + np.exp(-data * 2))).astype("float32")
    np.save(model_dir / "inputs.npy", data[None, None])
    np.save(model_dir / "outputs.npy", out[None, None])

    desc = make_description(model_dir, patch_size=(16, 64, 64))

    assert [str(c) for c in desc.covers] == ["cover.png"]
    img = np.array(Image.open(model_dir / "cover.png").convert("RGB"))
    h, w = img.shape[:2]
    # input | output side by side
    assert w > h
    side = (w - 4) // 2
    input_half = img[:, :side]
    output_half = img[:, w - side :]
    # both halves must show visible content
    assert input_half.max() > 200
    assert input_half.mean() > 32
    assert output_half.max() > 200

    # the packaged test tensors must be left untouched
    assert np.load(model_dir / "inputs.npy").min() == pytest.approx(data.min())
    assert np.load(model_dir / "inputs.npy").max() == pytest.approx(data.max())
    assert np.load(model_dir / "outputs.npy").max() == pytest.approx(out.max())


def test_package_includes_cover(model_dir):
    zip_path = model_dir / "model.zip"
    with chdir(model_dir):
        desc = make_description(model_dir)
        desc.package(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "cover.png" in names
