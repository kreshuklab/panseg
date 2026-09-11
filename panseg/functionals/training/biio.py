import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromFileDescr,
    Author,
    AxisId,
    BatchAxis,
    ChannelAxis,
    CiteEntry,
    FileDescr,
    Identifier,
    InputTensorDescr,
    IntervalOrRatioDataDescr,
    ModelDescr,
    OutputTensorDescr,
    ParameterizedSize,
    PytorchStateDictWeightsDescr,
    SizeReference,
    SpaceInputAxis,
    SpaceOutputAxis,
    TensorId,
    Version,
    WeightsDescr,
    ZeroMeanUnitVarianceDescr,
)
from imageio.v3 import imwrite
from pydantic import ValidationError

from panseg.functionals.training.augs import PercentileNormalizer

PANSEG_CITATION = CiteEntry(
    text=(
        "Wolny, A. et al. Accurate and versatile 3D segmentation of plant "
        "tissues at cellular resolution, eLife 9:e57613"
    ),
    doi="10.7554/eLife.57613",
)

_AUTHOR_LINE_RE = re.compile(r"^(?P<name>.+?)\s*<(?P<email>[^>]*)>$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+")
_URL_RE = re.compile(r"https?://\S+")


def _split_lines(text: str | list[str] | None) -> list[str]:
    lines = text.splitlines() if isinstance(text, str) else text or []
    return [stripped for line in lines if (stripped := line.strip())]


def parse_authors(authors: str | list[str] | None) -> list[Author]:
    parsed: list[Author] = []
    for line in _split_lines(authors):
        if "<" in line or ">" in line:
            match = _AUTHOR_LINE_RE.match(line)
            if match is None:
                raise ValueError(
                    f"Invalid author line, expected 'Name <email>': {line!r}"
                )
            name = match.group("name").strip()
            email = match.group("email").strip()
            if not name:
                raise ValueError(f"Invalid author line, missing name: {line!r}")
            if not _EMAIL_RE.match(email):
                raise ValueError(f"Invalid author line, invalid email: {line!r}")
            try:
                parsed.append(Author(name=name, email=email))
            except ValidationError as e:
                raise ValueError(f"Invalid author line: {line!r}") from e
        else:
            try:
                parsed.append(Author(name=line))
            except ValidationError as e:
                raise ValueError(f"Invalid author line: {line!r}") from e
    return parsed


def parse_citations(citations: str | list[str] | None) -> list[CiteEntry]:
    parsed: list[CiteEntry] = []
    for line in _split_lines(citations):
        # maxsplit=1 keeps the identifier out of the citation text.
        # A DOI or URL is mandatory: CiteEntry raises
        # "Either 'doi' or 'url' is required" for text-only citations, even
        # though the JSON schema docs mark doi/url as optional.
        identifier, *rest = line.split(maxsplit=1)
        text = rest[0].strip() if rest else ""
        doi_match = _DOI_RE.search(identifier)
        url_match = _URL_RE.search(identifier)
        if doi_match is not None:
            doi = doi_match.group(0).rstrip(".,;)")
            parsed.append(CiteEntry(text=text or doi, doi=doi))
        elif url_match is not None:
            url = url_match.group(0).rstrip(".,;)")
            parsed.append(CiteEntry(text=text or url, url=url))
        else:
            raise ValueError(
                f"Invalid citation line, expected '<DOI or URL> [free text]': {line!r}\n"
                "bioimage.io requires a DOI or URL for every citation."
            )
    return parsed


def _pick_2d_slice(data: np.ndarray) -> np.ndarray:
    """Extract a representative 2D slice from a test tensor.

    (1, C, [Z,] Y, X) -> (Y, X), taking the middle z-slice and the first channel.
    """
    data = data[0]
    if data.ndim == 4:
        data = data[:, data.shape[1] // 2, :]
    if data.ndim == 3:
        data = data[0]
    return data


def _normalize_for_display(
    img: np.ndarray, pmin: float = 1.0, pmax: float = 99.6
) -> np.ndarray:
    """Map an image to [0, 1] using a percentile window.

    Purely for visualization: real (e.g. z-scored) data can have a few
    extreme outliers that make plain min-max rendering come out black.
    """
    return np.clip(PercentileNormalizer(pmin=pmin, pmax=pmax)(img), 0.0, 1.0)


def _make_cover(test_in: np.ndarray, test_out: np.ndarray) -> Path:
    """Render an input | output cover image from the test tensors.

    Side by side so both halves stay visible.
    """
    in_img = (_normalize_for_display(_pick_2d_slice(test_in)) * 255).astype("uint8")
    out_img = (_normalize_for_display(_pick_2d_slice(test_out)) * 255).astype("uint8")
    gap = np.full((in_img.shape[0], 4, 1), 255, dtype="uint8")
    canvas = np.concatenate([in_img[:, :, None], gap, out_img[:, :, None]], axis=1)
    canvas = np.repeat(canvas, 3, axis=2)
    imwrite("cover.png", canvas)
    return Path("cover.png")


def _write_documentation(documentation: str, model_name: str) -> Path:
    content = (
        f"# {model_name}\n\n"
        f"{documentation.strip()}\n\n"
        "# Validation\n\n"
        "This model was validated with the packaged test tensors:\n"
        "- input: `test_in.npy`\n"
        "- output: `test_out.npy`\n"
    )
    Path("README.md").write_text(content, encoding="utf-8")
    return Path("README.md")


def make_model_description(
    weights: Path,
    model_name: str,
    in_channels: int,
    out_channels: int,
    feature_maps: int | list[int] | tuple[int, ...],
    patch_size: tuple[int, int, int],
    dimensionality: Literal["2D", "3D"],
    layer_order: str,
    modality: str,
    output_type: str,
    description: str,
    resolution: tuple[float, float, float],
    test_in: Path,
    test_out: Path,
    panseg_config: Path,
    authors: str | list[str] | None = None,
    additional_citations: str | list[str] | None = None,
    license: str | None = None,
    documentation: str | None = None,
):

    if dimensionality == "3D":
        in_axes = [
            BatchAxis(),
            ChannelAxis(
                channel_names=[Identifier(f"in_ch_{i}") for i in range(in_channels)]
            ),
            SpaceInputAxis(
                id=AxisId("z_in"),
                size=ParameterizedSize(min=patch_size[0], step=1),
                scale=resolution[0],
                unit="micrometer",
            ),
            SpaceInputAxis(
                id=AxisId("y_in"),
                size=ParameterizedSize(min=patch_size[1], step=1),
                scale=resolution[1],
                unit="micrometer",
            ),
            SpaceInputAxis(
                id=AxisId("x_in"),
                size=ParameterizedSize(min=patch_size[2], step=1),
                scale=resolution[2],
                unit="micrometer",
            ),
        ]
    elif dimensionality == "2D":
        in_axes = [
            BatchAxis(),
            ChannelAxis(
                channel_names=[Identifier(f"in_ch_{i}") for i in range(in_channels)]
            ),
            SpaceInputAxis(
                id=AxisId("y_in"),
                size=ParameterizedSize(min=patch_size[1], step=1),
                scale=resolution[1],
                unit="micrometer",
            ),
            SpaceInputAxis(
                id=AxisId("x_in"),
                size=ParameterizedSize(min=patch_size[2], step=1),
                scale=resolution[2],
                unit="micrometer",
            ),
        ]
    else:
        raise ValueError("Unknown dimension")

    input_desc = InputTensorDescr(
        description="model input",
        id=TensorId("input"),
        axes=in_axes,
        data=IntervalOrRatioDataDescr(type="float32"),
        test_tensor=FileDescr(source=test_in),
        preprocessing=[ZeroMeanUnitVarianceDescr()],
    )

    if dimensionality == "3D":
        out_axes = [
            BatchAxis(),
            ChannelAxis(
                channel_names=[Identifier(f"out_ch_{i}") for i in range(out_channels)]
            ),
            SpaceOutputAxis(
                id=AxisId("z_out"),
                size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("z_in")),
                scale=resolution[0],
                unit="micrometer",
            ),
            SpaceOutputAxis(
                id=AxisId("y_out"),
                size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("y_in")),
                scale=resolution[1],
                unit="micrometer",
            ),
            SpaceOutputAxis(
                id=AxisId("x_out"),
                size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("x_in")),
                scale=resolution[2],
                unit="micrometer",
            ),
        ]
    elif dimensionality == "2D":
        out_axes = [
            BatchAxis(),
            ChannelAxis(
                channel_names=[Identifier(f"out_ch_{i}") for i in range(out_channels)]
            ),
            SpaceOutputAxis(
                id=AxisId("y_out"),
                size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("y_in")),
                scale=resolution[1],
                unit="micrometer",
            ),
            SpaceOutputAxis(
                id=AxisId("x_out"),
                size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("x_in")),
                scale=resolution[2],
                unit="micrometer",
            ),
        ]

    output_desc = OutputTensorDescr(
        id=TensorId("output"),
        description="model output",
        axes=out_axes,
        data=IntervalOrRatioDataDescr(type="float32"),
        test_tensor=FileDescr(source=test_out),
    )

    pytorch_version = Version(torch.__version__)

    if dimensionality == "3D":
        net_id = Identifier("UNet3D")
    elif dimensionality == "2D":
        net_id = Identifier("UNet2D")

    pytorch_architecture = ArchitectureFromFileDescr(
        source=Path(__file__).parent / "model.py",
        callable=net_id,
        kwargs={
            "in_channels": in_channels,
            "out_channels": out_channels,
            "f_maps": feature_maps,
            "layer_order": layer_order,
        },
    )

    documentation_path = (
        _write_documentation(documentation, model_name)
        if documentation and documentation.strip()
        else None
    )

    # Render the cover ourselves: the spec's auto-generated cover uses
    # min-max normalization and a diagonal split, which renders black for
    # real (outlier-rich, spatially inhomogeneous) data.
    cover_path = _make_cover(np.load(test_in), np.load(test_out))

    model_desc = ModelDescr(
        name=model_name,
        description=description,
        tags=["UNet", modality, output_type],
        authors=parse_authors(authors),
        cite=[PANSEG_CITATION, *parse_citations(additional_citations)],
        license=license,
        documentation=documentation_path,
        covers=[cover_path],
        inputs=[input_desc],
        outputs=[output_desc],
        weights=WeightsDescr(
            pytorch_state_dict=PytorchStateDictWeightsDescr(
                source=weights,
                architecture=pytorch_architecture,
                pytorch_version=pytorch_version,
            )
        ),
        attachments=[FileDescr(source=panseg_config)],
    )

    return model_desc
