import logging
import re
from fractions import Fraction
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

logger = logging.getLogger(__name__)

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


_LENGTH_UNITS: tuple[tuple[str, float], ...] = (
    ("millimeter", 1e3),
    ("micrometer", 1.0),
    ("nanometer", 1e-3),
    ("picometer", 1e-6),
    ("femtometer", 1e-9),
)
"""bioimageio.spec `SpaceUnit`s panseg may declare, with their size in micrometers."""

_MAX_AXIS_SIZE = 2**20
"""Largest axis size for which `_exact_scale` guarantees exact size arithmetic."""


def _exact_scale(resolution: float) -> tuple[float, str]:
    """Express a voxel size in micrometers as an exactly representable scale/unit pair.

    bioimageio.spec computes axis sizes with float expressions like
    `size * scale_in / scale_out`, and validates test tensors by *exact* float
    equality of the result (see `SizeReference.get_size` and the `test_tensor`
    validation in `bioimageio.spec.model.v0_5`). For scales such as 0.24 this is
    inexact (`43 * 0.24 / 0.24` evaluates to 42.999...), so such a description
    rejects valid test tensors at construction time and reports wrong expected
    output shapes in `bioimageio.core.test_model`.

    Integer scales make `size * scale` exactly representable for every plausible
    size, so all these computations return exact integers. We therefore express
    the voxel size as an integer count of a finer unit (0.24 micrometer becomes
    240 nanometer); the physical length is unchanged. Voxel sizes that are exact
    as floats themselves (e.g. 0.5, 1.0) are kept in micrometers.
    """
    # a float is `numerator / 2**j`; `size * resolution` is exact for all sizes
    # up to _MAX_AXIS_SIZE as long as the numerator stays below 2**53 / _MAX_AXIS_SIZE
    if Fraction(resolution).numerator < 2**53 // _MAX_AXIS_SIZE:
        return resolution, "micrometer"

    for unit, micrometers_per_unit in _LENGTH_UNITS:
        value = resolution / micrometers_per_unit
        rounded = round(value)
        if rounded >= 1 and abs(value - rounded) <= 1e-9 * max(1.0, abs(value)):
            return float(rounded), unit

    logger.warning(
        "Voxel size %s micrometers cannot be represented exactly as a bioimageio"
        " axis scale; model export or validation may fail",
        resolution,
    )
    return resolution, "micrometer"


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
    final_sigmoid: bool,
    axis_min_sizes: tuple[int, int, int],
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

    # input and output axes share the same scale/unit pair: the output axis sizes
    # are declared as SizeReferences to the input axes
    (z_scale, z_unit), (y_scale, y_unit), (x_scale, x_unit) = (
        _exact_scale(r) for r in resolution
    )

    if dimensionality == "3D":
        in_axes = [
            BatchAxis(),
            ChannelAxis(
                channel_names=[Identifier(f"in_ch_{i}") for i in range(in_channels)]
            ),
            SpaceInputAxis(
                id=AxisId("z"),
                size=ParameterizedSize(min=axis_min_sizes[0], step=1),
                scale=z_scale,
                unit=z_unit,
            ),
            SpaceInputAxis(
                id=AxisId("y"),
                size=ParameterizedSize(min=axis_min_sizes[1], step=1),
                scale=y_scale,
                unit=y_unit,
            ),
            SpaceInputAxis(
                id=AxisId("x"),
                size=ParameterizedSize(min=axis_min_sizes[2], step=1),
                scale=x_scale,
                unit=x_unit,
            ),
        ]
    elif dimensionality == "2D":
        in_axes = [
            BatchAxis(),
            ChannelAxis(
                channel_names=[Identifier(f"in_ch_{i}") for i in range(in_channels)]
            ),
            SpaceInputAxis(
                id=AxisId("y"),
                size=ParameterizedSize(min=axis_min_sizes[1], step=1),
                scale=y_scale,
                unit=y_unit,
            ),
            SpaceInputAxis(
                id=AxisId("x"),
                size=ParameterizedSize(min=axis_min_sizes[2], step=1),
                scale=x_scale,
                unit=x_unit,
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
                id=AxisId("z"),
                size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("z")),
                scale=z_scale,
                unit=z_unit,
            ),
            SpaceOutputAxis(
                id=AxisId("y"),
                size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("y")),
                scale=y_scale,
                unit=y_unit,
            ),
            SpaceOutputAxis(
                id=AxisId("x"),
                size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("x")),
                scale=x_scale,
                unit=x_unit,
            ),
        ]
    elif dimensionality == "2D":
        out_axes = [
            BatchAxis(),
            ChannelAxis(
                channel_names=[Identifier(f"out_ch_{i}") for i in range(out_channels)]
            ),
            SpaceOutputAxis(
                id=AxisId("y"),
                size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("y")),
                scale=y_scale,
                unit=y_unit,
            ),
            SpaceOutputAxis(
                id=AxisId("x"),
                size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("x")),
                scale=x_scale,
                unit=x_unit,
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

    # the exported callable returns probabilities (its forward applies the
    # final activation): published models keep their [0, 1] output contract
    if dimensionality == "3D":
        net_id = Identifier("UNet3DProbabilities")
    elif dimensionality == "2D":
        net_id = Identifier("UNet2DProbabilities")

    pytorch_architecture = ArchitectureFromFileDescr(
        source=Path(__file__).parent / "model.py",
        callable=net_id,
        kwargs={
            "in_channels": in_channels,
            "out_channels": out_channels,
            "f_maps": feature_maps,
            "final_sigmoid": final_sigmoid,
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
        tags=["UNet", "PanSeg", modality, output_type],
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
