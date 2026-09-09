import re
import shutil
from pathlib import Path
from typing import Literal

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
from pydantic import ValidationError

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


def parse_authors(authors: list[str] | None) -> list[Author]:
    parsed: list[Author] = []
    for line in authors or []:
        stripped = line.strip()
        if not stripped:
            continue
        if "<" in stripped or ">" in stripped:
            match = _AUTHOR_LINE_RE.match(stripped)
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
                parsed.append(Author(name=stripped))
            except ValidationError as e:
                raise ValueError(f"Invalid author line: {line!r}") from e
    return parsed


def parse_citations(citations: list[str] | None) -> list[CiteEntry]:
    parsed: list[CiteEntry] = []
    for line in citations or []:
        stripped = line.strip()
        if not stripped:
            continue
        doi_match = _DOI_RE.search(stripped)
        url_match = _URL_RE.search(stripped)
        if doi_match is not None:
            parsed.append(
                CiteEntry(text=stripped, doi=doi_match.group(0).rstrip(".,;)"))
            )
        elif url_match is not None:
            parsed.append(
                CiteEntry(text=stripped, url=url_match.group(0).rstrip(".,;)"))
            )
        else:
            raise ValueError(
                f"Invalid citation line, must contain a DOI or URL: {line!r}"
            )
    return parsed


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
    authors: list[str] | None = None,
    additional_citations: list[str] | None = None,
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

    model_desc = ModelDescr(
        name=model_name,
        description=description,
        tags=["UNet", modality, output_type],
        authors=parse_authors(authors),
        cite=[PANSEG_CITATION, *parse_citations(additional_citations)],
        license=license,
        documentation=documentation_path,
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

    # Covers auto-generated from the test tensors live in a tmpdir and are
    # dropped by the exclude_unset packaging serialization. Copy them into the
    # model directory and assign explicitly so they end up in the package.
    if model_desc.covers:
        copied_covers = []
        for i, cover in enumerate(model_desc.covers):
            cover_name = "cover.png" if i == 0 else f"cover_{i}.png"
            shutil.copy(Path(cover), cover_name)
            copied_covers.append(Path(cover_name))
        model_desc.covers = copied_covers

    return model_desc
