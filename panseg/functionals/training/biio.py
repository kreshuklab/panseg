from pathlib import Path
from typing import Literal

import torch
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromFileDescr,
    AxisId,
    BatchAxis,
    ChannelAxis,
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

    model_desc = ModelDescr(
        name=model_name,
        description=description,
        tags=["UNet", modality, output_type],
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
