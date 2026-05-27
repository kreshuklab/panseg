from bioimageio.spec.model.v0_5 import (
    AxisId,
    Identifier,
    InputTensorDescr,
    OutputTensorDescr,
    ParameterizedSize,
    SizeReference,
    SpaceInputAxis,
    SpaceOutputAxis,
    TensorId,
)


def make_model_description(
    dim="3d",
    scale=(1.0, 1.0, 1.0),
):
    if dim == "3d":
        axes = [
            SpaceInputAxis(
                id=AxisId("z"), size=ParameterizedSize(min=32, step=1), scale=scale[0]
            ),
            SpaceInputAxis(
                id=AxisId("y"), size=ParameterizedSize(min=32, step=1), scale=scale[1]
            ),
            SpaceInputAxis(
                id=AxisId("x"), size=ParameterizedSize(min=32, step=1), scale=scale[2]
            ),
        ]
    elif dim == "2d":
        axes = [
            SpaceInputAxis(
                id=AxisId("y"), size=ParameterizedSize(min=32, step=1), scale=scale[1]
            ),
            SpaceInputAxis(
                id=AxisId("x"), size=ParameterizedSize(min=32, step=1), scale=scale[2]
            ),
        ]
    else:
        raise ValueError("Unknown dimension")

    input_desc = InputTensorDescr(
        description="model input",
        axes=axes,
        id=TensorId(),
    )
