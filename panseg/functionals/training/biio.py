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

dim = "3d"
if dim == "3d":
    axes = [SpaceInputAxis(id=AxisId("z"), size=ParameterizedSize)]

input_desc = InputTensorDescr(
    description="model input",
    axes=axes,
    id=TensorId(),
)
