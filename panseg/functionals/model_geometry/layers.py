"""Model modules mapped to the vocabulary the reach replay understands.

This is the single place that maps model modules to the replay vocabulary. It
special-cases `Encoder`, `Decoder`, `InterpolateUpsampling` and the container
types `nn.Sequential`/`nn.ModuleList`, and it refuses every other module that
is not provably transparent: an unrecognised module that exposes a spatial
attribute, or an unrecognised container with children, may change the spatial
layout in a way the replay has no rule for. Any new block type in
`panseg.functionals.training.model` must be registered here, or the geometry
will fail closed rather than under-approximate the reach.
"""

from dataclasses import dataclass

from torch import nn

from panseg.functionals.training.model import (
    AbstractUNet,
    Decoder,
    Encoder,
    InterpolateUpsampling,
)

_POOLING_2D = (nn.MaxPool2d, nn.AvgPool2d)
_POOLING_3D = (nn.MaxPool3d, nn.AvgPool3d)
_CONV_2D = (nn.Conv2d,)
_CONV_3D = (nn.Conv3d,)

_AXES_2D = (1, 2)
_AXES_3D = (0, 1, 2)

# Layers that resample the spatial grid in a way the replay has no rule for.
# The tuple is not exhaustive by construction; `_check_unknown` also fails
# closed on any other module that exposes a spatial attribute.
_UNSUPPORTED_SPATIAL = (
    nn.Upsample,
    nn.UpsamplingNearest2d,
    nn.UpsamplingBilinear2d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.FractionalMaxPool2d,
    nn.FractionalMaxPool3d,
    nn.LPPool2d,
)

# Attributes through which an unrecognised module can change the spatial
# layout. A module that has one of them is not an activation, norm or
# container, and silently treating it as identity would under-approximate the
# reach.
_SPATIAL_ATTRIBUTES = (
    "kernel_size",
    "stride",
    "padding",
    "output_size",
    "scale_factor",
)

# Captured at import so a patched `torch.nn.DataParallel` (tests, plugins) does
# not break the unwrapping below.
_DATA_PARALLEL = nn.DataParallel


def _spatial_layer(module: nn.Module) -> tuple[tuple[int, ...], bool] | None:
    """Return `(axes, is_pooling)` for a spatial layer, or None."""
    if isinstance(module, _POOLING_3D):
        return _AXES_3D, True
    if isinstance(module, _POOLING_2D):
        return _AXES_2D, True
    if isinstance(module, _CONV_3D):
        return _AXES_3D, False
    if isinstance(module, _CONV_2D):
        return _AXES_2D, False
    return None


def is_2d_model(model: nn.Module) -> bool:
    """Whether `model` operates on 2D spatial data.

    Decided by the model's own conv/pool layers, so wrappers such as
    ``nn.DataParallel`` and ``torch.compile`` are transparent.
    """
    saw_2d = False
    saw_3d = False
    for module in model.modules():
        spatial = _spatial_layer(module)
        if spatial is None:
            continue
        axes, _ = spatial
        saw_2d = saw_2d or len(axes) == 2
        saw_3d = saw_3d or len(axes) == 3
    return saw_2d and not saw_3d


def _per_axis(
    kernel_size: int | tuple[int, ...], axes: tuple[int, ...]
) -> tuple[int, ...]:
    """Broadcast an isotropic kernel size to one value per spatial axis."""
    if isinstance(kernel_size, int):
        return (kernel_size,) * len(axes)
    return kernel_size


def _full_axes(value, axes: tuple[int, ...], fill: int) -> tuple[int, int, int]:
    """Broadcast a per-layer value to one entry per (Z, Y, X) axis."""
    full = [fill, fill, fill]
    for axis, size in zip(axes, _per_axis(value, axes)):
        full[axis] = int(size)
    return (full[0], full[1], full[2])


def _axis_name(axis: int) -> str:
    return "ZYX"[axis]


def _check_unknown(module: nn.Module) -> None:
    """Refuse a module that may change the spatial layout but has no replay rule.

    Childless modules without a spatial attribute (activations, normalisations,
    dropout, ``nn.Identity``) stay transparent. Anything else is refused rather
    than treated as identity: an unregistered spatial module, or an unregistered
    container whose children would otherwise never be traversed, would
    under-approximate the reach without a signal.
    """
    name = type(module).__name__
    if isinstance(module, _UNSUPPORTED_SPATIAL):
        raise NotImplementedError(
            f"Model geometry for {name} is not implemented; the reach replay "
            "does not model its resampling. PanSeg architectures upsample "
            "through `InterpolateUpsampling`."
        )
    spatial = [
        attribute
        for attribute in _SPATIAL_ATTRIBUTES
        if getattr(module, attribute, None) is not None
    ]
    if spatial:
        raise NotImplementedError(
            f"Model geometry for {name} is not implemented; it exposes the "
            f"spatial attributes {spatial}, so the reach replay cannot assume "
            "it is transparent. Register the layer type with `model_geometry` "
            "or use a supported layer."
        )
    if next(module.children(), None) is not None:
        raise NotImplementedError(
            f"Model geometry for {name} is not implemented; it is an "
            "unregistered container, so the reach replay would skip its "
            "children. Register the container type with `model_geometry` or "
            "build it from `nn.Sequential`/`nn.ModuleList`."
        )


def _check_conv(
    module: nn.Module,
    axes: tuple[int, ...],
    radii: tuple[int, int, int],
) -> None:
    """Refuse a convolution whose size effect the replay does not model.

    The replay assumes a size-preserving, phase-consistent convolution: an odd
    kernel with `padding = kernel_size // 2`, unit stride, unit dilation and
    zero padding. The caller only routes `nn.Conv2d`/`nn.Conv3d` here, so the
    assertion narrows the type without a runtime change.
    """
    assert isinstance(module, (nn.Conv2d, nn.Conv3d))
    name = type(module).__name__
    if isinstance(module.padding, str):
        raise NotImplementedError(
            f"Model geometry does not model {name} with padding="
            f"{module.padding!r}; only an explicit padding of "
            "kernel_size // 2 preserves the size."
        )
    if module.padding_mode != "zeros":
        raise NotImplementedError(
            f"Model geometry does not model {name} with padding_mode="
            f"{module.padding_mode!r}; only zero padding is supported."
        )
    kernels = _per_axis(module.kernel_size, axes)
    paddings = _per_axis(module.padding, axes)
    strides = _per_axis(module.stride, axes)
    dilations = _per_axis(module.dilation, axes)
    for axis, kernel, padding, stride, dilation in zip(
        axes, kernels, paddings, strides, dilations
    ):
        where = f"{name} on the {_axis_name(axis)} axis"
        if kernel != 2 * radii[axis] + 1:
            raise NotImplementedError(
                f"Model geometry does not model {where} with "
                f"kernel_size={kernel}; only odd kernels preserve the size."
            )
        if padding != radii[axis]:
            raise NotImplementedError(
                f"Model geometry does not model {where} with padding={padding}; "
                f"only padding=kernel_size // 2 (here {radii[axis]}) preserves "
                "the size."
            )
        if stride != 1:
            raise NotImplementedError(
                f"Model geometry does not model {where} with stride={stride}; "
                "only a unit stride preserves the size."
            )
        if dilation != 1:
            raise NotImplementedError(
                f"Model geometry does not model {where} with dilation={dilation}; "
                "only a unit dilation is supported."
            )


def _check_pool(
    module: nn.Module,
    axes: tuple[int, ...],
) -> None:
    """Refuse a pooling layer whose size effect the replay does not model.

    Pooling may only downsample by its kernel/stride grid: no padding, no
    dilation and no `ceil_mode` (which would add a partial cell at the far
    edge). The caller only routes the four supported pooling types here, so the
    assertion narrows the type without a runtime change.
    """
    assert isinstance(module, (nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool2d, nn.AvgPool3d))
    name = type(module).__name__
    if module.ceil_mode:
        raise NotImplementedError(
            f"Model geometry does not model {name} with ceil_mode=True; only "
            "floor-mode pooling is supported."
        )
    paddings = _per_axis(module.padding, axes)
    dilations = _per_axis(getattr(module, "dilation", 1), axes)
    for axis, padding, dilation in zip(axes, paddings, dilations):
        where = f"{name} on the {_axis_name(axis)} axis"
        if padding != 0:
            raise NotImplementedError(
                f"Model geometry does not model {where} with padding={padding}; "
                "only padding=0 is supported."
            )
        if dilation != 1:
            raise NotImplementedError(
                f"Model geometry does not model {where} with dilation={dilation}; "
                "only a unit dilation is supported."
            )


@dataclass(frozen=True)
class Conv:
    """A convolution; `radii` is the per-axis reach in level-local cells."""

    radii: tuple[int, int, int]


@dataclass(frozen=True)
class Pool:
    kernels: tuple[int, int, int]
    strides: tuple[int, int, int]


@dataclass(frozen=True)
class SaveSkip:
    """Mark the end of an encoder level whose output the decoder concatenates."""


@dataclass(frozen=True)
class JoinEncoderSkip:
    """Nearest-neighbour upsample to the next saved encoder size, plus its union."""


Op = Conv | Pool | SaveSkip | JoinEncoderSkip


def module_ops(module: nn.Module) -> list[Op]:
    """Ordered per-axis operations of a module subtree."""
    if isinstance(module, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        raise NotImplementedError(
            "Model geometry for transposed convolutions is not implemented; "
            "PanSeg architectures use nearest-neighbour upsampling."
        )
    if isinstance(module, Encoder):
        ops: list[Op] = [] if module.pooling is None else module_ops(module.pooling)
        return ops + module_ops(module.basic_module)
    if isinstance(module, Decoder):
        return [JoinEncoderSkip()] + module_ops(module.basic_module)
    if isinstance(module, InterpolateUpsampling):
        return [JoinEncoderSkip()]
    if isinstance(module, (nn.Sequential, nn.ModuleList)):
        ops = []
        for child in module:
            ops.extend(module_ops(child))
        return ops
    spatial = _spatial_layer(module)
    if spatial is None:
        _check_unknown(module)
        return []
    axes, is_pooling = spatial
    if is_pooling:
        _check_pool(module, axes)
        stride = module.stride if module.stride is not None else module.kernel_size
        return [
            Pool(
                kernels=_full_axes(module.kernel_size, axes, 1),
                strides=_full_axes(stride, axes, 1),
            )
        ]
    kernel = _full_axes(module.kernel_size, axes, 1)
    radii = (kernel[0] // 2, kernel[1] // 2, kernel[2] // 2)
    _check_conv(module, axes, radii)
    return [Conv(radii=radii)]


def ordered_ops(model: nn.Module) -> list[Op]:
    """Ordered operations of `model`, with encoder skips marked for the decoder."""
    # DataParallel and torch.compile wrap the architecture transparently; the
    # spatial layers, and with them the encoder/decoder structure, are inside.
    while isinstance(model, _DATA_PARALLEL):
        model = model.module
    if hasattr(model, "_orig_mod"):  # torch.compile's OptimizedModule
        model = model._orig_mod

    if not isinstance(model, AbstractUNet):
        return module_ops(model)

    ops: list[Op] = []
    encoders = list(model.encoders)
    for index, encoder in enumerate(encoders):
        ops.extend(module_ops(encoder))
        if index < len(encoders) - 1:
            # The deepest encoder output feeds the decoder directly, so only
            # the shallower levels need a saved skip.
            ops.append(SaveSkip())
    for decoder in model.decoders:
        ops.extend(module_ops(decoder))
    ops.extend(module_ops(model.final_conv))
    return ops


def touches_axis(op: Conv | Pool, axis: int) -> bool:
    """Whether `op` acts on `axis` (a unit kernel/stride axis is transparent)."""
    if isinstance(op, Conv):
        return op.radii[axis] > 0
    return op.kernels[axis] > 1 or op.strides[axis] > 1
