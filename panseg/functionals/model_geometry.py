"""Architecture-dependent input geometry shared by training, inference and probes.

A model's *architecture* fixes four spatial facts:

- ``min_input``: the smallest input shape each axis accepts, i.e. the product
  of the pooling kernels that downsample that axis. Below it, the deepest
  pooling layer has nothing left to pool.
- ``increment``: the smallest shape change each axis tolerates, i.e. the
  product of the pooling *strides* that downsample that axis. Patch sizes,
  strides and model inputs are only comparable across overlapping predictions
  when they are multiples of it (the *phase period*).
- ``halo``: the context each side needs for an exact output, i.e. the smallest
  multiple of ``increment`` whose kept crop sees no artificial tensor border.
- ``effective_min_input``: the smallest input that both runs *and* leaves a
  non-empty artifact-free interior, rounded up to ``increment``. This is the
  smallest tile a consumer that crops the halo should use, and therefore the
  minimum a model description declares.

`min_input` and `increment` coincide for the current architectures (every pool
halves), but they do not in general: a pool with kernel 3 and stride 2 needs
three input cells to run yet advances the phase by two. `halo` is *not*
computed from a closed form: pooling and nearest-neighbour upsampling are not
shift-invariant, so the exact reach is replayed per axis and per phase. The
replay is the authoritative source; see ``docs/chapters/advanced/halo.md``.

The invariant that ties running and cropping together is::

    model_input_shape(patch, halo) >= min_input

which :meth:`ModelGeometry.check_patch` enforces. ``increment`` is a derivation
and BioImage.IO contract, not a validation gate: manual patch/halo overrides are
used verbatim. Prediction, training, the viewer and the model zoo all resolve
geometry through :func:`model_geometry`, following the same "resolved in
exactly one place" approach as ``panseg.functionals.precision``.
"""

import logging
from dataclasses import dataclass

from torch import nn

from panseg.functionals.training.model import (
    AbstractUNet,
    Decoder,
    Encoder,
    InterpolateUpsampling,
)

logger = logging.getLogger(__name__)

_POOLING_2D = (nn.MaxPool2d, nn.AvgPool2d)
_POOLING_3D = (nn.MaxPool3d, nn.AvgPool3d)
_CONV_2D = (nn.Conv2d,)
_CONV_3D = (nn.Conv3d,)

_AXES_2D = (1, 2)
_AXES_3D = (0, 1, 2)

# Upper bound on the replay size multiplier. `_reach_table` doubles it until the
# sampled positions are safely in the interior, so a few doublings suffice for
# every realistic architecture; the cap only protects against a malformed model.
_MAX_REPLAY_MULTIPLIER = 512


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


@dataclass(frozen=True)
class _Conv:
    """A convolution; `radii` is the per-axis reach in level-local cells."""

    radii: tuple[int, int, int]


@dataclass(frozen=True)
class _Pool:
    kernels: tuple[int, int, int]
    strides: tuple[int, int, int]


@dataclass(frozen=True)
class _SaveSkip:
    """Mark the end of an encoder level whose output the decoder concatenates."""


@dataclass(frozen=True)
class _JoinSkip:
    """Nearest-neighbour upsample to the next saved encoder size, plus its union."""


_Op = _Conv | _Pool | _SaveSkip | _JoinSkip


def _module_ops(module: nn.Module) -> list[_Op]:
    """Ordered per-axis operations of a module subtree."""
    if isinstance(module, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        raise NotImplementedError(
            "Model geometry for transposed convolutions is not implemented; "
            "PanSeg architectures use nearest-neighbour upsampling."
        )
    if isinstance(module, Encoder):
        ops: list[_Op] = [] if module.pooling is None else _module_ops(module.pooling)
        return ops + _module_ops(module.basic_module)
    if isinstance(module, Decoder):
        return [_JoinSkip()] + _module_ops(module.basic_module)
    if isinstance(module, InterpolateUpsampling):
        return [_JoinSkip()]
    if isinstance(module, (nn.Sequential, nn.ModuleList)):
        ops = []
        for child in module:
            ops.extend(_module_ops(child))
        return ops
    spatial = _spatial_layer(module)
    if spatial is None:
        return []
    axes, is_pooling = spatial
    if is_pooling:
        stride = module.stride if module.stride is not None else module.kernel_size
        return [
            _Pool(
                kernels=_full_axes(module.kernel_size, axes, 1),
                strides=_full_axes(stride, axes, 1),
            )
        ]
    return [
        _Conv(
            radii=tuple(
                radius // 2 for radius in _full_axes(module.kernel_size, axes, 1)
            )
        )
    ]


def _ordered_ops(model: nn.Module) -> list[_Op]:
    """Ordered operations of `model`, with encoder skips marked for the decoder."""
    if not isinstance(model, AbstractUNet):
        return _module_ops(model)

    ops: list[_Op] = []
    encoders = list(model.encoders)
    for index, encoder in enumerate(encoders):
        ops.extend(_module_ops(encoder))
        if index < len(encoders) - 1:
            # The deepest encoder output feeds the decoder directly, so only
            # the shallower levels need a saved skip.
            ops.append(_SaveSkip())
    for decoder in model.decoders:
        ops.extend(_module_ops(decoder))
    ops.extend(_module_ops(model.final_conv))
    return ops


def _conv_support(
    support: dict[int, set[int]], size: int, radius: int
) -> dict[int, set[int]]:
    if radius == 0:
        return support
    return {
        p: set().union(
            *(support.get(p + offset, set()) for offset in range(-radius, radius + 1))
        )
        for p in range(size)
    }


def _pool_support(
    support: dict[int, set[int]], size: int, kernel: int, stride: int
) -> tuple[dict[int, set[int]], int]:
    out_size = (size - kernel) // stride + 1
    if out_size < 1:
        out_size = 1
    return (
        {
            q: set().union(
                *(support.get(stride * q + offset, set()) for offset in range(kernel))
            )
            for q in range(out_size)
        },
        out_size,
    )


def _replay(
    ops: list[_Op], axis: int, model_input: int, track_radius_sum: bool = False
) -> tuple[dict[int, set[int]], int, int]:
    """Replay the per-axis supports over a model input of `model_input` voxels.

    Returns the support of every final output position, the final size, and the
    classic symmetric radius sum `S` (only when requested).
    """
    support: dict[int, set[int]] = {i: {i} for i in range(model_input)}
    size = model_input
    saves: list[tuple[dict[int, set[int]], int]] = []
    radius_sum = 0

    for op in ops:
        if isinstance(op, _Conv):
            if track_radius_sum:
                # `size` is the current level's size; the classic reach scales
                # each radius by the level stride, i.e. `model_input // size`
                radius_sum += int(op.radii[axis]) * (model_input // size)
            support = _conv_support(support, size, int(op.radii[axis]))
        elif isinstance(op, _Pool):
            support, size = _pool_support(
                support, size, int(op.kernels[axis]), int(op.strides[axis])
            )
        elif isinstance(op, _SaveSkip):
            saves.append((support, size))
        else:  # _JoinSkip
            encoder_support, encoder_size = saves.pop()
            source_size = size
            support = {
                q: support.get((q * source_size) // encoder_size, set())
                | encoder_support.get(q, set())
                for q in range(encoder_size)
            }
            size = encoder_size

    return support, size, radius_sum


def _reach_table(
    ops: list[_Op], axis: int, increment: int
) -> tuple[dict[int, tuple[int, int]], int]:
    """Exact per-phase `(left, right)` reach of the interior.

    The replay runs on an aligned model input of growing size until the sampled
    positions no longer touch either tensor edge. For an aligned input the
    reach depends only on `p mod increment` in the interior, so the table holds
    one entry per occurring phase. Returns the table and the classic radius sum
    `S` of that axis.
    """
    multiplier = 8
    while True:
        model_input = increment * multiplier
        support, final_size, radius_sum = _replay(
            ops, axis, model_input, track_radius_sum=True
        )
        lo = final_size // 4
        hi = final_size - lo
        positions = range(lo, max(lo + 1, hi))
        table: dict[int, tuple[int, int]] = {}
        interior = True
        scale = model_input / final_size
        for q in positions:
            voxels = support[q]
            if not voxels:
                continue
            if min(voxels) <= 0 or max(voxels) >= model_input - 1:
                interior = False
                break
            # The final output grid may be coarser than the model input (a
            # model without upsampling); report the reach around the input
            # position that cell represents.
            anchor = int(round(q * scale))
            phase = anchor % increment
            table.setdefault(phase, (anchor - min(voxels), max(voxels) - anchor))
        if interior and table:
            return table, radius_sum
        multiplier *= 2
        if multiplier > _MAX_REPLAY_MULTIPLIER:
            raise RuntimeError(
                "Could not find an interior model input for the reach replay "
                f"along axis {axis} (increment {increment}); the architecture "
                "is too deep or too strongly downsampling."
            )


def _exact_halo(table: dict[int, tuple[int, int]], increment: int) -> int:
    """Smallest multiple of `increment` whose aligned crop is clean.

    The crop contains every phase of the period, at positions `h + t` for
    `t = 0 … increment - 1` (a patch of exactly one period). A position is
    clean when `p - L(p) > 0` and `p + R(p) < M - 1`; substituting `M = Q + 2h`
    with `Q = increment` and `p = h + t` gives the two bounds below. Requiring
    both for every phase makes the result independent of the patch size, as
    long as the patch is a multiple of the period.
    """
    needed = 1
    for phase, (left, right) in table.items():
        needed = max(
            needed,
            left - phase + 1,
            phase + right - increment + 2,
        )
    return ((needed + increment - 1) // increment) * increment


@dataclass(frozen=True)
class ModelGeometry:
    """Spatial input geometry fixed by a model architecture."""

    min_input: tuple[int, int, int]
    """Smallest input shape (Z, Y, X) the model accepts."""
    increment: tuple[int, int, int]
    """Phase period (Z, Y, X): the per-axis product of the pooling strides."""
    halo: tuple[int, int, int]
    """Context (Z, Y, X) each side of a patch needs for an exact output."""
    is_2d: bool
    """Whether the model consumes one Z slice at a time."""

    @property
    def effective_min_input(self) -> tuple[int, int, int]:
        """Smallest (Z, Y, X) input with a non-empty, artifact-free interior.

        `min_input` on its own is only the smallest input that *runs*; a tile
        that small can leave nothing to crop after removing `halo`. The
        effective minimum is therefore `max(min_input, 2 * halo + 1)` rounded up
        to a multiple of `increment`, i.e. the smallest tile whose cropped
        interior still has one pixel and a valid phase. An axis without halo
        keeps its `min_input`.
        """
        return _as_3d(
            tuple(
                _round_up(max(minimum, 2 * halo + 1), increment)
                for minimum, increment, halo in zip(
                    self.min_input, self.increment, self.halo
                )
            )
        )

    def aligned_shape(self, shape: tuple[int, ...]) -> tuple[int, int, int]:
        """Round `shape` (Z, Y, X) up to the per-axis phase period."""
        return _as_3d(
            tuple(
                _round_up(size, increment)
                for size, increment in zip(_as_3d(shape), self.increment)
            )
        )

    def model_input_shape(
        self,
        patch: tuple[int, int, int],
        halo: tuple[int, int, int] | None = None,
    ) -> tuple[int, int, int]:
        """Shape fed to the model for `patch`, with `halo` on both sides.

        `halo` defaults to the theoretical halo of this architecture; pass the
        resolved halo when the caller overrides it.
        """
        return model_input_shape(patch, self.halo if halo is None else halo)

    def tensor_input_shape(
        self,
        patch: tuple[int, int, int],
        halo: tuple[int, int, int] | None = None,
    ) -> tuple[int, ...]:
        """The tensor's spatial shape: the model input without Z for 2D."""
        return _without_z(self.model_input_shape(patch, halo), self.is_2d)

    def tensor_effective_min_input(self) -> tuple[int, ...]:
        """The tensor's smallest spatial shape: `effective_min_input` sans Z for 2D."""
        return _without_z(self.effective_min_input, self.is_2d)

    def check_patch(
        self,
        patch: tuple[int, int, int],
        halo: tuple[int, int, int] | None = None,
    ) -> None:
        """Reject a patch whose model input is below the architectural minimum.

        Only runnability is enforced. Manual halos override every automatic
        calculation, so no alignment or minimum-halo check happens here.

        Raises:
            ValueError: naming the failing axis (Z, Y, X), its size and the
                minimum.
        """
        patch = _as_3d(patch)
        model_input = self.model_input_shape(patch, halo)
        failing = [
            f"{axis} ({size} < {min_size})"
            for axis, size, min_size in zip(
                ("Z", "Y", "X"), model_input, self.min_input
            )
            if size < min_size
        ]
        if failing:
            halo = self.halo if halo is None else halo
            raise ValueError(
                f"Patch shape {patch} with halo {_as_3d(halo)} has "
                f"model input shape {model_input}, which is below the model's "
                f"architectural minimum {self.min_input} in: {', '.join(failing)}.\n"
                "Increase the patch size or halo."
            )


def model_geometry(model: nn.Module) -> ModelGeometry:
    """Derive the input geometry of `model` from its ordered spatial layers.

    The traversal collects the per-axis convolution radii, pooling
    kernels/strides and upsampling joins, then replays the supports of a 1D
    aligned model input per axis. The halo is the smallest multiple of the phase
    period whose aligned crop is clean; it is never assumed from a closed form
    (nearest-neighbour upsampling is not shift-invariant).

    An axis the model does not downsample (*e.g.* Z of a 2D model) gets
    `increment = 1` and `halo = 0`: its reach has a single phase and needs no
    context.

    Raises:
        NotImplementedError: for spatial layers the replay does not model
            (transposed convolutions).

    References:
        - For further details on halos in the context of neural networks, see:
          https://doi.org/10.6028/jres.126.009
    """
    ops = _ordered_ops(model)

    min_input = [1, 1, 1]
    increment = [1, 1, 1]
    for op in ops:
        if isinstance(op, _Pool):
            for axis in range(3):
                min_input[axis] *= int(op.kernels[axis])
                increment[axis] *= int(op.strides[axis])

    halo = [0, 0, 0]
    radius_sums = [0, 0, 0]
    for axis in range(3):
        if increment[axis] == 1 and all(
            not (isinstance(op, (_Conv, _Pool)) and _touches_axis(op, axis))
            for op in ops
        ):
            continue
        table, radius_sums[axis] = _reach_table(ops, axis, increment[axis])
        halo[axis] = _exact_halo(table, increment[axis])

    geometry = ModelGeometry(
        min_input=(min_input[0], min_input[1], min_input[2]),
        increment=(increment[0], increment[1], increment[2]),
        halo=(halo[0], halo[1], halo[2]),
        # The modules, not the model class, decide this: wrappers such as
        # nn.DataParallel and torch.compile are transparent to the traversal.
        is_2d=is_2d_model(model),
    )
    logger.info(
        "Derived model geometry: conv radius sum S=%s -> exact halo %s "
        "(increment %s, minimum input %s).",
        tuple(radius_sums),
        geometry.halo,
        geometry.increment,
        geometry.min_input,
    )
    return geometry


def _touches_axis(op: _Conv | _Pool, axis: int) -> bool:
    """Whether `op` acts on `axis` (a unit kernel/stride axis is transparent)."""
    if isinstance(op, _Conv):
        return op.radii[axis] > 0
    return op.kernels[axis] > 1 or op.strides[axis] > 1


def model_input_shape(
    patch: tuple[int, int, int], halo: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Shape fed to the model for `patch`: the patch plus the halo on both sides."""
    patch = _as_3d(patch)
    halo = _as_3d(halo)
    return (
        patch[0] + 2 * halo[0],
        patch[1] + 2 * halo[1],
        patch[2] + 2 * halo[2],
    )


def tensor_input_shape(
    model: nn.Module,
    patch: tuple[int, int, int],
    halo: tuple[int, int, int],
) -> tuple[int, ...]:
    """The tensor's spatial shape for `model`: the model input without Z for 2D.

    For callers that probe a model without resolving its full geometry.
    """
    return _without_z(model_input_shape(patch, halo), is_2d_model(model))


def _round_up(size: int, increment: int) -> int:
    return ((int(size) + int(increment) - 1) // int(increment)) * int(increment)


def _without_z(
    shape: tuple[int, int, int], is_2d: bool
) -> tuple[int, int, int] | tuple[int, int]:
    return shape[1:] if is_2d else shape


def _as_3d(shape) -> tuple[int, int, int]:
    return (int(shape[0]), int(shape[1]), int(shape[2]))
