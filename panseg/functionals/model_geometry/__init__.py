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
shift-invariant, so the exact reach is replayed per axis and per phase (see
``.replay``). The replay is the authoritative source; see
``docs/chapters/advanced/halo.md``.

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

from panseg.functionals.model_geometry.layers import (
    Conv,
    Pool,
    is_2d_model,
    ordered_ops,
    touches_axis,
)
from panseg.functionals.model_geometry.replay import exact_halo, reach_table

logger = logging.getLogger(__name__)

# Fraction of the patch size the sliding window advances by default. The stride
# is rounded down to a multiple of the phase period so that overlapping patches
# compute the same voxel at the same phase (see `stride_shape`).
DEFAULT_STRIDE_RATIO = 0.75

__all__ = [
    "ModelGeometry",
    "model_geometry",
    "aligned_shape",
    "aligned_down",
    "model_input_shape",
    "tensor_input_shape",
    "is_2d_model",
    "stride_shape",
    "DEFAULT_STRIDE_RATIO",
]


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
        floor = tuple(
            max(minimum, 2 * halo + 1)
            for minimum, halo in zip(self.min_input, self.halo)
        )
        return aligned_shape(floor, self.increment)

    def aligned_shape(self, shape: tuple[int, ...]) -> tuple[int, int, int]:
        """Round `shape` (Z, Y, X) up to the per-axis phase period."""
        return aligned_shape(shape, self.increment)

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

    def stride_shape(
        self,
        patch: tuple[int, int, int],
        stride_ratio: float = DEFAULT_STRIDE_RATIO,
    ) -> tuple[int, int, int]:
        """Sliding-window stride for `patch`, aligned to this geometry's increment."""
        return stride_shape(patch, self.increment, stride_ratio)

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
        NotImplementedError: for spatial layers the replay does not model:
            transposed convolutions, convolutions that are not size-preserving
            (even kernel, ``padding != kernel_size // 2``, stride or dilation
            other than 1, non-zero padding mode), pooling with padding, dilation
            or ``ceil_mode``, and unrecognised modules that expose spatial
            attributes.

    References:
        - For further details on halos in the context of neural networks, see:
          https://doi.org/10.6028/jres.126.009
    """
    ops = ordered_ops(model)

    min_input = [1, 1, 1]
    increment = [1, 1, 1]
    for op in ops:
        if isinstance(op, Pool):
            for axis in range(3):
                min_input[axis] *= int(op.kernels[axis])
                increment[axis] *= int(op.strides[axis])

    halo = [0, 0, 0]
    radius_sums = [0, 0, 0]
    for axis in range(3):
        if increment[axis] == 1 and all(
            not (isinstance(op, (Conv, Pool)) and touches_axis(op, axis)) for op in ops
        ):
            continue
        table, radius_sums[axis] = reach_table(ops, axis, increment[axis])
        halo[axis] = exact_halo(table, increment[axis])

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


def stride_shape(
    patch_shape: tuple[int, int, int],
    increment: tuple[int, int, int],
    stride_ratio: float = DEFAULT_STRIDE_RATIO,
) -> tuple[int, int, int]:
    """Stride for sliding-window prediction, aligned to the phase period.

    The stride must be a multiple of the per-axis phase period (`increment`):
    otherwise overlapping patches compute the same voxel at different phases,
    and a voxel that is exact in one patch can carry an artefact in the next.
    It must never exceed the patch size, and never be below one voxel. A patch
    below one phase period cannot stride on the grid and returns 1.

    Args:
        patch_shape: patch shape (Z, Y, X).
        increment: per-axis phase period (Z, Y, X).
        stride_ratio: fraction of the patch to advance by. Defaults to 0.75.

    Returns:
        stride shape (Z, Y, X).
    """
    stride = []
    for patch, period in zip(patch_shape, increment):
        if patch < period:
            stride.append(1)
            continue
        stepped = (int(patch * stride_ratio) // period) * period
        stride.append(min(patch, max(period, stepped)))
    return (stride[0], stride[1], stride[2])


def aligned_shape(shape, increment) -> tuple[int, int, int]:
    """Round `shape` (Z, Y, X) up to the per-axis phase period `increment`.

    Accepts any sequence of three sizes, including numpy arrays, so callers do
    not need a second rounding implementation.
    """
    return _aligned(shape, increment, up=True)


def aligned_down(shape, increment) -> tuple[int, int, int]:
    """Round `shape` (Z, Y, X) down to the per-axis phase period `increment`."""
    return _aligned(shape, increment, up=False)


def _aligned(shape, increment, up: bool) -> tuple[int, int, int]:
    values = [
        -(-int(size) // int(period)) * int(period)
        if up
        else (int(size) // int(period)) * int(period)
        for size, period in zip(_as_3d(tuple(shape)), increment)
    ]
    return (values[0], values[1], values[2])


def _without_z(
    shape: tuple[int, int, int], is_2d: bool
) -> tuple[int, int, int] | tuple[int, int]:
    return shape[1:] if is_2d else shape


def _as_3d(shape) -> tuple[int, int, int]:
    return (int(shape[0]), int(shape[1]), int(shape[2]))
