"""Shape arithmetic shared by model geometry and its consumers.

These helpers know nothing about a model's architecture: they add a halo,
round a shape to a phase period, or derive a sliding-window stride. Keeping
them apart from :class:`~panseg.functionals.model_geometry.ModelGeometry`
leaves the value object and its derivation readable.
"""

# Fraction of the patch size the sliding window advances by default. The stride
# is rounded down to a multiple of the phase period so that overlapping patches
# compute the same voxel at the same phase (see `stride_shape`).
DEFAULT_STRIDE_RATIO = 0.75


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
