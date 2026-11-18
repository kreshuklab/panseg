"""Exact per-axis reach of an architecture, replayed over a 1D support.

The halo is never computed from a closed form: pooling and nearest-neighbour
upsampling are not shift-invariant, so the reach of every output cell is
replayed per axis and per phase. This replay is the authoritative source; see
``docs/chapters/advanced/halo.md``.
"""

from collections.abc import Sequence
from functools import lru_cache

from panseg.functionals.model_geometry.layers import Conv, Op, Pool, SaveSkip

# Upper bound on the replay size multiplier. `_reach_table` doubles it until the
# sampled positions are safely in the interior, so a few doublings suffice for
# every realistic architecture; the cap only protects against a malformed model.
_MAX_REPLAY_MULTIPLIER = 512


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


def replay(
    ops: Sequence[Op], axis: int, model_input: int
) -> tuple[dict[int, set[int]], int]:
    """Replay the per-axis supports over a model input of `model_input` voxels.

    Returns the support of every final output position and the final size.
    """
    support: dict[int, set[int]] = {i: {i} for i in range(model_input)}
    size = model_input
    saves: list[tuple[dict[int, set[int]], int]] = []

    for op in ops:
        if isinstance(op, Conv):
            support = _conv_support(support, size, int(op.radii[axis]))
        elif isinstance(op, Pool):
            support, size = _pool_support(
                support, size, int(op.kernels[axis]), int(op.strides[axis])
            )
        elif isinstance(op, SaveSkip):
            saves.append((support, size))
        else:  # JoinEncoderSkip
            encoder_support, encoder_size = saves.pop()
            source_size = size
            support = {
                q: support.get((q * source_size) // encoder_size, set())
                | encoder_support.get(q, set())
                for q in range(encoder_size)
            }
            size = encoder_size

    return support, size


def _reach_table(
    ops: Sequence[Op], axis: int, increment: int
) -> dict[int, tuple[int, int]]:
    """Uncached exact per-phase `(left, right)` reach of the interior.

    The replay runs on an aligned model input of growing size until the sampled
    positions no longer touch either tensor edge. For an aligned input the
    reach depends only on `p mod increment` in the interior, so the table holds
    one entry per occurring phase.
    """
    replay_multiplier = 8
    while True:
        model_input = increment * replay_multiplier
        support, final_size = replay(ops, axis, model_input)
        lo = final_size // 4
        hi = final_size - lo
        positions = range(lo, max(lo + 1, hi))
        table: dict[int, tuple[int, int]] = {}
        interior = True
        for q in positions:
            voxels = support[q]
            if not voxels:
                continue
            if min(voxels) <= 0 or max(voxels) >= model_input - 1:
                interior = False
                break
            # A U-Net output sits at full resolution and its index is the input
            # position. A model without upsampling has a coarser output grid,
            # where output `q` represents input cell `increment * q` (the
            # per-axis product of the pooling strides); other phases do not
            # occur and must not be invented.
            anchor = q if final_size == model_input else q * increment
            phase = anchor % increment
            table.setdefault(phase, (anchor - min(voxels), max(voxels) - anchor))
        if interior and table:
            return table
        replay_multiplier *= 2
        if replay_multiplier > _MAX_REPLAY_MULTIPLIER:
            raise RuntimeError(
                "Could not find an interior model input for the reach replay "
                f"along axis {axis} (increment {increment}); the architecture "
                "is too deep or too strongly downsampling."
            )


# `maxsize=None` is safe: the key space is (number of distinct architectures) ×
# 3 axes. The op dataclasses are frozen, so the op tuple is hashable and a model
# keeps the same key across predictions.
@lru_cache(maxsize=None)
def _reach_table_cached(
    ops: tuple[Op, ...], axis: int, increment: int
) -> tuple[tuple[int, tuple[int, int]], ...]:
    table = _reach_table(ops, axis, increment)
    return tuple(sorted(table.items()))


def reach_table(
    ops: Sequence[Op], axis: int, increment: int
) -> dict[int, tuple[int, int]]:
    """Cached per (op sequence, axis, period); returns a fresh dict per call."""
    items = _reach_table_cached(tuple(ops), axis, increment)
    return dict(items)


def exact_halo(table: dict[int, tuple[int, int]], increment: int) -> int:
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
