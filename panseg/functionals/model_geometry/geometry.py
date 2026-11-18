"""The model-geometry value object and its derivation from an architecture."""

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
from panseg.functionals.model_geometry.shapes import (
    DEFAULT_STRIDE_RATIO,
    _as_3d,
    _without_z,
    aligned_shape,
    model_input_shape,
    stride_shape,
)

logger = logging.getLogger(__name__)


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

    @classmethod
    def from_model(cls, model: nn.Module) -> "ModelGeometry":
        """Derive the input geometry of `model` from its ordered spatial layers.

        The traversal collects the per-axis convolution radii, pooling
        kernels/strides and upsampling joins, then replays the supports of a 1D
        aligned model input per axis. The halo is the smallest multiple of the
        phase period whose aligned crop is clean; it is never assumed from a
        closed form (nearest-neighbour upsampling is not shift-invariant).

        An axis the model does not downsample (*e.g.* Z of a 2D model) gets
        `increment = 1` and `halo = 0`: its reach has a single phase and needs
        no context.

        Raises:
            NotImplementedError: for spatial layers the replay does not model:
                transposed convolutions, convolutions that are not
                size-preserving (even kernel, ``padding != kernel_size // 2``,
                stride or dilation other than 1, non-zero padding mode),
                pooling with padding, dilation or ``ceil_mode``, and
                unrecognised modules that expose spatial attributes.

        References:
            - For further details on halos in the context of neural networks,
              see: https://doi.org/10.6028/jres.126.009
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
        for axis in range(3):
            if increment[axis] == 1 and all(
                not (isinstance(op, (Conv, Pool)) and touches_axis(op, axis))
                for op in ops
            ):
                continue
            table = reach_table(ops, axis, increment[axis])
            halo[axis] = exact_halo(table, increment[axis])

        geometry = cls(
            min_input=(min_input[0], min_input[1], min_input[2]),
            increment=(increment[0], increment[1], increment[2]),
            halo=(halo[0], halo[1], halo[2]),
            # The modules, not the model class, decide this: wrappers such as
            # nn.DataParallel and torch.compile are transparent to the traversal.
            is_2d=is_2d_model(model),
        )
        logger.info(
            "Derived model geometry: exact halo %s (increment %s, minimum input %s).",
            geometry.halo,
            geometry.increment,
            geometry.min_input,
        )
        return geometry

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

    def check_input(self, shape) -> None:
        """Reject an image below the architectural minimum, before probing.

        Args:
            shape: image shape (Z, Y, X).

        Raises:
            ValueError: for a too thin Z axis with the 2D-model hint, else for
                the failing Y/X axes, naming each axis, its size and the
                minimum.
        """
        shape = _as_3d(shape)
        min_depth = self.min_input[0]
        if shape[0] < min_depth:
            raise ValueError(
                f"This 3D model needs at least {min_depth} Z slices, but the image has "
                f"{shape[0]}. Use a 2D model for images with fewer Z slices."
            )
        failing = _axis_violations(("Y", "X"), shape[1:], self.min_input[1:])
        if failing:
            raise ValueError(
                f"This model needs an image of at least {self.min_input} (Z, Y, X) "
                f"to run, but the image shape is {shape}; it is too small in: "
                f"{', '.join(failing)}. Use a smaller model or a larger image."
            )

    def check_patch(
        self,
        patch: tuple[int, int, int],
        halo: tuple[int, int, int] | None = None,
    ) -> None:
        """Reject a patch whose model input is below the architectural minimum.

        Only runnability is enforced. Manual halos override every automatic
        calculation, so no alignment or minimum-halo check happens here; they
        are used verbatim.

        A 2D model additionally needs a one-slice-deep patch without a Z halo,
        since it predicts one Z slice at a time.

        Raises:
            ValueError: naming the failing axis (Z, Y, X), its size and the
                minimum, or the offending 2D patch depth or Z halo.
        """
        patch = _as_3d(patch)
        halo = self.halo if halo is None else _as_3d(halo)
        model_input = self.model_input_shape(patch, halo)
        failing = _axis_violations(("Z", "Y", "X"), model_input, self.min_input)
        if failing:
            raise ValueError(
                f"Patch shape {patch} with halo {halo} has "
                f"model input shape {model_input}, which is below the model's "
                f"architectural minimum {self.min_input} in: {', '.join(failing)}.\n"
                "Increase the patch size or halo."
            )
        if self.is_2d and patch[0] != 1:
            raise ValueError(
                f"Patch shape {tuple(patch)} has {patch[0]} Z slices, but 2D models "
                "predict one Z slice at a time. Set the Z patch size to 1, "
                "or leave the patch unset to derive a feasible one."
            )
        if self.is_2d and halo[0] != 0:
            raise ValueError(
                f"Halo shape {tuple(halo)} has a Z halo of {halo[0]}, but "
                "2D models predict one Z slice at a time. Set the Z halo to 0."
            )


def model_geometry(model: nn.Module) -> ModelGeometry:
    """Derive the input geometry of `model`; alias for `ModelGeometry.from_model`."""
    return ModelGeometry.from_model(model)


def tensor_input_shape(
    model: nn.Module,
    patch: tuple[int, int, int],
    halo: tuple[int, int, int],
) -> tuple[int, ...]:
    """The tensor's spatial shape for `model`: the model input without Z for 2D.

    For callers that probe a model without resolving its full geometry.
    """
    return _without_z(model_input_shape(patch, halo), is_2d_model(model))


def _axis_violations(names, sizes, minima) -> list[str]:
    """Format the axes whose `size` is below `minima` as `"axis (size < min)"`."""
    return [
        f"{name} ({size} < {minimum})"
        for name, size, minimum in zip(names, sizes, minima)
        if size < minimum
    ]
