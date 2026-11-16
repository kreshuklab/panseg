"""Architecture-dependent input geometry shared by training, inference and probes.

A model's *architecture* fixes two spatial facts:

- ``min_input``: the smallest input shape each axis accepts, i.e. the product
  of the pooling kernels that downsample that axis. Below it, the deepest
  pooling layer has nothing left to pool.
- ``halo``: the context each side needs for an exact output, i.e. how far a
  convolution in the decoder path can see. Without it, predictions at patch
  borders carry artifacts.

They are independent: a model with only 1x1 convolutions has zero halo but can
still require a pooled minimum. Both are derived here from a single per-axis
traversal of the model, so the two notions of "level" cannot diverge, and 2D
models fall out of the traversal (they only pool Y and X) instead of needing
per-call special cases.

The one invariant that ties them together is::

    model_input_shape(patch, halo) >= min_input

which :meth:`ModelGeometry.check_patch` enforces. Prediction, training, the
viewer and the model zoo all resolve geometry through :func:`model_geometry`,
following the same "resolved in exactly one place" approach as
``panseg.functionals.precision``.
"""

from dataclasses import dataclass

from torch import nn

from panseg.functionals.training.model import InterpolateUpsampling

_POOLING_2D = (nn.MaxPool2d, nn.AvgPool2d)
_POOLING_3D = (nn.MaxPool3d, nn.AvgPool3d)
_CONV_2D = (nn.Conv2d,)
_CONV_3D = (nn.Conv3d,)

_AXES_2D = (1, 2)
_AXES_3D = (0, 1, 2)


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


@dataclass(frozen=True)
class ModelGeometry:
    """Spatial input geometry fixed by a model architecture."""

    min_input: tuple[int, int, int]
    """Smallest input shape (Z, Y, X) the model accepts."""
    halo: tuple[int, int, int]
    """Context (Z, Y, X) each side of a patch needs for an exact output."""
    is_2d: bool
    """Whether the model consumes one Z slice at a time."""

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

    def check_patch(
        self,
        patch: tuple[int, int, int],
        halo: tuple[int, int, int] | None = None,
    ) -> None:
        """Reject a patch whose model input is below the architectural minimum.

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
    """Derive the input geometry of `model` in one per-axis traversal.

    Pooling multiplies the axis minimum and moves one level down; a decoder
    upsampling moves one level back up. Each convolution then contributes
    ``2**level * (kernel // 2)`` to the halo of each axis it operates on.
    """
    min_input = [1, 1, 1]
    halo = [0, 0, 0]
    level = 0
    saw_2d = False
    saw_3d = False

    for module in model.modules():
        spatial = _spatial_layer(module)
        if spatial is None:
            if isinstance(module, InterpolateUpsampling):
                level -= 1
            continue
        axes, is_pooling = spatial
        for axis, size in zip(axes, _per_axis(module.kernel_size, axes)):
            if is_pooling:
                min_input[axis] *= size
            else:
                halo[axis] += 2**level * (size // 2)
        if is_pooling:
            level += 1
        saw_2d = saw_2d or len(axes) == 2
        saw_3d = saw_3d or len(axes) == 3

    return ModelGeometry(
        min_input=(min_input[0], min_input[1], min_input[2]),
        halo=(halo[0], halo[1], halo[2]),
        # The modules, not the model class, decide this: wrappers such as
        # nn.DataParallel and torch.compile are transparent to the traversal.
        is_2d=saw_2d and not saw_3d,
    )


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


def _without_z(
    shape: tuple[int, int, int], is_2d: bool
) -> tuple[int, int, int] | tuple[int, int]:
    return shape[1:] if is_2d else shape


def _as_3d(shape) -> tuple[int, int, int]:
    return (int(shape[0]), int(shape[1]), int(shape[2]))
