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

which :meth:`ModelGeometry.check_patch` enforces on a patch;
:meth:`ModelGeometry.check_input` applies the same minimum to a whole image.
``increment`` is a derivation and BioImage.IO contract, not a validation gate:
manual patch/halo overrides are used verbatim. Prediction, training, the viewer
and the model zoo all resolve geometry through :func:`model_geometry`, following
the same "resolved in exactly one place" approach as
``panseg.functionals.precision``.

The package is split by concern: :mod:`.layers` maps model modules to the replay
vocabulary, :mod:`.replay` computes the exact reach, :mod:`.shapes` holds the
pure shape arithmetic, and :mod:`.geometry` owns the value object and its
derivation. This module is the public facade.
"""

from panseg.functionals.model_geometry.geometry import (
    ModelGeometry,
    model_geometry,
    tensor_input_shape,
)
from panseg.functionals.model_geometry.layers import is_2d_model
from panseg.functionals.model_geometry.shapes import (
    DEFAULT_STRIDE_RATIO,
    aligned_down,
    aligned_shape,
    model_input_shape,
    stride_shape,
)

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
