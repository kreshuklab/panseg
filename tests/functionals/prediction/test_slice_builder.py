"""SliceBuilder patch-shape handling.

Issue #6: the legacy 64 px Y/X floor is gone. The only lower bound on a patch
is the model's architectural minimum, which the prediction layer checks.
"""

import numpy as np

from panseg.functionals.prediction.utils.slice_builder import SliceBuilder


def test_slice_builder_accepts_patch_below_64_px():
    raw = np.zeros((2, 60, 70), dtype="float32")

    builder = SliceBuilder(
        raw, label_dataset=None, patch_shape=(1, 57, 57), stride_shape=(1, 57, 57)
    )

    assert builder.raw_slices == [
        (slice(z, z + 1), slice(y, y + 57), slice(x, x + 57))
        for z in (0, 1)
        for y in (0, 3)
        for x in (0, 13)
    ]
