"""Tests for the raw-array padding the predictor feeds on.

Both helpers pad the raw array before it is patchified, at opposite ends:
`pad_to_shape` extends the volume up to the phase grid, `mirror_pad` mirrors
around each patch. Neither may touch a leading channel axis.
"""

import numpy as np
import pytest

from panseg.functionals.prediction.utils.array_dataset import mirror_pad, pad_to_shape


class TestPadToShape:
    def test_no_op_for_an_aligned_zyx_array(self):
        image = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        assert pad_to_shape(image, (2, 3, 4)) is image

    def test_pads_zyx_at_the_high_end_only(self):
        image = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        padded = pad_to_shape(image, (4, 6, 8))

        assert padded.shape == (4, 6, 8)
        # the original extent sits at the low end, untouched
        np.testing.assert_array_equal(padded[:2, :3, :4], image)

    def test_pads_czyx_without_touching_the_channel_axis(self):
        image = np.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4)

        padded = pad_to_shape(image, (4, 6, 8))

        assert padded.shape == (2, 4, 6, 8)
        np.testing.assert_array_equal(padded[:, :2, :3, :4], image)

    def test_raises_when_the_target_is_smaller(self):
        with pytest.raises(ValueError, match="smaller"):
            pad_to_shape(np.zeros((4, 6, 8)), (4, 6, 7))

    def test_allows_a_pad_larger_than_the_axis(self):
        # a single-slice volume padded up to the aligned domain: numpy's
        # reflect mode repeats the reflection of a size-1 axis
        image = np.arange(3 * 4).reshape(1, 3, 4)

        padded = pad_to_shape(image, (8, 3, 4))

        assert padded.shape == (8, 3, 4)
        np.testing.assert_array_equal(padded, np.broadcast_to(image, (8, 3, 4)))


class TestMirrorPad:
    def test_pads_zyx_symmetrically(self):
        image = np.arange(4 * 5 * 6).reshape(4, 5, 6)

        padded = mirror_pad(image, (1, 2, 0), multichannel=False)

        assert padded.shape == (6, 9, 6)
        np.testing.assert_array_equal(padded[1:-1, 2:-2, :], image)
        # the low edge is the reflection, not a copy of the edge itself
        np.testing.assert_array_equal(padded[0, 2:-2, :], image[1])

    def test_pads_czyx_only_on_the_spatial_axes(self):
        image = np.arange(2 * 4 * 5 * 6).reshape(2, 4, 5, 6)

        padded = mirror_pad(image, (1, 2, 0), multichannel=True)

        assert padded.shape == (2, 6, 9, 6)
        np.testing.assert_array_equal(padded[:, 1:-1, 2:-2, :], image)
