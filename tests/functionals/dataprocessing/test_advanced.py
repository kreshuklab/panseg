import numpy as np

from panseg.functionals.dataprocessing.advanced_dataprocessing import (
    fix_over_under_segmentation_from_nuclei,
    get_bbox,
    remove_false_positives_by_foreground_probability,
)
from panseg.functionals.proofreading.utils import get_bboxes as get_bboxes_proof


def test_get_bbox_mask_slices():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[0:2, 1:2, 2:3] = True
    slices = get_bbox(mask, pixel_tolerance=0)
    assert slices[0] == slice(0, 2, None)
    assert slices[1] == slice(1, 2, None)
    assert slices[2] == slice(2, 3, None)


def test_remove_false_positives_by_foreground_probability():
    seg = np.ones((10, 10, 10), dtype=np.uint16)
    seg[2:8, 2:8, 2:8] += 20
    prob = np.zeros((10, 10, 10), dtype=np.float32)
    prob[2:8, 2:8, 2:8] += 0.4
    prob[3:7, 3:7, 3:7] += 0.4

    kept, _ = remove_false_positives_by_foreground_probability(
        seg, prob, np.mean(prob[2:8, 2:8, 2:8] * 0.99)
    )
    assert np.sum(kept == 1) == 216
    assert np.sum(kept == 2) == 0
    assert np.sum(kept == 0) == 1000 - 216

    kept, removed = remove_false_positives_by_foreground_probability(
        seg, prob, np.mean(prob[2:8, 2:8, 2:8] * 1.01)
    )
    assert np.sum(kept == 1) == 0
    assert np.sum(kept == 0) == 1000
    assert np.sum(removed == 1) == 784
    assert np.sum(removed == 2) == 216


def test_fix_over_under_segmentation_from_nuclei(complex_test_data):
    """
    Test the fix_over_under_segmentation_from_nuclei function with complex input data.

    Args:
        complex_test_data (tuple): A tuple containing cell segmentation array,
                                   nuclei segmentation array, and boundary probability map.

    Tests:
        - Verifies the initial state of the input data.
        - Ensures under-segmented regions are split correctly.
        - Ensures over-segmented regions are merged correctly.
    """
    cell_seg, nuclei_seg, boundary_pmap = complex_test_data

    # Check that the input data is as expected
    assert len(np.unique(cell_seg[2:6, 2:6, 2:6])) == 1, (
        "Initial region should have 1 unique label."
    )
    assert len(np.unique(cell_seg[1:4, 1:4, 6:9])) == 1, (
        "Initial region should have 1 unique label."
    )
    assert len(np.unique(cell_seg[6:10, 6:10, 6:10])) == 2, (
        "Initial region should have 2 unique labels."
    )

    corrected_seg = fix_over_under_segmentation_from_nuclei(
        cell_seg=cell_seg,
        nuclei_seg=nuclei_seg,
        threshold_merge=0.3,
        threshold_split=0.6,
        quantile_min=0.1,
        quantile_max=0.9,
        boundary=boundary_pmap,
    )

    # Check under-segmented regions are split
    assert len(np.unique(corrected_seg[2:6, 2:6, 2:6])) == 2, (
        "Undersegmentation not split as expected."
    )
    assert len(np.unique(corrected_seg[1:4, 1:4, 6:9])) == 2, (
        "Undersegmentation not split as expected."
    )

    # Check over-segmented regions are merged
    assert len(np.unique(corrected_seg[6:10, 6:10, 6:10])) == 1, (
        "Oversegmentation not merged as expected."
    )
