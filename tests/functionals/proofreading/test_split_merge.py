import numpy as np

from panseg.functionals.proofreading.split_merge_tools import (
    _merge_from_seeds,
    _split_from_seed,
    split_merge_from_seeds,
)


def test_merge_from_seeds_all_to_one():
    """Test basic merging functionality"""
    segmentation = np.array(
        [
            [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
            [[2, 1, 2], [1, 2, 1], [2, 1, 2]],
            [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
        ]
    )

    region_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
    region_bbox = np.array([[0, 0, 0], [3, 3, 3]])

    bboxes = {1: np.array([[0, 0, 0], [2, 2, 2]]), 2: np.array([[0, 0, 0], [2, 2, 2]])}

    all_idx = np.array([1, 2])

    result_seg, result_slice, result_bboxes = _merge_from_seeds(
        segmentation, region_slice, region_bbox, bboxes, all_idx
    )

    assert result_seg.shape == segmentation.shape

    expected_result = np.ones_like(segmentation)
    np.testing.assert_array_equal(result_seg, expected_result)
    assert all([rs == slice(0, 3, None) for rs in result_slice])

    assert len(result_bboxes) == 1
    assert np.all(result_bboxes[1] == np.array([[0, 0, 0], [3, 3, 3]]))


def test_merge_from_seeds_with_other_label():
    """Test merging when 0 is in the indices"""
    # Create a simple 3D segmentation
    segmentation = np.array(
        [
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[1, 0, 1], [0, 1, 0], [3, 3, 1]],
            [[0, 1, 0], [1, 0, 1], [3, 1, 0]],
        ]
    )

    region_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
    region_bbox = np.array([[0, 0, 0], [3, 3, 3]])

    bboxes = {
        0: np.array([[0, 0, 0], [3, 3, 3]]),
        1: np.array([[0, 0, 0], [3, 3, 3]]),
        3: np.array([[1, 2, 0], [3, 3, 2]]),
    }

    all_idx = np.array([0, 1])

    result_seg, result_slice, result_bboxes = _merge_from_seeds(
        segmentation, region_slice, region_bbox, bboxes, all_idx
    )

    assert result_seg.shape == segmentation.shape

    expected_result = np.zeros_like(segmentation)
    expected_result[segmentation == 0] = 0
    expected_result[segmentation == 1] = 0
    expected_result[segmentation == 3] = 3

    np.testing.assert_array_equal(result_seg, expected_result)
    assert all([rs == slice(0, 3, None) for rs in result_slice])

    # Check that bboxes was updated with the new label
    assert np.array_equal(result_bboxes[0], region_bbox)
    assert len(result_bboxes) == 2
    assert np.all(result_bboxes[0] == np.array([[0, 0, 0], [3, 3, 3]]))
    assert np.all(result_bboxes[3] == bboxes[3])


def test_merge_from_seeds_single_label():
    """Test merging with a single label"""
    segmentation = np.array(
        [
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        ]
    )

    region_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
    region_bbox = np.array([[0, 0, 0], [3, 3, 3]])

    bboxes = {3: np.array([[0, 0, 0], [3, 3, 3]])}

    all_idx = np.array([3])

    result_seg, result_slice, result_bboxes = _merge_from_seeds(
        segmentation, region_slice, region_bbox, bboxes, all_idx
    )

    assert result_seg.shape == segmentation.shape
    np.testing.assert_array_equal(result_seg, segmentation)

    assert 3 in result_bboxes
    assert len(result_bboxes) == 1
    assert np.array_equal(result_bboxes[3], region_bbox)


def test_merge_from_seeds_2d():
    """Test merging functionality with 2D segmentation"""
    segmentation = np.array(
        [
            [1, 2, 1],
            [2, 1, 2],
            [1, 2, 1],
        ]
    )

    region_slice = (slice(0, 3), slice(0, 3))
    region_bbox = np.array([[0, 0], [2, 2]])

    bboxes = {1: np.array([[0, 0], [2, 2]]), 2: np.array([[0, 0], [2, 2]])}

    all_idx = np.array([1, 2])

    result_seg, result_slice, result_bboxes = _merge_from_seeds(
        segmentation, region_slice, region_bbox, bboxes, all_idx
    )

    assert result_seg.shape == segmentation.shape

    expected_result = np.zeros_like(segmentation)
    expected_result[segmentation == 1] = 1
    expected_result[segmentation == 2] = 1
    expected_result[(segmentation != 1) & (segmentation != 2)] = segmentation[
        (segmentation != 1) & (segmentation != 2)
    ]

    np.testing.assert_array_equal(result_seg, expected_result)

    assert 1 in result_bboxes
    assert np.array_equal(result_bboxes[1], region_bbox)


def test_split_from_seed_single_seed():
    """Test splitting with one seed - should merge segments but leave others unchanged"""
    segmentation = np.array(
        [
            [[1, 1, 2], [1, 1, 2], [2, 2, 2]],
            [[1, 1, 2], [1, 1, 2], [2, 2, 2]],
            [[1, 1, 2], [1, 1, 2], [2, 2, 2]],
        ]
    )

    # Use explicit image data instead of random
    image = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]],
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 0.1, 0.2]],
        ]
    )

    seeds_list = (
        [0, 0, 0, 1, 1, 2],
        [0, 1, 2, 0, 1, 2],
        [0, 0, 0, 0, 0, 0],
    )
    seeds_values = np.array([1, 1, 1, 1, 1, 1])

    region_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
    all_idx = np.array([1, 2])
    offsets = np.array([0, 0, 0])
    bboxes = {1: np.array([[0, 0, 0], [2, 2, 2]]), 2: np.array([[0, 0, 0], [2, 2, 2]])}
    max_label = 2

    result_seg, result_slice, result_bboxes = _split_from_seed(
        segmentation,
        seeds_list,
        region_slice,
        all_idx,
        offsets,
        bboxes,
        image,
        seeds_values,
        max_label,
    )

    assert result_seg.shape == segmentation.shape

    assert all([rs == slice(0, 3, None) for rs in result_slice])

    assert len(result_bboxes) == 3


def test_split_from_seed_two_seeds_same_segment():
    """Test splitting with two seeds on the same segment"""
    segmentation = np.array(
        [
            [[1, 1, 1], [1, 2, 2], [1, 2, 2]],
            [[1, 1, 1], [1, 2, 2], [1, 2, 2]],
            [[1, 1, 1], [1, 2, 2], [1, 2, 2]],
        ]
    )

    # Use explicit image data instead of random
    image = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]],
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 0.1, 0.2]],
        ]
    )

    seeds_list = (
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
    )
    seeds_values = np.array([1, 1, 1, 1])  # All seeds have value 1

    region_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
    all_idx = np.array([1, 2])
    offsets = np.array([0, 0, 0])
    bboxes = {1: np.array([[0, 0, 0], [2, 2, 2]]), 2: np.array([[0, 0, 0], [2, 2, 2]])}
    max_label = 2

    result_seg, result_slice, result_bboxes = _split_from_seed(
        segmentation,
        seeds_list,
        region_slice,
        all_idx,
        offsets,
        bboxes,
        image,
        seeds_values,
        max_label,
    )

    assert result_seg.shape == segmentation.shape
    assert all([rs == slice(0, 3, None) for rs in result_slice])
    assert len(result_bboxes) == 3


def test_split_from_seed_two_seeds_different_segments():
    """Test splitting with two seeds on different segments"""
    # Create a simple 3D segmentation
    segmentation = np.array(
        [
            [[1, 1, 1], [2, 2, 2], [1, 1, 1]],
            [[1, 1, 1], [2, 2, 2], [1, 1, 1]],
            [[1, 1, 1], [2, 2, 2], [1, 1, 1]],
        ]
    )

    # Create explicit image data instead of random
    image = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]],
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 0.1, 0.2]],
        ]
    )

    # Define seeds for splitting - two seeds on segment 1 and one on segment 2
    seeds_list = (
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
    )
    seeds_values = np.array([1, 1, 2])  # Seeds have values 1, 1, 2

    region_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
    all_idx = np.array([1, 2])
    offsets = np.array([0, 0, 0])
    bboxes = {1: np.array([[0, 0, 0], [2, 2, 2]]), 2: np.array([[0, 0, 0], [2, 2, 2]])}
    max_label = 2

    result_seg, result_slice, result_bboxes = _split_from_seed(
        segmentation,
        seeds_list,
        region_slice,
        all_idx,
        offsets,
        bboxes,
        image,
        seeds_values,
        max_label,
    )

    assert result_seg.shape == segmentation.shape
    assert all([rs == slice(0, 3, None) for rs in result_slice])
    assert len(result_bboxes) == 4


def test_split_from_seed_with_zero_segmentation():
    """Test splitting behavior with zero in segmentation"""
    segmentation = np.array(
        [
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        ]
    )

    image = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]],
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 0.1, 0.2]],
        ]
    )

    seeds_list = (
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
    )
    seeds_values = np.array([1, 1, 1, 1])  # All seeds have value 1

    region_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
    all_idx = np.array([0, 1])
    offsets = np.array([0, 0, 0])
    bboxes = {0: np.array([[0, 0, 0], [2, 2, 2]]), 1: np.array([[0, 0, 0], [2, 2, 2]])}
    max_label = 1

    result_seg, result_slice, result_bboxes = _split_from_seed(
        segmentation,
        seeds_list,
        region_slice,
        all_idx,
        offsets,
        bboxes,
        image,
        seeds_values,
        max_label,
    )

    assert result_seg.shape == segmentation.shape
    assert all([rs == slice(0, 3, None) for rs in result_slice])
    assert len(result_bboxes) == 3


def test_split_from_seed_preserves_unrelated_segments():
    """Test that segments not in seeds_list remain unchanged"""
    segmentation = np.array(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        ]
    )

    image = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]],
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 0.1, 0.2]],
        ]
    )

    seeds_list = (
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
    )
    seeds_values = np.array([1, 1, 1, 1])

    region_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
    all_idx = np.array([1, 2])
    offsets = np.array([0, 0, 0])
    bboxes = {
        1: np.array([[0, 0, 0], [2, 2, 2]]),
        2: np.array([[0, 0, 0], [2, 2, 2]]),
        3: np.array([[0, 0, 0], [2, 2, 2]]),
    }
    max_label = 3

    result_seg, result_slice, result_bboxes = _split_from_seed(
        segmentation,
        seeds_list,
        region_slice,
        all_idx,
        offsets,
        bboxes,
        image,
        seeds_values,
        max_label,
    )

    assert result_seg.shape == segmentation.shape
    assert all([rs == slice(0, 3, None) for rs in result_slice])
    assert len(result_bboxes) == 4


def test_split_merge_from_seeds_mocked(mocker):
    seeds = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )

    segmentation = np.array(
        [
            [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
            [[1, 2, 1], [2, 2, 2], [1, 2, 1]],
            [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
        ]
    )

    image = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]],
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 0.1, 0.2]],
        ]
    )

    bboxes = {1: np.array([[0, 0, 0], [2, 2, 2]]), 2: np.array([[0, 0, 0], [2, 2, 2]])}
    max_label = 2
    correct_labels = set()

    mock_merge_result = (
        np.array(
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ]
        ),
        (slice(0, 3), slice(0, 3), slice(0, 3)),
        {1: np.array([[0, 0, 0], [3, 3, 3]])},
    )

    mock_split_result = (
        np.array(
            [
                [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
                [[1, 2, 1], [2, 2, 2], [1, 2, 1]],
                [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
            ]
        ),
        (slice(0, 3), slice(0, 3), slice(0, 3)),
        {1: np.array([[0, 0, 0], [3, 3, 3]]), 2: np.array([[0, 0, 0], [3, 3, 3]])},
    )

    mock_merge = mocker.patch(
        "panseg.functionals.proofreading.split_merge_tools._merge_from_seeds",
        return_value=mock_merge_result,
    )

    mock_split = mocker.patch(
        "panseg.functionals.proofreading.split_merge_tools._split_from_seed",
        return_value=mock_split_result,
    )

    result_seg, result_slice, result_bboxes = split_merge_from_seeds(
        seeds, segmentation, image, bboxes, max_label, correct_labels
    )

    mock_merge.assert_called_once()
    mock_split.assert_not_called()

    seeds_multi = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )
    seeds_multi[1, 1, 1] = 2

    mock_merge.reset_mock()
    mock_split.reset_mock()

    result_seg2, result_slice2, result_bboxes2 = split_merge_from_seeds(
        seeds_multi, segmentation, image, bboxes, max_label, correct_labels
    )

    mock_split.assert_called_once()
    mock_merge.assert_not_called()
