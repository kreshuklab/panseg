import os

import numpy as np
import pytest
from numba.typed import List

from panseg.functionals.proofreading import utils

os.environ["NUMBA_JIT_COVERAGE"] = "1"


def test_get_bboxes3D():
    segmentation = np.zeros((10, 10, 10), dtype=int)
    segmentation[:2, :2, :2] = 1
    segmentation[3:5, 3:6, 3:7] = 2
    labels_idx = [0, 1, 2]
    typed_labels_idx = List()
    [typed_labels_idx.append(x) for x in labels_idx]

    bboxes = utils._get_bboxes3D(segmentation, typed_labels_idx)

    assert all([k in bboxes for k in labels_idx])
    assert np.all(bboxes[1] == [[0, 0, 0], [2, 2, 2]])
    assert np.all(bboxes[2] == [[3, 3, 3], [5, 6, 7]])
    assert np.all(bboxes[0] == [segmentation.shape, [0, 0, 0]])


def test_get_bboxes2D():
    segmentation = np.zeros((10, 10), dtype=int)
    segmentation[:2, :2] = 1
    segmentation[3:5, 3:6] = 2
    labels_idx = [0, 1, 2]
    typed_labels_idx = List()
    [typed_labels_idx.append(x) for x in labels_idx]

    bboxes = utils._get_bboxes2D(segmentation, typed_labels_idx)

    assert all([k in bboxes for k in labels_idx])
    assert np.all(bboxes[1] == [[0, 0], [2, 2]])
    assert np.all(bboxes[2] == [[3, 3], [5, 6]])
    assert np.all(bboxes[0] == [segmentation.shape, [0, 0]])


def test_p_get_bboxes_2D(mocker):
    mocks = mocker.patch.multiple(
        "panseg.functionals.proofreading.utils",
        _get_bboxes2D=mocker.DEFAULT,
        _get_bboxes3D=mocker.DEFAULT,
    )
    arr = np.zeros((10, 10), dtype=int)

    utils._get_bboxes(arr, [])
    mocks["_get_bboxes2D"].assert_called_once()
    mocks["_get_bboxes3D"].assert_not_called()


def test_get_bboxes_3D(mocker):
    mocks = mocker.patch.multiple(
        "panseg.functionals.proofreading.utils",
        _get_bboxes2D=mocker.DEFAULT,
        _get_bboxes3D=mocker.DEFAULT,
    )
    arr = np.zeros((10, 10, 10), dtype=int)

    utils._get_bboxes(arr, [])
    mocks["_get_bboxes3D"].assert_called_once()
    mocks["_get_bboxes2D"].assert_not_called()


def test_p_get_bboxes_4D(mocker):
    mocks = mocker.patch.multiple(
        "panseg.functionals.proofreading.utils",
        _get_bboxes2D=mocker.DEFAULT,
        _get_bboxes3D=mocker.DEFAULT,
    )
    arr = np.zeros((10, 10, 10, 10), dtype=int)

    with pytest.raises(ValueError):
        utils._get_bboxes(arr, [])


def test_get_bboxes():
    segmentation = np.zeros((10, 10), dtype=int)
    segmentation[:2, :2] = 1
    segmentation[3:5, 4:6] = 2
    out = utils.get_bboxes(segmentation)

    assert np.all(out[0] == [[7, 7], [3, 3]])
    assert np.all(out[1] == [[0, 0], [5, 5]])
    assert np.all(out[2] == [[0, 1], [8, 9]])


def test_get_bboxes_slack():
    segmentation = np.zeros((10, 10), dtype=int)
    segmentation[:2, :2] = 1
    segmentation[3:5, 4:6] = 2
    out = utils.get_bboxes(segmentation, slack=(-1, 2, 2))

    assert np.all(out[0] == [[8, 8], [2, 2]])
    assert np.all(out[1] == [[0, 0], [4, 4]])
    assert np.all(out[2] == [[1, 2], [7, 8]])


def test_get_bboxes_no_slack():
    segmentation = np.zeros((10, 10), dtype=int)
    segmentation[:2, :2] = 1
    segmentation[3:5, 4:6] = 2
    out = utils.get_bboxes(segmentation, slack=(-1, 0, 0))

    assert np.all(out[0] == [[10, 10], [0, 0]])
    assert np.all(out[1] == [[0, 0], [2, 2]])
    assert np.all(out[2] == [[3, 4], [5, 6]])


def test_get_idx_slice():
    bboxes = {
        0: [[7, 7], [3, 3]],
        1: [[0, 0], [8, 4]],
        2: [[0, 1], [7, 8]],
    }
    out = utils.get_idx_slice(1, bboxes)
    assert out[0] == (slice(0, 8), slice(0, 4))
    assert np.all(out[1] == bboxes[1])
    assert np.all(out[2] == [0, 0])

    out = utils.get_idx_slice([1, 2], bboxes)
    assert out[0] == (slice(0, 8), slice(0, 8))
    assert np.all(out[1] == [[0, 0], [8, 8]])
    assert np.all(out[2] == [0, 0])
