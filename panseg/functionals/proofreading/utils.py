import numba
import numpy as np
from numba.typed import List


@numba.njit
def _get_bboxes3D(segmentation, labels_idx):
    shape = segmentation.shape

    bboxes = {}
    for idx in labels_idx:
        _x = np.array([[shape[0], shape[1], shape[2]], [0, 0, 0]])
        bboxes[idx] = _x

    for z in range(shape[0]):
        for x in range(shape[1]):
            for y in range(shape[2]):
                value = segmentation[z, x, y]
                if value > 0:
                    zmin, xmin, ymin = bboxes[value][0]
                    zmax, xmax, ymax = bboxes[value][1]

                    if z < zmin:
                        bboxes[value][0, 0] = z
                    if x < xmin:
                        bboxes[value][0, 1] = x
                    if y < ymin:
                        bboxes[value][0, 2] = y

                    if z >= zmax:
                        bboxes[value][1, 0] = z + 1
                    if x >= xmax:
                        bboxes[value][1, 1] = x + 1
                    if y >= ymax:
                        bboxes[value][1, 2] = y + 1
    return bboxes


@numba.njit
def _get_bboxes2D(segmentation, labels_idx):
    shape = segmentation.shape

    bboxes = {}
    for idx in labels_idx:
        _x = np.array([[shape[0], shape[1]], [0, 0]])
        bboxes[idx] = _x

    for x in range(shape[0]):
        for y in range(shape[1]):
            value = segmentation[x, y]
            if value > 0:
                xmin, ymin = bboxes[value][0]
                xmax, ymax = bboxes[value][1]

                if x < xmin:
                    bboxes[value][0, 0] = x
                if y < ymin:
                    bboxes[value][0, 1] = y

                if x >= xmax:
                    bboxes[value][1, 0] = x + 1
                if y >= ymax:
                    bboxes[value][1, 1] = y + 1
    return bboxes


def _get_bboxes(segmentation, labels_idx):
    typed_labels_idx = List()
    [typed_labels_idx.append(x) for x in labels_idx]
    if len(segmentation.shape) == 3:
        return _get_bboxes3D(segmentation, typed_labels_idx)
    elif len(segmentation.shape) == 2:
        return _get_bboxes2D(segmentation, typed_labels_idx)
    else:
        raise ValueError("Segmentation shape not supported")


def get_bboxes(segmentation, slack=(1, 3, 3)):
    """Get the bounding boxes of all values in the segmentation

    Bounding boxes are half-open intervals [lower, upper)
    """
    segmentation = segmentation.astype("int64")
    labels_idx = np.unique(segmentation)
    bboxes = _get_bboxes(segmentation, labels_idx)

    slack = np.array(slack)

    if len(segmentation.shape) == 2:
        slack = slack[1:]

    bboxes_out = {}
    for key, values in bboxes.items():
        values[0] = np.maximum(values[0] - slack, np.zeros_like(slack))
        values[1] = np.minimum(values[1] + slack, segmentation.shape)
        bboxes_out[int(key)] = values
    return bboxes_out


def get_idx_slice(
    indices, bboxes_dict: dict
) -> tuple[tuple[slice], np.ndarray, np.ndarray]:
    """Merge bounding boxes to get a bigger bounding box

    Returns:
        tuple over all dimensions containing a slice
        array containing the min and max coordinates
        array containing the first dimension of the former array
    """
    if isinstance(indices, int):
        indices = [indices]

    list_min_values = [bboxes_dict[idx][0] for idx in indices]
    list_max_values = [bboxes_dict[idx][1] for idx in indices]
    min_values = np.min(np.stack(list_min_values), axis=0)
    max_values = np.max(np.stack(list_max_values), axis=0)
    values = np.stack([min_values, max_values])

    return (
        tuple([slice(_min, _max) for _min, _max in zip(*values)]),
        values,
        min_values,
    )
