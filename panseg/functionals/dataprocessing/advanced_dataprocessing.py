import copy
import logging

import numba
import numpy as np
import tqdm
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential, watershed

from panseg.functionals.proofreading.utils import get_bboxes

logger = logging.getLogger(__name__)


def get_bbox(
    mask: np.ndarray, pixel_tolerance: int = 0
) -> tuple[tuple[slice, slice, slice], int, int, int]:
    """
    Returns the bounding box around a binary mask with optional padding.
    Bounding boxes are half-open intervals [lower,upper)

    Args:
        mask (np.ndarray): Binary mask to calculate the bounding box.
        pixel_tolerance (int): Padding around the bounding box.

    Returns:
        tuple[tuple[slice, slice, slice], int, int, int]: Bounding box slices and minimum coordinates.
    """

    bboxes_d = get_bboxes(mask, (pixel_tolerance) * 3)
    return tuple([slice(a[0], a[1], None) for a in zip(*bboxes_d[1])])


def get_quantile_mask(
    counts: np.ndarray, quantile_range: tuple[float, float] = (0.2, 0.99)
) -> np.ndarray:
    """
    Filters counts by quantiles.

    Args:
        counts (np.ndarray): Array of counts to filter.
        quantile_range (tuple[float, float]): Lower and upper quantiles.

    Returns:
        np.ndarray: Boolean mask indicating which counts are within the quantile range.
    """
    lower_mask = counts > np.quantile(counts, quantile_range[0])
    upper_mask = counts < np.quantile(counts, quantile_range[1])
    return np.logical_and(lower_mask, upper_mask)


@numba.njit(parallel=True)
def numba_find_overlaps(
    cell_seg: np.ndarray, nuc_seg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds overlaps between cell segmentation and nuclei segmentation.

    Args:
        cell_seg (np.ndarray): Cell segmentation array.
        nuc_seg (np.ndarray): Nuclei segmentation array.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Cell sizes, nuclei sizes, and pixel overlap counts.
    """
    shape_x, shape_y, shape_z = cell_seg.shape

    cell_counts = np.zeros(cell_seg.max() + 1, dtype=np.uint16)
    n_counts = np.zeros(nuc_seg.max() + 1, dtype=np.uint16)
    overlap_counts = np.zeros((cell_seg.max() + 1, nuc_seg.max() + 1), dtype=np.uint16)

    for x in numba.prange(shape_x):
        for y in range(shape_y):
            for z in range(shape_z):
                seg_id = cell_seg[x, y, z]
                if seg_id > 0:
                    cell_counts[seg_id] += 1

                n_id = nuc_seg[x, y, z]
                if n_id > 0:
                    n_counts[n_id] += 1

                if n_id > 0 and seg_id > 0:
                    overlap_counts[seg_id, n_id] += 1

    return cell_counts, n_counts, overlap_counts


def find_potential_under_seg(
    nuclei_counts: np.ndarray,
    cell_counts: np.ndarray,
    intersection_counts: np.ndarray,
    threshold: float = 0.9,
    quantiles_clip: tuple[float, float] = (0.2, 0.99),
) -> dict[int, dict[str, np.ndarray]]:
    """
    Identifies potential under-segmentation by analyzing overlap between cells and nuclei.

    Args:
        nuclei_counts (np.ndarray): Array of nuclei sizes.
        cell_counts (np.ndarray): Array of cell sizes.
        intersection_counts (np.ndarray): Overlap between cells and nuclei.
        threshold (float): Minimum overlap ratio to consider a cell under-segmented.
        quantiles_clip (tuple[float, float]): Quantile range for filtering nuclei.

    Returns:
        dict[int, dict[str, np.ndarray]]: Mapping of cell indices to their overlap profile with nuclei.
    """
    nuclei_mask = get_quantile_mask(nuclei_counts, quantiles_clip)
    cell_assignment = {}

    for cell_idx in range(cell_counts.shape[0]):
        nuclei_idx = np.nonzero(intersection_counts[cell_idx])[0]
        overlap_ratios = (
            intersection_counts[cell_idx, nuclei_idx] / nuclei_counts[nuclei_idx]
        )

        under_seg_nuclei = nuclei_idx[
            (overlap_ratios > threshold) & nuclei_mask[nuclei_idx]
        ]

        if len(under_seg_nuclei) > 1:
            cell_assignment[cell_idx] = {
                "n_idx": nuclei_idx,
                "under_seg_idx": under_seg_nuclei,
                "is_under_seg": True,
                "r_intersection": overlap_ratios,
            }

    return cell_assignment


def find_potential_over_seg(
    nuclei_counts: np.ndarray, intersection_counts: np.ndarray, threshold: float = 0.3
) -> dict[int, dict[str, np.ndarray]]:
    """
    Identifies potential over-segmentation by analyzing overlap between cells and nuclei.

    Args:
        nuclei_counts (np.ndarray): Array of nuclei sizes.
        intersection_counts (np.ndarray): Overlap between cells and nuclei.
        threshold (float): Minimum overlap ratio to consider a nucleus over-segmented.

    Returns:
        dict[int, dict[str, np.ndarray]]: Mapping of nuclei indices to their overlap profile with cells.
    """
    nuclei_assignment = {}

    for nuclei_idx in range(nuclei_counts.shape[0]):
        cell_idx = np.nonzero(intersection_counts[:, nuclei_idx])[0]
        overlap_ratios = (
            intersection_counts[cell_idx, nuclei_idx] / nuclei_counts[nuclei_idx]
        )

        over_seg_cells = cell_idx[overlap_ratios > threshold]

        if len(over_seg_cells) > 1:
            nuclei_assignment[nuclei_idx] = {
                "c_idx": cell_idx,
                "over_seg_idx": over_seg_cells,
                "is_over_seg": True,
                "r_intersection": overlap_ratios,
            }

    return nuclei_assignment


def split_from_seeds(
    segmentation: np.ndarray,
    boundary_pmap: np.ndarray,
    seeds: np.ndarray,
    all_idx: list[int],
) -> np.ndarray:
    """
    Splits a segmentation using seeded watershed.

    Args:
        segmentation (np.ndarray): Input segmentation array.
        boundary_pmap (np.ndarray): Boundary probability map.
        seeds (np.ndarray): Seed markers for watershed.
        all_idx (list[int]): List of indices to split.

    Returns:
        np.ndarray: Segmentation array after splitting.
    """
    segmentation_copy = copy.deepcopy(segmentation)
    mask = np.isin(segmentation_copy, all_idx)
    bbox = get_bbox(mask)

    cropped_boundary_pmap = boundary_pmap[bbox]
    cropped_seeds = seeds[bbox]
    cropped_mask = np.isin(segmentation_copy[bbox], all_idx)

    smoothed_pmap = gaussian(
        cropped_boundary_pmap / cropped_boundary_pmap.max(), sigma=2.0
    )
    local_seg = watershed(smoothed_pmap, markers=cropped_seeds, compactness=0.001)

    local_seg += segmentation_copy.max() + 1
    segmentation_copy[bbox][cropped_mask] = local_seg[cropped_mask]

    return segmentation_copy


def fix_under_segmentation(
    segmentation: np.ndarray,
    nuclei_segmentation: np.ndarray,
    boundary_pmap: np.ndarray,
    cell_assignments: dict[int, dict[str, np.ndarray]],
    cell_idx: list[int] | None = None,
) -> np.ndarray:
    """
    Attempts to fix cell under-segmentation by splitting cells with multiple nuclei.

    Args:
        segmentation (np.ndarray): Input cell segmentation.
        nuclei_segmentation (np.ndarray): Nuclei segmentation.
        boundary_pmap (np.ndarray): Boundary probability map.
        cell_assignments (dict[int, dict[str, np.ndarray]]): Under-segmentation information for cells.
        cell_idx (list[int] | None): Specific cell indices to process. If None, process all.

    Returns:
        np.ndarray: Segmentation array after fixing under-segmentation.
    """
    segmentation_copy = copy.deepcopy(segmentation)
    logger.info("Fixing under-segmentation...")

    for c_idx, assignment in tqdm.tqdm(cell_assignments.items()):
        if cell_idx is None or c_idx in cell_idx:
            seeds = np.zeros_like(segmentation_copy)
            for i, n_idx in enumerate(assignment["under_seg_idx"]):
                seeds[nuclei_segmentation == n_idx] = i + 1

            segmentation_copy = split_from_seeds(
                segmentation_copy, boundary_pmap, seeds, all_idx=[c_idx]
            )

    return segmentation_copy


def fix_over_segmentation(
    segmentation: np.ndarray,
    nuclei_assignments: dict[int, dict[str, np.ndarray]],
    nuclei_idx: list[int] | None = None,
) -> np.ndarray:
    """
    Attempts to fix cell over-segmentation by merging cells that share the same nucleus.

    Args:
        segmentation (np.ndarray): Input cell segmentation.
        nuclei_assignments (dict[int, dict[str, np.ndarray]]): Over-segmentation information for nuclei.
        nuclei_idx (list[int] | None): Specific nuclei indices to process. If None, process all.

    Returns:
        np.ndarray: Segmentation array after fixing over-segmentation.
    """
    segmentation_copy = copy.deepcopy(segmentation)
    logger.info("Fixing over-segmentation...")

    for n_idx, assignment in tqdm.tqdm(nuclei_assignments.items()):
        if nuclei_idx is None or n_idx in nuclei_idx:
            target_value = assignment["over_seg_idx"][0]
            for c_idx in assignment["over_seg_idx"]:
                segmentation_copy[segmentation == c_idx] = target_value

    return segmentation_copy


def fix_over_under_segmentation_from_nuclei(
    cell_seg: np.ndarray,
    nuclei_seg: np.ndarray,
    threshold_merge: float,
    threshold_split: float,
    quantile_min: float,
    quantile_max: float,
    boundary: np.ndarray | None = None,
) -> np.ndarray:
    """
    Correct over-segmentation and under-segmentation of cells based on nuclei information.

    This function uses information from nuclei segmentation to refine cell segmentation by first identifying
    over-segmented cells (cells mistakenly split into multiple segments) and merging them. It then corrects
    under-segmented cells (multiple nuclei within a single cell) by splitting them based on nuclei position
    and optional boundary information.

    Args:
        cell_seg (np.ndarray): A 2D or 3D array of segmented cells, where each integer represents a unique cell.
        nuclei_seg (np.ndarray): A 2D or 3D array of segmented nuclei, matching the shape of `cell_seg`.
            Used to guide merging and splitting.
        threshold_merge (float, optional): A value between 0 and 1. Cells with less than this fraction of nuclei overlap
            are considered over-segmented and will be merged. Default is 0.33.
        threshold_split (float, optional): A value between 0 and 1. Cells with more than this fraction of nuclei overlap
            are considered under-segmented and will be split. Default is 0.66.
        quantile_min (float, optional): The lower size limit for nuclei, as a fraction (0-1). Nuclei smaller than this
            quantile are ignored. Default is 0.3.
        quantile_max (float, optional): The upper size limit for nuclei, as a fraction (0-1). Nuclei larger than this
            quantile are ignored. Default is 0.99.
        boundary (np.ndarray | None, optional): Optional boundary map of the same shape as `cell_seg`. High values
            indicate cell boundaries and help refine splitting. If None, all regions are treated equally.

    Returns:
        np.ndarray: Corrected cell segmentation array.
    """
    # Find overlaps between cells and nuclei
    cell_counts, nuclei_counts, cell_nuclei_counts = numba_find_overlaps(
        cell_seg, nuclei_seg
    )

    # Identify over-segmentation and correct it
    nuclei_assignments = find_potential_over_seg(
        nuclei_counts, cell_nuclei_counts, threshold=threshold_merge
    )
    corrected_seg = fix_over_segmentation(cell_seg, nuclei_assignments)

    # Identify under-segmentation and correct it
    cell_counts, nuclei_counts, cell_nuclei_counts = numba_find_overlaps(
        corrected_seg, nuclei_seg
    )
    cell_assignments = find_potential_under_seg(
        nuclei_counts,
        cell_counts,
        cell_nuclei_counts,
        threshold=threshold_split,
        quantiles_clip=(quantile_min, quantile_max),
    )

    boundary_pmap = np.ones_like(cell_seg) if boundary is None else boundary
    return fix_under_segmentation(
        corrected_seg, nuclei_seg, boundary_pmap, cell_assignments, cell_idx=None
    )


def remove_false_positives_by_foreground_probability(
    segmentation: np.ndarray,
    foreground: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits an instance segmentation into two based on a foreground probability threshold.

    1. Relabels the input segmentation sequentially (preserving 0 as background).
    2. Computes the mean foreground probability for each region.
    3. Assigns regions with mean >= threshold to the `kept` map; the rest to the `removed` map.
    4. Relabels both outputs sequentially for compact labeling.

    Args:
        segmentation (np.ndarray): 2D or 3D segmentation array; each non-zero integer is a distinct region.
        foreground (np.ndarray): Same shape as `segmentation`, values in [0, 1] representing per-pixel probabilities.
        threshold (float): Regions with mean probability below this are considered false positives.

    Returns:
        kept (np.ndarray): Segmentation of regions with mean probability >= threshold, relabeled sequentially.
        removed (np.ndarray): Segmentation of regions with mean probability < threshold, relabeled sequentially.
    """
    # TODO: use `relabel_sequential` to recover the original labels

    if segmentation.shape != foreground.shape:
        raise ValueError("Segmentation and probability map must have the same shape.")
    if foreground.min() < 0 or foreground.max() > 1:
        raise ValueError("Foreground must be a probability map with values in [0, 1].")

    instances, _, _ = relabel_sequential(
        segmentation
    )  # The label 0 is assumed to denote the bg and is never remapped.

    kept = np.zeros_like(instances, dtype=instances.dtype)
    removed = np.zeros_like(instances, dtype=instances.dtype)

    # Measure and split
    regions = regionprops(instances)
    for region in tqdm.tqdm(regions, desc="Splitting regions"):
        bbox = region.bbox
        if instances.ndim == 3:
            slices = (
                slice(bbox[0], bbox[3]),
                slice(bbox[1], bbox[4]),
                slice(bbox[2], bbox[5]),
            )
        else:
            slices = (
                slice(bbox[0], bbox[2]),
                slice(bbox[1], bbox[3]),
            )

        mask = instances[slices] == region.label
        prob = foreground[slices][mask]
        mean_prob = prob.mean() if prob.size > 0 else 0.0

        if mean_prob >= threshold:
            kept[slices][mask] = region.label
        else:
            removed[slices][mask] = region.label

    # Relabel outputs so labels are compact
    kept, _, _ = relabel_sequential(kept)
    removed, _, _ = relabel_sequential(removed)

    return kept, removed
