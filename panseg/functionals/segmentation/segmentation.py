import logging
import math
import multiprocessing
import os

import nifty
import numpy as np
from elf.segmentation import (
    GaspFromAffinities,
    project_node_labels_to_pixels,
    stacked_watershed,
)
from elf.segmentation import lifted_multicut as lmc
from elf.segmentation.features import (
    compute_rag,
    lifted_problem_from_probabilities,
    lifted_problem_from_segmentation,
)
from elf.segmentation.multicut import multicut_kernighan_lin
from elf.segmentation.watershed import (
    apply_size_filter,
    blockwise_two_pass_watershed,
    distance_transform_watershed,
)
from vigra.filters import gaussianSmoothing

from panseg.functionals.segmentation.utils import compute_mc_costs, shift_affinities

logger = logging.getLogger(__name__)

try:
    import SimpleITK as sitk  # type: ignore[import]

    SIMPLE_ITK_INSTALLED = True
except ImportError:
    SIMPLE_ITK_INSTALLED = False


# Conservative estimate of the temporary working set per voxel of a halo-expanded block
# while 'distance_transform_watershed' runs on it: thresholded mask (1B) and its inverted
# copy (1B), distance transform and smoothed copy (2x4B), seed mask and labeled seeds
# (1B + 4B), smoothed input and weight map (2x4B), watershed height/marker/output copies
# (3x4B), size-filter temporaries (2x1B) and the second watershed copies (2x4B), the
# uint64 result (8B) and the pass-two relabel copies (2x8B).
_WS_BYTES_PER_VOXEL = 64

# Full-volume allocations that exist regardless of the block layout: the float32 input
# copy, the uint64 output and the full-volume relabel copy inside the two-pass watershed.
_FIXED_BYTES_PER_VOXEL = 24

# Fraction of the currently available RAM used as the default memory budget.
_DEFAULT_MEMORY_FRACTION = 0.5


def _effective_cpu_count() -> int:
    """Return the number of CPU cores available to this process.

    Unlike 'multiprocessing.cpu_count', this respects CPU affinity masks and container
    restrictions where the platform supports it (all platforms on Python >= 3.13, Linux
    and other Unix platforms on earlier versions), so it does not report host cores that
    the process may not use.

    Returns:
        int: The number of available CPU cores.
    """
    # 'os.process_cpu_count' is available on all platforms since Python 3.13.
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        return process_cpu_count()
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return multiprocessing.cpu_count()


def _sysconf_first(*names: str) -> int | None:
    """Return the first 'os.sysconf' value that exists on this platform, or None.

    The names exposed by 'os.sysconf' vary between Unix platforms (e.g. macOS exposes
    'SC_PHYS_PAGES' where Linux exposes 'SC_AVPHYS_PAGES').
    """
    for name in names:
        try:
            return int(os.sysconf(name))
        except (AttributeError, OSError, ValueError):
            continue
    return None


def _available_memory_windows() -> int | None:
    """Return the amount of available physical memory on Windows, if detectable.

    Uses the WinAPI call 'GlobalMemoryStatusEx' via ctypes (no extra dependency).
    """
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dw_length", ctypes.c_ulong),
                ("dw_memory_load", ctypes.c_ulong),
                ("ull_total_phys", ctypes.c_ulonglong),
                ("ull_avail_phys", ctypes.c_ulonglong),
                ("ull_total_page_file", ctypes.c_ulonglong),
                ("ull_avail_page_file", ctypes.c_ulonglong),
                ("ull_total_virtual", ctypes.c_ulonglong),
                ("ull_avail_virtual", ctypes.c_ulonglong),
                ("ull_avail_extended_virtual", ctypes.c_ulonglong),
            ]

        windll = getattr(ctypes, "windll", None)  # only exists on Windows
        if windll is None:
            return None
        stat = MemoryStatusEx()
        stat.dw_length = ctypes.sizeof(MemoryStatusEx)
        if windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ull_avail_phys)
    except (AttributeError, OSError):
        pass
    return None


def _available_memory_bytes() -> int | None:
    """Return the amount of memory currently available on the system, if detectable.

    Uses 'os.sysconf' on Unix and 'GlobalMemoryStatusEx' on Windows.

    Returns:
        Optional[int]: Available memory in bytes, or None if it cannot be determined
        on this platform.
    """
    pages = _sysconf_first("SC_AVPHYS_PAGES", "SC_PHYS_PAGES")
    page_size = _sysconf_first("SC_PAGE_SIZE", "SC_PAGESIZE")
    if pages is not None and page_size is not None:
        return pages * page_size
    return _available_memory_windows()


def _default_block_shape(
    shape: tuple[int, ...], n_threads: int | None
) -> tuple[int, ...]:
    """Derive a block shape for blockwise processing of a volume with the given shape.

    Starting from a single block, dimensions are greedily split (always the one with the
    largest extent per block) until the total number of blocks reaches max(2, 2 * n_threads)
    or until no dimension can be split further without yielding block dimensions of less
    than 32 voxels.

    Args:
        shape (tuple[int, ...]): Shape of the volume to split.
        n_threads (Optional[int]): Number of threads to target. If None, the number of
            CPU cores available to this process is used.

    Returns:
        tuple[int, ...]: The derived block shape.
    """
    n_threads = _effective_cpu_count() if n_threads is None else n_threads
    target_blocks = max(2, 2 * n_threads)
    blocks = [1] * len(shape)
    while int(np.prod(blocks)) < target_blocks:
        splittable = [d for d in range(len(shape)) if shape[d] / (blocks[d] + 1) >= 32]
        if not splittable:
            break
        split_dim = max(splittable, key=lambda d: shape[d] / blocks[d])
        blocks[split_dim] += 1
    return tuple(int(np.ceil(shape[d] / blocks[d])) for d in range(len(shape)))


def _default_halo(block_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Derive a halo for blockwise processing with the given block shape.

    Args:
        block_shape (tuple[int, ...]): Block shape to derive the halo from.

    Returns:
        tuple[int, ...]: The derived halo, capped at 32 voxels per dimension.
    """
    return tuple(min(32, b // 3) for b in block_shape)


def _blockwise_n_workers(
    n_voxels: int,
    block_shape: tuple[int, ...],
    halo: tuple[int, ...],
    n_threads: int,
    memory_budget: int | None,
) -> int:
    """Derive the number of blocks to process concurrently in the blockwise mode.

    The working set of a block (the halo-expanded block plus the temporary arrays of the
    distance transform watershed) is estimated from '_WS_BYTES_PER_VOXEL'. The number of
    concurrently processed blocks is capped so that the total estimated working set,
    including the full-volume allocations that exist regardless of blocking, stays within
    the memory budget. Excess blocks are processed in waves.

    Args:
        n_voxels (int): Number of voxels of the volume.
        block_shape (tuple[int, ...]): Block shape in voxels.
        halo (tuple[int, ...]): Halo in voxels added around each block.
        n_threads (int): Requested number of threads (upper bound of the return value).
        memory_budget (Optional[int]): Memory budget in bytes. If None, 50% of the
            currently available RAM is used, or no capping (with a warning) is applied
            if the available memory cannot be determined.

    Returns:
        int: Number of blocks to process concurrently (at least 1, at most n_threads).
    """
    if memory_budget is None:
        available = _available_memory_bytes()
        budget = (
            None if available is None else int(_DEFAULT_MEMORY_FRACTION * available)
        )
    else:
        budget = int(memory_budget)

    if budget is None:
        logger.warning(
            "dt_watershed: could not determine the available memory; running "
            "without a memory budget. Pass 'memory_budget' explicitly to bound "
            "the peak memory usage."
        )
        return n_threads

    fixed_bytes = _FIXED_BYTES_PER_VOXEL * n_voxels
    if fixed_bytes >= budget:
        logger.warning(
            "dt_watershed: the volume itself needs an estimated %.2f GB which exceeds "
            "the memory budget of %.2f GB; blockwise processing may exhaust memory.",
            fixed_bytes / 1e9,
            budget / 1e9,
        )
        return n_threads

    outer_shape = tuple(int(b) + 2 * int(h) for b, h in zip(block_shape, halo))
    per_block_bytes = math.prod(outer_shape) * _WS_BYTES_PER_VOXEL
    n_workers = max(1, min(n_threads, (budget - fixed_bytes) // per_block_bytes))
    if n_workers < n_threads:
        logger.info(
            "dt_watershed: limiting blockwise watershed to %d concurrent blocks "
            "(from %d threads) to stay within the memory budget of %.2f GB.",
            n_workers,
            n_threads,
            budget / 1e9,
        )
    return n_workers


def dt_watershed(
    boundary_pmaps: np.ndarray,
    threshold: float = 0.5,
    sigma_seeds: float = 1.0,
    stacked: bool = False,
    sigma_weights: float = 2.0,
    min_size: int = 100,
    alpha: float = 1.0,
    pixel_pitch: tuple[int, ...] | None = None,
    apply_nonmax_suppression: bool = False,
    n_threads: int | None = None,
    blockwise: bool = False,
    block_shape: tuple[int, ...] | None = None,
    halo: tuple[int, ...] | None = None,
    mask: np.ndarray | None = None,
    memory_budget: int | None = None,
) -> np.ndarray:
    """Performs watershed segmentation using distance transforms on boundary probability maps.

    This function applies the distance transform watershed algorithm to the input boundary
    probability maps, either slice-by-slice or in original shape depending on the 'stacked' parameter.
    The watershed method is applied to the boundary probability maps with optional pre-processing
    like thresholding, smoothing, and masking.

    Args:
        boundary_pmaps (np.ndarray): Input array of boundary probability maps, often obtained
            from deep learning models. Each pixel/voxel value represents the probability of
            being part of a boundary.
        threshold (float, optional): Threshold value applied to the boundary probability map
            before computing the distance transform. Values below this threshold are considered
            background. Defaults to 0.5.
        sigma_seeds (float, optional): Standard deviation for Gaussian smoothing applied to
            the seed map (used for initializing the watershed regions). Higher values result
            in more smoothed seeds. Defaults to 1.0.
        stacked (bool, optional): If True, performs watershed segmentation on each 2D slice of
            a 3D volume independently (slice-by-slice). If False, performs watershed segmentation
            in 3D for volumetric data or 2D for 2D input. Defaults to False.
        sigma_weights (float, optional): Standard deviation for Gaussian smoothing applied
            to the weight map. The weight map combines the distance transform and input map
            to guide the watershed. Larger values result in smoother weight maps. Defaults to 2.0.
        min_size (int, optional): Minimum size of the segmented regions to retain. Regions
            smaller than this size are removed. Defaults to 100.
        alpha (float, optional): Blending factor to combine the input boundary probability maps
            and the distance transform when constructing the weight map. A higher alpha
            prioritizes the input maps. Defaults to 1.0.
        pixel_pitch (Optional[tuple[int, ...]], optional): Voxel anisotropy factors (e.g., spacing
            along different axes) to use during the distance transform. If None, the distances are
            computed isotropically. For anisotropic volumes, this should match the voxel spacing.
            Defaults to None.
        apply_nonmax_suppression (bool, optional): If True, applies non-maximum suppression to
            the detected seeds, reducing seed redundancy. This requires the Nifty library.
            Defaults to False.
        n_threads (Optional[int], optional): Number of threads to use for parallel processing in
            2D mode (stacked mode) and 3D blockwise mode. If None, the default number of threads
            (all CPU cores available to this process) will be used. Defaults to None.
        blockwise (bool, optional): If True, runs the 3D watershed blockwise across multiple
            CPU cores instead of a single full-volume pass. Falls back to the single-pass
            watershed (with a warning) if the input is 2D, if the volume has fewer than
            2,000,000 voxels, or if the block layout yields fewer than two blocks. Ignored
            when 'stacked' is True. Defaults to False.
        block_shape (Optional[tuple[int, ...]], optional): Block shape in voxels for the
            blockwise mode. If None, it is derived from the volume shape and 'n_threads'.
            Dimensions are clamped to the volume shape. Defaults to None.
        halo (Optional[tuple[int, ...]], optional): Halo size in voxels added around each block
            in the blockwise mode. If None, it is derived from the block shape. Dimensions are
            clamped to half of the corresponding block dimension. Defaults to None.
        mask (Optional[np.ndarray], optional): A binary mask that excludes certain regions from
            segmentation. Only regions within the mask will be considered. If None, all regions
            are included. Must have the same shape as 'boundary_pmaps'. Defaults to None.
        memory_budget (Optional[int], optional): Approximate memory budget in bytes for the
            blockwise mode. The number of blocks processed concurrently is capped so that the
            estimated working set stays within this budget; remaining blocks are processed in
            waves. If None, 50% of the currently available RAM is used; if the available
            memory cannot be determined, no capping is applied and a warning is emitted.
            Defaults to None.

    Returns:
        np.ndarray: A labeled segmentation map where each region is assigned a unique label.

    """
    # Prepare the keyword arguments for the watershed function
    boundary_pmaps = boundary_pmaps.astype("float32", copy=False)
    n_threads = _effective_cpu_count() if n_threads is None else n_threads
    ws_kwargs = {
        "threshold": threshold,
        "sigma_seeds": sigma_seeds,
        "sigma_weights": sigma_weights,
        "min_size": min_size,
        "alpha": alpha,
        "pixel_pitch": pixel_pitch,
        "apply_nonmax_suppression": apply_nonmax_suppression,
    }
    if stacked:
        # Apply watershed slice by slice (for 3D data)
        segmentation, _ = stacked_watershed(
            boundary_pmaps,
            ws_function=distance_transform_watershed,
            n_threads=n_threads,
            mask=mask,
            **ws_kwargs,
        )
    elif blockwise:
        shape = boundary_pmaps.shape
        if block_shape is None:
            resolved_block_shape = _default_block_shape(shape, n_threads)
        else:
            # Nifty requires the block shape to fit within the volume
            resolved_block_shape = tuple(
                min(int(b), int(s)) for b, s in zip(block_shape, shape)
            )
        if halo is None:
            resolved_halo = _default_halo(resolved_block_shape)
        else:
            # Nifty requires the halo to be smaller than the block size
            resolved_halo = tuple(
                min(int(h), b // 2) for h, b in zip(halo, resolved_block_shape)
            )
        n_blocks = int(
            np.prod(
                np.ceil(
                    np.array(shape, dtype="float64") / np.array(resolved_block_shape)
                )
            )
        )
        if (
            boundary_pmaps.ndim == 3
            and boundary_pmaps.size >= 2_000_000
            and n_blocks >= 2
        ):
            # Cap the number of concurrently processed blocks so that the estimated
            # working set stays within the memory budget; excess blocks run in waves.
            n_workers = _blockwise_n_workers(
                boundary_pmaps.size,
                resolved_block_shape,
                resolved_halo,
                n_threads,
                memory_budget,
            )
            # Apply watershed blockwise in two passes (for large 3D data)
            logger.debug(
                "Running blockwise watershed, "
                f"shape:{resolved_block_shape}, halo: {resolved_halo}, "
                f"concurrent blocks: {n_workers}"
            )
            segmentation, _ = blockwise_two_pass_watershed(
                boundary_pmaps,
                resolved_block_shape,
                resolved_halo,
                ws_function=distance_transform_watershed,
                n_threads=n_workers,
                mask=mask,
                **ws_kwargs,
            )
        else:
            if boundary_pmaps.ndim != 3:
                reason = "the input is 2D"
            elif boundary_pmaps.size < 2_000_000:
                reason = "the volume has fewer than 2,000,000 voxels"
            else:
                reason = "the block layout yields fewer than two blocks"
            logger.warning(
                "dt_watershed: blockwise mode not applicable (%s); "
                "falling back to the single-pass watershed.",
                reason,
            )
            segmentation, _ = distance_transform_watershed(
                boundary_pmaps, mask=mask, **ws_kwargs
            )
    else:
        # Apply watershed in 3D for 3D data or in 2D for 2D data
        segmentation, _ = distance_transform_watershed(
            boundary_pmaps, mask=mask, **ws_kwargs
        )

    return segmentation


def gasp(
    boundary_pmaps: np.ndarray,
    superpixels: np.ndarray | None = None,
    gasp_linkage_criteria: str = "average",
    beta: float = 0.5,
    post_minsize: int = 100,
    n_threads: int = 6,
) -> np.ndarray:
    """
    Perform segmentation using the GASP algorithm with affinity maps.

    Args:
        boundary_pmaps (np.ndarray): Cell boundary prediction.
        superpixels (Optional[np.ndarray]): Superpixel segmentation. If None, GASP will be run from the pixels. Default is None.
        gasp_linkage_criteria (str): Linkage criteria for GASP. Default is 'average'.
        beta (float): Beta parameter for GASP. Small values steer towards under-segmentation, while high values bias towards over-segmentation. Default is 0.5.
        post_minsize (int): Minimum size of the segments after GASP. Default is 100.
        n_threads (int): Number of threads used for GASP. Default is 6.

    Returns:
        np.ndarray: GASP output segmentation.
    """
    remove_singleton = False
    if superpixels is not None:
        assert boundary_pmaps.shape == superpixels.shape, (
            "Shape mismatch between boundary_pmaps and superpixels."
        )
        if superpixels.ndim == 2:  # Ensure superpixels is 3D if provided
            superpixels = superpixels[None, ...]
            boundary_pmaps = boundary_pmaps[None, ...]
            remove_singleton = True

    # Prepare the arguments for running GASP
    run_GASP_kwargs = {
        "linkage_criteria": gasp_linkage_criteria,
        "add_cannot_link_constraints": False,
        "use_efficient_implementations": False,
    }

    # Interpret boundary_pmaps as affinities and prepare for GASP
    boundary_pmaps = boundary_pmaps.astype("float32")
    affinities = np.stack([boundary_pmaps] * 3, axis=0)

    offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    # Shift is required to correct aligned affinities
    affinities = shift_affinities(affinities, offsets=offsets)

    # invert affinities
    affinities = 1 - affinities

    # Initialize and run GASP
    gasp_instance = GaspFromAffinities(
        offsets,
        superpixel_generator=None
        if superpixels is None
        else (lambda *args, **kwargs: superpixels),
        run_GASP_kwargs=run_GASP_kwargs,
        n_threads=n_threads,
        beta_bias=beta,
    )
    segmentation, _ = gasp_instance(affinities)

    # Apply size filtering if specified
    if post_minsize > 0:
        segmentation, _ = apply_size_filter(
            segmentation.astype("uint32"), boundary_pmaps, post_minsize
        )

    if remove_singleton:
        segmentation = segmentation[0]

    return segmentation


def mutex_ws(
    boundary_pmaps: np.ndarray,
    superpixels: np.ndarray | None = None,
    beta: float = 0.5,
    post_minsize: int = 100,
    n_threads: int = 6,
) -> np.ndarray:
    """
    Wrapper around gasp with mutex_watershed as linkage criteria.

    Args:magicgui
        boundary_pmaps (np.ndarray): cell boundary prediction. 3D array of shape (Z, Y, X) with values between 0 and 1.
        superpixels (np.ndarray): superpixel segmentation. Must have the same shape as boundary_pmaps.
            If None, GASP will be run from the pixels. (default: None)
        beta (float): beta parameter for GASP. A small value will steer the segmentation towards under-segmentation.
            While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after GASP. (default: 100)
        n_threads (int): number of threads used for GASP. (default: 6)

    Returns:
        segmentation (np.ndarray): MutexWS output segmentation

    """
    return gasp(
        boundary_pmaps=boundary_pmaps,
        superpixels=superpixels,
        gasp_linkage_criteria="mutex_watershed",
        beta=beta,
        post_minsize=post_minsize,
        n_threads=n_threads,
    )


def multicut(
    boundary_pmaps: np.ndarray,
    superpixels: np.ndarray,
    beta: float = 0.5,
    post_minsize: int = 50,
) -> np.ndarray:
    """
    Multicut segmentation from boundary prediction.

    Args:
        boundary_pmaps (np.ndarray): cell boundary prediction, 3D array of shape (Z, Y, X) with values between 0 and 1.
        superpixels (np.ndarray): superpixel segmentation. Must have the same shape as boundary_pmaps.
        beta (float): beta parameter for the Multicut. A small value will steer the segmentation towards
            under-segmentation. While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after Multicut. (default: 100)

    Returns:
        segmentation (np.ndarray): Multicut output segmentation
    """

    # elf requires label arrays with 32/64-bit integer dtypes
    superpixels = superpixels.astype("uint32")
    rag = compute_rag(superpixels)

    # Prob -> edge costs
    boundary_pmaps = boundary_pmaps.astype("float32")
    costs = compute_mc_costs(boundary_pmaps, rag, superpixels, beta=beta)

    # Solving Multicut
    node_labels = multicut_kernighan_lin(rag, costs)
    segmentation = nifty.tools.take(node_labels, superpixels)

    # run size threshold
    if post_minsize > 0:
        segmentation, _ = apply_size_filter(
            segmentation.astype("uint32"), boundary_pmaps, post_minsize
        )
    return segmentation


def lifted_multicut_from_nuclei_pmaps(
    boundary_pmaps: np.ndarray,
    nuclei_pmaps: np.ndarray,
    superpixels: np.ndarray,
    beta: float = 0.5,
    post_minsize: int = 50,
) -> np.ndarray:
    """
    Lifted Multicut segmentation from boundary prediction and nuclei prediction.

    Args:
        boundary_pmaps (np.ndarray): cell boundary prediction, 3D array of shape (Z, Y, X) with values between 0 and 1.
        nuclei_pmaps (np.ndarray): nuclei prediction. Must have the same shape as boundary_pmaps and
            with values between 0 and 1.
        superpixels (np.ndarray): superpixel segmentation. Must have the same shape as boundary_pmaps.
        beta (float): beta parameter for the Multicut. A small value will steer the segmentation towards
            under-segmentation. While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after Multicut. (default: 100)

    Returns:
        segmentation (np.ndarray): Multicut output segmentation
    """
    if nuclei_pmaps.max() > 1 or nuclei_pmaps.min() < 0:
        raise ValueError("nuclei_pmaps should be between 0 and 1")

    # elf requires label arrays with 32/64-bit integer dtypes
    superpixels = superpixels.astype("uint32")

    # compute the region adjacency graph
    rag = compute_rag(superpixels)

    # compute multi cut edges costs
    boundary_pmaps = boundary_pmaps.astype("float32")
    costs = compute_mc_costs(boundary_pmaps, rag, superpixels, beta)

    # assert nuclei pmaps are floats
    nuclei_pmaps = nuclei_pmaps.astype("float32")
    input_maps = [nuclei_pmaps]
    assignment_threshold = 0.9

    # compute lifted multicut features from boundary pmaps
    lifted_uvs, lifted_costs = lifted_problem_from_probabilities(
        rag,
        superpixels.astype("uint32"),
        input_maps,
        assignment_threshold,
        graph_depth=4,
    )

    # solve the full lifted problem using the kernighan lin approximation introduced in
    # http://openaccess.thecvf.com/content_iccv_2015/html/Keuper_Efficient_Decomposition_of_ICCV_2015_paper.html
    node_labels = lmc.lifted_multicut_kernighan_lin(
        rag, costs, lifted_uvs, lifted_costs
    )
    segmentation = project_node_labels_to_pixels(rag, superpixels, node_labels)

    # run size threshold
    if post_minsize > 0:
        segmentation, _ = apply_size_filter(
            segmentation.astype("uint32"), boundary_pmaps, post_minsize
        )
    return segmentation


def lifted_multicut_from_nuclei_segmentation(
    boundary_pmaps: np.ndarray,
    nuclei_seg: np.ndarray,
    superpixels: np.ndarray,
    beta: float = 0.5,
    post_minsize: int = 50,
) -> np.ndarray:
    """
    Lifted Multicut segmentation from boundary prediction and nuclei segmentation.

    Args:
        boundary_pmaps (np.ndarray): cell boundary prediction, 3D array of shape (Z, Y, X) with values between 0 and 1.
        nuclei_seg (np.ndarray): Nuclei segmentation. Must have the same shape as boundary_pmaps.
        superpixels (np.ndarray): superpixel segmentation. Must have the same shape as boundary_pmaps.
        beta (float): beta parameter for the Multicut. A small value will steer the segmentation towards
            under-segmentation. While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after Multicut. (default: 100)

    Returns:
        segmentation (np.ndarray): Multicut output segmentation
    """
    # elf requires label arrays with 32/64-bit integer dtypes
    superpixels = superpixels.astype("uint32")

    # compute the region adjacency graph
    rag = compute_rag(superpixels)

    # compute multi cut edges costs
    boundary_pmaps = boundary_pmaps.astype("float32")
    costs = compute_mc_costs(boundary_pmaps, rag, superpixels, beta)
    max_cost = np.abs(np.max(costs))
    lifted_uvs, lifted_costs = lifted_problem_from_segmentation(
        rag,
        superpixels,
        nuclei_seg,
        overlap_threshold=0.2,
        graph_depth=4,
        same_segment_cost=5 * max_cost,
        different_segment_cost=-5 * max_cost,
    )

    # solve the full lifted problem using the kernighan lin approximation introduced in
    # http://openaccess.thecvf.com/content_iccv_2015/html/Keuper_Efficient_Decomposition_of_ICCV_2015_paper.html
    lifted_costs = lifted_costs.astype("float64")
    node_labels = lmc.lifted_multicut_kernighan_lin(
        rag, costs, lifted_uvs, lifted_costs
    )
    segmentation = project_node_labels_to_pixels(rag, superpixels, node_labels)

    # run size threshold
    if post_minsize > 0:
        segmentation, _ = apply_size_filter(
            segmentation.astype("uint32"), boundary_pmaps, post_minsize
        )
    return segmentation


def simple_itk_watershed(
    boundary_pmaps: np.ndarray,
    threshold: float = 0.5,
    sigma: float = 1.0,
    minsize: int = 100,
) -> np.ndarray:
    """
    Simple itk watershed segmentation.

    Args:
        boundary_pmaps (np.ndarray): cell boundary prediction. 3D array of shape (Z, Y, X) with values between 0 and 1.
        threshold (float): threshold for the watershed segmentation. (default: 0.5)
        sigma (float): sigma for the gaussian smoothing. (default: 1.0)
        minsize (int): minimal size of the segments after segmentation. (default: 100)

    Returns:
        segmentation (np.ndarray): watershed output segmentation (using SimpleITK)

    """
    if not SIMPLE_ITK_INSTALLED:
        raise ValueError("please install sitk before running this process")

    if sigma > 0:
        # fix ws sigma length
        # ws sigma cannot be shorter than pmaps dims
        max_sigma = (np.array(boundary_pmaps.shape) - 1) / 3
        ws_sigma = np.minimum(max_sigma, np.ones(max_sigma.ndim) * sigma)
        boundary_pmaps = gaussianSmoothing(boundary_pmaps, ws_sigma)

    # Itk watershed + size filtering
    itk_pmaps = sitk.GetImageFromArray(boundary_pmaps)
    itk_segmentation = sitk.MorphologicalWatershed(
        itk_pmaps, threshold, markWatershedLine=False, fullyConnected=False
    )
    itk_segmentation = sitk.RelabelComponent(itk_segmentation, minsize)
    segmentation = sitk.GetArrayFromImage(itk_segmentation).astype(np.uint16)
    return segmentation


def simple_itk_watershed_from_markers(
    boundary_pmaps: np.ndarray, seeds: np.ndarray
) -> np.ndarray:
    if not SIMPLE_ITK_INSTALLED:
        raise ValueError("please install sitk before running this process")

    itk_pmaps = sitk.GetImageFromArray(boundary_pmaps)
    itk_seeds = sitk.GetImageFromArray(seeds)
    segmentation = sitk.MorphologicalWatershedFromMarkers(
        itk_pmaps, itk_seeds, markWatershedLine=False, fullyConnected=False
    )
    return sitk.GetArrayFromImage(segmentation).astype("uint32")
