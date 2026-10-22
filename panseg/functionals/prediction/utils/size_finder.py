import contextlib
import logging
import threading
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from panseg.functionals.training.model import UNet2D

logger = logging.getLogger(__name__)

# Voxel-budget multiplier applied to the probed maximum patch shape whenever a derived
# patch shape fails the batch-size-1 OOM check (see `find_feasible_patch_and_halo_shapes`).
FEASIBILITY_SHRINK = 0.75

# Share of the free memory a CPU patch may take: prediction holds more than one forward pass
CPU_MEMORY_FRACTION = 0.5
# Input that measures a model's CPU memory per voxel: 16 per axis allows 5 levels of pooling
CPU_PROBE_SHAPES = {2: (16, 16), 3: (16, 16, 16)}
# Smallest and largest side of a CPU patch, the largest being the former fixed shapes
CPU_PATCH_SIDE = {2: (64, 1024), 3: (64, 256)}
# Floor for the resident-memory fallback, in case reused memory hides the forward pass
MIN_BYTES_PER_VOXEL = 1024
# SliceBuilder's smallest patch in Y and X
MIN_PATCH_YX = 64


@contextlib.contextmanager
def _quiet_oom_probing(device: str):
    """Temporarily cap the CUDA caching allocator during GPU probing.

    The search loops below probe the GPU by repeatedly triggering intentional
    out-of-memory errors. Since torch 2.13, every failed ``cudaMalloc`` also
    emits a C++ warning (``CUDACachingAllocator.cpp:3933``), which floods the
    log. Capping the process at its current footprint plus the currently free
    memory - the most it can physically grow by - makes oversized allocations
    get rejected by the allocator's ``per_process_memory_fraction`` policy
    before ``cudaMalloc`` is called, i.e. without the warning, while still
    raising the same catchable OOM ``RuntimeError``. The previous fraction is
    restored on exit.

    On machines without a CUDA device this is a no-op.
    """
    if not torch.cuda.is_available():
        yield
        return
    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    reserved = torch.cuda.memory.memory_reserved(index)
    # Staying strictly below the device total keeps the cap active (a fraction
    # of 1.0 disables it).
    allowed = min(reserved + free, total - 1)
    previous = torch.cuda.memory.get_per_process_memory_fraction(index)
    torch.cuda.memory.set_per_process_memory_fraction(allowed / total, index)
    try:
        yield
    finally:
        torch.cuda.memory.set_per_process_memory_fraction(previous, index)


def _is_2d_model(model: nn.Module) -> bool:
    if isinstance(model, nn.DataParallel):
        model = model.module
    return isinstance(model, UNet2D)


def _cgroup_headroom(
    proc_cgroup: Path = Path("/proc/self/cgroup"), root: Path = Path("/sys/fs/cgroup")
) -> int | None:
    """Memory left under this process's cgroup limits, e.g. a Slurm job's, or None.

    A limit set on any ancestor group applies, so the groups are walked up to the root.
    Handles cgroup v2 (`memory.max`) and v1 (`memory.limit_in_bytes`).
    """
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return None
    headroom = None
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            base, limit_file, usage_file = root, "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            base = root / "memory"
            limit_file, usage_file = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        group = base / path.lstrip("/")
        while True:
            try:
                limit = (group / limit_file).read_text().strip()
                usage = int((group / usage_file).read_text())
            except (OSError, ValueError):
                limit = None
            if (
                limit not in (None, "max") and int(limit) < 2**60
            ):  # v1 reports no limit as ~2**63
                room = int(limit) - usage
                headroom = room if headroom is None else min(headroom, room)
            if group == base or group.parent == group:
                break
            group = group.parent
    return headroom


def _available_memory() -> int:
    """Bytes this process can still use: free system memory, capped by a cgroup limit."""
    available = psutil.virtual_memory().available
    headroom = _cgroup_headroom()
    return available if headroom is None else min(available, headroom)


def _forward_bytes_per_voxel(
    model: nn.Module, in_channels: int, shape: tuple[int, ...]
) -> float | None:
    """Peak memory held at once by one CPU forward pass, per voxel, or None.

    Read from the profiler's record of every allocation and free, which grows exactly
    with the voxel count. That record is not public API, so None is returned when it
    is missing or another profiler is running.
    """
    x = torch.randn((1, in_channels) + shape)
    try:
        with (
            torch.no_grad(),
            profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof,
        ):
            model(x)
        events = prof.profiler.kineto_results.events()
        allocations = sorted(
            (event.start_ns(), event.nbytes())
            for event in events
            if event.name() == "[memory]"
        )
    except (AttributeError, RuntimeError):
        return None
    held = peak = 0
    for _, nbytes in allocations:
        held += nbytes
        peak = max(peak, held)
    return peak / x.numel() if peak > 0 else None


def _forward_memory(model: nn.Module, in_channels: int, shape: tuple[int, ...]) -> int:
    """Peak growth of this process's resident memory during one CPU forward pass."""
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    done = threading.Event()

    def sample():
        nonlocal peak
        while not done.is_set():
            peak = max(peak, process.memory_info().rss)
            done.wait(0.002)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        with torch.no_grad():
            model(torch.randn((1, in_channels) + shape))
    finally:
        done.set()
        sampler.join()
    return max(peak, process.memory_info().rss) - baseline


def _probe_cpu_patch_shape(
    model: nn.Module, in_channels: int, nn_dim: int
) -> tuple[int, int, int]:
    """Largest CPU patch whose forward pass fits in a share of the free memory.

    CPU allocations do not fail the way CUDA ones do: memory runs into swap or the
    process is killed. So instead of probing for OOM, the memory per voxel is measured
    on a tiny input with the profiler. Without the profiler's record, the growth of
    the resident memory is used, which overestimates it on small inputs.
    """
    budget = CPU_MEMORY_FRACTION * _available_memory()
    model.eval()
    shape = CPU_PROBE_SHAPES[nn_dim]
    bytes_per_voxel = _forward_bytes_per_voxel(model, in_channels, shape)
    if bytes_per_voxel is None:
        logger.warning(
            "PyTorch's allocation record is unavailable, so CPU memory is estimated "
            "from the resident memory, which overestimates it."
        )
        resident = _forward_memory(model, in_channels, shape) / np.prod(shape)
        bytes_per_voxel = max(resident, MIN_BYTES_PER_VOXEL)

    low, high = CPU_PATCH_SIDE[nn_dim]
    side = int((budget / bytes_per_voxel) ** (1 / nn_dim)) // 16 * 16
    if side < low:
        logger.warning(
            f"Only {budget / 1e9:.1f} GB of memory is free for a CPU patch, which needs "
            f"{bytes_per_voxel / 1e3:.1f} kB per voxel. Using the smallest patch, "
            f"{low} per side, which may not fit."
        )
    side = min(max(side, low), high)
    logger.info(
        f"CPU patch: {bytes_per_voxel / 1e3:.1f} kB per voxel and "
        f"{budget / 1e9:.1f} GB to use give at most {side} per side"
    )
    return (1, side, side) if nn_dim == 2 else (side, side, side)


def _is_oom_error(error: RuntimeError, device: str) -> bool:
    """Classify a device probe failure as an out-of-memory error.

    CUDA reports "CUDA out of memory", but Apple MPS reports memory failures
    with different wording: "The device has run out of memory" for general
    exhaustion, and "Invalid buffer size: N GiB" when a single allocation
    exceeds the 4 GiB Metal buffer cap.
    """
    message = str(error).lower()
    if "out of memory" in message:
        return True
    return "mps" in device and "invalid buffer size" in message


def derive_patch_and_halo_shapes(
    full_volume_shape: tuple[int, int, int],
    max_patch_shape: tuple[int, int, int],
    min_halo_shape: tuple[int, int, int],
    both_sides: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Recommend an optimal patch shape and halo size for a given 3D sample. Since convolving
    kernels are often isotropic, adjusting and fitting an initially isotropic maximum patch
    shape to the full sample image helps achieve a more balanced patch shape while ensuring
    optimal hardware utilization and accurate predictions.

    Args:
        full_volume_shape (tuple[int, int, int]):
            The shape of the full 3D volume (Z, Y, X).
        max_patch_shape (tuple[int, int, int]):
            The maximum feasible patch shape for the GPU (Z, Y, X).
        min_halo_shape (tuple[int, int, int]):
            The minimum required halo size per side (Z, Y, X).
        both_sides (bool, optional):
            If True, the `min_halo_shape` is applied symmetrically on both sides of the patch.
            Defaults to False.

    Returns:
        tuple[tuple[int, int, int], tuple[int, int, int]]:
            - Recommended patch shape `(Z, Y, X)`.
            - Halo size on one side `(Z, Y, X)`.

    Notes:
        - If the total voxel capacity of `max_patch_shape` is larger than or equal to the entire
          volume, we simply return the full volume shape with zero halo.
        - Otherwise, we find which dimensions actually require tiling and set a minimum halo
          there (0 otherwise).
    """
    shape_volume = np.array(full_volume_shape)
    shape_patch_max = np.array(max_patch_shape)
    shape_halo_min = (
        np.array(min_halo_shape) // 2 if both_sides else np.array(min_halo_shape)
    )

    n_voxels_patch = np.prod(shape_patch_max)
    n_voxels_sample = np.prod(shape_volume)

    if n_voxels_patch >= n_voxels_sample:
        return tuple(shape_volume), (0, 0, 0)
    else:  # otherwise at least one dim needs to be tiled, count_shrink < 3.
        dim_shrink = shape_patch_max > shape_volume
        halo_shape = np.where(dim_shrink, 0, shape_halo_min)
        count_shrink = dim_shrink.sum()

        adjusted_patch_shape = np.minimum(shape_volume, shape_patch_max)
        if count_shrink == 2:
            remaining_dim = np.flatnonzero(~dim_shrink)
            adjusted_patch_shape[remaining_dim] = n_voxels_patch // np.prod(
                shape_volume[dim_shrink]
            )
        elif count_shrink == 1:
            remaining_dims = np.flatnonzero(~dim_shrink)
            max_area = n_voxels_patch // shape_volume[dim_shrink]
            adjusted_patch_shape[remaining_dims] = np.sqrt(max_area).astype(int)
        else:  # count_shrink == 0.
            assert np.all(shape_patch_max <= shape_volume)
            adjusted_patch_shape = shape_patch_max

        adjusted_patch_shape = np.minimum(adjusted_patch_shape, shape_volume)

        patch_size = adjusted_patch_shape - halo_shape * 2
        # Trim the halo rather than let a patch fall below SliceBuilder's minimum in Y and X
        trim = (patch_size[1:] < MIN_PATCH_YX) & (
            adjusted_patch_shape[1:] >= MIN_PATCH_YX
        )
        if np.any(trim):
            halo_shape[1:] = np.where(
                trim, (adjusted_patch_shape[1:] - MIN_PATCH_YX) // 2, halo_shape[1:]
            )
            patch_size = adjusted_patch_shape - halo_shape * 2
            logger.debug(f"Halo trimmed to {tuple(halo_shape)} to keep 64 in Y and X")
        if np.any(patch_size < 1):
            if np.any(adjusted_patch_shape < 1):
                raise RuntimeError(
                    "Error during patch size calculation!\n"
                    f"Adjusted patch shape {adjusted_patch_shape}"
                )
            logger.debug("Halo bigger than patch size, setting halo to (0,0,0)")
            halo_shape = np.array((0, 0, 0))
            patch_size = adjusted_patch_shape

        return tuple(patch_size), tuple(halo_shape)


def probe_max_patch_shape(
    model: nn.Module,
    in_channels: int,
    device: str,
    # ) -> tuple[int, int, int] | tuple[int, int]:
) -> tuple[int, int, int]:
    """Determine the maximum feasible patch shape for a given model based on
    available GPU memory using binary search.

    1. This is merely a good quick guess. However, an exact maximum size causes problems.
    2. If isotropic shape, then the `best_n` is 128 for 2080 Ti, 186 for 3090,
       but for the sake of speed, cap it at 50 and let batch size handle the rest.
    3. On CPU the shape is sized from the free memory instead, see `_probe_cpu_patch_shape`.
    """

    nn_dim = 2 if _is_2d_model(model) else 3

    if device == "cpu":
        return _probe_cpu_patch_shape(model, in_channels, nn_dim)

    with _quiet_oom_probing(device):
        model = model.to(device)
        model.eval()

        if nn_dim == 3:  # use binary search for 3D
            low, high = (2, 50)
            best_n = low

            with torch.no_grad():
                while low <= high:
                    mid = (low + high) // 2
                    patch_shape = (16 * mid,) * nn_dim
                    x = None
                    try:
                        x = torch.randn((1, in_channels) + patch_shape).to(device)
                        _ = model(x)
                        best_n = mid  # Update best_n if successful
                        low = mid + 1  # Try larger patches
                    except RuntimeError as e:
                        if _is_oom_error(e, device):
                            errs = str(e).split(".", maxsplit=1)
                            logger.info(
                                f"Encountered '{errs[0]}' at patch shape {patch_shape}, "
                                "reducing it."
                            )
                            logger.debug(f"{errs[-1]}")

                            high = mid - 1  # Try smaller patches
                        else:
                            del model
                            raise
                    finally:
                        del x
                        torch.cuda.empty_cache()

        else:  # use linear search for 2D
            best_n = 200

            with torch.no_grad():
                while best_n > 16:
                    patch_shape = (16 * best_n,) * nn_dim
                    x = None
                    try:
                        x = torch.randn((1, in_channels) + patch_shape).to(device)
                        _ = model(x)
                        break
                    except RuntimeError as e:
                        if _is_oom_error(e, device):
                            best_n -= 20
                        else:
                            del model
                            raise
                    finally:
                        del x
                        torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

        return (
            (1, 16 * best_n, 16 * best_n)
            if nn_dim == 2
            else (16 * best_n, 16 * best_n, 16 * best_n)
        )


def find_feasible_patch_and_halo_shapes(
    model: nn.Module,
    in_channels: int,
    full_volume_shape: tuple[int, int, int],
    min_halo_shape: tuple[int, int, int],
    device: str,
    max_attempts: int = 5,
    both_sides: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Recommend a patch shape and halo size that is verified to fit on the device.

    `probe_max_patch_shape` probes the GPU with isotropic inputs and returns a tight maximum
    patch shape with no safety margin. `derive_patch_and_halo_shapes` then redistributes that
    voxel budget to match the aspect ratio of the full volume, so the shape that is actually
    fed to the model at prediction time can require more GPU memory than the probed maximum
    (peak memory does not depend on the voxel count alone), and the amount of free memory can
    change between probing and prediction. On such machines the batch size determination
    OOMs even for batch size 1 and prediction aborts (see issue #578).

    To prevent that, the shape derived here is verified with an actual batch-size-1 forward
    pass via `will_CUDA_OOM`. If it does not fit, the voxel budget of the probed maximum is
    shrunk by `FEASIBILITY_SHRINK` and the shape is derived and verified again, until a
    feasible shape is found or `max_attempts` is reached.

    Args:
        model (nn.Module): The model that will run the prediction.
        in_channels (int): Number of input channels to the model.
        full_volume_shape (tuple[int, int, int]): The shape of the full 3D volume (Z, Y, X).
        min_halo_shape (tuple[int, int, int]): The minimum required halo size per side (Z, Y, X).
        device (str): The computation device ('cpu', 'cuda', etc.).
        max_attempts (int, optional): Maximum number of derive-and-verify attempts.
            Defaults to 5.
        both_sides (bool, optional): If True, the `min_halo_shape` is applied symmetrically
            on both sides of the patch. Defaults to False.

    Returns:
        tuple[tuple[int, int, int], tuple[int, int, int]]:
            - Verified patch shape `(Z, Y, X)`.
            - Halo size on one side `(Z, Y, X)`.

    Raises:
        RuntimeError: If no feasible patch shape is found within `max_attempts` attempts.
    """
    max_patch_shape = probe_max_patch_shape(model, in_channels, device)

    if device == "cpu":
        return derive_patch_and_halo_shapes(
            full_volume_shape, max_patch_shape, min_halo_shape, both_sides
        )
    patch, halo = (None, None)
    for attempt in range(max_attempts):
        patch, halo = derive_patch_and_halo_shapes(
            full_volume_shape, max_patch_shape, min_halo_shape, both_sides
        )
        if not will_CUDA_OOM(
            model, in_channels, patch, halo, batch_size=1, device=device
        ):
            if attempt > 0:
                logger.info(
                    f"Using reduced patch shape {patch} and halo {halo} after "
                    f"{attempt} shrink(s) of the maximum patch shape."
                )
            return patch, halo
        logger.warning(
            f"Auto-determined patch shape {patch} with halo {halo} does not fit on "
            f"{device}; retrying with a {1 - FEASIBILITY_SHRINK:.0%} smaller voxel budget."
        )
        max_patch_shape = tuple(
            max(1, int(dim * FEASIBILITY_SHRINK ** (1 / 3))) for dim in max_patch_shape
        )

    raise RuntimeError(
        f"Could not determine a feasible patch shape on {device} after {max_attempts} "
        f"attempts (last tried patch shape {patch} with halo {halo}). "
        "Please reduce the patch size manually or free up GPU memory."
    )


def find_batch_size(
    model: nn.Module,
    in_channels: int,
    patch_shape: tuple[int, int, int],
    patch_halo: tuple[int, int, int],
    device: str,
) -> int:
    """Determine the maximum feasible batch size for a given model based on available GPU memory.

    Args:
        model (nn.Module): The model to be used for prediction.
        in_channels (int): Number of input channels to the model.
        patch_shape (tuple[int, int, int]): Dimensions of the input patches.
        patch_halo (tuple[int, int, int]): Halo size used for patch augmentation.
        device (str): Device to perform the computation on ('cuda' or 'cpu').

    Returns:
        int: The largest batch size that can be used without causing memory overflow.
    """
    if device == "cpu":
        return 1

    actual_patch_shape = tuple(patch_shape[i] + 2 * patch_halo[i] for i in range(3))
    if isinstance(model, UNet2D):
        actual_patch_shape = actual_patch_shape[1:]

    with _quiet_oom_probing(device):
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                x = None
                try:
                    x = torch.randn((batch_size, in_channels) + actual_patch_shape).to(
                        device
                    )
                    _ = model(x)
                except RuntimeError as e:
                    if _is_oom_error(e, device):
                        errs = str(e).split(".", maxsplit=1)
                        logger.info(
                            f"Encountered '{errs[0]}' at batch size {batch_size}, halving it."
                        )
                        logger.debug(f"{errs[-1]}")
                        batch_size //= 2
                        break
                    else:
                        logger.warning(
                            f"Encountered '{e}' at batch size {batch_size}, "
                            "unexpected but continuing with halved batch size."
                        )
                        batch_size //= 2
                        break
                finally:
                    del x
                    torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        if batch_size == 0:
            raise RuntimeError(
                f"Could not determine a feasible batch size for patch size "
                f"{patch_shape} and halo {patch_halo}. Please reduce the patch size."
            )
        return batch_size


def will_CUDA_OOM(
    model: nn.Module,
    in_channels: int,
    patch_shape: tuple[int, int, int],
    patch_halo: tuple[int, int, int],
    batch_size: int,
    device: str,
) -> bool:
    """Determine if a given batch size will cause an out-of-memory (OOM)
    error on the specified device.

    Args:
        model (nn.Module): The model to be used for prediction.
        in_channels (int): Number of input channels to the model.
        patch_shape (tuple[int, int, int]): Dimensions of the input patches.
        patch_halo (tuple[int, int, int]): Halo size used for patch augmentation.
        batch_size (int): Number of samples per batch.
        device (str): Device to perform the computation on ('cuda' or 'cpu').

    Returns:
        bool: True if the batch size will cause an OOM error, False otherwise.
    """
    if device == "cpu":
        return False  # CPU does not have CUDA OOM errors

    # Calculate the actual patch shape including the halo
    actual_patch_shape = tuple(patch_shape[i] + 2 * patch_halo[i] for i in range(3))

    # If the model is a 2D UNet, adjust the patch shape to 2D
    if isinstance(model, UNet2D):
        actual_patch_shape = actual_patch_shape[1:]

    with _quiet_oom_probing(device):
        model = model.to(device)
        model.eval()

        OOM_error = False
        x = None
        try:
            with torch.no_grad():
                x = torch.randn((batch_size, in_channels) + actual_patch_shape).to(
                    device
                )
                _ = model(x)
        except RuntimeError as e:
            if _is_oom_error(e, device):
                OOM_error = True
                logger.info(
                    f"Using patch shape {patch_shape}, halo {patch_halo}, "
                    f"and batch size {batch_size} will cause OOM."
                )
            else:
                raise  # Re-raise if it's not an OOM error
        finally:
            del x
            del model
            torch.cuda.empty_cache()

        return OOM_error
