import contextlib
import logging

import numpy as np
import torch
from torch import nn
from torch.optim.adam import Adam

from panseg.functionals.model_geometry import (
    ModelGeometry,
    is_2d_model,
    model_geometry,
    tensor_input_shape,
)
from panseg.functionals.precision import AmpSetting, amp_context
from panseg.functionals.training.losses import DiceLoss

logger = logging.getLogger(__name__)

# Fraction of the patch size the sliding window advances by default. The stride
# is rounded down to a multiple of the phase period so that overlapping patches
# compute the same voxel at the same phase (see `get_stride_shape`).
DEFAULT_STRIDE_RATIO = 0.75


def get_stride_shape(
    patch_shape: tuple[int, int, int],
    increment: tuple[int, int, int],
    stride_ratio: float = DEFAULT_STRIDE_RATIO,
) -> tuple[int, int, int]:
    """Stride for sliding-window prediction, aligned to the phase period.

    The stride must be a multiple of the per-axis phase period (`increment`):
    otherwise overlapping patches compute the same voxel at different phases,
    and a voxel that is exact in one patch can carry an artefact in the next.
    It must never exceed the patch size, and never be below one voxel.

    Args:
        patch_shape: patch shape (Z, Y, X).
        increment: per-axis phase period (Z, Y, X).
        stride_ratio: fraction of the patch to advance by. Defaults to 0.75.

    Returns:
        stride shape (Z, Y, X).
    """
    stride = []
    for patch, period in zip(patch_shape, increment):
        if patch < period:
            stride.append(1)
            continue
        stepped = (int(patch * stride_ratio) // period) * period
        stride.append(min(patch, max(period, stepped)))
    return (stride[0], stride[1], stride[2])


# Voxel-budget multiplier applied to the probed maximum patch shape whenever a derived
# patch shape fails the batch-size-1 OOM check (see `find_feasible_patch_and_halo_shapes`).
FEASIBILITY_SHRINK = 0.75


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


_BN_MODULES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


@contextlib.contextmanager
def _preserve_bn_stats(model: nn.Module):
    """Snapshot and restore the BatchNorm running stats of `model`.

    The training-mode batch-size probe runs forward passes in train mode (probe
    fidelity: the measured memory must match the real training steps), which
    updates `running_mean`, `running_var` and `num_batches_tracked`. Those
    updates must not leak into the training run, so the buffers are restored
    bit-exactly when the probe ends, including when the probe raises.
    """
    saved = [
        (
            module,
            module.running_mean.clone(),
            module.running_var.clone(),
            module.num_batches_tracked.clone(),
        )
        for module in model.modules()
        if isinstance(module, _BN_MODULES) and module.track_running_stats
    ]
    try:
        yield
    finally:
        for module, running_mean, running_var, num_batches_tracked in saved:
            module.running_mean.copy_(running_mean)
            module.running_var.copy_(running_var)
            module.num_batches_tracked.copy_(num_batches_tracked)


@contextlib.contextmanager
def _preserve_parameters(model: nn.Module):
    """Snapshot and restore the parameters of `model` bit-exactly.

    The training probe runs real optimizer steps, which mutate the weights.
    The snapshot lives on CPU so it does not inflate the device memory the
    probe measures. Restored on exit, including when the probe raises.
    """
    saved = [
        (parameter, parameter.detach().to("cpu", copy=True))
        for parameter in model.parameters()
    ]
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, snapshot in saved:
                parameter.copy_(snapshot)


def channels_last_3d_enabled(model: nn.Module, device: str, requested: bool) -> bool:
    """Whether the model actually runs in channels_last_3d: 3D models on CUDA only."""
    return requested and torch.device(device).type == "cuda" and not is_2d_model(model)


def _to_probe_device(
    model: nn.Module, device: str, channels_last_3d: bool
) -> tuple[nn.Module, bool]:
    """Move `model` to `device` in the probe memory format.

    Returns the model and the resolved channels_last_3d flag.
    """
    channels_last_3d = channels_last_3d_enabled(model, device, channels_last_3d)
    model = model.to(device)
    if channels_last_3d:
        model = model.to(memory_format=torch.channels_last_3d)
    return model, channels_last_3d


def _to_probe_layout(x: torch.Tensor, channels_last_3d: bool) -> torch.Tensor:
    return x.to(memory_format=torch.channels_last_3d) if channels_last_3d else x


def is_oom_error(error: RuntimeError, device: str) -> bool:
    """Classify a device probe failure as an out-of-memory error.

    CUDA reports "CUDA out of memory", but Apple MPS reports memory failures
    with different wording: "The device has run out of memory" for general
    exhaustion, and "Invalid buffer size: N GiB" when a single allocation
    exceeds the 4 GiB Metal buffer cap. cuDNN can also report a failed
    workspace allocation as ``CUDNN_STATUS_ALLOC_FAILED``.
    """
    message = str(error).lower()
    if "out of memory" in message or "cudnn_status_alloc_failed" in message:
        return True
    return "mps" in device and "invalid buffer size" in message


def derive_patch_and_halo_shapes(
    full_volume_shape: tuple[int, int, int],
    max_patch_shape: tuple[int, int, int],
    min_halo_shape: tuple[int, int, int],
    both_sides: bool = False,
    is_2d: bool = False,
    increment: tuple[int, int, int] = (1, 1, 1),
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Recommend an optimal patch shape and halo size for a given 3D sample. Since convolving
    kernels are often isotropic, adjusting and fitting an initially isotropic maximum patch
    shape to the full sample image helps achieve a more balanced patch shape while ensuring
    optimal hardware utilization and accurate predictions.

    The derived patch is aligned to the model's phase period (`increment`), and the
    halo is applied on every spatial axis, including axes that fit in a single
    patch: the model input, not the patch, is what the memory budget sizes and
    what the halo must surround.

    Args:
        full_volume_shape (tuple[int, int, int]):
            The shape of the full 3D volume (Z, Y, X). It is rounded up to the
            phase period, matching the aligned domain prediction pads to.
        max_patch_shape (tuple[int, int, int]):
            The maximum feasible model-input shape for the GPU (Z, Y, X). Its
            voxel product is the budget for `patch + 2 * halo`.
        min_halo_shape (tuple[int, int, int]):
            The minimum required halo size per side (Z, Y, X).
        both_sides (bool, optional):
            If True, the `min_halo_shape` is applied symmetrically on both sides of the patch.
            Defaults to False.
        is_2d (bool, optional):
            If True, the model is 2D and predicts one Z slice at a time: patch depth 1,
            Z halo 0, and the voxel budget goes to Y and X only. Defaults to False.
        increment (tuple[int, int, int], optional):
            Per-axis phase period the patch and model input are aligned to.
            Defaults to (1, 1, 1).

    Returns:
        tuple[tuple[int, int, int], tuple[int, int, int]]:
            - Recommended patch shape `(Z, Y, X)`, a multiple of `increment`.
            - Halo size on one side `(Z, Y, X)`.

    Raises:
        RuntimeError: If `increment + 2 * halo` alone exceeds the voxel budget.
    """
    shape_volume = np.array(full_volume_shape, dtype=np.int64)
    shape_patch_max = np.array(max_patch_shape, dtype=np.int64)
    shape_increment = np.array(increment, dtype=np.int64)
    halo_shape = (
        np.array(min_halo_shape, dtype=np.int64) // 2
        if both_sides
        else np.array(min_halo_shape, dtype=np.int64)
    )
    if is_2d:  # one Z slice per patch
        shape_volume[0] = shape_patch_max[0] = 1
        shape_increment[0] = 1
        halo_shape[0] = 0

    # The caller pads the raw volume up to the aligned domain; derive on that
    # same domain so patch sizes and origins stay on the phase grid.
    shape_volume = _round_up(shape_volume, shape_increment)

    budget = int(np.prod(shape_patch_max))
    min_model_input = shape_increment + 2 * halo_shape
    if int(np.prod(min_model_input)) > budget:
        raise RuntimeError(
            f"The aligned minimum model input {tuple(min_model_input)} "
            f"(increment {tuple(shape_increment)} plus 2 x halo "
            f"{tuple(halo_shape)}) already exceeds the available voxel budget "
            f"{budget} (max patch shape {tuple(shape_patch_max)})."
        )

    # A volume that fits as one model input keeps its full extent and the
    # halo: the halo is needed even when the volume is not tiled.
    if int(np.prod(shape_volume + 2 * halo_shape)) <= budget:
        return (
            tuple(int(size) for size in shape_volume),
            tuple(int(h) for h in halo_shape),
        )

    max_patch = _round_down(np.minimum(shape_volume, shape_patch_max), shape_increment)
    max_patch = np.maximum(max_patch, shape_increment)
    max_model_input = max_patch + 2 * halo_shape

    if int(np.prod(max_model_input)) <= budget:
        return (
            tuple(int(size) for size in max_patch),
            tuple(int(h) for h in halo_shape),
        )

    # Binary-search the largest fraction of the per-axis maximum that fits the
    # model-input budget, then align each axis down to the phase period.
    low, high = 0.0, 1.0
    best = min_model_input.copy()
    for _ in range(64):
        fraction = (low + high) / 2
        candidate = _round_down(
            min_model_input + fraction * (max_model_input - min_model_input),
            shape_increment,
        )
        if int(np.prod(candidate)) <= budget:
            best = candidate
            low = fraction
        else:
            high = fraction

    # Spend any leftover budget by growing the smallest axes first, which keeps
    # the model input as balanced as the budget allows. A handful of passes is
    # enough because every axis is rounded down to the phase period first.
    for _ in range(3 * len(best)):
        grown = False
        for axis in np.argsort(best):
            candidate = best.copy()
            candidate[axis] += shape_increment[axis]
            if candidate[axis] - 2 * halo_shape[axis] > shape_volume[axis]:
                continue
            if int(np.prod(candidate)) <= budget:
                best = candidate
                grown = True
                break
        if not grown:
            break

    patch = best - 2 * halo_shape
    return tuple(int(size) for size in patch), tuple(int(h) for h in halo_shape)


def _round_up(shape, increment):
    return ((shape + increment - 1) // increment) * increment


def _round_down(shape, increment):
    return np.array(shape, dtype=np.int64) // increment * increment


def probe_max_patch_shape(
    model: nn.Module,
    in_channels: int,
    device: str,
    amp: AmpSetting = "auto",
    channels_last_3d: bool = False,
    # ) -> tuple[int, int, int] | tuple[int, int]:
) -> tuple[int, int, int]:
    """Determine the maximum feasible patch shape for a given model based on
    available GPU memory using binary search.

    Each probe is the real inference forward: eval mode, ``torch.no_grad()``,
    the same precision mode (`amp`) as the later prediction, and the requested
    channels_last_3d layout. The probed maximum is therefore not sized for fp32
    memory when the model will run in bf16.

    The isotropic edge is capped at 50 * 16 voxels; larger patches are handled
    by the batch size. The largest successful probe sits at the allocator
    boundary and is order-dependent, so a shape derived from it can fail a
    later verification. Above 256^3 the result also loses one 16-voxel step of
    headroom, which keeps the prediction below the large-patch slowdown.
    """

    nn_dim = 2 if is_2d_model(model) else 3

    if device == "cpu":
        return (1, 1024, 1024) if nn_dim == 2 else (256, 256, 256)

    with _quiet_oom_probing(device):
        model, cl3d = _to_probe_device(model, device, channels_last_3d)
        model.eval()

        if nn_dim == 3:  # use binary search for 3D
            low, high = (2, 50)
            best_n = low

            with torch.no_grad(), amp_context(amp, device):
                while low <= high:
                    mid = (low + high) // 2
                    patch_shape = (16 * mid,) * nn_dim
                    x = None
                    try:
                        x = _to_probe_layout(
                            torch.randn((1, in_channels) + patch_shape).to(device), cl3d
                        )
                        _ = model(x)
                        best_n = mid  # Update best_n if successful
                        low = mid + 1  # Try larger patches
                    except RuntimeError as e:
                        if is_oom_error(e, device):
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

            with torch.no_grad(), amp_context(amp, device):
                while best_n > 16:
                    patch_shape = (16 * best_n,) * nn_dim
                    x = None
                    try:
                        x = _to_probe_layout(
                            torch.randn((1, in_channels) + patch_shape).to(device), cl3d
                        )
                        _ = model(x)
                        break
                    except RuntimeError as e:
                        if is_oom_error(e, device):
                            best_n -= 20
                        else:
                            del model
                            raise
                    finally:
                        del x
                        torch.cuda.empty_cache()

        # one step of headroom above the 256^3 boundary (see above)
        if nn_dim == 3 and best_n > 16:
            best_n -= 1

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
    amp: AmpSetting = "auto",
    channels_last_3d: bool = False,
    geometry: ModelGeometry | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Recommend a patch shape and halo size that is verified to fit on the device.

    `probe_max_patch_shape` probes the GPU with isotropic inputs and returns a tight maximum
    patch shape. `derive_patch_and_halo_shapes` then redistributes that voxel budget to match
    the aspect ratio of the full volume, so the shape that is actually fed to the model at
    prediction time can require more GPU memory than the probed maximum (peak memory does not
    depend on the voxel count alone), and the amount of free memory can change between probing
    and prediction. On such machines the batch size determination OOMs even for batch size 1
    and prediction aborts (see issue #578).

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
        amp (str): precision mode the memory probes run in (``"auto"``/``"off"``),
            to match the precision of the prediction they size for. Defaults to
            ``"auto"``.
        channels_last_3d (bool, optional): size the probes in the prediction's
            channels_last_3d layout; default False (plain contiguous layout).
        geometry (ModelGeometry | None, optional): the model's input geometry,
            already resolved by the caller. Derived here when omitted.

    Returns:
        tuple[tuple[int, int, int], tuple[int, int, int]]:
            - Verified patch shape `(Z, Y, X)`.
            - Halo size on one side `(Z, Y, X)`.

    Raises:
        RuntimeError: If no feasible patch shape is found within `max_attempts` attempts.
        ValueError: If the derived patch's model input is below the model's
            architectural minimum (see `ModelGeometry.check_patch`).
    """
    geometry = model_geometry(model) if geometry is None else geometry
    max_patch_shape = probe_max_patch_shape(
        model, in_channels, device, amp=amp, channels_last_3d=channels_last_3d
    )
    is_2d = geometry.is_2d
    increment = geometry.increment

    def derive(max_shape):
        return derive_patch_and_halo_shapes(
            full_volume_shape,
            max_shape,
            min_halo_shape,
            both_sides,
            is_2d,
            increment,
        )

    if device == "cpu":
        patch, halo = derive(max_patch_shape)
        geometry.check_patch(patch, halo)
        return patch, halo
    patch, halo = (None, None)
    for attempt in range(max_attempts):
        patch, halo = derive(max_patch_shape)
        geometry.check_patch(patch, halo)
        if not will_CUDA_OOM(
            model,
            in_channels,
            patch,
            halo,
            batch_size=1,
            device=device,
            amp=amp,
            channels_last_3d=channels_last_3d,
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
            max(
                period,
                int(dim * FEASIBILITY_SHRINK ** (1 / 3)) // period * period,
            )
            for dim, period in zip(max_patch_shape, increment)
        )

    raise RuntimeError(
        f"Could not determine a feasible patch shape on {device} after {max_attempts} "
        f"attempts (last tried patch shape {patch} with halo {halo}). "
        "Please reduce the patch size manually or free up GPU memory."
    )


def _probe_target_channels(model: nn.Module, in_channels: int) -> int:
    """Number of channels the training probe's target must have.

    `DiceLoss` requires input and target to have the same shape, so the target
    follows the model's output, not its input (`in_channels` and `out_channels`
    may differ). The U-Nets expose it as ``final_conv.out_channels``; wrappers
    are unwrapped first. Models without that attribute fall back to
    `in_channels`.
    """
    while True:
        if isinstance(model, nn.DataParallel):
            model = model.module
        elif hasattr(model, "_orig_mod"):  # torch.compile's OptimizedModule
            model = model._orig_mod
        else:
            break
    out_channels = getattr(getattr(model, "final_conv", None), "out_channels", None)
    return in_channels if out_channels is None else out_channels


def _training_step(
    model: nn.Module, x: torch.Tensor, amp: AmpSetting, device: str
) -> None:
    """Run two real training steps as the memory probe.

    Each step mirrors `UNetTrainer.train_epoch`: the same `DiceLoss`, an `Adam`
    optimizer with the training defaults (``lr=1e-4, weight_decay=1e-5``), and
    the same autocast scopes (forward and loss, then backward and step).

    The second step matters: the trainer zeroes gradients only after the
    forward, so an iteration runs its forward with the previous iteration's
    gradients and the Adam moments still live. A one-step probe misses that
    peak.

    The probe stays eager even when training will use `torch.compile`:
    compilation lowers memory, so eager is conservative.

    Module-level and monkeypatchable so the GPU-free search tests can substitute
    the cheap synthetic backward.
    """
    # allocate the target before the forward, like a loader batch; its channel
    # count follows the model output, and it stays float32 for the real peak
    target = torch.randint(
        0,
        2,
        (x.shape[0], _probe_target_channels(model, x.shape[1])) + x.shape[2:],
        device=x.device,
        dtype=torch.float32,
    )
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    try:
        for _ in range(2):
            with amp_context(amp, device):
                output = model(x)
                loss = DiceLoss()(output, target)

            optimizer.zero_grad()
            with amp_context(amp, device):
                loss.backward()
                optimizer.step()
    finally:
        model.zero_grad(set_to_none=True)
        optimizer.state.clear()
        del optimizer


def _probe_batch(
    model: nn.Module, x: torch.Tensor, for_training: bool, amp: AmpSetting, device: str
) -> None:
    """Run one inference forward or one real training step as the probe.

    Runs under the resolved `amp` precision mode, like the compute it measures.
    """
    if for_training:
        _training_step(model, x, amp, device)
    else:
        with torch.no_grad(), amp_context(amp, device):
            model(x)


# Largest batch size the probe search considers
MAX_PROBE_BATCH_SIZE = 2048


def _probe_batch_feasible(
    model: nn.Module,
    x_shape: tuple[int, ...],
    batch_size: int,
    for_training: bool,
    amp: AmpSetting,
    device: str,
    channels_last_3d: bool = False,
) -> bool:
    """Probe one batch size; True if it fits, False on OOM (or an unexpected
    error, which is logged and treated as infeasible like before)."""
    x = None
    try:
        x = _to_probe_layout(
            torch.randn((batch_size,) + x_shape).to(device), channels_last_3d
        )
        _probe_batch(model, x, for_training, amp, device)
        return True
    except RuntimeError as e:
        if is_oom_error(e, device):
            errs = str(e).split(".", maxsplit=1)
            logger.info(f"Encountered '{errs[0]}' at batch size {batch_size}.")
            logger.debug(f"{errs[-1]}")
        else:
            logger.warning(
                f"Encountered '{e}' at batch size {batch_size}, "
                "treating the batch size as infeasible."
            )
        return False
    finally:
        if for_training:
            # drop the probe's gradients, they must not leak into training
            model.zero_grad(set_to_none=True)
        del x
        torch.cuda.empty_cache()


def find_batch_size(
    model: nn.Module,
    in_channels: int,
    patch_shape: tuple[int, int, int],
    patch_halo: tuple[int, int, int],
    device: str,
    for_training: bool = False,
    amp: AmpSetting = "auto",
    channels_last_3d: bool = False,
) -> int:
    """Determine the maximum feasible batch size for a given model based on available GPU memory.

    The search doubles the batch size until it no longer fits and then bisects
    the last interval, so the result is the true maximum.

    With `for_training=True` each candidate runs two real training steps
    (`_training_step`): forward, Dice loss, backward, and an Adam step,
    including the Adam moments and the previous iteration's gradients (the
    training loop zeroes them only after the next forward). The probe stays
    eager even when training will use ``torch.compile`` (compilation lowers
    memory, so eager is conservative). The model's parameters and BatchNorm
    running stats are restored bit-exactly after probing.

    Args:
        model (nn.Module): The model to be used for prediction or training.
        in_channels (int): Number of input channels to the model.
        patch_shape (tuple[int, int, int]): Dimensions of the input patches.
        patch_halo (tuple[int, int, int]): Halo size used for patch augmentation.
        device (str): Device to perform the computation on ('cuda' or 'cpu').
        for_training (bool): If True, run the real training step (forward, Dice
            loss, backward, and Adam step) for every probe. Training needs more
            memory than a forward pass (gradients and optimizer moments), so a
            forward-only probe overestimates the batch size. Defaults to False
            (inference).
        amp (str): precision mode the probe runs in (``"auto"``/``"off"``), to
            match the precision of the compute it measures. Defaults to
            ``"auto"``.
        channels_last_3d (bool, optional): size the probe in the prediction's
            channels_last_3d layout; default False (plain contiguous layout).

    Returns:
        int: The largest batch size that can be used without causing memory overflow.
    """
    if device == "cpu":
        return 1

    actual_patch_shape = tensor_input_shape(model, patch_shape, patch_halo)
    x_shape = (in_channels,) + actual_patch_shape

    was_training = model.training
    preserve_parameters = (
        _preserve_parameters(model) if for_training else contextlib.nullcontext()
    )
    # BN running stats only change in train mode, so the cloning is
    # training-only: the inference probe cannot pollute them.
    preserve_bn_stats = (
        _preserve_bn_stats(model) if for_training else contextlib.nullcontext()
    )
    with _quiet_oom_probing(device), preserve_bn_stats, preserve_parameters:
        model, cl3d = _to_probe_device(model, device, channels_last_3d)
        model.train(for_training)
        try:
            if not _probe_batch_feasible(
                model, x_shape, 1, for_training, amp, device, cl3d
            ):
                raise RuntimeError(
                    f"Could not determine a feasible batch size for patch size "
                    f"{patch_shape} and halo {patch_halo}, even at batch size 1. "
                    "Please reduce the patch size/halo or free up device memory."
                )
            feasible = 1
            candidate = 2
            while candidate <= MAX_PROBE_BATCH_SIZE and _probe_batch_feasible(
                model, x_shape, candidate, for_training, amp, device, cl3d
            ):
                feasible = candidate
                candidate *= 2
            if candidate <= MAX_PROBE_BATCH_SIZE:
                # candidate OOMed: bisect (feasible, candidate) for the true max
                infeasible = candidate
                while feasible + 1 < infeasible:
                    mid = (feasible + infeasible) // 2
                    if _probe_batch_feasible(
                        model, x_shape, mid, for_training, amp, device, cl3d
                    ):
                        feasible = mid
                    else:
                        infeasible = mid
            return feasible
        finally:
            model.train(was_training)
            del model
            torch.cuda.empty_cache()


def will_CUDA_OOM(
    model: nn.Module,
    in_channels: int,
    patch_shape: tuple[int, int, int],
    patch_halo: tuple[int, int, int],
    batch_size: int,
    device: str,
    amp: AmpSetting = "auto",
    channels_last_3d: bool = False,
) -> bool:
    """Determine if a given batch size will cause an out-of-memory (OOM)
    error on the specified device.

    The verification forward runs the real prediction forward: eval mode,
    ``torch.no_grad()``, the resolved `amp` precision mode, and the requested
    channels_last_3d layout.

    Args:
        model (nn.Module): The model to be used for prediction.
        in_channels (int): Number of input channels to the model.
        patch_shape (tuple[int, int, int]): Dimensions of the input patches.
        patch_halo (tuple[int, int, int]): Halo size used for patch augmentation.
        batch_size (int): Number of samples per batch.
        device (str): Device to perform the computation on ('cuda' or 'cpu').
        amp (str): precision mode the probe runs in (``"auto"``/``"off"``).
        channels_last_3d (bool, optional): size the probe in the prediction's
            channels_last_3d layout; default False (plain contiguous layout).

    Returns:
        bool: True if the batch size will cause an OOM error, False otherwise.
    """
    if device == "cpu":
        return False  # CPU does not have CUDA OOM errors

    # The tensor's spatial shape: patch plus halo, without Z for 2D models
    actual_patch_shape = tensor_input_shape(model, patch_shape, patch_halo)

    with _quiet_oom_probing(device):
        model, cl3d = _to_probe_device(model, device, channels_last_3d)
        model.eval()

        OOM_error = False
        x = None
        try:
            with torch.no_grad(), amp_context(amp, device):
                x = _to_probe_layout(
                    torch.randn((batch_size, in_channels) + actual_patch_shape).to(
                        device
                    ),
                    cl3d,
                )
                _ = model(x)
        except RuntimeError as e:
            if is_oom_error(e, device):
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
