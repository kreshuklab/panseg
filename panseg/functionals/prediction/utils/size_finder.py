import logging

import numpy as np
import torch
from torch import nn

from panseg.functionals.training.model import UNet2D

logger = logging.getLogger(__name__)


def _is_2d_model(model: nn.Module) -> bool:
    if isinstance(model, nn.DataParallel):
        model = model.module
    return isinstance(model, UNet2D)


def find_patch_and_halo_shapes(
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

        patch_size = adjusted_patch_shape - halo_shape * 2
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


def find_a_max_patch_shape(
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
    """

    nn_dim = 2 if _is_2d_model(model) else 3

    if device == "cpu":
        return (1, 1024, 1024) if nn_dim == 2 else (256, 256, 256)

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
                    if "out of memory" in str(e):
                        errs = str(e).split(".", maxsplit=1)
                        logger.info(
                            f"Encountered '{errs[0]}' at patch shape {patch_shape}, "
                            "reducing it."
                        )
                        logger.debug(f"{errs[-1]}")

                        high = mid - 1  # Try smaller patches
                    else:
                        logger.warning(
                            f"Encountered '{e}' at patch shape {patch_shape}, "
                            "unexpected but continuing."
                        )
                        high = mid - 1  # Try smaller patches
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
                    if "out of memory" in str(e):
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
                if "out of memory" in str(e):
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

    model = model.to(device)
    model.eval()

    OOM_error = False
    x = None
    try:
        with torch.no_grad():
            x = torch.randn((batch_size, in_channels) + actual_patch_shape).to(device)
            _ = model(x)
    except RuntimeError as e:
        if "out of memory" in str(e):
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
