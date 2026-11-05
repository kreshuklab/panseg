import logging
from typing import Callable

import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

from panseg.functionals.precision import (
    AmpMode,
    AmpSetting,
    amp_context,
    validate_amp,
)
from panseg.functionals.prediction.utils.array_dataset import (
    ArrayDataset,
    default_prediction_collate,
    remove_padding,
)
from panseg.functionals.prediction.utils.size_finder import (
    channels_last_3d_enabled,
    find_batch_size,
    is_2d_model,
    is_oom_error,
    will_CUDA_OOM,
)
from panseg.functionals.training.embeddings import embeddings_to_affinities
from panseg.functionals.transfer import to_device

logger = logging.getLogger(__name__)

# Empirical amortization threshold: below this many forward passes the one-time
# compile cost outweighs the per-batch saving.
COMPILE_MIN_BATCHES = 200


class ArrayPredictor:
    """Predictor class for applying a model to a dataset and returning the results as numpy arrays.

    This predictor applies a given model on a dataset and accumulates the results into numpy arrays.
    The probability maps are computed in batches and memory utilization is carefully managed to fit
    within available system RAM. For large datasets that do not fit in memory, consider using
    `LazyPredictor` instead.

    If a batch does not fit on the device during the prediction loop (the one-off probe can be
    optimistic because repeated forwards can fragment the caching allocator), the run restarts
    with half the batch size, down to batch size 1. If batch size 1 does not fit either,
    prediction fails with the patch, halo, and batch size in the error message.

    Based on pytorch-3dunet StandardPredictor:
    https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/predictor.py

    Args:
        model (nn.Module): A trained model used for prediction.
        in_channels (int): Number of input channels to the model.
        out_channels (int): Number of output channels from the model.
        device (str): Device to use for prediction.
        patch (tuple[int, int, int]): Patch size used for prediction.
        patch_halo (tuple[int, int, int]): Mirror padding around the patch.
        single_batch_mode (bool): If True, the batch size will be set to 1.
        headless (bool): If True, use DataParallel if multiple GPUs are available.
        is_embedding (bool, optional): If True, convert model output to embeddings. Defaults to False.
        verbose_logging (bool, optional): If True, enable verbose logging. Defaults to False.
        disable_tqdm (bool, optional): If True, disable tqdm progress bars. Defaults to False.
        amp (AmpSetting, optional): precision mode, see `unet_prediction`.
        channels_last_3d (bool, optional): opt in to the channels_last_3d
            memory format for 3D models on CUDA, see `unet_prediction`. The
            memory probes size in the same layout. Defaults to False because
            the measured result is stack-dependent: on torch 2.13/CUDA
            13/cuDNN 9.26 it is 1.7x slower than contiguous.
        compile (bool | None, optional): torch.compile control. None (default)
            compiles automatically only on CUDA and only when the number of
            prediction batches amortizes the one-time compile cost
            (COMPILE_MIN_BATCHES); True forces compilation; False forces eager
            execution. In some environments triton additionally needs
            C_INCLUDE_PATH=/usr/local/cuda/include; a failed compile falls back
            to eager with a warning.

    Attributes:
        batch_size (int): Batch size actually used. It starts at the probed size and may shrink if
            a batch runs out of memory during prediction.
        device (str): Device where the model will be run.
        model (nn.Module): Model configured for evaluation.
        out_channels (int): Number of channels expected in the output.
        patch (tuple[int, int, int]): Patch size used for prediction.
        patch_halo (tuple[int, int, int]): Halo size around each patch.
        verbose_logging (bool): Flag to enable detailed logging.
        disable_tqdm (bool): Flag to disable tqdm progress bars during prediction.
        is_embedding (bool): Flag to determine if the output should be treated as embeddings.
        tracker (Optional[PBar_Tracker]): Relais progress bar information from task to widget.
    """

    def __init__(
        self,
        model: nn.Module,
        in_channels: int,
        out_channels: int,
        device: str,
        patch: tuple[int, int, int],
        patch_halo: tuple[int, int, int],
        single_batch_mode: bool,
        headless: bool,
        is_embedding: bool = False,
        verbose_logging: bool = False,
        disable_tqdm: bool = False,
        tracker=None,
        amp: AmpSetting = "auto",
        channels_last_3d: bool = False,
        compile: bool | None = None,
    ):
        self.device = device
        self.tracker = tracker
        self.patch = patch
        self.patch_halo = patch_halo
        self.amp: AmpMode = validate_amp(amp)

        # activation to apply on top of `forward` (see
        # `AbstractUNet.output_activation`), captured before any
        # DataParallel/compile wrapping. It is the identity for the
        # `UNet*Probabilities` export classes, whose `forward` already activates,
        # and the identity for models without the entry (e.g. SpocoNet): a
        # double activation is impossible by construction.
        output_activation = getattr(model, "output_activation", None)
        self._activate: Callable[[torch.Tensor], torch.Tensor] = (
            output_activation() if output_activation is not None else nn.Identity()
        )

        # the probes must size for the memory format of the prediction loop
        self.channels_last_3d = channels_last_3d_enabled(
            model, device, channels_last_3d
        )

        if single_batch_mode:  # then check if OOM happens at batch size 1
            self.batch_size = 1
            if device != "cpu" and will_CUDA_OOM(
                model,
                in_channels,
                patch,
                patch_halo,
                self.batch_size,
                device,
                amp=self.amp,
                channels_last_3d=self.channels_last_3d,
            ):
                raise RuntimeError(
                    f"Prediction will run out of memory on {device} for "
                    f"{self._shapes_description()}, even at batch size "
                    f"{self.batch_size}. Please reduce the patch size/halo or free "
                    "up device memory."
                )
        else:  # find the max batch size without causing OOM, may be [1, 2, 4, 8, 16, ...]
            self.batch_size = find_batch_size(
                model,
                in_channels,
                patch,
                patch_halo,
                device,
                amp=self.amp,
                channels_last_3d=self.channels_last_3d,
            )

        logger.info(f"Using batch size of {self.batch_size} for prediction")

        # Use all available GPUs for headless mode
        if torch.cuda.device_count() > 1 and device != "cpu" and headless:
            model = nn.DataParallel(model)
            logger.info(
                f"Using {torch.cuda.device_count()} GPUs for prediction. "
                f"Increasing batch size to {torch.cuda.device_count()} * {self.batch_size}"
            )
            self.batch_size *= torch.cuda.device_count()
            self.device = "cuda"

        self.model = model.to(self.device)
        self.out_channels = out_channels
        self.verbose_logging = verbose_logging
        self.disable_tqdm = disable_tqdm
        self.is_embedding = is_embedding

        # channels_last_3d is opt-in and, when requested, only applies to 3D
        # models on CUDA (a memory-layout change, outputs are unaffected); keep
        # it off on CPU and for 2D models. Idempotent: the probes may have
        # converted the model already.
        if self.channels_last_3d:
            self.model = self.model.to(memory_format=torch.channels_last_3d)

        # `torch.compile` is decided per `__call__` from the number of batches
        # (see `_maybe_compile`); `None` means auto.
        self.compile = compile
        self._eager_model = self.model
        self._compiled = False
        # a failed compile (eager fallback) is remembered so later `__call__`s
        # do not retry a known-failing compile
        self._compile_failed = False

    def _shapes_description(self) -> str:
        return f"patch shape {self.patch} with halo {self.patch_halo}"

    def _should_compile(self, num_batches: int) -> bool:
        if self.compile is not None:
            return self.compile
        return (
            torch.device(self.device).type == "cuda"
            and num_batches >= COMPILE_MIN_BATCHES
        )

    def _maybe_compile(self, num_batches: int) -> None:
        if self._compiled or self._compile_failed:
            return
        if self._should_compile(num_batches):
            self._enable_compile(num_batches)
        elif self.compile is None:
            if torch.device(self.device).type != "cuda":
                logger.info(
                    "Skipping torch.compile: device %s is not CUDA.", self.device
                )
            else:
                logger.info(
                    "Skipping torch.compile: %d batches < %d required to amortize "
                    "the compile cost.",
                    num_batches,
                    COMPILE_MIN_BATCHES,
                )

    def _enable_compile(self, num_batches: int) -> None:
        # Capture the execution-eager model after DataParallel / channels_last_3d
        # wrapping so the fallback in `_forward` is exactly that model.
        self._eager_model = self.model
        try:
            self.model = torch.compile(self.model)
            self._compiled = True
            logger.info("Using torch.compile for %d batches.", num_batches)
        except Exception as e:
            self._compile_failed = True
            logger.warning("torch.compile is unavailable (%s); running eager.", e)

    def _forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Model forward, degrading `torch.compile` to eager on first-forward failure.

        `torch.compile` wraps lazily and only fails at the first forward (e.g.
        triton cannot find ``cuda.h``). Such a failure is retried once with the
        eager model — nothing has been accumulated for the failed batch — so
        prediction continues instead of aborting mid-run.
        """
        try:
            return self.model(input_)
        except RuntimeError as e:
            if not self._compiled or is_oom_error(e, self.device):
                raise
            logger.warning(
                "torch.compile failed at inference (%s); falling back to eager. "
                "In some environments triton needs C_INCLUDE_PATH=/usr/local/cuda/include.",
                e,
            )
            self.model = self._eager_model
            self._compiled = False
            self._compile_failed = True
            return self.model(input_)

    def __call__(self, test_dataset: Dataset) -> np.ndarray:
        assert isinstance(test_dataset, ArrayDataset), (
            "Dataset must be an instance of ArrayDataset"
        )
        assert self.patch_halo == test_dataset.halo_shape, (
            f"Predictor halo shape {self.patch_halo} does not match dataset halo shape {test_dataset.halo_shape}"
        )

        # dimensionality of the output prediction
        volume_shape = self.volume_shape(test_dataset)
        model_is_2d = is_2d_model(self.model)
        if self.is_embedding:
            if model_is_2d:
                # outputs 1-affinities in XY
                out_channels = 2
            else:
                # outputs 1-affinities in XYZ
                out_channels = 3
        else:
            out_channels = self.out_channels

        batch_size = self.batch_size
        while True:
            try:
                prediction = self._predict_with_batch_size(
                    test_dataset, batch_size, out_channels, volume_shape, model_is_2d
                )
            except RuntimeError as error:
                if not is_oom_error(error, self.device):
                    raise
                smaller_batch_size = batch_size // 2
                if smaller_batch_size < 1:
                    raise RuntimeError(
                        f"Prediction ran out of memory on {self.device} for "
                        f"{self._shapes_description()}, even at batch size "
                        f"{batch_size}. Please reduce the patch size/halo or free "
                        "up device memory."
                    ) from error
                logger.warning(
                    f"Prediction ran out of memory at batch size {batch_size}, "
                    f"retrying with batch size {smaller_batch_size}."
                )
                batch_size = smaller_batch_size
                self.batch_size = batch_size
            else:
                return prediction

    def _predict_with_batch_size(
        self,
        test_dataset: Dataset,
        batch_size: int,
        out_channels: int,
        volume_shape: tuple[int, int, int],
        model_is_2d: bool,
    ) -> np.ndarray:
        """Run the full prediction loop with one batch size.

        Raises the original `RuntimeError` if a batch does not fit, so `__call__`
        can retry the whole run with a smaller batch. The partially filled output
        arrays are discarded together with this frame.
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=True,
            collate_fn=default_prediction_collate,
        )
        num_batches = len(test_loader)

        if self.verbose_logging:
            logger.info(f"Running prediction on {num_batches} batches")

        prediction_maps_shape = (out_channels,) + volume_shape

        if self.verbose_logging:
            logger.info(
                f"The shape of the output prediction maps (CDHW): {prediction_maps_shape}"
            )
            logger.info(f"Using patch_halo: {self.patch_halo}")
            # allocate prediction and normalization arrays
            logger.info("Allocating prediction and normalization arrays...")

        # initialize the output prediction arrays
        prediction_map = np.zeros(prediction_maps_shape, dtype="float32")
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_mask = np.zeros(prediction_maps_shape, dtype="uint8")

        # run prediction
        # Sets the module in evaluation mode explicitly
        # It is necessary for batchnorm/dropout layers if present (the final
        # activation is applied explicitly, see `_activate`)
        self.model.eval()
        # decide on `torch.compile` from the actual number of batches; the 2D
        # check above deliberately sees the uncompiled model
        self._maybe_compile(num_batches)
        # Run prediction on the entire input dataset
        if self.tracker is not None:
            self.tracker.total = num_batches
            self.tracker.progress = 0
        try:
            # no cuDNN autotuning: at the batch counts prediction typically sees
            # the per-shape tuning cost does not amortize, and it gave no
            # steady-state gain in the prediction loop either; the forward
            # passes run in the same precision mode as the memory probes
            with amp_context(self.amp, self.device), torch.no_grad():
                for input_, indices in tqdm.tqdm(
                    test_loader, disable=self.disable_tqdm
                ):
                    if self.tracker is not None:
                        self.tracker.progress += 1
                    # the loader pins the host batches, so a pinned batch copies to
                    # the GPU asynchronously (`to_device`); the copy is ordered
                    # before the forward pass on the same stream
                    input_ = to_device(input_, self.device)  # padded with halo
                    if self.channels_last_3d:
                        input_ = input_.to(memory_format=torch.channels_last_3d)
                    # forward pass
                    if model_is_2d:
                        # remove the singleton z-dimension from the input
                        input_ = torch.squeeze(input_, dim=-3)
                        prediction = self._forward(input_)
                        # add the singleton z-dimension to the output
                        prediction = torch.unsqueeze(prediction, dim=-3)
                    else:
                        prediction = self._forward(input_)

                    if self.is_embedding:
                        if model_is_2d:
                            offsets = [[-1, 0], [0, -1]]
                        else:
                            offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
                        # keep today's numbers: this branch consumes the model's
                        # activated output (Softmax for sparse models), not raw
                        # embeddings; the semantics are undecided, see
                        # KRiedmiller/panseg#5
                        prediction = self._activate(prediction)
                        # convert embeddings to affinities
                        prediction = embeddings_to_affinities(
                            prediction, offsets, delta=0.5
                        )
                        # average across channels and invert (i.e. 1-affinities)
                        prediction = 1 - prediction.mean(dim=1)
                    # removing halo from the prediction
                    prediction = remove_padding(prediction, self.patch_halo)
                    # convert to numpy array; the accumulated maps are float32
                    # whatever the compute precision (bf16 cannot convert to numpy),
                    # and contiguous whatever the channels_last_3d input layout
                    prediction = prediction.float()
                    if not self.is_embedding:
                        # logits -> probabilities per patch, before the
                        # overlap-averaging below, in fp32 after the cast above (the
                        # torch.compile forward graph stays activation-free)
                        prediction = self._activate(prediction)
                    prediction = prediction.contiguous().cpu().numpy()

                    channel_slice = slice(0, out_channels)
                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # add channel dimension to the index
                        index = (channel_slice,) + tuple(index)
                        # accumulate probabilities into the output prediction array
                        prediction_map[index] += pred
                        # count voxel visits for normalization
                        normalization_mask[index] += 1
        except RuntimeError:
            # Drop the GPU tensors still referenced by this frame before the
            # caller retries with a smaller batch. The traceback keeps the frame
            # (and its locals) alive until the exception is handled.
            input_ = prediction = None
            if self.device != "cpu":
                torch.cuda.empty_cache()
            raise

        if self.verbose_logging:
            logger.info("Prediction finished")

        # normalize results and return
        return prediction_map / normalization_mask

    @staticmethod
    def volume_shape(dataset: Dataset) -> tuple[int, int, int]:
        raw = dataset.raw
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]
