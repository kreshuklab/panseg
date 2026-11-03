import logging

import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

from panseg.functionals.prediction.utils.array_dataset import (
    ArrayDataset,
    default_prediction_collate,
    remove_padding,
)
from panseg.functionals.prediction.utils.size_finder import (
    _is_2d_model,
    _is_oom_error,
    find_batch_size,
    will_CUDA_OOM,
)
from panseg.functionals.training.embeddings import embeddings_to_affinities

logger = logging.getLogger(__name__)


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
    ):
        self.device = device
        self.tracker = tracker
        self.patch = patch
        self.patch_halo = patch_halo

        if single_batch_mode:  # then check if OOM happens at batch size 1
            self.batch_size = 1
            if device != "cpu" and will_CUDA_OOM(
                model, in_channels, patch, patch_halo, self.batch_size, device
            ):
                raise RuntimeError(
                    f"Prediction will run out of memory on {device} for "
                    f"{self._shapes_description()}, even at batch size "
                    f"{self.batch_size}. Please reduce the patch size/halo or free "
                    "up device memory."
                )
        else:  # find the max batch size without causing OOM, may be [1, 2, 4, 8, 16, ...]
            self.batch_size = find_batch_size(
                model, in_channels, patch, patch_halo, device
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

    def _shapes_description(self) -> str:
        return f"patch shape {self.patch} with halo {self.patch_halo}"

    def __call__(self, test_dataset: Dataset) -> np.ndarray:
        assert isinstance(test_dataset, ArrayDataset), (
            "Dataset must be an instance of ArrayDataset"
        )
        assert self.patch_halo == test_dataset.halo_shape, (
            f"Predictor halo shape {self.patch_halo} does not match dataset halo shape {test_dataset.halo_shape}"
        )

        # dimensionality of the output prediction
        volume_shape = self.volume_shape(test_dataset)
        is_2d_model = _is_2d_model(self.model)
        if self.is_embedding:
            if is_2d_model:
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
                    test_dataset, batch_size, out_channels, volume_shape, is_2d_model
                )
            except RuntimeError as error:
                if not _is_oom_error(error, self.device):
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
        is_2d_model: bool,
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

        if self.verbose_logging:
            logger.info(f"Running prediction on {len(test_loader)} batches")

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
        # It is necessary for batchnorm/dropout layers if present as well as final Sigmoid/Softmax to be applied
        self.model.eval()
        # Run prediction on the entire input dataset
        if self.tracker is not None:
            self.tracker.total = len(test_loader)
            self.tracker.progress = 0
        try:
            with torch.no_grad():
                for input_, indices in tqdm.tqdm(
                    test_loader, disable=self.disable_tqdm
                ):
                    if self.tracker is not None:
                        self.tracker.progress += 1
                    input_ = input_.to(
                        self.device
                    )  # input is padded with halo in dataset __getitem__
                    # forward pass
                    if is_2d_model:
                        # remove the singleton z-dimension from the input
                        input_ = torch.squeeze(input_, dim=-3)
                        prediction = self.model(input_)
                        # add the singleton z-dimension to the output
                        prediction = torch.unsqueeze(prediction, dim=-3)
                    else:
                        prediction = self.model(input_)

                    if self.is_embedding:
                        if is_2d_model:
                            offsets = [[-1, 0], [0, -1]]
                        else:
                            offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
                        # convert embeddings to affinities
                        prediction = embeddings_to_affinities(
                            prediction, offsets, delta=0.5
                        )
                        # average across channels and invert (i.e. 1-affinities)
                        prediction = 1 - prediction.mean(dim=1)
                    # removing halo from the prediction
                    prediction = remove_padding(prediction, self.patch_halo)
                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

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
