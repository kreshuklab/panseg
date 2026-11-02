import logging

import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

from panseg.functionals.precision import amp_context, cudnn_benchmark, validate_amp
from panseg.functionals.prediction.utils.array_dataset import (
    ArrayDataset,
    default_prediction_collate,
    remove_padding,
)
from panseg.functionals.prediction.utils.size_finder import (
    _is_2d_model,
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
        amp (str, optional): precision mode (``"auto"``: bf16 autocast on
            supported CUDA devices, else fp32; ``"off"``: always fp32). The
            memory probes run in the same mode. Defaults to "auto".

    Attributes:
        batch_size (int): Calculated batch size based on device capabilities and model requirements.
        device (str): Device where the model will be run.
        model (nn.Module): Model configured for evaluation.
        out_channels (int): Number of channels expected in the output.
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
        amp: str = "auto",
    ):
        self.device = device
        self.tracker = tracker
        self.amp = validate_amp(amp)

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
            ):
                raise RuntimeError(
                    "OOM error will happen. Please reduce the patch size/halo."
                )
        else:  # find the max batch size without causing OOM, may be [0, 1, 2, 4, 8, 16, 32, 64, 128]
            self.batch_size = find_batch_size(
                model, in_channels, patch, patch_halo, device, amp=self.amp
            )
            if self.batch_size < 1:
                raise RuntimeError(
                    "Could not determine a feasible batch size for the given model and patch size/halo."
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
        self.patch_halo = patch_halo
        self.verbose_logging = verbose_logging
        self.disable_tqdm = disable_tqdm
        self.is_embedding = is_embedding

    def __call__(self, test_dataset: Dataset) -> np.ndarray:
        assert isinstance(test_dataset, ArrayDataset), (
            "Dataset must be an instance of ArrayDataset"
        )
        assert self.patch_halo == test_dataset.halo_shape, (
            f"Predictor halo shape {self.patch_halo} does not match dataset halo shape {test_dataset.halo_shape}"
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=default_prediction_collate,
        )

        if self.verbose_logging:
            logger.info(f"Running prediction on {len(test_loader)} batches")

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
        # every patch has the same shape, so cuDNN autotuning pays off; the
        # forward passes run in the same precision mode as the memory probes
        with cudnn_benchmark(), amp_context(self.amp, self.device), torch.no_grad():
            for input_, indices in tqdm.tqdm(test_loader, disable=self.disable_tqdm):
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
                # convert to numpy array; the accumulated maps are float32
                # whatever the compute precision (bf16 cannot convert to numpy)
                prediction = prediction.float().cpu().numpy()

                channel_slice = slice(0, out_channels)
                # for each batch sample
                for pred, index in zip(prediction, indices):
                    # add channel dimension to the index
                    index = (channel_slice,) + tuple(index)
                    # accumulate probabilities into the output prediction array
                    prediction_map[index] += pred
                    # count voxel visits for normalization
                    normalization_mask[index] += 1

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
