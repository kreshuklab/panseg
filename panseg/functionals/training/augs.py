import logging
import random
from typing import Callable

import numpy as np
import torch
from scipy.ndimage import convolve, gaussian_filter, map_coordinates, rotate
from skimage import measure
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries

GLOBAL_RANDOM_STATE_SEED = 47
GLOBAL_RANDOM_STATE = np.random.RandomState(GLOBAL_RANDOM_STATE_SEED)

logger = logging.getLogger(__name__)
logger.info(
    f"Using fixed random state {GLOBAL_RANDOM_STATE_SEED} for reproducibility, set GLOBAL_RANDOM_STATE to None to use random state based on time"
)


class Compose:
    """
    Composes several transforms together.

    Args:
        transforms: list of transforms to compose.
    """

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, m):
        for t in self.transforms:
            m = t(m)
        return m


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, axis_prob=0.5, **kwargs):
        assert random_state is not None, "RandomState cannot be None"
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m):
        assert m.ndim in [3, 4], "Supports only 3D (DxHxW) or 4D (CxDxHxW) images"

        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, **kwargs):
        self.random_state = random_state
        # always rotate around z-axis
        self.axis = (1, 2)

    def __call__(self, m):
        assert m.ndim in [3, 4], "Supports only 3D (DxHxW) or 4D (CxDxHxW) images"

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, self.axis)
        else:
            channels = [np.rot90(m[c], k, self.axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(
        self,
        random_state,
        angle_spectrum=30,
        axes=None,
        mode="reflect",
        order=0,
        **kwargs,
    ):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(
                m,
                angle,
                axes=axis,
                reshape=False,
                order=self.order,
                mode=self.mode,
                cval=-1,
            )
        else:
            channels = [
                rotate(
                    m[c],
                    angle,
                    axes=axis,
                    reshape=False,
                    order=self.order,
                    mode=self.mode,
                    cval=-1,
                )
                for c in range(m.shape[0])
            ]
            m = np.stack(channels, axis=0)

        return m


class RandomContrast:
    """
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(
        self,
        random_state,
        alpha=(0.5, 1.5),
        mean=0.0,
        execution_probability=0.1,
        **kwargs,
    ):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (m - self.mean)
            return np.clip(result, -1, 1)

        return m


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=0 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(
        self,
        random_state,
        spline_order,
        alpha=2000,
        sigma=50,
        execution_probability=0.1,
        apply_3d=True,
        **kwargs,
    ):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim in [3, 4]

            if m.ndim == 3:
                volume_shape = m.shape
            else:
                volume_shape = m[0].shape

            if self.apply_3d:
                dz = (
                    gaussian_filter(
                        self.random_state.randn(*volume_shape),
                        self.sigma,
                        mode="reflect",
                    )
                    * self.alpha
                )
            else:
                dz = np.zeros_like(m)

            dy, dx = [
                gaussian_filter(
                    self.random_state.randn(*volume_shape), self.sigma, mode="reflect"
                )
                * self.alpha
                for _ in range(2)
            ]

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(
                np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing="ij"
            )
            indices = z + dz, y + dy, x + dx

            if m.ndim == 3:
                return map_coordinates(
                    m, indices, order=self.spline_order, mode="reflect"
                )
            else:
                channels = [
                    map_coordinates(c, indices, order=self.spline_order, mode="reflect")
                    for c in m
                ]
                return np.stack(channels, axis=0)

        return m


class CropToFixed:
    def __init__(self, random_state, size=(256, 256), centered=False, **kwargs):
        self.random_state = random_state
        self.crop_y, self.crop_x = size
        self.centered = centered

    def __call__(self, m):
        def _padding(pad_total):
            half_total = pad_total // 2
            return (half_total, pad_total - half_total)

        def _rand_range_and_pad(crop_size, max_size):
            """
            Returns a tuple:
                max_value (int) for the corner dimension. The corner dimension is chosen as `self.random_state(max_value)`
                pad (int): padding in both directions; if crop_size is lt max_size the pad is 0
            """
            if crop_size < max_size:
                return max_size - crop_size, (0, 0)
            else:
                return 1, _padding(crop_size - max_size)

        def _start_and_pad(crop_size, max_size):
            if crop_size < max_size:
                return (max_size - crop_size) // 2, (0, 0)
            else:
                return 0, _padding(crop_size - max_size)

        assert m.ndim in (3, 4)
        if m.ndim == 3:
            _, y, x = m.shape
        else:
            _, _, y, x = m.shape

        if not self.centered:
            y_range, y_pad = _rand_range_and_pad(self.crop_y, y)
            x_range, x_pad = _rand_range_and_pad(self.crop_x, x)

            y_start = self.random_state.randint(y_range)
            x_start = self.random_state.randint(x_range)

        else:
            y_start, y_pad = _start_and_pad(self.crop_y, y)
            x_start, x_pad = _start_and_pad(self.crop_x, x)

        if m.ndim == 3:
            result = m[
                :, y_start : y_start + self.crop_y, x_start : x_start + self.crop_x
            ]
            return np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode="reflect")
        else:
            channels = []
            for c in range(m.shape[0]):
                result = m[c][
                    :, y_start : y_start + self.crop_y, x_start : x_start + self.crop_x
                ]
                channels.append(
                    np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode="reflect")
                )
            return np.stack(channels, axis=0)


class StandardLabelToBoundary:
    def __init__(
        self,
        ignore_index=None,
        append_label=False,
        mode="thick",
        foreground=False,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.mode = mode
        self.foreground = foreground

    def __call__(self, m):
        assert m.ndim == 3, f"Only 3D labels supported, dimensionality: {m.ndim}"

        boundaries = find_boundaries(m, connectivity=2, mode=self.mode)
        boundaries = boundaries.astype("int32")

        results = []
        if self.foreground:
            foreground = (m > 0).astype("uint8")
            results.append(_recover_ignore_index(foreground, m, self.ignore_index))

        results.append(_recover_ignore_index(boundaries, m, self.ignore_index))

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


class Standardize:
    """
    Apply Z-score normalization (0-mean, 1-std) to a given input tensor.

    Args:
        eps (float, optional): A small value added to the denominator for
            numerical stability. Defaults to 1e-10.
        mean (Optional[float]): The mean to use for standardization. If None,
            the mean of the input is used. Defaults to None.
        std (Optional[float]): The standard deviation to use for standardization.
            If None, the std of the input is used. Defaults to None.
        channelwise (bool, optional): Whether to apply the normalization
            channel-wise. Channel must be first dimension. Defaults to False.

    Raises:
        AssertionError: If mean or std is provided, both must be provided.
    """

    def __init__(self, mean=None, std=None, channelwise=False, eps=1e-10, **kwargs):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise

    def __call__(self, m):
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per-channel
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)


class PercentileNormalizer:
    def __init__(self, pmin=1, pmax=99.6, channelwise=False, eps=1e-10, **kwargs):
        self.eps = eps
        self.pmin = pmin
        self.pmax = pmax
        self.channelwise = channelwise

    def __call__(self, m):
        if self.channelwise:
            axes = list(range(m.ndim))
            # average across channels
            axes = tuple(axes[1:])
            pmin = np.percentile(m, self.pmin, axis=axes, keepdims=True)
            pmax = np.percentile(m, self.pmax, axis=axes, keepdims=True)
        else:
            pmin = np.percentile(m, self.pmin)
            pmax = np.percentile(m, self.pmax)

        return (m - pmin) / (pmax - pmin + self.eps)


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, min_value, max_value, **kwargs):
        assert max_value > min_value
        self.min_value = min_value
        self.value_range = max_value - min_value

    def __call__(self, m):
        norm_0_1 = (m - self.min_value) / self.value_range
        return np.clip(2 * norm_0_1 - 1, -1, 1)


class AdditiveGaussianNoise:
    def __init__(
        self, random_state, scale=(0.0, 1.0), execution_probability=0.1, **kwargs
    ):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.scale = scale

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            std = self.random_state.uniform(self.scale[0], self.scale[1])
            gaussian_noise = self.random_state.normal(0, std, size=m.shape)
            return m + gaussian_noise
        return m


class AdditivePoissonNoise:
    def __init__(
        self, random_state, lam=(0.0, 1.0), execution_probability=0.1, **kwargs
    ):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.lam = lam

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            lam = self.random_state.uniform(self.lam[0], self.lam[1])
            poisson_noise = self.random_state.poisson(lam, size=m.shape)
            return m + poisson_noise
        return m


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor.

    Args:
        expand_dims (bool): if True, adds a channel dimension to the input data
        dtype (np.dtype): the desired output data type
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], "Supports only 3D (DxHxW) or 4D (CxDxHxW) images"
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class Relabel:
    """
    Relabel a numpy array of labels into a consecutive numbers, e.g.
    [10, 10, 0, 6, 6] -> [2, 2, 0, 1, 1]. Useful when one has an instance segmentation volume
    at hand and would like to create a one-hot-encoding for it. Without a consecutive labeling the task would be harder.
    """

    def __init__(self, append_original=False, run_cc=True, ignore_label=None, **kwargs):
        self.append_original = append_original
        self.ignore_label = ignore_label
        self.run_cc = run_cc

        if ignore_label is not None:
            assert append_original, (
                "ignore_label present, so append_original must be true, so that one can localize the ignore region"
            )

    def __call__(self, m):
        orig = m
        if self.run_cc:
            # assign 0 to the ignore region
            m = measure.label(m, background=self.ignore_label)

        _, unique_labels = np.unique(m, return_inverse=True)
        result = unique_labels.reshape(m.shape)
        if self.append_original:
            result = np.stack([result, orig])
        return result


class Identity:
    def __init__(self, **kwargs):
        pass

    def __call__(self, m):
        return m


class RgbToLabel:
    def __call__(self, img):
        img = np.array(img)
        assert img.ndim == 3 and img.shape[2] == 3
        return img[..., 0] * 65536 + img[..., 1] * 256 + img[..., 2]


class LabelToTensor:
    def __call__(self, m):
        m = np.array(m)
        return torch.from_numpy(m.astype(dtype="int64"))


class GaussianBlur3D:
    def __init__(self, sigma=[0.1, 2.0], execution_probability=0.5, **kwargs):
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, x):
        if random.random() < self.execution_probability:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = gaussian(x, sigma=sigma)
            return x
        return x


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input


class Augmenter:
    """Image augmentations for training.
    Only for 3d images. Inputs must have a channel dimension, labels cannot.

    Attributes:
        seed: Seed for all random operations.
    """

    def __init__(self):
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)

    def raw_transform(self, stats):
        return Compose(
            [
                Standardize(mean=stats["mean"], std=stats["std"]),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(
                    np.random.RandomState(self.seed),
                    axes=[[2, 1]],
                    angle_spectrum=45,
                    mode="reflect",
                ),
                GaussianBlur3D(),
                AdditiveGaussianNoise(np.random.RandomState()),
                AdditivePoissonNoise(np.random.RandomState()),
                ToTensor(expand_dims=False),
            ]
        )

    def label_transform(self):
        return Compose(
            [
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(
                    np.random.RandomState(self.seed),
                    axes=[[2, 1]],
                    angle_spectrum=45,
                    mode="reflect",
                ),
                StandardLabelToBoundary(),
                ToTensor(expand_dims=False),
            ]
        )


def get_test_augmentations(raw: np.ndarray | None, expand_dims: bool = True) -> Compose:
    """
    Constructs a set of data transformations for inference.
    Uses global mean and standard deviation of the provided raw data if available;
    otherwise, it calculatesthese statistics on-the-fly per patch.

    Args:
        raw (Optional[ndarray]): The raw data to compute global statistics.
                                 If None, statistics are computedduring transformation per patch.
        expand_dims (bool): if True, adds a channel dimension to the input data.

    Returns:
        Compose: A composed transformation of standardization and tensor conversion.
    """
    mean = np.mean(raw) if raw is not None else None
    std = np.std(raw) if raw is not None else None

    return Compose([Standardize(mean=mean, std=std), ToTensor(expand_dims=expand_dims)])
