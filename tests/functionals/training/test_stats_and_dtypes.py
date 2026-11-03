"""Raw-loading dtype and dataset statistics (findings 6+7 of the perf analysis).

The raw volume is loaded as float32 and kept float32 through `Standardize` and
`ToTensor` (the old pipeline promoted uint16 -> float64 -> float32, two extra
full copies per patch). `calculate_stats` computes the percentiles in one
combined pass without materializing a concatenated copy; its results must stay
float32-accurate and are stored as float32.
"""

import random
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import torch

from panseg.functionals.training.augs import (
    AdditiveGaussianNoise,
    AdditivePoissonNoise,
    Augmenter,
    Standardize,
    ToTensor,
)
from panseg.functionals.training.h5dataset import (
    HDF5Dataset,
    _percentiles,
    calculate_stats,
)


def _reference_stats(images):
    """The old `calculate_stats` algorithm: full concatenation copy plus two
    separate percentile computations. Kept here as the independent source of
    truth the new implementation must match."""
    flat = np.concatenate([img.ravel() for img in images])
    return {
        "pmin": np.percentile(flat, 1),
        "pmax": np.percentile(flat, 99.6),
        "mean": np.mean(flat),
        "std": np.std(flat),
    }


@pytest.fixture()
def raw_u16():
    """A uint16 raw volume with realistic dynamic range."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 5000, size=(1, 8, 96, 96)).astype("uint16")


class TestCalculateStats:
    def test_matches_the_reference_implementation(self, raw_u16):
        reference = _reference_stats(raw_u16)

        for images in (raw_u16, raw_u16.astype(np.float32)):
            stats = calculate_stats(images)
            for key in ("pmin", "pmax", "mean", "std"):
                value = stats[key]
                assert value is not None, key
                np.testing.assert_allclose(
                    value, reference[key], rtol=1e-6, err_msg=key
                )

    @pytest.mark.parametrize(
        "values, qs",
        [
            (np.zeros(1000, dtype="uint16"), (1, 50, 99.6)),
            (np.full(7, 7, dtype="int16"), (1, 25, 99.6)),
            (np.arange(1001, dtype="int32"), (1, 25, 50, 75, 99.6)),
            (np.array([3], dtype="uint8"), (1, 50, 99.6)),
            (np.array([1, 2], dtype="int8"), (1, 33.3, 50, 66.6, 99.6)),
        ],
    )
    def test_integer_percentiles_match_numpy_exactly(self, values, qs):
        """The histogram percentile path must reproduce `np.percentile` exactly
        (numpy's 'linear' method), including the interpolation rounding."""
        rng = np.random.default_rng(3)
        noisy = (values.astype("int64") + rng.integers(0, 3, values.size)).astype(
            values.dtype
        )

        for data in (values, noisy):
            result = _percentiles(data.reshape(-1), tuple(qs))
            np.testing.assert_array_equal(result, np.percentile(data, qs))

    def test_integer_percentiles_randomized(self):
        rng = np.random.default_rng(5)
        qs = (1, 17.5, 50, 83.3, 99.6)
        for _ in range(20):
            n = int(rng.integers(1, 5000))
            spread = int(rng.integers(1, 2000))
            data = rng.integers(-spread, spread + 1, size=n).astype("int16")
            np.testing.assert_array_equal(
                _percentiles(data.reshape(-1), qs), np.percentile(data, qs)
            )

    def test_stores_the_stats_as_float32(self, raw_u16):
        stats = calculate_stats(raw_u16.astype(np.float32))

        for key in ("pmin", "pmax", "mean", "std"):
            assert np.asarray(stats[key]).dtype == np.float32, key

    def test_accepts_a_list_of_arrays(self, raw_u16):
        chunks = [raw_u16[0, :4], raw_u16[0, 4:]]
        reference = _reference_stats(raw_u16)

        stats = calculate_stats([chunk.astype(np.float32) for chunk in chunks])

        for key in ("pmin", "pmax", "mean", "std"):
            value = stats[key]
            assert value is not None, key
            np.testing.assert_allclose(value, reference[key], rtol=1e-6, err_msg=key)

    def test_global_normalization_off_returns_nones(self):
        assert calculate_stats(np.zeros((2, 2)), global_normalization=False) == {
            "pmin": None,
            "pmax": None,
            "mean": None,
            "std": None,
        }


class TestFloat32Pipeline:
    def test_dataset_raw_is_loaded_as_float32(self, tmp_path, raw_u16):
        label = np.zeros(raw_u16.shape[1:], dtype="uint8")
        label[:4] = 1
        file_path = tmp_path / "vol.h5"
        with h5py.File(file_path, "w") as f:
            f.create_dataset("raw", data=raw_u16)
            f.create_dataset("label", data=label)

        dataset = HDF5Dataset(
            file_path=file_path,
            augmenter=MagicMock(),
            patch_shape=(2, 64, 64),
            dimensionality="3D",
        )

        assert dataset.raw.dtype == np.float32
        assert np.array_equal(dataset.raw, raw_u16.astype(np.float32))

    def test_standardize_keeps_float32_patches(self):
        patch = np.random.default_rng(1).random((2, 8, 8), dtype=np.float32)
        # float64 stats must not up-cast the float32 patch
        standardize = Standardize(mean=np.float64(0.5), std=np.float64(2.0))

        out = standardize(patch)

        assert out.dtype == np.float32
        np.testing.assert_allclose(out, (patch - 0.5) / 2.0, rtol=1e-6, atol=1e-7)

    def test_noise_keeps_float32_patches(self):
        patch = np.random.default_rng(1).random((2, 8, 8), dtype=np.float32)

        for transform in (
            AdditiveGaussianNoise(np.random.RandomState(0), execution_probability=1.0),
            AdditivePoissonNoise(np.random.RandomState(0), execution_probability=1.0),
        ):
            out = transform(patch)
            assert out.dtype == np.float32, type(transform).__name__

    def test_to_tensor_does_not_copy_contiguous_float32(self):
        patch = np.random.default_rng(1).random((1, 8, 8), dtype=np.float32)

        tensor = ToTensor(expand_dims=False)(patch)

        assert tensor.dtype == torch.float32
        assert np.shares_memory(tensor.numpy(), patch), (
            "a contiguous float32 array must not be copied again"
        )

    def test_to_tensor_copies_other_dtypes_and_strides(self):
        for patch in (
            np.random.default_rng(1).random((1, 8, 8)),  # float64
            np.random.default_rng(1).random((1, 8, 8), dtype=np.float32)[
                :, :, ::-1
            ],  # negative strides
        ):
            tensor = ToTensor(expand_dims=False)(patch)

            assert tensor.dtype == torch.float32
            assert tensor.shape == (1, 8, 8)
            np.testing.assert_allclose(tensor.numpy(), np.ascontiguousarray(patch))

    def test_raw_transform_matches_the_float64_reference(self, raw_u16, monkeypatch):
        """The whole raw transform chain on the float32 pipeline must match the
        old float64 pipeline within float32 precision (the random draws are
        synchronized, so both chains see identical augmentations)."""
        real_random_state = np.random.RandomState

        def seeded_random_state(*args):
            return real_random_state(*(args if args else (47,)))

        monkeypatch.setattr(np.random, "RandomState", seeded_random_state)

        reference_stats = _reference_stats(raw_u16)
        raw_f32 = raw_u16.astype(np.float32)
        new_stats = calculate_stats(raw_f32)

        aug_reference, aug_new = Augmenter(), Augmenter()
        aug_reference.seed = aug_new.seed = 1234
        transform_reference = aug_reference.raw_transform(reference_stats)
        transform_new = aug_new.raw_transform(new_stats)

        random.seed(7)
        out_reference = transform_reference(raw_u16.astype(np.float64))
        random.seed(7)
        out_new = transform_new(raw_f32)

        assert out_reference.dtype == torch.float32
        assert out_new.dtype == torch.float32
        np.testing.assert_allclose(
            out_new.numpy(), out_reference.numpy(), rtol=1e-5, atol=1e-6
        )
