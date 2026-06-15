import numpy as np
import pytest
import torch

from panseg.functionals.training import augs


def test_compose(mocker):
    mocks = [mocker.Mock(), mocker.Mock()]
    compose = augs.Compose(mocks)

    compose("smth")

    [m.assert_called_once() for m in mocks]


def test_random_flip_4d(mocker, raw_zcyx_96x2x96x96):
    img = raw_zcyx_96x2x96x96
    mock_random = mocker.Mock()
    mock_random.uniform.return_value = 0

    rf = augs.RandomFlip(mock_random)

    np.testing.assert_equal(img, rf(img))
    mock_random.uniform.return_value = 1
    with pytest.raises(AssertionError):
        np.testing.assert_equal(img, rf(img))


def test_random_flip_3d(mocker, raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    mock_random = mocker.Mock()
    mock_random.uniform.return_value = 0

    rf = augs.RandomFlip(mock_random)

    np.testing.assert_equal(img, rf(img))
    mock_random.uniform.return_value = 1
    with pytest.raises(AssertionError):
        np.testing.assert_equal(img, rf(img))


def test_random_flip_2d(mocker, raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    mock_random = mocker.Mock()
    mock_random.uniform.return_value = 0

    rf = augs.RandomFlip(mock_random)

    with pytest.raises(AssertionError):
        rf(img)


def test_random_rotate_90_3d(mocker, raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    mock_random = mocker.Mock()
    mock_random.randint.return_value = 0
    rr = augs.RandomRotate90(mock_random)

    np.testing.assert_equal(img, rr(img))
    mock_random.randint.return_value = 1
    # When k=1, we expect rotation to occur
    result = rr(img)
    assert result.shape == img.shape
    # Verify that rotation actually happened
    assert not np.array_equal(img, result) or np.array_equal(
        np.rot90(img, 1, (1, 2)), result
    )


def test_random_rotate_90_4d(mocker, raw_zcyx_96x2x96x96):
    img = raw_zcyx_96x2x96x96
    mock_random = mocker.Mock()
    mock_random.randint.return_value = 0
    rr = augs.RandomRotate90(mock_random)

    np.testing.assert_equal(img, rr(img))
    mock_random.randint.return_value = 2
    # When k=2, we expect rotation to occur
    result = rr(img)
    assert result.shape == img.shape
    # Verify that rotation actually happened
    assert not np.array_equal(img, result) or np.array_equal(
        np.rot90(img, 2, (1, 2)), result
    )


def test_random_rotate_90_invalid_ndim(mocker):
    mock_random = mocker.Mock()
    rr = augs.RandomRotate90(mock_random)

    with pytest.raises(AssertionError):
        rr(np.random.rand(10, 10))


def test_random_rotate_3d(mocker, raw_cell_3d_100x128x128):
    mock_random = mocker.Mock()
    mock_random.randint.side_effect = [0, 10]  # axis, angle
    rr = augs.RandomRotate(mock_random, axes=[(0, 1), (1, 2)])

    result = rr(raw_cell_3d_100x128x128)
    assert result.shape == raw_cell_3d_100x128x128.shape
    mock_random.randint.assert_any_call(len([(0, 1), (1, 2)]))
    mock_random.randint.assert_any_call(-30, 30)


def test_random_rotate_4d(mocker, raw_zcyx_96x2x96x96):
    mock_random = mocker.Mock()
    mock_random.randint.side_effect = [1, -15]  # axis, angle
    rr = augs.RandomRotate(mock_random, axes=[(0, 1), (1, 2)])

    result = rr(raw_zcyx_96x2x96x96)
    assert result.shape == raw_zcyx_96x2x96x96.shape
    mock_random.randint.assert_any_call(len([(0, 1), (1, 2)]))
    mock_random.randint.assert_any_call(-30, 30)


def test_random_contrast_2d(mocker, raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    mock_random = mocker.Mock()
    mock_random.uniform.return_value = 0.5  # Below execution probability
    rc = augs.RandomContrast(mock_random, execution_probability=1.0)

    assert not np.array_equal(img, rc(img))
    assert mock_random.uniform.call_count == 2


def test_random_contrast_3d_no_change(mocker, raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    mock_random = mocker.Mock()
    mock_random.uniform.side_effect = [
        0.0,
        1.0,
    ]  # First call for execution, second for alpha
    rc = augs.RandomContrast(mock_random, execution_probability=1.0, alpha=(0.5, 1.5))

    result = rc(img)
    assert result.shape == img.shape
    np.testing.assert_equal(img, result)
    mock_random.uniform.assert_called()


def test_random_contrast_3d_change(mocker, raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    mock_random = mocker.Mock()
    mock_random.uniform.side_effect = [
        0.0,
        1.5,
    ]  # First call for execution, second for alpha
    rc = augs.RandomContrast(mock_random, execution_probability=1.0, alpha=(0.5, 1.5))

    result = rc(img)
    assert result.shape == img.shape
    assert not np.array_equal(img, result)
    mock_random.uniform.assert_called()


def test_random_contrast_4d(mocker, raw_zcyx_96x2x96x96):
    img = raw_zcyx_96x2x96x96
    mock_random = mocker.Mock()
    mock_random.uniform.side_effect = [
        0.0,
        0.8,
    ]  # First call for execution, second for alpha
    rc = augs.RandomContrast(mock_random, execution_probability=1.0, alpha=(0.5, 1.5))

    result = rc(img)
    assert result.shape == img.shape
    assert np.all(result >= -1) and np.all(result <= 1)
    assert not np.array_equal(img, result)
    mock_random.uniform.assert_called()


def test_random_contrast_no_execution(mocker, raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    mock_random = mocker.Mock()
    mock_random.uniform.return_value = 0.2  # Above execution probability
    rc = augs.RandomContrast(mock_random, execution_probability=0.1)

    np.testing.assert_equal(img, rc(img))
    mock_random.uniform.assert_called_once()


def test_elastic_deformation_3d(mocker, raw_cell_3d_100x128x128):
    mock_random = mocker.Mock()
    mock_random.uniform.return_value = 0.05  # below execution probability
    mock_random.randn.return_value = np.ones((100, 128, 128))

    ed = augs.ElasticDeformation(mock_random, spline_order=1, execution_probability=0.1)
    result = ed(raw_cell_3d_100x128x128)

    # Should perform deformation
    assert result.shape == raw_cell_3d_100x128x128.shape
    assert not np.array_equal(result, raw_cell_3d_100x128x128)


def test_elastic_deformation_4d(mocker, raw_zcyx_96x2x96x96):
    img = raw_zcyx_96x2x96x96.reshape((2, 96, 96, 96))
    mock_random = mocker.Mock()
    mock_random.uniform.return_value = 0.05  # below execution probability
    mock_random.randn.return_value = np.ones((96, 96, 96))

    ed = augs.ElasticDeformation(mock_random, spline_order=1, execution_probability=0.1)
    result = ed(img)

    # Should perform deformation
    assert result.shape == img.shape
    assert not np.array_equal(result, img)


def test_elastic_deformation_not_applied(mocker, raw_cell_3d_100x128x128):
    mock_random = mocker.Mock()
    mock_random.uniform.return_value = 0.2  # above execution probability

    ed = augs.ElasticDeformation(mock_random, spline_order=1, execution_probability=0.1)
    result = ed(raw_cell_3d_100x128x128)

    # Should not apply deformation
    np.testing.assert_equal(result, raw_cell_3d_100x128x128)


def test_elastic_deformation_2d_raises(mocker, raw_cell_2d_96x96):
    mock_random = mocker.Mock()
    mock_random.uniform.return_value = 0.05

    ed = augs.ElasticDeformation(mock_random, spline_order=1, execution_probability=0.1)

    with pytest.raises(AssertionError):
        ed(raw_cell_2d_96x96)


def test_crop_to_fixed_3d_centered(raw_cell_3d_100x128x128, mocker):
    """Test 3D image with centered cropping"""
    img = raw_cell_3d_100x128x128
    mock_random_state = mocker.Mock()
    cf = augs.CropToFixed(mock_random_state, size=(64, 64), centered=True)

    result = cf(img)
    assert result.shape == (1, img.shape[0], 64, 64) or result.shape == (
        img.shape[0],
        64,
        64,
    )


def test_crop_to_fixed_3d_not_centered(raw_cell_3d_100x128x128, mocker):
    """Test 3D image with non-centered cropping"""
    img = raw_cell_3d_100x128x128
    mock_random_state = mocker.Mock()
    mock_random_state.randint.return_value = 0
    cf = augs.CropToFixed(mock_random_state, size=(64, 64), centered=False)

    result = cf(img)
    assert result.shape == (img.shape[0], 64, 64)


def test_crop_to_fixed_4d(raw_zcyx_96x2x96x96, mocker):
    """Test 4D image with centered cropping"""
    img = raw_zcyx_96x2x96x96
    mock_random_state = mocker.Mock()
    mock_random_state.randint.return_value = 0
    cf = augs.CropToFixed(mock_random_state, size=(64, 64), centered=True)

    result = cf(img)
    assert result.shape == (img.shape[0], img.shape[1], 64, 64)


def test_crop_to_fixed_padding(raw_cell_3d_100x128x128):
    """Test error when crop size is larger than image size"""
    img = raw_cell_3d_100x128x128
    cf = augs.CropToFixed(np.random, size=(1000, 1000), centered=True)
    result = cf(img)
    assert result.shape == (img.shape[0], 1000, 1000)


def test_standard_label_to_boundary_3d(raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128

    transform = augs.StandardLabelToBoundary(foreground=True)
    result = transform(img)

    assert result.shape == (2, 100, 128, 128)  # 2 channels: boundary + foreground
    assert result.dtype == np.int32


def test_standard_label_to_boundary_with_ignore_index(raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128

    transform = augs.StandardLabelToBoundary(ignore_index=255)
    result = transform(img)

    assert result.shape == (1, 100, 128, 128)
    assert result.dtype == np.int32


def test_standard_label_to_boundary_append_label(raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    transform = augs.StandardLabelToBoundary(append_label=True, foreground=True)
    result = transform(img)

    assert result.shape == (
        3,
        100,
        128,
        128,
    )


def test_standard_label_to_boundary_invalid_dimension():
    img = np.random.randint(0, 2, (10, 10))  # 2D image
    transform = augs.StandardLabelToBoundary()

    with pytest.raises(AssertionError, match="Only 3D labels supported"):
        transform(img)


def test_standardize_2d_no_channelwise(raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    standard = augs.Standardize()
    result = standard(img)

    expected_mean = np.mean(result)
    expected_std = np.std(result)

    assert np.isclose(expected_mean, 0.0, atol=1e-6)
    assert np.isclose(expected_std, 1.0, atol=1e-6)


def test_standardize_2d_with_mean_std(raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    mean, std = 0.5, 2.0
    standard = augs.Standardize(mean=mean, std=std)
    result = standard(img)

    expected = (img - mean) / std
    np.testing.assert_array_almost_equal(result, expected)


def test_standardize_2d_channelwise(raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    std = augs.Standardize(channelwise=True)
    result = std(img)

    expected_mean = np.mean(result)
    expected_std = np.std(result)

    assert np.isclose(expected_mean, 0.0, atol=1e-6)
    assert np.isclose(expected_std, 1.0, atol=1e-6)


def test_standardize_3d_channelwise(raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    std = augs.Standardize(channelwise=True)
    result = std(img)

    axes = (1, 2)  # spatial axes
    mean = np.mean(img, axis=axes, keepdims=True)
    std_val = np.std(img, axis=axes, keepdims=True)
    expected = (img - mean) / np.clip(std_val, a_min=1e-10, a_max=None)

    np.testing.assert_array_almost_equal(result, expected)


def test_standardize_4d_channelwise(raw_zcyx_96x2x96x96):
    img = raw_zcyx_96x2x96x96
    std = augs.Standardize(channelwise=True)
    result = std(img)

    axes = (1, 2, 3)  # expects czyx
    mean = np.mean(img, axis=axes, keepdims=True)
    std_val = np.std(img, axis=axes, keepdims=True)
    expected = (img - mean) / np.clip(std_val, a_min=1e-10, a_max=None)

    np.testing.assert_array_almost_equal(result, expected)


def test_standardize_assertion_error():
    with pytest.raises(AssertionError):
        augs.Standardize(mean=0.5)

    with pytest.raises(AssertionError):
        augs.Standardize(std=0.5)


def test_percentile_normalizer_2d(raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    normalizer = augs.PercentileNormalizer(pmin=1, pmax=99.6)

    result = normalizer(img)

    assert result.shape == img.shape

    assert result.max() > 1
    assert result.max() < 1.1
    assert result.min() < 0
    assert result.min() > -0.1

    # Check that original image is not modified
    assert np.array_equal(img, raw_cell_2d_96x96)


def test_percentile_normalizer_3d(raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    normalizer = augs.PercentileNormalizer(pmin=1, pmax=99.6)

    result = normalizer(img)

    assert result.shape == img.shape

    assert result.max() > 1
    assert result.max() < 1.1
    assert result.min() < 0
    assert result.min() > -0.1


def test_percentile_normalizer_channelwise(raw_zcyx_96x2x96x96):
    img = raw_zcyx_96x2x96x96
    normalizer = augs.PercentileNormalizer(pmin=1, pmax=99.999, channelwise=True)

    result = normalizer(img)

    assert result.shape == img.shape
    assert result.max() > 1
    assert result.max() < 1.1
    assert result.min() < 0
    assert result.min() > -0.1

    for c in range(img.shape[0]):
        channel_data = img[c]
        pmin = np.percentile(channel_data, 1)
        pmax = np.percentile(channel_data, 99.6)
        expected_min = (channel_data.min() - pmin) / (pmax - pmin + 1e-10)
        expected_max = (channel_data.max() - pmin) / (pmax - pmin + 1e-10)
        assert result[c].min() >= expected_min - 1e-6
        assert result[c].max() <= expected_max + 1e-6


def test_percentile_normalizer_extreme_percentiles(raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    normalizer = augs.PercentileNormalizer(pmin=0, pmax=100)

    result = normalizer(img)

    assert result.min() >= 0
    assert result.max() <= 1


def test_normalize_2d(raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    norm = augs.Normalize(0, 255)
    img[0, 0] = 0
    img[0, 1] = 255

    result = norm(img)

    assert np.all(result >= -1) and np.all(result <= 1)

    expected_min = np.clip(2 * (0 - 0) / 255 - 1, -1, 1)
    expected_max = np.clip(2 * (255 - 0) / 255 - 1, -1, 1)
    assert result.min() == expected_min
    assert result.max() == expected_max


def test_normalize_3d(raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    norm = augs.Normalize(0, 1000)
    img[0, 0] = 0
    img[0, 1] = 1000
    result = norm(img)

    assert np.all(result >= -1) and np.all(result <= 1)

    expected_min = np.clip(2 * (0 - 0) / 1000 - 1, -1, 1)
    expected_max = np.clip(2 * (1000 - 0) / 1000 - 1, -1, 1)
    assert result.min() == expected_min
    assert result.max() == expected_max


def test_normalize_4d(raw_zcyx_96x2x96x96):
    img = raw_zcyx_96x2x96x96
    norm = augs.Normalize(0, 1)
    img[0, 0] = 0
    img[0, 1] = 1
    result = norm(img)

    assert np.all(result >= -1) and np.all(result <= 1)

    expected_min = np.clip(2 * (0 - 0) / 1 - 1, -1, 1)
    expected_max = np.clip(2 * (1 - 0) / 1 - 1, -1, 1)
    assert result.min() == expected_min
    assert result.max() == expected_max


def test_normalize_invalid_range():
    with pytest.raises(AssertionError):
        augs.Normalize(10, 5)


def test_additive_gaussian_noise_no_noise(mocker, raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    mock_random = mocker.Mock()
    mock_random.uniform.return_value = 0.2  # > execution_probability
    agn = augs.AdditiveGaussianNoise(mock_random, execution_probability=0.1)

    np.testing.assert_equal(img, agn(img))
    mock_random.uniform.assert_called_once()


def test_additive_gaussian_noise_with_noise(mocker, raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    mock_random = mocker.Mock()
    mock_random.uniform.side_effect = [
        0.05,
        0.5,
    ]  # First call < prob, second call for std
    mock_random.normal.side_effect = np.random.RandomState().normal
    agn = augs.AdditiveGaussianNoise(
        mock_random, execution_probability=0.1, scale=(0.1, 0.2)
    )

    result = agn(img)
    assert result.shape == img.shape
    assert not np.array_equal(img, result)
    assert mock_random.uniform.call_count == 2
    mock_random.normal.assert_called_once()


def test_additive_gaussian_noise_different_shapes(
    mocker, raw_cell_3d_100x128x128, raw_zcyx_96x2x96x96
):
    for img in [raw_cell_3d_100x128x128, raw_zcyx_96x2x96x96]:
        mock_random = mocker.Mock()
        mock_random.uniform.side_effect = [0.05, 0.5]
        mock_random.normal.side_effect = np.random.RandomState().normal
        agn = augs.AdditiveGaussianNoise(mock_random, execution_probability=0.1)

        result = agn(img)
        assert result.shape == img.shape
        assert not np.array_equal(img, result)


def test_to_tensor_2d_no_expand(raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    tt = augs.ToTensor(expand_dims=False)
    with pytest.raises(AssertionError):
        tt(img)


def test_to_tensor_4d_no_expand(raw_zcyx_96x2x96x96):
    img = raw_zcyx_96x2x96x96
    tt = augs.ToTensor(expand_dims=False)

    result = tt(img)
    expected = torch.from_numpy(img.astype(dtype=np.float32))

    assert torch.equal(result, expected)
    assert result.shape == (96, 2, 96, 96)


def test_to_tensor_3d_expand(raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    tt = augs.ToTensor(expand_dims=True)

    result = tt(img)
    expected = torch.from_numpy(np.expand_dims(img, axis=0).astype(dtype=np.float32))

    assert torch.equal(result, expected)
    assert result.shape == (1, 100, 128, 128)


def test_to_tensor_invalid_dims():
    img = np.random.rand(5, 10, 10, 10, 10)  # 5D array
    tt = augs.ToTensor(expand_dims=False)

    with pytest.raises(AssertionError):
        tt(img)


def test_relabel_2d(raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    relabel = augs.Relabel()
    result = relabel(img)

    assert result.shape == img.shape
    unique_labels = np.unique(result)
    np.testing.assert_array_equal(unique_labels, np.arange(len(unique_labels)))


def test_relabel_3d(raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    relabel = augs.Relabel()
    result = relabel(img)

    assert result.shape == img.shape
    unique_labels = np.unique(result)
    np.testing.assert_array_equal(unique_labels, np.arange(len(unique_labels)))


def test_relabel_append_original(raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    relabel = augs.Relabel(append_original=True)
    result = relabel(img)

    # Check that output has double the channels
    assert result.shape[0] == 2
    assert result.shape[1:] == img.shape
    assert result[1].shape == img.shape
    np.testing.assert_array_equal(result[1], img)


def test_relabel_with_ignore_label(raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    relabel = augs.Relabel(ignore_label=0, append_original=True)
    result = relabel(img)

    assert result.shape[0] == 2
    assert result.shape[1:] == img.shape


def test_relabel_no_cc(raw_cell_2d_96x96):
    img = raw_cell_2d_96x96
    relabel = augs.Relabel(run_cc=False)
    result = relabel(img)

    assert result.shape == img.shape
    unique_labels = np.unique(result)
    np.testing.assert_array_equal(unique_labels, np.arange(len(unique_labels)))


def test_identity():
    assert augs.Identity()("smth") == "smth"


def test_rgb_to_label():
    inp = np.arange(30).reshape((2, 5, 3))
    out = augs.RgbToLabel()(inp)
    assert out.shape == (2, 5)


def test_label_to_tensor():
    inp = np.arange(30).reshape((2, 5, 3))
    out = augs.LabelToTensor()(inp)
    assert isinstance(out, torch.Tensor)


def test_gaussian_blur_3d_execution_probability(mocker, raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    mock_random = mocker.patch("panseg.functionals.training.augs.random")
    mock_random.random.return_value = 0.6  # Above probability threshold

    gb = augs.GaussianBlur3D(execution_probability=0.5)
    result = gb(img)

    assert result is img


def test_gaussian_blur_3d_execution(mocker, raw_cell_3d_100x128x128):
    img = raw_cell_3d_100x128x128
    mock_random = mocker.patch("panseg.functionals.training.augs.random")
    mock_random.random.return_value = 0.3  # Below probability threshold
    mock_random.uniform.return_value = 0.5

    gb = augs.GaussianBlur3D(execution_probability=0.5)
    result = gb(img)

    assert not np.array_equal(result, img)


def test_recover_ignore_index():
    orig = np.arange(9).reshape((3, 3))
    inp = np.zeros_like(orig)
    ignore_idx = 8

    out = augs._recover_ignore_index(inp, orig, ignore_idx)
    assert out is inp
    assert out[2, 2] == 8
    assert np.all(out[:2, :2] == 0)


def test_augmenter():
    augmenter = augs.Augmenter()
    augmenter.raw_transform({"mean": 0.8, "std": 0.5})
    augmenter.label_transform()


def test_get_test_augmentations():
    augs.get_test_augmentations(None)
    augs.get_test_augmentations(np.array((1, 2, 3)))
