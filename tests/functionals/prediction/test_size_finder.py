import numpy as np
import pytest
import torch
from torch import nn

from panseg.functionals.prediction.utils import size_finder
from panseg.functionals.prediction.utils.size_finder import (
    derive_patch_and_halo_shapes,
    find_batch_size,
    find_feasible_patch_and_halo_shapes,
    probe_max_patch_shape,
    will_CUDA_OOM,
)
from panseg.functionals.training.model import UNet2D, UNet3D

# Fixed scenario for the feasibility-loop unit tests (no GPU required, will_CUDA_OOM is faked)
VOL = (48, 1536, 1536)
HALO = (0, 44, 44)
PROBED_MAX = (256, 256, 256)


@pytest.mark.parametrize(
    "full_volume_shape, max_patch_shape, min_halo_shape, expected",
    [  # Here halo is 1-sided
        (
            (1, 12000, 50000),
            (192, 192, 192),
            (44, 44, 44),
            ((1, 2572, 2572), (0, 44, 44)),
        ),
        ((1, 120, 500), (192, 192, 192), (44, 44, 44), ((1, 120, 500), (0, 0, 0))),
        ((95, 120, 500), (192, 192, 192), (44, 44, 44), ((95, 120, 500), (0, 0, 0))),
        ((95, 120, 5000), (192, 192, 192), (44, 44, 44), ((95, 120, 532), (0, 0, 44))),
        ((5000, 120, 95), (192, 192, 192), (44, 44, 44), ((532, 120, 95), (44, 0, 0))),
        (
            (100, 1000, 1000),
            (192, 192, 192),
            (44, 44, 44),
            ((100, 178, 178), (0, 44, 44)),
        ),
        (
            (1000, 1000, 1000),
            (192, 192, 192),
            (44, 44, 44),
            ((104, 104, 104), (44, 44, 44)),
        ),
        (
            (1000, 1000, 1000),
            (192, 192, 192),
            (100, 100, 100),
            ((192, 192, 192), (0, 0, 0)),
        ),
        ((16, 8, 8), (10, 10, 10), (4, 4, 4), ((7, 8, 8), (4, 0, 0))),
        ((16, 10, 10), (10, 10, 10), (4, 4, 4), ((2, 2, 2), (4, 4, 4))),
        ((12, 12, 8), (10, 10, 10), (4, 4, 4), ((3, 3, 8), (4, 4, 0))),
        # Anisotropic volume (issue #146): the sqrt redistribution of the voxel budget
        # used to push a remaining dimension past the sample size, which then crashed
        # SliceBuilder. The derived patch must never exceed the sample in any dimension.
        (
            (100, 300, 5000),
            (256, 256, 256),
            (0, 0, 0),
            ((100, 300, 409), (0, 0, 0)),
        ),
        (
            (100, 300, 5000),
            (256, 256, 256),
            (44, 44, 44),
            ((100, 212, 321), (0, 44, 44)),
        ),
    ],
)
def test_derive_patch_and_halo_shapes(
    full_volume_shape, max_patch_shape, min_halo_shape, expected
):
    result = derive_patch_and_halo_shapes(
        full_volume_shape, max_patch_shape, min_halo_shape
    )
    assert result == expected

    double_halo_out = np.array(result[1]) * 2
    assert np.prod(result[0] + double_halo_out) <= np.prod(max_patch_shape)

    double_halo_shape = (
        min_halo_shape[0] * 2,
        min_halo_shape[1] * 2,
        min_halo_shape[2] * 2,
    )
    result = derive_patch_and_halo_shapes(
        full_volume_shape, max_patch_shape, double_halo_shape, both_sides=True
    )
    assert result == expected


@pytest.mark.parametrize(
    "full_volume_shape, max_patch_shape, min_halo_shape, expected",
    [  # 2D models (issue #612): patch depth 1 and no Z halo, the budget goes to Y and X
        ((20, 512, 512), (1, 3200, 3200), (0, 44, 44), ((1, 512, 512), (0, 0, 0))),
        ((50, 1024, 1024), (1, 3200, 3200), (0, 44, 44), ((1, 1024, 1024), (0, 0, 0))),
        ((50, 1024, 1024), (1, 1024, 1024), (0, 44, 44), ((1, 1024, 1024), (0, 0, 0))),
        (
            (50, 5000, 5000),
            (1, 3200, 3200),
            (0, 44, 44),
            ((1, 3112, 3112), (0, 44, 44)),
        ),
        ((50, 5000, 300), (1, 3200, 3200), (0, 44, 44), ((1, 5000, 300), (0, 0, 0))),
        ((1, 3000, 800), (1, 1024, 1024), (0, 44, 44), ((1, 1222, 800), (0, 44, 0))),
        (
            (1, 20000, 1000),
            (1, 3200, 3200),
            (0, 44, 44),
            ((1, 10152, 1000), (0, 44, 0)),
        ),
        ((20, 512, 512), (1, 3200, 3200), (4, 44, 44), ((1, 512, 512), (0, 0, 0))),
    ],
)
def test_derive_patch_and_halo_shapes_2d(
    full_volume_shape, max_patch_shape, min_halo_shape, expected
):
    patch, halo = derive_patch_and_halo_shapes(
        full_volume_shape, max_patch_shape, min_halo_shape, is_2d=True
    )
    assert (patch, halo) == expected
    assert patch[0] == 1 and halo[0] == 0
    assert all(p <= s for p, s in zip(patch[1:], full_volume_shape[1:]))
    assert np.prod(np.array(patch) + np.array(halo) * 2) <= np.prod(max_patch_shape)

    double_halo_shape = tuple(h * 2 for h in min_halo_shape)
    result = derive_patch_and_halo_shapes(
        full_volume_shape,
        max_patch_shape,
        double_halo_shape,
        both_sides=True,
        is_2d=True,
    )
    assert result == expected


class FakeOomProbe:
    """Stand-in for will_CUDA_OOM: OOMs iff the actual forward shape exceeds `threshold` voxels."""

    def __init__(self):
        self.threshold = None  # None -> never OOM
        self.verified_shapes = []

    def __call__(
        self,
        model,
        in_channels,
        patch_shape,
        patch_halo,
        batch_size,
        device,
        amp="auto",
        channels_last_3d=False,
    ):
        actual = tuple(patch_shape[i] + 2 * patch_halo[i] for i in range(3))
        self.verified_shapes.append(actual)
        if self.threshold is None:
            return False
        return int(np.prod(actual)) > self.threshold


@pytest.fixture()
def fake_oom_probe(monkeypatch):
    """Deterministic stand-ins for the GPU-probing pieces of the feasibility loop."""
    probe = FakeOomProbe()
    monkeypatch.setattr(size_finder, "will_CUDA_OOM", probe)
    monkeypatch.setattr(
        size_finder,
        "probe_max_patch_shape",
        lambda model, in_channels, device, amp="auto", **kwargs: PROBED_MAX,
    )
    return probe


def test_find_feasible_patch_and_halo_shapes_drives_oom_probe(fake_oom_probe):
    expected = derive_patch_and_halo_shapes(VOL, PROBED_MAX, HALO)
    result = find_feasible_patch_and_halo_shapes(
        nn.Module(), 1, VOL, HALO, device="cuda"
    )
    assert result == expected
    assert len(fake_oom_probe.verified_shapes) == 1


def test_find_feasible_patch_and_halo_shapes_shrinks_until_verified(
    fake_oom_probe,
):
    fake_oom_probe.threshold = (
        15_000_000  # first attempt (16.8M voxels) OOMs, the shrunken one fits
    )

    result = find_feasible_patch_and_halo_shapes(
        nn.Module(), 1, VOL, HALO, device="cuda"
    )

    result_voxels = int(np.prod([result[0][i] + 2 * result[1][i] for i in range(3)]))
    assert result_voxels <= fake_oom_probe.threshold
    assert result_voxels < int(np.prod(fake_oom_probe.verified_shapes[0]))
    assert len(fake_oom_probe.verified_shapes) == 2


def test_find_feasible_patch_and_halo_shapes_raises_when_nothing_fits(
    fake_oom_probe,
):
    fake_oom_probe.threshold = 1  # every attempt OOMs

    with pytest.raises(
        RuntimeError, match="Could not determine a feasible patch shape"
    ):
        find_feasible_patch_and_halo_shapes(
            nn.Module(), 1, VOL, HALO, device="cuda", max_attempts=3
        )
    # short-circuit after the first failed check of each attempt
    assert len(fake_oom_probe.verified_shapes) == 3


def test_find_feasible_patch_and_halo_shapes_skips_verification_on_cpu(
    fake_oom_probe,
):
    result = find_feasible_patch_and_halo_shapes(
        nn.Module(), 1, VOL, HALO, device="cpu"
    )

    assert result == derive_patch_and_halo_shapes(VOL, PROBED_MAX, HALO)
    assert fake_oom_probe.verified_shapes == []


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_find_feasible_patch_and_halo_shapes_2d_model_keeps_depth_1(
    fake_oom_probe, monkeypatch, device
):
    monkeypatch.setattr(
        size_finder,
        "probe_max_patch_shape",
        lambda model, in_channels, device, amp="auto", **kwargs: (1, 3200, 3200),
    )

    result = find_feasible_patch_and_halo_shapes(FakeUNet2D(), 1, VOL, HALO, device)

    assert result == ((1, 1536, 1536), (0, 0, 0))  # used to be Z = 4
    if device == "cuda":
        assert fake_oom_probe.verified_shapes == [(1, 1536, 1536)]


def test_find_feasible_patch_and_halo_shapes_2d_model_shrinks_yx_only(
    fake_oom_probe, monkeypatch
):
    monkeypatch.setattr(
        size_finder,
        "probe_max_patch_shape",
        lambda model, in_channels, device, amp="auto", **kwargs: (1, 3200, 3200),
    )
    fake_oom_probe.threshold = 9_000_000  # the first attempt (10.24M voxels) OOMs

    result = find_feasible_patch_and_halo_shapes(
        FakeUNet2D(), 1, (48, 4000, 4000), HALO, device="cuda"
    )

    assert result == ((1, 2819, 2819), (0, 44, 44))
    assert fake_oom_probe.verified_shapes == [(1, 3200, 3200), (1, 2907, 2907)]


class _FakeDeviceTensor:
    """Stand-in for a `torch.randn` result: carries the requested shape and
    absorbs `.to(device)` without touching any real device."""

    def __init__(self, shape):
        self.shape = tuple(shape)

    def to(self, *args, **kwargs):
        return self


class _FakeOutput:
    """Stand-in for a model output: supports the `.sum().backward()` the
    training-mode batch-size probe runs."""

    def __init__(self, model, voxels):
        self.model = model
        self.voxels = voxels

    def sum(self):
        return self

    def backward(self):
        self.model.backward_calls += 1
        threshold = self.model.backward_oom_threshold
        if threshold is not None and self.voxels > threshold:
            raise RuntimeError("CUDA out of memory. Tried to allocate 999.00 GiB.")


def _raise_fake_device_error(model, x):
    # in_channels is 1 in all these tests, so the total input size is the voxel count
    voxels = int(np.prod(x.shape))
    if (
        model.unexpected_error_above is not None
        and voxels > model.unexpected_error_above
    ):
        raise RuntimeError("unexpected error, not an OOM")
    if model.oom_voxel_threshold is not None and voxels > model.oom_voxel_threshold:
        raise RuntimeError("CUDA out of memory. Tried to allocate 999.00 GiB.")
    return _FakeOutput(model, voxels)


class _FakeUNetProbeMixin:
    """Shared probe double for the OOM unit tests: `forward` raises a fake CUDA
    OOM (or a non-OOM error) once the input exceeds a configured voxel count,
    and `to` never moves the model to a real device. No real forward pass ever
    runs.

    Both fakes keep subclassing `UNet3D`/`UNet2D` so `is_2d_model` routes to the
    matching code path.
    """

    def __init__(
        self,
        oom_voxel_threshold=None,
        unexpected_error_above=None,
        failure=None,
        backward_oom_threshold=None,
    ):
        super().__init__(1, 1, f_maps=8, num_levels=2)
        self.oom_voxel_threshold = oom_voxel_threshold
        self.unexpected_error_above = unexpected_error_above
        self.failure = failure
        self.backward_oom_threshold = backward_oom_threshold
        self.backward_calls = 0
        self.seen_input_shapes = []

    def to(self, *args, **kwargs):
        return self

    def forward(self, x):
        self.seen_input_shapes.append(tuple(x.shape))
        if self.failure is not None:
            raise RuntimeError(self.failure)
        return _raise_fake_device_error(self, x)


class FakeUNet3D(_FakeUNetProbeMixin, UNet3D):
    """3D probe double (see `_FakeUNetProbeMixin`)."""


class FakeUNet2D(_FakeUNetProbeMixin, UNet2D):
    """2D variant of `FakeUNet3D` so that `is_2d_model` routes to the 2D code paths."""


def _record_cudnn_benchmark(model: nn.Module) -> list[bool]:
    """Record `torch.backends.cudnn.benchmark` inside every forward pass."""
    seen: list[bool] = []

    def hook(module, inputs, output):
        seen.append(torch.backends.cudnn.benchmark)

    model.register_forward_hook(hook)
    return seen


@pytest.fixture()
def fake_cuda(monkeypatch):
    """Route the torch calls the GPU probes make away from any real device, so
    the search loops run deterministically without a GPU."""
    monkeypatch.setattr(
        torch, "randn", lambda shape, *args, **kwargs: _FakeDeviceTensor(shape)
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


@pytest.mark.parametrize(
    ("oom_voxel_threshold", "expected"),
    [
        (150_000, (48, 48, 48)),  # n=3: below the boundary, no headroom
        (300_000, (64, 64, 64)),  # n=4: below the boundary, no headroom
        (2_500_000, (128, 128, 128)),  # n=8: below the boundary, no headroom
        # n=17 probes 272^3 successfully; one-step headroom -> 256^3 (n=16)
        (22_000_000, (256, 256, 256)),
    ],
)
def test_probe_max_patch_shape_3d_binary_search(
    fake_cuda, oom_voxel_threshold, expected
):
    model = FakeUNet3D(oom_voxel_threshold=oom_voxel_threshold)

    assert probe_max_patch_shape(model, 1, "cuda:0") == expected


def test_probe_max_patch_shape_3d_caps_at_50_with_one_step_headroom(fake_cuda):
    assert probe_max_patch_shape(FakeUNet3D(), 1, "cuda:0") == (784, 784, 784)


def test_probe_max_patch_shape_3d_floors_at_low_on_persistent_oom(fake_cuda):
    model = FakeUNet3D(oom_voxel_threshold=0)  # every candidate OOMs

    assert probe_max_patch_shape(model, 1, "cuda:0") == (32, 32, 32)


def test_probe_max_patch_shape_3d_reraises_non_oom_errors(fake_cuda):
    model = FakeUNet3D(unexpected_error_above=0)  # every candidate fails, non-OOM

    with pytest.raises(RuntimeError, match="unexpected error, not an OOM"):
        probe_max_patch_shape(model, 1, "cuda:0")


def test_probe_max_patch_shape_3d_treats_mps_buffer_size_error_as_oom(fake_cuda):
    # Apple MPS rejects single allocations above its 4 GiB buffer cap with
    # "Invalid buffer size" instead of an "out of memory" message.
    model = FakeUNet3D(failure="Invalid buffer size: 4.29 GiB")

    assert probe_max_patch_shape(model, 1, "mps") == (32, 32, 32)


@pytest.mark.parametrize(
    ("message", "device", "expected"),
    [
        ("CUDA out of memory. Tried to allocate 2.00 GiB.", "cuda:0", True),
        ("The device has run out of memory.", "mps", True),
        ("Invalid buffer size: 4.29 GiB", "mps", True),
        ("Invalid buffer size: 4.29 GiB", "mps:0", True),
        ("Invalid buffer size: 4.29 GiB", "cuda:0", False),
        ("cuDNN error: CUDNN_STATUS_ALLOC_FAILED", "cuda:0", True),
        ("MPS backend error: illegal instruction", "mps", False),
        ("boom", "mps", False),
    ],
)
def test_is_oom_error(message, device, expected):
    assert size_finder.is_oom_error(RuntimeError(message), device) is expected


def test_probe_max_patch_shape_2d_returns_max_on_first_fit(fake_cuda):
    assert probe_max_patch_shape(FakeUNet2D(), 1, "cuda:0") == (1, 3200, 3200)


def test_probe_max_patch_shape_2d_linear_search(fake_cuda):
    # best_n=160 (2560^2 voxels) fits, best_n=180 (2880^2) OOMs
    model = FakeUNet2D(oom_voxel_threshold=7_000_000)

    assert probe_max_patch_shape(model, 1, "cuda:0") == (1, 2560, 2560)


def test_probe_max_patch_shape_2d_floors_on_persistent_oom(fake_cuda):
    # the -20 step overshoots the `best_n > 16` loop bound: best_n ends up at 0
    model = FakeUNet2D(oom_voxel_threshold=0)  # every candidate OOMs

    assert probe_max_patch_shape(model, 1, "cuda:0") == (1, 0, 0)


def test_probe_max_patch_shape_2d_reraises_non_oom_errors(fake_cuda):
    model = FakeUNet2D(failure="boom")

    with pytest.raises(RuntimeError, match="boom"):
        probe_max_patch_shape(model, 1, "cuda:0")


@pytest.mark.parametrize("model_factory", [FakeUNet3D, FakeUNet2D])
def test_probe_max_patch_shape_runs_without_cudnn_autotuning(fake_cuda, model_factory):
    """Inference patch probing must not enable cuDNN autotuning."""
    model = model_factory(oom_voxel_threshold=300_000)
    seen = _record_cudnn_benchmark(model)

    probe_max_patch_shape(model, 1, "cuda:0")

    assert seen, "no probe forward ran"
    assert not any(seen), "the patch probe enabled cuDNN autotuning"


def test_will_cuda_oom_runs_without_cudnn_autotuning(fake_cuda):
    model = FakeUNet3D()
    seen = _record_cudnn_benchmark(model)

    assert will_CUDA_OOM(model, 1, (64, 64, 64), (0, 0, 0), 1, "cuda:0") is False

    assert seen, "no verification forward ran"
    assert not any(seen), "the OOM verification enabled cuDNN autotuning"


@pytest.mark.parametrize(("for_training", "expected"), [(False, False), (True, True)])
def test_probe_batch_autotuning_only_for_training(for_training, expected):
    """The training probe measures the benchmarked training loop; inference
    probes match the (now autotuning-free) prediction loop."""
    model = FakeUNet3D()
    seen = _record_cudnn_benchmark(model)
    x = _FakeDeviceTensor((1, 1, 16, 16, 16))

    size_finder._probe_batch(model, x, for_training, amp="off", device="cuda:0")

    assert seen == [expected]


def test_will_cuda_oom_detects_oom(fake_cuda):
    model = FakeUNet3D(oom_voxel_threshold=0)

    assert will_CUDA_OOM(model, 1, (64, 64, 64), (0, 0, 0), 1, "cuda:0") is True


def test_will_cuda_oom_false_when_input_fits(fake_cuda):
    model = FakeUNet3D()

    assert will_CUDA_OOM(model, 1, (64, 64, 64), (0, 0, 0), 1, "cuda:0") is False


def test_will_cuda_oom_includes_halo_in_input_shape(fake_cuda):
    model = FakeUNet3D(oom_voxel_threshold=64**3 - 1)  # fits only without the halo

    assert will_CUDA_OOM(model, 1, (64, 64, 64), (4, 4, 4), 1, "cuda:0") is True
    assert model.seen_input_shapes == [(1, 1, 72, 72, 72)]


def test_will_cuda_oom_2d_model_slices_first_dim(fake_cuda):
    model = FakeUNet2D()

    assert will_CUDA_OOM(model, 1, (1, 16, 16), (0, 2, 2), 1, "cuda:0") is False
    assert model.seen_input_shapes == [(1, 1, 20, 20)]


def test_will_cuda_oom_reraises_non_oom_errors(fake_cuda):
    model = FakeUNet3D(failure="boom")

    with pytest.raises(RuntimeError, match="boom"):
        will_CUDA_OOM(model, 1, (64, 64, 64), (0, 0, 0), 1, "cuda:0")


def test_find_batch_size_halves_on_oom(fake_cuda):
    # 16^3 = 4096 voxels per sample: batch 4 (16384) fits, batch 8 (32768) OOMs
    model = FakeUNet3D(oom_voxel_threshold=20_000)

    assert find_batch_size(model, 1, (16, 16, 16), (0, 0, 0), "cuda:0") == 4


def test_find_batch_size_halves_on_unexpected_error(fake_cuda):
    model = FakeUNet3D(unexpected_error_above=20_000)

    assert find_batch_size(model, 1, (16, 16, 16), (0, 0, 0), "cuda:0") == 4


def test_find_batch_size_includes_halo_in_input_shape(fake_cuda):
    # with halo (8, 8, 8) a single sample needs 32^3 = 32768 > 20_000 voxels
    model = FakeUNet3D(oom_voxel_threshold=20_000)

    with pytest.raises(RuntimeError, match="Could not determine a feasible batch size"):
        find_batch_size(model, 1, (16, 16, 16), (8, 8, 8), "cuda:0")


def test_find_batch_size_raises_when_batch_one_ooms(fake_cuda):
    model = FakeUNet3D(oom_voxel_threshold=0)

    with pytest.raises(RuntimeError, match="Could not determine a feasible batch size"):
        find_batch_size(model, 1, (16, 16, 16), (0, 0, 0), "cuda:0")


def test_find_batch_size_2d_model_slices_first_dim(fake_cuda):
    # (16, 16) = 256 voxels per sample: 78 samples (19968) fit, 79 (20224) OOMs
    model = FakeUNet2D(oom_voxel_threshold=20_000)

    assert find_batch_size(model, 1, (1, 16, 16), (0, 0, 0), "cuda:0") == 78
    assert all(len(shape) == 4 for shape in model.seen_input_shapes)


def test_find_batch_size_for_training_halves_on_backward_oom(fake_cuda):
    # Forward fits up to batch 4 (16384 voxels), backward only batch 1 (4096).
    # A forward-only probe would overestimate the training batch size.
    model = FakeUNet3D(oom_voxel_threshold=20_000, backward_oom_threshold=5_000)

    assert find_batch_size(model, 1, (16, 16, 16), (0, 0, 0), "cuda:0") == 4
    assert (
        find_batch_size(model, 1, (16, 16, 16), (0, 0, 0), "cuda:0", for_training=True)
        == 1
    )
    # the inference probe never runs backward; the training probe runs it for
    # the fitting batch 1 and for the OOMing batch 2
    assert model.backward_calls == 2


def test_find_batch_size_restores_training_mode(fake_cuda):
    model = FakeUNet3D()
    model.eval()

    find_batch_size(model, 1, (16, 16, 16), (0, 0, 0), "cuda:0", for_training=True)

    assert not model.training


class _BnProbeModel(nn.Module):
    """Probe double whose forward runs a *real* BatchNorm update on a real
    tensor before honoring the OOM oracle: train-mode probes genuinely pollute
    `running_mean`/`running_var`/`num_batches_tracked`, like the real model."""

    def __init__(self, oom_voxel_threshold=None):
        super().__init__()
        self.bn = nn.BatchNorm3d(2)
        self.oom_voxel_threshold = oom_voxel_threshold
        self.seen_input_shapes = []
        self.bn_updates = 0

    def to(self, *args, **kwargs):
        return self

    def forward(self, x):
        self.seen_input_shapes.append(tuple(x.shape))
        voxels = int(np.prod(x.shape))
        # BN update first, then the oracle: even a failing probe may leave the
        # running stats mutated
        t = torch.arange(2 * 4**3, dtype=torch.float32).reshape(1, 2, 4, 4, 4)
        before = int(self.bn.num_batches_tracked)
        out = self.bn(t).sum()
        if int(self.bn.num_batches_tracked) > before:
            self.bn_updates += 1
        if self.oom_voxel_threshold is not None and voxels > self.oom_voxel_threshold:
            raise RuntimeError("CUDA out of memory. Tried to allocate 999.00 GiB.")
        return out


def _bn_buffers(model):
    return {name: buf.clone() for name, buf in model.named_buffers()}


def test_find_batch_size_for_training_restores_bn_stats_bit_exact(fake_cuda):
    """The train-mode probe must not leak BatchNorm running stats into training."""
    model = _BnProbeModel()
    before = _bn_buffers(model)

    find_batch_size(model, 1, (16, 16, 16), (0, 0, 0), "cuda:0", for_training=True)

    assert model.bn_updates > 0, "the probe must actually run train-mode forwards"
    after = _bn_buffers(model)
    assert after.keys() == before.keys()
    for name in before:
        assert torch.equal(after[name], before[name]), name


def test_find_batch_size_restores_bn_stats_when_batch_one_ooms(fake_cuda):
    model = _BnProbeModel(oom_voxel_threshold=0)
    before = _bn_buffers(model)

    with pytest.raises(RuntimeError, match="Could not determine a feasible batch size"):
        find_batch_size(model, 1, (16, 16, 16), (0, 0, 0), "cuda:0", for_training=True)

    assert model.bn_updates > 0
    after = _bn_buffers(model)
    for name in before:
        assert torch.equal(after[name], before[name]), name


def test_find_batch_size_bisects_to_the_true_maximum(fake_cuda):
    # 16^3 = 4096 voxels/sample: 6 samples (24576) fit, 7 (28672) OOMs.
    # The old power-of-2 search settled for 4.
    model = FakeUNet3D(oom_voxel_threshold=25_000)

    assert find_batch_size(model, 1, (16, 16, 16), (0, 0, 0), "cuda:0") == 6


@pytest.mark.parametrize("max_samples", [1, 2, 3, 5, 6, 7, 12, 13, 16, 31])
def test_find_batch_size_result_within_power_of_two_and_true_max(
    fake_cuda, max_samples
):
    """The result is never below the old power-of-2 value and never above the
    true maximum (a fake monotone OOM oracle makes both bounds exact)."""
    model = FakeUNet3D(oom_voxel_threshold=max_samples * 4096)

    result = find_batch_size(model, 1, (16, 16, 16), (0, 0, 0), "cuda:0")

    power_of_two = 2 ** (max_samples.bit_length() - 1)
    assert power_of_two <= result <= max_samples
    assert result == max_samples


def test_find_batch_size_caps_at_2048(fake_cuda):
    model = FakeUNet3D()  # never OOMs

    assert find_batch_size(model, 1, (16, 16, 16), (0, 0, 0), "cuda:0") == 2048


def _patch_cuda_memory(monkeypatch, previous_fraction=1.0):
    """Fake the torch CUDA memory APIs `_quiet_oom_probing` uses and record every
    fraction that gets set, so the tests run deterministically without a GPU."""
    set_calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8, 16))
    monkeypatch.setattr(torch.cuda.memory, "memory_reserved", lambda device: 2)
    monkeypatch.setattr(
        torch.cuda.memory,
        "get_per_process_memory_fraction",
        lambda device: previous_fraction,
    )
    monkeypatch.setattr(
        torch.cuda.memory,
        "set_per_process_memory_fraction",
        lambda fraction, device: set_calls.append(fraction),
    )
    return set_calls


def test_quiet_oom_probing_caps_allocator_and_restores(monkeypatch):
    set_calls = _patch_cuda_memory(monkeypatch)

    with size_finder._quiet_oom_probing("cuda:0"):
        pass

    # cap at footprint (2) + free (8) of 16 total, then restore the previous 1.0
    assert set_calls == [10 / 16, 1.0]


def test_quiet_oom_probing_restores_fraction_on_error(monkeypatch):
    set_calls = _patch_cuda_memory(monkeypatch, previous_fraction=0.7)

    with (
        pytest.raises(RuntimeError, match="boom"),
        size_finder._quiet_oom_probing("cuda:0"),
    ):
        raise RuntimeError("boom")

    assert set_calls == [10 / 16, 0.7]


def test_quiet_oom_probing_noop_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda.memory,
        "set_per_process_memory_fraction",
        lambda *args, **kwargs: pytest.fail("allocator touched without CUDA"),
    )

    with size_finder._quiet_oom_probing("cuda:0"):
        pass
