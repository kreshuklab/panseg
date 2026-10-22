import logging
from types import SimpleNamespace

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
    [  # The halo is trimmed so the patch keeps SliceBuilder's 64 in Y and X
        (
            (1000, 1000, 1000),
            (100, 100, 100),
            (44, 44, 44),
            ((12, 64, 64), (44, 18, 18)),
        ),
        (
            (1000, 1000, 1000),
            (128, 128, 128),
            (44, 44, 44),
            ((40, 64, 64), (44, 32, 32)),
        ),
        ((1, 3000, 3000), (1, 140, 140), (0, 44, 44), ((1, 64, 64), (0, 38, 38))),
    ],
)
def test_derive_patch_and_halo_shapes_keeps_64_in_yx(
    full_volume_shape, max_patch_shape, min_halo_shape, expected
):
    patch, halo = derive_patch_and_halo_shapes(
        full_volume_shape, max_patch_shape, min_halo_shape
    )

    assert (patch, halo) == expected
    assert np.prod(np.array(patch) + np.array(halo) * 2) <= np.prod(max_patch_shape)


class FakeOomProbe:
    """Stand-in for will_CUDA_OOM: OOMs iff the actual forward shape exceeds `threshold` voxels."""

    def __init__(self):
        self.threshold = None  # None -> never OOM
        self.verified_shapes = []

    def __call__(self, model, in_channels, patch_shape, patch_halo, batch_size, device):
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
        lambda model, in_channels, device: PROBED_MAX,
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
    assert len(fake_oom_probe.verified_shapes) == 3


def test_find_feasible_patch_and_halo_shapes_skips_verification_on_cpu(
    fake_oom_probe,
):
    result = find_feasible_patch_and_halo_shapes(
        nn.Module(), 1, VOL, HALO, device="cpu"
    )

    assert result == derive_patch_and_halo_shapes(VOL, PROBED_MAX, HALO)
    assert fake_oom_probe.verified_shapes == []


class _FakeDeviceTensor:
    """Stand-in for a `torch.randn` result: carries the requested shape and
    absorbs `.to(device)` without touching any real device."""

    def __init__(self, shape):
        self.shape = tuple(shape)

    def to(self, *args, **kwargs):
        return self


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
    return None


class FakeUNet3D(UNet3D):
    """UNet3D double for the OOM unit tests: `forward` raises a fake CUDA OOM (or
    a non-OOM error) once the input exceeds a configured voxel count, and `to`
    never moves the model to a real device. No forward pass ever runs."""

    def __init__(
        self, oom_voxel_threshold=None, unexpected_error_above=None, failure=None
    ):
        super().__init__(1, 1, f_maps=8, num_levels=2)
        self.oom_voxel_threshold = oom_voxel_threshold
        self.unexpected_error_above = unexpected_error_above
        self.failure = failure
        self.seen_input_shapes = []

    def to(self, *args, **kwargs):
        return self

    def forward(self, x):
        self.seen_input_shapes.append(tuple(x.shape))
        if self.failure is not None:
            raise RuntimeError(self.failure)
        return _raise_fake_device_error(self, x)


class FakeUNet2D(UNet2D):
    """2D variant of `FakeUNet3D` so that `_is_2d_model` routes to the 2D code paths."""

    def __init__(
        self, oom_voxel_threshold=None, unexpected_error_above=None, failure=None
    ):
        super().__init__(1, 1, f_maps=8, num_levels=2)
        self.oom_voxel_threshold = oom_voxel_threshold
        self.unexpected_error_above = unexpected_error_above
        self.failure = failure
        self.seen_input_shapes = []

    def to(self, *args, **kwargs):
        return self

    def forward(self, x):
        self.seen_input_shapes.append(tuple(x.shape))
        if self.failure is not None:
            raise RuntimeError(self.failure)
        return _raise_fake_device_error(self, x)


@pytest.fixture()
def fake_cuda(monkeypatch):
    """Route the torch calls the GPU probes make away from any real device, so
    the search loops run deterministically without a GPU."""
    monkeypatch.setattr(
        torch, "randn", lambda shape, *args, **kwargs: _FakeDeviceTensor(shape)
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_probe_max_patch_shape_3d_binary_search(fake_cuda):
    # n=4 (64^3 voxels) fits, n=5 (80^3) OOMs
    model = FakeUNet3D(oom_voxel_threshold=300_000)

    assert probe_max_patch_shape(model, 1, "cuda:0") == (64, 64, 64)


def test_probe_max_patch_shape_3d_caps_at_50(fake_cuda):
    assert probe_max_patch_shape(FakeUNet3D(), 1, "cuda:0") == (800, 800, 800)


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


def test_probe_max_patch_shape_3d_reraises_buffer_size_error_on_cuda(fake_cuda):
    model = FakeUNet3D(failure="Invalid buffer size: 4.29 GiB")

    with pytest.raises(RuntimeError, match="Invalid buffer size"):
        probe_max_patch_shape(model, 1, "cuda:0")


@pytest.mark.parametrize(
    ("message", "device", "expected"),
    [
        ("CUDA out of memory. Tried to allocate 2.00 GiB.", "cuda:0", True),
        ("The device has run out of memory.", "mps", True),
        ("Invalid buffer size: 4.29 GiB", "mps", True),
        ("Invalid buffer size: 4.29 GiB", "mps:0", True),
        ("Invalid buffer size: 4.29 GiB", "cuda:0", False),
        ("MPS backend error: illegal instruction", "mps", False),
        ("boom", "mps", False),
    ],
)
def test_is_oom_error(message, device, expected):
    assert size_finder._is_oom_error(RuntimeError(message), device) is expected


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
    # (16, 16) = 256 voxels per sample: batch 64 (16384) fits, batch 128 OOMs
    model = FakeUNet2D(oom_voxel_threshold=20_000)

    assert find_batch_size(model, 1, (1, 16, 16), (0, 0, 0), "cuda:0") == 64
    assert all(len(shape) == 4 for shape in model.seen_input_shapes)


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


class FakeCpuMemory:
    """Stand-in for the CPU memory readings: fixed free memory and memory per voxel."""

    def __init__(self):
        self.available = 64e9
        self.bytes_per_voxel = 10_000
        self.profiled = True  # False: the profiler's allocation record is missing
        self.probed_shapes = []

    def profiled_bytes_per_voxel(self, model, in_channels, shape):
        self.probed_shapes.append(tuple(shape))
        return self.bytes_per_voxel if self.profiled else None

    def resident_growth(self, model, in_channels, shape):
        self.probed_shapes.append(tuple(shape))
        return int(self.bytes_per_voxel * np.prod(shape))


@pytest.fixture()
def fake_cpu_memory(monkeypatch):
    fake = FakeCpuMemory()
    monkeypatch.setattr(size_finder, "_available_memory", lambda: fake.available)
    monkeypatch.setattr(
        size_finder, "_forward_bytes_per_voxel", fake.profiled_bytes_per_voxel
    )
    monkeypatch.setattr(size_finder, "_forward_memory", fake.resident_growth)
    return fake


@pytest.mark.parametrize(
    ("model_cls", "available", "bytes_per_voxel", "expected"),
    [
        (FakeUNet3D, 64e9, 10_000, (144, 144, 144)),
        (FakeUNet3D, 4e9, 10_000, (64, 64, 64)),  # floored at the smallest side
        (FakeUNet3D, 1e12, 1_000, (256, 256, 256)),  # the former fixed shape caps it
        (FakeUNet2D, 64e9, 10_000, (1, 1024, 1024)),
        (FakeUNet2D, 8e9, 20_000, (1, 432, 432)),
    ],
)
def test_probe_max_patch_shape_cpu_follows_free_memory(
    fake_cpu_memory, model_cls, available, bytes_per_voxel, expected
):
    fake_cpu_memory.available = available
    fake_cpu_memory.bytes_per_voxel = bytes_per_voxel
    model = model_cls(failure="forward must not run")

    assert probe_max_patch_shape(model, 1, "cpu") == expected
    nn_dim = 2 if model_cls is FakeUNet2D else 3
    assert fake_cpu_memory.probed_shapes == [size_finder.CPU_PROBE_SHAPES[nn_dim]]
    assert model.seen_input_shapes == []


def test_probe_max_patch_shape_cpu_warns_when_memory_is_short(fake_cpu_memory, caplog):
    fake_cpu_memory.available = 1e9

    with caplog.at_level(logging.WARNING):
        assert probe_max_patch_shape(FakeUNet3D(), 1, "cpu") == (64, 64, 64)
    assert "may not fit" in caplog.text


def test_probe_max_patch_shape_cpu_falls_back_to_resident_memory(
    fake_cpu_memory, caplog
):
    fake_cpu_memory.profiled = False

    with caplog.at_level(logging.WARNING):
        assert probe_max_patch_shape(FakeUNet3D(), 1, "cpu") == (144, 144, 144)
    assert "allocation record is unavailable" in caplog.text
    assert fake_cpu_memory.probed_shapes == [(16, 16, 16), (16, 16, 16)]


def test_probe_max_patch_shape_cpu_floors_hidden_resident_memory(fake_cpu_memory):
    # Memory freed by an earlier pass can be reused, so the resident growth may be 0
    fake_cpu_memory.profiled = False
    fake_cpu_memory.available = 4e9
    fake_cpu_memory.bytes_per_voxel = 0

    assert probe_max_patch_shape(FakeUNet3D(), 1, "cpu") == (112, 112, 112)


def test_forward_bytes_per_voxel_grows_exactly_with_voxels():
    model = UNet3D(1, 1, f_maps=16, num_levels=2).eval()

    small = size_finder._forward_bytes_per_voxel(model, 1, (16, 16, 16))
    large = size_finder._forward_bytes_per_voxel(model, 1, (16, 32, 32))

    assert small > 0
    assert large == pytest.approx(small, rel=0.01)


class _ProfileWithoutRecord:
    """A profiler whose allocation record is missing, as in another PyTorch version."""

    profiler = SimpleNamespace()

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _ProfileAlreadyRunning(_ProfileWithoutRecord):
    def __enter__(self):
        raise RuntimeError("Can't disable Kineto profiler when it's not running")


@pytest.mark.parametrize("profile", [_ProfileWithoutRecord, _ProfileAlreadyRunning])
def test_forward_bytes_per_voxel_none_without_the_record(monkeypatch, profile):
    monkeypatch.setattr(size_finder, "profile", profile)
    model = UNet3D(1, 1, f_maps=16, num_levels=2).eval()

    assert size_finder._forward_bytes_per_voxel(model, 1, (16, 16, 16)) is None


def test_forward_memory_measures_a_real_pass():
    model = UNet3D(1, 1, f_maps=16, num_levels=2).eval()

    assert size_finder._forward_memory(model, 1, (32, 64, 64)) > 0


def test_available_memory_takes_the_cgroup_limit(monkeypatch):
    monkeypatch.setattr(
        size_finder.psutil, "virtual_memory", lambda: SimpleNamespace(available=32e9)
    )
    monkeypatch.setattr(size_finder, "_cgroup_headroom", lambda: 6e9)
    assert size_finder._available_memory() == 6e9

    monkeypatch.setattr(size_finder, "_cgroup_headroom", lambda: None)
    assert size_finder._available_memory() == 32e9


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_headroom_v2_takes_the_tightest_ancestor(tmp_path):
    root = tmp_path / "cgroup"
    _write(root / "slurm/job_1/memory.max", "8000000000\n")
    _write(root / "slurm/job_1/memory.current", "2000000000\n")
    _write(root / "slurm/job_1/step_0/memory.max", "max\n")
    _write(root / "slurm/job_1/step_0/memory.current", "1000000000\n")
    proc_cgroup = tmp_path / "proc_cgroup"
    proc_cgroup.write_text("0::/slurm/job_1/step_0\n")

    assert size_finder._cgroup_headroom(proc_cgroup, root) == 6_000_000_000


def test_cgroup_headroom_v1(tmp_path):
    root = tmp_path / "cgroup"
    group = root / "memory/slurm/uid_1/job_2"
    _write(group / "memory.limit_in_bytes", "4000000000\n")
    _write(group / "memory.usage_in_bytes", "1000000000\n")
    proc_cgroup = tmp_path / "proc_cgroup"
    proc_cgroup.write_text(
        "12:memory:/slurm/uid_1/job_2\n3:cpu,cpuacct:/slurm/uid_1/job_2\n"
    )

    assert size_finder._cgroup_headroom(proc_cgroup, root) == 3_000_000_000


def test_cgroup_headroom_none_without_a_limit(tmp_path):
    root = tmp_path / "cgroup"
    _write(root / "memory/memory.limit_in_bytes", "9223372036854771712\n")
    _write(root / "memory/memory.usage_in_bytes", "1000\n")
    proc_cgroup = tmp_path / "proc_cgroup"
    proc_cgroup.write_text("12:memory:/\n")

    assert size_finder._cgroup_headroom(proc_cgroup, root) is None
    assert size_finder._cgroup_headroom(tmp_path / "missing", root) is None
