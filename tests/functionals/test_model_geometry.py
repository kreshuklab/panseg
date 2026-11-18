"""Tests for the shared model-geometry helpers.

`model_geometry` is the single place that derives architecture-dependent input
geometry from a model: the smallest input each axis accepts (`min_input`), the
phase period it steps by (`increment`) and the context each side needs for an
exact output (`halo`). The halo is replayed from the architecture's ordered
layers, never assumed from a closed form, so the reach table is checked against
a real autograd support here.
"""

import pytest
import torch
from torch import nn

from panseg.functionals.model_geometry import (
    ModelGeometry,
    is_2d_model,
    model_geometry,
    model_input_shape,
    stride_shape,
)
from panseg.functionals.model_geometry.layers import ordered_ops
from panseg.functionals.model_geometry.replay import (
    _reach_table_cached,
    exact_halo,
    reach_table,
    replay,
)
from panseg.functionals.training.model import UNet2D, UNet3D

STANDARD_HALO = (48, 48, 48)
STANDARD_EFFECTIVE_MINIMUM = (104, 104, 104)


class TestPackageLayout:
    """Geometry is a package: introspection, replay and the public API.

    `layers` maps model modules to the replay vocabulary, `replay` holds the 1D
    support replay, and the package root keeps the documented geometry API.
    """

    def test_the_package_root_keeps_the_public_geometry_api(self):
        import panseg.functionals.model_geometry as geometry

        for name in (
            "ModelGeometry",
            "model_geometry",
            "aligned_shape",
            "aligned_down",
            "model_input_shape",
            "tensor_input_shape",
            "is_2d_model",
            "stride_shape",
        ):
            assert hasattr(geometry, name), name

    def test_the_replay_and_layers_helpers_live_in_their_own_modules(self):
        model = nn.Sequential(nn.Conv3d(1, 1, 3, padding=1))

        ops = ordered_ops(model)
        support, final_size, _ = replay(ops, axis=0, model_input=8)
        table, _ = reach_table(ops, axis=0, increment=1)

        assert final_size == 8
        assert support[4] == {3, 4, 5}
        assert exact_halo(table, 1) == 2


class TestModelGeometryTraversal:
    def test_is_2d_model_is_decided_by_the_spatial_layers(self):
        model = UNet2D(1, 1, f_maps=[4, 4])

        assert is_2d_model(model)
        # wrappers are transparent: the spatial layers decide
        assert is_2d_model(nn.Sequential(model))
        assert not is_2d_model(UNet3D(1, 1, f_maps=[4, 4]))

    def test_from_model_matches_the_function(self):
        model = UNet3D(in_channels=1, out_channels=1, f_maps=[4, 4, 4, 4])

        assert ModelGeometry.from_model(model) == model_geometry(model)

    def test_unet3d_halo_increment_and_min_input(self):
        model = UNet3D(in_channels=1, out_channels=1, f_maps=[4, 4, 4, 4])

        geometry = model_geometry(model)

        # three 2x pools on every axis; the exact halo is 48, not the classic 44
        assert geometry.min_input == (8, 8, 8)
        assert geometry.increment == (8, 8, 8)
        assert geometry.halo == STANDARD_HALO
        assert not geometry.is_2d

    def test_unet2d_has_no_z_geometry_and_stays_one_slice_deep(self):
        model = UNet2D(in_channels=1, out_channels=1, f_maps=[4, 4, 4, 4])

        geometry = model_geometry(model)

        assert geometry.min_input == (1, 8, 8)
        assert geometry.increment == (1, 8, 8)
        assert geometry.halo == (0, 48, 48)
        assert geometry.is_2d

    @pytest.mark.parametrize(
        "model",
        [
            # architecture kwargs of the efficient-chipmunk RDF
            UNet3D(in_channels=1, out_channels=2, f_maps=16),
            # architecture kwargs of the pioneering-rhino RDF
            UNet2D(
                in_channels=1,
                out_channels=1,
                final_sigmoid=True,
                f_maps=64,
                layer_order="gcr",
                num_groups=8,
                is_segmentation=True,
            ),
        ],
        ids=["efficient-chipmunk", "pioneering-rhino"],
    )
    def test_rdf_architectures_use_the_exact_halo(self, model):
        assert model_geometry(model).halo in {STANDARD_HALO, (0, 48, 48)}

    def test_anisotropic_conv_kernels_contribute_per_axis(self):
        model = nn.Sequential(nn.Conv3d(1, 1, kernel_size=(1, 3, 5), padding=(0, 1, 2)))

        geometry = model_geometry(model)

        # a conv-only axis has a single phase; its halo is the per-axis radius + 1
        assert geometry.increment == (1, 1, 1)
        assert geometry.halo == (0, 2, 3)

    def test_average_pooling_counts_like_max_pooling(self):
        model = nn.Sequential(
            nn.AvgPool3d(2),
            nn.Conv3d(1, 1, 3, padding=1),
            nn.Conv3d(1, 1, 3, padding=1),
        )

        geometry = model_geometry(model)

        # pooling cell 2q feeds 2q, 2q+1; two convs at that level add radius 2
        # per side, so each output needs [2q-4, 2q+5] -> halo 6
        assert geometry.min_input == (2, 2, 2)
        assert geometry.increment == (2, 2, 2)
        assert geometry.halo == (6, 6, 6)

    def test_anisotropic_pooling_sets_the_per_axis_period(self):
        model = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 4, 4)),
            nn.Conv3d(1, 1, 3, padding=1),
        )

        geometry = model_geometry(model)

        assert geometry.min_input == (2, 4, 4)
        assert geometry.increment == (2, 4, 4)

    def test_pool_kernel_and_stride_can_differ(self):
        """A kernel-3, stride-2 pool runs on 3 cells but steps the phase by 2."""
        model = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(1, 1, 3, padding=1),
            nn.Conv3d(1, 1, 3, padding=1),
        )

        geometry = model_geometry(model)

        assert geometry.min_input == (3, 3, 3)
        assert geometry.increment == (2, 2, 2)
        assert geometry.min_input != geometry.increment
        assert geometry.halo == (6, 6, 6)
        # a kernel-3, stride-2 pool leaves a coarser output grid: only the
        # aligned phase exists, and the replay must not invent others
        table, _ = reach_table(ordered_ops(model), axis=0, increment=2)
        assert set(table) == {0}

    def test_one_by_one_convs_get_their_reach_from_pooling(self):
        model = nn.Sequential(
            nn.MaxPool3d(2),
            nn.MaxPool3d(2),
            nn.MaxPool3d(2),
            nn.Conv3d(1, 1, 1),
            nn.Conv3d(1, 1, 1),
        )

        geometry = model_geometry(model)

        # the 1x1 convs add no radius; a pooled output still reads a whole
        # 8-voxel cell, so the halo is one full phase period
        assert geometry.min_input == (8, 8, 8)
        assert geometry.increment == (8, 8, 8)
        assert geometry.halo == (8, 8, 8)

    def test_transposed_convolutions_are_rejected(self):
        model = nn.Sequential(nn.ConvTranspose3d(1, 1, 2, stride=2))

        with pytest.raises(NotImplementedError, match="transposed convolution"):
            model_geometry(model)

    @pytest.mark.parametrize(
        "config",
        [
            {"f_maps": [4, 4]},
            {"f_maps": [4, 4, 4, 4]},
            {"f_maps": 8, "num_levels": 5},
        ],
    )
    @pytest.mark.parametrize("model_class", [UNet3D, UNet2D])
    def test_halo_always_reaches_the_architectural_minimum(self, model_class, config):
        """`halo + 1 >= min_input` on every axis a model pools.

        The automatic path derives patches with the exact halo, so this
        invariant is what keeps every auto-derived shape valid.
        """
        geometry = model_geometry(model_class(1, 1, **config))

        for axis, (halo, minimum) in enumerate(zip(geometry.halo, geometry.min_input)):
            if geometry.is_2d and axis == 0:
                continue
            assert halo + 1 >= minimum


class TestReplaySizeInvariant:
    """The replay's size must equal the size a real forward pass produces."""

    @pytest.mark.parametrize(
        "config",
        [
            {"f_maps": [4, 4]},
            {"f_maps": [4, 4, 4, 4]},
            {"f_maps": 8, "num_levels": 5},
        ],
    )
    # `(axis, dim)` pairs are explicit: a 2D model has no Z tensor axis, so its
    # Y/X replay axes are 1/2 and its output dims 2/3.
    @pytest.mark.parametrize(
        ("model_class", "axis", "dim"),
        [
            (UNet3D, 0, 2),
            (UNet3D, 1, 3),
            (UNet3D, 2, 4),
            (UNet2D, 1, 2),
            (UNet2D, 2, 3),
        ],
        ids=["3d-z", "3d-y", "3d-x", "2d-y", "2d-x"],
    )
    def test_the_replay_matches_the_real_forward_size(
        self, model_class, axis, dim, config
    ):
        model = model_class(1, 1, layer_order="c", **config).eval()
        spatial = [64] * (2 if model_class is UNet2D else 3)

        with torch.no_grad():
            out = model(torch.zeros(1, 1, *spatial))

        assert replay(ordered_ops(model), axis, 64)[1] == out.shape[dim]


class _UnknownSpatialLayer(nn.Module):
    """A layer type the replay has no rule for, but that changes the grid."""

    def __init__(self):
        super().__init__()
        self.kernel_size = 2
        self.stride = 2

    def forward(self, x):  # pragma: no cover - never called
        raise AssertionError("the geometry must be refused before a forward pass")


class _UnknownContainer(nn.Module):
    """A wrapper the replay has no rule for, holding a spatial child."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 2)

    def forward(self, x):  # pragma: no cover - never called
        raise AssertionError("the geometry must be refused before a forward pass")


class TestUnsupportedLayers:
    """The replay refuses layers whose size effect it does not model."""

    def test_conv_padding_other_than_the_radius_is_rejected(self):
        with pytest.raises(NotImplementedError, match="padding"):
            model_geometry(UNet3D(1, 1, f_maps=[4, 4, 4, 4], conv_padding=2))

        with pytest.raises(NotImplementedError, match="padding"):
            model_geometry(nn.Sequential(nn.Conv3d(1, 1, 3, padding=2)))

    def test_even_conv_kernels_are_rejected(self):
        with pytest.raises(NotImplementedError, match="kernel_size"):
            model_geometry(nn.Sequential(nn.Conv3d(1, 1, 2, padding=1)))

    def test_pooling_with_padding_is_rejected(self):
        with pytest.raises(NotImplementedError, match="padding"):
            model_geometry(nn.Sequential(nn.MaxPool3d(2, padding=1)))

    def test_pooling_with_dilation_is_rejected(self):
        with pytest.raises(NotImplementedError, match="dilation"):
            model_geometry(nn.Sequential(nn.MaxPool3d(2, dilation=2)))

    def test_unsupported_resampling_layers_are_rejected(self):
        model = nn.Sequential(
            nn.Conv3d(1, 1, 3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(1, 1, 3, padding=1),
        )

        with pytest.raises(NotImplementedError, match="Upsample"):
            model_geometry(model)

    def test_unrecognised_modules_with_spatial_attributes_are_rejected(self):
        model = nn.Sequential(
            nn.Conv3d(1, 1, 3, padding=1),
            _UnknownSpatialLayer(),
            nn.Conv3d(1, 1, 3, padding=1),
        )

        with pytest.raises(NotImplementedError, match="_UnknownSpatialLayer"):
            model_geometry(model)

    def test_unrecognised_containers_are_rejected_not_traversed_as_identity(self):
        model = nn.Sequential(
            nn.Conv3d(1, 1, 3, padding=1),
            _UnknownContainer(),
            nn.Conv3d(1, 1, 3, padding=1),
        )

        with pytest.raises(NotImplementedError, match="_UnknownContainer"):
            model_geometry(model)

    def test_childless_modules_without_spatial_attributes_stay_transparent(self):
        model = nn.Sequential(nn.Conv3d(1, 1, 3, padding=1), nn.Identity(), nn.ReLU())

        geometry = model_geometry(model)

        assert geometry.halo == (2, 2, 2)
        assert len(ordered_ops(model)) == 1


def _avg_pool_oracle(model: nn.Module, model_input: int, axis: int, positions):
    """Per-position `(left, right)` reach from autograd, on float64 tensors.

    Avg pooling replaces max pooling so the gradient follows every input voxel
    of a pooling cell; max pooling would follow only the argmax and
    under-report the structural support. Returns `{p: (left, right)}`.
    """
    model = model.double()
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.MaxPool2d, nn.MaxPool3d)):
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            pooling = nn.AvgPool2d if isinstance(module, nn.MaxPool2d) else nn.AvgPool3d
            setattr(
                parent,
                parts[-1],
                pooling(module.kernel_size, module.stride or module.kernel_size),
            )

    is_2d = isinstance(model, UNet2D)
    spatial = [model_input] * (2 if is_2d else 3)
    x = torch.zeros([1, 1, *spatial], dtype=torch.float64)
    result = {}
    for p in positions:
        x.grad = None
        x.requires_grad_(True)
        out = model(x)
        index = [0, 0] + [model_input // 2] * len(spatial)
        index[axis + 2 - (1 if is_2d else 0)] = p
        out[tuple(index)].backward()
        nonzero = torch.nonzero(x.grad[0, 0])
        axis_index = axis - 1 if is_2d else axis
        low = int(nonzero[:, axis_index].min())
        high = int(nonzero[:, axis_index].max())
        result[p] = (p - low, high - p)
    return result


class TestReachReplay:
    """The production reach table against a real autograd support."""

    @pytest.mark.parametrize("f_maps", [[2, 2, 2], [2, 2, 2, 2]], ids=["3lvl", "4lvl"])
    @pytest.mark.parametrize("model_class", [UNet3D, UNet2D], ids=["3d", "2d"])
    def test_the_reach_table_matches_the_autograd_support(self, f_maps, model_class):
        axis = 0 if model_class is UNet3D else 1
        increment = 2 ** (len(f_maps) - 1)
        # sample one full period in the middle of the full-resolution grid,
        # far enough from both edges for the deepest reach
        positions = range(64, 64 + increment)

        supports = []
        table = None
        for seed in range(2):
            torch.manual_seed(seed)
            model = model_class(
                in_channels=1,
                out_channels=1,
                f_maps=f_maps,
                layer_order="c",
                final_sigmoid=True,
            )
            table, _ = reach_table(ordered_ops(model), axis, increment)
            supports.append(
                _avg_pool_oracle(model, model_input=128, axis=axis, positions=positions)
            )

        # the union over seeds is the structural support; random weights only
        # affect which of it has non-zero gradient
        for position in positions:
            left = min(support[position][0] for support in supports)
            right = max(support[position][1] for support in supports)
            assert table[position % increment] == (left, right), position

    def test_the_reach_table_matches_an_anisotropic_conv_stack(self):
        model = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 3, 5), padding=(0, 1, 2)),
            nn.Conv3d(1, 1, kernel_size=(3, 1, 5), padding=(1, 0, 2)),
        )
        ops = ordered_ops(model)

        for axis in range(3):
            table, _ = reach_table(ops, axis, increment=1)
            oracle = _avg_pool_oracle(model, model_input=16, axis=axis, positions=[8])
            assert table[0] == oracle[8], axis

    def test_the_derived_halo_is_minimal(self):
        model = UNet3D(1, 1, f_maps=[4, 4, 4, 4])
        table, _ = reach_table(ordered_ops(model), 0, increment=8)

        assert exact_halo(table, 8) == 48
        # one phase period less leaves at least one crop position unclean
        h, q = 40, 8
        m = q + 2 * h
        assert any(
            h + t - table[t][0] <= 0 or h + t + table[t][1] >= m - 1 for t in range(8)
        )


class TestReachTableCache:
    """The replay is pure, so its reach table is cached per op sequence.

    `model_geometry` runs per prediction and on every model change in the
    viewer; the op dataclasses are frozen, so the op tuple is a valid key.
    """

    def test_reach_table_is_cached(self):
        _reach_table_cached.cache_clear()
        ops = ordered_ops(nn.Sequential(nn.Conv3d(1, 1, 3, padding=1)))

        first, _ = reach_table(ops, axis=0, increment=1)
        second, _ = reach_table(ops, axis=0, increment=1)

        assert _reach_table_cached.cache_info().hits >= 1
        assert first == second

        # a caller mutating its dict must not poison the cache
        first[0] = (-1, -1)
        third, _ = reach_table(ops, axis=0, increment=1)
        assert third == second

    def test_reach_table_cache_is_keyed_by_axis_and_increment(self):
        _reach_table_cached.cache_clear()
        ops = ordered_ops(
            nn.Sequential(
                nn.Conv3d(1, 1, kernel_size=(1, 3, 5), padding=(0, 1, 2)),
                nn.Conv3d(1, 1, kernel_size=(3, 1, 5), padding=(1, 0, 2)),
            )
        )

        reach_table(ops, axis=0, increment=1)
        assert _reach_table_cached.cache_info().misses == 1

        reach_table(ops, axis=1, increment=1)
        reach_table(ops, axis=0, increment=2)
        assert _reach_table_cached.cache_info().misses == 3

        reach_table(ops, axis=0, increment=1)
        info = _reach_table_cached.cache_info()
        assert info.hits == 1
        assert info.misses == 3


class TestEffectiveMinInput:
    """`effective_min_input` = smallest input that runs, crops and stays aligned."""

    def test_applies_the_halo_floor_and_rounds_to_the_increment(self):
        geometry = ModelGeometry(
            min_input=(8, 8, 8),
            increment=(8, 8, 8),
            halo=STANDARD_HALO,
            is_2d=False,
        )

        assert geometry.effective_min_input == STANDARD_EFFECTIVE_MINIMUM

    def test_keeps_min_input_where_it_already_exceeds_the_floor(self):
        geometry = ModelGeometry(
            min_input=(16, 32, 64), increment=(8, 8, 8), halo=(4, 8, 12), is_2d=False
        )

        assert geometry.effective_min_input == (16, 32, 64)

    def test_a_zero_halo_axis_keeps_its_min_input(self):
        geometry = ModelGeometry(
            min_input=(8, 16, 16), increment=(8, 8, 8), halo=(0, 4, 4), is_2d=False
        )

        assert geometry.effective_min_input == (8, 16, 16)

    def test_tensor_variant_uses_yx_for_2d(self):
        geometry = ModelGeometry(
            min_input=(1, 4, 4), increment=(1, 8, 8), halo=(0, 20, 20), is_2d=True
        )

        assert geometry.effective_min_input == (1, 48, 48)
        assert geometry.tensor_effective_min_input() == (48, 48)

    def test_tensor_variant_keeps_z_for_3d(self):
        geometry = ModelGeometry(
            min_input=(8, 8, 8),
            increment=(8, 8, 8),
            halo=STANDARD_HALO,
            is_2d=False,
        )

        assert geometry.tensor_effective_min_input() == STANDARD_EFFECTIVE_MINIMUM

    @pytest.mark.parametrize(
        "config",
        [
            {"f_maps": [4, 4, 4, 4]},
            {"f_maps": 8, "num_levels": 5},
        ],
    )
    @pytest.mark.parametrize("model_class", [UNet3D, UNet2D])
    def test_leaves_one_pixel_of_interior_that_the_model_accepts(
        self, model_class, config
    ):
        geometry = model_geometry(model_class(1, 1, **config))
        # cropping the halo from a tile of the effective minimum leaves the
        # patch the model actually produces artifact-free
        effective, halo = geometry.effective_min_input, geometry.halo
        patch = (
            effective[0] - 2 * halo[0],
            effective[1] - 2 * halo[1],
            effective[2] - 2 * halo[2],
        )

        assert all(pixels >= 1 for pixels in patch)
        # and that cropped tile is still a valid model input, aligned to the phase
        geometry.check_patch(patch)
        assert tuple(
            size % increment
            for size, increment in zip(
                geometry.model_input_shape(patch), geometry.increment
            )
        ) == (0, 0, 0)

    @pytest.mark.parametrize("model_class", [UNet3D, UNet2D], ids=["3d", "2d"])
    def test_the_effective_minimum_runs_through_a_real_unet(self, model_class):
        """The effective minimum is aligned, so the decoder sees exact halvings."""
        model = model_class(in_channels=1, out_channels=1, f_maps=[4, 4, 4, 4])
        geometry = model_geometry(model)
        shape = geometry.tensor_effective_min_input()

        with torch.no_grad():
            out = model(torch.zeros(1, 1, *shape))

        assert tuple(out.shape[2:]) == shape


class TestAlignedShape:
    def test_rounds_each_axis_up_to_its_increment(self):
        geometry = ModelGeometry(
            min_input=(1, 8, 8), increment=(1, 8, 8), halo=(0, 48, 48), is_2d=True
        )

        assert geometry.aligned_shape((7, 101, 128)) == (7, 104, 128)

    def test_leaves_an_aligned_shape_unchanged(self):
        geometry = model_geometry(UNet3D(1, 1, f_maps=[4, 4, 4, 4]))

        assert geometry.aligned_shape((16, 104, 8)) == (16, 104, 8)


class TestStrideShape:
    @pytest.mark.parametrize(
        "patch_shape, increment, expected",
        [
            ((96, 96, 96), (8, 8, 8), (72, 72, 72)),
            # round down to a multiple of the period, never below it
            ((100, 100, 100), (8, 8, 8), (72, 72, 72)),
            ((8, 8, 8), (8, 8, 8), (8, 8, 8)),
            ((4, 4, 4), (8, 8, 8), (1, 1, 1)),
            ((1, 64, 64), (1, 8, 8), (1, 48, 48)),
            # the stride never exceeds the patch; a sub-period patch does not stride
            ((8, 8, 8), (1, 16, 16), (6, 1, 1)),
        ],
    )
    def test_stride_shape_is_aligned_to_the_phase_period(
        self, patch_shape, increment, expected
    ):
        assert stride_shape(patch_shape, increment) == expected

    def test_model_geometry_stride_shape_uses_its_own_increment(self):
        geometry = model_geometry(UNet3D(1, 1, f_maps=[4, 4, 4, 4]))

        assert geometry.increment == (8, 8, 8)
        assert geometry.stride_shape((96, 96, 96)) == (72, 72, 72)


class TestModelInputShape:
    def test_adds_the_halo_on_both_sides(self):
        assert model_input_shape((4, 8, 8), (1, 2, 3)) == (6, 12, 14)

    def test_tensor_shape_drops_z_for_2d_models(self):
        geometry = model_geometry(UNet2D(1, 1, f_maps=[4, 4, 4, 4]))

        assert geometry.tensor_input_shape((1, 16, 16), (0, 2, 2)) == (20, 20)

    def test_tensor_shape_keeps_z_for_3d_models(self):
        geometry = model_geometry(UNet3D(1, 1, f_maps=[4, 4, 4, 4]))

        assert geometry.tensor_input_shape((8, 16, 16), (8, 48, 48)) == (
            24,
            112,
            112,
        )


class TestCheckInput:
    @pytest.fixture()
    def geometry(self):
        return model_geometry(UNet3D(1, 1, f_maps=[4, 4, 4, 4]))

    def test_check_input_names_the_failing_axis(self, geometry):
        with pytest.raises(ValueError, match=r"too small in: Y \(7 < 8\)"):
            geometry.check_input((16, 7, 64))

        with pytest.raises(ValueError, match=r"too small in: Y \(7 < 8\), X \(7 < 8\)"):
            geometry.check_input((16, 7, 7))

    def test_check_input_hints_a_2d_model_for_thin_z(self, geometry):
        with pytest.raises(
            ValueError,
            match=r"needs at least 8 Z slices, but the image has 4"
            r".*Use a 2D model for images with fewer Z slices",
        ):
            geometry.check_input((4, 64, 64))


class TestCheckPatch:
    @pytest.fixture()
    def geometry(self):
        return model_geometry(UNet3D(1, 1, f_maps=[4, 4, 4, 4]))

    @pytest.mark.parametrize(
        ("patch", "halo", "match"),
        [
            ((7, 8, 8), (0, 0, 0), r"Z \(7 < 8\)"),
            ((8, 7, 7), (0, 0, 0), r"Y \(7 < 8\).*X \(7 < 8\)"),
        ],
    )
    def test_error_names_the_failing_axis(self, geometry, patch, halo, match):
        with pytest.raises(ValueError, match=match):
            geometry.check_patch(patch, halo)

    def test_accepts_the_minimum_and_a_halo_padded_small_patch(self, geometry):
        geometry.check_patch((8, 8, 8), (0, 0, 0))
        # the halo is part of the model input, so a 4 px patch can run
        geometry.check_patch((4, 4, 4), (2, 2, 2))

    def test_uses_the_theoretical_halo_when_none_is_given(self, geometry):
        # a tiny patch is fine with the 48 px halo
        geometry.check_patch((1, 1, 1))

    def test_does_not_gate_on_alignment_or_manual_halo(self, geometry):
        """Manual halos are used verbatim; only runnability is enforced."""
        # unaligned patch and halo, but the model input is large enough
        geometry.check_patch((8, 8, 8), (3, 0, 0))

    def test_2d_model_rejects_a_z_halo_larger_than_the_minimum(self):
        geometry = model_geometry(UNet2D(1, 1, f_maps=[4, 4, 4, 4]))

        # min_input Z is 1, so an empty Z patch is rejected
        with pytest.raises(ValueError, match=r"Z \(0 < 1\)"):
            geometry.check_patch((0, 8, 8), (0, 0, 0))

    def test_2d_model_accepts_one_z_slice(self):
        geometry = model_geometry(UNet2D(1, 1, f_maps=[4, 4, 4, 4]))

        geometry.check_patch((1, 8, 8), (0, 0, 0))

    def test_check_patch_rejects_a_2d_patch_with_depth(self):
        geometry = model_geometry(UNet2D(1, 1, f_maps=[4, 4, 4, 4]))

        with pytest.raises(
            ValueError,
            match=r"has 2 Z slices, but 2D models predict one Z slice at a time"
            r".*Set the Z patch size to 1",
        ):
            geometry.check_patch((2, 8, 8), (0, 0, 0))

    def test_check_patch_rejects_a_2d_z_halo(self):
        geometry = model_geometry(UNet2D(1, 1, f_maps=[4, 4, 4, 4]))

        with pytest.raises(
            ValueError,
            match=r"has a Z halo of 1, but 2D models predict one Z slice at a time"
            r".*Set the Z halo to 0",
        ):
            geometry.check_patch((1, 8, 8), (1, 48, 48))

    def test_check_patch_accepts_the_2d_defaults(self):
        geometry = model_geometry(UNet2D(1, 1, f_maps=[4, 4, 4, 4]))

        # the derived patch is one slice deep with no Z halo
        geometry.check_patch((1, 8, 8))
        geometry.check_patch((1, 8, 8), (0, 0, 0))
