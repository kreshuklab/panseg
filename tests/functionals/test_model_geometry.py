"""Tests for the shared model-geometry helpers.

`model_geometry` is the single place that derives architecture-dependent input
geometry from a model: the smallest input each axis accepts (`min_input`) and
the context each side needs for an exact output (`halo`). Both come from one
per-axis traversal so they cannot diverge, and every consumer (prediction,
training, the viewer, the model zoo) asks the same module.
"""

import pytest
from torch import nn

from panseg.functionals.model_geometry import (
    is_2d_model,
    model_geometry,
    model_input_shape,
)
from panseg.functionals.training.model import UNet2D, UNet3D


class TestModelGeometryTraversal:
    def test_is_2d_model_is_decided_by_the_spatial_layers(self):
        model = UNet2D(1, 1, f_maps=[4, 4])

        assert is_2d_model(model)
        # wrappers are transparent: the spatial layers decide
        assert is_2d_model(nn.Sequential(model))
        assert not is_2d_model(UNet3D(1, 1, f_maps=[4, 4]))

    def test_unet3d_halo_and_min_input(self):
        model = UNet3D(in_channels=1, out_channels=1, f_maps=[4, 4, 4, 4])

        geometry = model_geometry(model)

        # three 2x pools on every axis, and the classic 44 px U-Net halo
        assert geometry.min_input == (8, 8, 8)
        assert geometry.halo == (44, 44, 44)
        assert not geometry.is_2d

    def test_unet2d_has_no_z_halo_and_stays_one_slice_deep(self):
        model = UNet2D(in_channels=1, out_channels=1, f_maps=[4, 4, 4, 4])

        geometry = model_geometry(model)

        assert geometry.min_input == (1, 8, 8)
        assert geometry.halo == (0, 44, 44)
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
    def test_rdf_architectures_keep_the_44_px_halo(self, model):
        assert model_geometry(model).halo in {(44, 44, 44), (0, 44, 44)}

    def test_anisotropic_conv_kernels_contribute_per_axis(self):
        model = nn.Sequential(nn.Conv3d(1, 1, kernel_size=(1, 3, 5)))

        geometry = model_geometry(model)

        # kernel // 2: 0 on Z, 1 on Y, 2 on X
        assert geometry.halo == (0, 1, 2)

    def test_average_pooling_counts_like_max_pooling(self):
        model = nn.Sequential(
            nn.AvgPool3d(2),
            nn.Conv3d(1, 1, 3),
            nn.Conv3d(1, 1, 3),
        )

        geometry = model_geometry(model)

        # both convs sit one level down: 2 * (1 + 1)
        assert geometry.halo == (4, 4, 4)
        assert geometry.min_input == (2, 2, 2)

    def test_pooling_kernel_multiplies_min_input_per_axis(self):
        model = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 4, 4)))

        assert model_geometry(model).min_input == (2, 4, 4)

    def test_one_by_one_convs_need_no_halo_but_still_need_the_pooled_input(self):
        model = nn.Sequential(
            nn.MaxPool3d(2),
            nn.MaxPool3d(2),
            nn.MaxPool3d(2),
            nn.Conv3d(1, 1, 1),
            nn.Conv3d(1, 1, 1),
        )

        geometry = model_geometry(model)

        assert geometry.halo == (0, 0, 0)
        assert geometry.min_input == (8, 8, 8)

    @pytest.mark.parametrize(
        "config",
        [
            {"f_maps": [4, 4]},
            {"f_maps": [4, 4, 4, 4]},
            {"f_maps": 8, "num_levels": 5},
            {"f_maps": [4, 4, 4, 4], "conv_kernel_size": 5, "pool_kernel_size": 2},
            {"f_maps": [4, 4, 4, 4], "conv_kernel_size": 7, "pool_kernel_size": 4},
        ],
    )
    @pytest.mark.parametrize("model_class", [UNet3D, UNet2D])
    def test_theoretical_halo_always_reaches_the_architectural_minimum(
        self, model_class, config
    ):
        """`halo + 1 >= min_input` on every axis a model pools.

        The automatic path derives patches with the theoretical halo, so this
        invariant is what keeps every auto-derived shape valid.
        """
        geometry = model_geometry(model_class(1, 1, **config))

        for axis, (halo, minimum) in enumerate(zip(geometry.halo, geometry.min_input)):
            if geometry.is_2d and axis == 0:
                continue
            assert halo + 1 >= minimum


class TestModelInputShape:
    def test_adds_the_halo_on_both_sides(self):
        assert model_input_shape((4, 8, 8), (1, 2, 3)) == (6, 12, 14)

    def test_tensor_shape_drops_z_for_2d_models(self):
        geometry = model_geometry(UNet2D(1, 1, f_maps=[4, 4, 4, 4]))

        assert geometry.tensor_input_shape((1, 16, 16), (0, 2, 2)) == (20, 20)

    def test_tensor_shape_keeps_z_for_3d_models(self):
        geometry = model_geometry(UNet3D(1, 1, f_maps=[4, 4, 4, 4]))

        assert geometry.tensor_input_shape((8, 16, 16), (4, 44, 44)) == (
            16,
            104,
            104,
        )


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
        # a tiny patch is fine with the 44 px halo
        geometry.check_patch((1, 1, 1))

    def test_2d_model_rejects_a_z_halo_larger_than_the_minimum(self):
        geometry = model_geometry(UNet2D(1, 1, f_maps=[4, 4, 4, 4]))

        # min_input Z is 1, so an empty Z patch is rejected
        with pytest.raises(ValueError, match=r"Z \(0 < 1\)"):
            geometry.check_patch((0, 8, 8), (0, 0, 0))

    def test_2d_model_accepts_one_z_slice(self):
        geometry = model_geometry(UNet2D(1, 1, f_maps=[4, 4, 4, 4]))

        geometry.check_patch((1, 8, 8), (0, 0, 0))
