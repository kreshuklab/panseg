"""fp32 precision of the loss reductions under AMP.

The loss reductions must stay exact under `amp: auto` (bf16 autocast): Dice
sums over many thousand voxels lose significant precision in bf16, so the
loss computes in float32 even when the model runs in bf16.
"""

import pytest
import torch

from panseg.functionals.training.losses import DiceLoss, compute_per_channel_dice


class TestDiceLossPrecision:
    """Dice reductions must compute in fp32 even from bf16 logits.

    The reference in each test is computed from the *same values* cast to
    float32, so fp32 reductions match bit-exactly while bf16 products/sums do
    not (the squared-product rounding alone is up to 0.35 % per element).
    """

    def test_dice_loss_output_is_fp32_for_bf16_logits(self):
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(2, 1, 32, 32, 32, generator=g).to(torch.bfloat16)
        target = torch.rand(2, 1, 32, 32, 32, generator=g)

        loss = DiceLoss()(logits, target)

        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)

    def test_dice_reductions_exact_for_bf16_input(self):
        # 0.74609375 is the bf16 value whose squared product has the largest
        # rounding bias: a bf16 reduction is visibly off for it.
        n = 2**18
        target = torch.ones(1, 1, n)
        x_bf16 = torch.full((1, 1, n), 0.74609375, dtype=torch.bfloat16)

        reference = compute_per_channel_dice(x_bf16.float(), target)

        torch.testing.assert_close(
            compute_per_channel_dice(x_bf16, target), reference, rtol=1e-4, atol=0
        )

    def test_dice_loss_exact_for_bf16_input_default_path(self):
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(2, 1, 64, 64, 64, generator=g) * 2
        target = torch.rand(2, 1, 64, 64, 64, generator=g)
        logits_bf16 = logits.to(torch.bfloat16)

        reference = DiceLoss()(logits_bf16.float(), target)
        loss = DiceLoss()(logits_bf16, target)

        assert float(loss) == pytest.approx(float(reference), rel=2e-4)
