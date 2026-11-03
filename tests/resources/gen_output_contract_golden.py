"""Generate ``tests/resources/output_contract_golden.pt`` (the output-contract fence).

Golden values of every externally visible output, captured from the code
**before** the double-sigmoid fix (eval-mode ``forward()`` still applied the
final activation). They fence the output contract: after ``forward()`` returns
logits, the ``unet_prediction`` probability maps, the embedding path,
``test_out.npy`` semantics and the exported BioImage.IO callable must reproduce
these values.

Only regenerate the golden if the fence itself has to move (a deliberate,
documented contract change), and only after the change has been reviewed: the
file is the independent pre-change record, so regenerating it from the new code
drops that record. Adjust the two "pre-change" lines that use
``model.predict`` to ``model(x)`` when generating against pre-change code.

Run from the checkout root::

    /home/kriedmiller/software/miniforge3/envs/panseg-fork/bin/python \\
        tests/resources/gen_output_contract_golden.py [OUT_PATH]
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from panseg.functionals.prediction import unet_prediction  # noqa: E402
from panseg.functionals.prediction.utils.array_predictor import (
    ArrayPredictor,  # noqa: E402
)
from panseg.functionals.training.model import UNet2D, UNet3D  # noqa: E402
from tests.functionals.prediction.helpers import (  # noqa: E402
    MODEL_CONFIG,
    make_dataset,
)

DEFAULT_OUT = Path(__file__).resolve().parent / "output_contract_golden.pt"


def make_3d(final_sigmoid: bool, out_channels: int = 1) -> UNet3D:
    return UNet3D(
        in_channels=1,
        out_channels=out_channels,
        final_sigmoid=final_sigmoid,
        f_maps=[8, 16],
        num_groups=4,
        layer_order="gcr",
    )


def make_2d() -> UNet2D:
    return UNet2D(
        in_channels=1,
        out_channels=2,
        final_sigmoid=True,
        f_maps=[8, 16],
        num_groups=4,
        layer_order="gcr",
    )


def main(out_path: Path = DEFAULT_OUT) -> None:
    torch.manual_seed(0)
    model_3d = make_3d(final_sigmoid=True)
    torch.manual_seed(1)
    model_3d_softmax = make_3d(final_sigmoid=False, out_channels=2)
    torch.manual_seed(2)
    model_2d = make_2d()

    g = torch.Generator().manual_seed(3)
    input_3d = torch.randn(1, 1, 8, 16, 16, generator=g)
    input_2d = torch.randn(1, 1, 32, 32, generator=g)

    # Train mode returns logits in every version. Pre-change, *eval* mode
    # returned the activated output -- on current code that value is
    # `predict()` (the fix pins `predict()` to the old eval-mode output).
    with torch.no_grad():
        model_3d.train()
        gold_logits_3d = model_3d(input_3d)
        model_3d.eval()
        gold_probs_3d = model_3d.predict(input_3d)

        model_3d_softmax.train()
        gold_logits_softmax_3d = model_3d_softmax(input_3d)
        model_3d_softmax.eval()
        gold_softmax_3d = model_3d_softmax.predict(input_3d)

        model_2d.train()
        gold_logits_2d = model_2d(input_2d)
        model_2d.eval()
        gold_probs_2d = model_2d.predict(input_2d)

    raw = np.random.default_rng(0).random((8, 96, 96)).astype("float32")
    patch, halo = (8, 64, 64), (2, 8, 8)

    # Inference boundary: the full unet_prediction pipeline (zoo config
    # loading, patch + halo, overlap-averaging).
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        torch.save(model_3d.state_dict(), model_dir / "best_checkpoint.pytorch")
        (model_dir / "config_train.yml").write_text(MODEL_CONFIG)
        gold_infer_3d = unet_prediction(
            raw=raw,
            input_layout="ZYX",
            model_name=None,
            model_id=None,
            patch=patch,
            patch_halo=halo,
            config_path=model_dir / "config_train.yml",
            model_weights_path=model_dir / "best_checkpoint.pytorch",
            device="cpu",
            disable_tqdm=True,
        )

    # Embedding boundary: today's numbers are computed from the model's
    # activated (Softmax) output.
    dataset = make_dataset(raw, patch=patch, halo=halo)
    predictor = ArrayPredictor(
        model=model_3d_softmax,
        in_channels=1,
        out_channels=1,
        device="cpu",
        patch=patch,
        patch_halo=halo,
        single_batch_mode=True,
        headless=False,
        is_embedding=True,
        disable_tqdm=True,
    )
    gold_embed_3d = predictor(dataset)

    torch.save(
        {
            "state_unet3d": model_3d.state_dict(),
            "state_unet3d_softmax": model_3d_softmax.state_dict(),
            "state_unet2d": model_2d.state_dict(),
            "state_keys_unet3d": sorted(model_3d.state_dict().keys()),
            "state_keys_unet2d": sorted(model_2d.state_dict().keys()),
            "input_3d": input_3d,
            "input_2d": input_2d,
            "gold_logits_3d": gold_logits_3d,
            "gold_probs_3d": gold_probs_3d,
            "gold_logits_softmax_3d": gold_logits_softmax_3d,
            "gold_softmax_3d": gold_softmax_3d,
            "gold_logits_2d": gold_logits_2d,
            "gold_probs_2d": gold_probs_2d,
            "raw_3d": torch.from_numpy(raw),
            "patch": torch.tensor(patch),
            "halo": torch.tensor(halo),
            "gold_infer_3d": torch.from_numpy(gold_infer_3d),
            "gold_embed_3d": torch.from_numpy(gold_embed_3d),
        },
        out_path,
    )
    print(f"wrote {out_path}")
    print("shapes:", gold_infer_3d.shape, gold_embed_3d.shape)
    for name, arr in [
        ("infer", gold_infer_3d),
        ("embed", gold_embed_3d),
        ("softmax3d", gold_softmax_3d.detach().numpy()),
    ]:
        print(f"{name}: range [{arr.min()}, {arr.max()}] std {arr.std()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
