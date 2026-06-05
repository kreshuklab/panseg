import shutil
from contextlib import chdir
from pathlib import Path

import numpy as np

from panseg.functionals.training.biio import make_model_description


def test_make_model_description(tmp_path):
    weights = (
        Path(__file__).parent.parent.parent
        / "resources"
        / "models"
        / "best_checkpoint.pytorch"
    )
    shutil.copy(weights, tmp_path)

    inputs = tmp_path / "inputs.npy"
    np.save(inputs, np.random.rand(1, 1, 16, 50, 64))
    outputs = tmp_path / "outputs.npy"
    np.save(outputs, np.random.rand(1, 1, 16, 50, 64))

    with chdir(tmp_path):
        make_model_description(
            weights=Path("best_checkpoint.pytorch"),
            model_name="dummy_model",
            in_channels=1,
            out_channels=1,
            feature_maps=64,
            patch_size=(16, 32, 64),
            dimensionality="3D",
            modality="mod",
            output_type="boundaries",
            description="dummy model",
            resolution=(0.5, 0.02, 2),
            test_in=inputs,
            test_out=outputs,
            panseg_config=Path("best_checkpoint.pytorch"),
        )
