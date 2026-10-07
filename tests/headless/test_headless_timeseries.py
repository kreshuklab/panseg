"""End-to-end headless run of the spec's complete example timeseries workflow.

The committed resliced OME-TIFF anchor (TZYX, 32x32 frames, too small for a
real U-Net patch) goes through import -> set_t_spacing -> gaussian ->
unet prediction -> dt watershed -> clustering -> set-biggest -> tiff export.
Only the model inference is mocked (``unet_prediction`` in the task module);
every task body, the frame-by-frame loop and all I/O run for real. No
network, no model weights, so the test is fast and not marked slow.
"""

from pathlib import Path

import numpy as np
import yaml

from panseg.core.image import ImageLayout, PanSegImage, import_image
from panseg.headless.headless import run_headless_workflow

TEST_FILES = Path(__file__).resolve().parent.parent / "resources"
RESLICED_TZYX_ANCHOR = TEST_FILES / "ome_tiff_examples" / "4D-series.ome.tif"
EXAMPLE_WORKFLOW_YAML = TEST_FILES / "test_workflow_t_spacing.yaml"

T_SPACING = 5.0


def _mock_unet_prediction(raw: np.ndarray, **kwargs) -> np.ndarray:
    """Deterministic boundary map: a central high-probability blob."""
    pmap = np.zeros((1,) + raw.shape, dtype="float32")
    pmap[:, :, 4:-4, 4:-4] = 0.9
    return pmap


def _example_timeseries_config(tmp_path: Path) -> Path:
    with open(EXAMPLE_WORKFLOW_YAML, "r") as f:
        config = yaml.safe_load(f)

    config["runner"] = "serial"
    config["inputs"]["input_path"] = str(RESLICED_TZYX_ANCHOR)
    config["inputs"]["export_directory"] = str(tmp_path / "output")

    workflow_path = tmp_path / "example_timeseries_workflow.yaml"
    with open(workflow_path, "w") as f:
        yaml.safe_dump(config, f)
    return workflow_path


def test_example_timeseries_workflow_runs_headless(mocker, tmp_path: Path):
    mocker.patch(
        "panseg.tasks.prediction_tasks.unet_prediction",
        side_effect=_mock_unet_prediction,
    )
    workflow_path = _example_timeseries_config(tmp_path)

    run_headless_workflow(workflow_path)

    results = list((tmp_path / "output").glob("*.tiff"))
    assert len(results) == 1, results

    reimported = import_image(
        path=results[0],
        image_name="reimported",
        semantic_type="segmentation",
        stack_layout="TZYX",
    )
    assert isinstance(reimported, PanSegImage)
    assert reimported.image_layout == ImageLayout.TZYX
    assert reimported.shape == (4, 2, 32, 32)
    assert reimported.properties.t_spacing == T_SPACING
