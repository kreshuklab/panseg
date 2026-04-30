from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import yaml

from panseg.core.image import PanSegImage
from panseg.headless.headless import run_headless_workflow
from panseg.io.tiff import create_tiff
from panseg.io.voxelsize import VoxelSize
from panseg.tasks.dataprocessing_tasks import gaussian_smoothing_task
from panseg.tasks.io_tasks import (
    export_image_task,
    import_image_task,
    merge_channels_task,
)
from panseg.tasks.workflow_handler import workflow_handler


def create_random_tiff(tmpdir, name, shape=(32, 32), layout="YX") -> Path:
    path_tiff = Path(tmpdir) / name
    create_tiff(
        path_tiff,
        np.random.rand(*shape).astype("float32"),
        voxel_size=VoxelSize(voxels_size=(1, 1, 1), unit="um"),
        layout=layout,
    )
    return path_tiff


def test_create_workflow(tmp_path):
    # Create an empty tiff file
    path_tiff = create_random_tiff(tmp_path, "test.tiff")

    workflow_handler.clean_dag()

    ps_1 = import_image_task(
        input_path=path_tiff, key="raw", semantic_type="raw", stack_layout="YX"
    )
    assert isinstance(ps_1, PanSegImage)
    ps_2 = gaussian_smoothing_task(image=ps_1, sigma=1.0)
    assert isinstance(ps_2, PanSegImage)
    export_image_task(
        image=ps_2,
        export_directory=path_tiff.parent,
        name_pattern="{image_name}_export",
        scale_to_origin=True,
    )

    workflow_handler.save_to_yaml(tmp_path / "workflow.yaml")

    dag = workflow_handler.dag
    assert len(dag.list_tasks) == 3
    assert len(dag.inputs[0].keys()) == 3

    # Run the headless workflow

    path_tiff_1 = create_random_tiff(tmp_path, "test1.tiff")
    path_tiff_2 = create_random_tiff(tmp_path, "test2.tiff")

    with open(tmp_path / "workflow.yaml", "r") as file:
        config = yaml.safe_load(file)

    job_list = [
        {
            "input_path": str(path_tiff_1),
            "export_directory": str(tmp_path / "output"),
            "name_pattern": "{file_name}_export",
        },
        {
            "input_path": str(path_tiff_2),
            "export_directory": str(tmp_path / "output"),
            "name_pattern": "{file_name}_export",
        },
    ]

    config["inputs"] = job_list

    with open(tmp_path / "workflow.yaml", "w") as file:
        yaml.dump(config, file)

    run_headless_workflow(tmp_path / "workflow.yaml")

    results_dir = tmp_path / "output"
    results = list(results_dir.glob("*"))
    assert len(results) == 2, results


def test_create_workflow_channels(tmp_path):
    # Create an empty tiff file
    path_tiff = create_random_tiff(tmp_path, "test.tiff", (4, 64, 32), "CYX")

    workflow_handler.clean_dag()

    ps_1s = import_image_task(
        input_path=path_tiff, key="raw", semantic_type="raw", stack_layout="CYX"
    )
    assert isinstance(ps_1s, list)
    ps_2 = gaussian_smoothing_task(image=ps_1s[0], sigma=1.0)
    assert isinstance(ps_2, PanSegImage)
    ps_3 = merge_channels_task(
        **{f"image_{i}": ps for i, ps in enumerate(ps_1s + [ps_2])}
    )
    assert isinstance(ps_3, PanSegImage)
    np.testing.assert_allclose(ps_1s[0].get_data(), ps_3.get_data()[0])
    np.testing.assert_allclose(ps_2.get_data(), ps_3.get_data()[4])
    export_image_task(
        image=ps_3,
        export_directory=path_tiff.parent,
        name_pattern="{image_name}_export",
        scale_to_origin=True,
    )

    workflow_handler.save_to_yaml(tmp_path / "workflow.yaml")

    dag = workflow_handler.dag
    assert len(dag.list_tasks) == 4
    assert len(dag.inputs[0].keys()) == 3

    # Run the headless workflow

    path_tiff_1 = create_random_tiff(tmp_path, "test1.tiff", (4, 64, 32), "CYX")
    path_tiff_2 = create_random_tiff(tmp_path, "test2.tiff", (4, 64, 32), "CYX")

    with open(tmp_path / "workflow.yaml", "r") as file:
        config = yaml.safe_load(file)

    job_list = [
        {
            "input_path": str(path_tiff_1),
            "export_directory": str(tmp_path / "output"),
            "name_pattern": "{file_name}_export",
        },
        {
            "input_path": str(path_tiff_2),
            "export_directory": str(tmp_path / "output"),
            "name_pattern": "{file_name}_export",
        },
    ]

    config["inputs"] = job_list

    with open(tmp_path / "workflow.yaml", "w") as file:
        yaml.dump(config, file)

    run_headless_workflow(tmp_path / "workflow.yaml")

    results_dir = tmp_path / "output"
    results = list(results_dir.glob("*"))
    assert len(results) == 2, results
