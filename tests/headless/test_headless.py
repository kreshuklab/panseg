import logging
from pathlib import Path

import numpy as np
import pytest
import tifffile
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
from panseg.tasks.segmentation_tasks import aio_watershed_task, dt_watershed_task
from panseg.tasks.workflow_handler import workflow_handler
from tests.conftest import write_still_tiff


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
    assert len(dag.inputs.keys()) == 3

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


def test_blockwise_watershed_headless(tmp_path, caplog):
    # (128, 128, 128) is large enough (>= 2M voxels) so the blockwise path
    # is taken by the functional instead of falling back to single-pass
    path_tiff = create_random_tiff(tmp_path, "test.tiff", (128, 128, 128), "ZYX")

    workflow_handler.clean_dag()

    ps_1 = import_image_task(
        input_path=path_tiff, key="raw", semantic_type="raw", stack_layout="ZYX"
    )
    assert isinstance(ps_1, PanSegImage)
    ps_2 = dt_watershed_task(image=ps_1, blockwise=True)
    assert isinstance(ps_2, PanSegImage)
    export_image_task(
        image=ps_2,
        export_directory=path_tiff.parent,
        name_pattern="{image_name}_export",
        scale_to_origin=True,
    )

    workflow_handler.save_to_yaml(tmp_path / "workflow.yaml")

    # the saved workflow must carry the blockwise parameter of the dt-watershed task
    with open(tmp_path / "workflow.yaml", "r") as file:
        config = yaml.safe_load(file)

    dt_watershed_entries = [
        task for task in config["list_tasks"] if task["func"] == "dt_watershed_task"
    ]
    assert len(dt_watershed_entries) == 1
    assert dt_watershed_entries[0]["parameters"]["blockwise"] is True

    # Run the headless workflow

    path_tiff_1 = create_random_tiff(tmp_path, "test1.tiff", (128, 128, 128), "ZYX")
    path_tiff_2 = create_random_tiff(tmp_path, "test2.tiff", (128, 128, 128), "ZYX")

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

    with caplog.at_level(
        logging.WARNING, logger="panseg.functionals.segmentation.segmentation"
    ):
        run_headless_workflow(tmp_path / "workflow.yaml")
    # (128, 128, 128) >= 2M voxels, so the blockwise path is genuinely taken
    assert "falling back" not in caplog.text

    results_dir = tmp_path / "output"
    results = list(results_dir.glob("*"))
    assert len(results) == 2, results


def test_blockwise_aio_watershed_headless(tmp_path, caplog):
    # (128, 128, 128) is large enough (>= 2M voxels) so the blockwise path
    # is taken by the functional instead of falling back to single-pass
    path_tiff = create_random_tiff(tmp_path, "test.tiff", (128, 128, 128), "ZYX")

    workflow_handler.clean_dag()

    ps_1 = import_image_task(
        input_path=path_tiff, key="raw", semantic_type="raw", stack_layout="ZYX"
    )
    assert isinstance(ps_1, PanSegImage)
    ps_2 = aio_watershed_task(image=ps_1, nuclei=None, blockwise=True, mode="gasp")
    assert isinstance(ps_2, PanSegImage)
    export_image_task(
        image=ps_2,
        export_directory=path_tiff.parent,
        name_pattern="{image_name}_export",
        scale_to_origin=True,
    )

    workflow_handler.save_to_yaml(tmp_path / "workflow.yaml")

    # the saved workflow must carry the blockwise parameter of the aio-watershed task
    with open(tmp_path / "workflow.yaml", "r") as file:
        config = yaml.safe_load(file)

    aio_watershed_entries = [
        task for task in config["list_tasks"] if task["func"] == "aio_watershed_task"
    ]
    assert len(aio_watershed_entries) == 1
    assert aio_watershed_entries[0]["parameters"]["blockwise"] is True
    assert aio_watershed_entries[0]["parameters"]["mode"] == "gasp"

    # Run the headless workflow

    path_tiff_1 = create_random_tiff(tmp_path, "test1.tiff", (128, 128, 128), "ZYX")
    path_tiff_2 = create_random_tiff(tmp_path, "test2.tiff", (128, 128, 128), "ZYX")

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

    with caplog.at_level(
        logging.WARNING, logger="panseg.functionals.segmentation.segmentation"
    ):
        run_headless_workflow(tmp_path / "workflow.yaml")
    # (128, 128, 128) >= 2M voxels, so the blockwise path is genuinely taken
    assert "falling back" not in caplog.text

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
    assert len(dag.inputs.keys()) == 3

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


# --- Multi-file time series import (multifile-timeseries-import spec) ---


def test_headless_dir_of_dirs_stacks_one_series_per_subdirectory(tmp_path):
    """A directory of subdirectories maps one stacked job per subdirectory."""
    input_directory = tmp_path / "inputs"
    (input_directory / "series_a").mkdir(parents=True)
    (input_directory / "series_b").mkdir(parents=True)
    for stem, value in (("a1", 1), ("a2", 2)):
        write_still_tiff(input_directory / "series_a" / f"{stem}.tiff", (16, 16), value)
    for stem, value in (("b1", 10), ("b2", 20)):
        write_still_tiff(input_directory / "series_b" / f"{stem}.tiff", (16, 16), value)

    path_tiff = create_random_tiff(tmp_path, "test.tiff")
    workflow_handler.clean_dag()
    ps_1 = import_image_task(
        input_path=path_tiff, key="raw", semantic_type="raw", stack_layout="YX"
    )
    assert isinstance(ps_1, PanSegImage)
    export_image_task(
        image=ps_1,
        export_directory=tmp_path / "output",
        name_pattern="{file_name}_export",
        scale_to_origin=True,
    )
    workflow_handler.save_to_yaml(tmp_path / "workflow.yaml")

    with open(tmp_path / "workflow.yaml", "r") as file:
        config = yaml.safe_load(file)
    config["inputs"] = [
        {
            "input_path": str(input_directory),
            "export_directory": str(tmp_path / "output"),
            "name_pattern": "{file_name}_export",
        }
    ]
    with open(tmp_path / "workflow.yaml", "w") as file:
        yaml.dump(config, file)

    run_headless_workflow(tmp_path / "workflow.yaml")

    results = sorted((tmp_path / "output").glob("*_export.ome.tiff"))
    assert len(results) == 2, results
    assert {path.name for path in results} == {
        "a1_export.ome.tiff",
        "b1_export.ome.tiff",
    }
    for path in results:
        with tifffile.TiffFile(path) as tiff:
            assert tiff.series[0].axes == "TYX"
            assert tiff.series[0].shape[0] == 2  # SizeT=2


def test_headless_list_input_stacks_one_job(tmp_path):
    path_tiff = create_random_tiff(tmp_path, "test.tiff")
    workflow_handler.clean_dag()
    ps_1 = import_image_task(
        input_path=path_tiff, key="raw", semantic_type="raw", stack_layout="YX"
    )
    assert isinstance(ps_1, PanSegImage)
    export_image_task(
        image=ps_1,
        export_directory=tmp_path / "output",
        name_pattern="{file_name}_export",
        scale_to_origin=True,
    )
    workflow_handler.save_to_yaml(tmp_path / "workflow.yaml")

    first = write_still_tiff(tmp_path / "a1.tiff", (16, 16), 1)
    second = write_still_tiff(tmp_path / "a2.tiff", (16, 16), 2)

    with open(tmp_path / "workflow.yaml", "r") as file:
        config = yaml.safe_load(file)
    config["inputs"] = [
        {
            "input_path": [str(first), str(second)],
            "export_directory": str(tmp_path / "output"),
            "name_pattern": "{file_name}_export",
        }
    ]
    with open(tmp_path / "workflow.yaml", "w") as file:
        yaml.dump(config, file)

    run_headless_workflow(tmp_path / "workflow.yaml")

    results = list((tmp_path / "output").glob("*_export.ome.tiff"))
    assert len(results) == 1, results
    with tifffile.TiffFile(results[0]) as tiff:
        assert tiff.series[0].axes == "TYX"
        assert tiff.series[0].shape[0] == 2


def test_headless_files_win_over_subdirectories(tmp_path):
    """A directory with both files and subdirectories keeps one job per file."""
    directory = tmp_path / "inputs"
    directory.mkdir()
    write_still_tiff(directory / "a1.tiff", (16, 16), 1)
    write_still_tiff(directory / "a2.tiff", (16, 16), 2)
    (directory / "subdir").mkdir()
    write_still_tiff(directory / "subdir" / "b1.tiff", (16, 16), 10)

    from panseg.headless.headless import collect_jobs_list
    from panseg.tasks.workflow_handler import RunTimeInputSchema

    jobs = collect_jobs_list(
        {"input_path": str(directory)},
        {"input_path": RunTimeInputSchema(is_input_file=True)},
    )
    assert len(jobs) == 2
    assert all(isinstance(job["input_path"], Path) for job in jobs)


def test_headless_empty_subdirectory_rejected(tmp_path):
    directory = tmp_path / "inputs"
    directory.mkdir()
    (directory / "empty").mkdir()

    from panseg.headless.headless import parse_import_image_task

    with pytest.raises(ValueError, match="empty"):
        parse_import_image_task(directory, allow_dir=True)


def test_headless_nonexistent_list_entry_rejected(tmp_path):
    from panseg.headless.headless import parse_import_image_task

    existing = write_still_tiff(tmp_path / "a1.tiff", (16, 16), 1)
    with pytest.raises(ValueError, match="missing.tif"):
        parse_import_image_task(
            [str(existing), str(tmp_path / "missing.tif")], allow_dir=False
        )


def test_headless_gui_saved_multifile_workflow_reruns(tmp_path):
    """A workflow saved from a multi-file GUI import re-runs headless.

    The DAG stores the actual input values, so the saved YAML carries the
    file list itself; running the saved file without overrides must work.
    """
    first = write_still_tiff(tmp_path / "a1.tiff", (16, 16), 1)
    second = write_still_tiff(tmp_path / "a2.tiff", (16, 16), 2)

    workflow_handler.clean_dag()
    ps_1 = import_image_task(
        input_path=(first, second),
        key="raw",
        semantic_type="segmentation",
        stack_layout="YX",
    )
    assert isinstance(ps_1, PanSegImage)
    export_image_task(
        image=ps_1,
        export_directory=tmp_path / "output",
        name_pattern="{file_name}_export",
        scale_to_origin=True,
    )
    workflow_handler.save_to_yaml(tmp_path / "workflow.yaml")

    with open(tmp_path / "workflow.yaml", "r") as file:
        config = yaml.safe_load(file)
    saved_input = config["inputs"]["input_path"]
    assert isinstance(saved_input, list)
    assert sorted(saved_input) == [str(first), str(second)]

    run_headless_workflow(tmp_path / "workflow.yaml")

    results = list((tmp_path / "output").glob("*_export.ome.tiff"))
    assert len(results) == 1, results
    with tifffile.TiffFile(results[0]) as tiff:
        assert tiff.series[0].shape[0] == 2


def _timeseries_workflow(tmp_path: Path) -> dict:
    """A workflow recorded from a multi-file (time series) import, with the
    stacked (T-bearing) stack_layout the GUI now prefills and records, a
    frame-mapped processing step, and the export."""
    first = write_still_tiff(tmp_path / "a1.tiff", (16, 16), 1)
    second = write_still_tiff(tmp_path / "a2.tiff", (16, 16), 2)

    workflow_handler.clean_dag()
    ps_1 = import_image_task(
        input_path=(first, second),
        key="raw",
        semantic_type="segmentation",
        stack_layout="TYX",
    )
    assert isinstance(ps_1, PanSegImage)
    ps_2 = gaussian_smoothing_task(image=ps_1, sigma=1.0)
    assert isinstance(ps_2, PanSegImage)
    export_image_task(
        image=ps_2,
        export_directory=tmp_path / "output",
        name_pattern="{file_name}_export",
        scale_to_origin=True,
    )
    workflow_handler.save_to_yaml(tmp_path / "workflow.yaml")

    with open(tmp_path / "workflow.yaml", "r") as file:
        config = yaml.safe_load(file)

    import_entry = next(
        task for task in config["list_tasks"] if task["func"] == "import_image_task"
    )
    # the exported workflow pins the timeseries layout it was recorded from
    assert import_entry["parameters"]["stack_layout"] == "TYX"
    return config


def _run_timeseries_workflow(tmp_path: Path, config: dict, input_path, output_name):
    output_dir = tmp_path / output_name
    config["inputs"] = [
        {
            "input_path": input_path,
            "export_directory": str(output_dir),
            "name_pattern": "{file_name}_export",
        }
    ]
    workflow_path = tmp_path / f"{output_name}.yaml"
    with open(workflow_path, "w") as file:
        yaml.dump(config, file)
    run_headless_workflow(workflow_path)
    return output_dir


def test_headless_timeseries_workflow_runs_on_timeseries_file(tmp_path):
    """The exported workflow takes a single T-bearing file: the recorded
    layout matches the file exactly."""
    config = _timeseries_workflow(tmp_path)

    series = tmp_path / "series.tiff"
    create_tiff(
        series,
        np.arange(2 * 16 * 16, dtype="float32").reshape(2, 16, 16),
        VoxelSize(),
        layout="TYX",
    )

    output_dir = _run_timeseries_workflow(
        tmp_path, config, str(series.with_suffix(".ome.tiff")), "out_ts_file"
    )
    results = list(output_dir.glob("*_export.ome.tiff"))
    assert len(results) == 1, results
    with tifffile.TiffFile(results[0]) as tiff:
        assert tiff.series[0].axes == "TYX"
        assert tiff.series[0].shape[0] == 2


def test_headless_timeseries_workflow_runs_on_list_input(tmp_path):
    """The exported workflow takes a list of stills: the recorded T is
    stripped and stacking supplies the time axis."""
    config = _timeseries_workflow(tmp_path)

    b1 = write_still_tiff(tmp_path / "b1.tiff", (16, 16), 10)
    b2 = write_still_tiff(tmp_path / "b2.tiff", (16, 16), 20)

    output_dir = _run_timeseries_workflow(
        tmp_path, config, [str(b1), str(b2)], "out_list"
    )
    results = list(output_dir.glob("*_export.ome.tiff"))
    assert len(results) == 1, results
    with tifffile.TiffFile(results[0]) as tiff:
        assert tiff.series[0].axes == "TYX"
        assert tiff.series[0].shape[0] == 2


def test_headless_timeseries_workflow_runs_on_dir_of_dirs(tmp_path):
    """The exported workflow takes a directory of subdirectories: one
    stacked time series per subdirectory."""
    config = _timeseries_workflow(tmp_path)

    input_directory = tmp_path / "inputs"
    (input_directory / "series_a").mkdir(parents=True)
    (input_directory / "series_b").mkdir(parents=True)
    write_still_tiff(input_directory / "series_a" / "a1.tiff", (16, 16), 1)
    write_still_tiff(input_directory / "series_a" / "a2.tiff", (16, 16), 2)
    write_still_tiff(input_directory / "series_b" / "b1.tiff", (16, 16), 10)
    write_still_tiff(input_directory / "series_b" / "b2.tiff", (16, 16), 20)

    output_dir = _run_timeseries_workflow(
        tmp_path, config, str(input_directory), "out_dod"
    )
    results = sorted(output_dir.glob("*_export.ome.tiff"))
    assert len(results) == 2, results
    for path in results:
        with tifffile.TiffFile(path) as tiff:
            assert tiff.series[0].axes == "TYX"
            assert tiff.series[0].shape[0] == 2


def test_headless_timeseries_workflow_errors_on_dir_of_stills(tmp_path):
    """A time series workflow over a directory of still files errors: the
    directory-of-files shape runs one job per file, and a single-file job
    must match the recorded layout exactly - a still file is not a series."""
    config = _timeseries_workflow(tmp_path)

    file_dir = tmp_path / "plain"
    file_dir.mkdir()
    write_still_tiff(file_dir / "p1.tiff", (16, 16), 5)
    write_still_tiff(file_dir / "p2.tiff", (16, 16), 6)

    output_dir = tmp_path / "out_files"
    config["inputs"] = [
        {
            "input_path": str(file_dir),
            "export_directory": str(output_dir),
            "name_pattern": "{file_name}_export",
        }
    ]
    workflow_path = tmp_path / "out_files.yaml"
    with open(workflow_path, "w") as file:
        yaml.dump(config, file)

    with pytest.raises(ValueError, match="incompatible with chosen layout"):
        run_headless_workflow(workflow_path)


def test_timeseries_recording_equivalent_to_single_file_recording(tmp_path):
    """A workflow recorded from multiple files is identical to one recorded
    from a single time series file: both record the stacked (T-bearing)
    stack_layout, and the DAGs differ only in the recorded input_path."""
    first = write_still_tiff(tmp_path / "a1.tiff", (16, 16), 1)
    second = write_still_tiff(tmp_path / "a2.tiff", (16, 16), 2)
    series = tmp_path / "series.tiff"
    create_tiff(
        series,
        np.arange(2 * 16 * 16, dtype="float32").reshape(2, 16, 16),
        VoxelSize(),
        layout="TYX",
    )

    dags = {}
    for label, input_path in (
        ("multifile", (first, second)),
        ("single-file", series.with_suffix(".ome.tiff")),
    ):
        workflow_handler.clean_dag()
        ps_1 = import_image_task(
            input_path=input_path,
            key="raw",
            semantic_type="segmentation",
            stack_layout="TYX",
        )
        assert isinstance(ps_1, PanSegImage)
        export_image_task(
            image=ps_1,
            export_directory=tmp_path / f"output_{label}",
            name_pattern="{file_name}_export",
            scale_to_origin=True,
        )
        workflow_handler.save_to_yaml(tmp_path / f"workflow_{label}.yaml")
        with open(tmp_path / f"workflow_{label}.yaml", "r") as file:
            dags[label] = yaml.safe_load(file)

    tasks = {
        label: [(task["func"], task["parameters"]) for task in dag["list_tasks"]]
        for label, dag in dags.items()
    }
    assert tasks["multifile"] == tasks["single-file"]


def test_headless_timeseries_workflow_runs_on_dir_of_timeseries_files(tmp_path):
    """A time series workflow over a directory of T-bearing files runs one
    series per file: each single-file job matches the recorded layout
    exactly, exactly like a workflow recorded from one T-bearing file."""
    config = _timeseries_workflow(tmp_path)

    ts_dir = tmp_path / "series_files"
    ts_dir.mkdir()
    for stem, n_t in (("t1", 2), ("t2", 3)):
        create_tiff(
            ts_dir / f"{stem}.tiff",
            np.arange(n_t * 16 * 16, dtype="float32").reshape(n_t, 16, 16),
            VoxelSize(),
            layout="TYX",
        )

    output_dir = tmp_path / "out_ts_files"
    config["inputs"] = [
        {
            "input_path": str(ts_dir),
            "export_directory": str(output_dir),
            "name_pattern": "{file_name}_export",
        }
    ]
    workflow_path = tmp_path / "out_ts_files.yaml"
    with open(workflow_path, "w") as file:
        yaml.dump(config, file)

    run_headless_workflow(workflow_path)

    results = sorted(output_dir.glob("*_export.ome.tiff"))
    assert {path.name for path in results} == {
        "t1.ome_export.ome.tiff",
        "t2.ome_export.ome.tiff",
    }
    shapes = {}
    for path in results:
        with tifffile.TiffFile(path) as tiff:
            assert tiff.series[0].axes == "TYX"
            shapes[path.name] = tiff.series[0].shape[0]
    assert shapes == {"t1.ome_export.ome.tiff": 2, "t2.ome_export.ome.tiff": 3}
