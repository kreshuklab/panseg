import logging
from pathlib import Path
from typing import Any, Literal

import yaml

from panseg.headless.basic_runner import SerialRunner
from panseg.io import allowed_data_format
from panseg.io.io import natural_sort_key
from panseg.tasks.workflow_handler import RunTimeInputSchema

logger = logging.getLogger(__name__)

Runners = Literal["serial"]

_implemented_runners = {"serial": SerialRunner}


def validate_config(config: dict):
    if "inputs" not in config:
        raise ValueError(
            "The workflow configuration does not contain an 'inputs' section."
        )

    if "infos" not in config:
        raise ValueError(
            "The workflow configuration does not contain an 'infos' section."
        )

    if "list_tasks" not in config:
        raise ValueError(
            "The workflow configuration does not contain an 'list_tasks' section."
        )

    if "runner" not in config:
        logger.warning(
            "The workflow configuration does not contain a 'runner' section. Using the default serial runner."
        )
        config["runner"] = "serial"

    return config


def parse_import_image_task(input_path, allow_dir: bool) -> list[Path | list[Path]]:
    if isinstance(input_path, str):
        input_path = Path(input_path)

    if isinstance(input_path, list):
        # one atomic job: the files are stacked as one time series. Directories
        # are not allowed inside a list - a directory of files is its own
        # multi-job input format.
        paths = [
            Path(entry) if isinstance(entry, str) else entry for entry in input_path
        ]
        for path in paths:
            if not path.exists():
                raise ValueError(f"File {path} does not exist.")
            if not path.is_file():
                raise ValueError(
                    f"{path} is not a file, a list input must contain files only."
                )
        return [sorted(paths, key=natural_sort_key)]

    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path} does not exist.")

    if input_path.is_file():
        if input_path.suffix.lower() not in allowed_data_format:
            raise ValueError(f"No valid files found in {input_path}.")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Path {input_path} is not a file or a directory.")

    if not allow_dir:
        raise ValueError(
            f"Directory {input_path} is not allowed when multiple input files are expected."
        )

    entries = list(input_path.glob("*"))
    # a .zarr dataset is a directory but counts as a file
    files: list[Path | list[Path]] = [
        entry for entry in entries if entry.suffix.lower() in allowed_data_format
    ]
    if files:
        # one job per file, as before
        return files

    # directory of subdirectories: one stacked job per subdirectory
    jobs: list[Path | list[Path]] = []
    for subdirectory in sorted(
        (
            entry
            for entry in entries
            if entry.is_dir() and entry.suffix.lower() not in allowed_data_format
        ),
        key=natural_sort_key,
    ):
        series_files = sorted(
            (
                entry
                for entry in subdirectory.glob("*")
                if entry.suffix.lower() in allowed_data_format
            ),
            key=natural_sort_key,
        )
        if not series_files:
            raise ValueError(f"No valid files found in {subdirectory}.")
        jobs.append(series_files)

    if not jobs:
        raise ValueError(f"No valid files found in {input_path}.")

    return jobs


def collect_jobs_list(
    inputs: dict | list[dict], inputs_schema: dict[str, RunTimeInputSchema]
) -> list[dict[str, Any]]:
    """
    Parse the inputs and create a list of jobs to run.
    """

    if isinstance(inputs, dict):
        inputs = [inputs]

    num_is_input_file = sum(
        [1 for schema in inputs_schema.values() if schema.is_input_file]
    )

    if num_is_input_file == 0:
        raise ValueError(
            "No input files found in the inputs schema. The workflow cannot run."
        )
    elif num_is_input_file > 1:
        allow_dir = False
    else:
        allow_dir = True

    all_jobs = []  # disconnected graphs go here
    # inputs: disconnected from the beginning, not by subdirectory
    for input_dict in inputs:
        if not isinstance(input_dict, dict):
            raise ValueError(f"Input {input_dict} should be a dictionary.")

        inputs_files = {}
        for name, schema in inputs_schema.items():
            if schema.is_input_file:
                if name not in inputs_files:
                    inputs_files[name] = []

                inputs_files[name].extend(
                    parse_import_image_task(input_dict[name], allow_dir=allow_dir)
                )

        list_len = [len(files) for files in inputs_files.values()]
        if len(set(list_len)) != 1:
            raise ValueError(
                f"Inputs have different number of files. found {inputs_files}"
            )

        list_files = list(zip(*inputs_files.values()))
        list_keys = list(inputs_files.keys())
        list_jobs = [dict(zip(list_keys, files)) for files in list_files]

        # each job = one subdirectory or input file
        for job in list_jobs:
            for key, value in input_dict.items():
                if key not in job:
                    job[key] = value

            # avoid name collision from subdirs
            if (
                len(list_jobs) > 1
                and isinstance(job[list_keys[0]], list)
                and "name_pattern" in job
            ):
                subdirs = ""
                for input_key in list_keys:
                    subdir = job[input_key][0].parent.name
                    subdirs = subdirs + subdir
                npat: str = job["name_pattern"]
                job["name_pattern"] = npat.replace(
                    "{file_name}", subdirs + "_{file_name}"
                )

            all_jobs.append(job)

    return all_jobs


def run_headles_workflow_from_config(config: dict, path: str | Path):
    config = validate_config(config)

    inputs = config["inputs"]
    inputs_schema = config["infos"]["inputs_schema"]
    inputs_schema = {k: RunTimeInputSchema(**v) for k, v in inputs_schema.items()}

    jobs_list = collect_jobs_list(inputs, inputs_schema)

    runner = config.get("runner")
    if runner not in _implemented_runners:
        raise ValueError(f"Runner {runner} is not implemented.")

    runner = _implemented_runners[runner](path)

    for job_input in jobs_list:
        logger.info(f"Submitting job with input: {job_input}")
        runner.submit_job(job_input)

    logger.info("All jobs have been submitted. Running the workflow...")


def run_headless_workflow(path: str | Path):
    """
    Run a workflow in headless mode using the provided configuration file.

    Args:
        path (str | Path): Path to the workflow configuration file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Workflow configuration file {path} does not exist.")

    with path.open("r") as file:
        config = yaml.safe_load(file)

    run_headles_workflow_from_config(config, path=path)
