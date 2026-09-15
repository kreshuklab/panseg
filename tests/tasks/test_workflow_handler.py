from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from panseg.tasks.workflow_handler import (
    DAG,
    NodeType,
    RunTimeInputSchema,
    Task,
    WorkflowHandler,
    prune_dag,
)


def make_leaf_dag(inputs: dict, extra_inputs: dict | None = None) -> DAG:
    dag = DAG()
    dag.inputs = {**inputs, **(extra_inputs or {})}
    for key in dag.inputs:
        dag.infos.inputs_schema[key] = RunTimeInputSchema(
            description=f"Description of {key}"
        )
    dag.list_tasks.append(
        Task(
            func="export_image_task",
            images_inputs={
                "image": "some_image",
                **{key: key for key in inputs},
            },
            parameters={},
            outputs=[],
            node_type=NodeType.LEAF,
        )
    )
    return dag


def write_workflow_yaml(path: Path, inputs) -> Path:
    config = {
        "infos": {
            "inputs_schema": {
                key: {"description": f"Description of {key}"}
                for key in (inputs[0] if isinstance(inputs, list) else inputs)
            }
        },
        "inputs": inputs,
        "list_tasks": [],
    }
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


def test_dag_inputs_default_is_empty_dict():
    assert DAG().inputs == {}


def test_dag_inputs_accepts_plain_dict():
    dag = DAG(inputs={"input_path": "image.tiff"})
    assert dag.inputs == {"input_path": "image.tiff"}


def test_dag_inputs_converts_legacy_single_entry_list():
    dag = DAG(inputs=[{"input_path": "image.tiff", "export_directory": "/tmp"}])
    assert dag.inputs == {"input_path": "image.tiff", "export_directory": "/tmp"}


def test_dag_inputs_converts_legacy_empty_list():
    assert DAG(inputs=[]).inputs == {}


def test_dag_inputs_rejects_multi_entry_list():
    with pytest.raises(ValidationError, match="single workflow"):
        DAG(inputs=[{"input_path": "a.tiff"}, {"input_path": "b.tiff"}])


def test_prune_dag_keeps_reachable_inputs():
    dag = make_leaf_dag(
        {"export_directory": "/tmp", "name_pattern": "{file_name}_export"},
        extra_inputs={"unconnected_input": "value"},
    )

    pruned = prune_dag(dag)

    assert pruned.inputs == {
        "export_directory": "/tmp",
        "name_pattern": "{file_name}_export",
    }
    assert "unconnected_input" not in pruned.infos.inputs_schema


def test_from_yaml_accepts_legacy_list_inputs(tmp_path):
    path = write_workflow_yaml(
        tmp_path / "legacy.yaml",
        [{"input_path": "image.tiff", "export_directory": "/tmp"}],
    )

    dag = WorkflowHandler().from_yaml(path)._dag

    assert dag.inputs == {"input_path": "image.tiff", "export_directory": "/tmp"}


def test_from_yaml_accepts_dict_inputs(tmp_path):
    path = write_workflow_yaml(
        tmp_path / "current.yaml",
        {"input_path": "image.tiff", "export_directory": "/tmp"},
    )

    dag = WorkflowHandler().from_yaml(path)._dag

    assert dag.inputs == {"input_path": "image.tiff", "export_directory": "/tmp"}


def test_from_yaml_rejects_batch_list_inputs(tmp_path):
    path = write_workflow_yaml(
        tmp_path / "batch.yaml",
        [
            {"input_path": "a.tiff", "export_directory": "/tmp"},
            {"input_path": "b.tiff", "export_directory": "/tmp"},
        ],
    )

    with pytest.raises(ValidationError, match="batch configuration"):
        WorkflowHandler().from_yaml(path)


def test_save_to_yaml_writes_dict_inputs(tmp_path):
    handler = WorkflowHandler()
    handler._dag = make_leaf_dag({"export_directory": "/tmp"})

    path = tmp_path / "workflow.yaml"
    handler.save_to_yaml(path)

    with open(path) as f:
        config = yaml.safe_load(f)

    assert isinstance(config["inputs"], dict)
    assert config["inputs"] == {"export_directory": "/tmp"}
