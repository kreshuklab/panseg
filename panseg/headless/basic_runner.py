import logging
from pathlib import Path

import yaml

from panseg.core.image import PanSegImage
from panseg.tasks.workflow_handler import DAG, Task, WorkflowHandler

logger = logging.getLogger(__name__)


class SerialRunner:
    """
    SerialRunner is a class that runs a workflow in a single thread.
    """

    def __init__(self, dag_path: str | Path):
        if isinstance(dag_path, str):
            dag_path = Path(dag_path)

        self.dag_path = dag_path

        if not dag_path.exists():
            raise FileNotFoundError(f"File {dag_path} not found")

        self.func_registry = WorkflowHandler().func_registry
        logger.info(self.func_registry.list_funcs())

    def find_next_task(self, dag: DAG, var_set: set[str]):
        """Return the next task to run based on the current var_set"""
        for task in dag.list_tasks:
            required_inputs = set(task.images_inputs.values())
            if required_inputs.issubset(var_set):
                dag.list_tasks.remove(task)
                return task
        return None

    def run_task(self, task: Task, var_space: dict):
        """Run a task and update the var_space"""
        # Get inputs from var_space
        inputs = {}
        for name, image_name in task.images_inputs.items():
            inputs[name] = var_space[image_name]

        # run the task
        func = self.func_registry.get_func(task.func)
        outputs = func(**inputs, **task.parameters)

        if isinstance(outputs, PanSegImage):
            outputs = [outputs]

        elif outputs is None:
            outputs = []

        assert isinstance(outputs, (list, tuple)), (
            f"Task {task.func} should return a list of PanSegImage, got {type(outputs)}"
        )
        assert len(outputs) == len(task.outputs), (
            f"Task {task.func} should return {len(task.outputs)} outputs, got {len(outputs)}"
        )

        for name, output in zip(task.outputs, outputs, strict=True):
            var_space[name] = output

        return var_space

    def clean_var_space(self, dag: DAG, var_space: dict):
        all_remaining_required_inputs = set()
        for task in dag.list_tasks:
            required_inputs = set(task.images_inputs.values())
            all_remaining_required_inputs = all_remaining_required_inputs.union(
                required_inputs
            )

        list_key_to_delete = []
        for var in var_space.keys():
            if var not in all_remaining_required_inputs:
                list_key_to_delete.append(var)

        for key in list_key_to_delete:
            del var_space[key]
        return var_space

    def _parse_input(self, inputs: dict[str, str] | list[dict[str, str]]) -> list[dict]:
        if isinstance(inputs, dict):
            inputs = [inputs]

        return inputs

    def _load_dag(self) -> DAG:
        with self.dag_path.open("r") as f:
            config = yaml.safe_load(f)

        # A headless configuration may list multiple jobs under "inputs";
        # the DAG only carries the task graph and the input schema. The job
        # inputs are passed to submit_job.
        config.pop("inputs", None)

        return DAG(**config)

    def submit_job(self, inputs: dict[str, str]):
        """Submit a job to the runner

        Args:
            inputs (dict): A dictionary containing the input variables for the workflow

        Returns:
            bool: True if the job has been submitted successfully
        """
        dag = self._load_dag()

        var_space = {}
        for key in dag.list_inputs:
            if key not in inputs:
                raise ValueError(f"Missing input variable {key}")
            var_space[key] = inputs[key]

        while dag.list_tasks:
            # Find next task to run
            next_task = self.find_next_task(dag, set(var_space.keys()))
            if next_task is None:
                raise ValueError(
                    "No task to run next, the computation graph might be corrupted"
                )

            # Run the task
            var_space = self.run_task(next_task, var_space)

            # Remove from var_space the variables that are not needed anymore
            var_space = self.clean_var_space(dag, var_space)

        if var_space:
            raise ValueError(
                "Some variables are still in the var_space, the computation graph might be corrupted"
            )
        return True
