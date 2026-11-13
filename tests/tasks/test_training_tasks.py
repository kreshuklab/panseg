"""Unit tests for training tasks."""

from panseg.tasks.training_tasks import unet_training_task
from panseg.tasks.workflow_handler import Task_message


def _make_dataset_dir(tmp_path):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "train").mkdir(parents=True)
    (dataset_dir / "val").mkdir()
    return dataset_dir


def _run_task(dataset_dir, **overrides):
    kwargs = {
        "dataset_dir": dataset_dir,
        "image": None,
        "segmentation": None,
        "model_name": "test_model",
        "in_channels": 1,
        "out_channels": 1,
        "feature_maps": 16,
        "patch_size": (16, 64, 64),
        "max_num_iters": 1,
        "dimensionality": "3D",
        "device": "cpu",
        "layer_order": "bcr",
    }
    kwargs.update(overrides)
    return unet_training_task(**kwargs)


def test_unet_training_task_defaults(tmp_path, mocker):
    m_unet_training = mocker.patch("panseg.tasks.training_tasks.unet_training")
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "train").mkdir(parents=True)
    (dataset_dir / "val").mkdir()

    unet_training_task(
        dataset_dir=dataset_dir,
        image=None,
        segmentation=None,
        model_name="test_model",
        in_channels=1,
        out_channels=1,
        feature_maps=16,
        patch_size=(16, 64, 64),
        max_num_iters=1,
        dimensionality="3D",
        device="cpu",
        layer_order="bcr",
    )

    m_unet_training.assert_called_once()
    call_kwargs = m_unet_training.call_args.kwargs
    assert call_kwargs["authors"] is None
    assert call_kwargs["additional_citations"] is None
    assert call_kwargs["license"] is None
    assert call_kwargs["documentation"] is None


def test_unet_training_task_reports_unexpected_runtime_error(tmp_path, mocker):
    """A failure like an OOM must not be reported as a finished training run."""
    m_unet_training = mocker.patch("panseg.tasks.training_tasks.unet_training")
    m_unet_training.side_effect = RuntimeError("CUDA out of memory.")
    m_log = mocker.patch("panseg.tasks.training_tasks.log")

    result = _run_task(_make_dataset_dir(tmp_path))

    # the task framework turns the re-raised error into a user-facing message
    assert isinstance(result, Task_message)
    assert "CUDA out of memory" in result.message
    m_log.assert_any_call(
        "Training failed: CUDA out of memory.",
        thread="train_gui_task",
        level="ERROR",
    )
    assert not any("Finished training" in str(call) for call in m_log.call_args_list)


def test_unet_training_task_reports_too_small_patch(tmp_path, mocker):
    m_unet_training = mocker.patch("panseg.tasks.training_tasks.unet_training")
    m_unet_training.side_effect = RuntimeError("Output size is too small")
    mocker.patch("panseg.tasks.training_tasks.log")

    result = _run_task(_make_dataset_dir(tmp_path))

    assert isinstance(result, Task_message)
    assert result.message == "Patch size too small for model architecture!"
