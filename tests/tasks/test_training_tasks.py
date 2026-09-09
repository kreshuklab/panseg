"""Unit tests for training tasks."""

from pathlib import Path

import pytest

from panseg.tasks.training_tasks import unet_training_task


@pytest.fixture
def dataset_dir(tmp_path):
    directory = tmp_path / "dataset"
    (directory / "train").mkdir(parents=True)
    (directory / "val").mkdir()
    return directory


def test_unet_training_task_forwards_fair_metadata(dataset_dir, mocker):
    m_unet_training = mocker.patch("panseg.tasks.training_tasks.unet_training")

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
        authors=["Jane Doe <jane@example.com>"],
        additional_citations=["Smith, J. et al. Some result. doi:10.1234/x.y"],
        license="MIT",
        documentation="A very good model.",
    )

    m_unet_training.assert_called_once()
    call_kwargs = m_unet_training.call_args.kwargs
    assert call_kwargs["authors"] == ["Jane Doe <jane@example.com>"]
    assert call_kwargs["additional_citations"] == [
        "Smith, J. et al. Some result. doi:10.1234/x.y"
    ]
    assert call_kwargs["license"] == "MIT"
    assert call_kwargs["documentation"] == "A very good model."


def test_unet_training_task_defaults(dataset_dir, mocker):
    m_unet_training = mocker.patch("panseg.tasks.training_tasks.unet_training")

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
