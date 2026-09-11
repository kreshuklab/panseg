"""Unit tests for training tasks."""

from panseg.tasks.training_tasks import unet_training_task


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
