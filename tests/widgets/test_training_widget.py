from pathlib import Path

import pytest

from panseg.functionals.training.model import UNet2D, UNet3D
from panseg.functionals.training.train import find_h5_files
from panseg.viewer_napari.widgets.training import Training_Tab


@pytest.fixture
def training_tab():
    return Training_Tab(None)


def test_get_container(training_tab):
    container = training_tab.get_container()
    assert len(container) == 3


def test_unet_training_run(training_tab, mocker, tmp_path):
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "panseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("panseg.viewer_napari.widgets.training.schedule_task")
    m_model_path = mocker.patch(
        "panseg.viewer_napari.widgets.training.PATH_PANSEG_MODELS", new=tmp_path
    )

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )
    m_log.assert_called_with("Starting training task", thread="train_gui")
    m_get_models.assert_not_called()
    m_schedule.assert_called_once()


def test_unet_training_no_dataset(training_tab, mocker):
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "panseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("panseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset=None,
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )
    m_log.assert_called_with("Please choose a dataset to load!", thread="train_gui")
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()


def test_unet_training_no_image(
    training_tab,
    mocker,
    napari_segmentation,
    napari_raw,
):
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "panseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("panseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Current Data",
        dataset=None,
        image=None,
        segmentation=napari_segmentation,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )
    m_log.assert_called_with(
        "Please choose a raw image and a segmentation to train!", thread="train_gui"
    )
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()

    training_tab.widget_unet_training(
        from_disk="Current Data",
        dataset=None,
        image=napari_raw,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )
    m_log.assert_called_with(
        "Please choose a raw image and a segmentation to train!", thread="train_gui"
    )
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()


def test_unet_training_channels(
    training_tab,
    mocker,
    napari_segmentation,
    napari_raw,
):
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "panseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("panseg.viewer_napari.widgets.training.schedule_task")

    m_additional_inputs = [mocker.Mock() for _ in range(3)]
    for m in m_additional_inputs:
        m.value = napari_raw
    training_tab.additional_inputs = m_additional_inputs
    training_tab.widget_unet_training(
        from_disk="Current Data",
        dataset=None,
        image=napari_raw,
        segmentation=napari_segmentation,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )
    m_log.assert_called_with("Starting training task", thread="train_gui")
    m_get_models.assert_not_called()
    m_schedule.assert_called_once()
    assert m_schedule.call_args.kwargs["task_kwargs"]["image"] == [napari_raw] * 4


def test_unet_training_none(training_tab, mocker):
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "panseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("panseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality=None,  # <-- error
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()

    m_log.reset_mock()
    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type=None,  # <-- error
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()


def test_unet_training_custom(training_tab, mocker):
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "panseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("panseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality=training_tab.CUSTOM,  # <-- custom
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()

    m_log.reset_mock()
    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type=training_tab.CUSTOM,
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()


def test_unet_training_no_name(training_tab, mocker):
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "panseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("panseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="",  # <-- error
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()


def test_unet_training_feature_maps(training_tab, mocker, tmp_path):
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "panseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("panseg.viewer_napari.widgets.training.schedule_task")
    m_model_path = mocker.patch(
        "panseg.viewer_napari.widgets.training.PATH_PANSEG_MODELS", new=tmp_path
    )

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality="boundaries",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_called_once()
    assert isinstance(m_schedule.call_args[1]["task_kwargs"]["feature_maps"], int)

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16, 16, 16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality="boundaries",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_schedule.assert_called()
    assert isinstance(m_schedule.call_args[1]["task_kwargs"]["feature_maps"], list)


def test_unet_training_pretrained(training_tab, mocker, tmp_path):
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "panseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_get_models.return_value = [1, 2, mocker.sentinel]
    m_schedule = mocker.patch("panseg.viewer_napari.widgets.training.schedule_task")
    m_model_path = mocker.patch(
        "panseg.viewer_napari.widgets.training.PATH_PANSEG_MODELS", new=tmp_path
    )

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained="SOMETHING",  # <--
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        device="cpu",
        modality="boundaries",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_called_with("SOMETHING")
    m_schedule.assert_called_once()
    assert m_schedule.call_args[1]["task_kwargs"]["pre_trained"] == mocker.sentinel


def test_on_from_disk_change(training_tab, mocker):
    training_tab.widget_unet_training.from_disk.value = "Disk"
    m_show_image = mocker.patch.object(training_tab.widget_unet_training.image, "show")
    m_show_seg = mocker.patch.object(
        training_tab.widget_unet_training.segmentation, "show"
    )
    m_show_dataset = mocker.patch.object(
        training_tab.widget_unet_training.dataset, "show"
    )
    m_hide_image = mocker.patch.object(training_tab.widget_unet_training.image, "hide")
    m_hide_seg = mocker.patch.object(
        training_tab.widget_unet_training.segmentation, "hide"
    )
    m_hide_dataset = mocker.patch.object(
        training_tab.widget_unet_training.dataset, "hide"
    )
    training_tab.widget_unet_training.from_disk.value = "Current Data"

    m_show_image.assert_called_once()
    m_show_seg.assert_called_once()
    m_show_dataset.assert_not_called()

    m_hide_image.assert_not_called()
    m_hide_seg.assert_not_called()
    m_hide_dataset.assert_called_once()


def test_on_dimensionality_change(training_tab):
    training_tab.widget_unet_training.patch_size.value = [9, 8, 7]
    training_tab.widget_unet_training.dimensionality.value = "2D"
    assert training_tab.previous_z_patch_size == 9
    assert training_tab.widget_unet_training.patch_size.value == (1, 8, 7)

    training_tab.widget_unet_training.dimensionality.value = "3D"
    assert (
        training_tab.widget_unet_training.patch_size[0].value
        == training_tab.previous_z_patch_size
    )


def test_on_custom_modality_change(training_tab, mocker):
    m_show = mocker.patch.object(
        training_tab.widget_unet_training.custom_modality, "show"
    )
    m_hide = mocker.patch.object(
        training_tab.widget_unet_training.custom_modality, "hide"
    )

    training_tab.widget_unet_training.modality.value = training_tab.CUSTOM
    m_show.assert_called_once()
    m_hide.assert_not_called()

    training_tab.widget_unet_training.modality.value = "confocal"
    m_hide.assert_called_once()
    m_show.assert_called_once()


def test_on_custom_output_type_change(training_tab, mocker):
    m_show = mocker.patch.object(
        training_tab.widget_unet_training.custom_output_type, "show"
    )
    m_hide = mocker.patch.object(
        training_tab.widget_unet_training.custom_output_type, "hide"
    )

    training_tab.widget_unet_training.output_type.value = training_tab.CUSTOM
    m_show.assert_called_once()
    m_hide.assert_not_called()

    training_tab.widget_unet_training.output_type.value = "boundaries"
    m_hide.assert_called_once()
    m_show.assert_called_once()


def test_on_dataset_change(training_tab, mocker):
    m_finder = mocker.patch(
        "panseg.viewer_napari.widgets.training.find_h5_files",
        wraps=find_h5_files,
    )

    test_data_dir = Path(__file__).parent.parent / "resources" / "data"
    assert test_data_dir.exists(), f"Test data directory not found: {test_data_dir}"

    training_tab.widget_unet_training.dataset.value = test_data_dir / "Noneexistent"
    m_finder.assert_not_called()

    training_tab.widget_unet_training.dataset.value = test_data_dir.parent
    m_finder.assert_not_called()

    training_tab.widget_unet_training.dataset.value = test_data_dir
    m_finder.assert_called_once()
    assert training_tab.widget_unet_training.resolution.value == (0.28, 0.13, 0.13)


def test_on_pretrained_changed(training_tab, mocker):
    m_zoo = mocker.patch("panseg.viewer_napari.widgets.training.model_zoo")
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_zoo.get_model_by_name.return_value = (
        mocker.Mock(spec=UNet3D),
        {"f_maps": [16], "in_channels": 1, "out_channels": 1},
        "path",
    )

    training_tab.widget_unet_training.pretrained.value = None
    m_zoo.get_model_by_name.assert_not_called()
    m_zoo.get_model_description.assert_not_called()

    training_tab.widget_unet_training.pretrained.choices = ["my_model"]
    training_tab.widget_unet_training.pretrained.value = "my_model"
    m_zoo.get_model_by_name.assert_called_with("my_model")
    m_zoo.get_model_description.assert_called_with("my_model")
    m_log.assert_not_called()


def test_on_pretrained_changed_channels(training_tab, mocker):
    m_zoo = mocker.patch("panseg.viewer_napari.widgets.training.model_zoo")
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_zoo.get_model_by_name.return_value = (
        mocker.Mock(spec=UNet3D),
        {"f_maps": [16], "in_channels": 2, "out_channels": 1},
        "path",
    )

    training_tab.widget_unet_training.pretrained.value = None
    m_zoo.get_model_by_name.assert_not_called()
    m_zoo.get_model_description.assert_not_called()

    training_tab.widget_unet_training.pretrained.choices = ["my_model"]
    training_tab.widget_unet_training.pretrained.value = "my_model"
    m_zoo.get_model_by_name.assert_called_with("my_model")
    m_zoo.get_model_description.assert_called_with("my_model")
    m_log.assert_called_with(
        "Model incompatible chosen data!\nModel channels: "
        "(2, 1)\nData channels: "
        "(1, 1)",
        thread="training",
        level="ERROR",
    )


def test_on_pretrained_changed_wrong_2Dmodel(training_tab, mocker):
    m_zoo = mocker.patch("panseg.viewer_napari.widgets.training.model_zoo")
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_zoo.get_model_by_name.return_value = (
        mocker.Mock(spec=UNet2D),
        {"f_maps": [16], "in_channels": 1, "out_channels": 1},
        "path",
    )

    training_tab.widget_unet_training.pretrained.value = None
    m_zoo.get_model_by_name.assert_not_called()
    m_zoo.get_model_description.assert_not_called()

    training_tab.widget_unet_training.pretrained.choices = ["my_model"]
    training_tab.widget_unet_training.pretrained.value = "my_model"
    m_zoo.get_model_by_name.assert_called_with("my_model")
    m_zoo.get_model_description.assert_called_with("my_model")
    m_log.assert_called_with(
        "Model incompatible with chosen data!\nModel is a 2D model, but data is 3D",
        thread="training",
        level="ERROR",
    )


def test_on_pretrained_changed_wrong_3Dmodel(training_tab, mocker):
    m_zoo = mocker.patch("panseg.viewer_napari.widgets.training.model_zoo")
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    m_zoo.get_model_by_name.return_value = (
        mocker.Mock(spec=UNet3D),
        {"f_maps": [16], "in_channels": 1, "out_channels": 1},
        "path",
    )
    training_tab.widget_unet_training.dimensionality.value = "2D"

    training_tab.widget_unet_training.pretrained.value = None
    m_zoo.get_model_by_name.assert_not_called()
    m_zoo.get_model_description.assert_not_called()

    training_tab.widget_unet_training.pretrained.choices = ["my_model"]
    training_tab.widget_unet_training.pretrained.value = "my_model"
    m_zoo.get_model_by_name.assert_called_with("my_model")
    m_zoo.get_model_description.assert_called_with("my_model")
    m_log.assert_called_with(
        "Model incompatible with chosen data!\nModel is a 3D model, but data is 2D",
        thread="training",
        level="ERROR",
    )


def test_update_dimensionality(training_tab):
    # YX, ?
    training_tab.in_shape = (1, 1)
    training_tab.out_shape = (1, 1)
    training_tab.update_dimensionality()
    assert training_tab.widget_unet_training.dimensionality.value == "2D"

    # CYX, YX
    training_tab.in_shape = (1, 1, 1)
    training_tab.out_shape = (1, 1)
    training_tab.update_dimensionality()
    assert training_tab.widget_unet_training.dimensionality.value == "2D"

    # ZYX, ZYX
    training_tab.in_shape = (1, 1, 1)
    training_tab.out_shape = (1, 1, 1)
    training_tab.update_dimensionality()
    assert training_tab.widget_unet_training.dimensionality.value == "3D"

    # ZYX, CZYX
    training_tab.in_shape = (1, 1, 1)
    training_tab.out_shape = (1, 1, 1, 1)
    training_tab.update_dimensionality()
    assert training_tab.widget_unet_training.dimensionality.value == "3D"

    # CZYX, ?
    training_tab.in_shape = (1, 1, 1, 1)
    training_tab.out_shape = (1, 1, 1)
    training_tab.update_dimensionality()
    assert training_tab.widget_unet_training.dimensionality.value == "3D"


def test_update_dimensionality_error(training_tab):
    # YX, CZYX
    training_tab.in_shape = (1, 1)
    training_tab.out_shape = (1, 1, 1, 1)
    with pytest.raises(ValueError):
        training_tab.update_dimensionality()

    # CZYX, YX
    training_tab.in_shape = (1, 1, 1, 1)
    training_tab.out_shape = (1, 1)
    with pytest.raises(ValueError):
        training_tab.update_dimensionality()

    # X, YX
    training_tab.in_shape = (1,)
    training_tab.out_shape = (1, 1)
    with pytest.raises(ValueError):
        training_tab.update_dimensionality()

    # ZYX, X
    training_tab.in_shape = (1, 1, 1)
    training_tab.out_shape = (1,)
    with pytest.raises(ValueError):
        training_tab.update_dimensionality()


def test_update_channels(training_tab):
    ch = training_tab.widget_unet_training.channels
    # YX, YX
    training_tab.in_shape = (2, 2)
    training_tab.out_shape = (2, 2)
    training_tab.update_channels()
    assert ch.value == (1, 1)

    # CYX, YX
    training_tab.in_shape = (2, 2, 2)
    training_tab.out_shape = (2, 2)
    training_tab.update_channels()
    assert ch.value == (2, 1)

    # ZYX, ZYX
    training_tab.in_shape = (2, 2, 2)
    training_tab.out_shape = (2, 2, 2)
    training_tab.update_channels()
    assert ch.value == (1, 1)

    # CZYX, ZYX
    training_tab.in_shape = (2, 2, 2, 2)
    training_tab.out_shape = (2, 2, 2)
    training_tab.update_channels()
    assert ch.value == (2, 1)


def test_update_channels_error(training_tab, mocker):
    m_log = mocker.patch("panseg.viewer_napari.widgets.training.log")
    ch = training_tab.widget_unet_training.channels

    # YX, ZYX
    training_tab.in_shape = (2, 2)
    training_tab.out_shape = (2, 2, 2)
    training_tab.update_channels()
    assert ch.value == (1, 2)
    m_log.assert_called_once()
    m_log.reset_mock()

    # YX, CZYX
    training_tab.in_shape = (2, 2)
    training_tab.out_shape = (2, 2, 2, 2)
    training_tab.update_channels()
    m_log.assert_called_once()
    m_log.reset_mock()

    # ZYX, CYX
    training_tab.in_shape = (2, 2, 2)
    training_tab.out_shape = (2, 2, 2)
    training_tab.widget_unet_training.dimensionality.value = "2D"
    m_log.assert_called_once()
    m_log.reset_mock()
    training_tab.update_channels()
    assert ch.value == (2, 2)
    m_log.assert_called_once()
    m_log.reset_mock()

    # ZYX, CZYX
    training_tab.in_shape = (2, 2, 2)
    training_tab.out_shape = (2, 2, 2, 2)
    training_tab.update_channels()
    assert ch.value == (1, 2)
    m_log.assert_called_once()
    m_log.reset_mock()

    # CZYX, YX
    training_tab.in_shape = (2, 2, 2, 2)
    training_tab.out_shape = (2, 2)
    training_tab.update_channels()
    m_log.assert_called_once()
    m_log.reset_mock()

    # CZYX, CZYX
    training_tab.in_shape = (2, 2, 2, 2)
    training_tab.out_shape = (2, 2, 2, 2)
    training_tab.update_channels()
    assert ch.value == (2, 2)
    m_log.assert_called_once()
    m_log.reset_mock()

    # X, CZYX
    training_tab.in_shape = (2,)
    training_tab.out_shape = (2, 2, 2, 2)
    training_tab.update_channels()
    m_log.assert_called_once()
    m_log.reset_mock()


def test_update_layer_selection(
    training_tab,
    mocker,
    napari_raw,
    napari_segmentation,
    make_napari_viewer_proxy,
):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)

    sentinel = mocker.sentinel
    sentinel.value = napari_raw
    sentinel.type = "inserted"
    assert training_tab.widget_unet_training.image.value is None
    assert training_tab.widget_unet_training.segmentation.value is None

    training_tab.update_layer_selection(sentinel)
    assert napari_raw in training_tab.widget_unet_training.image.choices
    assert training_tab.widget_unet_training.segmentation.value is None

    viewer.add_layer(napari_segmentation)
    sentinel.value = napari_segmentation
    sentinel.type = "inserted"
    training_tab.update_layer_selection(sentinel)
    assert napari_segmentation in training_tab.widget_unet_training.segmentation.choices


def test_on_image_change(training_tab, mocker, napari_raw):
    m_update_dimensionality = mocker.patch.object(training_tab, "update_dimensionality")
    training_tab._on_image_change(napari_raw)
    m_update_dimensionality.assert_called_once()


def test_on_segmentation_change(training_tab, mocker, napari_segmentation):
    m_update_dimensionality = mocker.patch.object(training_tab, "update_dimensionality")
    training_tab._on_image_change(napari_segmentation)
    m_update_dimensionality.assert_called_once()
