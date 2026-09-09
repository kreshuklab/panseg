import sys
from pathlib import Path

import pytest

from panseg.__version__ import __version__
from panseg.run_panseg import create_parser, launch_napari, main

EXAMPLE_TRAIN_CONFIG = (
    Path(__file__).parent.parent / "panseg" / "resources" / "config_train_example.yaml"
)


def test_create_parser():
    parser = create_parser()
    assert len(parser._actions) == 8


def test_launch_napari(mocker):
    mock = mocker.patch("panseg.viewer_napari.viewer.Panseg_viewer")
    launch_napari()
    mock.assert_called_once()


def test_main_version(mocker):
    sys.argv = ["panseg", "-v"]
    mock = mocker.patch("panseg.run_panseg.print")
    with pytest.raises(SystemExit):
        main()
    mock.assert_called_with(__version__)


@pytest.fixture
def mock_check_version(mocker):
    return mocker.patch("panseg.run_panseg.check_version")


def test_launch_training_example_config(mock_check_version, mocker):
    m_unet_training = mocker.patch(
        "panseg.functionals.training.train.unet_training", autospec=True
    )
    sys.argv = ["panseg", "--train", str(EXAMPLE_TRAIN_CONFIG)]

    main()

    m_unet_training.assert_called_once()
    kwargs = m_unet_training.call_args.kwargs
    assert kwargs["dataset_dir"] == "TRAIN_DIR"
    assert kwargs["model_name"] == "custom_model"
    assert kwargs["in_channels"] == 1
    assert kwargs["out_channels"] == 2
    assert kwargs["feature_maps"] == [32, 64, 128, 256, 512]
    assert kwargs["patch_size"] == [80, 160, 160]
    assert kwargs["max_num_iters"] == 50000
    assert kwargs["dimensionality"] == "3D"
    assert kwargs["sparse"] is False
    assert kwargs["device"] == "cuda"


def test_launch_training_top_level_config(mock_check_version, mocker, tmp_path):
    m_unet_training = mocker.patch(
        "panseg.functionals.training.train.unet_training", autospec=True
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dataset_dir: some_dir\n"
        "model_name: my_model\n"
        "in_channels: 1\n"
        "out_channels: 1\n"
        "feature_maps: 32\n"
        "patch_size: [16, 64, 64]\n"
        "max_num_iters: 5\n"
        "dimensionality: 3D\n"
        "sparse: false\n"
        "device: cpu\n"
    )
    sys.argv = ["panseg", "--train", str(config_path)]

    main()

    m_unet_training.assert_called_once()
    assert m_unet_training.call_args.kwargs["model_name"] == "my_model"
    assert m_unet_training.call_args.kwargs["device"] == "cpu"


def test_launch_training_unknown_keys(mock_check_version, mocker, tmp_path):
    m_unet_training = mocker.patch(
        "panseg.functionals.training.train.unet_training", autospec=True
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "training:\n  model_name: my_model\n  not_a_param: 1\n  another_bad_key: 2\n"
    )
    sys.argv = ["panseg", "--train", str(config_path)]

    with pytest.raises(ValueError) as exc_info:
        main()

    message = str(exc_info.value)
    assert "not_a_param" in message
    assert "another_bad_key" in message
    m_unet_training.assert_not_called()
