import sys
from pathlib import Path

import pytest
import yaml

from panseg.__version__ import __version__
from panseg.functionals.precision import validate_amp
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
    assert kwargs["out_channels"] == 1
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


def test_launch_training_accepts_amp_key(mock_check_version, mocker, tmp_path):
    """`amp: auto|off` from the training YAML flows into `unet_training`."""
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
        "amp: off\n"
    )
    sys.argv = ["panseg", "--train", str(config_path)]

    main()

    m_unet_training.assert_called_once()
    # YAML parses a bare `off` as a boolean; it must resolve to the "off" mode
    assert validate_amp(m_unet_training.call_args.kwargs["amp"]) == "off"


_BASE_TRAIN_CONFIG = {
    "dataset_dir": "some_dir",
    "model_name": "my_model",
    "in_channels": 1,
    "out_channels": 1,
    "feature_maps": 32,
    "patch_size": [16, 64, 64],
    "max_num_iters": 5,
    "dimensionality": "3D",
    "sparse": False,
    "device": "cpu",
}


def _write_train_config(tmp_path: Path, **extra) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({**_BASE_TRAIN_CONFIG, **extra}))
    return config_path


def _launch(config_path: Path, m_unet_training) -> dict:
    sys.argv = ["panseg", "--train", str(config_path)]
    main()
    m_unet_training.assert_called_once()
    return m_unet_training.call_args.kwargs


def test_launch_training_translates_the_val_slice_builder_stride(
    mock_check_version, mocker, tmp_path
):
    """The `loaders.val.slice_builder` section written into `config_train.yml`
    is a valid input config: its `stride_shape` becomes `val_slice_builder`."""
    m_unet_training = mocker.patch(
        "panseg.functionals.training.train.unet_training", autospec=True
    )
    config_path = _write_train_config(
        tmp_path,
        loaders={
            "val": {
                "slice_builder": {
                    "name": "SliceBuilder",
                    "patch_shape": [16, 64, 64],
                    "stride_shape": [8, 32, 32],
                }
            }
        },
    )

    kwargs = _launch(config_path, m_unet_training)

    assert kwargs["val_slice_builder"] == {"stride_shape": [8, 32, 32]}
    # `loaders` is the written record, not a `unet_training` argument
    assert "loaders" not in kwargs


def test_launch_training_without_loaders_keeps_the_val_grid_default(
    mock_check_version, mocker, tmp_path
):
    """Omitting the section leaves `val_slice_builder` unset (non-overlapping)."""
    m_unet_training = mocker.patch(
        "panseg.functionals.training.train.unet_training", autospec=True
    )

    kwargs = _launch(_write_train_config(tmp_path), m_unet_training)

    assert "val_slice_builder" not in kwargs


def test_launch_training_rejects_two_val_stride_spellings(
    mock_check_version, mocker, tmp_path
):
    m_unet_training = mocker.patch(
        "panseg.functionals.training.train.unet_training", autospec=True
    )
    config_path = _write_train_config(
        tmp_path,
        val_slice_builder={"stride_shape": [4, 32, 32]},
        loaders={
            "val": {"slice_builder": {"stride_shape": [8, 32, 32]}},
        },
    )

    with pytest.raises(ValueError, match="val_slice_builder"):
        _launch(config_path, m_unet_training)

    m_unet_training.assert_not_called()


@pytest.mark.parametrize(
    "slice_builder, match",
    [
        # val patches are never filtered: a filter name would be a no-op
        ({"name": "FilterSliceBuilder", "stride_shape": [8, 32, 32]}, "SliceBuilder"),
        # val patches are always the training patch size
        ({"patch_shape": [8, 32, 32], "stride_shape": [8, 32, 32]}, "patch_size"),
        # val patches are never filtered: a threshold would be a no-op
        ({"stride_shape": [8, 32, 32], "threshold": 0.1}, "slice_builder"),
    ],
    ids=["filter-name", "patch-shape", "threshold"],
)
def test_launch_training_rejects_dead_val_slice_builder_keys(
    mock_check_version, mocker, tmp_path, slice_builder, match
):
    m_unet_training = mocker.patch(
        "panseg.functionals.training.train.unet_training", autospec=True
    )
    config_path = _write_train_config(
        tmp_path, loaders={"val": {"slice_builder": slice_builder}}
    )

    with pytest.raises(ValueError, match=match):
        _launch(config_path, m_unet_training)

    m_unet_training.assert_not_called()


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
