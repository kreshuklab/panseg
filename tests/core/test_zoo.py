import logging
import os
from pathlib import Path

import pytest
import torch

import panseg.core.zoo as zoo_module
from panseg import FILE_BEST_MODEL_PYTORCH, FILE_CONFIG_TRAIN_YAML
from panseg.core.zoo import model_zoo
from panseg.functionals.training.model import UNet2D
from tests.conftest import IS_CUDA_AVAILABLE

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# test some modes (3D and 2D)
MODEL_NAMES = [
    "confocal_2D_unet_ovules_ds2x",
    "generic_confocal_3D_unet",
    "lightsheet_2D_unet_root_ds1x",
    "generic_light_sheet_3D_unet",
]


class TestPanSegModelZoo:
    """Test the PanSeg model zoo"""

    @pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA is not available")
    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS,
        reason="Github workflows do not allow model download for security reason",
    )
    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_model_output_normalisation(self, model_name):
        model, _, model_path = model_zoo.get_model_by_name(
            model_name, model_update=True
        )
        assert model_path.exists()
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        if isinstance(model, UNet2D):
            x = torch.randn(4, 1, 260, 260)
        else:
            x = torch.randn(4, 1, 80, 160, 160)
        y = model(x)
        # assert output normalized
        assert torch.all(0 <= y) and torch.all(y <= 1)


MODEL_IDS = [  # These two models has halo 44 on each side
    "efficient-chipmunk",  # Qin Yu's 3D nuclear segmentation model.
    "pioneering-rhino",  # Adrian's 2D cell-wall segmentation model.
]


class TestBioImageIOModelZoo:
    """Test the BioImage.IO model zoo"""

    model_zoo.refresh_bioimageio_zoo_urls()

    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_get_model_by_id(self, model_id):
        """Try to load a model from the BioImage.IO model zoo by ID."""
        model, _, model_path = model_zoo.get_model_by_id(model_id)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        if (
            "model_state_dict" in state
        ):  # Model weights format may vary between versions
            state = state["model_state_dict"]
        model.load_state_dict(state)

    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_halo_computation_for_bioimageio_model(self, model_id):
        """Compute the halo for a BioImage.IO model."""
        model, _, _ = model_zoo.get_model_by_id(model_id)
        halo = model_zoo.compute_halo(model)
        assert halo == 44


ZOO_MODEL_NAME = "confocal_2D_unet_ovules_ds2x"
ZOO_MODEL_URL = (
    "https://zenodo.org/record/7772709/files/confocal_2D_unet_ovules_ds2x.pytorch"
)


class TestCheckModelsVerification:
    """check_models must verify files on disk and fail loudly when missing."""

    def test_files_present_no_download_attempted(self, mocker, monkeypatch, tmp_path):
        model_dir = tmp_path / "generic_confocal_3D_unet"
        model_dir.mkdir()
        (model_dir / FILE_CONFIG_TRAIN_YAML).write_text("model: {}")
        (model_dir / FILE_BEST_MODEL_PYTORCH).write_bytes(b"weights")
        monkeypatch.setattr(zoo_module, "PATH_PANSEG_MODELS", tmp_path)
        spy = mocker.patch.object(model_zoo, "_download_model_files")

        model_zoo.check_models("generic_confocal_3D_unet")

        spy.assert_not_called()

    def test_download_succeeds_verified_and_logged(
        self, mocker, monkeypatch, tmp_path, caplog
    ):
        monkeypatch.setattr(zoo_module, "PATH_PANSEG_MODELS", tmp_path)

        def fake_download(model_url, out_dir, config_only=False):
            model_dir = Path(out_dir)
            (model_dir / FILE_CONFIG_TRAIN_YAML).write_text("model: {}")
            if not config_only:
                (model_dir / FILE_BEST_MODEL_PYTORCH).write_bytes(b"weights")

        mocker.patch.object(
            model_zoo, "_download_model_files", side_effect=fake_download
        )

        with caplog.at_level(logging.INFO, logger="panseg.core.zoo"):
            model_zoo.check_models(ZOO_MODEL_NAME)

        assert f"Download finished for {ZOO_MODEL_NAME}" in caplog.text

    def test_download_leaves_files_missing_raises_with_name_and_url(
        self, mocker, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(zoo_module, "PATH_PANSEG_MODELS", tmp_path)
        # no-op downloader: simulates a download that failed silently
        mocker.patch.object(model_zoo, "_download_model_files")

        with pytest.raises(FileNotFoundError) as excinfo:
            model_zoo.check_models(ZOO_MODEL_NAME)

        message = str(excinfo.value)
        assert ZOO_MODEL_NAME in message
        assert ZOO_MODEL_URL in message

    def test_config_only_skip_when_config_present(self, mocker, monkeypatch, tmp_path):
        model_dir = tmp_path / "generic_confocal_3D_unet"
        model_dir.mkdir()
        (model_dir / FILE_CONFIG_TRAIN_YAML).write_text("model: {}")
        monkeypatch.setattr(zoo_module, "PATH_PANSEG_MODELS", tmp_path)
        spy = mocker.patch.object(model_zoo, "_download_model_files")

        # only the config was requested and it is present: no download
        model_zoo.check_models("generic_confocal_3D_unet", config_only=True)

        spy.assert_not_called()
