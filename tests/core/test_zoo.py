import logging
import os
import time
from pathlib import Path

import pytest
import requests
import torch

import panseg.core.zoo as zoo_module
from panseg import FILE_BEST_MODEL_PYTORCH, FILE_CONFIG_TRAIN_YAML
from panseg.core.zoo import Author, model_zoo, resolve_bioimageio_source
from panseg.functionals.model_geometry import model_geometry
from panseg.functionals.training.model import UNet2D, UNet3D
from tests.conftest import HYPHA_MODEL_LIST, IS_CUDA_AVAILABLE, MOCK_UNET_KWARGS

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
        # forward() returns logits; the normalized output comes from predict()
        y = model.predict(x)
        # assert output normalized
        assert torch.all(0 <= y) and torch.all(y <= 1)


MODEL_IDS = [  # These two models has halo 44 on each side
    "efficient-chipmunk",  # Qin Yu's 3D nuclear segmentation model.
    "pioneering-rhino",  # Adrian's 2D cell-wall segmentation model.
]


@pytest.fixture(scope="session")
def biio_model(tmp_path_factory):
    """Load each BioImage.IO model by id at most once per test session.

    `get_model_by_id` returns the weights as an in-memory reader that can
    only be consumed once, so the state dict is materialized to a file for
    reuse across tests.
    """
    cache = {}
    weights_dir = tmp_path_factory.mktemp("biio_models")

    def _load(model_id: str):
        if model_id not in cache:
            model, _, model_path = model_zoo.get_model_by_id(model_id)
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            state_path = weights_dir / f"{model_id}.pytorch"
            torch.save(state, state_path)
            cache[model_id] = (model, state_path)
        return cache[model_id]

    return _load


class TestBioImageIOModelZoo:
    """Test the BioImage.IO model zoo"""

    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_get_model_by_id(self, model_id, biio_model):
        """Try to load a model from the BioImage.IO model zoo by ID."""
        model, state_path = biio_model(model_id)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if (
            "model_state_dict" in state
        ):  # Model weights format may vary between versions
            state = state["model_state_dict"]
        model.load_state_dict(state)

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("efficient-chipmunk", (48, 48, 48)),
            # 2D models predict one Z slice at a time
            ("pioneering-rhino", (0, 48, 48)),
        ],
    )
    def test_halo_computation_for_bioimageio_model(
        self, model_id, expected, biio_model
    ):
        """Compute the halo for a BioImage.IO model."""
        model, _ = biio_model(model_id)
        assert model_geometry(model).halo == expected


@pytest.mark.parametrize(
    "mock_biio_package", [False, True], ids=["zip", "legacy"], indirect=True
)
def test_get_model_by_id_resolves_bioimageio_id(mock_biio_package, mocker):
    """A bioimage.io ID is resolved on its own, without the full model list, and
    the weights load whichever format PyTorch saved them in."""
    resolve = mocker.patch.object(
        zoo_module, "bioimageio_rdf_url", return_value=str(mock_biio_package)
    )
    refresh = mocker.spy(model_zoo, "refresh_bioimageio_zoo_urls")

    model, model_config, weights = model_zoo.get_model_by_id("mock-unet3d")

    resolve.assert_called_once_with("mock-unet3d")
    refresh.assert_not_called()
    assert isinstance(model, UNet3D)
    assert model_config["f_maps"] == MOCK_UNET_KWARGS["f_maps"]
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))


def _panseg_model_ids() -> list[str]:
    return [
        model_id for _, model_id in model_zoo.get_bioimageio_zoo_panseg_model_names()
    ]


@pytest.mark.parametrize("paginated", [False, True], ids=["list", "page"])
def test_refresh_bioimageio_zoo_reads_hypha_model_list(
    paginated, offline_bioimageio_zoo
):
    if paginated:
        offline_bioimageio_zoo.return_value = {
            "items": HYPHA_MODEL_LIST,
            "total": len(HYPHA_MODEL_LIST),
            "offset": 0,
            "limit": 10,
        }

    model_zoo.refresh_bioimageio_zoo_urls()

    assert model_zoo.models_bioimageio["supported"].to_dict() == {
        "efficient-chipmunk": True,
        "affable-shark": False,
        "hiding-tiger": False,
        "odd-tags": False,
    }
    assert _panseg_model_ids() == ["efficient-chipmunk"]


def test_refresh_bioimageio_zoo_caches_list_for_a_day(offline_bioimageio_zoo):
    model_zoo.refresh_bioimageio_zoo_urls()
    model_zoo.refresh_bioimageio_zoo_urls()
    assert offline_bioimageio_zoo.call_count == 1

    day_old = time.time() - zoo_module.BIOIMAGE_IO_LIST_MAX_AGE - 60
    os.utime(zoo_module.PATH_BIOIMAGE_IO_MODEL_LIST, (day_old, day_old))
    model_zoo.refresh_bioimageio_zoo_urls()
    assert offline_bioimageio_zoo.call_count == 2

    model_zoo.refresh_bioimageio_zoo_urls(force=True)
    assert offline_bioimageio_zoo.call_count == 3


def test_refresh_bioimageio_zoo_falls_back_to_cached_list(
    offline_bioimageio_zoo, caplog
):
    model_zoo.refresh_bioimageio_zoo_urls()
    offline_bioimageio_zoo.side_effect = requests.ConnectionError("offline")

    with caplog.at_level(logging.WARNING, logger="panseg.core.zoo"):
        model_zoo.refresh_bioimageio_zoo_urls(force=True)

    assert "using the copy from" in caplog.text
    assert _panseg_model_ids() == ["efficient-chipmunk"]


def test_refresh_bioimageio_zoo_without_network_or_cache(
    offline_bioimageio_zoo, caplog
):
    offline_bioimageio_zoo.side_effect = requests.ConnectionError("offline")

    with caplog.at_level(logging.WARNING, logger="panseg.core.zoo"):
        model_zoo.refresh_bioimageio_zoo_urls()

    assert "no BioImage.IO models are available" in caplog.text
    assert not model_zoo.has_bioimageio_models()
    assert model_zoo.get_bioimageio_zoo_panseg_model_names() == []
    assert model_zoo.get_bioimageio_zoo_other_model_names() == []


def test_refresh_bioimageio_zoo_uses_list_it_cannot_cache(
    offline_bioimageio_zoo, mocker, caplog
):
    mocker.patch.object(
        zoo_module, "write_text_atomic", side_effect=OSError("No space left")
    )

    with caplog.at_level(logging.WARNING, logger="panseg.core.zoo"):
        model_zoo.refresh_bioimageio_zoo_urls()

    assert "Could not cache the BioImage.IO model list" in caplog.text
    assert _panseg_model_ids() == ["efficient-chipmunk"]


@pytest.mark.parametrize(
    "response", [[], {"detail": "Internal Server Error"}], ids=["empty", "error"]
)
def test_refresh_bioimageio_zoo_keeps_cache_on_bad_response(
    response, offline_bioimageio_zoo
):
    model_zoo.refresh_bioimageio_zoo_urls()
    cached = zoo_module.PATH_BIOIMAGE_IO_MODEL_LIST.read_bytes()
    offline_bioimageio_zoo.return_value = response

    model_zoo.refresh_bioimageio_zoo_urls(force=True)

    assert zoo_module.PATH_BIOIMAGE_IO_MODEL_LIST.read_bytes() == cached
    assert _panseg_model_ids() == ["efficient-chipmunk"]


HYPHA_FILES_URL = "https://hypha.aicell.io/bioimage-io/artifacts/{}/files/"


class TestResolveBioImageIOSource:
    """Model IDs resolve to the RDF on Hypha; URLs and paths pass through."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        zoo_module.bioimageio_rdf_url.cache_clear()
        yield
        zoo_module.bioimageio_rdf_url.cache_clear()

    @pytest.mark.parametrize(
        ("files", "expected"),
        [
            (["bioimageio.yaml", "rdf.yaml", "weights.pt"], "bioimageio.yaml"),
            (["rdf.yaml", "weights.pt"], "rdf.yaml"),
        ],
    )
    def test_picks_rdf_file(self, requests_mock, files, expected):
        files_url = HYPHA_FILES_URL.format("affable-shark")
        requests_mock.get(
            files_url, json=[{"type": "file", "name": name} for name in files]
        )

        assert resolve_bioimageio_source("affable-shark") == files_url + expected
        # the full ID resolves the same, from the cache
        assert (
            resolve_bioimageio_source("bioimage-io/affable-shark")
            == files_url + expected
        )
        assert requests_mock.call_count == 1

    def test_model_without_rdf(self, requests_mock):
        requests_mock.get(
            HYPHA_FILES_URL.format("affable-shark"),
            json=[{"type": "file", "name": "weights.pt"}],
        )

        with pytest.raises(ValueError, match="has no bioimageio.yaml or rdf.yaml"):
            resolve_bioimageio_source("affable-shark")

    def test_unknown_model(self, requests_mock):
        requests_mock.get(
            HYPHA_FILES_URL.format("no-such-model"),
            status_code=404,
            json={"detail": "Artifact not found"},
        )

        with pytest.raises(ValueError, match="'no-such-model' not found"):
            resolve_bioimageio_source("no-such-model")

    def test_passes_through_urls(self, requests_mock):
        url = "https://example.org/rdf.yaml"
        assert resolve_bioimageio_source(url) == url
        assert requests_mock.call_count == 0

    @pytest.mark.parametrize(
        "source",
        ["smth", "philosophical_panda", "Efficient-Chipmunk", "affable-shark/1.1"],
    )
    def test_rejects_non_ids(self, requests_mock, source):
        with pytest.raises(ValueError, match="is not a BioImage.IO model ID"):
            resolve_bioimageio_source(source)
        assert requests_mock.call_count == 0

    def test_passes_through_local_paths(self, requests_mock, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "my-model").mkdir()
        package = tmp_path / "my-model.zip"
        package.touch()

        assert resolve_bioimageio_source("my-model") == "my-model"
        assert resolve_bioimageio_source(package) == package
        assert requests_mock.call_count == 0


def test_dimensionality_filter_is_spatial_only():
    """The zoo's dimensionality is a spatial property, with no T-dependent
    values: a TZYX image picks 3D models, a TYX image 2D models."""
    dimensionalities = {
        record["dimensionality"]
        for record in model_zoo.get_model_zoo_dict().values()
        if record["added_by"] != Author.USER and record["dimensionality"]
    }
    assert dimensionalities <= {"2D", "3D"}

    models_2d = model_zoo.list_models(
        use_custom_models=False, dimensionality_filter=["2D"]
    )
    models_3d = model_zoo.list_models(
        use_custom_models=False, dimensionality_filter=["3D"]
    )
    assert models_2d and models_3d
    assert not set(models_2d) & set(models_3d)


ZOO_MODEL_NAME = "confocal_2D_unet_ovules_ds2x"
ZOO_MODEL_URL = (
    "https://zenodo.org/records/7772709/files/confocal_2D_unet_ovules_ds2x.pytorch"
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
