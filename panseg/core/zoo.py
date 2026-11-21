"""Model Zoo Singleton"""

# pylint: disable=C0116,C0103

import json
import logging
import re
import time
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from shutil import copy2
from typing import Optional, Self

import numpy as np
import requests
from bioimageio.spec import InvalidDescr, load_description
from bioimageio.spec.model.v0_4 import ModelDescr as ModelDescr_v0_4
from bioimageio.spec.model.v0_5 import ModelDescr as ModelDescr_v0_5
from bioimageio.spec.utils import download
from pandas import DataFrame, concat
from pydantic import AliasChoices, BaseModel, Field, HttpUrl, model_validator

from panseg import (
    FILE_BEST_MODEL_PYTORCH,
    FILE_CONFIG_TRAIN_YAML,
    FILE_LAST_MODEL_PYTORCH,
    PATH_MODEL_ZOO,
    PATH_MODEL_ZOO_CUSTOM,
    PATH_PANSEG_MODELS,
)
from panseg.functionals.model_geometry import model_geometry
from panseg.functionals.training.model import (
    AbstractUNet,
    UNet2D,
)
from panseg.utils import (
    download_files,
    fetch_json,
    get_class,
    load_config,
    save_config,
    write_text_atomic,
)

logger_zoo = logging.getLogger(__name__)


class Author(str, Enum):
    BIOIMAGEIO = "bioimage.io"
    PANSEG = "panseg"
    USER = "user"


BIOIMAGE_IO_ARTIFACTS_URL = "https://hypha.aicell.io/bioimage-io/artifacts"
# Published models with their manifests; bioimage.io serves its collection from here.
BIOIMAGE_IO_COLLECTION_URL = (
    f"{BIOIMAGE_IO_ARTIFACTS_URL}/bioimage.io/children"
    '?filters={"type":"model"}&limit=10000'
)
# Newer uploads name the RDF `bioimageio.yaml`, older ones `rdf.yaml`;
# bioimage.io looks for them in this order.
BIOIMAGE_IO_RDF_FILE_NAMES = ("bioimageio.yaml", "rdf.yaml")
BIOIMAGE_IO_MODEL_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")
BIOIMAGE_IO_LOOKUP_TIMEOUT = (5, 30)
# Fetched again once a day; the cached copy stands in when a fetch fails.
PATH_BIOIMAGE_IO_MODEL_LIST = PATH_PANSEG_MODELS / "bioimageio_models.json"
BIOIMAGE_IO_LIST_MAX_AGE = 24 * 60 * 60  # seconds


def _hypha_model_artifacts(payload) -> list[dict]:
    """Model artifacts in a Hypha `children` response.

    Hypha answers with either a bare list or a page
    `{"items", "total", "offset", "limit"}`.
    """
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        artifacts = payload["items"]
        total = payload.get("total")
        if isinstance(total, int) and total > len(artifacts):
            logger_zoo.warning(
                f"BioImage.IO returned {len(artifacts)} of {total} models"
            )
    elif isinstance(payload, list):
        artifacts = payload
    else:
        raise ValueError(f"Unexpected BioImage.IO model list: {type(payload).__name__}")
    return [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("type") == "model"
        and isinstance(artifact.get("alias"), str)
        and isinstance(artifact.get("manifest"), dict)
    ]


def _download_bioimageio_model_list(path: Path) -> list[dict]:
    """Fetch the model list and cache it at `path`; only a valid list replaces it."""
    artifacts = _hypha_model_artifacts(
        fetch_json(BIOIMAGE_IO_COLLECTION_URL, timeout=BIOIMAGE_IO_LOOKUP_TIMEOUT)
    )
    if not artifacts:
        raise ValueError("BioImage.IO returned no models")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, json.dumps(artifacts))
    except OSError as e:
        logger_zoo.warning(f"Could not cache the BioImage.IO model list at {path}: {e}")
    return artifacts


def _read_bioimageio_model_list(path: Path) -> list[dict]:
    try:
        return _hypha_model_artifacts(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        logger_zoo.warning(f"Ignoring unreadable BioImage.IO model list {path}: {e}")
        return []


@lru_cache(maxsize=64)
def bioimageio_rdf_url(model_id: str) -> str:
    """URL of the RDF of a bioimage.io model, e.g. 'efficient-chipmunk'."""
    files_url = f"{BIOIMAGE_IO_ARTIFACTS_URL}/{model_id}/files/"
    try:
        files = fetch_json(files_url, timeout=BIOIMAGE_IO_LOOKUP_TIMEOUT)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise ValueError(f"BioImage.IO model '{model_id}' not found") from e
        raise
    if isinstance(files, dict):
        files = files.get("items", [])
    names = {file.get("name") for file in files if isinstance(file, dict)}
    for name in BIOIMAGE_IO_RDF_FILE_NAMES:
        if name in names:
            url = files_url + name
            logger_zoo.info(f"Resolved BioImage.IO model {model_id} to {url}")
            return url
    raise ValueError(
        f"BioImage.IO model '{model_id}' has no "
        f"{' or '.join(BIOIMAGE_IO_RDF_FILE_NAMES)}"
    )


def resolve_bioimageio_source(source):
    """RDF URL for a bioimage.io model ID; URLs and existing paths pass through."""
    if not isinstance(source, str) or "://" in source or Path(source).exists():
        return source
    model_id = source.removeprefix("bioimage-io/")
    if not BIOIMAGE_IO_MODEL_ID.fullmatch(model_id):
        # bioimageio.spec would look it up in the retired EBI id map
        raise ValueError(
            f"'{source}' is not a BioImage.IO model ID such as 'efficient-chipmunk', "
            "a URL, or an existing path"
        )
    return bioimageio_rdf_url(model_id)


class ModelZooRecord(BaseModel):
    """Model Zoo Record"""

    name: str
    url: Optional[HttpUrl] = Field(
        None, validation_alias=AliasChoices("model_url", "url")
    )
    path: Optional[str] = None
    description: Optional[str] = None
    resolution: Optional[tuple[float, float, float]] = None
    dimensionality: Optional[str] = None
    modality: Optional[str] = None
    recommended_patch_size: Optional[tuple[float, float, float]] = None
    output_type: Optional[str] = None
    doi: Optional[str] = None
    added_by: Optional[str] = None

    # BioImage.IO models specific fields. TODO: unify.
    id: Optional[str] = None
    name_display: Optional[str] = None
    supported: Optional[bool] = None

    @model_validator(mode="after")
    def check_one_id_present(self) -> Self:
        """Check that one of url (zenodo), path (custom/local) or id (bioimage.io) is present"""
        if self.url is None and self.path is None and self.id is None:
            raise ValueError(f"One of `url`, `path` or `id` must be present: {self}")
        return self

    @model_validator(mode="after")
    def check_id_fields_present(self) -> Self:
        if self.id is not None and (
            self.name_display is None or self.supported is None
        ):
            raise ValueError(
                f"If `id` exists, then `name_display` and `supported` must be present: {self}"
            )
        return self


class ModelZoo:
    """Model Zoo Singleton

    A model zoo DataFrame serves as a central repository for model metadata.
    The DataFrame is indexed by model name and contains columns for model metadata.
    The ModelZoo class provides methods to update, query and add records to the DataFrame.

    Records are added as ModelZooRecord instances, validated by the ModelZooRecord class.
    """

    _instance: Optional["ModelZoo"] = None

    _zoo_dict: dict = {}
    _zoo_custom_dict: dict = {}
    _bioimageio_zoo_collection: list = []
    _bioimageio_zoo_all_model_url_dict: dict = {}

    path_zoo: Path = PATH_MODEL_ZOO
    path_zoo_custom: Path = PATH_MODEL_ZOO_CUSTOM

    models: DataFrame
    models_bioimageio: DataFrame

    def __new__(cls, path_zoo, path_zoo_custom):
        if cls._instance is None:
            cls._instance = super(ModelZoo, cls).__new__(cls)
            cls._instance.refresh(path_zoo, path_zoo_custom)
        return cls._instance

    def _init_zoo_dict(
        self,
        path_zoo: Optional[Path] = None,
        path_zoo_custom: Optional[Path] = None,
    ) -> None:
        if path_zoo is not None:
            self.path_zoo = path_zoo
        if path_zoo_custom is not None:
            self.path_zoo_custom = path_zoo_custom

        zoo_dict = load_config(Path(self.path_zoo))
        self._zoo_dict = zoo_dict

        if self.path_zoo_custom is not None:
            zoo_custom_dict = load_config(Path(self.path_zoo_custom))
            if zoo_custom_dict is None:
                zoo_custom_dict = {}

            self._zoo_custom_dict = zoo_custom_dict

    def _init_zoo_df(self) -> None:
        records = []
        for name, model in self._zoo_dict.items():
            model["name"] = name
            records.append(ModelZooRecord(**model, added_by=Author.PANSEG).model_dump())

        for name, model in self._zoo_custom_dict.items():
            model["name"] = name
            records.append(ModelZooRecord(**model, added_by=Author.USER).model_dump())

        self.models = DataFrame(
            records,
            columns=list(ModelZooRecord.model_fields.keys()),
        ).set_index("name")
        self.models = self.models.replace({np.nan: None})

    def refresh(
        self,
        path_zoo: Optional[Path] = None,
        path_zoo_custom: Optional[Path] = None,
    ) -> None:
        self._init_zoo_dict(path_zoo, path_zoo_custom)
        self._init_zoo_df()

    def get_model_zoo_dict(self) -> dict:
        return self.models.to_dict(orient="index")

    def list_models(
        self,
        use_custom_models: bool = True,
        modality_filter: Optional[list[str]] = None,
        output_type_filter: Optional[list[str]] = None,
        dimensionality_filter: Optional[list[str]] = None,
    ) -> list[str]:
        """Return a list of model names, filtered by the specified criteria"""
        filtered_df: DataFrame = self.models

        if dimensionality_filter is not None:
            # `loc` to avoid pylint E1136 ; following lines have no warning because filtered_df now is a DataFrame
            filtered_df = filtered_df.loc[
                filtered_df.loc[:, "dimensionality"].isin(dimensionality_filter)
            ]

        if modality_filter is not None:
            filtered_df = filtered_df[filtered_df["modality"].isin(modality_filter)]

        if output_type_filter is not None:
            filtered_df = filtered_df[
                filtered_df["output_type"].isin(output_type_filter)
            ]

        if not use_custom_models:
            filtered_df = filtered_df[filtered_df["added_by"] != Author.USER]

        return filtered_df.index.tolist()

    def get_model_names(self) -> list[str]:
        return self.models.index.to_list()

    def _get_model_record(self, name) -> ModelZooRecord:
        return ModelZooRecord(name=name, **self.models.loc[name])

    def get_model_description(self, model_name: str) -> Optional[str]:
        return self._get_model_record(model_name).description

    def get_model_resolution(
        self, model_name: str
    ) -> Optional[tuple[float, float, float]]:
        return self._get_model_record(model_name).resolution

    def get_model_patch_size(
        self, model_name: str
    ) -> Optional[tuple[float, float, float]]:
        return self._get_model_record(model_name).recommended_patch_size

    def _get_unique_metadata(self, metadata_key: str) -> list[str]:
        metadata = self.models.loc[:, metadata_key].dropna().unique()
        return [str(x) for x in metadata]

    def get_unique_dimensionalities(self) -> list[str]:
        return self._get_unique_metadata("dimensionality")

    def get_unique_modalities(self) -> list[str]:
        return self._get_unique_metadata("modality")

    def get_unique_output_types(self) -> list[str]:
        return self._get_unique_metadata("output_type")

    def _add_model_record(self, model_record: ModelZooRecord) -> None:
        """Add a ModelZooRecord to the ModelZoo DataFrame"""
        models_new = DataFrame(
            [model_record.model_dump()],
            columns=list(ModelZooRecord.model_fields.keys()),
        ).set_index("name")
        self.models = concat([self.models, models_new], ignore_index=False)

    def add_custom_model(
        self,
        new_model_name: str,
        location: Path = Path.home(),
        resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
        description: str = "",
        dimensionality: str = "",
        modality: str = "",
        output_type: str = "",
    ) -> tuple[bool, Optional[str]]:
        """Add a custom trained model in the model zoo local record file"""

        dest_dir = PATH_PANSEG_MODELS / new_model_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        all_expected_files = {
            FILE_CONFIG_TRAIN_YAML,
            FILE_LAST_MODEL_PYTORCH,
            FILE_BEST_MODEL_PYTORCH,
        }
        found_files = set()

        recommended_patch_size = [80, 170, 170]
        for file_path in location.glob("*"):
            if file_path.name == "config_train.yml":
                try:
                    config_train = load_config(file_path)
                    recommended_patch_size = config_train["loaders"]["train"][
                        "slice_builder"
                    ]["patch_shape"]
                except Exception as e:
                    return False, f"Failed to load or parse config_train.yml: {e}"

            if file_path.name in all_expected_files:
                # panseg models are created by default in the model directory
                if not file_path.parent == dest_dir:
                    copy2(file_path, dest_dir)
                found_files.add(file_path.name)

        missing_files = all_expected_files - found_files
        if missing_files:
            msg = f"Missing required files in the specified directory: {', '.join(missing_files)}. Model cannot be loaded."
            return False, msg

        # Create and check the new model record
        new_model_record = {
            "path": str(location),
            "resolution": resolution,
            "description": description,
            "recommended_patch_size": recommended_patch_size,
            "dimensionality": dimensionality,
            "modality": modality,
            "output_type": output_type,
        }
        self._add_model_record(
            ModelZooRecord(
                name=new_model_name, **new_model_record, added_by=Author.USER
            )
        )

        # Update the custom zoo dictionary in ModelZoo and save to file
        self._zoo_custom_dict[new_model_name] = new_model_record
        save_config(self._zoo_custom_dict, self.path_zoo_custom)

        return True, None

    def _download_model_files(
        self, model_url: str, out_dir: Path, config_only: bool = False
    ) -> None:
        """Download model files and/or configuration based on the model URL."""
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        config_url = f"{model_url.rsplit('/', 1)[0]}/{FILE_CONFIG_TRAIN_YAML}"
        urls = {FILE_CONFIG_TRAIN_YAML: config_url}
        if not config_only:
            urls[FILE_BEST_MODEL_PYTORCH] = model_url
        download_files(urls, out_dir)

    def check_models(
        self, model_name: str, update_files: bool = False, config_only: bool = False
    ) -> None:
        """Check and download model files and configurations as needed.

        After the download step, verifies that the expected files are on disk
        and raises a FileNotFoundError naming the model (and its zoo URL when
        it has one) if anything is missing.
        """
        model_dir = PATH_PANSEG_MODELS / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        logger_zoo.debug(
            f"check_models, model_name: {model_name}, "
            f"update_files {update_files}, config_only {config_only}"
        )

        expected_files = [FILE_CONFIG_TRAIN_YAML]
        if not config_only:
            expected_files.append(FILE_BEST_MODEL_PYTORCH)

        model_url = None
        downloaded = False
        missing = [f for f in expected_files if not (model_dir / f).exists()]
        if missing or update_files:
            config = load_config(PATH_MODEL_ZOO)
            model_url = config.get(model_name, {}).get("model_url")
            if model_url:
                self._download_model_files(model_url, model_dir, config_only)
                downloaded = True
            else:
                logger_zoo.warning(
                    f"Model {model_name} not found in the models zoo configuration."
                )
            missing = [f for f in expected_files if not (model_dir / f).exists()]

        if missing:
            raise FileNotFoundError(
                f"Model '{model_name}' has missing files after the download step: "
                f"{', '.join(missing)} (expected in {model_dir}). "
                + (
                    f"Model URL: {model_url}"
                    if model_url is not None
                    else "The model was not found in the models zoo configuration."
                )
            )

        if downloaded:
            logger_zoo.info(f"Download finished for {model_name}")

    def _get_model_config_path_by_name(self, model_name: str) -> Path:
        """Return the path to the training configuration for a model in zoo."""
        self.check_models(model_name, config_only=True)
        return PATH_PANSEG_MODELS / model_name / FILE_CONFIG_TRAIN_YAML

    def get_model_config_by_name(self, model_name: str) -> dict:
        """Load the training configuration for a model in zoo."""
        config_path = self._get_model_config_path_by_name(model_name)
        return load_config(config_path)

    def _create_model_by_config(self, model_config: dict):
        """Create a model instance from a configuration."""
        model_class = get_class(
            model_config["name"], modules=["panseg.functionals.training.model"]
        )
        return model_class(**model_config)

    def get_model_by_config_path(
        self, config_path: Path, model_weights_path: Optional[Path] = None
    ):
        """Create a safari model (may or may not be in zoo) from a configuration file."""
        config_train = load_config(config_path)
        model_config = config_train.pop("model")
        model = self._create_model_by_config(model_config)
        if model_weights_path is None:
            model_weights_path = config_path.parent / FILE_BEST_MODEL_PYTORCH
        logger_zoo.info(f"Loading model from weights: {model_weights_path}")
        return model, model_config, model_weights_path

    def get_model_by_name(self, model_name: str, model_update: bool = False):
        """Load configuration for a model in zoo; return the model, configuration and path."""
        self.check_models(model_name, update_files=model_update, config_only=False)
        config_path = self._get_model_config_path_by_name(model_name)
        model_weights_path = PATH_PANSEG_MODELS / model_name / FILE_BEST_MODEL_PYTORCH
        logger_zoo.info(f"Loaded model from PanSeg zoo: {model_name}")
        return self.get_model_by_config_path(config_path, model_weights_path)

    def get_model_by_id(self, model_id: str):
        """Load model from BioImage.IO Model Zoo.

        `model_id` is a bioimage.io model ID such as 'efficient-chipmunk', or the
        URL or path of an RDF or model package.
        """
        rdf_source = resolve_bioimageio_source(model_id)
        model_description = load_description(rdf_source, perform_io_checks=False)

        # Check if description is `ResourceDescr`
        if isinstance(model_description, InvalidDescr):
            model_description.validation_summary.display()
            raise ValueError(f"Failed to load {model_id}")

        # Check `model_description` has `weights`
        if not isinstance(model_description, ModelDescr_v0_4) and not isinstance(
            model_description, ModelDescr_v0_5
        ):
            raise ValueError(
                f"Model description {model_id} is not in v0.4 or v0.5 BioImage.IO model description format. "
                "Only v0.4 and v0.5 formats are supported by BioImage.IO Spec and PanSeg."
            )

        # Check `model_description.weights` has `pytorch_state_dict`
        if model_description.weights.pytorch_state_dict is None:
            raise ValueError(f"Model {model_id} does not have PyTorch weights")

        # Spec format version v0.4 and v0.5 have different designs to store model architecture
        if isinstance(
            model_description, ModelDescr_v0_4
        ):  # then `pytorch_state_dict.architecture` is nn.Module
            architecture_callable = (
                model_description.weights.pytorch_state_dict.architecture
            )
            architecture_kwargs = model_description.weights.pytorch_state_dict.kwargs
        elif isinstance(
            model_description, ModelDescr_v0_5
        ):  # then it is `ArchitectureDescr` with `callable`
            architecture_callable = (
                model_description.weights.pytorch_state_dict.architecture.callable
            )
            architecture_kwargs = (
                model_description.weights.pytorch_state_dict.architecture.kwargs
            )
        else:
            raise ValueError(
                f"Unsupported model description format: {type(model_description).__name__}"
            )
        logger_zoo.info(
            f"Got {architecture_callable} model with kwargs {architecture_kwargs}."
        )

        # Create model from architecture and kwargs
        architecture = str(
            architecture_callable
        )  # e.g. 'panseg.functionals.training.model.UNet3D'
        architecture = "UNet3D" if "UNet3D" in architecture else "UNet2D"
        model_config = {
            "name": architecture,
            "in_channels": 1,
            "out_channels": 1,
            "layer_order": "gcr",
            "f_maps": 32,
            "num_groups": 8,
            "final_sigmoid": True,
        }
        model_config.update(architecture_kwargs)
        model = self._create_model_by_config(model_config)
        # the file description carries the sha256, so the download is verified and cached
        model_weights_path = download(model_description.weights.pytorch_state_dict)

        logger_zoo.info(f"Created {architecture} model with kwargs {model_config}.")
        logger_zoo.info(f"Loaded model from BioImage.IO Model Zoo: {model_id}")
        return model, model_config, model_weights_path

    def _init_bioimageio_zoo_df(self) -> None:
        records = []
        for _, model in self._bioimageio_zoo_all_model_url_dict.items():
            records.append(
                ModelZooRecord(**model, added_by=Author.BIOIMAGEIO).model_dump()
            )

        self.models_bioimageio = DataFrame(
            records,
            columns=list(ModelZooRecord.model_fields.keys()),
        ).set_index("id")

    def refresh_bioimageio_zoo_urls(self, force: bool = False) -> None:
        """Load the BioImage.IO model list into `models_bioimageio`.

        The list is cached on disk and fetched again once the copy is a day old,
        or when `force` is set. If the fetch fails, the cached copy is used, or no
        models at all, so this never fails for want of network.
        """
        path = PATH_BIOIMAGE_IO_MODEL_LIST
        fresh = (
            path.exists()
            and time.time() - path.stat().st_mtime < BIOIMAGE_IO_LIST_MAX_AGE
        )
        collection = _read_bioimageio_model_list(path) if fresh and not force else []
        if not collection:
            logger_zoo.info(
                f"Fetching the BioImage.IO model list from {BIOIMAGE_IO_COLLECTION_URL}"
            )
            try:
                collection = _download_bioimageio_model_list(path)
            except (requests.RequestException, ValueError) as e:
                collection = _read_bioimageio_model_list(path) if path.exists() else []
                if collection:
                    saved = datetime.fromtimestamp(path.stat().st_mtime)
                    logger_zoo.warning(
                        f"Could not fetch the BioImage.IO model list ({e}); "
                        f"using the copy from {saved:%Y-%m-%d %H:%M}"
                    )
                else:
                    logger_zoo.warning(
                        f"Could not fetch the BioImage.IO model list ({e}); "
                        "no BioImage.IO models are available"
                    )

        # `alias` is the model ID (e.g. 'efficient-chipmunk'), `manifest` its RDF.
        models = [
            {
                **artifact["manifest"],
                "id": artifact["alias"],
                "name": artifact["manifest"].get("name") or artifact["alias"],
            }
            for artifact in collection
        ]
        max_nickname_length = max((len(entry["id"]) for entry in models), default=0)

        def truncate_name(name, length=100):
            return name[:length] + "..." if len(name) > length else name

        def build_model_url_dict(filter_func=None):
            filtered_models = filter(filter_func, models) if filter_func else models
            # Keyed by ID: model names on bioimage.io are not unique.
            return {
                entry["id"]: {
                    "id": entry["id"],
                    "name": entry["name"],
                    "name_display": f"{entry['id']:<{max_nickname_length}}: {truncate_name(entry['name'])}",
                    "supported": self._is_panseg_model(entry),
                }
                for entry in filtered_models
            }

        self._bioimageio_zoo_collection = collection
        self._bioimageio_zoo_all_model_url_dict = build_model_url_dict()
        self._init_bioimageio_zoo_df()
        supported = sum(
            model["supported"]
            for model in self._bioimageio_zoo_all_model_url_dict.values()
        )
        logger_zoo.info(
            f"Loaded {len(models)} BioImage.IO models, {supported} of them for PanSeg"
        )

    def _is_panseg_model(self, collection_entry: dict) -> bool:
        """Determines if the 'tags' field in a collection entry contains the keyword 'panseg'."""
        tags = collection_entry.get("tags")
        if not isinstance(tags, list):
            return False

        # Normalize tags to lower case and remove non-alphanumeric characters
        normalized_tags = {
            "".join(filter(str.isalnum, tag.lower()))
            for tag in tags
            if isinstance(tag, str)
        }
        return not normalized_tags.isdisjoint({"plantseg", "panseg"})

    def has_bioimageio_models(self) -> bool:
        """Whether a non-empty BioImage.IO model list is loaded; never fetches it."""
        return hasattr(self, "models_bioimageio") and not self.models_bioimageio.empty

    def _get_bioimageio_zoo_model_names(
        self, supported: Optional[bool] = None
    ) -> list[tuple[str, str]]:
        if not hasattr(self, "models_bioimageio"):
            self.refresh_bioimageio_zoo_urls()
        models = self.models_bioimageio
        if supported is not None:
            # an empty frame has an object column, which can't be used as a mask
            models = models[models["supported"].astype(bool) == supported]
        return sorted(zip(models["name_display"], models.index))

    def get_bioimageio_zoo_all_model_names(self) -> list[tuple[str, str]]:
        """Return (display name, model ID) of every model in the BioImage.IO Model Zoo."""
        return self._get_bioimageio_zoo_model_names()

    def get_bioimageio_zoo_panseg_model_names(self) -> list[tuple[str, str]]:
        """Return (display name, model ID) of BioImage.IO models tagged 'panseg' or 'plantseg'."""
        return self._get_bioimageio_zoo_model_names(supported=True)

    def get_bioimageio_zoo_other_model_names(self) -> list[tuple[str, str]]:
        """Return (display name, model ID) of BioImage.IO models not tagged for PanSeg."""
        return self._get_bioimageio_zoo_model_names(supported=False)

    def compute_3D_halo_for_zoo_models(self, model_name: str) -> tuple[int, int, int]:
        model, _, _ = self.get_model_by_name(model_name)
        return model_geometry(model).halo

    def compute_3D_halo_for_bioimageio_models(
        self, model_id: str
    ) -> tuple[int, int, int]:
        model, _, _ = self.get_model_by_id(model_id)
        return model_geometry(model).halo

    def is_2D(self, module: AbstractUNet) -> bool:
        return isinstance(module, UNet2D)

    def is_2D_zoo_model(self, model_name: str) -> bool:
        model, _, _ = self.get_model_by_name(model_name)
        return self.is_2D(model)

    def is_2D_bioimageio_model(self, model_id: str) -> bool:
        model, _, _ = self.get_model_by_id(model_id)
        return self.is_2D(model)


model_zoo = ModelZoo(PATH_MODEL_ZOO, PATH_MODEL_ZOO_CUSTOM)
