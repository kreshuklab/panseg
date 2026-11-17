"""Initialise model registry at startup"""

import logging
from os import getenv
from pathlib import Path

import torch  # noqa: F401 ; required to patch #273
import yaml

from panseg.loggers import stream_handler

PATH_PANSEG_GLOBAL = Path(__file__).parent.resolve()

# Files in code repository
DIR_RESOURCES = "resources"
FILE_MODEL_ZOO = "models_zoo.yaml"
FILE_CONFIG_TRAIN_TEMPLATE = "config_train_template.yaml"
FILE_CONFIG_TRAIN_YAML = "config_train.yml"
FILE_BEST_MODEL_PYTORCH = "best_checkpoint.pytorch"
FILE_LAST_MODEL_PYTORCH = "last_checkpoint.pytorch"

PATH_RESOURCES = PATH_PANSEG_GLOBAL / DIR_RESOURCES

PATH_MODEL_ZOO = PATH_RESOURCES / FILE_MODEL_ZOO
PATH_TRAIN_TEMPLATE = PATH_RESOURCES / FILE_CONFIG_TRAIN_TEMPLATE

# Files in user home
DIR_PANSEG_MODELS = ".panseg_models"
DIR_CONFIGS = "configs"
FILE_MODEL_ZOO_CUSTOM = "custom_zoo.yaml"

PATH_HOME = Path(getenv("PANSEG_HOME", str(Path.home())))

PATH_PANSEG_MODELS = PATH_HOME / DIR_PANSEG_MODELS
PATH_CONFIGS = PATH_PANSEG_MODELS / DIR_CONFIGS
PATH_MODEL_ZOO_CUSTOM = PATH_PANSEG_MODELS / FILE_MODEL_ZOO_CUSTOM

PATH_CONFIGS.mkdir(parents=True, exist_ok=True)

if not PATH_MODEL_ZOO_CUSTOM.exists():
    with PATH_MODEL_ZOO_CUSTOM.open("w") as file:
        yaml.dump({}, file)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.info(f"Logger configured at initialisation. PanSeg logger name: {__name__}")
