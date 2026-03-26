import importlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import rmtree

import requests
import yaml
from packaging.version import Version

from panseg import PATH_PANSEG_MODELS

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load a YAML configuration file into a dictionary."""
    config_path = Path(config_path)
    with config_path.open("r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def save_config(config: dict, config_path: Path) -> None:
    """Save a dictionary to a YAML configuration file."""
    config_path = Path(config_path)
    with config_path.open("w") as file:
        yaml.dump(config, file)


def download_file(url: str, filename: Path) -> None:
    """Download a single file from a URL to a specified filename."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:147.0) Gecko/20100101 Firefox/147.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        # Use stream for large files
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(
                chunk_size=8192
            ):  # Adjust chunk size as needed
                f.write(chunk)
    except requests.RequestException as e:
        logger.warning(f"Failed to download {url}. Error: {e}")


def download_files(urls: dict, out_dir: Path) -> None:
    """Download files from URLs to a specified directory."""
    logger.debug(f"Queued download of files: {urls}")
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)  # Create the directory and any parent directories

    with ThreadPoolExecutor() as executor:
        futures = []
        for filename, url in urls.items():
            file_path = out_dir / filename
            if not file_path.exists():  # Skip download if file already exists
                logger.debug(f"Downloading file {filename} from {url}...")
                futures.append(executor.submit(download_file, url, file_path))
            else:
                logger.debug(f"File {filename} already exists. Skipping download.")

        for future in futures:
            future.result()  # Wait for all downloads to complete


def clean_models() -> None:
    """Delete all models in the model zoo after confirmation from the user."""
    for _ in range(3):
        answer = (
            input(
                "This will delete all models in the model zoo.Ensure you've backed up custom models. Continue? (y/n): "
            )
            .strip()
            .lower()
        )
        if answer == "y":
            try:
                rmtree(PATH_PANSEG_MODELS, ignore_errors=True)
                logger.info("All models deleted successfully.")
            except Exception as e:
                logger.warning(f"An error occurred while trying to delete models: {e}")
            finally:
                logger.info("Operation complete. PanSeg will now close.")
                break
        elif answer == "n":
            logger.info("Operation cancelled. No models were deleted.")
            break
        else:
            logger.warning("Invalid input, please type 'y' or 'n'.")


def check_version(
    current_version: str,
    panseg_url: str = "https://api.github.com/repos/kreshuklab/panseg/releases?per_page=100",
    silent: bool = False,
) -> tuple:
    """
    Check for the latest version of PanSeg available on GitHub.
    Parses the changelog and returns new or current features, depending on
    whether a new version is available or not.

    Args:
        current_version (str): The current version of PanSeg in use.
        panseg_url (str): The URL to check the latest release of PanSeg
            (default is the GitHub API URL).
        silent (bool): Silences logging

    Returns:
        result (str): Basic version statement
        feature_text (str): Multiline description of new/current features
    """
    try:
        crr_version = Version(current_version)
        response = requests.get(panseg_url)
        response.raise_for_status()  # Raises an HTTPError if the response status code was unsuccessful

        latest_version = Version("0.0.0")
        latest_release_version = Version("0.0.0")
        latest_body = ""
        latest_release_body = ""
        for rel in response.json():
            v = Version(rel["tag_name"])
            if rel["prerelease"]:
                if v > latest_version:
                    latest_version = v
                    latest_body = rel["body"]
            else:
                if v > latest_release_version:
                    latest_release_version = v
                    latest_release_body = rel["body"]

        new_features = []
        for match in re.findall(r"feat.*?: (.*?)\n", latest_body):
            new_features.append(match.split(" by @")[0])
        logger.debug(f"fetched new_features {new_features}")

        new_release_features = []
        for match in re.findall(r"feat.*?: (.*?)\n", latest_release_body):
            new_release_features.append(match.split(" by @")[0])

        result = ""
        feature_text_l = []

        if crr_version == latest_release_version:
            result = f"You are using the latest release of PanSeg: {current_version}"
            if not silent:
                logger.info(result)

            if new_release_features:
                feature_text_l.append("New features in this release:\n")
                feature_text_l.extend(new_release_features[:8])
        elif crr_version < latest_release_version:
            result = (
                f"You are using PanSeg {current_version}\n"
                f"New release of PanSeg available: {latest_release_version}.\n"
                "Please update to the latest version."
            )
            if not silent:
                logger.warning(result)

            if new_release_features:
                feature_text_l.append("New features in newest release:\n")
                feature_text_l.extend(new_release_features[:8])

        elif latest_release_version < crr_version < latest_version:
            result = (
                f"You are using PanSeg {current_version}\n"
                f"New version of PanSeg available: {latest_version}.\n"
                "Please update to the latest version."
            )
            if not silent:
                logger.warning(result)

            if new_features:
                feature_text_l.append("New features in newest version:\n")
                feature_text_l.extend(new_features[:8])

        elif crr_version >= latest_version:
            result = f"You are using a pre-release version of PanSeg: {current_version}"

            if crr_version == latest_version:
                if new_features:
                    feature_text_l.append("New features in this version:\n")
                    feature_text_l.extend(new_features[:8])
            if not silent:
                logger.info(result)

        logger.debug(
            f"Current: {crr_version}, latest version: {latest_version}, latest release: {latest_release_version}"
        )
        feature_text = "\n".join(feature_text_l)
        return result, feature_text

    except requests.RequestException as e:
        logger.warning(f"Could not check for new version. Error: {e}")
        return f"Could not check for new version. Error: {e}", ""
    except ValueError as e:
        logger.warning(f"Could not parse version information. Error: {e}")
        return f"Could not parse version information. Error: {e}", ""


def get_class(class_name, modules):
    """Get a class by name from a list of modules."""
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported class: {class_name}")
