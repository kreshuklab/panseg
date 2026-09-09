from unittest.mock import MagicMock

import pytest
import requests

import panseg.utils as utils_mod
from panseg.utils import check_version, download_file


@pytest.fixture
def mock_logger(mocker):
    """Fixture to mock the logger."""
    return mocker.patch("panseg.utils.logger")


def test_check_version_new_version(mock_logger, requests_mock):
    """Test when the latest version is newer than the current version."""
    current_version = "1.0.0"
    latest_version = "2.0.0"

    # Mock the API response
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/panseg/releases?per_page=100",
        json=[
            {
                "tag_name": latest_version,
                "prerelease": False,
                "body": "something\n"
                "feat: first feature by @me\n"
                "feat(specific): second feature\n"
                "some more text",
            }
        ],
    )

    logline, features = check_version(current_version)

    # Assert logger warning was called with appropriate message
    true_logline = (
        f"You are using PanSeg {current_version}\n"
        f"New release of PanSeg available: {latest_version}.\n"
        "Please update to the latest version."
    )
    mock_logger.warning.assert_called_once_with(true_logline)
    assert logline == true_logline
    assert (
        features == "New features in newest release:\n\nfirst feature\nsecond feature"
    )


def test_check_version_same_version(mock_logger, requests_mock):
    """Test when the current version is the same as the latest version."""
    current_version = "2.0.0"
    latest_version = "2.0.0"

    # Mock the API response
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/panseg/releases?per_page=100",
        json=[
            {
                "tag_name": latest_version,
                "prerelease": False,
                "body": "something\n"
                "feat: first feature by @me\n"
                "feat(specific): second feature\n"
                "some more text",
            }
        ],
    )

    logline, features = check_version(current_version)

    # Assert logger info was called with appropriate message
    true_logline = f"You are using the latest release of PanSeg: {current_version}"
    mock_logger.info.assert_called_once_with(true_logline)
    assert logline == true_logline
    assert features == "New features in this release:\n\nfirst feature\nsecond feature"


def test_check_version_old_version(mock_logger, requests_mock):
    """Test when the current version is newer than the latest version."""
    current_version = "2.0.0"
    latest_version = "1.9.0"

    # Mock the API response
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/panseg/releases?per_page=100",
        json=[
            {
                "tag_name": latest_version,
                "prerelease": False,
                "body": "something\n"
                "feat: first feature by @me\n"
                "feat(specific): second feature\n"
                "some more text",
            }
        ],
    )

    logline, features = check_version(current_version)

    # Assert logger info was called with appropriate message
    true_logline = f"You are using a pre-release version of PanSeg: {current_version}"

    mock_logger.info.assert_called_once_with(true_logline)
    assert logline == true_logline
    assert features == ""


def test_check_version_beta_version(mock_logger, requests_mock):
    """Test when the latest version is a beta version."""
    current_version = "2.0.0b1"
    latest_version = "2.0.0b3"

    # Mock the API response
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/panseg/releases?per_page=100",
        json=[
            {
                "tag_name": latest_version,
                "prerelease": True,
                "body": "something\n"
                "feat: first feature by @me\n"
                "feat(specific): second feature\n"
                "some more text",
            },
            {"tag_name": "1.8.0", "prerelease": False, "body": ""},
        ],
    )

    logline, features = check_version(current_version)

    # Assert logger info was called with appropriate message
    true_logline = (
        f"You are using PanSeg {current_version}\n"
        f"New version of PanSeg available: {latest_version}.\n"
        "Please update to the latest version."
    )
    mock_logger.warning.assert_called_once_with(true_logline)
    assert logline == true_logline
    assert (
        features == "New features in newest version:\n\nfirst feature\nsecond feature"
    )


def test_check_version_new_beta_version(mock_logger, requests_mock):
    """Test when the current version is older and the latest version is a beta."""
    current_version = "1.0.0"
    latest_version = "2.0.0b3"

    # Mock the API response
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/panseg/releases?per_page=100",
        json=[
            {"tag_name": latest_version, "prerelease": True, "body": ""},
        ],
    )

    check_version(current_version)

    # Assert logger warning was called with appropriate message
    mock_logger.warning.assert_called_once_with(
        f"You are using PanSeg {current_version}\n"
        f"New version of PanSeg available: {latest_version}.\n"
        "Please update to the latest version."
    )


def test_check_version_request_exception(mock_logger, requests_mock):
    """Test when the GitHub API request fails."""
    current_version = "1.0.0"

    # Mock the API to raise a RequestException
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/panseg/releases?per_page=100",
        exc=requests.RequestException,
    )

    check_version(current_version)

    # Assert logger warning was called with appropriate message
    mock_logger.warning.assert_called_once_with(
        "Could not check for new version. Error: "
    )


def test_check_version_value_error(mock_logger, requests_mock):
    """Test when the version format is invalid."""
    current_version = "1.0.0"

    # Mock the API response with an invalid version format
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/panseg/releases?per_page=100",
        json=[{"tag_name": "invalid_version", "prerelease": False, "body": ""}],
    )

    check_version(current_version)

    # Assert logger warning was called with appropriate message
    mock_logger.warning.assert_called_once_with(
        "Could not parse version information. Error: Invalid version: 'invalid_version'"
    )


def _ok_response(payload=b"fake-bytes"):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.iter_content.side_effect = lambda chunk_size=None: iter([payload])
    return resp


class TestDownloadFileResilience:
    """download_file must be timeout-bound, retried, atomic and loud."""

    def test_success_first_attempt_writes_file(self, tmp_path, mocker):
        url = "https://zenodo.org/record/1/files/a.pytorch"
        target = tmp_path / "a.pytorch"
        mock_get = mocker.patch.object(
            utils_mod.requests, "get", return_value=_ok_response(b"weights")
        )

        download_file(url, target)

        assert target.read_bytes() == b"weights"
        # every request must carry an explicit (connect, read) timeout
        for call in mock_get.call_args_list:
            timeout = call.kwargs.get("timeout")
            assert timeout is not None, "request was made without a timeout"

    def test_transient_failure_then_retry_succeeds(self, tmp_path, mocker):
        url = "https://zenodo.org/record/1/files/a.pytorch"
        target = tmp_path / "a.pytorch"
        mock_get = mocker.patch.object(
            utils_mod.requests,
            "get",
            side_effect=[
                requests.exceptions.HTTPError("504 Gateway Timeout"),
                _ok_response(b"weights"),
            ],
        )
        mock_sleep = mocker.patch.object(utils_mod.time, "sleep")

        download_file(url, target)

        assert target.read_bytes() == b"weights"
        assert mock_get.call_count == 2
        # exponential backoff starts at the initial delay (seconds scale)
        assert [c.args[0] for c in mock_sleep.call_args_list] == [
            utils_mod.INITIAL_RETRY_DELAY
        ]

    def test_all_attempts_fail_raises_and_leaves_no_residue(self, tmp_path, mocker):
        url = "https://zenodo.org/record/1/files/a.pytorch"
        target = tmp_path / "a.pytorch"
        mock_get = mocker.patch.object(
            utils_mod.requests,
            "get",
            side_effect=requests.exceptions.ConnectionError("connection reset"),
        )
        mock_sleep = mocker.patch.object(utils_mod.time, "sleep")
        mock_logger = mocker.patch.object(utils_mod, "logger")

        with pytest.raises(requests.RequestException) as excinfo:
            download_file(url, target)

        message = str(excinfo.value)
        assert url in message
        assert "connection reset" in message  # underlying cause is named
        assert mock_get.call_count == utils_mod.MAX_DOWNLOAD_ATTEMPTS
        # exponential backoff ramps from seconds up to ~30s across attempts
        assert [c.args[0] for c in mock_sleep.call_args_list] == [10, 20, 30, 30]
        # each failed attempt is logged as a warning
        assert mock_logger.warning.call_count == utils_mod.MAX_DOWNLOAD_ATTEMPTS
        # neither the final file nor any temp-file residue remains
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []
