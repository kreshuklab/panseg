import sys

import pytest

from panseg.__version__ import __version__
from panseg.run_panseg import create_parser, launch_napari, main


def test_create_parser():
    parser = create_parser()
    assert len(parser._actions) == 8


def test_launch_napari(mocker):
    mock = mocker.Mock()
    sys.modules["panseg.viewer_napari.viewer"] = mock
    launch_napari()
    mock.Panseg_viewer.assert_called_once()


def test_main_version(mocker):
    sys.argv = ["panseg", "-v"]
    mock = mocker.patch("panseg.run_panseg.print")
    with pytest.raises(SystemExit):
        main()
    mock.assert_called_with(__version__)
