import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from magicgui.widgets import ComboBox, Container, FloatSpinBox  # pyright: ignore

from panseg.core.image import TIME_UNIT_CHOICES
from panseg.workflow_gui.editor import Workflow_gui


@pytest.fixture
def mock_logger(mocker):
    """Fixture to mock the logger."""
    return mocker.patch("panseg.workflow_gui.widgets.logger")


@pytest.fixture
def gui(qtbot, request):
    workflow_yaml = request.getfixturevalue(request.param)
    gui = Workflow_gui(config_path=workflow_yaml, run=False)
    qtbot.addWidget(gui.main_window.native)
    return gui


@pytest.mark.parametrize("gui", ["workflow_yaml"], indirect=True)
def test_loading_dialog(gui, workflow_yaml):
    with open(workflow_yaml, "r") as f:
        parsed_in = yaml.safe_load(f)
    parsed_in["runner"] = "serial"
    assert parsed_in == gui.config

    gui.change_config.call_button.native.click()
    assert gui.config is None

    gui.loader_w.config_path.value = str(workflow_yaml)
    gui.loader_w.call_button.native.click()
    assert parsed_in == gui.config


@pytest.mark.parametrize(
    "gui, workflow",
    [
        (
            "workflow_complete_yaml",
            "workflow_complete_yaml",
        ),
        (
            "workflow_aio_yaml",
            "workflow_aio_yaml",
        ),
        (
            "workflow_yaml",
            "workflow_yaml",
        ),
        (
            "workflow_t_spacing_yaml",
            "workflow_t_spacing_yaml",
        ),
    ],
    indirect=["gui"],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="Testfiles contain only posix paths"
)
def test_load_save_unchanged_complete(gui, workflow, tmp_path, request):
    workflow_yaml = request.getfixturevalue(workflow)
    out_file = tmp_path / "test_workflow_out.yaml"
    with open(workflow_yaml, "r") as f:
        parsed_in = yaml.safe_load(f)
    parsed_in["runner"] = "serial"
    assert parsed_in == gui.config

    gui.save_b.native.click()
    gui.save.path.value = str(out_file)
    gui.save.call_button.native.click()

    with open(out_file, "r") as f:
        parsed_out = yaml.safe_load(f)

    # Step through task_list
    for k, v in parsed_in.items():
        if isinstance(v, list):
            for i, l in enumerate(v):
                if isinstance(l, dict):
                    for kk, vv in l.items():
                        assert vv == parsed_out.get(k)[i].get(kk)

        else:
            assert v == parsed_out.get(k)

    assert parsed_out == parsed_in


@pytest.mark.parametrize("gui", ["workflow_yaml"], indirect=["gui"])
def test_load_missing_config(gui, mock_logger, workflow_yaml):
    gui.change_config.call_button.native.click()

    gui.loader_w.config_path.value = str(workflow_yaml.with_name("NOT_EXISTING.yaml"))
    gui.loader_w.call_button.native.click()
    mock_logger.error.assert_called_with("File does not exist!")

    gui.loader_w.config_path.value = str(
        Path(__file__).resolve().parent.parent / "resources" / "rgb_3D.tif"
    )
    gui.loader_w.call_button.native.click()
    mock_logger.error.assert_called_with("Please provide a yaml file!")


@pytest.mark.parametrize("gui", ["workflow_yaml"], indirect=["gui"])
def test_invalid_config(gui):
    with patch("panseg.workflow_gui.editor.logger") as mock_logger:
        conf: dict = deepcopy(gui.config)

        conf_mod = deepcopy(conf)
        conf_mod.pop("list_tasks")
        gui.config = conf_mod
        gui.show_config()
        mock_logger.error.assert_called_once()
        mock_logger.reset_mock()

        conf_mod = deepcopy(conf)
        conf_mod.pop("infos")
        gui.config = conf_mod
        gui.show_config()
        mock_logger.error.assert_called_once()
        mock_logger.reset_mock()

        conf_mod = deepcopy(conf)
        conf_mod.pop("inputs")
        gui.config = conf_mod
        gui.show_config()
        mock_logger.error.assert_called_once()
        mock_logger.reset_mock()

        gui.config = conf["list_tasks"]
        gui.show_config()
        mock_logger.error.assert_called_once()
        mock_logger.reset_mock()


@pytest.mark.parametrize("gui", ["workflow_yaml"], indirect=["gui"])
def test_toggle_theme(gui):
    gui.toggle_theme()
    gui.toggle_theme()
    gui.toggle_theme()


def _collect_widgets(widget, found):
    found.append(widget)
    if isinstance(widget, Container):
        for child in widget:
            _collect_widgets(child, found)
    return found


def _t_spacing_node(parsed):
    nodes = [t for t in parsed["list_tasks"] if t["func"] == "set_t_spacing_task"]
    assert len(nodes) == 1
    return nodes[0]


def _t_spacing_spin_box(gui):
    spin_boxes = [
        w for w in _collect_widgets(gui.content, []) if isinstance(w, FloatSpinBox)
    ]
    t_spacing_boxes = [w for w in spin_boxes if w.label == "set t spacing"]
    assert len(t_spacing_boxes) == 1
    return t_spacing_boxes[0]


@pytest.mark.parametrize("gui", ["workflow_t_spacing_yaml"], indirect=["gui"])
def test_set_t_spacing_entry_renders_spin_box(gui, workflow_t_spacing_yaml):
    with open(workflow_t_spacing_yaml, "r") as f:
        parsed = yaml.safe_load(f)

    spin_box = _t_spacing_spin_box(gui)
    assert spin_box.value == _t_spacing_node(parsed)["parameters"]["t_spacing"]


@pytest.mark.parametrize("gui", ["workflow_t_spacing_yaml"], indirect=["gui"])
def test_set_t_spacing_entry_saves_parameter(gui, tmp_path):
    out_file = tmp_path / "test_workflow_t_spacing_out.yaml"

    _t_spacing_spin_box(gui).value = 12.5

    gui.save_b.native.click()
    gui.save.path.value = str(out_file)
    gui.save.call_button.native.click()

    with open(out_file, "r") as f:
        parsed_out = yaml.safe_load(f)

    assert _t_spacing_node(parsed_out)["parameters"]["t_spacing"] == 12.5


def _t_spacing_unit_combo(gui):
    combos = [w for w in _collect_widgets(gui.content, []) if isinstance(w, ComboBox)]
    unit_combos = [w for w in combos if w.label == "Unit"]
    assert len(unit_combos) == 1
    return unit_combos[0]


@pytest.mark.parametrize("gui", ["workflow_t_spacing_yaml"], indirect=["gui"])
def test_set_t_spacing_entry_renders_unit_combo(gui, workflow_t_spacing_yaml):
    with open(workflow_t_spacing_yaml, "r") as f:
        parsed = yaml.safe_load(f)

    combo = _t_spacing_unit_combo(gui)
    assert combo.value == _t_spacing_node(parsed)["parameters"]["t_unit"]
    assert list(combo.choices) == list(TIME_UNIT_CHOICES)


def _t_spacing_row(gui):
    """The container that directly groups the spin box and the unit combo."""
    spin_box = _t_spacing_spin_box(gui)
    combo = _t_spacing_unit_combo(gui)

    def walk(container):
        children = list(container)
        if spin_box in children and combo in children:
            return container
        for child in children:
            if isinstance(child, Container):
                found = walk(child)
                if found is not None:
                    return found
        return None

    return walk(gui.content)


@pytest.mark.parametrize("gui", ["workflow_t_spacing_yaml"], indirect=["gui"])
def test_set_t_spacing_number_and_unit_in_horizontal_row(gui):
    """The number and its unit are grouped in one horizontal container."""
    row = _t_spacing_row(gui)
    assert row is not None
    assert row.layout == "horizontal"


def test_set_t_spacing_entry_unit_defaults_for_old_yaml(
    qtbot, workflow_t_spacing_yaml, tmp_path
):
    """Workflow files written before the unit parameter exist carry no
    t_unit: the entry defaults to seconds."""
    with open(workflow_t_spacing_yaml, "r") as f:
        config = yaml.safe_load(f)
    _t_spacing_node(config)["parameters"].pop("t_unit", None)

    old_file = tmp_path / "old_workflow.yaml"
    with open(old_file, "w") as f:
        yaml.safe_dump(config, f)

    gui = Workflow_gui(config_path=old_file, run=False)
    qtbot.addWidget(gui.main_window.native)

    combo = _t_spacing_unit_combo(gui)
    assert combo.value == "s"


@pytest.mark.parametrize("gui", ["workflow_t_spacing_yaml"], indirect=["gui"])
def test_set_t_spacing_entry_saves_unit(gui, tmp_path):
    out_file = tmp_path / "test_workflow_t_spacing_out.yaml"

    _t_spacing_unit_combo(gui).value = "min"

    gui.save_b.native.click()
    gui.save.path.value = str(out_file)
    gui.save.call_button.native.click()

    with open(out_file, "r") as f:
        parsed_out = yaml.safe_load(f)

    assert _t_spacing_node(parsed_out)["parameters"]["t_unit"] == "min"
