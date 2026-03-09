import logging
import pprint
import webbrowser
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import yaml
from magicgui import magic_factory, magicgui
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSlider,
    FloatSpinBox,
    Label,
    LineEdit,
    MainWindow,
    PushButton,
    Slider,
    SpinBox,
)
from qt_material import apply_stylesheet
from qtpy import QtCore, QtGui

logger = logging.getLogger(__name__)


class Workflow_widgets:
    def __init__(self):
        self.main_window = MainWindow(layout="vertical", labels=False, scrollable=True)
        self.main_window.create_menu_item(
            menu_name="Theme", item_name="Switch", callback=self.toggle_theme
        )
        self.main_window.create_menu_item(
            menu_name="Help",
            item_name="Documentation",
            callback=self.show_online_docs,
        )
        self.theme = "dark"
        self.apply_theme(self.main_window.native)
        self.content = Container(layout="horizontal", labels=False)
        self.bottom_buttons = Container(layout="horizontal", labels=False)
        self.content.margins = (10, 10, 0, 0)
        self.bottom_buttons.margins = (10, 0, 0, 0)
        self.main_window.extend([self.content, self.bottom_buttons])
        self.changing_fields = {"tasks": {}, "inputs": {}}
        self.change_config.tooltip = (
            "Discard current changes and open a different yaml file."
        )

    def apply_theme(self, window):
        """Applies the dark or light theme to the given window.

        Changes Workflow_widgets.theme to 'dark' or 'light'
        """
        if self.theme == "dark":
            kwargs = {
                "theme": "dark_lightgreen.xml",
                "extra": {
                    "primaryTextColor": "#ffffff",
                },
            }
        else:
            kwargs = {
                "theme": "light_lightgreen.xml",
                "extra": {},
            }
        apply_stylesheet(window, **kwargs)

    def toggle_theme(self):
        if self.theme == "dark":
            self.theme = "light"
            self.apply_theme(self.main_window.native)
        else:
            self.theme = "dark"
            self.apply_theme(self.main_window.native)

    def show_online_docs(self):
        url = "https://kreshuklab.github.io/panseg/"
        webbrowser.open(url, new=0, autoraise=True)

    @magicgui(call_button="Exit")
    def exit(self):
        self.main_window.native.close()  # native only necessary in magicgui<=0.10.0

    @magicgui(call_button="Load new config")
    def change_config(self):
        self.show_loader()

    @magicgui(
        call_button="Load",
        config_path={
            "widget_type": "FileEdit",
            "label": "Path",
            "mode": "r",
            "tooltip": "Choose yaml to load",
            "filter": "*.y[a]ml",
        },
    )
    def loader_w(self, config_path: Path):
        """Loads a workflow from a yaml file"""

        logger.info(f"Loading {config_path}")
        if not config_path.exists():
            logger.error("File does not exist!")
            self.show_loader()
            return

        elif config_path.suffix not in [".yaml", ".yml"]:
            logger.error("Please provide a yaml file!")
            self.show_loader()
            return
        self.config_path = config_path

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            logger.debug(f"LOADED:\n{self.config}")
        self.show_config()

    def fill_config_c(self):
        """Fills content container with everything about one workflow."""

        logger.debug("Filling contents section")
        input_c = Container(layout="vertical", labels=False)
        input_c.margins = (0, 0, 0, 0)
        self.fill_input_c(input_c)
        self.content.append(input_c)  # pyright: ignore

        tasks_c = Container(layout="vertical", labels=False)
        tasks_c.margins = (0, 0, 0, 0)
        self.fill_tasks_c(tasks_c)
        self.content.append(tasks_c)  # pyright: ignore

    def fill_tasks_c(self, cont):
        """Fills tasks section of config container with all tasks trees"""

        task_tree = Task_tree(self.config["list_tasks"], self.config)
        tasks_container = task_tree.build_container()

        label = Label(value="Tasks:\n")
        font = QtGui.QFont()
        font.setBold(True)
        label.native.setFont(font)

        cont.append(label)
        cont.append(tasks_container)

        self.changing_fields["tasks"] = task_tree.changing_fields

    def fill_input_c(self, cont: Container):
        """Fills the input section of the config container"""

        logger.debug("Filling input section")

        # TODO:The inputs section is currently a list of dicts, should be just a dict (#429)
        inputs = self.config["inputs"][0]

        header_c = Container(
            widgets=[
                Label(value="Edit file paths and names:"),
                PushButton(text="Advanced Options", tooltip="Show tasks"),
            ],
            layout="horizontal",
            labels=False,
        )
        header_c[1].clicked.connect(lambda _: self.switch_advanced_view())
        header_c[0].min_width = 400

        font = QtGui.QFont()
        font.setBold(True)
        header_c[0].native.setFont(font)

        cont.append(header_c)
        field_tracker = self.changing_fields["inputs"] = {}

        names_d = list(filter(lambda item: item[0].startswith("name"), inputs.items()))
        inputs_d = list(
            filter(lambda item: item[0].startswith("input"), inputs.items())
        )
        export_d = list(
            filter(lambda item: item[0].startswith("export"), inputs.items())
        )

        for io_type, path in inputs_d:
            file_mode = "r"
            w = self.io_item(
                path={
                    "tooltip": self.config["infos"]["inputs_schema"][io_type][
                        "description"
                    ],
                    "widget_type": "FileEdit",
                    "label": io_type,
                    "value": path,
                    "mode": file_mode,
                }
            )
            cont.append(w)
            field_tracker[io_type] = w
            w.self.bind(self)

        for io_type, path in export_d:
            file_mode = "d"
            w = self.io_item(
                path={
                    "tooltip": self.config["infos"]["inputs_schema"][io_type][
                        "description"
                    ],
                    "widget_type": "FileEdit",
                    "label": io_type,
                    "value": path,
                    "mode": file_mode,
                }
            )
            cont.append(w)
            field_tracker[io_type] = w
            w.self.bind(self)

        for io_type, path in names_d:
            w = self.io_name(
                name={
                    "tooltip": self.config["infos"]["inputs_schema"][io_type][
                        "description"
                    ],
                    "widget_type": "LineEdit",
                    "label": io_type,
                    "value": path,
                }
            )
            cont.append(w)
            field_tracker[io_type] = w
            w.self.bind(self)

        self.reset_b = PushButton(
            text="Reset from file",
            tooltip="Overwrite ALL changes with content from the yaml file.",
        )
        self.reset_b.changed.connect(self.show_config)

        self.save_b = PushButton(text="Save as..", tooltip="Open the save dialog.")
        self.save.path.value = self.config_path
        self.save_b.changed.connect(self.save.show)
        self.save_b.changed.connect(lambda: self.apply_theme(self.save.native))

        controls_c = Container(
            widgets=[self.reset_b, self.save_b],
            layout="horizontal",
        )

        p_formatted = str(self.config_path)
        if len(p_formatted) > 50:
            p_formatted = "[..]" + str(self.config_path)[::-1][50::-1]
        info = Label(value=f"Currently editing:\n{p_formatted}")

        [w.show() for w in controls_c]
        cont.append(info)
        cont.append(controls_c)

    def switch_advanced_view(self, advanced: Optional[bool] = None):
        logger.debug("swich_advanced_view called")
        if advanced is not None:
            self.advanced = not advanced

        if self.advanced:
            self.advanced = False
            self.content[-1].hide()

        else:
            self.advanced = True
            self.content[-1].show()

        mwn = self.main_window.native
        c_size = self.content.native.sizeHint()
        b_size = self.bottom_buttons.native.sizeHint()
        mwn.resize(c_size + b_size + QtCore.QSize(0, 80))

    @magic_factory(call_button=False)
    def io_item(self, path: Path):
        logger.debug(f"Called io_item {path}")
        return str(path)

    @magic_factory(call_button=False)
    def io_name(self, name: str):
        logger.debug(f"Called io_name {name}")
        return name

    @magicgui(
        main_window=True,
        call_button="Save",
        path={
            "label": "Save to",
            "mode": "w",
            "tooltip": "Choose where to save the workflow",
            "filter": "*.y[a]ml",
        },
    )
    def save(self, path: Path):
        logger.debug(f"Called save {path}")
        if path.suffix not in [".yaml", ".yml"]:
            logging.warning("Please save as a yaml file!")
            return

        output = self.config.copy()
        for field, w in self.changing_fields["inputs"].items():
            output["inputs"][0][field] = w()
        logger.debug(f"IO part: {output['inputs'][0]}")

        for id, w in self.changing_fields["tasks"].items():
            for i, task in enumerate(output["list_tasks"]):
                if task["id"] == id:
                    update = w()
                    print("Task: ", task)
                    pprint.pp(update)

                    for k, v in update.items():
                        if isinstance(v, dict):
                            output["list_tasks"][i][k].update(v)
                        else:
                            output["list_tasks"][i].update(update)

        logger.debug(pprint.pformat(f"Tasks part: {output['list_tasks']}"))

        with open(path, "w") as f:
            f.write(yaml.safe_dump(output))
            logger.debug(f"Successfully written to {path}")

        save.hide()  # type: ignore # noqa

    @abstractmethod
    def show_config(self):
        pass

    @abstractmethod
    def show_loader(self):
        pass


class Task_node:
    def __init__(self, node: dict):
        self.parents = set()
        self.children = set()
        self.changing_fields = {}

        self.node_type: str
        self.func: str
        self.images_inputs: dict[str, str]
        self.outputs: list[str]
        self.parameters: dict
        self.id: str

        for k, v in node.items():
            setattr(self, k, v)

    def add_parent(self, parent_id: str):
        self.parents.add(parent_id)

    def add_child(self, child_id: str):
        self.children.add(child_id)

    def get_node_widget(self, config: dict):
        label = " ".join(f"{self.func}".split("_")[:-1])

        # @@@@@@ IO tasks @@@@@@
        if self.func == "import_image_task":
            w = ComboBox(
                label=label,
                value=self.images_inputs["input_path"],
                choices=list(
                    filter(
                        lambda s: s.startswith("input"),
                        config["inputs"][0],
                    )
                ),
            )
            # cont.append(Label(value=label))
            self.changing_fields[self.id] = lambda: {
                "images_inputs": {"input_path": w.value}
            }
            return Container(widgets=[w])

        elif self.func == "export_image_task":
            export_cont = Container(label=label)
            export_cont.append(
                ComboBox(  # pyright: ignore
                    label="Export slot",
                    value=self.images_inputs["export_directory"],
                    choices=list(
                        filter(
                            lambda s: s.startswith("export"),
                            config["inputs"][0],
                        )
                    ),
                )
            )
            export_cont.append(
                ComboBox(  # pyright: ignore
                    label="Name slot",
                    value=self.images_inputs["name_pattern"],
                    choices=list(
                        filter(
                            lambda s: s.startswith("name"),
                            config["inputs"][0],
                        )
                    ),
                )
            )
            export_cont.append(
                ComboBox(  # pyright: ignore
                    label="Export Mesh as",
                    value=self.parameters["export_mesh"],
                    choices=[None, "glb", "obj", "ply"],
                )
            )
            export_cont.append(
                CheckBox(  # pyright: ignore
                    label="Close Meshes",
                    value=self.parameters["close_mesh"],
                )
            )
            self.changing_fields[self.id] = lambda: {
                "images_inputs": {
                    "export_directory": export_cont[0].value,
                    "name_pattern": export_cont[1].value,
                },
                "parameters": {
                    "export_mesh": export_cont[2].value,
                    "close_mesh": export_cont[3].value,
                },
            }
            return Container(widgets=[export_cont])

        # @@@@@@ Preprocessing @@@@@@
        elif self.func == "gaussian_smoothing_task":
            w = FloatSlider(
                label=label,
                value=self.parameters["sigma"],
                min=0.1,
                max=10,
            )
            self.changing_fields[self.id] = lambda: {"parameters": {"sigma": w.value}}
            return Container(widgets=[w])

        elif self.func == "set_voxel_size_task":
            x, y, z = (
                FloatSpinBox(value=self.parameters["voxel_size"][0]),
                FloatSpinBox(value=self.parameters["voxel_size"][1]),
                FloatSpinBox(value=self.parameters["voxel_size"][2]),
            )
            w = Container(
                label=label,
                layout="horizontal",
                labels=False,
                widgets=(x, y, z),
            )
            self.changing_fields[self.id] = lambda: {
                "parameters": {
                    "voxel_size": [x.value, y.value, z.value],
                }
            }

            return Container(widgets=[w])

        elif self.func == "image_rescale_to_shape_task":
            x, y, z = (
                SpinBox(value=self.parameters["new_shape"][0]),
                SpinBox(value=self.parameters["new_shape"][1]),
                SpinBox(value=self.parameters["new_shape"][2]),
            )
            w = Container(
                label=label,
                layout="horizontal",
                labels=False,
                widgets=(x, y, z),
            )
            self.changing_fields[self.id] = lambda: {
                "parameters": {
                    "new_shape": [x.value, y.value, z.value],
                }
            }

            return Container(widgets=[w])

        elif self.func == "image_rescale_to_voxel_size_task":
            x, y, z, unit = (
                FloatSpinBox(value=self.parameters["new_voxels_size"][0], step=0.001),
                FloatSpinBox(value=self.parameters["new_voxels_size"][1], step=0.001),
                FloatSpinBox(value=self.parameters["new_voxels_size"][2], step=0.001),
                LineEdit(value=self.parameters["new_unit"]),
            )
            w = Container(
                label=label,
                layout="horizontal",
                labels=False,
                widgets=(x, y, z, unit),
            )
            self.changing_fields[self.id] = lambda: {
                "parameters": {
                    "new_unit": unit.value,
                    "new_voxels_size": [x.value, y.value, z.value],
                }
            }
            return Container(widgets=[w])

        elif self.func == "remove_false_positives_by_foreground_probability_task":
            w = FloatSlider(
                label="Remove false-positives\nthreshold",
                value=self.parameters["threshold"],
                min=0.0,
                max=1.0,
            )

            self.changing_fields[self.id] = lambda: {
                "parameters": {"threshold": w.value}
            }
            return Container(widgets=[w])

        # @@@@@@ Segmentation @@@@@@
        elif self.func == "unet_prediction_task":
            unet_cont = Container(label=label)
            unet_cont.append(
                LineEdit(  # pyright: ignore
                    label="Model:",
                    value=self.parameters["model_name"],
                ),
            )
            unet_cont.append(
                LineEdit(  # pyright: ignore
                    label="Device:",
                    value=self.parameters["device"],
                ),
            )
            self.changing_fields[self.id] = lambda: {
                "parameters": {
                    "model_name": unet_cont[0].value,
                    "device": unet_cont[1].value,
                }
            }
            return Container(widgets=[unet_cont])

        elif self.func == "biio_prediction_task":
            biio_cont = Container(label=label)
            biio_cont.append(
                LineEdit(  # pyright: ignore
                    label="Model:",
                    value=self.parameters["model_id"],
                ),
            )
            self.changing_fields[self.id] = lambda: {
                "parameters": {"model_id": biio_cont[-1].value}
            }
            return Container(widgets=[biio_cont])

        elif self.func == "dt_watershed_task":
            threshold = FloatSlider(
                label="Threshold",
                value=self.parameters["threshold"],
                min=0.0,
                max=1.0,
            )
            minsize = Slider(
                label="Min size",
                value=self.parameters["min_size"],
                min=1,
                max=1000,
            )
            cont = Container(
                widgets=[Label(value=label), threshold, minsize],
                layout="vertical",
            )
            cont.margins = (0, 0, 0, 0)

            self.changing_fields[self.id] = lambda: {
                "parameters": {"threshold": threshold.value, "min_size": minsize.value}
            }
            return Container(widgets=[cont])

        elif self.func == "clustering_segmentation_task":
            beta = FloatSlider(
                label="Beta",
                value=self.parameters["beta"],
                min=0.0,
                max=1.0,
            )
            minsize = Slider(
                label="Min size",
                value=self.parameters["post_min_size"],
                min=1,
                max=1000,
            )
            mode = ComboBox(
                label="Mode",
                value=self.parameters["mode"],
                choices=["gasp", "multicut", "mutex_ws", "lmc"],
            )
            cont = Container(
                widgets=[Label(value="Clustering Segmentation"), beta, minsize, mode],
                labels=True,
                layout="vertical",
            )
            cont.margins = (0, 0, 0, 0)

            self.changing_fields[self.id] = lambda: {
                "parameters": {
                    "beta": beta.value,
                    "post_min_size": minsize.value,
                    "mode": mode.value,
                }
            }
            return Container(widgets=[cont])

        elif self.func == "aio_watershed_task":
            threshold = FloatSlider(
                label="Threshold",
                value=self.parameters["threshold"],
                min=0.0,
                max=1.0,
            )
            minsize = Slider(
                label="Min size",
                value=self.parameters["min_size"],
                min=1,
                max=1000,
            )
            beta = FloatSlider(
                label="Beta",
                value=self.parameters["beta"],
                min=0.0,
                max=1.0,
            )
            mode = ComboBox(
                label="Mode",
                value=self.parameters["mode"],
                choices=["gasp", "multicut", "mutex_ws", "lmc"],
            )
            cont = Container(
                widgets=[Label(value=label), threshold, minsize, beta, mode],
                labels=True,
                layout="vertical",
            )
            cont.margins = (0, 0, 0, 0)

            self.changing_fields[self.id] = lambda: {
                "parameters": {
                    "threshold": threshold.value,
                    "min_size": minsize.value,
                    "beta": beta.value,
                    "mode": mode.value,
                }
            }
            return Container(widgets=[cont])

        elif self.func == "set_biggest_instance_to_zero_task":
            return Container(widgets=[Label(value="Set biggest instance to zero")])

        # @@@@@@ Catch-all @@@@@@
        else:
            return Container(
                widgets=[
                    Label(
                        value=pprint.pformat(getattr(self, "parameters")), label=label
                    )
                ]
            )


class Task_tree:
    def __init__(self, task_list: list[dict], config: dict):
        self.task_list = task_list
        self.config = config
        self.nodes = []
        self.roots = []
        self.leaves = []
        self.tree_ids = []
        self.changing_fields = {}

        for task in task_list:
            node = Task_node(task)
            self.nodes.append(node)
            if node.node_type == "root":
                self.roots.append(node)
            elif node.node_type == "leaf":
                self.leaves.append(node)

        for node in self.nodes:
            for output in node.outputs:
                for task in task_list:
                    if output in task["images_inputs"].values():
                        node.add_child(task["id"])

            for input in node.images_inputs.values():
                for task in task_list:
                    if input in task["outputs"]:
                        node.add_parent(task["id"])

        self._id_dict = {node.id: node for node in self.nodes}

    def from_id(self, id):
        return self._id_dict[id]

    def build_container(self):
        logger.debug("### Building new task tree widget ###")
        self.tree_ids = []
        super_cont = Container(layout="horizontal", labels=False)
        super_cont.margins = (0, 0, 0, 0)
        for node in self.roots:
            cont = Container(layout="vertical", labels=False)
            cont.margins = (0, 0, 0, 0)
            self._add_node_top_down(cont, node, nest=(len(self.nodes) > 8))
            super_cont.append(cont)  # pyright: ignore
        logger.debug("### Done building new task tree widget ###")
        return super_cont

    def _add_node_top_down(self, cont: Container, node: Task_node, nest=False):
        logger.debug(f"Adding {node.func}")
        w = node.get_node_widget(self.config)
        w.margins = (0, 0, 0, 0)
        cont.append(w)

        self.tree_ids.append(node.id)
        self.changing_fields.update(node.changing_fields)

        if len(node.children) == 0:
            return
        elif len(node.children) == 1:
            child = node.children.pop()
            if child in self.tree_ids:
                # avoid duplicates
                logger.debug(f"Ignoring a child {self.from_id(child).func}")
                return
            self._add_node_top_down(cont, self.from_id(child), nest)
        else:
            super_cont = Container(layout="horizontal", labels=True)
            for child in node.children:
                if child in self.tree_ids:
                    # avoid duplicates
                    logger.debug(f"Ignoring a child {self.from_id(child).func}")
                    continue
                if nest:
                    sub_cont = Container(layout="vertical", labels=False)
                    sub_cont.margins = (0, 0, 0, 0)
                    super_cont.append(sub_cont)  # pyright: ignore
                    self._add_node_top_down(sub_cont, self.from_id(child), nest=False)
                else:
                    self._add_node_top_down(cont, self.from_id(child), nest=False)

            cont.append(super_cont)
        return
