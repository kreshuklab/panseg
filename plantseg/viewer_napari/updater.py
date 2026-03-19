import subprocess

from napari.qt.threading import thread_worker

from plantseg import PATH_PLANTSEG_MODELS
from plantseg.viewer_napari import log


def update():
    @thread_worker
    def _update():
        try:
            # TODO: make sure correct package is installed!
            subprocess.run(
                ["conda", "install", "panseg", "-c", "conda-forge"],
                input="y\n",
                text=True,
                check=True,
            )
            log("Panseg installed, removing plantseg", thread="updater", level="INFO")
            subprocess.run(
                ["conda", "remove", "plant-seg", "plantseg"],
                input="y\n",
                text=True,
                check=True,
            )
            log("Old Plantseg uninstalled", thread="updater", level="INFO")
            if PATH_PLANTSEG_MODELS.exists():
                new_modelpath = PATH_PLANTSEG_MODELS.rename(".panseg_models")
                log(
                    f"Moved Plantseg models to {new_modelpath}",
                    thread="updater",
                    level="INFO",
                )
        except subprocess.CalledProcessError as e:
            log(
                f"Unable to update! If you have installed via git, please update your local repo!\n{e}",
                thread="updater",
                level="WARNING",
            )
            return

        log("Update finished, please restart!", thread="updater", level="INFO")

    log(
        "Starting update, might take a while!\nCheck the progress in the terminal.",
        thread="updater",
        level="INFO",
    )
    _update().start()
