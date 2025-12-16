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
            subprocess.run(
                ["conda", "remove", "plant-seg"],
                input="y\n",
                text=True,
                check=True,
            )
            if PATH_PLANTSEG_MODELS.exists():
                PATH_PLANTSEG_MODELS.rename(".panseg_models")
        except subprocess.CalledProcessError:
            log(
                "Unable to update! If you have installed via git, please update your local repo!",
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
