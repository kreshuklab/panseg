import logging

try:
    import pyside6
except ImportError:
    pass

from panseg.viewer_napari.loggers import napari_handler

logger_viewer_napari = logging.getLogger(__name__)
logger_viewer_napari.setLevel(logging.INFO)
logger_viewer_napari.addHandler(napari_handler)

# Avoid propagating to loggers.stream_handler
logger_viewer_napari.propagate = False


def log(
    message: str,
    thread: str,
    level: str = "info",
    logger: logging.Logger = logger_viewer_napari,
):
    """Wrapper function for logging into Napari GUI.

    For historical reasons, the `thread` argument is used to identify the widget that is logging.
    """
    logger.log(
        level=logging.getLevelNamesMapping()[level.upper()],
        msg=message,
        extra={"widget_name": thread},
    )


log(
    message=f"Napari logger configured. Napari logger name: {__name__}",
    thread="Napari GUI Logger",
    level="info",
)
