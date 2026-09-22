from typing import Any

from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari.utils.notifications import NotificationSeverity
from qtpy.QtCore import QTimer

# How long (ms) warning/error popups stay visible before auto-hiding. Info
# popups keep napari's built-in default of 4000 ms.
WARNING_ERROR_DISMISS_AFTER = 30_000

_original_from_notification: Any = None


def _auto_expand(dialog: Any) -> None:
    # Use expand() (not toggle_expansion()): the latter also stops the
    # auto-hide timer, which would keep the popup on screen forever.
    if dialog.isVisible() and not dialog.property("expanded"):
        dialog.expand()


def _patched_from_notification(cls: Any, notification: Any, parent: Any = None):
    dialog = _original_from_notification(notification, parent)
    if notification.severity >= NotificationSeverity.WARNING:
        # Set before show() runs so the dismiss timer uses this interval.
        dialog.DISMISS_AFTER = WARNING_ERROR_DISMISS_AFTER
    QTimer.singleShot(0, lambda: _auto_expand(dialog))
    return dialog


def configure_napari_notifications() -> None:
    """Expand notifications automatically.

    Napari popups show only the first line of a message and disappear after a
    few seconds. This wraps ``NapariQtNotification.from_notification`` so that
    all notifications open expanded. Warning and error notifications stay
    visible for ``WARNING_ERROR_DISMISS_AFTER`` milliseconds before
    auto-hiding; info-level notifications keep the default 4000 ms.
    """
    global _original_from_notification
    if _original_from_notification is not None:
        return
    _original_from_notification = NapariQtNotification.from_notification
    NapariQtNotification.from_notification = classmethod(  # pyright: ignore[reportAttributeAccessIssue]
        _patched_from_notification
    )
