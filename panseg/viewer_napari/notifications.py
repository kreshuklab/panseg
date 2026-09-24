from typing import Any

from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari.utils.notifications import NotificationSeverity
from qtpy.QtCore import QTimer

# Info popups keep napari's built-in default of 4000 ms.
WARNING_ERROR_DISMISS_AFTER = 30_000

_original_close: Any = None
_original_from_notification: Any = None


def _auto_expand(dialog: Any) -> None:
    if dialog.isVisible() and not dialog.property("expanded"):
        dialog.expand()


def _patched_from_notification(cls: Any, notification: Any, parent: Any = None):
    dialog = _original_from_notification(notification, parent)
    if notification.severity >= NotificationSeverity.WARNING:
        # Set before show() runs so the dismiss timer uses this interval.
        dialog.DISMISS_AFTER = WARNING_ERROR_DISMISS_AFTER
    QTimer.singleShot(0, lambda: _auto_expand(dialog))
    return dialog


def _patched_close(self: Any):
    # When a dialog is shown, napari stops the dismiss timers of all other
    # dialogs, expecting close() to restart the previous one's. That restart
    # is dead code in napari: it checks the instance list for self after
    # removing self from it, so it can never match, and older popups stay
    # on screen until hovered or closed by hand. Resume the previous
    # dialog's timer when the most recent one closes.
    instances = NapariQtNotification._instances
    was_most_recent = bool(instances) and instances[-1] is self
    closed = _original_close(self)
    if was_most_recent and self.parent() is not None:
        previous = [d for d in NapariQtNotification._instances if d is not self]
        if previous and not previous[-1].underMouse():
            previous[-1].timer_start()
    return closed


def configure_napari_notifications() -> None:
    """Expand notifications automatically.

    Napari popups show only the first line of a message and disappear after a
    few seconds. This wraps ``NapariQtNotification.from_notification`` so that
    all notifications open expanded. Warning and error notifications stay
    visible for ``WARNING_ERROR_DISMISS_AFTER`` milliseconds before
    auto-hiding; info-level notifications keep the default 4000 ms.

    It also wraps ``close`` to fix a napari bug where a new popup stops the
    dismiss timers of all older popups but never restarts them, leaving them
    on screen forever.
    """
    global _original_close, _original_from_notification
    if _original_from_notification is not None:
        return
    _original_from_notification = NapariQtNotification.from_notification
    NapariQtNotification.from_notification = classmethod(  # pyright: ignore[reportAttributeAccessIssue]
        _patched_from_notification
    )
    _original_close = NapariQtNotification.close
    NapariQtNotification.close = _patched_close  # pyright: ignore[reportAttributeAccessIssue]
