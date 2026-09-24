"""Tests for automatic expansion and timed dismissal of notifications."""

import pytest
from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari.utils.notifications import Notification, NotificationSeverity
from qtpy.QtCore import QPoint
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QApplication, QWidget

from panseg.viewer_napari.notifications import (
    WARNING_ERROR_DISMISS_AFTER,
    configure_napari_notifications,
)

MULTILINE = "First line of an important message\nLine two with details\nLine three"


def _is_visible(dialog):
    try:
        return dialog.isVisible()
    except RuntimeError:
        # The C++ object was deleted once the dialog closed
        # (WA_DeleteOnClose); a deleted dialog is not visible.
        return False


def _park_pointer_away(dialog):
    """Move the pointer off the dialog.

    On a headless X server the pointer rests at the screen center, which the
    expanded error dialog (taller than the others due to its action buttons)
    straddles. The hover stops the dismiss timer (napari's enterEvent) and no
    leaveEvent ever arrives on a headless display, so the timer would stay
    stopped and the dialog would never auto-hide.
    """
    screen = QApplication.primaryScreen().geometry()
    for point in (
        QPoint(0, 0),
        QPoint(screen.width() - 1, 0),
        QPoint(0, screen.height() - 1),
        QPoint(screen.width() - 1, screen.height() - 1),
    ):
        if not dialog.geometry().contains(point):
            QCursor().setPos(point)
            return


@pytest.fixture(autouse=True)
def _patched_notifications():
    # Intended app behavior, so leave the patch in place for the session.
    configure_napari_notifications()
    yield


@pytest.fixture
def show_notification(qapp):
    created = []

    def _show(severity):
        if isinstance(severity, Notification):
            notification = severity
        else:
            notification = Notification(MULTILINE, severity=severity)
        dialog = NapariQtNotification.from_notification(notification)
        # No parent, so the dismiss timer starts regardless of window focus.
        dialog.show()
        _park_pointer_away(dialog)
        created.append(dialog)
        return dialog

    yield _show

    for dialog in created:
        if _is_visible(dialog):
            dialog.close()
    assert not any(dialog in NapariQtNotification._instances for dialog in created)


def test_warning_notification_auto_expands_with_longer_timer(show_notification, qtbot):
    dialog = show_notification("warning")
    qtbot.wait(50)

    assert dialog.property("expanded")
    # Expansion must not stop the dismiss timer (that would keep the popup
    # on screen forever); it keeps running at the longer interval.
    assert dialog.timer.isActive()
    assert dialog.timer.interval() == WARNING_ERROR_DISMISS_AFTER


def test_warning_notification_auto_hides(show_notification, qtbot):
    dialog = show_notification("warning")
    qtbot.wait(50)

    # Shorten the dismiss timer and let it fire to confirm auto-hide works.
    dialog.timer.setInterval(50)
    dialog.timer.start()
    qtbot.wait(300)

    assert not _is_visible(dialog)


def test_error_notification_auto_expands_with_longer_timer(show_notification, qtbot):
    dialog = show_notification(Notification.from_exception(ValueError("boom")))
    qtbot.wait(50)

    assert dialog.property("expanded")
    assert dialog.timer.isActive()
    assert dialog.timer.interval() == WARNING_ERROR_DISMISS_AFTER


def test_info_notification_expands_with_default_timer(show_notification, qtbot):
    dialog = show_notification("info")
    qtbot.wait(50)

    assert dialog.property("expanded")
    assert dialog.timer.isActive()
    assert dialog.timer.interval() == NapariQtNotification.DISMISS_AFTER


def test_configure_is_idempotent(show_notification, qtbot):
    before = (
        NapariQtNotification.__dict__["from_notification"],
        NapariQtNotification.__dict__["close"],
    )
    configure_napari_notifications()
    assert (
        NapariQtNotification.__dict__["from_notification"],
        NapariQtNotification.__dict__["close"],
    ) == before

    dialog = show_notification(NotificationSeverity.WARNING)
    qtbot.wait(50)

    assert dialog.property("expanded")


class _active_parent(QWidget):
    """Parent that always reports an active window.

    Napari only starts the dismiss timer (and only stops the timers of
    older dialogs) when the parent window is active, which is a race in
    offscreen tests.
    """

    def isActiveWindow(self):
        return True


def test_older_notification_auto_hides_after_newest_closes(qapp, qtbot):
    parent = _active_parent()
    parent.show()

    first = NapariQtNotification.from_notification(
        Notification(MULTILINE, severity="warning"), parent
    )
    first.show()
    _park_pointer_away(first)
    second = NapariQtNotification.from_notification(
        Notification(MULTILINE, severity="warning"), parent
    )
    second.show()
    _park_pointer_away(second)
    qtbot.wait(50)

    # Showing the newest dialog stops the older dialog's dismiss timer;
    # napari is supposed to restart it when the newest dialog closes.
    assert not first.timer.isActive()
    assert second.timer.isActive()

    for dialog in (first, second):
        dialog.timer.stop()
        dialog.timer.setInterval(200)
    second.timer.start()

    qtbot.waitUntil(lambda: not _is_visible(second), timeout=5000)
    # The older dialog's timer must have been resumed...
    assert first.timer.isActive()
    # ...so the older dialog auto-hides as well.
    qtbot.waitUntil(lambda: not _is_visible(first), timeout=5000)

    parent.close()
