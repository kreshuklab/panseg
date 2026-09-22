"""Tests for automatic expansion and timed dismissal of notifications."""

import pytest
from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari.utils.notifications import Notification, NotificationSeverity

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
    before = NapariQtNotification.__dict__["from_notification"]
    configure_napari_notifications()
    assert NapariQtNotification.__dict__["from_notification"] is before

    dialog = show_notification(NotificationSeverity.WARNING)
    qtbot.wait(50)

    assert dialog.property("expanded")
