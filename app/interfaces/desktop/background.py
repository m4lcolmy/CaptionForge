"""Run slow work off the window's thread and hand the answer back to it.

A lookup waits on YouTube and an update waits on pip, so both run on a thread
of their own. Their result, or their exception, is delivered on the window's
thread, where it is safe to touch widgets.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal, Slot


class _Relay(QObject):
    """Carries one result across threads, then deletes itself."""

    done = Signal(object)
    failed = Signal(object)

    def __init__(
        self,
        on_done: Callable[[Any], None],
        on_failed: Callable[[BaseException], None],
        parent: QObject,
    ) -> None:
        super().__init__(parent)
        self._on_done = on_done
        self._on_failed = on_failed
        # Queued explicitly: the signal is emitted on the worker thread and
        # the callbacks must run on the thread this relay lives on.
        self.done.connect(self._deliver_done, Qt.ConnectionType.QueuedConnection)
        self.failed.connect(self._deliver_failed, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def _deliver_done(self, value: object) -> None:
        try:
            self._on_done(value)
        finally:
            self.deleteLater()

    @Slot(object)
    def _deliver_failed(self, error: object) -> None:
        try:
            assert isinstance(error, BaseException)
            self._on_failed(error)
        finally:
            self.deleteLater()


def run_in_background(
    work: Callable[[], Any],
    on_done: Callable[[Any], None],
    on_failed: Callable[[BaseException], None],
    *,
    parent: QObject,
    name: str = "captionforge-desktop",
) -> None:
    """Start ``work`` on a daemon thread; answer on ``parent``'s thread."""
    relay = _Relay(on_done, on_failed, parent)

    def run() -> None:
        try:
            result = work()
        except BaseException as exc:  # noqa: BLE001 - handed to on_failed
            relay.failed.emit(exc)
        else:
            relay.done.emit(result)

    threading.Thread(target=run, name=name, daemon=True).start()
