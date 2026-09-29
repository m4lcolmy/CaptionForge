"""Start the desktop app: one Qt window, one running copy per person.

A second launch does not open a second window. It finds the first copy through
a local socket, asks it to come forward, and exits. The lock that decides which
copy is first is released by the operating system if a copy crashes, so a
crash never leaves CaptionForge refusing to start.
"""

from __future__ import annotations

import os
import signal
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QLockFile,
    QMessageLogContext,
    QObject,
    QRect,
    QTimer,
    QtMsgType,
    Signal,
    qInstallMessageHandler,
)
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from app.core.config import Config
from app.core.constants import APP_NAME, VERSION
from app.core.logging_config import get_logger
from app.interfaces.desktop import theme
from app.interfaces.desktop.window import MainWindow
from app.interfaces.launcher import ENTRY_NAME, ICON_SOURCE

# The page is a single 40rem column, so the window only needs to be a little
# wider than that; the height leaves room for panels and a taskbar on 1080p.
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 800

# A first copy that holds the lock but is still starting needs a moment
# before its socket answers.
KNOCK_SECONDS = 5.0
KNOCK_TIMEOUT_MS = 500


def window_geometry(screen: QRect) -> QRect:
    """Size the window to fit the screen and place it in the middle."""
    width = min(WINDOW_WIDTH, int(screen.width() * 0.9))
    height = min(WINDOW_HEIGHT, int(screen.height() * 0.9))
    return QRect(
        screen.x() + (screen.width() - width) // 2,
        screen.y() + (screen.height() - height) // 2,
        width,
        height,
    )


def runtime_directory() -> Path:
    """Where per-session files go: the user's runtime folder when there is one."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    return Path(runtime) if runtime else Path(tempfile.gettempdir())


def _suffix() -> str:
    return f"-{os.getuid()}" if hasattr(os, "getuid") else ""


def lock_path() -> Path:
    """The lock only the first running copy holds, one per user."""
    return runtime_directory() / f"captionforge-desktop{_suffix()}.lock"


def instance_address() -> str:
    """The socket a second launch knocks on, one per user."""
    if os.name == "nt":
        # A named pipe, which is per-session already.
        return f"captionforge-desktop-{os.environ.get('USERNAME', 'user')}"
    return str(runtime_directory() / f"captionforge-desktop{_suffix()}.socket")


class SingleInstance(QObject):
    """Make sure only one copy runs, and let later launches wake it."""

    # A later launch asked to see the window.
    activated = Signal()

    def __init__(self, address: str, lock: Path) -> None:
        super().__init__()
        self._address = address
        self._lock = QLockFile(str(lock))
        # The lock is held for as long as the app runs, which can be hours:
        # only a dead owner makes it stale, never its age.
        self._lock.setStaleLockTime(0)
        self._server: QLocalServer | None = None

    def claim(self) -> bool:
        """Return True to run here, or False once the running copy has been woken."""
        if self._lock.tryLock(0):
            self._listen()
            return True
        deadline = time.monotonic() + KNOCK_SECONDS
        while time.monotonic() < deadline:
            if self._knock():
                return False
            time.sleep(0.1)
        # The owner holds the lock but never answers. Starting anyway is better
        # than leaving the person with nothing on screen.
        get_logger().warning("The running CaptionForge did not answer; starting anew")
        return True

    def release(self) -> None:
        """Stop answering and let the next launch be the first."""
        if self._server is not None:
            self._server.close()
            self._server = None
        self._lock.unlock()

    def _listen(self) -> None:
        # A socket file left by a crash would make listen() fail.
        QLocalServer.removeServer(self._address)
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        server.newConnection.connect(self._answer)
        if not server.listen(self._address):
            get_logger().warning(
                "A second launch will not find this window: {}", server.errorString()
            )
        self._server = server

    def _answer(self) -> None:
        assert self._server is not None
        while (connection := self._server.nextPendingConnection()) is not None:
            # Only this user can reach the socket, and knocking is all it does.
            connection.disconnected.connect(connection.deleteLater)
            self.activated.emit()

    def _knock(self) -> bool:
        socket = QLocalSocket()
        socket.connectToServer(self._address)
        if not socket.waitForConnected(KNOCK_TIMEOUT_MS):
            return False
        socket.write(b"show\n")
        socket.waitForBytesWritten(KNOCK_TIMEOUT_MS)
        socket.disconnectFromServer()
        return True


def _log_qt_message(
    mode: QtMsgType, _context: QMessageLogContext, message: str
) -> None:
    """Send Qt's own warnings to the log file instead of a terminal nobody sees."""
    if mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        get_logger().error("Qt: {}", message)
    else:
        get_logger().debug("Qt: {}", message)


def application() -> QApplication:
    """The one QApplication, named so the desktop matches it to its entry."""
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    # X11 reads the window's WM_CLASS instance from argv[0], and GNOME matches
    # that to StartupWMClass in captionforge.desktop. "python" would match
    # nothing, and the dock would show a generic icon.
    created = QApplication([ENTRY_NAME])
    created.setApplicationName(APP_NAME)
    created.setApplicationVersion(VERSION)
    created.setDesktopFileName(ENTRY_NAME)
    created.setWindowIcon(QIcon(str(ICON_SOURCE)))
    # Closing the window while a job runs only hides it; the window decides
    # when the app is done.
    created.setQuitOnLastWindowClosed(False)
    return created


@contextmanager
def quit_on_signals(target: QApplication) -> Iterator[None]:
    """Stop on Ctrl+C in a terminal, or on SIGTERM at logout.

    Python runs signal handlers only when it has the interpreter, and Qt's
    event loop is native code, so a timer hands control back a few times a
    second.
    """

    def stop(_number: int, _frame: Any) -> None:
        target.quit()

    previous: dict[signal.Signals, Any] = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        with suppress(ValueError, OSError):
            previous[number] = signal.signal(number, stop)
    wake = QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(250)
    try:
        yield
    finally:
        wake.stop()
        for number, handler in previous.items():
            with suppress(ValueError, OSError):
                signal.signal(number, handler)


def run(config: Config) -> bool:
    """Show the window until it is done; True if a running copy was woken."""
    app = application()
    instance = SingleInstance(instance_address(), lock_path())
    if not instance.claim():
        get_logger().info("Desktop launch woke the running window")
        return True

    qInstallMessageHandler(_log_qt_message)
    theme.theme().install(app)
    window = MainWindow(config)
    instance.activated.connect(window.bring_forward)
    window.finished.connect(app.quit)
    screen = QGuiApplication.primaryScreen()
    if screen is not None:
        window.setGeometry(window_geometry(screen.availableGeometry()))
    window.show()
    get_logger().info("Desktop window opened")
    try:
        with quit_on_signals(app):
            app.exec()
    finally:
        window.shutdown()
        instance.release()
    return False
