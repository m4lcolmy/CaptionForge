"""Run CaptionForge as a desktop application: one window, no terminal.

Desktop mode serves the same local page ``captionforge web`` serves. The only
differences are that nothing has to stay open in a terminal, a second launch
re-uses the server already running, and the server stops on its own once the
window that was using it is gone.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Config
from app.core.constants import APP_NAME
from app.core.exceptions import ConfigurationError
from app.core.logging_config import get_logger
from app.models.job import JobStatus

# The page is a single 40rem column, so the window only needs to be a little
# wider than that; the height leaves room for panels and a taskbar on 1080p.
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 800

# The page pings /api/health while it is open. Losing several pings in a row
# means the window is gone, so the server has nobody left to serve.
IDLE_GRACE_SECONDS = 90.0
WATCHDOG_INTERVAL_SECONDS = 5.0
STARTUP_TIMEOUT_SECONDS = 20.0
PROBE_TIMEOUT_SECONDS = 1.5

# Chromium-family browsers can open a page as a plain window with no tab strip
# and no address bar, which is as close to a native window as a browser gets.
APP_MODE_BROWSERS = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "brave-browser",
    "microsoft-edge",
    "vivaldi",
)

BUSY_STATUSES = frozenset(
    {
        JobStatus.PENDING,
        JobStatus.RUNNING,
        JobStatus.PREPARING_AUDIO,
        JobStatus.LOADING_MODEL,
        JobStatus.TRANSCRIBING,
        JobStatus.POST_PROCESSING,
        JobStatus.EXPORTING,
    }
)


@dataclass(frozen=True)
class Session:
    """Where a running desktop server is listening, and how to talk to it."""

    host: str
    port: int
    token: str

    @property
    def url(self) -> str:
        """The page address, token included, exactly as the window opens it."""
        return f"http://{self.host}:{self.port}/?t={self.token}"

    @property
    def health_url(self) -> str:
        """The endpoint used both as a readiness probe and as the heartbeat."""
        return f"http://{self.host}:{self.port}/api/health"


class Activity:
    """Remember when the page last spoke to the server."""

    def __init__(self) -> None:
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def touch(self) -> None:
        """Record a request that just arrived."""
        with self._lock:
            self._last = time.monotonic()

    def idle_seconds(self) -> float:
        """Seconds since the last request."""
        with self._lock:
            return time.monotonic() - self._last


class ActivityMiddleware:
    """ASGI wrapper that reports every request to an :class:`Activity`."""

    def __init__(self, app: Any, activity: Activity) -> None:
        self._app = app
        self._activity = activity

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Note the request, then hand it to the application unchanged."""
        if scope.get("type") == "http":
            self._activity.touch()
        await self._app(scope, receive, send)


def session_file() -> Path:
    """The file that records the running server, one per user."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    directory = Path(runtime) if runtime else Path(tempfile.gettempdir())
    suffix = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
    return directory / f"captionforge-desktop{suffix}.json"


def launch(
    config: Config,
    *,
    port: int = 0,
    window: bool = True,
    open_page: bool = True,
    announce: Callable[[str, bool], None] | None = None,
) -> None:
    """Serve the page and keep serving it until its window is closed."""
    running = running_session()
    if running is not None:
        if announce is not None:
            announce(running.url, True)
        if open_page:
            present(running.url, window=window)
        get_logger().info("Desktop launch joined the running server")
        return

    try:
        import uvicorn

        from app.interfaces.web.security import generate_token
        from app.interfaces.web.server import HOST, create_app, find_free_port
    except ImportError as exc:
        raise ConfigurationError(
            "The desktop app needs the web packages. Install them with "
            'pip install "captionforge[web]"',
            details=str(exc),
        ) from exc

    session = Session(HOST, port or find_free_port(), generate_token())
    application = create_app(config, token=session.token, port=session.port)
    activity = Activity()
    application.add_middleware(ActivityMiddleware, activity=activity)

    server = uvicorn.Server(
        uvicorn.Config(
            application, host=session.host, port=session.port, log_level="warning"
        )
    )
    serving = threading.Thread(target=server.run, name="captionforge-web", daemon=True)
    serving.start()
    stop_on_termination(server)

    try:
        if not wait_until_ready(session, serving):
            raise ConfigurationError(
                "CaptionForge could not start its local server.",
                details=f"No answer from {session.health_url}",
            )
        write_session(session)
        if announce is not None:
            announce(session.url, False)
        watching = threading.Thread(
            target=watch_for_idle,
            args=(server, activity, application.state.registry),
            name="captionforge-idle",
            daemon=True,
        )
        watching.start()
        if open_page:
            present(session.url, window=window, block_until=serving)
        else:
            wait_for(serving)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        serving.join(timeout=10)
        clear_session(session)


def stop_on_termination(server: Any) -> None:
    """Close the app on logout the same way closing the window closes it."""
    if threading.current_thread() is not threading.main_thread():
        return

    def stop(_number: int, _frame: Any) -> None:
        server.should_exit = True

    with contextlib.suppress(ValueError, OSError, AttributeError):
        signal.signal(signal.SIGTERM, stop)


def running_session() -> Session | None:
    """Return the server already serving this user, or None."""
    path = session_file()
    try:
        recorded = json.loads(path.read_text("utf-8"))
        session = Session(
            str(recorded["host"]), int(recorded["port"]), str(recorded["token"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if answers(session):
        return session
    path.unlink(missing_ok=True)
    return None


def answers(session: Session) -> bool:
    """Ask a recorded server whether it is still CaptionForge."""
    from app.interfaces.web.security import TOKEN_HEADER

    request = urllib.request.Request(
        session.health_url, headers={TOKEN_HEADER: session.token}
    )
    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as reply:
            payload = json.loads(reply.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False
    return bool(payload.get("app") == APP_NAME)


def write_session(session: Session) -> None:
    """Record this server so a second launch joins it instead of duplicating it."""
    path = session_file()
    try:
        path.write_text(
            json.dumps(
                {"host": session.host, "port": session.port, "token": session.token}
            ),
            encoding="utf-8",
        )
        path.chmod(0o600)
    except OSError as exc:
        # Losing the record only costs a duplicate server on the next launch.
        get_logger().warning("Could not record the desktop session: {}", exc)


def clear_session(session: Session) -> None:
    """Remove this server's record, leaving another instance's record alone."""
    path = session_file()
    try:
        recorded = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return
    if int(recorded.get("port", 0)) == session.port:
        path.unlink(missing_ok=True)


def wait_until_ready(session: Session, serving: threading.Thread) -> bool:
    """Poll the health endpoint until the server answers or gives up."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not serving.is_alive():
            return False
        if answers(session):
            return True
        time.sleep(0.1)
    return False


def watch_for_idle(server: Any, activity: Activity, registry: Any) -> None:
    """Stop the server once the page has stopped talking to it."""
    while not server.should_exit:
        time.sleep(WATCHDOG_INTERVAL_SECONDS)
        if activity.idle_seconds() < IDLE_GRACE_SECONDS:
            continue
        if busy(registry):
            # Work outlives its window: a transcription started before the
            # window closed still finishes and still writes its files.
            activity.touch()
            continue
        get_logger().info("Desktop server stopping: no page has asked for anything")
        server.should_exit = True


def busy(registry: Any) -> bool:
    """Report whether any job is still queued or running."""
    return any(record.status in BUSY_STATUSES for record in registry.recent())


@dataclass(frozen=True)
class Geometry:
    """A rectangle on the screen, in pixels."""

    x: int
    y: int
    width: int
    height: int


def window_geometry(screen: Geometry | None) -> Geometry:
    """Size the new window to fit the screen and place it in the middle."""
    if screen is None:
        return Geometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    width = min(WINDOW_WIDTH, int(screen.width * 0.9))
    height = min(WINDOW_HEIGHT, int(screen.height * 0.9))
    return Geometry(
        screen.x + (screen.width - width) // 2,
        screen.y + (screen.height - height) // 2,
        width,
        height,
    )


def primary_screen() -> Geometry | None:
    """Find the main monitor, or None when the screen cannot be measured."""
    return xrandr_primary() or tk_screen()


def xrandr_primary() -> Geometry | None:
    """Read the primary monitor from xrandr, so a second monitor is ignored."""
    if shutil.which("xrandr") is None:
        return None
    try:
        output = subprocess.run(
            ["xrandr", "--query"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_xrandr(output)


def parse_xrandr(output: str) -> Geometry | None:
    """Pick the primary monitor's rectangle, else the first connected one."""
    found: list[tuple[bool, Geometry]] = []
    for line in output.splitlines():
        match = re.search(r" connected (primary )?(\d+)x(\d+)\+(\d+)\+(\d+)", line)
        if match:
            width, height, x, y = (int(part) for part in match.groups()[1:])
            found.append((bool(match.group(1)), Geometry(x, y, width, height)))
    for primary, geometry in found:
        if primary:
            return geometry
    return found[0][1] if found else None


def tk_screen() -> Geometry | None:
    """Measure the screen with tkinter when xrandr is not there to ask."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        try:
            return Geometry(0, 0, root.winfo_screenwidth(), root.winfo_screenheight())
        finally:
            root.destroy()
    except Exception:  # noqa: BLE001 - no display or no Tk: just skip centring
        return None


def present(
    url: str, *, window: bool, block_until: threading.Thread | None = None
) -> str:
    """Show the page, then block for as long as that window lives."""
    viewer = pywebview() if window else None
    if viewer is not None:
        place = window_geometry(primary_screen())
        viewer.create_window(
            APP_NAME,
            url,
            width=place.width,
            height=place.height,
            x=place.x,
            y=place.y,
        )
        viewer.start()
        return "native window"
    description = open_browser(url, app_mode=window)
    if block_until is not None:
        wait_for(block_until)
    return description


def pywebview() -> Any | None:
    """Return the optional native-window library, or None when it is absent."""
    try:
        import webview
    except ImportError:
        return None
    return webview


def open_browser(url: str, *, app_mode: bool) -> str:
    """Open a chromeless browser window when one is available, else a tab."""
    if app_mode:
        binary = app_mode_browser()
        if binary is not None:
            # Only the window being opened is placed; one already open stays put.
            place = window_geometry(primary_screen())
            try:
                subprocess.Popen(
                    [
                        binary,
                        f"--app={url}",
                        f"--window-size={place.width},{place.height}",
                        f"--window-position={place.x},{place.y}",
                        # A browser that is already open would take over the
                        # launch and ignore the size; its own profile keeps
                        # this window a separate instance that honours it.
                        f"--user-data-dir={browser_profile_dir()}",
                        "--no-first-run",
                        "--no-default-browser-check",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return f"{Path(binary).name} window"
            except OSError as exc:
                get_logger().warning("Could not open an app window: {}", exc)
    webbrowser.open(url)
    return "default browser"


def browser_profile_dir() -> Path:
    """The browser profile the app window uses, apart from the user's own."""
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "share"
    return root / "captionforge" / "browser-window"


def app_mode_browser() -> str | None:
    """Find a Chromium-family browser that can open a page as its own window."""
    for candidate in APP_MODE_BROWSERS:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def wait_for(serving: threading.Thread) -> None:
    """Block until the server thread stops, staying responsive to Ctrl+C."""
    while serving.is_alive():
        serving.join(timeout=1.0)


def main() -> None:
    """Entry point for the ``captionforge-desktop`` launcher."""
    from app.core.logging_config import configure_logging

    config = Config.load()
    configure_logging(config)
    launch(config)


if __name__ == "__main__":
    main()
