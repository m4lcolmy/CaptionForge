"""Offline tests for desktop mode and the desktop entry it installs."""

import json
import platform
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Config
from app.interfaces import desktop, launcher
from app.models.job import JobStatus


class StubRecord:
    """A job record with only the field the watchdog reads."""

    def __init__(self, status: JobStatus) -> None:
        self.status = status


class StubRegistry:
    """A registry that reports a fixed set of jobs."""

    def __init__(self, *statuses: JobStatus) -> None:
        self._records = [StubRecord(status) for status in statuses]

    def recent(self) -> list[StubRecord]:
        """Return the fixed jobs, newest first like the real registry."""
        return self._records


class StubServer:
    """A uvicorn server stand-in that only records the stop request."""

    def __init__(self) -> None:
        self.should_exit = False


def test_activity_starts_fresh_and_ages() -> None:
    """Idle time grows from the moment the server starts, not from first use."""
    activity = desktop.Activity()

    assert activity.idle_seconds() < 1.0

    activity._last -= 30.0
    assert activity.idle_seconds() >= 30.0


def test_busy_covers_every_unfinished_stage() -> None:
    """Any stage between queued and exporting counts as work in progress."""
    assert desktop.busy(StubRegistry(JobStatus.TRANSCRIBING))
    assert desktop.busy(StubRegistry(JobStatus.COMPLETED, JobStatus.PENDING))
    assert not desktop.busy(StubRegistry(JobStatus.COMPLETED, JobStatus.FAILED))
    assert not desktop.busy(StubRegistry())


def test_watchdog_stops_an_unused_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no page asking for anything, the server stops on its own."""
    monkeypatch.setattr(desktop, "WATCHDOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(desktop, "IDLE_GRACE_SECONDS", 0.02)
    server = StubServer()

    desktop.watch_for_idle(server, desktop.Activity(), StubRegistry())

    assert server.should_exit is True


def test_watchdog_waits_for_a_running_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing the window never abandons a transcription that is still running."""
    monkeypatch.setattr(desktop, "WATCHDOG_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(desktop, "IDLE_GRACE_SECONDS", 0.02)
    server = StubServer()
    registry = StubRegistry(JobStatus.TRANSCRIBING)

    watching = threading.Thread(
        target=desktop.watch_for_idle, args=(server, desktop.Activity(), registry)
    )
    watching.start()
    time.sleep(0.2)
    running = not server.should_exit
    server.should_exit = True
    watching.join(timeout=2)

    assert running is True


def test_session_file_is_recorded_and_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch records where it listens so the next one joins it."""
    monkeypatch.setattr(desktop, "session_file", lambda: tmp_path / "session.json")
    session = desktop.Session("127.0.0.1", 8123, "token-value")

    desktop.write_session(session)
    recorded = json.loads((tmp_path / "session.json").read_text("utf-8"))

    assert recorded == {"host": "127.0.0.1", "port": 8123, "token": "token-value"}

    desktop.clear_session(session)
    assert not (tmp_path / "session.json").exists()


def test_a_second_instance_keeps_the_first_ones_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A duplicate that exits must not erase the server still running."""
    monkeypatch.setattr(desktop, "session_file", lambda: tmp_path / "session.json")
    desktop.write_session(desktop.Session("127.0.0.1", 8123, "first"))

    desktop.clear_session(desktop.Session("127.0.0.1", 9999, "second"))

    assert (tmp_path / "session.json").exists()


def test_a_dead_record_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record left behind by a crash is discarded, not opened."""
    monkeypatch.setattr(desktop, "session_file", lambda: tmp_path / "session.json")
    monkeypatch.setattr(desktop, "answers", lambda session: False)
    desktop.write_session(desktop.Session("127.0.0.1", 8123, "stale"))

    assert desktop.running_session() is None
    assert not (tmp_path / "session.json").exists()


def test_the_desktop_app_serves_the_same_page_and_stops_by_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real launch answers on localhost, then shuts down once nobody asks."""
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    assert fastapi is not None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(desktop, "session_file", lambda: tmp_path / "session.json")
    monkeypatch.setattr(desktop, "WATCHDOG_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(desktop, "IDLE_GRACE_SECONDS", 0.5)
    opened: list[str] = []

    launched = threading.Thread(
        target=desktop.launch,
        args=(Config(),),
        kwargs={"open_page": False, "announce": lambda url, joined: opened.append(url)},
        daemon=True,
    )
    launched.start()

    deadline = time.monotonic() + 20
    while not opened and time.monotonic() < deadline:
        time.sleep(0.05)
    assert opened, "the desktop app never reported a URL"

    session = desktop.running_session()
    assert session is not None
    assert session.url == opened[0]

    launched.join(timeout=20)
    assert not launched.is_alive()
    assert not (tmp_path / "session.json").exists()


def test_open_browser_prefers_a_window_over_a_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Chromium-family browser opens the page as its own window."""
    calls: list[list[str]] = []
    monkeypatch.setattr(desktop, "app_mode_browser", lambda: "/usr/bin/chromium")
    monkeypatch.setattr(
        desktop, "primary_screen", lambda: desktop.Geometry(0, 0, 1920, 1080)
    )
    monkeypatch.setattr(
        desktop.subprocess, "Popen", lambda command, **kwargs: calls.append(command)
    )

    described = desktop.open_browser("http://127.0.0.1:1/?t=x", app_mode=True)

    assert described == "chromium window"
    assert calls[0][1] == "--app=http://127.0.0.1:1/?t=x"
    assert f"--window-size={desktop.WINDOW_WIDTH},{desktop.WINDOW_HEIGHT}" in calls[0]
    assert "--window-position=510,140" in calls[0]
    # Its own profile, or an already open browser would ignore the size.
    assert any(flag.startswith("--user-data-dir=") for flag in calls[0])


def test_window_is_centred_on_the_primary_monitor() -> None:
    """The window lands in the middle of the main screen, not a side monitor."""
    output = (
        "HDMI-1 connected 2560x1440+1920+0 (normal) 600mm x 340mm\n"
        "eDP-1 connected primary 1920x1080+0+0 (normal) 344mm x 193mm\n"
        "DP-2 disconnected (normal left inverted right x axis y axis)\n"
    )

    screen = desktop.parse_xrandr(output)

    assert screen == desktop.Geometry(0, 0, 1920, 1080)
    assert desktop.window_geometry(screen) == desktop.Geometry(510, 140, 900, 800)


def test_window_shrinks_to_fit_a_small_screen() -> None:
    """On a 1366x768 laptop the window still fits, with a margin all round."""
    place = desktop.window_geometry(desktop.Geometry(0, 0, 1366, 768))

    assert (place.width, place.height) == (900, 691)
    assert (place.x, place.y) == (233, 38)


def test_open_browser_falls_back_to_the_usual_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without such a browser, the page opens in whatever the user uses."""
    opened: list[str] = []
    monkeypatch.setattr(desktop, "app_mode_browser", lambda: None)
    monkeypatch.setattr(desktop.webbrowser, "open", lambda url: opened.append(url))

    described = desktop.open_browser("http://127.0.0.1:1/?t=x", app_mode=True)

    assert described == "default browser"
    assert opened == ["http://127.0.0.1:1/?t=x"]


def test_desktop_entry_starts_without_a_terminal(tmp_path: Path) -> None:
    """The entry runs desktop mode from the folder it was installed in."""
    entry = launcher.desktop_entry(
        tmp_path, ("/opt/venv/bin/python", "-m", "app", "desktop")
    )

    assert "Terminal=false" in entry
    assert "Exec=/opt/venv/bin/python -m app desktop" in entry
    assert f"Path={tmp_path}" in entry
    assert "Icon=captionforge" in entry
    assert "Name=CaptionForge" in entry


def test_desktop_entry_quotes_a_path_with_spaces(tmp_path: Path) -> None:
    """An interpreter inside "My Projects" still starts."""
    entry = launcher.desktop_entry(tmp_path, ("/home/a b/python", "-m", "app"))

    assert 'Exec="/home/a b/python" -m app' in entry


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux desktop entry")
def test_install_and_uninstall_leave_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installing writes an entry and an icon; removing takes both away."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setattr(launcher, "_refresh_linux_caches", lambda: None)

    installed = launcher.install(tmp_path / "project")

    entry = tmp_path / "share" / "applications" / "captionforge.desktop"
    icon = (
        tmp_path
        / "share"
        / "icons"
        / "hicolor"
        / "scalable"
        / "apps"
        / "captionforge.svg"
    )
    assert entry.is_file()
    assert icon.is_file()
    assert set(installed.paths) == {entry, icon}
    assert installed.workdir == (tmp_path / "project").resolve()

    removed = launcher.uninstall()

    assert set(removed) == {entry, icon}
    assert not entry.exists()
    assert not icon.exists()


def test_uninstall_reports_nothing_when_no_entry_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing an entry that was never installed is quiet, not an error."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setattr(launcher, "_refresh_linux_caches", lambda: None)

    assert launcher.uninstall() == ()


def test_launch_joins_a_server_that_is_already_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second click opens the running app instead of starting a second one."""
    session = desktop.Session("127.0.0.1", 8123, "token")
    shown: list[str] = []
    monkeypatch.setattr(desktop, "running_session", lambda: session)
    monkeypatch.setattr(
        desktop, "present", lambda url, **kwargs: shown.append(url) or "window"
    )
    joined: list[Any] = []

    desktop.launch(Config(), announce=lambda url, was_joined: joined.append(was_joined))

    assert shown == [session.url]
    assert joined == [True]
