"""Offline tests for the Qt desktop window and the desktop entry it installs.

The window runs on Qt's offscreen platform, so these need no display. Its
services and job registry are stand-ins: nothing here reaches YouTube, pip or
a Whisper model.
"""

import builtins
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import (  # noqa: E402
    QBuffer,
    QByteArray,
    QIODevice,
    QMimeData,
    QPoint,
    QPointF,
    QRect,
    Qt,
    QUrl,
)
from PySide6.QtGui import (  # noqa: E402
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QPalette,
)
from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402

from app.adapters.package_updater import (  # noqa: E402
    PACKAGES,
    AvailableUpdate,
    InstalledUpdate,
)
from app.adapters.whisper_adapter import CudaStatus  # noqa: E402
from app.core.config import Config  # noqa: E402
from app.core.deepgram_key import DeepgramKey  # noqa: E402
from app.core.exceptions import (  # noqa: E402
    ConfigurationError,
    DeepgramKeyRejectedError,
    ExtractorRefusedError,
    LocalFileNotFoundError,
)
from app.interfaces import cli, desktop, launcher  # noqa: E402
from app.interfaces.desktop import application as desktop_app  # noqa: E402
from app.interfaces.desktop import theme  # noqa: E402
from app.interfaces.desktop.widgets import Chip, FileRow  # noqa: E402
from app.interfaces.desktop.window import (  # noqa: E402
    MEDIA_FILE_FILTER,
    MainWindow,
    Services,
    format_duration,
    format_size,
)
from app.interfaces.web.jobs import (  # noqa: E402
    JobRecord,
    JobRequest,
    MediaJobRequest,
    OutputFile,
)
from app.models.job import JobStatus  # noqa: E402
from app.models.media import MediaKind, MediaOptions, MediaVariant  # noqa: E402
from app.models.subtitle import (  # noqa: E402
    SubtitleDiscoveryResult,
    SubtitleSourceType,
    SubtitleTrack,
)
from app.models.video import VideoMetadata  # noqa: E402
from app.services.video_service import VideoInspection  # noqa: E402

URL = "https://youtu.be/qJFbKl6RjLU"
YTDLP = next(package for package in PACKAGES if package.name == "yt-dlp")


# ---------- stand-ins ----------


class StubRegistry:
    """A job registry that records requests and never runs anything."""

    def __init__(self) -> None:
        self.records: dict[str, JobRecord] = {}
        self.submitted: list[JobRequest | MediaJobRequest] = []
        self.stopped = False

    def _add(self, request: JobRequest | MediaJobRequest) -> JobRecord:
        record = JobRecord(id=f"job-{len(self.records) + 1}", request=request)
        self.records[record.id] = record
        self.submitted.append(request)
        return record

    def submit(self, request: JobRequest) -> JobRecord:
        return self._add(request)

    def submit_media(self, request: MediaJobRequest) -> JobRecord:
        return self._add(request)

    def get(self, job_id: str) -> JobRecord | None:
        return self.records.get(job_id)

    def recent(self) -> list[JobRecord]:
        return list(reversed(self.records.values()))

    def cancel(self, job_id: str) -> bool:
        record = self.records.get(job_id)
        if record is None:
            return False
        record.cancel_event.set()
        return True

    def shutdown(self) -> None:
        self.stopped = True


class StubUpdater:
    """An updater with a fixed answer, which installs whatever it is told."""

    def __init__(self, *updates: AvailableUpdate) -> None:
        self.updates = updates
        self.installed: list[list[str]] = []
        self.fresh: list[bool] = []

    def can_update(self) -> bool:
        return True

    def check(self, *, fresh: bool = False) -> tuple[AvailableUpdate, ...]:
        self.fresh.append(fresh)
        return self.updates

    def install(self, names: list[str]) -> tuple[InstalledUpdate, ...]:
        self.installed.append(list(names))
        return tuple(
            InstalledUpdate(update.name, update.latest, False)
            for update in self.updates
            if update.name in names
        )


def track(code: str, *, automatic: bool = False, translated: bool = False) -> Any:
    """A caption track with only the fields the window reads."""
    return SubtitleTrack(
        language_code=code,
        normalized_language_code=code,
        source_type=(
            SubtitleSourceType.AUTOMATIC if automatic else SubtitleSourceType.MANUAL
        ),
        is_automatic=automatic,
        is_translated=translated,
    )


def inspection(title: str = "How captions are made") -> VideoInspection:
    """A video with two real tracks, two translations, and three downloads."""
    video = VideoMetadata(
        video_id="qJFbKl6RjLU",
        title=title,
        channel_name="Example Channel",
        duration_seconds=3725,
        webpage_url=URL,
        original_url=URL,
        thumbnail_url="https://i.ytimg.com/vi/qJFbKl6RjLU/hqdefault.jpg",
    )
    selected = track("ar")
    discovery = SubtitleDiscoveryResult(
        video=video,
        manual_tracks=(selected,),
        automatic_tracks=(
            track("ar", automatic=True),
            track("fr", automatic=True, translated=True),
            track("de", automatic=True, translated=True),
        ),
        selected_track=selected,
        preferred_language="ar",
        selection_reason="manual track in the preferred language",
    )
    media = MediaOptions(
        variants=(
            MediaVariant(
                key="720",
                kind=MediaKind.VIDEO,
                label="720p",
                extension="mp4",
                height=720,
                estimated_bytes=50 * 1024 * 1024,
            ),
            MediaVariant(
                key="audio",
                kind=MediaKind.AUDIO,
                label="MP3",
                extension="mp3",
                estimated_bytes=8_500_000,
            ),
        )
    )
    return VideoInspection(discovery=discovery, media=media)


def local_inspection(path: Path) -> VideoInspection:
    """A file on this computer: no tracks, nothing to download."""
    video = VideoMetadata(
        title=path.stem,
        duration_seconds=95,
        webpage_url=path.as_uri(),
        original_url=str(path),
        local_path=path,
    )
    return VideoInspection(
        discovery=SubtitleDiscoveryResult(video=video, preferred_language="ar"),
        media=MediaOptions(),
    )


def carrying(*urls: QUrl) -> QMimeData:
    """What a file manager hands over when something is dragged out of it."""
    mime = QMimeData()
    mime.setUrls(list(urls))
    return mime


def drag_over(window: MainWindow, mime: QMimeData) -> QDragEnterEvent:
    event = QDragEnterEvent(
        QPoint(40, 40),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window, event)
    return event


def drop_on(window: MainWindow, mime: QMimeData) -> QDropEvent:
    event = QDropEvent(
        QPointF(40, 40),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window, event)
    return event


def png_bytes() -> bytes:
    """A real picture, so the thumbnail has something to decode."""
    image = QImage(32, 18, QImage.Format.Format_RGB32)
    image.fill(QColor("#00D68F"))
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(data.data())


def wait_until(condition: Callable[[], object], timeout: float = 5.0) -> None:
    """Let Qt deliver queued work until ``condition`` holds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("the window never got there")


def finish(record: JobRecord, *files: Path, **fields: Any) -> None:
    """Move a stub job to completed, as the registry's worker would."""
    with record.lock:
        record.files = [
            OutputFile(path.name, path, path.stat().st_size) for path in files
        ]
        for name, value in fields.items():
            setattr(record, name, value)
        record.status = JobStatus.COMPLETED
        record.stage = "Completed"
        record.percent = 100.0


def refuse_key(_value: str) -> tuple[DeepgramKey, bool | None]:
    """The default stand-in: no test reaches Deepgram unless it says so."""
    raise DeepgramKeyRejectedError("Deepgram did not accept that key.")


# ---------- fixtures ----------


@pytest.fixture(scope="session")
def qt_app() -> QApplication:
    """One styled application for every test, as a real session has."""
    created = desktop_app.application()
    theme.theme().install(created)
    theme.theme().apply(theme.LIGHT)
    return created


@pytest.fixture
def make_window(
    qt_app: QApplication, tmp_path: Path
) -> Iterator[Callable[..., MainWindow]]:
    """Build windows over stand-ins, and take them down after the test."""
    made: list[MainWindow] = []

    def build(**overrides: Any) -> MainWindow:
        opened: list[Path] = overrides.pop("opened", [])
        services = Services(
            inspect=overrides.pop("inspect", lambda url, language, allow: inspection()),
            updater=overrides.pop("updater", StubUpdater()),
            cuda=overrides.pop("cuda", lambda: CudaStatus(True, True, ())),
            fetch=overrides.pop("fetch", lambda url: png_bytes()),
            open_path=lambda path: opened.append(path) or True,
            choose_file=overrides.pop("choose_file", lambda parent, folder: None),
            deepgram_key=overrides.pop("deepgram_key", lambda: None),
            save_deepgram_key=overrides.pop("save_deepgram_key", refuse_key),
            forget_deepgram_key=overrides.pop("forget_deepgram_key", lambda: None),
        )
        window = MainWindow(
            overrides.pop("config", Config()),
            registry=overrides.pop("registry", StubRegistry()),
            services=services,
            preferences_file=overrides.pop("preferences_file", tmp_path / "prefs.json"),
        )
        assert not overrides, f"unknown overrides {sorted(overrides)}"
        window.resize(900, 800)
        window.show()
        made.append(window)
        return window

    yield build
    for window in made:
        window.shutdown()
        window.hide()
        window.deleteLater()
    QApplication.processEvents()


def looked_up(window: MainWindow, url: str = URL) -> MainWindow:
    """Type a link and wait for the lookup to fill the window."""
    window.url.setText(url)
    window.look_up()
    wait_until(lambda: not window.controls.isHidden())
    return window


# ---------- formatting, as the page formats ----------


def test_sizes_and_durations_read_like_the_page() -> None:
    assert format_size(512) == "512 B"
    assert format_size(8_500_000) == "8.1 MB"
    assert format_size(18_234) == "17.8 KB"
    assert format_duration(3725) == "1:02:05"
    assert format_duration(754) == "12:34"
    assert format_duration(None) == ""


# ---------- window placement ----------


def test_window_is_centred_on_the_screen() -> None:
    """The window lands in the middle of the main screen at its natural size."""
    place = desktop_app.window_geometry(QRect(0, 0, 1920, 1080))

    assert place == QRect(510, 140, 900, 800)


def test_window_shrinks_to_fit_a_small_screen() -> None:
    """On a 1366x768 laptop the window still fits, with a margin all round."""
    place = desktop_app.window_geometry(QRect(0, 0, 1366, 768))

    assert (place.width(), place.height()) == (900, 691)
    assert (place.x(), place.y()) == (233, 38)


def test_the_app_is_named_for_its_desktop_entry(qt_app: QApplication) -> None:
    """GNOME groups the window under the entry by these names."""
    assert qt_app.arguments()[0] == launcher.ENTRY_NAME
    assert qt_app.desktopFileName() == launcher.ENTRY_NAME
    assert qt_app.applicationName() == "CaptionForge"
    assert qt_app.quitOnLastWindowClosed() is False


# ---------- theme ----------


def test_the_theme_follows_the_system_colour_scheme(qt_app: QApplication) -> None:
    """Dark on a dark desktop, light otherwise, with the page's own tokens."""
    assert theme.palette_for(Qt.ColorScheme.Dark) is theme.DARK
    assert theme.palette_for(Qt.ColorScheme.Light) is theme.LIGHT
    assert theme.palette_for(Qt.ColorScheme.Unknown) is theme.LIGHT

    try:
        theme.theme().apply(theme.DARK)
        assert theme.current() is theme.DARK
        window = qt_app.palette().color(QPalette.ColorRole.Window)
        assert window.name().upper() == theme.DARK.bg
        assert theme.DARK.surface in qt_app.styleSheet()
    finally:
        theme.theme().apply(theme.LIGHT)


def test_the_theme_carries_the_page_tokens_verbatim() -> None:
    """Every colour in app.css appears here unchanged, light and dark."""
    css = (
        Path(__file__).parents[2] / "app" / "interfaces" / "web" / "static" / "app.css"
    ).read_text(encoding="utf-8")
    for value in (
        theme.LIME,
        theme.SPRING,
        theme.EMERALD,
        theme.TEAL,
        theme.ON_VIVID,
        *vars(theme.LIGHT).values(),
        *vars(theme.DARK).values(),
    ):
        if value.startswith("#"):
            assert value in css, value


# ---------- looking a video up ----------


def test_a_lookup_fills_every_section(make_window: Callable[..., MainWindow]) -> None:
    """Title, byline, picture, downloads, tracks and controls all appear."""
    window = looked_up(make_window())

    assert window.title.plain() == "How captions are made"
    assert window.byline.plain() == "Example Channel · 1:02:05"
    wait_until(window.thumb.has_picture)
    media = window.media_list.chips()
    # The MP3 first, then the videos, each with its estimated size.
    assert [(chip.text(), chip.kind) for chip in media] == [
        ("MP3", "~8.1 MB"),
        ("720p", "~50.0 MB"),
    ]
    tracks = window.track_list.chips()
    assert [chip.text() for chip in tracks] == ["ar", "ar", "+2 machine-translated"]
    assert [chip.picked for chip in tracks] == [True, False, False]
    assert window.selection.plain() == (
        "Will export the highlighted track (manual track in the preferred language)."
    )
    assert window.inspect_button.isEnabled()
    assert window.inspect_button.text() == "Look up"


def test_translations_stay_behind_one_chip(
    make_window: Callable[..., MainWindow],
) -> None:
    """The machine translations appear only when asked for."""
    window = looked_up(make_window())
    more = window.track_list.chips()[-1]

    more.click()

    shown = [chip for chip in window.track_list.chips() if not chip.isHidden()]
    assert [chip.text() for chip in shown] == ["ar", "ar", "fr", "de"]
    assert shown[-1].kind == "auto · translated"


def test_an_arabic_title_reads_right_to_left(
    make_window: Callable[..., MainWindow],
) -> None:
    """``<h2 dir="auto">``: the title's own script sets its direction."""
    window = make_window(inspect=lambda *args: inspection("سورة الملك"))
    looked_up(window)

    assert 'dir="rtl"' in window.title.text()
    assert 'dir="ltr"' in window.byline.text()


def test_a_refused_lookup_offers_a_newer_ytdlp(
    make_window: Callable[..., MainWindow],
) -> None:
    """YouTube refusing yt-dlp ticks the yt-dlp update, and installs nothing."""

    def refuse(*args: object) -> VideoInspection:
        raise ExtractorRefusedError("YouTube refused this request.")

    updater = StubUpdater(AvailableUpdate(YTDLP, "2026.8.19", "2026.9.20"))
    window = make_window(inspect=refuse, updater=updater)
    window.url.setText(URL)
    window.look_up()

    wait_until(lambda: not window.alert.isHidden() and updater.fresh[-1:] == [True])
    wait_until(lambda: window.ticked_updates() == ["yt-dlp"])
    assert window.alert_title.text() == "COULD NOT READ THAT VIDEO"
    assert window.alert_body.plain() == "YouTube refused this request."
    assert window.video.isHidden()
    assert window.update_button.isEnabled()
    assert "Nothing installs until you click" in window.updates_note.plain()
    assert updater.installed == []


def test_an_update_installs_exactly_what_was_ticked(
    make_window: Callable[..., MainWindow],
) -> None:
    """Update selected installs the ticked package and reports it."""
    updater = StubUpdater(AvailableUpdate(YTDLP, "2026.8.19", "2026.9.20"))
    window = make_window(updater=updater)
    wait_until(lambda: not window.updates.isHidden())
    assert window.ticked_updates() == []
    assert not window.update_button.isEnabled()

    window._update_boxes[0][1].setChecked(True)
    window.update_button.click()

    wait_until(lambda: window.updates_label.text() == "UPDATED")
    assert updater.installed == [["yt-dlp"]]
    assert window.updates_note.plain() == "Updated yt-dlp to 2026.9.20."
    assert window.update_list.isHidden()


# ---------- remembering choices ----------


def test_choices_are_remembered_for_the_next_window(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    """A toggled format and option come back in the next session."""
    window = make_window()
    txt = next(chip for chip in window.formats.chips() if chip.text() == "txt")
    txt.click()
    window.keep_audio.setChecked(True)
    window.close()

    saved = (tmp_path / "prefs.json").read_text(encoding="utf-8")
    assert '"txt"' in saved

    again = make_window()
    assert {chip.text() for chip in again.formats.chips() if chip.picked} >= {"txt"}
    assert again.keep_audio.isChecked()
    # A changed option is worth showing on arrival.
    assert again.options.is_open()


def test_an_unusable_graphics_card_cannot_be_picked(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    """A saved "cuda" falls back to Automatic when CUDA will not load."""
    (tmp_path / "prefs.json").write_text('{"device": "cuda"}', encoding="utf-8")
    window = make_window(cuda=lambda: CudaStatus(False, True, ("cuBLAS",)))

    card = window.device_cards["cuda"]
    wait_until(lambda: not card.isEnabled())
    assert window.chosen_device() == "auto"
    assert "cuBLAS will not load" in card.text.text()


# ---------- running jobs ----------


def test_one_click_on_a_quality_starts_that_download(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    """The chip is the whole request, and every other action waits for it."""
    registry = StubRegistry()
    opened: list[Path] = []
    window = looked_up(make_window(registry=registry, opened=opened))
    window.url.setText("https://youtu.be/somethingElse")
    mp3, video = window.media_list.chips()

    mp3.click()

    request = registry.submitted[-1]
    assert isinstance(request, MediaJobRequest)
    assert request.quality == "audio"
    # The chips describe the video that was looked up.
    assert request.url == URL
    assert mp3.picked and not video.isEnabled() and not mp3.isEnabled()
    assert not window.run_button.isEnabled()
    assert not window.progress.isHidden()

    produced = tmp_path / "talk.mp3"
    produced.write_bytes(b"x" * 2048)
    finish(
        registry.records["job-1"],
        produced,
        media_summary={"key": "audio", "label": "MP3", "kind": "audio"},
    )
    window._poll()

    assert window.progress.isHidden()
    assert window.results_label.text() == "SAVED THE AUDIO AS MP3"
    assert not mp3.picked and video.isEnabled()
    rows = window.files.findChildren(FileRow)
    assert len(rows) == 1
    rows[0].click()
    assert opened == [produced]


def test_captions_finish_with_the_transcription_details(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    """Get captions sends the chosen formats and reports what Whisper did."""
    registry = StubRegistry()
    window = looked_up(make_window(registry=registry))

    window.run_button.click()

    request = registry.submitted[-1]
    assert isinstance(request, JobRequest)
    assert request.url == URL
    assert request.formats == ("srt",)
    assert request.language == Config().default_language

    produced = tmp_path / "talk.srt"
    produced.write_text("1\n", encoding="utf-8")
    finish(
        registry.records["job-1"],
        produced,
        used_existing_captions=False,
        transcription_summary={
            "detected_language": "ar",
            "language_probability": 0.974,
            "model_name": "small",
            "device": "cpu",
            "compute_type": "int8",
        },
    )
    window._poll()

    assert window.results_label.text() == "TRANSCRIBED ON THIS COMPUTER"
    note = window.results_note.text()
    assert "small on cpu, detected ar (97% confident)" in note
    assert "Saved to" in note


def test_a_failed_job_says_why(make_window: Callable[..., MainWindow]) -> None:
    registry = StubRegistry()
    window = looked_up(make_window(registry=registry))
    window.run_button.click()
    record = registry.records["job-1"]
    with record.lock:
        record.status = JobStatus.FAILED
        record.error = "FFmpeg is not installed."

    window._poll()

    assert window.alert_title.text() == "JOB FAILED"
    assert window.alert_body.plain() == "FFmpeg is not installed."
    assert window.run_button.isEnabled()


def test_cancel_asks_the_job_to_stop(make_window: Callable[..., MainWindow]) -> None:
    registry = StubRegistry()
    window = looked_up(make_window(registry=registry))
    window.run_button.click()

    window.cancel_button.click()

    assert registry.records["job-1"].cancel_event.is_set()
    assert window.stage.text() == "Cancelling…"
    assert not window.cancel_button.isEnabled()


def test_a_job_needs_a_format(make_window: Callable[..., MainWindow]) -> None:
    registry = StubRegistry()
    window = looked_up(make_window(registry=registry))
    for chip in window.formats.chips():
        if chip.picked:
            chip.click()

    window.run_button.click()

    assert registry.submitted == []
    assert window.alert_title.text() == "PICK A FORMAT"


# ---------- lifetime ----------


def test_closing_an_idle_window_ends_the_app(
    make_window: Callable[..., MainWindow],
) -> None:
    window = make_window()
    ended: list[bool] = []
    window.finished.connect(lambda: ended.append(True))

    window.close()

    assert ended == [True]


def test_closing_never_abandons_a_running_job(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    """The window hides, the job finishes, and only then does the app end."""
    registry = StubRegistry()
    window = looked_up(make_window(registry=registry))
    window.run_button.click()
    window._idle_timer.setInterval(10)
    ended: list[bool] = []
    window.finished.connect(lambda: ended.append(True))

    window.close()

    assert window.isHidden()
    assert ended == []
    produced = tmp_path / "talk.srt"
    produced.write_text("1\n", encoding="utf-8")
    finish(registry.records["job-1"], produced, used_existing_captions=True)
    wait_until(lambda: ended == [True])


def test_a_second_launch_brings_the_first_forward(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """The second copy wakes the first and does not run itself."""
    address = str(tmp_path / "instance.socket")
    first = desktop_app.SingleInstance(address, tmp_path / "instance.lock")
    second = desktop_app.SingleInstance(address, tmp_path / "instance.lock")
    woken: list[bool] = []
    first.activated.connect(lambda: woken.append(True))
    try:
        assert first.claim() is True
        assert second.claim() is False
        wait_until(lambda: woken == [True])
    finally:
        first.release()

    third = desktop_app.SingleInstance(address, tmp_path / "instance.lock")
    try:
        assert third.claim() is True
    finally:
        third.release()


def test_a_crashed_copy_does_not_block_the_next_launch(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """A lock left by a process that died is taken over, not waited on."""
    lock = tmp_path / "instance.lock"
    script = (
        "import os, sys\n"
        "from PySide6.QtCore import QLockFile\n"
        "held = QLockFile(sys.argv[1]); held.setStaleLockTime(0)\n"
        "assert held.tryLock(0)\n"
        "os._exit(0)\n"
    )
    subprocess.run([sys.executable, "-c", script, str(lock)], check=True, timeout=60)
    assert lock.exists()

    instance = desktop_app.SingleInstance(str(tmp_path / "instance.socket"), lock)
    try:
        assert instance.claim() is True
    finally:
        instance.release()


def test_launch_explains_how_to_install_qt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the desktop extra, the error says which package to install."""
    real_import = builtins.__import__

    def without_qt(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "app.interfaces.desktop.application":
            raise ModuleNotFoundError("No module named 'PySide6'", name="PySide6")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_qt)

    with pytest.raises(ConfigurationError, match=r"captionforge\[desktop\]"):
        desktop.launch(Config())


def test_the_desktop_command_says_when_it_joined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second ``captionforge desktop`` from a terminal says what it did."""
    from typer.testing import CliRunner

    monkeypatch.setattr(desktop, "launch", lambda config: True)

    result = CliRunner().invoke(cli.app, ["desktop"])

    assert result.exit_code == 0
    assert "already open" in result.stdout


# ---------- the desktop entry ----------


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


def test_desktop_entry_matches_the_qt_window(tmp_path: Path) -> None:
    """The dock groups the window under this entry, and keeps it to one."""
    entry = launcher.desktop_entry(tmp_path, ("/opt/venv/bin/python", "-m", "app"))

    assert f"StartupWMClass={launcher.ENTRY_NAME}" in entry.splitlines()
    assert "SingleMainWindow=true" in entry.splitlines()


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


def test_chips_measure_bold_so_picking_never_reflows(qt_app: QApplication) -> None:
    """A chip keeps its width whether or not it is picked."""
    chip = Chip("1080p", "~173.6 MB")
    before = chip.sizeHint()

    chip.set_picked(True)

    assert chip.sizeHint() == before


# ---------- files from this computer ----------


def test_a_chosen_file_is_looked_up_straight_away(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    """Choosing is the whole request, and the file is read where it lies."""
    source = tmp_path / "talk.mp3"
    folders: list[str] = []
    seen: list[str] = []

    def choose(_parent: object, folder: str) -> str:
        folders.append(folder)
        return str(source)

    window = make_window(
        choose_file=choose,
        inspect=lambda url, language, allow: (
            seen.append(url) or local_inspection(source)
        ),
    )

    window.choose_button.click()
    wait_until(lambda: not window.controls.isHidden())

    assert folders == [str(Path.home())]
    assert seen == [str(source)]
    assert window.url.text() == str(source)
    assert window.title.plain() == "talk"
    assert window.byline.plain() == "1:35"
    # A file has no tracks and is always transcribed, and the window says so.
    assert window.run_button.text() == "Transcribe"
    assert window.tracks.isHidden() and window.media.isHidden()
    assert window.force.isHidden() and window.allow_translated.isHidden()

    # The next choice starts where the last one was found.
    window.choose_button.click()
    assert folders[-1] == str(tmp_path)


def test_closing_the_chooser_changes_nothing(
    make_window: Callable[..., MainWindow],
) -> None:
    seen: list[str] = []
    window = make_window(
        inspect=lambda url, language, allow: seen.append(url) or inspection()
    )
    window.choose_button.click()
    QApplication.processEvents()
    assert seen == []
    assert window.url.text() == ""
    assert window.controls.isHidden()


def test_a_link_after_a_file_brings_the_caption_parts_back(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    source = tmp_path / "talk.mp3"
    answers = iter([local_inspection(source), inspection()])
    window = make_window(inspect=lambda url, language, allow: next(answers))

    looked_up(window, str(source))
    assert window.run_button.text() == "Transcribe"
    window.controls.hide()
    looked_up(window)

    assert window.run_button.text() == "Get captions"
    assert not window.tracks.isHidden()
    assert not window.force.isHidden() and not window.allow_translated.isHidden()


def test_a_file_dropped_on_the_window_is_looked_up(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    source = tmp_path / "محاضرة.mp4"
    seen: list[str] = []
    window = make_window(
        inspect=lambda url, language, allow: (
            seen.append(url) or local_inspection(source)
        )
    )
    mime = carrying(QUrl.fromLocalFile(str(source)))

    assert drag_over(window, mime).isAccepted()
    # The field is marked as where the file will land.
    assert window.url.property("dropping") == "true"
    drop_on(window, mime)
    wait_until(lambda: not window.controls.isHidden())

    assert window.url.property("dropping") == "false"
    assert seen == [str(source)]


def test_only_files_on_this_computer_can_be_dropped(
    make_window: Callable[..., MainWindow],
) -> None:
    window = make_window()
    link = carrying(QUrl("https://youtu.be/qJFbKl6RjLU"))
    assert not drag_over(window, link).isAccepted()
    assert window.url.property("dropping") != "true"


def test_nothing_is_dropped_while_a_job_runs(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    seen: list[str] = []
    window = looked_up(
        make_window(
            inspect=lambda url, language, allow: seen.append(url) or inspection()
        )
    )
    window.run()
    mime = carrying(QUrl.fromLocalFile(str(tmp_path / "talk.mp4")))

    assert not drag_over(window, mime).isAccepted()
    drop_on(window, mime)
    QApplication.processEvents()

    assert seen == [URL]
    assert window.url.text() == URL


def test_the_fields_leave_dropped_files_to_the_window(
    make_window: Callable[..., MainWindow],
) -> None:
    """Otherwise a field would swallow the drop as the text of its address."""
    window = make_window()
    assert window.acceptDrops()
    assert not any(field.acceptDrops() for field in window.findChildren(QLineEdit))


def test_a_file_that_cannot_be_read_says_file(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    def inspect(url: str, language: str, allow: bool) -> VideoInspection:
        raise LocalFileNotFoundError(f"No file was found at {url}.")

    window = make_window(inspect=inspect)
    window.take_file(tmp_path / "gone.mp4")
    wait_until(lambda: not window.alert.isHidden())

    assert window.alert_title.text() == "COULD NOT READ THAT FILE"
    assert window.alert_body.plain().startswith("No file was found at")


def test_the_file_chooser_lists_media_in_either_case() -> None:
    assert "*.mp4" in MEDIA_FILE_FILTER and "*.MP4" in MEDIA_FILE_FILTER
    assert "*.opus" in MEDIA_FILE_FILTER
    assert MEDIA_FILE_FILTER.endswith(";;All files (*)")


# ---------- transcribe with: engine, model, key ----------

DEEPGRAM_KEY = "0123456789abcdef0123456789abcdef01234567"


def engine_chip(window: MainWindow, name: str) -> Chip:
    return next(chip for chip in window.engines.chips() if chip.text() == name)


def test_the_most_used_choices_sit_outside_more_options(
    make_window: Callable[..., MainWindow],
) -> None:
    """Engine, model and the captions override show without opening anything."""
    window = looked_up(make_window())
    assert not window.options.is_open()
    for control in (window.engines, window.model, window.force):
        assert control.isVisible()
    assert window.options.button.text().endswith("More options")
    assert window.model.currentData() == "small"
    assert engine_chip(window, "Whisper").picked


def test_switching_engines_swaps_the_models_and_keeps_each_pick(
    make_window: Callable[..., MainWindow],
) -> None:
    window = looked_up(make_window())
    window.model.setCurrentIndex(window.model.findData("medium"))

    engine_chip(window, "Deepgram").click()

    assert window.chosen_engine() == "deepgram"
    assert window.model.currentData() == "nova-3"
    assert window.model.findData("medium") == -1
    # The graphics card is Whisper's business, and there is no key yet.
    assert window.device_block.isHidden()
    assert window.key_row.isVisible() and window.key_saved.isHidden()
    assert "Deepgram" in window.hint.plain()
    assert "~21.3 MB" in window.engine_help.plain()

    engine_chip(window, "Whisper").click()

    assert window.model.currentData() == "medium"
    assert not window.device_block.isHidden() and window.key_row.isHidden()


def test_the_override_says_what_the_button_will_do(
    make_window: Callable[..., MainWindow],
) -> None:
    window = looked_up(make_window())
    assert window.run_button.text() == "Get captions"

    window.force.setChecked(True)

    assert window.run_button.text() == "Transcribe"
    assert window.selection.plain() == (
        "Will transcribe with Whisper instead of exporting the highlighted track."
    )


def test_deepgram_needs_a_key_only_when_it_would_be_used(
    make_window: Callable[..., MainWindow],
) -> None:
    registry = StubRegistry()
    window = looked_up(make_window(registry=registry))
    engine_chip(window, "Deepgram").click()

    window.force.setChecked(True)
    window.run_button.click()

    assert registry.submitted == []
    assert window.alert_title.text() == "ADD A DEEPGRAM KEY"
    assert window.deepgram_key.hasFocus()

    # Without the override the video's own captions are exported: no key needed.
    window.force.setChecked(False)
    window.run_button.click()
    assert len(registry.submitted) == 1


def test_a_pasted_key_is_checked_saved_and_used(
    make_window: Callable[..., MainWindow],
) -> None:
    pasted: list[str] = []

    def save(value: str) -> tuple[DeepgramKey, bool | None]:
        pasted.append(value)
        return DeepgramKey(DEEPGRAM_KEY, "file"), True

    registry = StubRegistry()
    window = looked_up(make_window(registry=registry, save_deepgram_key=save))
    engine_chip(window, "Deepgram").click()
    window.deepgram_key.setText(DEEPGRAM_KEY)

    window.key_save.click()
    wait_until(lambda: window.key_saved.isVisible())

    assert pasted == [DEEPGRAM_KEY]
    assert window.deepgram_key.text() == ""
    assert window.key_saved_text.plain() == "Deepgram key …4567 saved"
    window.force.setChecked(True)
    window.run_button.click()
    request = registry.submitted[-1]
    assert isinstance(request, JobRequest)
    assert (request.engine, request.model) == ("deepgram", "nova-3")


def test_a_refused_key_says_why_and_stays_unsaved(
    make_window: Callable[..., MainWindow],
) -> None:
    window = looked_up(make_window())
    engine_chip(window, "Deepgram").click()
    window.deepgram_key.setText("nope")
    window.key_save.click()
    wait_until(lambda: not window.alert.isHidden())
    assert window.alert_title.text() == "COULD NOT SAVE THE KEY"
    assert window.key_row.isVisible()


def test_a_key_from_the_environment_cannot_be_forgotten_here(
    make_window: Callable[..., MainWindow],
) -> None:
    window = make_window(deepgram_key=lambda: DeepgramKey(DEEPGRAM_KEY, "environment"))
    engine_chip(window, "Deepgram").click()
    assert window.key_saved_text.plain() == "Deepgram key …4567 set in the environment"
    assert window.key_forget.isHidden()


def test_deepgram_results_say_where_they_were_made(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    registry = StubRegistry()
    window = looked_up(make_window(registry=registry))
    window.run_button.click()
    produced = tmp_path / "talk.srt"
    produced.write_text("1\n", encoding="utf-8")
    finish(
        registry.records["job-1"],
        produced,
        used_existing_captions=False,
        transcription_summary={
            "engine": "deepgram",
            "detected_language": "tr",
            "language_probability": 0.9,
            "model_name": "nova-3",
            "device": "deepgram",
            "compute_type": "cloud",
        },
    )
    window._poll()

    assert window.results_label.text() == "TRANSCRIBED BY DEEPGRAM"
    assert "Deepgram nova-3, detected tr (90% confident)" in window.results_note.text()


def test_the_engine_and_both_models_are_remembered(
    make_window: Callable[..., MainWindow], tmp_path: Path
) -> None:
    window = make_window()
    window.model.setCurrentIndex(window.model.findData("base"))
    engine_chip(window, "Deepgram").click()
    window.close()

    saved = json.loads((tmp_path / "prefs.json").read_text(encoding="utf-8"))
    assert saved["engine"] == "deepgram"
    assert (saved["model"], saved["deepgram_model"]) == ("base", "nova-3")

    again = make_window()
    assert engine_chip(again, "Deepgram").picked
    assert again.model.currentData() == "nova-3"
    engine_chip(again, "Whisper").click()
    assert again.model.currentData() == "base"
