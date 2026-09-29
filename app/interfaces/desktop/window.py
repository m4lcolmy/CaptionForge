"""The CaptionForge window: the local page's sections, as native widgets.

Every section, heading and sentence is the page's own, in the page's order, and
behaves the way ``app.js`` makes it behave. The window calls the services the
web server calls, in the same process, so there is no server, port or token
between them. Two things differ because a desktop app can do better than a
page: finished files open with one click instead of downloading a second copy,
and closing the window never abandons a job that is still running.
"""

from __future__ import annotations

import html
import math
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QLabel,
    QLineEdit,
    QMainWindow,
    QRadioButton,
    QScrollArea,
    QWidget,
)

from app.adapters import package_updater
from app.adapters.whisper_adapter import CudaStatus
from app.core.config import Config
from app.core.constants import APP_NAME, SUPPORTED_OUTPUT_FORMATS, VERSION
from app.core.exceptions import CaptionForgeError
from app.core.logging_config import get_logger
from app.interfaces.desktop.background import run_in_background
from app.interfaces.desktop.theme import EMERALD, font
from app.interfaces.desktop.widgets import (
    Button,
    Chip,
    Chips,
    Disclosure,
    ElidedLabel,
    FileRow,
    Label,
    Mark,
    OptionCard,
    ProgressTrack,
    SectionLabel,
    Thumbnail,
    clear_layout,
    column,
    on_click,
    row,
)
from app.interfaces.web.errors import GENERIC_MESSAGE, update_may_help
from app.interfaces.web.jobs import JobRegistry, JobRequest, MediaJobRequest
from app.interfaces.web.preferences import (
    WebPreferences,
    load_preferences,
    resolve,
    save_preferences,
)
from app.models.job import JobStatus
from app.models.media import MediaKind, MediaOptions, MediaVariant
from app.models.subtitle import SubtitleTrack
from app.services.factory import create_video_service
from app.services.video_service import VideoInspection

POLL_INTERVAL_MS = 700
SAVE_DELAY_MS = 600
IDLE_CHECK_MS = 2000
THUMBNAIL_TIMEOUT_SECONDS = 10
THUMBNAIL_MAX_BYTES = 4 * 1024 * 1024

# ``main { max-width: 40rem; padding: 2.5rem 1.5rem 6rem; gap: 1.75rem }``,
# and the narrow-screen rule below 33rem.
COLUMN_WIDTH = 640
NARROW_WIDTH = 528

# Sizes are the on-disk faster-whisper downloads, rounded.
WHISPER_MODELS: tuple[tuple[str, str, str], ...] = (
    ("tiny", "Tiny", "~75 MB · fastest, roughest"),
    ("base", "Base", "~145 MB"),
    ("small", "Small", "~465 MB · a good default"),
    ("medium", "Medium", "~1.5 GB"),
    ("large-v3", "Large v3", "~3 GB · slowest, most accurate"),
)
CUSTOM_MODEL = ("", "Something else", "A model name or a folder on this computer")

DEVICES: tuple[tuple[str, str, str], ...] = (
    ("auto", "Automatic", "Graphics card when it works, otherwise the processor"),
    ("cuda", "Graphics card", "NVIDIA CUDA"),
    ("cpu", "Processor", "Slower, but always available"),
)

TERMINAL = frozenset({"completed", "failed", "cancelled"})

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


def fetch_bytes(url: str) -> bytes:
    """Download a thumbnail, refusing anything unreasonably large."""
    request = urllib.request.Request(url, headers={"User-Agent": APP_NAME})
    with urllib.request.urlopen(request, timeout=THUMBNAIL_TIMEOUT_SECONDS) as reply:
        return bytes(reply.read(THUMBNAIL_MAX_BYTES))


def open_path(path: Path) -> bool:
    """Open a file or a folder with whatever this computer uses for it."""
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


@dataclass
class Services:
    """Everything the window asks of the rest of CaptionForge.

    The real ones are the very objects the web server uses; tests hand the
    window stand-ins instead.
    """

    inspect: Callable[[str, str, bool], VideoInspection]
    updater: Any
    cuda: Callable[[], CudaStatus]
    fetch: Callable[[str], bytes]
    open_path: Callable[[Path], bool]

    @classmethod
    def for_config(cls, config: Config) -> Services:
        """The services a real session uses."""

        def inspect(url: str, language: str, allow_translated: bool) -> VideoInspection:
            return create_video_service(config).inspect_all(
                url, language, allow_translated=allow_translated
            )

        return cls(
            inspect=inspect,
            updater=package_updater.UPDATER,
            cuda=CudaStatus.probe,
            fetch=fetch_bytes,
            open_path=open_path,
        )


def js_round(value: float) -> int:
    """Round halves up, as ``Math.round`` does, so numbers match the page."""
    return math.floor(value + 0.5)


def format_duration(seconds: float | None) -> str:
    """``1:02:03`` or ``2:03``, the way the page writes a video's length."""
    if seconds is None:
        return ""
    total = js_round(seconds)
    hours, minutes, rest = total // 3600, (total % 3600) // 60, total % 60
    if hours:
        return f"{hours}:{minutes:02d}:{rest:02d}"
    return f"{minutes}:{rest:02d}"


def format_size(size: int) -> str:
    """Bytes as B, KB or MB with one decimal, the way the page writes them."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def cuda_reason(status: CudaStatus) -> str:
    """Say why the graphics card cannot be picked, with the fix when there is one."""
    if status.device_present and status.missing_libraries:
        # The card is there; CTranslate2 just cannot load what it links against.
        return (
            f"Detected, but {' and '.join(status.missing_libraries)} will not load "
            "— install nvidia-cublas-cu12 and nvidia-cudnn-cu12 to use it"
        )
    return "No NVIDIA graphics card detected on this computer"


def results_heading(job: dict[str, Any]) -> str:
    """What the results list is headed with, by what the job did."""
    if job.get("kind") == "media":
        media = job.get("media") or {}
        label = media.get("label", "file")
        return (
            f"Saved the audio as {label}"
            if media.get("kind") == "audio"
            else f"Saved the video at {label}"
        )
    return (
        "Exported the video's own captions"
        if job.get("used_existing_captions")
        else "Transcribed on this computer"
    )


def message_for(error: BaseException, context: str) -> str:
    """The short message a person sees; the technical cause goes to the log."""
    log = get_logger()
    if isinstance(error, CaptionForgeError):
        log.error(
            "Desktop {} failed user_message={} technical_cause={}",
            context,
            error.message,
            error.details or type(error).__name__,
        )
        return error.message
    log.opt(exception=error).error("Unexpected desktop {} failure", context)
    return GENERIC_MESSAGE


class Page(QWidget):
    """The scrolling column, 40rem wide at most and centred in the window."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("page")
        self.main = column(28)
        self.setLayout(self.main)
        self.narrow_changed: Callable[[bool], None] | None = None
        self._narrow: bool | None = None
        self._fit(COLUMN_WIDTH)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt's own name
        """Keep the column centred, and switch to the narrow spacing when small."""
        super().resizeEvent(event)
        self._fit(event.size().width())

    def _fit(self, width: int) -> None:
        narrow = width <= NARROW_WIDTH
        side = max(0, (width - COLUMN_WIDTH) // 2)
        if narrow:
            self.main.setContentsMargins(side + 18, 28, side + 18, 64)
            self.main.setSpacing(22)
        else:
            self.main.setContentsMargins(side + 24, 40, side + 24, 96)
            self.main.setSpacing(28)
        if narrow != self._narrow:
            self._narrow = narrow
            if self.narrow_changed is not None:
                self.narrow_changed(narrow)


class Bar(QFrame):
    """The top bar: the mark, the name, and the version and output folder."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("bar")
        layout = row(10, (24, 18, 24, 18))
        brand = QLabel(APP_NAME)
        brand.setObjectName("brand")
        self.meta = ElidedLabel()
        self.meta.setObjectName("meta")
        self.meta.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(Mark())
        layout.addWidget(brand)
        layout.addStretch(1)
        layout.addWidget(self.meta, 3)
        self.setLayout(layout)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt's own name
        """``.bar-meta { max-width: 55% }``."""
        super().resizeEvent(event)
        self.meta.setMaximumWidth(int(event.size().width() * 0.55))


class MainWindow(QMainWindow):
    """One window holding the whole workflow, from a link to finished files."""

    # Emitted once the app may exit: the window was closed and nothing it
    # started is still running.
    finished = Signal()

    def __init__(
        self,
        config: Config,
        *,
        registry: JobRegistry | None = None,
        services: Services | None = None,
        preferences_file: Path | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._registry = registry or JobRegistry(config)
        self._services = services or Services.for_config(config)
        self._preferences_file = preferences_file
        self._output_directory = config.default_output_folder.resolve()
        self._preferences = resolve(load_preferences(preferences_file), config)
        self._inspection: VideoInspection | None = None
        self._inspected_url = ""
        self._chosen_formats: set[str] = set()
        self._job_id: str | None = None
        self._running = False
        self._restoring = False
        self._update_ticket = 0
        self._thumb_ticket = 0
        self._update_boxes: list[tuple[OptionCard, QCheckBox]] = []

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self._save_preferences)
        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(IDLE_CHECK_MS)
        self._idle_timer.timeout.connect(self._leave_when_idle)

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(420, 420)
        self._build()
        self._start()

    # ---------- building ----------

    def _build(self) -> None:
        root = QWidget()
        layout = column(0)
        layout.addWidget(self._build_bar())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.page = Page()
        main = self.page.main
        main.addWidget(self._build_source())
        self.hint = Label(
            "CaptionForge reuses the video's own captions when there are any, "
            "and transcribes the audio on this computer when there aren't."
        )
        main.addWidget(self.hint)
        main.addWidget(self._build_alert())
        main.addWidget(self._build_updates())
        main.addWidget(self._build_video())
        main.addWidget(self._build_media())
        main.addWidget(self._build_tracks())
        main.addWidget(self._build_controls())
        main.addWidget(self._build_progress())
        main.addWidget(self._build_results())
        main.addStretch(1)
        self.page.narrow_changed = self._narrow
        self._narrow(False)
        scroll.setWidget(self.page)
        layout.addWidget(scroll, 1)
        root.setLayout(layout)
        self.setCentralWidget(root)

    def _build_bar(self) -> QWidget:
        self.bar = Bar()
        return self.bar

    def _build_source(self) -> QWidget:
        box = QWidget()
        self._source_layout = row(10)
        self.url = QLineEdit()
        self.url.setPlaceholderText("Paste a YouTube video link")
        self.url.setAccessibleName("YouTube video link")
        self.url.returnPressed.connect(self.look_up)
        self.inspect_button = Button("Look up", "ghost")
        on_click(self.inspect_button, self.look_up)
        self._source_layout.addWidget(self.url, 1)
        self._source_layout.addWidget(self.inspect_button)
        box.setLayout(self._source_layout)
        return box

    def _build_alert(self) -> QWidget:
        self.alert = QFrame()
        self.alert.setObjectName("alert")
        layout = column(3, (17, 14, 17, 14))
        self.alert_title = QLabel()
        self.alert_title.setObjectName("alert-title")
        self.alert_title.setFont(font(13, weight=700, spacing=0.5))
        self.alert_body = Label(role="body")
        self.alert_body.setObjectName("alert-body")
        self.alert_body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.alert_title)
        layout.addWidget(self.alert_body)
        self.alert.setLayout(layout)
        self.alert.hide()
        return self.alert

    def _build_updates(self) -> QWidget:
        self.updates = QWidget()
        layout = column(8)
        self.updates_label = SectionLabel("Updates available")
        self.update_list = QWidget()
        self._update_layout = column(6)
        self.update_list.setLayout(self._update_layout)
        self.update_actions = QWidget()
        actions = row(8)
        self.update_button = Button("Update selected", "ghost")
        self.update_button.setEnabled(False)
        on_click(self.update_button, self.install_updates)
        self.update_later = Button("Not now", "quiet")
        on_click(self.update_later, self.updates.hide)
        actions.addWidget(self.update_button)
        actions.addWidget(self.update_later)
        actions.addStretch(1)
        self.update_actions.setLayout(actions)
        self.updates_note = Label("Nothing is installed until you tick it.")
        for widget in (
            self.updates_label,
            self.update_list,
            self.update_actions,
            self.updates_note,
        ):
            layout.addWidget(widget)
        self.updates.setLayout(layout)
        self.updates.hide()
        return self.updates

    def _build_video(self) -> QWidget:
        self.video = QWidget()
        layout = row(16)
        self.thumb = Thumbnail()
        text = column(4)
        self.title = Label(role="title")
        # ``<h2 dir="auto">``: an Arabic title reads right to left.
        self.title.follow_direction()
        self.byline = Label()
        text.addWidget(self.title)
        text.addWidget(self.byline)
        text.addStretch(1)
        # ``align-items: flex-start``.
        layout.addWidget(self.thumb, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(text, 1)
        self.video.setLayout(layout)
        self.video.hide()
        return self.video

    def _build_media(self) -> QWidget:
        self.media = QWidget()
        layout = column(8)
        self.media_list = Chips()
        self.media_note = Label("One click starts it. Sizes are estimates.")
        layout.addWidget(SectionLabel("Download the file"))
        layout.addWidget(self.media_list)
        layout.addWidget(self.media_note)
        self.media.setLayout(layout)
        self.media.hide()
        return self.media

    def _build_tracks(self) -> QWidget:
        self.tracks = QWidget()
        layout = column(0)
        self.track_list = Chips()
        self.selection = Label()
        layout.addWidget(SectionLabel("Caption tracks"))
        layout.addWidget(self.track_list)
        layout.addSpacing(2)
        layout.addWidget(self.selection)
        self.tracks.setLayout(layout)
        self.tracks.hide()
        return self.tracks

    def _build_controls(self) -> QWidget:
        self.controls = QFrame()
        self.controls.setObjectName("controls")
        layout = column(20, (20, 20, 20, 20))

        fields = row(20)
        language_field = column(0)
        language_field.addWidget(SectionLabel("Language"))
        self.language = QLineEdit()
        self.language.setAccessibleName("Language")
        self.language.setFixedWidth(88)
        language_field.addWidget(self.language)
        language_field.addStretch(1)
        formats_field = column(0)
        formats_field.addWidget(SectionLabel("Formats"))
        self.formats = Chips()
        formats_field.addWidget(self.formats)
        formats_field.addStretch(1)
        fields.addLayout(language_field)
        fields.addLayout(formats_field, 1)
        layout.addLayout(fields)

        self.options = Disclosure("Options", self._build_options())
        layout.addWidget(self.options)

        self.run_button = Button("Get captions", "go")
        on_click(self.run_button, self.run)
        layout.addWidget(self.run_button)
        self.controls.setLayout(layout)
        self.controls.hide()
        return self.controls

    def _build_options(self) -> QWidget:
        body = QWidget()
        grid = column(10)
        self.force = QCheckBox("Always transcribe, ignore existing captions")
        self.allow_translated = QCheckBox("Accept machine-translated tracks")
        self.timestamped = QCheckBox("Timestamps in the TXT file")
        self.overwrite = QCheckBox("Replace files instead of numbering them")
        self.postprocess = QCheckBox("Clean and reflow the subtitles")
        self.postprocess.setChecked(True)
        self.keep_audio = QCheckBox("Keep the prepared audio")
        for box in (
            self.force,
            self.allow_translated,
            self.timestamped,
            self.overwrite,
            self.postprocess,
            self.keep_audio,
        ):
            box.setCursor(Qt.CursorShape.PointingHandCursor)
            grid.addWidget(box)

        models = column(0)
        models.addWidget(SectionLabel("Whisper model"))
        self.model_list = column(6)
        self.model_group = QButtonGroup(self)
        models.addLayout(self.model_list)
        self.model_custom = QLineEdit()
        self.model_custom.setAccessibleName("Custom model name or folder")
        self.model_custom.setPlaceholderText(
            "Model name or folder, e.g. tarteel-ai/whisper-base-ar-quran"
        )
        self.model_custom.hide()
        models.addSpacing(8)
        models.addWidget(self.model_custom)
        models.addSpacing(7)
        models.addWidget(
            Label(
                "Only used when the video has no caption track to reuse. Bigger "
                "models are more accurate and slower, and each one downloads once "
                "the first time you pick it.",
                role="help",
            )
        )
        grid.addLayout(models)

        devices = column(0)
        devices.addWidget(SectionLabel("Device"))
        device_list = column(6)
        self.device_group = QButtonGroup(self)
        self.device_cards: dict[str, OptionCard] = {}
        for value, name, note in DEVICES:
            radio = QRadioButton()
            radio.setProperty("value", value)
            card = OptionCard(radio, name, note)
            self.device_group.addButton(radio)
            self.device_cards[value] = card
            device_list.addWidget(card)
        self.device_cards["auto"].indicator.setChecked(True)
        devices.addLayout(device_list)
        grid.addLayout(devices)

        names = column(0)
        names.addWidget(SectionLabel("Names and spellings"))
        self.prompt = QLineEdit()
        self.prompt.setAccessibleName("Names and spellings")
        self.prompt.setPlaceholderText("Speaker names, places, recurring terms")
        names.addWidget(self.prompt)
        names.addSpacing(7)
        names.addWidget(
            Label(
                "Whisper reads this before it listens, so unusual words come out "
                "spelled the way you write them here. Write the names as you want "
                "them to appear, separated by commas. It is a hint, not a filter: "
                "nothing is dropped for being absent from the list.",
                role="help",
            )
        )
        grid.addLayout(names)
        body.setLayout(grid)
        return body

    def _build_progress(self) -> QWidget:
        self.progress = QWidget()
        layout = column(11)
        head = row(16)
        self.stage = QLabel("Starting")
        self.stage.setObjectName("stage")
        self.stage.setWordWrap(True)
        self.pct = QLabel("0%")
        self.pct.setObjectName("pct")
        head.addWidget(self.stage, 1)
        head.addWidget(self.pct)
        self.track = ProgressTrack()
        self.cancel_button = Button("Cancel", "quiet")
        on_click(self.cancel_button, self.cancel)
        layout.addLayout(head)
        layout.addWidget(self.track)
        # ``.btn-quiet { align-self: flex-start }``.
        layout.addWidget(self.cancel_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.progress.setLayout(layout)
        self.progress.hide()
        return self.progress

    def _build_results(self) -> QWidget:
        self.results = QWidget()
        layout = column(11)
        self.results_label = SectionLabel("Done")
        self.files = QWidget()
        self._files_layout = column(6)
        self.files.setLayout(self._files_layout)
        self.results_note = Label()
        self.results_note.linkActivated.connect(self._open_output_folder)
        layout.addWidget(self.results_label)
        layout.addWidget(self.files)
        layout.addWidget(self.results_note)
        self.results.setLayout(layout)
        self.results.hide()
        return self.results

    def _narrow(self, narrow: bool) -> None:
        """``@media (max-width: 33rem) { .source { flex-direction: column } }``."""
        self._source_layout.setDirection(
            QBoxLayout.Direction.TopToBottom
            if narrow
            else QBoxLayout.Direction.LeftToRight
        )

    # ---------- startup ----------

    def _start(self) -> None:
        self.bar.meta.set_full_text(f"v{VERSION} · {self._output_directory}")
        self._apply_preferences()
        self._watch_for_changes()
        run_in_background(
            self._services.cuda,
            self._apply_cuda,
            self._cuda_unknown,
            parent=self,
            name="captionforge-cuda",
        )
        self.offer_updates()

    def _apply_preferences(self) -> None:
        saved = self._preferences
        self._restoring = True
        try:
            self.language.setText(
                saved.get("language") or self._config.default_language
            )
            self.prompt.setText(saved.get("prompt") or "")
            for box, key in (
                (self.force, "force"),
                (self.overwrite, "overwrite"),
                (self.keep_audio, "keep_audio"),
                (self.timestamped, "timestamped_txt"),
                (self.allow_translated, "allow_translated"),
            ):
                box.setChecked(bool(saved.get(key)))
            # Cleaning subtitles is on unless it was explicitly turned off.
            self.postprocess.setChecked(saved.get("postprocess") is not False)
            self._chosen_formats = set(self._initial_formats())
            self._render_formats()
            self._render_models()
            self._select_device(saved.get("device") or self._config.whisper_device)
        finally:
            self._restoring = False
        # Options that differ from the defaults are worth showing on arrival.
        if (
            self.force.isChecked()
            or self.overwrite.isChecked()
            or self.keep_audio.isChecked()
            or self.timestamped.isChecked()
            or self.allow_translated.isChecked()
            or not self.postprocess.isChecked()
            or self.prompt.text()
        ):
            self.options.set_open(True)

    def _initial_formats(self) -> list[str]:
        saved = [
            item
            for item in self._preferences.get("formats") or []
            if item in SUPPORTED_OUTPUT_FORMATS
        ]
        return saved or ["srt"]

    def _render_formats(self) -> None:
        self.formats.clear()
        for name in sorted(SUPPORTED_OUTPUT_FORMATS):
            chip = Chip(name)
            chip.set_picked(name in self._chosen_formats)
            on_click(chip, partial(self._toggle_format, chip))
            self.formats.add(chip)

    def _toggle_format(self, chip: Chip) -> None:
        name = chip.text()
        if name in self._chosen_formats:
            self._chosen_formats.discard(name)
        else:
            self._chosen_formats.add(name)
        chip.set_picked(name in self._chosen_formats)
        self.remember_choices()

    def _render_models(self) -> None:
        wanted = self._preferences.get("model") or self._config.default_whisper_model
        known = any(value == wanted for value, _, _ in WHISPER_MODELS)
        for value, name, note in (*WHISPER_MODELS, CUSTOM_MODEL):
            radio = QRadioButton()
            radio.setProperty("value", value)
            self.model_group.addButton(radio)
            self.model_list.addWidget(OptionCard(radio, name, note))
            radio.setChecked((not known) if value == "" else value == wanted)
        self.model_custom.setVisible(not known)
        if not known:
            self.model_custom.setText(wanted)
        self.model_group.buttonToggled.connect(self._model_toggled)

    def _model_toggled(self, button: QRadioButton, checked: bool) -> None:
        if not checked:
            return
        custom = button.property("value") == ""
        self.model_custom.setVisible(custom)
        if custom:
            self.model_custom.setFocus()

    def _select_device(self, wanted: str) -> None:
        card = self.device_cards.get(wanted)
        if card is not None and card.isEnabled():
            card.indicator.setChecked(True)

    def _apply_cuda(self, status: CudaStatus) -> None:
        if status.available:
            return
        card = self.device_cards["cuda"]
        card.setEnabled(False)
        card.text.set_note(cuda_reason(status))
        if card.indicator.isChecked():
            self._restoring = True
            try:
                self.device_cards["auto"].indicator.setChecked(True)
            finally:
                self._restoring = False

    def _cuda_unknown(self, error: BaseException) -> None:
        get_logger().warning("Could not tell whether CUDA works: {}", error)
        self._apply_cuda(CudaStatus(False, False, ()))

    # ---------- remembering ----------

    def current_preferences(self) -> dict[str, Any]:
        """The choices on screen, in the shape the preferences file stores."""
        return {
            "language": self.language.text().strip() or None,
            "formats": [
                name
                for name in sorted(SUPPORTED_OUTPUT_FORMATS)
                if name in self._chosen_formats
            ],
            "model": self.chosen_model(),
            "device": self.chosen_device(),
            "prompt": self.prompt.text().strip() or None,
            "force": self.force.isChecked(),
            "overwrite": self.overwrite.isChecked(),
            "keep_audio": self.keep_audio.isChecked(),
            "timestamped_txt": self.timestamped.isChecked(),
            "postprocess": self.postprocess.isChecked(),
            "allow_translated": self.allow_translated.isChecked(),
        }

    def remember_choices(self) -> None:
        """Save the choices shortly after the last change."""
        if self._restoring:
            return
        self._save_timer.start()

    def _save_preferences(self) -> None:
        try:
            save_preferences(
                WebPreferences(**self.current_preferences()), self._preferences_file
            )
        except Exception as exc:  # noqa: BLE001 - remembering is a convenience
            # Never interrupt the real work for it.
            get_logger().warning("Could not remember the desktop choices: {}", exc)

    def _flush_preferences(self) -> None:
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._save_preferences()

    def _watch_for_changes(self) -> None:
        for field in (self.language, self.prompt, self.model_custom):
            field.textChanged.connect(lambda _text: self.remember_choices())
        for box in (
            self.force,
            self.allow_translated,
            self.timestamped,
            self.overwrite,
            self.postprocess,
            self.keep_audio,
        ):
            box.toggled.connect(lambda _checked: self.remember_choices())
        for group in (self.model_group, self.device_group):
            group.buttonToggled.connect(self._choice_toggled)

    def _choice_toggled(self, _button: QRadioButton, checked: bool) -> None:
        # A radio group reports the one it unchecked too; save once, not twice.
        if checked:
            self.remember_choices()

    def chosen_model(self) -> str | None:
        """The picked model's name, or the custom one typed in."""
        picked = self.model_group.checkedButton()
        if picked is None:
            return None
        value = str(picked.property("value"))
        return value or self.model_custom.text().strip() or None

    def chosen_device(self) -> str | None:
        """auto, cuda or cpu."""
        picked = self.device_group.checkedButton()
        return str(picked.property("value")) if picked is not None else None

    # ---------- view helpers ----------

    def show_alert(self, title: str, message: str) -> None:
        """Show the red box with a heading and a sentence."""
        self.alert_title.setText(title.upper())
        self.alert_body.setText(message)
        self.alert.show()

    def clear_alert(self) -> None:
        """Hide the red box."""
        self.alert.hide()

    def set_busy(self, busy: bool) -> None:
        """Stop every other action while one thing runs."""
        self._running = busy
        self.inspect_button.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        # One click starts a download, so every other one has to stop working
        # while a job runs: two writes to the output folder at once help nobody.
        for chip in self.media_list.chips():
            chip.setEnabled(not busy)
        # Swapping a package out from under a running job helps nobody either.
        self._sync_update_button()

    # ---------- inspection ----------

    def look_up(self) -> None:
        """Read the video behind the link: its details, tracks and downloads."""
        url = self.url.text().strip()
        if not url:
            self.url.setFocus()
            return
        if self._running:
            return
        self.clear_alert()
        self._stop_polling()
        self.results.hide()
        self.progress.hide()
        self.set_busy(True)
        self.inspect_button.setText("Looking up…")
        language = self.language.text().strip() or self._config.default_language
        allow_translated = self.allow_translated.isChecked()
        run_in_background(
            lambda: self._services.inspect(url, language, allow_translated),
            lambda result: self._inspected(url, result),
            self._inspect_failed,
            parent=self,
            name="captionforge-inspect",
        )

    def _inspected(self, url: str, result: VideoInspection) -> None:
        self._inspection = result
        self._inspected_url = url
        self.set_busy(False)
        self.inspect_button.setText("Look up")
        self.render_inspection(result)

    def _inspect_failed(self, error: BaseException) -> None:
        self.set_busy(False)
        self.inspect_button.setText("Look up")
        self.video.hide()
        self.media.hide()
        self.tracks.hide()
        self.show_alert("Could not read that video", message_for(error, "lookup"))
        if isinstance(error, CaptionForgeError) and update_may_help(error):
            self.offer_updates(refused=True)

    def render_inspection(self, result: VideoInspection) -> None:
        """Fill the video, download, track and control sections from one lookup."""
        discovery = result.discovery
        video = discovery.video
        self.title.setText(video.title)
        parts = [video.channel_name, format_duration(video.duration_seconds)]
        self.byline.setText(" · ".join(part for part in parts if part))
        self._thumb_ticket += 1
        self.thumb.set_pixmap(None)
        if video.thumbnail_url:
            self.thumb.show()
            self._fetch_thumbnail(video.thumbnail_url, self._thumb_ticket)
        else:
            self.thumb.hide()
        self.video.show()
        self.render_media(result.media)

        # YouTube publishes ~155 machine translations of its own transcription.
        # Show the real tracks; keep the translations behind one chip.
        every = [*discovery.manual_tracks, *discovery.automatic_tracks]
        real = [track for track in every if not track.is_translated]
        translated = [track for track in every if track.is_translated]
        selected = discovery.selected_track

        self.track_list.clear()
        for track in real:
            self.track_list.add(self._track_chip(track, selected))
        if not every:
            self.track_list.add(Label("None published for this video.", wrap=False))
        if translated:
            more = Chip(f"+{len(translated)} machine-translated")

            def expand() -> None:
                more.hide()
                more.setEnabled(False)
                for track in translated:
                    self.track_list.add(self._track_chip(track, selected))

            on_click(more, expand)
            self.track_list.add(more)

        self.selection.setText(
            f"Will export the highlighted track "
            f"({discovery.selection_reason or 'preferred match'})."
            if selected
            else "No matching track, so the audio will be transcribed on this computer."
        )
        self.tracks.show()
        self.controls.show()

    def _fetch_thumbnail(self, url: str, ticket: int) -> None:
        def arrived(data: bytes) -> None:
            if ticket != self._thumb_ticket:
                return
            image = QImage.fromData(data)
            if not image.isNull():
                self.thumb.set_pixmap(QPixmap.fromImage(image))

        run_in_background(
            lambda: self._services.fetch(url),
            arrived,
            lambda error: get_logger().debug("Thumbnail not shown: {}", error),
            parent=self,
            name="captionforge-thumbnail",
        )

    @staticmethod
    def _track_chip(track: SubtitleTrack, selected: SubtitleTrack | None) -> Chip:
        kind = " · ".join(
            part
            for part in (
                "auto" if track.is_automatic else "manual",
                "translated" if track.is_translated else None,
            )
            if part
        )
        chip = Chip(track.language_code, kind, interactive=False)
        chip.set_picked(
            selected is not None
            and track.language_code == selected.language_code
            and track.is_automatic == selected.is_automatic
        )
        return chip

    def render_media(self, media: MediaOptions | None) -> None:
        """One chip per quality the video publishes, the MP3 first."""
        # Audio first: it is the smallest file and the most common request, and
        # it anchors a row that otherwise reads as an undifferentiated ladder.
        variants = list(media.variants) if media is not None else []
        ordered = [item for item in variants if item.kind is MediaKind.AUDIO] + [
            item for item in variants if item.kind is not MediaKind.AUDIO
        ]
        self.media_list.clear()
        if not ordered:
            # Nothing offered is not an error worth a paragraph; the row just goes.
            self.media.hide()
            return
        for variant in ordered:
            size = (
                f"~{format_size(variant.estimated_bytes)}"
                if variant.estimated_bytes
                else ""
            )
            chip = Chip(variant.label, size)
            chip.setEnabled(not self._running)
            on_click(chip, partial(self.start_media, variant, chip))
            self.media_list.add(chip)
        self.media.show()

    def start_media(self, variant: MediaVariant, chip: Chip) -> None:
        """Download one quality straight away: the click is the whole request."""
        if self._running:
            return
        self.clear_alert()
        self.results.hide()
        self.set_busy(True)
        chip.set_picked(True)
        self._set_progress(f"Starting {variant.label}", 0, animate=False)
        self.cancel_button.setEnabled(True)
        self.progress.show()
        try:
            record = self._registry.submit_media(
                MediaJobRequest(
                    # The chips describe the video that was looked up, even if
                    # the field has been edited since.
                    url=self._inspected_url or self.url.text().strip(),
                    quality=variant.key,
                    overwrite=self.overwrite.isChecked(),
                )
            )
        except Exception as exc:  # noqa: BLE001 - shown, never raised
            self.progress.hide()
            self.set_busy(False)
            chip.set_picked(False)
            self.show_alert(
                "Could not start the download", message_for(exc, "download")
            )
            return
        self._follow(record.id)

    def _clear_media_choice(self) -> None:
        for chip in self.media_list.chips():
            chip.set_picked(False)

    # ---------- jobs ----------

    def run(self) -> None:
        """Export the captions, or transcribe when there are none to export."""
        if not self._chosen_formats:
            self.show_alert("Pick a format", "Choose at least one output format.")
            return
        if self._running:
            return
        self.clear_alert()
        self.results.hide()
        self.set_busy(True)
        self._set_progress("Queued", 0, animate=False)
        self.cancel_button.setEnabled(True)
        self.progress.show()
        preferences = self.current_preferences()
        try:
            record = self._registry.submit(
                JobRequest(
                    url=self.url.text().strip(),
                    language=preferences["language"],
                    formats=tuple(preferences["formats"]),
                    model=preferences["model"],
                    device=preferences["device"],
                    prompt=preferences["prompt"],
                    force=preferences["force"],
                    overwrite=preferences["overwrite"],
                    keep_audio=preferences["keep_audio"],
                    timestamped_txt=preferences["timestamped_txt"],
                    postprocess=preferences["postprocess"],
                    allow_translated=preferences["allow_translated"],
                )
            )
        except Exception as exc:  # noqa: BLE001 - shown, never raised
            self.progress.hide()
            self.set_busy(False)
            self.show_alert("Could not start", message_for(exc, "job"))
            return
        self.remember_choices()
        self._follow(record.id)

    def cancel(self) -> None:
        """Ask the running job to stop at its next checkpoint."""
        if not self._job_id:
            return
        self.cancel_button.setEnabled(False)
        self.stage.setText("Cancelling…")
        if not self._registry.cancel(self._job_id):
            self.show_alert("Could not cancel", "That job is no longer available.")

    def _follow(self, job_id: str) -> None:
        self._job_id = job_id
        self._poll_timer.start()
        self._poll()

    def _stop_polling(self) -> None:
        self._poll_timer.stop()

    def _set_progress(
        self, stage: str, percent: float, *, animate: bool = True
    ) -> None:
        self.stage.setText(stage)
        self.pct.setText(f"{js_round(percent)}%")
        self.track.set_percent(percent, animate=animate)

    def _poll(self) -> None:
        if not self._job_id:
            return
        record = self._registry.get(self._job_id)
        if record is None:
            self._stop_polling()
            self.set_busy(False)
            self.progress.hide()
            self.show_alert("Lost track of the job", "That job is no longer available.")
            return
        job = record.snapshot()
        self._set_progress(str(job["stage"]), float(job["percent"]))
        if job["status"] not in TERMINAL:
            return

        self._stop_polling()
        self.set_busy(False)
        self._clear_media_choice()
        self.progress.hide()
        if job["status"] == "failed":
            self.show_alert(
                "Job failed", job["error"] or "CaptionForge could not finish."
            )
            if job["update_may_help"]:
                self.offer_updates(refused=True)
            return
        if job["status"] == "cancelled":
            self.show_alert("Cancelled", "No incomplete output was kept.")
            return
        self.render_results(job)

    def render_results(self, job: dict[str, Any]) -> None:
        """List the files a job wrote; one click opens each one."""
        self.results_label.setText(results_heading(job))
        clear_layout(self._files_layout)
        for produced in job["files"]:
            path = Path(produced["path"])
            file_row = FileRow(produced["name"], format_size(produced["size_bytes"]))
            on_click(file_row, partial(self._open_file, path))
            self._files_layout.addWidget(file_row)
        folder = html.escape(str(self._output_directory))
        notes = [
            f'Saved to <a href="folder" style="color:{EMERALD}; '
            f'text-decoration:none">{folder}</a>'
        ]
        info = job.get("transcription")
        if job.get("kind") != "media" and info:
            probability = info.get("language_probability")
            confident = (
                f" ({js_round(probability * 100)}% confident)"
                if probability is not None
                else ""
            )
            notes.append(
                html.escape(
                    f"{info['model_name']} on {info['device']}, "
                    f"detected {info['detected_language']}{confident}"
                )
            )
        self.results_note.set_markup(" · ".join(notes))
        self.results.show()

    def _open_file(self, path: Path) -> None:
        if not path.is_file():
            self.show_alert(
                "Could not open the file", "That file is no longer available."
            )
            return
        if not self._services.open_path(path):
            self.show_alert(
                "Could not open the file",
                f"Nothing on this computer opens it. It is saved at {path}.",
            )

    def _open_output_folder(self, _link: str = "") -> None:
        if not self._services.open_path(self._output_directory):
            self.show_alert(
                "Could not open the folder", f"It is at {self._output_directory}."
            )

    # ---------- updates ----------

    def offer_updates(self, *, refused: bool = False) -> None:
        """Offer newer packages. Nothing is installed without a tick."""
        # The check on arrival can answer after a later one made for a refusal;
        # only the newest question's answer may redraw the row.
        self._update_ticket += 1
        ticket = self._update_ticket
        updater = self._services.updater
        enabled = self._config.check_for_updates

        def check() -> tuple[bool, tuple[Any, ...]]:
            if not enabled or not updater.can_update():
                return (False, ())
            # After a refusal the answer has to be current: yt-dlp may have
            # shipped a fix since the window last asked.
            return (True, tuple(updater.check(fresh=refused)))

        run_in_background(
            check,
            lambda answer: self._updates_arrived(ticket, refused, answer),
            # Offline, or pip is unhappy. The work in the window matters more.
            lambda error: get_logger().info("Update check skipped: {}", error),
            parent=self,
            name="captionforge-updates",
        )

    def _updates_arrived(
        self, ticket: int, refused: bool, answer: tuple[bool, tuple[Any, ...]]
    ) -> None:
        if ticket != self._update_ticket:
            return
        checked, updates = answer
        ytdlp = any(update.name == "yt-dlp" for update in updates)
        if refused and checked and not ytdlp:
            self.alert_body.setText(
                self.alert_body.plain()
                + " yt-dlp is already the newest release, so YouTube may be "
                "limiting this connection. Try again later."
            )
        self._render_updates(updates, {"yt-dlp"} if refused and ytdlp else set())
        if refused and ytdlp:
            self.updates_note.setText(
                "A newer yt-dlp usually fixes this. Nothing installs until you "
                "click Update selected."
            )

    def _render_updates(self, updates: tuple[Any, ...], ticked: set[str]) -> None:
        clear_layout(self._update_layout)
        self._update_boxes = []
        if not updates:
            self.updates.hide()
            return
        for update in updates:
            box = QCheckBox()
            box.setProperty("package", update.name)
            box.setChecked(update.name in ticked)
            box.toggled.connect(lambda _checked: self._sync_update_button())
            card = OptionCard(
                box,
                f"{update.name} {update.installed} → {update.latest}",
                update.package.mission,
            )
            self._update_boxes.append((card, box))
            self._update_layout.addWidget(card)
        self.updates_label.setText("Updates available")
        self.updates_note.setText("Nothing is installed until you tick it.")
        self.update_list.show()
        self.update_actions.show()
        self._sync_update_button()
        self.updates.show()

    def ticked_updates(self) -> list[str]:
        """The packages the person ticked."""
        return [
            str(box.property("package"))
            for _, box in self._update_boxes
            if box.isChecked()
        ]

    def _sync_update_button(self) -> None:
        self.update_button.setEnabled(not self._running and bool(self.ticked_updates()))

    def install_updates(self) -> None:
        """Install exactly the ticked packages, and nothing else."""
        packages = self.ticked_updates()
        if not packages or self._running:
            return
        cards = [card for card, _ in self._update_boxes]
        self.set_busy(True)
        for card in cards:
            card.setEnabled(False)
        self.update_later.setEnabled(False)
        self.update_button.setText("Updating…")

        def settle() -> None:
            for card in cards:
                card.setEnabled(True)
            self.update_later.setEnabled(True)
            self.update_button.setText("Update selected")
            self.set_busy(False)

        def installed(updated: tuple[Any, ...]) -> None:
            settle()
            self._report_updates(packages, list(updated))

        def failed(error: BaseException) -> None:
            settle()
            self.show_alert("Could not update", message_for(error, "update"))

        updater = self._services.updater
        run_in_background(
            lambda: updater.install(packages),
            installed,
            failed,
            parent=self,
            name="captionforge-update",
        )

    def _report_updates(self, requested: list[str], updated: list[Any]) -> None:
        kept: list[tuple[OptionCard, QCheckBox]] = []
        for card, box in self._update_boxes:
            if box.property("package") in requested:
                self._update_layout.removeWidget(card)
                card.hide()
                card.deleteLater()
            else:
                kept.append((card, box))
        self._update_boxes = kept
        now = [item for item in updated if not item.restart_needed]
        later = [item for item in updated if item.restart_needed]
        lines: list[str] = []
        if now:
            names = ", ".join(f"{item.name} to {item.version}" for item in now)
            lines.append(f"Updated {names}.")
        if later:
            names = ", ".join(f"{item.name} {item.version}" for item in later)
            lines.append(f"Installed {names}; restart CaptionForge to start using it.")
        if not updated:
            lines.append("Everything you picked was already up to date.")
        if any(item.name == "yt-dlp" for item in now) and not self.alert.isHidden():
            self.clear_alert()
            lines.append("Try again.")
        remaining = bool(self._update_boxes)
        self.update_list.setVisible(remaining)
        self.update_actions.setVisible(remaining)
        self.updates_label.setText("Updates available" if remaining else "Updated")
        self.updates_note.setText(" ".join(lines))
        self._sync_update_button()

    # ---------- lifetime ----------

    def has_unfinished_work(self) -> bool:
        """Whether any download or transcription is still queued or running."""
        return any(record.status in BUSY_STATUSES for record in self._registry.recent())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt's own name
        """Close now, or once the work that is still running has finished."""
        self._flush_preferences()
        if self.has_unfinished_work():
            # Work outlives its window: a transcription started before the
            # window closed still finishes and still writes its files.
            event.ignore()
            self.hide()
            self._idle_timer.start()
            get_logger().info("Desktop window closed; finishing the running job first")
            return
        event.accept()
        self.finished.emit()

    def _leave_when_idle(self) -> None:
        if self.has_unfinished_work():
            return
        self._idle_timer.stop()
        get_logger().info("Desktop app stopping: the last job has finished")
        self.finished.emit()

    def bring_forward(self) -> None:
        """Show the window again, on top, for a second launch."""
        self._idle_timer.stop()
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def shutdown(self) -> None:
        """Save what is pending and stop the job threads."""
        self._flush_preferences()
        self._stop_polling()
        self._registry.shutdown()


__all__ = ["MainWindow", "Services", "format_duration", "format_size"]
