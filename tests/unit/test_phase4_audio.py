"""Offline coverage for Phase 4 audio preparation."""

import subprocess
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner
from yt_dlp.utils import DownloadError

from app.adapters.ffmpeg_adapter import FFmpegAdapter
from app.adapters.ytdlp_adapter import Extractor, YtDlpAdapter
from app.core.config import Config
from app.core.exceptions import (
    AudioConversionError,
    AudioDownloadError,
    AudioFormatUnavailableError,
    FFmpegNotFoundError,
    SubtitleDiscoveryError,
)
from app.interfaces import cli
from app.models.subtitle import SubtitleDiscoveryResult
from app.services.audio_service import AudioService
from app.utils.file_utils import create_job_directory
from tests.conftest import VIDEO_ID, VIDEO_URL, FakeExtractor, make_track


class AudioExtractor(FakeExtractor):
    """Fake yt-dlp extractor that materializes the configured output."""

    options: dict[str, Any]

    def extract_info(self, url: str, *, download: bool) -> dict[str, Any]:
        result = super().extract_info(url, download=download)
        template = Path(self.options["outtmpl"].replace("%(ext)s", "webm"))
        template.write_bytes(b"audio")
        for hook in self.options["progress_hooks"]:
            hook({"downloaded_bytes": 5, "total_bytes": 10})
        return dict(result)


def test_audio_only_download_and_progress(tmp_path: Path) -> None:
    extractor = AudioExtractor({"id": VIDEO_ID})
    captured: dict[str, Any] = {}

    def factory(options: dict[str, Any]) -> AbstractContextManager[Extractor]:
        captured.update(options)
        extractor.options = options
        return extractor

    progress: list[float | None] = []
    path = YtDlpAdapter(factory).download_audio(
        VIDEO_ID, tmp_path, lambda _message, value: progress.append(value)
    )

    assert captured["format"] == "bestaudio"
    assert captured["skip_download"] is False
    assert captured["noplaylist"] is True
    assert path.read_bytes() == b"audio"
    assert progress == [50.0]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (DownloadError("network secret"), AudioDownloadError),
        (
            DownloadError("Requested format is not available"),
            AudioFormatUnavailableError,
        ),
    ],
)
def test_audio_download_errors_are_translated(
    tmp_path: Path, error: Exception, expected: type[Exception]
) -> None:
    adapter = YtDlpAdapter(lambda _options: FakeExtractor(error=error))
    with pytest.raises(expected):
        adapter.download_audio(VIDEO_ID, tmp_path)


def test_ffmpeg_command_defaults() -> None:
    command = FFmpegAdapter("/opt/ffmpeg").build_conversion_command(
        Path("input.webm"), Path("output.wav")
    )
    assert command == [
        "/opt/ffmpeg",
        "-y",
        "-i",
        "input.webm",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "output.wav",
    ]


def test_successful_ffmpeg_conversion(tmp_path: Path) -> None:
    destination = tmp_path / "prepared.wav"

    def runner(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"RIFF")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    assert (
        FFmpegAdapter(runner=runner).convert(tmp_path / "source.webm", destination)
        == destination
    )


def test_ffmpeg_missing() -> None:
    def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    with pytest.raises(FFmpegNotFoundError):
        FFmpegAdapter(runner=runner).version()


def test_ffmpeg_conversion_failure_hides_process_error(tmp_path: Path) -> None:
    def runner(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, command, stderr="internal secret")

    with pytest.raises(AudioConversionError, match="could not convert"):
        FFmpegAdapter(runner=runner).convert(
            tmp_path / "source", tmp_path / "output.wav"
        )


def test_temporary_folder_creation(tmp_path: Path) -> None:
    from uuid import uuid4

    result = create_job_directory(tmp_path / "nested", uuid4())
    assert result.is_dir()
    assert result.parent == tmp_path / "nested"


class StubVideoService:
    def __init__(self, result: SubtitleDiscoveryResult) -> None:
        self.result = result

    def inspect(self, _url: str, _language: str) -> SubtitleDiscoveryResult:
        return self.result


class StubAudioDownloader:
    fail = False

    def download_audio(
        self, _video_id: str, destination: Path, _progress: object
    ) -> Path:
        source = destination / "source.webm"
        source.write_bytes(b"audio")
        if self.fail:
            raise AudioDownloadError("Download failed")
        return source


class StubFFmpeg:
    fail = False

    def convert(self, _source: Path, destination: Path, **_kwargs: object) -> Path:
        if self.fail:
            raise AudioConversionError("Conversion failed")
        destination.write_bytes(b"RIFF")
        return destination


def make_service(
    tmp_path: Path,
    video_metadata: object,
    *,
    track: bool = False,
    download_fail: bool = False,
    conversion_fail: bool = False,
) -> AudioService:
    selected = make_track("ar") if track else None
    result = SubtitleDiscoveryResult(
        video=video_metadata,  # type: ignore[arg-type]
        selected_track=selected,
        preferred_language="ar",
    )
    downloader = StubAudioDownloader()
    downloader.fail = download_fail
    ffmpeg = StubFFmpeg()
    ffmpeg.fail = conversion_fail
    return AudioService(
        StubVideoService(result),  # type: ignore[arg-type]
        downloader,  # type: ignore[arg-type]
        ffmpeg,  # type: ignore[arg-type]
        Config(temp_directory=tmp_path),
    )


def test_cleanup_after_success(tmp_path: Path, video_metadata: object) -> None:
    path = make_service(tmp_path, video_metadata).prepare(VIDEO_URL, "ar")
    assert path.is_file()
    assert not list(tmp_path.glob("captionforge-*/*"))


@pytest.mark.parametrize("failure", ["download", "conversion"])
def test_cleanup_after_failure(
    tmp_path: Path, video_metadata: object, failure: str
) -> None:
    service = make_service(
        tmp_path,
        video_metadata,
        download_fail=failure == "download",
        conversion_fail=failure == "conversion",
    )
    with pytest.raises((AudioDownloadError, AudioConversionError)):
        service.prepare(VIDEO_URL, "ar")
    assert not list(tmp_path.glob("captionforge-*"))


def test_keep_temp_behavior(tmp_path: Path, video_metadata: object) -> None:
    path = make_service(tmp_path, video_metadata).prepare(
        VIDEO_URL, "ar", keep_temp=True
    )
    assert path.parent.name.startswith("captionforge-")
    assert (path.parent / "source.webm").exists()


def test_existing_caption_stops_unless_forced(
    tmp_path: Path, video_metadata: object
) -> None:
    service = make_service(tmp_path, video_metadata, track=True)
    with pytest.raises(SubtitleDiscoveryError, match="already exist"):
        service.prepare(VIDEO_URL, "ar")
    assert service.prepare(VIDEO_URL, "ar", force=True).is_file()


def test_prepare_audio_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = tmp_path / "prepared.wav"
    prepared.write_bytes(b"RIFF")

    class StubService:
        def __init__(self, *_args: object) -> None:
            pass

        def prepare(self, *_args: object, **_kwargs: object) -> Path:
            return prepared

    monkeypatch.setattr(cli, "AudioService", StubService)
    result = CliRunner().invoke(
        cli.app, ["prepare-audio", VIDEO_URL, "--force", "--output-temp", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert str(prepared) in result.stdout
    assert "Phase 5" in result.stdout
