"""Offline tests for video and audio files on this computer.

FFmpeg and FFprobe are stand-ins here, and any attempt to reach YouTube fails
the test outright: a local file must never be looked up, or fetched, online.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from app.adapters.ffmpeg_adapter import FFmpegAdapter, MediaProbe
from app.adapters.whisper_adapter import WhisperAdapter
from app.core.config import Config
from app.core.exceptions import (
    FFmpegNotFoundError,
    LocalFileNotFoundError,
    MediaFormatUnavailableError,
    NoAudioStreamError,
    UnreadableMediaFileError,
)
from app.interfaces import cli
from app.models.subtitle import SubtitleDiscoveryResult
from app.models.transcription import TranscriptionResult, TranscriptionSegment
from app.models.video import VideoMetadata
from app.services.audio_service import AudioService
from app.services.export_service import ExportService
from app.services.media_service import MediaService
from app.services.subtitle_service import SubtitleService
from app.services.transcription_service import TranscriptionService
from app.services.video_service import VideoService
from app.utils.local_media import local_media_path, resolve_media_file
from tests.conftest import VIDEO_ID

# ---------- stand-ins ----------


class NoYouTube:
    """A yt-dlp adapter that fails the test if anything asks it for anything."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"YouTube was asked to {name} for a local file")


class StubFFmpeg:
    """FFprobe reports a fixed answer; FFmpeg writes a token WAV."""

    def __init__(self, probe: MediaProbe | None = None) -> None:
        self.answer = probe or MediaProbe(duration_seconds=125.4, has_audio=True)
        self.probed: list[Path] = []
        self.converted: list[Path] = []

    def probe(self, source: Path) -> MediaProbe:
        self.probed.append(source)
        return self.answer

    def convert(self, source: Path, destination: Path, **_: Any) -> Path:
        self.converted.append(source)
        destination.write_bytes(b"RIFF")
        return destination


class StubWhisper:
    """Transcribes anything into one Arabic word."""

    calls = 0

    def select_device(self, requested: str) -> str:
        return "cpu" if requested == "auto" else requested

    @staticmethod
    def select_compute_type(requested: str, device: str) -> str:
        return WhisperAdapter.select_compute_type(requested, device)

    def transcribe(self, _audio: Path, **kwargs: Any) -> TranscriptionResult:
        self.calls += 1
        return TranscriptionResult(
            segments=(
                TranscriptionSegment(
                    index=1, start_seconds=0, end_seconds=1, text="محاضرة"
                ),
            ),
            detected_language="ar",
            model_name=str(kwargs["model_name"]),
            device="cpu",
            compute_type="int8",
        )


def media_file(tmp_path: Path, name: str = "lecture 3.mp4") -> Path:
    """A file that exists; its bytes never matter to the stand-ins."""
    path = tmp_path / "videos" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a video")
    return path


def video_service(ffmpeg: StubFFmpeg) -> VideoService:
    return VideoService(
        NoYouTube(),  # type: ignore[arg-type]
        SubtitleService(),
        Config(),
        ffmpeg,  # type: ignore[arg-type]
    )


def completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


def flat(text: str) -> str:
    """Terminal output with Rich's wrapping and panel borders taken out."""
    return " ".join(text.replace("│", " ").split())


# ---------- telling a file from a link ----------


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("/home/me/Videos/talk.mkv", "/home/me/Videos/talk.mkv"),
        ("  /home/me/Videos/talk.mkv\n", "/home/me/Videos/talk.mkv"),
        # Windows' "Copy as path" and shells both wrap paths in quotes.
        ('"/home/me/My Videos/talk.mkv"', "/home/me/My Videos/talk.mkv"),
        ("'/home/me/My Videos/talk.mkv'", "/home/me/My Videos/talk.mkv"),
        # File managers and drags hand over URIs, percent-encoded.
        ("file:///home/me/My%20Videos/talk.mkv", "/home/me/My Videos/talk.mkv"),
        ("file://localhost/home/me/a.mp3", "/home/me/a.mp3"),
        (
            "file:///home/me/%D9%85%D8%AD%D8%A7%D8%B6%D8%B1%D8%A9.mp3",
            "/home/me/محاضرة.mp3",
        ),
        ("file:///C:/Users/me/talk.mp4", "C:/Users/me/talk.mp4"),
        (r"C:\Users\me\talk.mp4", r"C:\Users\me\talk.mp4"),
        ("./talk.mp4", "talk.mp4"),
        # A bare name with a media extension is a file, even a missing one.
        ("talk.MP4", "talk.MP4"),
    ],
)
def test_typed_paths_name_files(typed: str, expected: str) -> None:
    assert local_media_path(typed) == Path(expected)


def test_a_home_relative_path_is_expanded() -> None:
    assert local_media_path("~/talk.mp3") == Path.home() / "talk.mp3"


@pytest.mark.parametrize(
    "typed",
    [
        "https://youtu.be/qJFbKl6RjLU",
        "https://www.youtube.com/watch?v=qJFbKl6RjLU",
        "youtu.be/qJFbKl6RjLU",
        "www.youtube.com/watch?v=qJFbKl6RjLU",
        "file://server/share/talk.mp4",
        "",
        "   ",
    ],
)
def test_links_are_not_files(typed: str) -> None:
    assert local_media_path(typed) is None


def test_an_existing_file_counts_whatever_its_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "recording").write_bytes(b"x")
    assert local_media_path("recording") == Path("recording")


def test_a_missing_file_is_named_in_the_error(tmp_path: Path) -> None:
    with pytest.raises(LocalFileNotFoundError, match="No file was found at"):
        resolve_media_file(tmp_path / "gone.mp4")


def test_a_folder_is_not_a_file(tmp_path: Path) -> None:
    with pytest.raises(LocalFileNotFoundError, match="is a folder"):
        resolve_media_file(tmp_path)


def test_a_file_resolves_to_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = media_file(tmp_path)
    monkeypatch.chdir(source.parent)
    assert resolve_media_file(Path(source.name)) == source.resolve()


# ---------- FFprobe ----------


def test_probe_reads_duration_and_sound() -> None:
    seen: dict[str, Any] = {}

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["command"] = command
        seen.update(kwargs)
        return completed(
            json.dumps(
                {
                    "format": {"duration": "125.400000"},
                    "streams": [
                        {"codec_type": "video"},
                        {"codec_type": "audio"},
                    ],
                }
            )
        )

    probe = FFmpegAdapter("/opt/ff/bin/ffmpeg", runner=runner).probe(Path("/a.mp4"))

    assert probe == MediaProbe(duration_seconds=125.4, has_audio=True)
    assert seen["command"][0] == "/opt/ff/bin/ffprobe"
    assert seen["command"][-1] == "/a.mp4"
    assert seen["encoding"] == "utf-8"


def test_probe_falls_back_to_stream_durations() -> None:
    output = json.dumps(
        {
            "format": {"duration": "N/A"},
            "streams": [{"codec_type": "audio", "duration": "61.5"}],
        }
    )
    adapter = FFmpegAdapter(runner=lambda *_a, **_k: completed(output))
    assert adapter.probe(Path("/a.opus")).duration_seconds == 61.5


def test_probe_sees_a_file_without_sound() -> None:
    output = json.dumps({"streams": [{"codec_type": "video"}]})
    adapter = FFmpegAdapter(runner=lambda *_a, **_k: completed(output))
    assert adapter.probe(Path("/a.mp4")) == MediaProbe(None, has_audio=False)


def test_probe_refusal_is_reported_without_ffprobe_output() -> None:
    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, command, stderr="Invalid data found")

    with pytest.raises(UnreadableMediaFileError) as caught:
        FFmpegAdapter(runner=runner).probe(Path("/notes.txt"))
    assert caught.value.message == (
        "CaptionForge cannot read that file as video or audio."
    )
    assert caught.value.details == "Invalid data found"


def test_missing_ffprobe_says_what_is_missing() -> None:
    def runner(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("ffprobe")

    with pytest.raises(FFmpegNotFoundError, match="FFprobe"):
        FFmpegAdapter(runner=runner).probe(Path("/a.mp4"))


@pytest.mark.parametrize(
    ("ffmpeg", "ffprobe"),
    [
        ("ffmpeg", "ffprobe"),
        ("/usr/local/bin/ffmpeg", "/usr/local/bin/ffprobe"),
        ("/opt/tools/ffmpeg.exe", "/opt/tools/ffprobe.exe"),
        ("/opt/tools/encoder", "/opt/tools/ffprobe"),
    ],
)
def test_ffprobe_is_found_beside_ffmpeg(ffmpeg: str, ffprobe: str) -> None:
    assert FFmpegAdapter(ffmpeg).probe_executable == ffprobe


# ---------- inspection ----------


def test_a_file_is_inspected_without_youtube(tmp_path: Path) -> None:
    source = media_file(tmp_path, "محاضرة 3.mp4")
    ffmpeg = StubFFmpeg()

    result = video_service(ffmpeg).inspect_all(f'"{source}"', "ar")

    video = result.discovery.video
    assert video.is_local and video.video_id is None
    assert video.local_path == source.resolve()
    # The name it already has, so lecture.mp4 exports lecture.srt.
    assert video.title == "محاضرة 3"
    assert video.duration_seconds == 125
    assert video.webpage_url.startswith("file://")
    assert result.discovery.selected_track is None
    assert result.media.variants == ()
    assert ffmpeg.probed == [source.resolve()]


def test_a_file_without_sound_is_refused_at_lookup(tmp_path: Path) -> None:
    ffmpeg = StubFFmpeg(MediaProbe(duration_seconds=2.0, has_audio=False))
    with pytest.raises(NoAudioStreamError, match="no sound"):
        video_service(ffmpeg).inspect_all(str(media_file(tmp_path)), "ar")


def test_a_missing_file_is_never_mistaken_for_a_bad_link(tmp_path: Path) -> None:
    with pytest.raises(LocalFileNotFoundError):
        video_service(StubFFmpeg()).inspect_all(str(tmp_path / "gone.mp4"), "ar")


def test_a_video_comes_from_exactly_one_place(tmp_path: Path) -> None:
    common = {"title": "t", "webpage_url": "u", "original_url": "u"}
    with pytest.raises(ValidationError):
        VideoMetadata(**common)
    with pytest.raises(ValidationError):
        VideoMetadata(video_id=VIDEO_ID, local_path=tmp_path, **common)


# ---------- audio and transcription ----------


def local_discovery(source: Path) -> SubtitleDiscoveryResult:
    return SubtitleDiscoveryResult(
        video=VideoMetadata(
            title=source.stem,
            duration_seconds=60,
            webpage_url=source.as_uri(),
            original_url=str(source),
            local_path=source,
        ),
        preferred_language="ar",
    )


def test_audio_is_converted_from_the_file_where_it_lies(tmp_path: Path) -> None:
    source = media_file(tmp_path)
    ffmpeg = StubFFmpeg()
    service = AudioService(
        video_service(ffmpeg),
        NoYouTube(),  # type: ignore[arg-type]
        ffmpeg,  # type: ignore[arg-type]
        Config(temp_directory=tmp_path / "temp"),
    )

    prepared = service.prepare(
        str(source), "ar", discovery=local_discovery(source), force=True
    )

    assert ffmpeg.converted == [source]
    assert prepared.read_bytes() == b"RIFF"
    # The original is read, never moved, copied into temp, or removed.
    assert source.read_bytes() == b"not really a video"
    assert not list((tmp_path / "temp").glob("captionforge-*/"))


def test_a_file_is_transcribed_and_named_after_itself(tmp_path: Path) -> None:
    source = media_file(tmp_path)
    ffmpeg = StubFFmpeg()
    config = Config(
        default_output_folder=tmp_path / "output", temp_directory=tmp_path / "temp"
    )
    videos = video_service(ffmpeg)
    whisper = StubWhisper()
    service = TranscriptionService(
        videos,
        NoYouTube(),  # type: ignore[arg-type]
        SubtitleService(),
        AudioService(
            videos,
            NoYouTube(),  # type: ignore[arg-type]
            ffmpeg,  # type: ignore[arg-type]
            config,
        ),
        whisper,  # type: ignore[arg-type]
        ExportService(),
        config,
    )
    messages: list[str] = []

    result = service.process(
        str(source),
        language="ar",
        formats=("srt", "json"),
        progress=lambda message, _percent: messages.append(message),
    )

    assert messages[0] == "Reading the file"
    assert result.used_existing_captions is False
    assert whisper.calls == 1
    assert [path.name for path in result.paths] == ["lecture 3.srt", "lecture 3.json"]
    exported = json.loads(result.paths[1].read_text(encoding="utf-8"))
    assert exported["video"]["video_id"] is None
    assert exported["video"]["local_path"] == str(source.resolve())
    assert source.is_file()
    assert not list((tmp_path / "temp").iterdir())


def test_a_file_offers_nothing_to_download(tmp_path: Path) -> None:
    service = MediaService(
        video_service(StubFFmpeg()),
        NoYouTube(),  # type: ignore[arg-type]
        Config(),
    )
    with pytest.raises(MediaFormatUnavailableError, match="already on this computer"):
        service.download(str(media_file(tmp_path)), "best")


# ---------- the command line ----------


class StubCliVideoService:
    """Answers every inspection with the given file."""

    def __init__(self, discovery: SubtitleDiscoveryResult) -> None:
        self.discovery = discovery

    def inspect(self, *_: Any, **__: Any) -> SubtitleDiscoveryResult:
        return self.discovery


def test_inspect_describes_a_file_without_track_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = media_file(tmp_path)
    monkeypatch.setattr(
        cli,
        "_create_video_service",
        lambda _config=None: StubCliVideoService(local_discovery(source)),
    )

    result = CliRunner().invoke(cli.app, ["inspect", str(source)])

    assert result.exit_code == 0, result.output
    output = flat(result.stdout)
    assert "File Metadata" in output
    assert "lecture 3" in output
    assert "Manual Subtitles" not in output
    assert "'captionforge transcribe' transcribes its audio locally" in output


def test_extract_sends_a_file_to_transcribe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = media_file(tmp_path)

    class StubVideoService(StubCliVideoService):
        def __init__(self, *_: Any) -> None:
            super().__init__(local_discovery(source))

    monkeypatch.setattr(cli, "VideoService", StubVideoService)

    result = CliRunner().invoke(cli.app, ["extract", str(source)])

    assert result.exit_code == 1
    assert "no caption track to export" in flat(result.stderr)
    assert "Run 'captionforge transcribe'" in flat(result.stderr)


def test_a_missing_file_is_invalid_input(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["inspect", str(tmp_path / "gone.mp4")])
    assert result.exit_code == 2
    assert "No file was found" in flat(result.stderr)


def test_help_says_files_work() -> None:
    for command in ("transcribe", "inspect", "prepare-audio"):
        result = CliRunner().invoke(cli.app, [command, "--help"])
        assert "audio file on this computer" in flat(result.stdout), command
