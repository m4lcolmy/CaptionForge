"""FFmpeg adapter for transcription-friendly audio conversion."""

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import (
    AudioConversionError,
    FFmpegNotFoundError,
    UnreadableMediaFileError,
)

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]

# Reading a file's header is quick; anything this slow is a stalled mount.
PROBE_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class MediaProbe:
    """What FFprobe found in a file: how long it runs, and whether it has sound."""

    duration_seconds: float | None
    has_audio: bool


class FFmpegAdapter:
    """Construct and execute safe, non-shell FFmpeg commands."""

    def __init__(
        self, executable: Path | str = "ffmpeg", runner: ProcessRunner | None = None
    ) -> None:
        self.executable = str(executable)
        self._runner = runner or subprocess.run

    def version(self) -> str:
        """Return the first FFmpeg version line."""
        result = self._execute([self.executable, "-version"], conversion=False)
        return (result.stdout or "").splitlines()[0] or "Unknown version"

    @property
    def probe_executable(self) -> str:
        """FFprobe ships beside FFmpeg, so a configured FFmpeg path locates it."""
        path = Path(self.executable)
        name = path.name.replace("ffmpeg", "ffprobe")
        if name == path.name:
            name = "ffprobe"
        if path.parent == Path("."):
            return name
        return str(path.with_name(name))

    def probe(self, source: Path) -> MediaProbe:
        """Read a local file's duration and whether it carries any sound."""
        command = [
            self.probe_executable,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(source),
        ]
        unreadable = "CaptionForge cannot read that file as video or audio."
        try:
            result = self._runner(
                command,
                check=True,
                capture_output=True,
                # Tags are UTF-8 whatever the system locale says, and a stray
                # byte in one is no reason to refuse the whole file.
                encoding="utf-8",
                errors="replace",
                timeout=PROBE_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise FFmpegNotFoundError(
                "FFprobe, which comes with FFmpeg, is not installed or is not "
                "beside the configured FFmpeg."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise UnreadableMediaFileError(unreadable, details=exc.stderr) from exc
        except subprocess.TimeoutExpired as exc:
            raise UnreadableMediaFileError(
                "FFprobe took too long to read that file.", details=str(exc)
            ) from exc
        except OSError as exc:
            raise UnreadableMediaFileError(
                "FFprobe could not be started.", details=str(exc)
            ) from exc
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise UnreadableMediaFileError(unreadable, details=str(exc)) from exc
        if not isinstance(payload, dict):
            raise UnreadableMediaFileError(unreadable, details="unexpected output")
        streams = [
            item for item in payload.get("streams") or () if isinstance(item, dict)
        ]
        container = payload.get("format")
        duration = (
            _seconds(container.get("duration")) if isinstance(container, dict) else None
        )
        if duration is None:
            # Some containers only time their streams.
            timed = [_seconds(item.get("duration")) for item in streams]
            duration = max(
                (value for value in timed if value is not None), default=None
            )
        return MediaProbe(
            duration_seconds=duration,
            has_audio=any(item.get("codec_type") == "audio" for item in streams),
        )

    def build_conversion_command(
        self,
        source: Path,
        destination: Path,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        audio_format: str = "wav",
    ) -> list[str]:
        """Build a PCM WAV conversion command."""
        codec = "pcm_s16le" if audio_format.lower() == "wav" else "pcm_s16le"
        return [
            self.executable,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            codec,
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(destination),
        ]

    def convert(
        self,
        source: Path,
        destination: Path,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        audio_format: str = "wav",
    ) -> Path:
        """Convert audio and verify that FFmpeg produced a non-empty file."""
        command = self.build_conversion_command(
            source,
            destination,
            sample_rate=sample_rate,
            channels=channels,
            audio_format=audio_format,
        )
        self._execute(command, conversion=True)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise AudioConversionError(
                "FFmpeg completed but did not produce a usable audio file."
            )
        return destination

    def _execute(
        self, command: Sequence[str], *, conversion: bool
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._runner(
                list(command),
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise FFmpegNotFoundError(
                "FFmpeg is not installed or the configured executable path is invalid."
            ) from exc
        except subprocess.CalledProcessError as exc:
            message = (
                "FFmpeg could not convert the audio."
                if conversion
                else "FFmpeg is installed but could not be executed."
            )
            raise AudioConversionError(message, details=exc.stderr) from exc
        except OSError as exc:
            raise AudioConversionError(
                "FFmpeg could not be started.", details=str(exc)
            ) from exc


def _seconds(value: Any) -> float | None:
    """Read FFprobe's decimal-string durations, ignoring "N/A" and nonsense."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 and seconds != float("inf") else None
