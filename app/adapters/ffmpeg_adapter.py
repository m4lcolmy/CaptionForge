"""FFmpeg adapter for transcription-friendly audio conversion."""

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from app.core.exceptions import AudioConversionError, FFmpegNotFoundError

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


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
                "FFmpeg could not convert the downloaded audio."
                if conversion
                else "FFmpeg is installed but could not be executed."
            )
            raise AudioConversionError(message, details=exc.stderr) from exc
        except OSError as exc:
            raise AudioConversionError(
                "FFmpeg could not be started.", details=str(exc)
            ) from exc
