"""Audio fallback orchestration for Phase 4."""

import shutil
from collections.abc import Callable
from pathlib import Path

from app.adapters.ffmpeg_adapter import FFmpegAdapter
from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.config import Config
from app.core.exceptions import (
    CaptionForgeError,
    CleanupError,
    ProcessingInterruptedError,
    SubtitleDiscoveryError,
    TemporaryDirectoryError,
)
from app.core.retry import retry_call
from app.models.job import Job, JobStatus
from app.models.subtitle import SubtitleDiscoveryResult
from app.services.video_service import VideoService
from app.utils.file_utils import cleanup_path, create_job_directory, ensure_disk_space

ProgressCallback = Callable[[str, float | None], None]


class AudioService:
    """Prepare audio only when caption selection cannot satisfy the request."""

    def __init__(
        self,
        video_service: VideoService,
        ytdlp: YtDlpAdapter,
        ffmpeg: FFmpegAdapter,
        config: Config,
    ) -> None:
        self._video_service = video_service
        self._ytdlp = ytdlp
        self._ffmpeg = ffmpeg
        self._config = config

    def prepare(
        self,
        url: str,
        language: str,
        *,
        temporary_directory: Path | None = None,
        keep_temp: bool | None = None,
        force: bool = False,
        progress: ProgressCallback | None = None,
        discovery: SubtitleDiscoveryResult | None = None,
    ) -> Path:
        """Inspect, download audio only, convert it, and return the WAV path."""
        notify = progress or (lambda _message, _percent: None)
        notify("Validating video and checking captions", None)
        discovery = discovery or self._video_service.inspect(url, language)
        if discovery.selected_track is not None and not force:
            raise SubtitleDiscoveryError(
                f"Matching '{discovery.preferred_language}' captions already exist; "
                "audio preparation was skipped. Use --force to override."
            )

        base = temporary_directory or self._config.temp_directory
        preserve = self._config.keep_temp_files if keep_temp is None else keep_temp
        job = Job(source_url=url, language=language)
        job.transition(JobStatus.RUNNING, stage="inspection")
        job_directory: Path | None = None
        try:
            created_directory = create_job_directory(base, job.id)
            job_directory = created_directory
            job.temporary_directory = job_directory
            # Conservative floor covering compressed input plus PCM output.
            duration = discovery.video.duration_seconds or 600
            ensure_disk_space(
                job_directory,
                max(self._config.minimum_free_disk_bytes, duration * 64000),
            )
            job.transition(JobStatus.PREPARING_AUDIO, stage="audio_download")
            source = retry_call(
                lambda: self._ytdlp.download_audio(
                    discovery.video.video_id, created_directory, notify
                ),
                attempts=self._config.retry_count,
                delay_seconds=self._config.retry_delay_seconds,
                operation_name="audio_download",
            )
            job.downloaded_audio_path = source
            notify("Converting audio to mono 16 kHz PCM WAV", None)
            inside_output = job_directory / f"prepared.{self._config.audio_format}"
            self._ffmpeg.convert(
                source,
                inside_output,
                sample_rate=self._config.audio_sample_rate,
                channels=self._config.audio_channels,
                audio_format=self._config.audio_format,
            )
            if preserve:
                final = inside_output
            else:
                final = base / f"captionforge-{job.id}.{self._config.audio_format}"
                shutil.move(str(inside_output), final)
                cleanup_path(job_directory)
                job_directory = None
            job.prepared_audio_path = final
            job.transition(JobStatus.COMPLETED)
            notify("Audio preparation complete", 100.0)
            return final.resolve()
        except KeyboardInterrupt as exc:
            job.transition(JobStatus.CANCELLED)
            job.error_message = "Interrupted"
            self._cleanup_after_failure(job_directory, preserve)
            raise ProcessingInterruptedError(
                "Audio preparation was interrupted; temporary files were cleaned up."
            ) from exc
        except CaptionForgeError as exc:
            job.transition(JobStatus.FAILED)
            job.error_message = exc.message
            self._cleanup_after_failure(job_directory, preserve)
            raise
        except OSError as exc:
            job.transition(JobStatus.FAILED)
            self._cleanup_after_failure(job_directory, preserve)
            raise TemporaryDirectoryError(
                "The audio workspace could not be accessed.", details=str(exc)
            ) from exc

    @staticmethod
    def _cleanup_after_failure(path: Path | None, preserve: bool) -> None:
        if path is not None and not preserve:
            try:
                cleanup_path(path)
            except CleanupError as cleanup_error:
                raise CleanupError(
                    "Audio preparation failed and temporary cleanup also failed.",
                    details=cleanup_error.details,
                ) from cleanup_error
