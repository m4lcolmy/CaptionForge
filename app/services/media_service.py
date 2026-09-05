"""Whole-file downloads: an MP4 at a chosen quality, or an MP3 of the audio.

This is the one part of CaptionForge that fetches the video itself. Everything
else deliberately touches metadata, caption tracks, or an audio-only stream, so
downloads keep their own service, their own errors, and their own disk checks.
"""

import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.config import Config
from app.core.exceptions import (
    CaptionForgeError,
    CleanupError,
    MediaDownloadCancelledError,
    MediaFormatUnavailableError,
)
from app.core.logging_config import get_logger
from app.models.media import MediaDownloadResult, MediaOptions, MediaVariant
from app.models.video import VideoMetadata
from app.services.video_service import VideoService
from app.utils.file_utils import (
    available_stem,
    cleanup_path,
    create_job_directory,
    ensure_disk_space,
    ensure_output_directory,
    sanitize_filename,
)

ProgressCallback = Callable[[str, float | None], None]

# A quality's estimate comes from the stream's own bitrate, which understates
# a variable-bitrate file. Reserve half again as much before starting.
SIZE_SAFETY_FACTOR = 1.5


class MediaService:
    """List what a video offers to download, and write the chosen file."""

    def __init__(
        self,
        video_service: VideoService,
        ytdlp: YtDlpAdapter,
        config: Config,
    ) -> None:
        self._video_service = video_service
        self._ytdlp = ytdlp
        self._config = config

    def options(self, url: str) -> tuple[VideoMetadata, MediaOptions]:
        """Report the qualities this video actually publishes."""
        inspection = self._video_service.inspect_all(url, self._config.default_language)
        return inspection.discovery.video, inspection.media

    def download(
        self,
        url: str,
        quality: str,
        *,
        output_directory: Path | None = None,
        overwrite: bool = False,
        progress: ProgressCallback | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> MediaDownloadResult:
        """Download one variant into the output folder without replacing files."""
        notify = progress or (lambda _message, _percent: None)
        stop = cancelled or (lambda: False)
        log = get_logger()
        notify("Checking what this video offers", None)
        video, available = self.options(url)
        variant = available.resolve(quality)
        if variant is None:
            raise MediaFormatUnavailableError(
                f"'{quality}' is not something this video offers. "
                f"Available: {describe(available)}."
            )
        if stop():
            raise MediaDownloadCancelledError("The download was cancelled.")

        directory = ensure_output_directory(
            output_directory or self._config.default_output_folder
        )
        workspace = create_job_directory(self._config.temp_directory, uuid4())
        try:
            ensure_disk_space(
                workspace,
                max(
                    self._config.minimum_free_disk_bytes,
                    int((variant.estimated_bytes or 0) * SIZE_SAFETY_FACTOR),
                ),
            )
            log.info(
                "Downloading media video_id={} variant={}", video.video_id, variant.key
            )
            produced = self._ytdlp.download_media(
                video.video_id,
                workspace,
                variant,
                progress_callback=notify,
                cancelled=stop,
            )
            destination = self._destination(
                directory, video, produced, variant, overwrite=overwrite
            )
            ensure_disk_space(directory, produced.stat().st_size)
            # Across filesystems this is a copy, so the temporary file only
            # disappears once the finished one is completely written.
            shutil.move(str(produced), destination)
        except CaptionForgeError:
            self._discard(workspace)
            raise
        except OSError as exc:
            self._discard(workspace)
            raise MediaFormatUnavailableError(
                "The downloaded file could not be moved into the output folder.",
                details=str(exc),
            ) from exc
        self._discard(workspace)
        notify("Download complete", 100.0)
        return MediaDownloadResult(path=destination, variant=variant, video=video)

    def _destination(
        self,
        directory: Path,
        video: VideoMetadata,
        produced: Path,
        variant: MediaVariant,
        *,
        overwrite: bool,
    ) -> Path:
        """Name the file after the video, never over an existing download."""
        # Trust what FFmpeg produced over what was requested: a video with no
        # separate MP4 stream can come back as the container yt-dlp had.
        extension = produced.suffix.lower().lstrip(".") or variant.extension
        stem = sanitize_filename(video.title, fallback=video.video_id)
        suffix = "" if variant.is_audio else f" [{variant.label}]"
        stem = f"{stem}{suffix}"
        if not overwrite:
            stem = available_stem(directory, stem, (extension,))
        return directory / f"{stem}.{extension}"

    @staticmethod
    def _discard(workspace: Path) -> None:
        """Remove the workspace, and never fail a finished download over it."""
        try:
            cleanup_path(workspace)
        except CleanupError as exc:
            get_logger().warning(
                "Could not clean the download workspace technical_cause={}", exc.details
            )


def describe(options: MediaOptions) -> str:
    """List the offered qualities the way the CLI and errors should say them."""
    return ", ".join(variant.key for variant in options.variants) or "nothing"
