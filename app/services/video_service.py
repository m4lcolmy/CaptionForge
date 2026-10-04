"""Application service coordinating YouTube and local-file inspection."""

from dataclasses import dataclass
from pathlib import Path

from app.adapters.ffmpeg_adapter import FFmpegAdapter
from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.config import Config
from app.core.exceptions import (
    LiveStreamNotSupportedError,
    NoAudioStreamError,
    VideoUnavailableError,
)
from app.core.logging_config import get_logger
from app.core.retry import retry_call
from app.models.media import MediaOptions
from app.models.subtitle import SubtitleDiscoveryResult
from app.models.video import VideoMetadata
from app.services.subtitle_service import SubtitleService
from app.utils.local_media import local_media_path, resolve_media_file
from app.utils.url_utils import extract_youtube_video_id


@dataclass(frozen=True)
class VideoInspection:
    """Everything one metadata request yields: captions and downloadable files."""

    discovery: SubtitleDiscoveryResult
    media: MediaOptions


class VideoService:
    """Validate URLs, enforce supported-video rules, and coordinate discovery."""

    def __init__(
        self,
        adapter: YtDlpAdapter,
        subtitle_service: SubtitleService,
        config: Config | None = None,
        ffmpeg: FFmpegAdapter | None = None,
    ) -> None:
        self._adapter = adapter
        self._subtitle_service = subtitle_service
        self._config = config or Config()
        self._ffmpeg = ffmpeg or FFmpegAdapter(self._config.ffmpeg_executable)

    def inspect(
        self,
        url: str,
        preferred_language: str,
        *,
        allow_translated: bool = False,
    ) -> SubtitleDiscoveryResult:
        """Inspect one non-live YouTube video, or a file, without downloading."""
        return self.inspect_all(
            url, preferred_language, allow_translated=allow_translated
        ).discovery

    def inspect_all(
        self,
        url: str,
        preferred_language: str,
        *,
        allow_translated: bool = False,
    ) -> VideoInspection:
        """Inspect once and keep both the caption tracks and the file qualities.

        One YouTube metadata request already describes every stream, so asking
        again to list downloads would only cost the person another wait.
        """
        log = get_logger()
        log.info("Inspecting input URL: {}", url)
        local = local_media_path(url)
        if local is not None:
            return self._inspect_file(local, url, preferred_language)
        video_id = extract_youtube_video_id(url)
        log.info("Normalized YouTube video ID: {}", video_id)
        inspection = retry_call(
            lambda: self._adapter.inspect(video_id, url),
            attempts=self._config.retry_count,
            delay_seconds=self._config.retry_delay_seconds,
            operation_name="youtube_metadata",
        )
        video = inspection.video
        if video.is_live or video.live_status in {"is_live", "is_upcoming"}:
            raise LiveStreamNotSupportedError(
                "This live stream is not supported in the current version."
            )
        if video.availability in {"private", "subscriber_only", "premium_only"}:
            raise VideoUnavailableError(
                "The video could not be accessed with its current availability."
            )
        discovery = self._subtitle_service.discover(
            video,
            inspection.manual_tracks,
            inspection.automatic_tracks,
            preferred_language,
            allow_translated=allow_translated,
        )
        return VideoInspection(discovery=discovery, media=inspection.media)

    def _inspect_file(
        self, path: Path, original: str, preferred_language: str
    ) -> VideoInspection:
        """Describe a file on this computer: no caption tracks, nothing to download.

        Its sound is the only thing CaptionForge can use, so a file without any
        is refused here, before anyone waits for a model to load.
        """
        source = resolve_media_file(path)
        probe = self._ffmpeg.probe(source)
        if not probe.has_audio:
            raise NoAudioStreamError("That file has no sound to transcribe.")
        get_logger().info(
            "Local file inspected path={} duration_seconds={}",
            source,
            probe.duration_seconds,
        )
        video = VideoMetadata(
            # The name the person knows it by: lecture.mp4 exports lecture.srt.
            title=source.stem,
            duration_seconds=(
                round(probe.duration_seconds) if probe.duration_seconds else None
            ),
            webpage_url=source.as_uri(),
            original_url=original.strip(),
            local_path=source,
        )
        discovery = self._subtitle_service.discover(video, (), (), preferred_language)
        return VideoInspection(discovery=discovery, media=MediaOptions())
