"""Application service coordinating YouTube inspection."""

from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.config import Config
from app.core.exceptions import LiveStreamNotSupportedError, VideoUnavailableError
from app.core.logging_config import get_logger
from app.core.retry import retry_call
from app.models.subtitle import SubtitleDiscoveryResult
from app.services.subtitle_service import SubtitleService
from app.utils.url_utils import extract_youtube_video_id


class VideoService:
    """Validate URLs, enforce supported-video rules, and coordinate discovery."""

    def __init__(
        self,
        adapter: YtDlpAdapter,
        subtitle_service: SubtitleService,
        config: Config | None = None,
    ) -> None:
        self._adapter = adapter
        self._subtitle_service = subtitle_service
        self._config = config or Config()

    def inspect(self, url: str, preferred_language: str) -> SubtitleDiscoveryResult:
        """Inspect one non-live YouTube video without downloading content."""
        log = get_logger()
        log.info("Inspecting input URL: {}", url)
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
        return self._subtitle_service.discover(
            video,
            inspection.manual_tracks,
            inspection.automatic_tracks,
            preferred_language,
        )
