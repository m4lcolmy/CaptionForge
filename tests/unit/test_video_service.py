"""Tests for supported-video application rules."""

import pytest

from app.adapters.ytdlp_adapter import YtDlpInspection
from app.core.exceptions import LiveStreamNotSupportedError
from app.models.video import VideoMetadata
from app.services.subtitle_service import SubtitleService
from app.services.video_service import VideoService
from tests.conftest import VIDEO_ID, VIDEO_URL


class StubAdapter:
    """Adapter stub returning a prepared inspection."""

    def __init__(self, video: VideoMetadata) -> None:
        self.video = video

    def inspect(self, video_id: str, original_url: str) -> YtDlpInspection:
        assert video_id == VIDEO_ID
        assert original_url == VIDEO_URL
        return YtDlpInspection(self.video, (), ())


def test_live_video_is_rejected(video_metadata: VideoMetadata) -> None:
    """Active streams are outside Phase 2 scope."""
    live = video_metadata.model_copy(update={"is_live": True, "live_status": "is_live"})
    service = VideoService(StubAdapter(live), SubtitleService())  # type: ignore[arg-type]

    with pytest.raises(LiveStreamNotSupportedError):
        service.inspect(VIDEO_URL, "ar")
