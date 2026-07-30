"""Reusable compact yt-dlp and domain fixtures."""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any

import pytest
from yt_dlp.utils import DownloadError

from app.adapters.ytdlp_adapter import Extractor
from app.models.subtitle import SubtitleSourceType, SubtitleTrack
from app.models.video import VideoMetadata

VIDEO_ID = "qJFbKl6RjLU"
VIDEO_URL = f"https://youtu.be/{VIDEO_ID}?si=wdoe8oQzasIgydBk"


class FakeExtractor(AbstractContextManager["FakeExtractor"]):
    """Configurable context-managed extractor used by adapter tests."""

    def __init__(
        self, response: Mapping[str, Any] | None = None, error: Exception | None = None
    ) -> None:
        self.response = response or {}
        self.error = error
        self.requested_url: str | None = None
        self.download: bool | None = None

    def __enter__(self) -> "FakeExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def extract_info(self, url: str, *, download: bool) -> Mapping[str, Any]:
        self.requested_url = url
        self.download = download
        if self.error:
            raise self.error
        return self.response


def extractor_factory(
    extractor: Extractor,
) -> Any:
    """Create a factory compatible with YtDlpAdapter dependency injection."""

    def factory(options: dict[str, Any]) -> AbstractContextManager[Extractor]:
        assert options["skip_download"] is True
        assert options["writesubtitles"] is False
        return extractor  # type: ignore[return-value]

    return factory


@pytest.fixture
def video_with_manual_arabic_subtitles() -> dict[str, Any]:
    """Compact metadata with manual Arabic and English subtitles."""
    return {
        "id": VIDEO_ID,
        "title": "Example video",
        "channel": "Example Channel",
        "channel_id": "channel-1",
        "duration": 213,
        "webpage_url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
        "upload_date": "20260730",
        "is_live": False,
        "live_status": "not_live",
        "availability": "public",
        "subtitles": {
            "ar": [{"ext": "vtt"}, {"ext": "srt"}, {"ext": "vtt"}],
            "en": [{"ext": "vtt"}],
        },
        "automatic_captions": {},
    }


@pytest.fixture
def video_with_automatic_arabic_captions() -> dict[str, Any]:
    """Compact metadata with automatic Arabic captions."""
    return {
        "id": VIDEO_ID,
        "title": "Automatic captions",
        "automatic_captions": {"ar": [{"ext": "json3"}, {"ext": "vtt"}]},
    }


@pytest.fixture
def video_with_arabic_variants() -> dict[str, Any]:
    """Compact metadata containing regional Arabic tracks."""
    return {
        "id": VIDEO_ID,
        "title": "Arabic variants",
        "subtitles": {"ar-SA": [{"ext": "vtt"}]},
        "automatic_captions": {"ar_EG": [{"ext": "vtt"}]},
    }


@pytest.fixture
def video_without_subtitles() -> dict[str, Any]:
    """Compact metadata without caption tracks."""
    return {"id": VIDEO_ID, "title": "No subtitles"}


@pytest.fixture
def video_with_only_english_subtitles() -> dict[str, Any]:
    """Compact metadata containing only English subtitles."""
    return {
        "id": VIDEO_ID,
        "title": "English only",
        "subtitles": {"en": [{"ext": "vtt"}]},
    }


@pytest.fixture
def live_video() -> dict[str, Any]:
    """Compact metadata for an active stream."""
    return {
        "id": VIDEO_ID,
        "title": "Live",
        "is_live": True,
        "live_status": "is_live",
    }


@pytest.fixture
def video_with_missing_optional_fields() -> dict[str, Any]:
    """Smallest successful metadata response."""
    return {"id": VIDEO_ID, "title": "Minimal"}


@pytest.fixture
def private_video_error() -> DownloadError:
    """Representative yt-dlp private-video failure."""
    return DownloadError("ERROR: Private video")


@pytest.fixture
def video_metadata() -> VideoMetadata:
    """Stable video model for service and CLI tests."""
    return VideoMetadata(
        video_id=VIDEO_ID,
        title="Example video",
        channel_name="Example Channel",
        duration_seconds=213,
        webpage_url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
        original_url=VIDEO_URL,
    )


def make_track(
    code: str, source: SubtitleSourceType = SubtitleSourceType.MANUAL
) -> SubtitleTrack:
    """Create a concise track for selection tests."""
    normalized = code.replace("_", "-")
    language, *region = normalized.split("-")
    normalized = "-".join([language.lower(), *[part.upper() for part in region]])
    return SubtitleTrack(
        language_code=code,
        normalized_language_code=normalized,
        language_name="Arabic" if language.lower() == "ar" else "English",
        source_type=source,
        is_automatic=source is SubtitleSourceType.AUTOMATIC,
        available_formats=("vtt",),
        track_count=1,
    )
