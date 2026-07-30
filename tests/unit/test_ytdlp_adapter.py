"""Offline tests for yt-dlp metadata mapping and error translation."""

from datetime import date

import pytest
from yt_dlp.utils import DownloadError

from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.exceptions import (
    MetadataRetrievalError,
    PrivateVideoError,
    VideoUnavailableError,
)
from tests.conftest import VIDEO_ID, VIDEO_URL, FakeExtractor, extractor_factory


def test_successful_metadata_and_manual_track_mapping(
    video_with_manual_arabic_subtitles: dict[str, object],
) -> None:
    """Metadata and formats should map without any download request."""
    extractor = FakeExtractor(video_with_manual_arabic_subtitles)
    adapter = YtDlpAdapter(extractor_factory(extractor))

    result = adapter.inspect(VIDEO_ID, VIDEO_URL)

    assert result.video.upload_date == date(2026, 7, 30)
    assert result.video.duration_seconds == 213
    assert result.manual_tracks[0].normalized_language_code == "ar"
    assert result.manual_tracks[0].available_formats == ("srt", "vtt")
    assert extractor.download is False


def test_automatic_track_mapping_and_malformed_entries(
    video_with_automatic_arabic_captions: dict[str, object],
) -> None:
    """Automatic tracks should be separate and malformed entries ignored."""
    video_with_automatic_arabic_captions["automatic_captions"] = {
        "ar_EG": [
            {"ext": "vtt"},
            {"ext": "VTT"},
            {"url": "missing-ext"},
            "malformed",
        ],
        "": [{"ext": "vtt"}],
    }
    adapter = YtDlpAdapter(
        extractor_factory(FakeExtractor(video_with_automatic_arabic_captions))
    )

    result = adapter.inspect(VIDEO_ID, VIDEO_URL)

    assert len(result.automatic_tracks) == 1
    track = result.automatic_tracks[0]
    assert track.normalized_language_code == "ar-EG"
    assert track.available_formats == ("vtt",)
    assert track.track_count == 3
    assert track.is_automatic is True


def test_missing_optional_metadata_is_safe(
    video_with_missing_optional_fields: dict[str, object],
) -> None:
    """Absent optional fields and malformed dates should not fail mapping."""
    video_with_missing_optional_fields["upload_date"] = "invalid"
    adapter = YtDlpAdapter(
        extractor_factory(FakeExtractor(video_with_missing_optional_fields))
    )

    video = adapter.inspect(VIDEO_ID, VIDEO_URL).video

    assert video.channel_name is None
    assert video.duration_seconds is None
    assert video.upload_date is None


@pytest.mark.parametrize(
    ("message", "exception_type"),
    [
        ("ERROR: Private video", PrivateVideoError),
        ("ERROR: Video unavailable", VideoUnavailableError),
        ("ERROR: extractor failed", MetadataRetrievalError),
    ],
)
def test_download_errors_are_translated(
    message: str, exception_type: type[Exception]
) -> None:
    """Raw yt-dlp errors should never cross the adapter boundary."""
    adapter = YtDlpAdapter(
        extractor_factory(FakeExtractor(error=DownloadError(message)))
    )

    with pytest.raises(exception_type):
        adapter.inspect(VIDEO_ID, VIDEO_URL)


def test_generic_extractor_errors_are_translated() -> None:
    """Unexpected extractor failures should become metadata errors."""
    adapter = YtDlpAdapter(extractor_factory(FakeExtractor(error=RuntimeError("boom"))))

    with pytest.raises(MetadataRetrievalError):
        adapter.inspect(VIDEO_ID, VIDEO_URL)
