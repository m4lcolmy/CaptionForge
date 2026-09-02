"""Offline tests for yt-dlp metadata mapping and error translation."""

from datetime import date
from pathlib import Path

import pytest
from yt_dlp.utils import DownloadError

from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.exceptions import (
    AudioDownloadError,
    AudioStreamForbiddenError,
    MetadataRetrievalError,
    PrivateVideoError,
    SubtitleStreamForbiddenError,
    VideoUnavailableError,
)
from app.models.subtitle import SubtitleSourceType, SubtitleTrack
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


class _FailingDownloader:
    """Extractor that fails the download step with a given yt-dlp error."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __call__(self, _options: dict[str, object]) -> "_FailingDownloader":
        return self

    def __enter__(self) -> "_FailingDownloader":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def extract_info(self, url: str, *, download: bool) -> dict[str, object]:
        raise DownloadError(self.message)


@pytest.mark.parametrize(
    "message",
    [
        "ERROR: unable to download video data: HTTP Error 403: Forbidden",
        "ERROR: The page needs to be reloaded",
        "ERROR: Sign in to confirm you're not a bot",
        "ERROR: nsig extraction failed",
    ],
)
def test_stale_extractor_audio_failures_are_not_retried(
    message: str, tmp_path: Path
) -> None:
    """A refused stream needs an upgrade, so retrying must not be advertised."""
    adapter = YtDlpAdapter(_FailingDownloader(message))

    with pytest.raises(AudioStreamForbiddenError) as caught:
        adapter.download_audio(VIDEO_ID, tmp_path)

    assert caught.value.retryable is False
    assert "yt-dlp" in caught.value.message


def test_stale_extractor_caption_failures_are_not_retried() -> None:
    """The same refusal on a caption track gets the same actionable message."""
    adapter = YtDlpAdapter(_FailingDownloader("HTTP Error 403: Forbidden"))
    track = SubtitleTrack(
        language_code="ar",
        normalized_language_code="ar",
        source_type=SubtitleSourceType.AUTOMATIC,
        is_automatic=True,
    )

    with pytest.raises(SubtitleStreamForbiddenError) as caught:
        adapter.download_subtitle(VIDEO_ID, track)

    assert caught.value.retryable is False


def test_ordinary_audio_failures_stay_retryable(tmp_path: Path) -> None:
    """Genuine transient failures must keep their retry behaviour."""
    adapter = YtDlpAdapter(_FailingDownloader("ERROR: connection reset by peer"))

    with pytest.raises(AudioDownloadError) as caught:
        adapter.download_audio(VIDEO_ID, tmp_path)

    assert not isinstance(caught.value, AudioStreamForbiddenError)
    assert caught.value.retryable is True
