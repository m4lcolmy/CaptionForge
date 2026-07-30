"""Tests for YouTube URL parsing."""

import pytest

from app.core.exceptions import InvalidYouTubeUrlError, UnsupportedYouTubeUrlError
from app.utils.url_utils import extract_youtube_video_id
from tests.conftest import VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/watch?v={VIDEO_ID}",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?t=30&utm_source=test",
        f"https://youtube.com/watch?list=PL123&v={VIDEO_ID}&t=1",
    ],
)
def test_supported_video_urls(url: str) -> None:
    """Supported URL forms should resolve to the same canonical video ID."""
    assert extract_youtube_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com/playlist?list=PL123",
        "https://youtube.com/channel/UC123",
        "https://youtube.com/@example",
        "https://youtube.com/results?search_query=test",
    ],
)
def test_unsupported_youtube_resources(url: str) -> None:
    """Known non-video YouTube resources should be rejected explicitly."""
    with pytest.raises(UnsupportedYouTubeUrlError):
        extract_youtube_video_id(url)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "https://example.com/watch?v=qJFbKl6RjLU",
        "https://youtube.com/watch?v=short",
        "https://youtu.be/qJFbKl6RjLU/extra",
        "ftp://youtube.com/watch?v=qJFbKl6RjLU",
    ],
)
def test_invalid_urls(url: str) -> None:
    """Malformed and unrelated URLs should fail before extraction."""
    with pytest.raises(InvalidYouTubeUrlError):
        extract_youtube_video_id(url)


def test_playlist_only_watch_url() -> None:
    """A watch URL containing no video ID is playlist-only."""
    with pytest.raises(UnsupportedYouTubeUrlError):
        extract_youtube_video_id("https://youtube.com/watch?list=PL123")
