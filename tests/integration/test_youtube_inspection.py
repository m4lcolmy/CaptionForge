"""Optional live metadata inspection test.

Run explicitly with ``pytest -m integration``. The chosen public URL can still
change upstream, so this supplements rather than replaces offline tests.
"""

import os

import pytest

from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.services.subtitle_service import SubtitleService
from app.services.video_service import VideoService


@pytest.mark.integration
def test_public_video_metadata_only() -> None:
    """Inspect an explicitly supplied public URL without downloading content."""
    url = os.getenv("CAPTIONFORGE_INTEGRATION_VIDEO_URL")
    if not url:
        pytest.skip("Set CAPTIONFORGE_INTEGRATION_VIDEO_URL to run this test")

    result = VideoService(YtDlpAdapter(), SubtitleService()).inspect(url, "ar")

    assert result.video.video_id
    assert result.video.title
