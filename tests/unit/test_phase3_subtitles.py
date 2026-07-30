"""Offline parsing, cleanup, and caption-download tests."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from app.adapters.ytdlp_adapter import Extractor, YtDlpAdapter
from app.models.subtitle import RawSubtitle, SubtitleSourceType
from app.services.subtitle_service import SubtitleService
from tests.conftest import VIDEO_ID, FakeExtractor, make_track


def test_caption_download_uses_ytdlp_without_media(tmp_path: Path) -> None:
    """The adapter should request one subtitle while retaining skip_download."""
    captured: dict[str, Any] = {}

    class DownloadingExtractor(FakeExtractor):
        def extract_info(self, url: str, *, download: bool) -> dict[str, Any]:
            output = Path(captured["options"]["outtmpl"]).parent
            (output / f"{VIDEO_ID}.ar.vtt").write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nمرحبا\n",
                encoding="utf-8",
            )
            captured["download"] = download
            return {}

    def factory(options: dict[str, Any]) -> AbstractContextManager[Extractor]:
        captured["options"] = options
        return DownloadingExtractor()

    raw = YtDlpAdapter(factory).download_subtitle(VIDEO_ID, make_track("ar"))

    assert raw.content.endswith("مرحبا\n")
    assert captured["download"] is True
    assert captured["options"]["skip_download"] is True
    assert captured["options"]["writesubtitles"] is True
    assert captured["options"]["writeautomaticsub"] is False
    assert captured["options"]["subtitleslangs"] == ["ar"]


def test_automatic_caption_download_sets_automatic_option() -> None:
    """Automatic caption selection should map to yt-dlp's automatic flag."""
    captured: dict[str, Any] = {}

    def factory(options: dict[str, Any]) -> AbstractContextManager[Extractor]:
        captured.update(options)
        output = Path(options["outtmpl"]).parent
        extractor = FakeExtractor({})
        original = extractor.extract_info

        def extract_info(url: str, *, download: bool) -> dict[str, Any]:
            original(url, download=download)
            (output / f"{VIDEO_ID}.ar.vtt").write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nنص\n",
                encoding="utf-8",
            )
            return {}

        extractor.extract_info = extract_info  # type: ignore[method-assign]
        return extractor

    track = make_track("ar", SubtitleSourceType.AUTOMATIC)
    YtDlpAdapter(factory).download_subtitle(VIDEO_ID, track)

    assert captured["writeautomaticsub"] is True
    assert captured["writesubtitles"] is False


def test_vtt_parsing_cleanup_arabic_duplicates_and_timestamps() -> None:
    """Cleanup should preserve Arabic while removing markup and fixing overlap."""
    raw = RawSubtitle(
        format="vtt",
        content="""WEBVTT

00:00:00.000 --> 00:00:02.000
<c.color>مَرْحَبًا، بالعالم!</c>

00:00:01.500 --> 00:00:03.000
  مَرْحَبًا،   بالعالم! 

00:00:02.500 --> 00:00:02.000
هذا نص عربي؟

00:00:04.000 --> 00:00:05.000
[Music]
""",
    )

    segments = SubtitleService().parse_and_clean(raw, make_track("ar"))

    assert [item.text for item in segments] == [
        "مَرْحَبًا، بالعالم!",
        "هذا نص عربي؟",
    ]
    assert segments[0].text.encode("utf-8").decode("utf-8") == segments[0].text
    assert segments[1].end_seconds > segments[1].start_seconds
    assert segments[0].end_seconds <= segments[1].start_seconds


def test_srt_parsing_accepts_comma_timestamps() -> None:
    """SRT and VTT should share the common internal representation."""
    raw = RawSubtitle(
        format="srt",
        content="1\n00:00:01,250 --> 00:00:02,500\nHello\n",
    )

    segment = SubtitleService().parse_and_clean(raw, make_track("en"))[0]

    assert segment.index == 1
    assert segment.start_seconds == 1.25
    assert segment.end_seconds == 2.5


def test_json3_parsing() -> None:
    """yt-dlp JSON3 fallback data should remain parseable."""
    raw = RawSubtitle(
        format="json3",
        content='{"events":[{"tStartMs":500,"dDurationMs":1000,'
        '"segs":[{"utf8":"أهلاً"},{"utf8":"!"}]}]}',
    )

    segment = SubtitleService().parse_and_clean(raw, make_track("ar"))[0]

    assert segment.text == "أهلاً!"
    assert segment.start_seconds == 0.5
