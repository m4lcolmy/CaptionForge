"""Guards for caption source quality: format choice, translations, fidelity."""

import json
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.config import Config
from app.models.subtitle import (
    RawSubtitle,
    SubtitleSourceType,
    SubtitleTrack,
)
from app.services.postprocessing_service import PostProcessingService
from app.services.subtitle_service import SubtitleService
from tests.conftest import VIDEO_ID, VIDEO_URL, make_track

# A YouTube rolling/karaoke auto-caption track: every cue repeats the previous
# line and appends the next one, tripling the word count.
ROLLING_VTT = """WEBVTT

00:00:05.200 --> 00:00:07.240
السلام عليكم اخواني

00:00:07.240 --> 00:00:10.120
السلام عليكم اخواني واخواتي حياكم الله

00:00:10.120 --> 00:00:10.130
واخواتي حياكم الله

00:00:10.130 --> 00:00:13.400
واخواتي حياكم الله باذن الله تعالى
"""

# The same track in json3: one cue per phrase, no repetition.
JSON3 = json.dumps(
    {
        "events": [
            {
                "tStartMs": 5200,
                "dDurationMs": 2040,
                "segs": [{"utf8": "السلام عليكم اخواني"}],
            },
            {
                "tStartMs": 7240,
                "dDurationMs": 2880,
                "segs": [{"utf8": "واخواتي حياكم الله"}],
            },
            {
                "tStartMs": 10120,
                "dDurationMs": 3280,
                "segs": [{"utf8": "باذن الله تعالى"}],
            },
        ]
    }
)

ARABIC_TRACK = SubtitleTrack(
    language_code="ar",
    normalized_language_code="ar",
    source_type=SubtitleSourceType.AUTOMATIC,
    is_automatic=True,
)


def rolling_join(texts: list[str]) -> list[str]:
    """Reconstruct the spoken word sequence from possibly overlapping cues."""
    words: list[str] = []
    for text in texts:
        incoming = text.split()
        overlap = 0
        for size in range(min(len(words), len(incoming)), 0, -1):
            if words[-size:] == incoming[:size]:
                overlap = size
                break
        words.extend(incoming[overlap:])
    return words


class RecordingDownloader(AbstractContextManager["RecordingDownloader"]):
    """Extractor that records options and writes the caption file yt-dlp would."""

    options: dict[str, Any] = {}

    def __init__(self, options: dict[str, Any]) -> None:
        RecordingDownloader.options = options

    def __enter__(self) -> "RecordingDownloader":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def extract_info(self, url: str, *, download: bool) -> Mapping[str, Any]:
        directory = Path(RecordingDownloader.options["outtmpl"]).parent
        (directory / f"{VIDEO_ID}.ar.json3").write_text(JSON3, encoding="utf-8")
        return {"id": VIDEO_ID}


def test_caption_download_prefers_json3_over_rolling_vtt() -> None:
    """json3 must be requested first; VTT auto-captions are rolling duplicates."""
    adapter = YtDlpAdapter(extractor_factory=RecordingDownloader)

    raw = adapter.download_subtitle(VIDEO_ID, ARABIC_TRACK)

    formats = RecordingDownloader.options["subtitlesformat"].split("/")
    assert formats[0] == "json3"
    assert formats.index("json3") < formats.index("vtt")
    assert raw.format == "json3"


def test_json3_needs_no_repair_while_rolling_vtt_does() -> None:
    """The two formats must agree on words; only VTT needs de-duplication."""
    service = SubtitleService()
    from_vtt = service.parse_and_clean(
        RawSubtitle(content=ROLLING_VTT, format="vtt"), ARABIC_TRACK
    )
    from_json3 = service.parse_and_clean(
        RawSubtitle(content=JSON3, format="json3"), ARABIC_TRACK
    )

    raw_vtt_cues = service.parse_and_clean(
        RawSubtitle(content=ROLLING_VTT, format="vtt"),
        ARABIC_TRACK,
        postprocess=False,
    )
    # VTT arrives inflated and must be repaired; json3 arrives already correct.
    assert len(raw_vtt_cues) > len(from_json3)
    assert len(from_json3) == 3
    assert rolling_join([item.text for item in from_vtt]) == rolling_join(
        [item.text for item in from_json3]
    )


def test_post_processing_preserves_every_spoken_word() -> None:
    """De-duplication may re-cut cues but must never drop speech."""
    service = SubtitleService()
    before = service.parse_and_clean(
        RawSubtitle(content=ROLLING_VTT, format="vtt"),
        ARABIC_TRACK,
        postprocess=False,
    )
    after = service.parse_and_clean(
        RawSubtitle(content=ROLLING_VTT, format="vtt"), ARABIC_TRACK
    )

    assert rolling_join([item.text for item in before]) == rolling_join(
        [item.text for item in after]
    )


def test_deliberate_repetition_survives_by_default() -> None:
    """Rhetorical repetition is speech, not an artifact, unless opted in."""
    candidates = [(0.0, 3.0, "نصره للشريعه نصره للشريعه هذه السلسله")]

    kept = PostProcessingService().process_candidates(candidates, "ar")
    collapsed = PostProcessingService(
        Config(collapse_repeated_phrases=True)
    ).process_candidates(candidates, "ar")

    assert kept[0].text.split()[:4] == ["نصره", "للشريعه", "نصره", "للشريعه"]
    assert collapsed[0].text.split()[:2] == ["نصره", "للشريعه"]
    assert "نصره للشريعه نصره" not in collapsed[0].text


def test_machine_translated_tracks_are_flagged(
    video_with_missing_optional_fields: dict[str, Any],
) -> None:
    """YouTube marks translated caption tracks with tlang= in the URL."""
    video_with_missing_optional_fields["automatic_captions"] = {
        "ar": [{"ext": "json3", "url": "https://youtube.com/api/timedtext?lang=ar"}],
        "tr": [
            {
                "ext": "json3",
                "url": "https://youtube.com/api/timedtext?lang=ar&tlang=tr",
            }
        ],
    }
    extractor = _StubExtractor(video_with_missing_optional_fields)

    inspection = YtDlpAdapter(lambda _options: extractor).inspect(VIDEO_ID, VIDEO_URL)

    by_code = {
        track.normalized_language_code: track for track in inspection.automatic_tracks
    }
    assert by_code["ar"].is_translated is False
    assert by_code["tr"].is_translated is True


def test_translated_tracks_rank_below_no_track_unless_allowed() -> None:
    """A translation of a transcription must not pre-empt local transcription."""
    translated = make_track("tr", SubtitleSourceType.AUTOMATIC).model_copy(
        update={"is_translated": True}
    )
    service = SubtitleService()

    excluded, _ = service.select_track([], [translated], "tr")
    included, _ = service.select_track([], [translated], "tr", allow_translated=True)

    assert excluded is None
    assert included == translated


def test_discovery_reports_skipped_translations(
    video_metadata: Any,
) -> None:
    """Callers need to know a translated track existed to explain the fallback."""
    translated = make_track("tr", SubtitleSourceType.AUTOMATIC).model_copy(
        update={"is_translated": True}
    )

    discovery = SubtitleService().discover(video_metadata, [], [translated], "tr")

    assert discovery.selected_track is None
    assert SubtitleService.translated_matches(discovery) == (translated,)


class _StubExtractor(AbstractContextManager["_StubExtractor"]):
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response

    def __enter__(self) -> "_StubExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def extract_info(self, url: str, *, download: bool) -> Mapping[str, Any]:
        return self.response
