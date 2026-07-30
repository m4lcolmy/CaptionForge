"""Subtitle selection, parsing, and conservative cleanup."""

import json
import re
from collections.abc import Sequence

from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.config import Config
from app.core.exceptions import SubtitleDiscoveryError, SubtitleParseError
from app.core.logging_config import get_logger
from app.core.retry import retry_call
from app.models.subtitle import (
    RawSubtitle,
    SubtitleDiscoveryResult,
    SubtitleSegment,
    SubtitleSourceType,
    SubtitleTrack,
)
from app.models.video import VideoMetadata
from app.utils.arabic_text_utils import clean_caption_text
from app.utils.language_utils import base_language, normalize_language_code
from app.utils.time_utils import parse_timestamp

_TIMING_LINE = re.compile(r"^\s*(?P<start>\S+)\s+-->\s+(?P<end>\S+)(?:\s+.*)?$")


class SubtitleService:
    """Apply deterministic preferred-language caption selection rules."""

    def __init__(self, config: Config | None = None) -> None:
        from app.services.postprocessing_service import PostProcessingService

        self._postprocessor = PostProcessingService(config)
        self._config = config or Config()

    def discover(
        self,
        video: VideoMetadata,
        manual_tracks: Sequence[SubtitleTrack],
        automatic_tracks: Sequence[SubtitleTrack],
        preferred_language: str,
    ) -> SubtitleDiscoveryResult:
        """Select the best track and return a stable discovery result."""
        preferred = normalize_language_code(preferred_language)
        if preferred is None:
            raise SubtitleDiscoveryError(
                f"The preferred language code '{preferred_language}' is invalid."
            )
        selected, match_type = self.select_track(
            manual_tracks, automatic_tracks, preferred
        )
        reason = self._selection_reason(selected, match_type)
        get_logger().info(
            "Subtitle selection: preferred_language={}; selected={}",
            preferred,
            selected.normalized_language_code if selected else "none",
        )
        return SubtitleDiscoveryResult(
            video=video,
            manual_tracks=tuple(manual_tracks),
            automatic_tracks=tuple(automatic_tracks),
            selected_track=selected,
            preferred_language=preferred,
            selection_reason=reason,
        )

    def select_track(
        self,
        manual_tracks: Sequence[SubtitleTrack],
        automatic_tracks: Sequence[SubtitleTrack],
        preferred_language: str,
    ) -> tuple[SubtitleTrack | None, str | None]:
        """Select using manual-exact, manual-base, auto-exact, auto-base priority."""
        preferred = normalize_language_code(preferred_language)
        if preferred is None:
            return None, None
        preferred_base = base_language(preferred)
        priorities = (
            (manual_tracks, True, "exact"),
            (manual_tracks, False, "base"),
            (automatic_tracks, True, "exact"),
            (automatic_tracks, False, "base"),
        )
        for tracks, exact, match_type in priorities:
            matches = [
                track
                for track in tracks
                if (
                    track.normalized_language_code == preferred
                    if exact
                    else base_language(track.normalized_language_code) == preferred_base
                )
            ]
            if matches:
                return (
                    min(
                        matches,
                        key=lambda track: (
                            track.normalized_language_code != preferred_base,
                            track.normalized_language_code,
                            track.language_code,
                        ),
                    ),
                    match_type,
                )
        return None, None

    def retrieve_and_parse(
        self,
        adapter: YtDlpAdapter,
        video_id: str,
        track: SubtitleTrack,
        *,
        postprocess: bool = True,
    ) -> tuple[SubtitleSegment, ...]:
        """Retrieve the selected raw track, then parse and clean it."""
        raw = retry_call(
            lambda: adapter.download_subtitle(video_id, track),
            attempts=self._config.retry_count,
            delay_seconds=self._config.retry_delay_seconds,
            operation_name="subtitle_download",
        )
        return self.parse_and_clean(raw, track, postprocess=postprocess)

    def parse_and_clean(
        self,
        raw: RawSubtitle,
        track: SubtitleTrack,
        *,
        postprocess: bool = True,
    ) -> tuple[SubtitleSegment, ...]:
        """Parse supported raw caption data into the common representation."""
        extension = raw.format.lower()
        if extension in {"vtt", "srt"}:
            candidates = self._parse_timed_text(raw.content)
        elif extension == "json3":
            candidates = self._parse_json3(raw.content)
        else:
            raise SubtitleParseError(
                f"Unsupported downloaded caption format: {extension}"
            )
        if postprocess:
            segments = self._postprocessor.process_candidates(
                candidates, track.normalized_language_code
            )
        else:
            segments = self._clean_segments(candidates, track.normalized_language_code)
        if not segments:
            raise SubtitleParseError(
                "The downloaded caption track contains no usable text."
            )
        return segments

    @staticmethod
    def _parse_timed_text(content: str) -> list[tuple[float, float, str]]:
        lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segments: list[tuple[float, float, str]] = []
        index = 0
        while index < len(lines):
            match = _TIMING_LINE.match(lines[index])
            if not match:
                index += 1
                continue
            try:
                start = parse_timestamp(match.group("start"))
                end = parse_timestamp(match.group("end"))
            except ValueError as exc:
                raise SubtitleParseError(
                    "A downloaded caption timestamp is invalid.", details=str(exc)
                ) from exc
            index += 1
            text_lines: list[str] = []
            while index < len(lines) and lines[index].strip():
                text_lines.append(lines[index])
                index += 1
            segments.append((start, end, "\n".join(text_lines)))
        if not segments:
            raise SubtitleParseError(
                "The downloaded caption track contains no segments."
            )
        return segments

    @staticmethod
    def _parse_json3(content: str) -> list[tuple[float, float, str]]:
        try:
            payload = json.loads(content)
            events = payload["events"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SubtitleParseError(
                "The downloaded JSON caption track is invalid.", details=str(exc)
            ) from exc
        segments = []
        for event in events:
            if not isinstance(event, dict) or "segs" not in event:
                continue
            start = float(event.get("tStartMs", 0)) / 1000
            duration = float(event.get("dDurationMs", 0)) / 1000
            text = "".join(
                str(part.get("utf8", ""))
                for part in event["segs"]
                if isinstance(part, dict)
            )
            segments.append((start, start + duration, text))
        if not segments:
            raise SubtitleParseError(
                "The downloaded caption track contains no segments."
            )
        return segments

    @staticmethod
    def _clean_segments(
        candidates: Sequence[tuple[float, float, str]], language: str
    ) -> tuple[SubtitleSegment, ...]:
        cleaned: list[SubtitleSegment] = []
        previous_text: str | None = None
        for start, end, raw_text in candidates:
            text = clean_caption_text(raw_text, fix_punctuation=False)
            if not text or text == previous_text:
                continue
            start = max(0.0, start)
            if cleaned and start < cleaned[-1].end_seconds:
                previous = cleaned[-1]
                if start > previous.start_seconds:
                    cleaned[-1] = previous.model_copy(update={"end_seconds": start})
                else:
                    start = previous.end_seconds
            end = max(end, start + 0.001)
            cleaned.append(
                SubtitleSegment(
                    index=len(cleaned) + 1,
                    start_seconds=start,
                    end_seconds=end,
                    text=text,
                    language=language,
                )
            )
            previous_text = text
        if not cleaned:
            raise SubtitleParseError(
                "The downloaded caption track contains no usable text."
            )
        return tuple(cleaned)

    @staticmethod
    def _selection_reason(
        track: SubtitleTrack | None, match_type: str | None
    ) -> str | None:
        if track is None or match_type is None:
            return None
        source = (
            "manual" if track.source_type is SubtitleSourceType.MANUAL else "automatic"
        )
        name = track.language_name or track.normalized_language_code
        return f"Selected {source} {name} caption using a {match_type} match."
