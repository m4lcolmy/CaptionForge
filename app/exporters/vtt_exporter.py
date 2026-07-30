"""WebVTT subtitle exporter."""

from collections.abc import Sequence

from app.models.subtitle import SubtitleSegment
from app.utils.time_utils import format_timestamp


def render_vtt(segments: Sequence[SubtitleSegment]) -> str:
    """Render segments as UTF-8-ready WebVTT text."""
    blocks = ["WEBVTT"]
    for segment in segments:
        start = format_timestamp(segment.start_seconds, separator=".")
        end = format_timestamp(segment.end_seconds, separator=".")
        blocks.append(f"{start} --> {end}\n{segment.text}")
    return "\n\n".join(blocks) + "\n"
