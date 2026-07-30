"""SubRip subtitle exporter."""

from collections.abc import Sequence

from app.models.subtitle import SubtitleSegment
from app.utils.time_utils import format_timestamp


def render_srt(segments: Sequence[SubtitleSegment]) -> str:
    """Render segments as UTF-8-ready SRT text."""
    blocks = []
    for index, segment in enumerate(segments, 1):
        start = format_timestamp(segment.start_seconds)
        end = format_timestamp(segment.end_seconds)
        blocks.append(f"{index}\n{start} --> {end}\n{segment.text}")
    return "\n\n".join(blocks) + "\n"
