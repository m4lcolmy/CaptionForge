"""Plain-text transcript exporter."""

from collections.abc import Sequence

from app.models.subtitle import SubtitleSegment
from app.utils.time_utils import format_timestamp


def render_txt(
    segments: Sequence[SubtitleSegment], *, timestamped: bool = False
) -> str:
    """Render a plain or timestamped transcript."""
    lines = []
    for segment in segments:
        prefix = (
            f"[{format_timestamp(segment.start_seconds, separator='.')}]\t"
            if timestamped
            else ""
        )
        lines.append(f"{prefix}{segment.text}")
    return "\n".join(lines) + "\n"
