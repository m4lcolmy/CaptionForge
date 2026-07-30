"""Structured JSON subtitle exporter."""

import json
from collections.abc import Sequence

from app.models.subtitle import SubtitleSegment, SubtitleTrack
from app.models.video import VideoMetadata


def render_json(
    video: VideoMetadata,
    track: SubtitleTrack,
    segments: Sequence[SubtitleSegment],
) -> str:
    """Render metadata, selection, source type, and segments as JSON."""
    payload = {
        "video": video.model_dump(mode="json"),
        "selected_language": track.normalized_language_code,
        "caption_source_type": track.source_type.value,
        "segments": [segment.model_dump(mode="json") for segment in segments],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
