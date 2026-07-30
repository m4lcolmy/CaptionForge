"""Tests for domain data models."""

import pytest
from pydantic import ValidationError

from app.models.subtitle import SubtitleSegment


def test_subtitle_segment_accepts_valid_timing() -> None:
    """A segment ending after it starts should validate."""
    segment = SubtitleSegment(
        index=1,
        start_seconds=0,
        end_seconds=1.5,
        text="مرحبا",
        language="ar",
    )

    assert segment.text == "مرحبا"


def test_subtitle_segment_rejects_invalid_timing() -> None:
    """A segment cannot end before it starts."""
    with pytest.raises(ValidationError):
        SubtitleSegment(
            index=1,
            start_seconds=2,
            end_seconds=1,
            text="Invalid",
            language="en",
        )
