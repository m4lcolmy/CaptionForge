"""Subtitle timestamp parsing and formatting helpers."""

import re

_TIMESTAMP = re.compile(
    r"^\s*(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):"
    r"(?P<seconds>\d{1,2})(?:[.,](?P<millis>\d{1,3}))?\s*$"
)


def parse_timestamp(value: str) -> float:
    """Parse an SRT/VTT timestamp into seconds."""
    match = _TIMESTAMP.match(value)
    if not match:
        raise ValueError(f"Invalid subtitle timestamp: {value}")
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    millis = (match.group("millis") or "0").ljust(3, "0")
    if minutes > 59 or seconds > 59:
        raise ValueError(f"Invalid subtitle timestamp: {value}")
    return hours * 3600 + minutes * 60 + seconds + int(millis) / 1000


def format_timestamp(seconds: float, *, separator: str = ",") -> str:
    """Format seconds as an SRT/VTT timestamp."""
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{separator}{milliseconds:03d}"


def distribute_duration(
    start: float, end: float, weights: list[int]
) -> list[tuple[float, float]]:
    """Divide a time range proportionally while avoiding floating-point gaps."""
    if not weights:
        return []
    duration = max(0.001, end - start)
    total = max(1, sum(weights))
    boundaries = [start]
    consumed = 0
    for weight in weights[:-1]:
        consumed += weight
        boundaries.append(start + duration * consumed / total)
    boundaries.append(start + duration)
    return list(zip(boundaries, boundaries[1:], strict=False))
