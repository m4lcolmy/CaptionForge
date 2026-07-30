"""Pure YouTube URL parsing and validation utilities."""

import re
from urllib.parse import parse_qs, urlparse

from app.core.exceptions import InvalidYouTubeUrlError, UnsupportedYouTubeUrlError

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }
)
_UNSUPPORTED_PATH_PREFIXES = (
    "/playlist",
    "/channel",
    "/user",
    "/c/",
    "/@",
    "/results",
    "/feed",
)


def is_valid_video_id(value: str) -> bool:
    """Return whether a value is a canonical YouTube video identifier."""
    return bool(_VIDEO_ID_PATTERN.fullmatch(value.strip()))


def extract_youtube_video_id(url: str) -> str:
    """Validate an individual-video YouTube URL and return its video ID."""
    candidate = url.strip()
    if not candidate:
        raise InvalidYouTubeUrlError(
            "The provided URL is not a valid YouTube video URL."
        )
    if "://" not in candidate:
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or host not in _YOUTUBE_HOSTS:
        raise InvalidYouTubeUrlError(
            "The provided URL is not a valid YouTube video URL."
        )

    path = parsed.path.rstrip("/")
    query = parse_qs(parsed.query)
    video_id: str | None = None

    if host in {"youtu.be", "www.youtu.be"}:
        parts = [part for part in path.split("/") if part]
        video_id = parts[0] if len(parts) == 1 else None
    elif path == "/watch":
        values = query.get("v", [])
        video_id = values[0] if values else None
        if video_id is None and "list" in query:
            raise UnsupportedYouTubeUrlError(
                "This URL points to a playlist instead of an individual video."
            )
    elif path.startswith(("/shorts/", "/embed/", "/v/")):
        parts = [part for part in path.split("/") if part]
        video_id = parts[1] if len(parts) == 2 else None
    elif path.startswith(_UNSUPPORTED_PATH_PREFIXES):
        raise UnsupportedYouTubeUrlError(
            "This YouTube URL does not point to a supported individual video."
        )
    else:
        raise InvalidYouTubeUrlError(
            "The provided URL is not a valid YouTube video URL."
        )

    if video_id is None or not is_valid_video_id(video_id):
        raise InvalidYouTubeUrlError(
            "The provided URL contains an invalid YouTube video ID."
        )
    return video_id


def canonical_youtube_url(video_id: str) -> str:
    """Build the canonical webpage URL for a validated video ID."""
    if not is_valid_video_id(video_id):
        raise InvalidYouTubeUrlError("The YouTube video ID is invalid.")
    return f"https://www.youtube.com/watch?v={video_id}"
