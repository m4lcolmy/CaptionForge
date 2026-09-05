"""Application-wide immutable constants."""

from typing import Final

APP_NAME: Final[str] = "CaptionForge"
VERSION: Final[str] = "0.7.0"
ENV_PREFIX: Final[str] = "CAPTIONFORGE_"
ENV_FILE: Final[str] = ".env"
LOG_FILE_NAME: Final[str] = "captionforge_{time:YYYY-MM-DD}.log"
LOG_ROTATION: Final[str] = "10 MB"
LOG_RETENTION: Final[str] = "14 days"
SUPPORTED_OUTPUT_FORMATS: Final[frozenset[str]] = frozenset(
    {"srt", "vtt", "txt", "json", "docx"}
)
# Heights offered for an MP4 download, highest first. Only the ones a video
# actually publishes are ever shown, so this is a filter and not a promise.
MEDIA_VIDEO_HEIGHTS: Final[tuple[int, ...]] = (2160, 1440, 1080, 720, 480, 360)
MEDIA_AUDIO_KEY: Final[str] = "audio"
MEDIA_AUDIO_BITRATE_KBPS: Final[int] = 192
MEDIA_VIDEO_EXTENSION: Final[str] = "mp4"
MEDIA_AUDIO_EXTENSION: Final[str] = "mp3"


class ExitCode:
    """Process exit codes used by the command-line interface."""

    SUCCESS: Final[int] = 0
    FAILURE: Final[int] = 1
    INVALID_INPUT: Final[int] = 2
    VIDEO_UNAVAILABLE: Final[int] = 3
    METADATA_FAILURE: Final[int] = 4
