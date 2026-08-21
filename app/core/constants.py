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


class ExitCode:
    """Process exit codes used by the command-line interface."""

    SUCCESS: Final[int] = 0
    FAILURE: Final[int] = 1
    INVALID_INPUT: Final[int] = 2
    VIDEO_UNAVAILABLE: Final[int] = 3
    METADATA_FAILURE: Final[int] = 4
