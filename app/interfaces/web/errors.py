"""Translate CaptionForge errors into HTTP responses without leaking internals."""

from http import HTTPStatus

from app.core.exceptions import (
    AudioDownloadError,
    CaptionForgeError,
    ConfigurationError,
    DocxNotInstalledError,
    FFmpegNotFoundError,
    InsufficientDiskSpaceError,
    InvalidYouTubeUrlError,
    LiveStreamNotSupportedError,
    MediaDownloadError,
    MediaFormatUnavailableError,
    MetadataRetrievalError,
    SubtitleDownloadError,
    UnsupportedModelError,
    UnsupportedYouTubeUrlError,
    ValidationError,
    VideoUnavailableError,
    WhisperNotInstalledError,
)

GENERIC_MESSAGE = (
    "CaptionForge could not complete the request. See the log for technical details."
)

_STATUS_BY_ERROR: tuple[tuple[type[CaptionForgeError], int], ...] = (
    (InvalidYouTubeUrlError, HTTPStatus.BAD_REQUEST),
    (UnsupportedYouTubeUrlError, HTTPStatus.BAD_REQUEST),
    (LiveStreamNotSupportedError, HTTPStatus.BAD_REQUEST),
    (UnsupportedModelError, HTTPStatus.BAD_REQUEST),
    # More specific than MediaDownloadError below it, so it has to come first.
    (MediaFormatUnavailableError, HTTPStatus.BAD_REQUEST),
    (ValidationError, HTTPStatus.BAD_REQUEST),
    (ConfigurationError, HTTPStatus.BAD_REQUEST),
    (VideoUnavailableError, HTTPStatus.NOT_FOUND),
    (InsufficientDiskSpaceError, HTTPStatus.INSUFFICIENT_STORAGE),
    (WhisperNotInstalledError, HTTPStatus.SERVICE_UNAVAILABLE),
    (DocxNotInstalledError, HTTPStatus.SERVICE_UNAVAILABLE),
    (FFmpegNotFoundError, HTTPStatus.SERVICE_UNAVAILABLE),
    (MetadataRetrievalError, HTTPStatus.BAD_GATEWAY),
    (SubtitleDownloadError, HTTPStatus.BAD_GATEWAY),
    (AudioDownloadError, HTTPStatus.BAD_GATEWAY),
    (MediaDownloadError, HTTPStatus.BAD_GATEWAY),
)


def status_for(error: CaptionForgeError) -> int:
    """Map an application error to a stable HTTP status code."""
    for error_type, status in _STATUS_BY_ERROR:
        if isinstance(error, error_type):
            return int(status)
    return int(HTTPStatus.INTERNAL_SERVER_ERROR)


def body_for(error: CaptionForgeError) -> dict[str, object]:
    """Build the response body, carrying the short message and never the cause."""
    return {
        "error": error.message,
        "code": type(error).__name__,
        "retryable": error.retryable,
    }
