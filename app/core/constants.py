"""Application-wide immutable constants."""

from typing import Final

APP_NAME: Final[str] = "CaptionForge"
VERSION: Final[str] = "0.9.0"
ENV_PREFIX: Final[str] = "CAPTIONFORGE_"
ENV_FILE: Final[str] = ".env"
LOG_FILE_NAME: Final[str] = "captionforge_{time:YYYY-MM-DD}.log"
LOG_ROTATION: Final[str] = "10 MB"
LOG_RETENTION: Final[str] = "14 days"
SUPPORTED_OUTPUT_FORMATS: Final[frozenset[str]] = frozenset(
    {"srt", "vtt", "txt", "json", "docx"}
)
# "whisper" runs on this computer; "deepgram" sends the audio to Deepgram.
TRANSCRIPTION_ENGINES: Final[frozenset[str]] = frozenset({"whisper", "deepgram"})
DEEPGRAM_API_URL: Final[str] = "https://api.deepgram.com/v1"
# Mono Opus at this rate is ample for speech recognition, and about a fifth of
# the 16 kHz PCM WAV that Whisper reads: roughly 21 MB an hour to upload.
DEEPGRAM_UPLOAD_FORMAT: Final[str] = "ogg"
DEEPGRAM_UPLOAD_BITRATE_KBPS: Final[int] = 48
# Heights offered for an MP4 download, highest first. Only the ones a video
# actually publishes are ever shown, so this is a filter and not a promise.
MEDIA_VIDEO_HEIGHTS: Final[tuple[int, ...]] = (2160, 1440, 1080, 720, 480, 360)
MEDIA_AUDIO_KEY: Final[str] = "audio"
MEDIA_AUDIO_BITRATE_KBPS: Final[int] = 192
MEDIA_VIDEO_EXTENSION: Final[str] = "mp4"
MEDIA_AUDIO_EXTENSION: Final[str] = "mp3"
# Extensions that mark typed input as a file on this computer rather than a
# link, and that the desktop file chooser lists. FFmpeg reads far more than
# this; any other file can still be chosen or pasted by its full path.
LOCAL_MEDIA_EXTENSIONS: Final[tuple[str, ...]] = (
    "mp4", "mkv", "mov", "webm", "avi", "m4v", "wmv", "flv", "ts", "mts",
    "mpg", "mpeg", "3gp",
    "mp3", "m4a", "aac", "wav", "flac", "ogg", "oga", "opus", "wma", "aif",
    "aiff", "amr", "mka",
)  # fmt: skip


class ExitCode:
    """Process exit codes used by the command-line interface."""

    SUCCESS: Final[int] = 0
    FAILURE: Final[int] = 1
    INVALID_INPUT: Final[int] = 2
    VIDEO_UNAVAILABLE: Final[int] = 3
    METADATA_FAILURE: Final[int] = 4
