"""CaptionForge exception hierarchy."""


class CaptionForgeError(Exception):
    """Base class for all expected CaptionForge errors."""

    def __init__(self, message: str, *, details: str | None = None) -> None:
        self.message = message
        self.details = details
        rendered = f"{message}: {details}" if details else message
        super().__init__(rendered)

    @property
    def retryable(self) -> bool:
        """Whether repeating the failed operation may reasonably succeed."""
        return False


class ConfigurationError(CaptionForgeError):
    """Raised when application configuration is invalid or unavailable."""


class VideoError(CaptionForgeError):
    """Base class for video inspection failures."""


class InvalidYouTubeUrlError(VideoError):
    """Raised when input is not a valid individual YouTube video URL."""


class UnsupportedYouTubeUrlError(VideoError):
    """Raised when a YouTube URL targets an unsupported resource type."""


class VideoUnavailableError(VideoError):
    """Raised when a video cannot be accessed."""


class PrivateVideoError(VideoUnavailableError):
    """Raised when a video is private."""


class LiveStreamNotSupportedError(VideoError):
    """Raised when a video is live or scheduled to become live."""


class MetadataRetrievalError(VideoError):
    """Raised when YouTube metadata retrieval fails."""

    @property
    def retryable(self) -> bool:
        return True


class SubtitleError(CaptionForgeError):
    """Base class for subtitle failures."""


class SubtitleDiscoveryError(SubtitleError):
    """Raised when subtitle metadata cannot be interpreted."""


class SubtitleDownloadError(SubtitleError):
    """Raised when a selected subtitle track cannot be downloaded."""

    @property
    def retryable(self) -> bool:
        return True


class SubtitleParseError(SubtitleError):
    """Raised when downloaded subtitle data cannot be parsed."""


class AudioError(CaptionForgeError):
    """Base class for audio preparation failures."""


class AudioDownloadError(AudioError):
    """Raised when an audio-only download fails."""

    @property
    def retryable(self) -> bool:
        return True


class AudioFormatUnavailableError(AudioDownloadError):
    """Raised when YouTube provides no usable audio stream."""


class FFmpegNotFoundError(AudioError):
    """Raised when the configured FFmpeg executable is unavailable."""


class AudioConversionError(AudioError):
    """Raised when FFmpeg cannot prepare the downloaded audio."""


class TemporaryDirectoryError(AudioError):
    """Raised when a temporary workspace cannot be used."""


class InsufficientDiskSpaceError(AudioError):
    """Raised when the temporary filesystem has insufficient free space."""


class CleanupError(AudioError):
    """Raised when temporary artifacts cannot be removed."""


class ProcessingInterruptedError(AudioError):
    """Raised when audio preparation is interrupted."""


class TranscriptionError(CaptionForgeError):
    """Base class for local transcription failures."""


class WhisperNotInstalledError(TranscriptionError):
    """Raised when the optional faster-whisper dependency is unavailable."""


class ModelLoadError(TranscriptionError):
    """Raised when a Whisper model cannot be loaded or downloaded."""

    @property
    def retryable(self) -> bool:
        return True


class UnsupportedModelError(ModelLoadError):
    """Raised when a requested model name is invalid."""


class CudaUnavailableError(TranscriptionError):
    """Raised when CUDA was explicitly requested but is unavailable."""


class GpuMemoryError(TranscriptionError):
    """Raised when GPU memory is exhausted."""


class InvalidComputeTypeError(TranscriptionError):
    """Raised when the selected compute type is unsupported."""


class AudioTranscriptionError(TranscriptionError):
    """Raised when prepared audio cannot be decoded or transcribed."""


class EmptyTranscriptionError(TranscriptionError):
    """Raised when transcription produces no usable speech segments."""


class TranscriptionCancelledError(TranscriptionError):
    """Raised when the user cancels an active transcription."""


class ExportError(CaptionForgeError):
    """Raised when subtitle output cannot be safely exported."""


class DocxNotInstalledError(ExportError):
    """Raised when the optional python-docx dependency is unavailable."""


class ValidationError(CaptionForgeError):
    """Raised when application-level input validation fails."""
