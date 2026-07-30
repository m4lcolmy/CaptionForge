"""Engine-independent transcription models."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TranscriptionSegment(BaseModel):
    """A timestamped segment returned by a transcription adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1)
    language: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    no_speech_probability: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_timing(self) -> "TranscriptionSegment":
        """Ensure the segment ends after it starts."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class TranscriptionResult(BaseModel):
    """Stable result detached from faster-whisper implementation objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segments: tuple[TranscriptionSegment, ...]
    detected_language: str = Field(min_length=1)
    language_probability: float | None = Field(default=None, ge=0, le=1)
    duration_seconds: float | None = Field(default=None, ge=0)
    model_name: str = Field(min_length=1)
    device: str = Field(min_length=1)
    compute_type: str = Field(min_length=1)
