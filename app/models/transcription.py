"""Engine-independent transcription models."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WordTiming(BaseModel):
    """One word with the timing the engine aligned it to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    text: str = Field(min_length=1)
    probability: float | None = Field(default=None, ge=0, le=1)


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
    words: tuple[WordTiming, ...] = ()

    @model_validator(mode="after")
    def validate_timing(self) -> "TranscriptionSegment":
        """Ensure the segment ends after it starts."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class TranscriptionResult(BaseModel):
    """Stable result detached from any engine's implementation objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segments: tuple[TranscriptionSegment, ...]
    detected_language: str = Field(min_length=1)
    language_probability: float | None = Field(default=None, ge=0, le=1)
    duration_seconds: float | None = Field(default=None, ge=0)
    model_name: str = Field(min_length=1)
    device: str = Field(min_length=1)
    compute_type: str = Field(min_length=1)
    # "whisper" ran on this computer; "deepgram" ran on Deepgram's servers,
    # where device and compute type describe nothing a person chose.
    engine: str = Field(default="whisper", min_length=1)
