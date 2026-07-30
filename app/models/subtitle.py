"""Subtitle segment model."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.video import VideoMetadata


class SubtitleSourceType(StrEnum):
    """Origin of a subtitle track."""

    MANUAL = "manual"
    AUTOMATIC = "automatic"


class SubtitleTrack(BaseModel):
    """Stable description of one available subtitle language track."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_code: str = Field(min_length=1)
    normalized_language_code: str = Field(min_length=1)
    language_name: str | None = None
    source_type: SubtitleSourceType
    is_automatic: bool
    available_formats: tuple[str, ...] = ()
    track_count: int = Field(default=0, ge=0)


class SubtitleDiscoveryResult(BaseModel):
    """Metadata, discovered tracks, and preferred-language selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    video: VideoMetadata
    manual_tracks: tuple[SubtitleTrack, ...] = ()
    automatic_tracks: tuple[SubtitleTrack, ...] = ()
    selected_track: SubtitleTrack | None = None
    preferred_language: str
    selection_reason: str | None = None


class SubtitleSegment(BaseModel):
    """A timestamped subtitle text segment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1)
    language: str = Field(min_length=2)
    speaker: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    no_speech_probability: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_timing(self) -> "SubtitleSegment":
        """Ensure the segment ends after it starts."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class RawSubtitle(BaseModel):
    """Downloaded caption content before parsing and cleanup."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    format: str = Field(min_length=1)
