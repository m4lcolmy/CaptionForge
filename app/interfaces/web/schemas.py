"""Request and response models for the local web API."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.constants import SUPPORTED_OUTPUT_FORMATS


class InspectRequest(BaseModel):
    """A request to inspect one YouTube video."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2048)
    language: str | None = Field(default=None, max_length=32)
    allow_translated: bool = False


class JobRequestBody(BaseModel):
    """A request to export captions, falling back to local transcription."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2048)
    language: str | None = Field(default=None, max_length=32)
    formats: tuple[str, ...] = ()
    model: str | None = Field(default=None, max_length=256)
    device: str | None = Field(default=None, max_length=32)
    compute_type: str | None = Field(default=None, max_length=32)
    prompt: str | None = Field(default=None, max_length=2048)
    force: bool = False
    overwrite: bool = False
    keep_audio: bool = False
    timestamped_txt: bool = False
    postprocess: bool = True
    allow_translated: bool = False

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject unknown output formats before any work is scheduled."""
        unknown = sorted(set(value) - SUPPORTED_OUTPUT_FORMATS)
        if unknown:
            supported = ", ".join(sorted(SUPPORTED_OUTPUT_FORMATS))
            raise ValueError(
                f"Unsupported format(s): {', '.join(unknown)}. Choose from {supported}."
            )
        return value


class MediaJobRequestBody(BaseModel):
    """A request to download the whole file as an MP4 or an MP3."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2048)
    # A key the page read back from an inspection: "audio", or a height such
    # as "1080". The service resolves it against what the video really offers.
    quality: str = Field(min_length=1, max_length=16)
    overwrite: bool = False
