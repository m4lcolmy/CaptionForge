"""Public video metadata model."""

from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.utils.url_utils import is_valid_video_id


class VideoMetadata(BaseModel):
    """Stable metadata for one video: on YouTube, or a file on this computer.

    Exactly one of ``video_id`` and ``local_path`` is set. Everything that
    needs YouTube (caption tracks, stream downloads) reads ``video_id``;
    everything that needs only sound reads whichever is there.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    video_id: str | None = None
    title: str = Field(min_length=1)
    channel_name: str | None = None
    channel_id: str | None = None
    duration_seconds: int | None = Field(default=None, ge=0)
    webpage_url: str = Field(min_length=1)
    original_url: str = Field(min_length=1)
    thumbnail_url: str | None = None
    upload_date: date | None = None
    is_live: bool = False
    live_status: str | None = None
    availability: str | None = None
    age_limit: int | None = Field(default=None, ge=0)
    description: str | None = None
    local_path: Path | None = None

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str | None) -> str | None:
        """Require YouTube's canonical eleven-character identifier."""
        if value is not None and not is_valid_video_id(value):
            raise ValueError("video_id must be a valid YouTube video ID")
        return value

    @model_validator(mode="after")
    def validate_source(self) -> "VideoMetadata":
        """A video comes from exactly one place."""
        if (self.video_id is None) == (self.local_path is None):
            raise ValueError("set exactly one of video_id and local_path")
        return self

    @property
    def is_local(self) -> bool:
        """Whether this is a file on this computer rather than a YouTube video."""
        return self.local_path is not None
