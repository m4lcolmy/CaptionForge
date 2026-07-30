"""Public video metadata model."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils.url_utils import is_valid_video_id


class VideoMetadata(BaseModel):
    """Stable metadata for one YouTube video."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    video_id: str
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

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str) -> str:
        """Require YouTube's canonical eleven-character identifier."""
        if not is_valid_video_id(value):
            raise ValueError("video_id must be a valid YouTube video ID")
        return value
