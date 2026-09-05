"""Downloadable media models: what a video offers and what was written."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import MEDIA_AUDIO_KEY
from app.models.video import VideoMetadata


class MediaKind(StrEnum):
    """Whether a download keeps the picture or only the sound."""

    VIDEO = "video"
    AUDIO = "audio"


class MediaVariant(BaseModel):
    """One download the page can offer, already known to exist for this video."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=16)
    kind: MediaKind
    label: str = Field(min_length=1)
    extension: str = Field(min_length=1)
    height: int | None = Field(default=None, ge=1)
    estimated_bytes: int | None = Field(default=None, ge=0)

    @property
    def is_audio(self) -> bool:
        """Whether this variant produces an audio-only file."""
        return self.kind is MediaKind.AUDIO


class MediaOptions(BaseModel):
    """Every download one video offers, best first, audio last."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variants: tuple[MediaVariant, ...] = ()

    def video_variants(self) -> tuple[MediaVariant, ...]:
        """Return only the variants that keep the picture, highest first."""
        return tuple(item for item in self.variants if item.kind is MediaKind.VIDEO)

    def audio_variant(self) -> MediaVariant | None:
        """Return the audio-only variant, when the video has an audio stream."""
        return next(
            (item for item in self.variants if item.kind is MediaKind.AUDIO), None
        )

    def resolve(self, requested: str) -> MediaVariant | None:
        """Find the variant a request names, tolerating a height that is gone.

        A saved link, a typed ``--quality 1080``, and a stale page can all ask
        for a height this video does not publish. Stepping down to the best
        available one below it is what the person meant; failing is not.
        """
        wanted = requested.strip().lower()
        if not wanted:
            return None
        exact = next((item for item in self.variants if item.key == wanted), None)
        if exact is not None:
            return exact
        videos = self.video_variants()
        if wanted in {MEDIA_AUDIO_KEY, "mp3", "sound"}:
            return self.audio_variant()
        if wanted in {"best", "highest", "max"}:
            return videos[0] if videos else None
        height = _requested_height(wanted)
        if height is None or not videos:
            return None
        below = [item for item in videos if (item.height or 0) <= height]
        return below[0] if below else videos[-1]


class MediaDownloadResult(BaseModel):
    """One finished download and the file it produced."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    variant: MediaVariant
    video: VideoMetadata


def _requested_height(value: str) -> int | None:
    """Read 1080, 1080p, or 1080P as a height, and anything else as nothing."""
    digits = value[:-1] if value.endswith("p") else value
    return int(digits) if digits.isdigit() else None
