"""Job data model."""

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    """Lifecycle states for caption and transcription processing jobs."""

    PENDING = "pending"
    RUNNING = "running"
    PREPARING_AUDIO = "preparing_audio"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    POST_PROCESSING = "post_processing"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """A requested CaptionForge processing job."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    source_url: str = Field(min_length=1)
    language: str = Field(default="ar", min_length=2)
    output_directory: Path = Path("output")
    output_formats: tuple[str, ...] = ("srt", "vtt")
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error_message: str | None = None
    temporary_directory: Path | None = None
    downloaded_audio_path: Path | None = None
    prepared_audio_path: Path | None = None
    progress_percent: float = Field(default=0, ge=0, le=100)
    current_stage: str = "created"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure_stage: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)

    def transition(self, status: JobStatus, *, stage: str | None = None) -> None:
        """Move the job to a new state and maintain lifecycle timestamps."""
        now = datetime.now(UTC)
        if self.started_at is None and status is not JobStatus.PENDING:
            self.started_at = now
        self.status = status
        self.current_stage = stage or status.value
        self.updated_at = now
        if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            self.finished_at = now
            if self.started_at is not None:
                self.duration_seconds = max(
                    0.0, (self.finished_at - self.started_at).total_seconds()
                )
            if status is JobStatus.FAILED:
                self.failure_stage = self.current_stage
