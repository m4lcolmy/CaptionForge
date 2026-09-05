"""In-memory job registry that runs blocking workflows off the event loop."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import Config
from app.core.exceptions import (
    CaptionForgeError,
    MediaDownloadCancelledError,
    TranscriptionCancelledError,
)
from app.core.logging_config import get_logger
from app.interfaces.web.errors import GENERIC_MESSAGE
from app.models.job import JobStatus
from app.services.factory import create_media_service, create_transcription_service

REMEMBERED_JOBS = 50


@dataclass(frozen=True)
class JobRequest:
    """Everything the workflow needs, resolved from one API request."""

    url: str
    language: str | None = None
    formats: tuple[str, ...] = ()
    model: str | None = None
    device: str | None = None
    compute_type: str | None = None
    prompt: str | None = None
    force: bool = False
    overwrite: bool = False
    keep_audio: bool = False
    timestamped_txt: bool = False
    postprocess: bool = True
    allow_translated: bool = False


@dataclass(frozen=True)
class MediaJobRequest:
    """A whole-file download: one video, one quality the page offered."""

    url: str
    quality: str
    overwrite: bool = False


@dataclass
class OutputFile:
    """One file the workflow wrote to disk."""

    name: str
    path: Path
    size_bytes: int


@dataclass
class JobRecord:
    """Mutable state of one job, safe to read from the event loop."""

    id: str
    request: JobRequest | MediaJobRequest
    status: JobStatus = JobStatus.PENDING
    stage: str = "Queued"
    percent: float = 0.0
    error: str | None = None
    error_code: str | None = None
    files: list[OutputFile] = field(default_factory=list)
    used_existing_captions: bool | None = None
    transcription_summary: dict[str, Any] | None = None
    media_summary: dict[str, Any] | None = None
    video_title: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def kind(self) -> str:
        """Which of the two workflows this job runs; the page labels by it."""
        return "media" if isinstance(self.request, MediaJobRequest) else "captions"

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-ready view of the job taken under the lock."""
        with self.lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status.value,
                "stage": self.stage,
                "percent": round(self.percent, 1),
                "error": self.error,
                "error_code": self.error_code,
                "video_title": self.video_title,
                "used_existing_captions": self.used_existing_captions,
                "transcription": self.transcription_summary,
                "media": self.media_summary,
                "cancel_requested": self.cancel_event.is_set(),
                "created_at": self.created_at.isoformat(),
                "finished_at": (
                    self.finished_at.isoformat() if self.finished_at else None
                ),
                "files": [
                    {
                        "name": file.name,
                        "size_bytes": file.size_bytes,
                        "path": str(file.path),
                    }
                    for file in self.files
                ],
            }

    def file_named(self, name: str) -> OutputFile | None:
        """Look a produced file up by name; never join untrusted input to a path."""
        with self.lock:
            return next((file for file in self.files if file.name == name), None)


class JobRegistry:
    """Owns the worker thread and the bounded history of recent jobs."""

    def __init__(self, config: Config, *, max_workers: int = 1) -> None:
        self._config = config
        self._records: OrderedDict[str, JobRecord] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="captionforge-job"
        )
        # Downloads get their own single worker. They are short next to a
        # transcription, and queueing one behind an hour of Whisper would make
        # a one-click download feel broken.
        self._media_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="captionforge-media"
        )

    def submit(self, request: JobRequest) -> JobRecord:
        """Register a job and hand it to the worker thread."""
        return self._enqueue(request, self._executor, self._run)

    def submit_media(self, request: MediaJobRequest) -> JobRecord:
        """Register a whole-file download and hand it to the download thread."""
        return self._enqueue(request, self._media_executor, self._run_media)

    def _enqueue(
        self,
        request: JobRequest | MediaJobRequest,
        executor: ThreadPoolExecutor,
        worker: Callable[[JobRecord], None],
    ) -> JobRecord:
        """Remember one job and start it on the lane that belongs to it."""
        record = JobRecord(id=uuid4().hex, request=request)
        with self._lock:
            self._records[record.id] = record
            while len(self._records) > REMEMBERED_JOBS:
                self._records.popitem(last=False)
        executor.submit(worker, record)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        """Return one job by identifier, or None when it is unknown or evicted."""
        with self._lock:
            return self._records.get(job_id)

    def recent(self) -> list[JobRecord]:
        """Return remembered jobs, newest first."""
        with self._lock:
            return list(reversed(self._records.values()))

    def cancel(self, job_id: str) -> bool:
        """Ask a running job to stop at its next checkpoint."""
        record = self.get(job_id)
        if record is None:
            return False
        record.cancel_event.set()
        return True

    def shutdown(self) -> None:
        """Cancel outstanding work and stop the worker thread."""
        with self._lock:
            records = list(self._records.values())
        for record in records:
            record.cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._media_executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, record: JobRecord) -> None:
        """Execute one workflow on the worker thread."""
        log = get_logger().bind(job_id=record.id)
        request = record.request
        assert isinstance(request, JobRequest)
        if record.cancel_event.is_set():
            self._finish(record, JobStatus.CANCELLED, stage="Cancelled")
            return
        with record.lock:
            record.status = JobStatus.RUNNING
            record.stage = "Starting"
        try:
            service = create_transcription_service(self._config)
            result = service.process(
                request.url,
                language=request.language,
                model_name=request.model,
                device=request.device,
                compute_type=request.compute_type,
                formats=request.formats or None,
                keep_audio=request.keep_audio,
                force=request.force,
                overwrite=request.overwrite,
                timestamped_txt=request.timestamped_txt,
                postprocess=request.postprocess,
                allow_translated=request.allow_translated,
                initial_prompt=request.prompt,
                progress=lambda message, percent: self._report(
                    record, message, percent
                ),
                cancelled=record.cancel_event.is_set,
            )
        except TranscriptionCancelledError:
            log.info("Web job cancelled")
            self._finish(record, JobStatus.CANCELLED, stage="Cancelled")
            return
        except CaptionForgeError as exc:
            log.error(
                "Web job failed user_message={} technical_cause={}",
                exc.message,
                exc.details or type(exc).__name__,
            )
            self._finish(
                record,
                JobStatus.FAILED,
                stage="Failed",
                error=exc.message,
                error_code=type(exc).__name__,
            )
            return
        except Exception as exc:  # noqa: BLE001 - boundary of the worker thread
            log.exception("Unexpected web job failure technical_cause={}", exc)
            self._finish(
                record,
                JobStatus.FAILED,
                stage="Failed",
                error=GENERIC_MESSAGE,
                error_code="UnexpectedError",
            )
            return
        self._complete(record, result)

    def _run_media(self, record: JobRecord) -> None:
        """Download one whole file on the download thread."""
        log = get_logger().bind(job_id=record.id)
        request = record.request
        assert isinstance(request, MediaJobRequest)
        if record.cancel_event.is_set():
            self._finish(record, JobStatus.CANCELLED, stage="Cancelled")
            return
        with record.lock:
            record.status = JobStatus.RUNNING
            record.stage = "Starting"
        try:
            result = create_media_service(self._config).download(
                request.url,
                request.quality,
                overwrite=request.overwrite,
                progress=lambda message, percent: self._report(
                    record, message, percent
                ),
                cancelled=record.cancel_event.is_set,
            )
        except MediaDownloadCancelledError:
            log.info("Web download cancelled")
            self._finish(record, JobStatus.CANCELLED, stage="Cancelled")
            return
        except CaptionForgeError as exc:
            log.error(
                "Web download failed user_message={} technical_cause={}",
                exc.message,
                exc.details or type(exc).__name__,
            )
            self._finish(
                record,
                JobStatus.FAILED,
                stage="Failed",
                error=exc.message,
                error_code=type(exc).__name__,
            )
            return
        except Exception as exc:  # noqa: BLE001 - boundary of the worker thread
            log.exception("Unexpected web download failure technical_cause={}", exc)
            self._finish(
                record,
                JobStatus.FAILED,
                stage="Failed",
                error=GENERIC_MESSAGE,
                error_code="UnexpectedError",
            )
            return
        self._complete_media(record, result)

    @staticmethod
    def _complete_media(record: JobRecord, result: Any) -> None:
        """Record the single produced file and the quality it came from."""
        path = Path(result.path).resolve()
        with record.lock:
            record.files = [
                OutputFile(
                    name=path.name,
                    path=path,
                    size_bytes=path.stat().st_size if path.is_file() else 0,
                )
            ]
            record.media_summary = {
                "key": result.variant.key,
                "label": result.variant.label,
                "kind": result.variant.kind.value,
                "extension": result.variant.extension,
            }
            record.video_title = result.video.title
            record.status = JobStatus.COMPLETED
            record.stage = "Completed"
            record.percent = 100.0
            record.finished_at = datetime.now(UTC)

    @staticmethod
    def _report(record: JobRecord, message: str, percent: float | None) -> None:
        """Copy a workflow progress callback into the record."""
        with record.lock:
            record.stage = message
            if percent is not None:
                record.percent = max(record.percent, min(100.0, percent))

    def _complete(self, record: JobRecord, result: Any) -> None:
        """Record produced files and transcription metadata."""
        files: list[OutputFile] = []
        for path in result.paths:
            resolved = Path(path).resolve()
            files.append(
                OutputFile(
                    name=resolved.name,
                    path=resolved,
                    size_bytes=(resolved.stat().st_size if resolved.is_file() else 0),
                )
            )
        summary: dict[str, Any] | None = None
        if result.transcription is not None:
            info = result.transcription
            summary = {
                "detected_language": info.detected_language,
                "language_probability": info.language_probability,
                "model_name": info.model_name,
                "device": info.device,
                "compute_type": info.compute_type,
            }
        with record.lock:
            record.files = files
            record.used_existing_captions = result.used_existing_captions
            record.transcription_summary = summary
            record.video_title = result.video.title
            record.status = JobStatus.COMPLETED
            record.stage = "Completed"
            record.percent = 100.0
            record.finished_at = datetime.now(UTC)

    @staticmethod
    def _finish(
        record: JobRecord,
        status: JobStatus,
        *,
        stage: str,
        error: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Move a job to a terminal state."""
        with record.lock:
            record.status = status
            record.stage = stage
            record.error = error
            record.error_code = error_code
            record.finished_at = datetime.now(UTC)


__all__ = [
    "JobRecord",
    "JobRegistry",
    "JobRequest",
    "MediaJobRequest",
    "OutputFile",
]
