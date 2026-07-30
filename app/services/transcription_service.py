"""Caption-first orchestration for local Whisper transcription."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.adapters.whisper_adapter import WhisperAdapter
from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.config import Config
from app.core.exceptions import (
    CaptionForgeError,
    CleanupError,
    TranscriptionCancelledError,
)
from app.core.logging_config import get_logger
from app.core.retry import retry_call
from app.models.job import Job, JobStatus
from app.models.subtitle import SubtitleSegment, SubtitleSourceType, SubtitleTrack
from app.models.transcription import TranscriptionResult
from app.models.video import VideoMetadata
from app.services.audio_service import AudioService
from app.services.export_service import ExportService
from app.services.postprocessing_service import PostProcessingService
from app.services.subtitle_service import SubtitleService
from app.services.video_service import VideoService
from app.utils.file_utils import cleanup_path

ProgressCallback = Callable[[str, float | None], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class TranscriptionWorkflowResult:
    """Files and processing metadata returned by the complete workflow."""

    paths: tuple[Path, ...]
    video: VideoMetadata
    used_existing_captions: bool
    transcription: TranscriptionResult | None = None
    prepared_audio_path: Path | None = None


class TranscriptionService:
    """Choose captions first, otherwise prepare audio and run Whisper."""

    def __init__(
        self,
        video_service: VideoService,
        ytdlp: YtDlpAdapter,
        subtitle_service: SubtitleService,
        audio_service: AudioService,
        whisper: WhisperAdapter,
        export_service: ExportService,
        config: Config,
    ) -> None:
        self._video_service = video_service
        self._ytdlp = ytdlp
        self._subtitle_service = subtitle_service
        self._audio_service = audio_service
        self._whisper = whisper
        self._export_service = export_service
        self._config = config

    def process(
        self,
        url: str,
        *,
        language: str | None = None,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        formats: Sequence[str] | None = None,
        output_directory: Path | None = None,
        keep_audio: bool = False,
        force: bool = False,
        overwrite: bool = False,
        timestamped_txt: bool = False,
        postprocess: bool = True,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
    ) -> TranscriptionWorkflowResult:
        """Inspect, select the appropriate source, transcribe if needed, and export."""
        notify = progress or (lambda _message, _percent: None)
        is_cancelled = cancelled or (lambda: False)
        selected_language = language or self._config.whisper_language
        inspection_language = selected_language or self._config.default_language
        job = Job(source_url=url, language=inspection_language)
        job.transition(JobStatus.RUNNING, stage="inspection")
        log = get_logger().bind(job_id=str(job.id), stage=job.current_stage)
        audio_path: Path | None = None
        preserve_audio = keep_audio or self._config.keep_temp_files
        try:
            self._check_cancelled(is_cancelled)
            notify("Inspecting video", 5.0)
            discovery = self._video_service.inspect(url, inspection_language)
            track = discovery.selected_track
            requested_formats = formats or self._config.default_output_formats
            destination = output_directory or self._config.default_output_folder

            if track is not None and not force:
                notify("Using existing captions", 25.0)
                segments = self._subtitle_service.retrieve_and_parse(
                    self._ytdlp,
                    discovery.video.video_id,
                    track,
                    postprocess=postprocess,
                )
                self._check_cancelled(is_cancelled)
                job.status = JobStatus.EXPORTING
                notify("Exporting", 85.0)
                paths = self._export_service.export(
                    discovery.video,
                    track,
                    segments,
                    requested_formats,
                    destination,
                    timestamped_txt=timestamped_txt,
                    overwrite=overwrite,
                )
                job.status = JobStatus.COMPLETED
                notify("Completed", 100.0)
                return TranscriptionWorkflowResult(
                    paths=paths,
                    video=discovery.video,
                    used_existing_captions=True,
                )

            job.status = JobStatus.PREPARING_AUDIO
            notify("Preparing audio", 10.0)
            audio_path = self._audio_service.prepare(
                url,
                inspection_language,
                keep_temp=True,
                force=True,
                discovery=discovery,
                progress=lambda message, percent: notify(
                    message,
                    None if percent is None else 10.0 + percent * 0.15,
                ),
            )
            job.prepared_audio_path = audio_path
            self._check_cancelled(is_cancelled)
            job.status = JobStatus.LOADING_MODEL

            def report_transcription(message: str, percent: float | None) -> None:
                job.status = (
                    JobStatus.LOADING_MODEL
                    if message == "Loading model"
                    else JobStatus.TRANSCRIBING
                )
                if percent is not None:
                    job.progress_percent = percent
                notify(message, percent)

            selected_model = model_name or self._config.default_whisper_model
            selected_device = device or self._config.whisper_device
            log.info(
                "Transcription selected_method=whisper video_id={} model={} device={}",
                discovery.video.video_id,
                selected_model,
                selected_device,
            )
            transcription = retry_call(
                lambda: self._whisper.transcribe(
                    audio_path,
                    model_name=selected_model,
                    device=selected_device,
                    compute_type=compute_type or self._config.whisper_compute_type,
                    language=selected_language,
                    beam_size=self._config.whisper_beam_size,
                    vad_enabled=self._config.whisper_vad_enabled,
                    min_silence_duration_ms=(
                        self._config.whisper_min_silence_duration_ms
                    ),
                    download_root=self._config.whisper_model_download_directory,
                    progress=report_transcription,
                    cancelled=is_cancelled,
                ),
                attempts=self._config.retry_count,
                delay_seconds=self._config.retry_delay_seconds,
                operation_name="whisper_model_or_transcription",
            )
            job.status = JobStatus.POST_PROCESSING
            notify("Post-processing", 88.0)
            segments = self.to_subtitle_segments(transcription)
            if postprocess:
                segments = PostProcessingService(self._config).process(segments)
            track = SubtitleTrack(
                language_code=transcription.detected_language,
                normalized_language_code=transcription.detected_language,
                language_name=None,
                source_type=SubtitleSourceType.AUTOMATIC,
                is_automatic=True,
            )
            self._check_cancelled(is_cancelled)
            job.status = JobStatus.EXPORTING
            notify("Exporting", 92.0)
            paths = self._export_service.export(
                discovery.video,
                track,
                segments,
                requested_formats,
                destination,
                timestamped_txt=timestamped_txt,
                overwrite=overwrite,
            )
            job.status = JobStatus.COMPLETED
            notify("Completed", 100.0)
            return TranscriptionWorkflowResult(
                paths=paths,
                video=discovery.video,
                used_existing_captions=False,
                transcription=transcription,
                prepared_audio_path=audio_path if preserve_audio else None,
            )
        except KeyboardInterrupt as exc:
            job.transition(JobStatus.CANCELLED)
            raise TranscriptionCancelledError("Transcription was cancelled.") from exc
        except TranscriptionCancelledError:
            job.transition(JobStatus.CANCELLED)
            raise
        except CaptionForgeError as exc:
            job.error_message = exc.message
            job.transition(JobStatus.FAILED, stage=job.current_stage)
            raise
        finally:
            if audio_path is not None and not preserve_audio:
                try:
                    if audio_path.parent.name.startswith("captionforge-"):
                        cleanup_path(audio_path.parent)
                    else:
                        cleanup_path(audio_path)
                except CleanupError:
                    if job.status is JobStatus.COMPLETED:
                        raise
            if job.status is JobStatus.COMPLETED:
                job.transition(JobStatus.COMPLETED)
            log.info(
                "Job finished status={} failure_stage={} duration_seconds={}",
                job.status,
                job.failure_stage,
                job.duration_seconds,
            )

    @staticmethod
    def to_subtitle_segments(
        result: TranscriptionResult,
    ) -> tuple[SubtitleSegment, ...]:
        """Convert adapter output into the common exporter representation."""
        return tuple(
            SubtitleSegment(
                index=segment.index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                text=segment.text,
                language=segment.language or result.detected_language,
                confidence=segment.confidence,
                no_speech_probability=segment.no_speech_probability,
            )
            for segment in result.segments
        )

    @staticmethod
    def _check_cancelled(cancelled: CancelCallback) -> None:
        if cancelled():
            raise TranscriptionCancelledError("Transcription was cancelled.")
