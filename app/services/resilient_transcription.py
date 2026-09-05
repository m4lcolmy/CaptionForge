"""Transcription that degrades instead of failing when a GPU runs out of memory.

Three mechanisms combine here, and each covers a gap the others leave:

*Pre-flight* asks the driver how much memory is actually free and starts on the
richest plan predicted to fit, so a modest card quietly runs a smaller
configuration instead of discovering the limit by crashing.

*The ladder* handles the cases pre-flight cannot predict - another process
taking memory mid-run, allocator fragmentation, an unusually dense stretch of
speech. Each failure steps to the next-cheapest plan, and the last rung is CPU,
which has no device memory to exhaust. That terminal rung is what turns "job
failed" into "job finished, slower".

*Checkpoint and resume* stops a late failure from being expensive. Segments are
recorded as the engine emits them, and the next rung restarts from the end of
the last completed segment rather than from zero. Because the prepared audio is
mono 16 kHz PCM, that restart is an exact byte offset into the file, and because
the voice-activity filter ends segments in silence, the cut never lands
mid-word.

Note that splitting long audio into pieces is deliberately *not* used to reduce
memory: Whisper is a fixed 30-second-window model, so peak memory is identical
for a one-minute file and an eight-hour one. Chunking earns its place here only
as a resume boundary.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.adapters.gpu_memory import format_bytes, read_vram
from app.adapters.whisper_adapter import WhisperAdapter
from app.core.exceptions import (
    EmptyTranscriptionError,
    GpuMemoryError,
    TranscriptionCancelledError,
)
from app.core.logging_config import get_logger
from app.models.transcription import TranscriptionResult, TranscriptionSegment
from app.services.transcription_plan import (
    DEFAULT_HEADROOM_BYTES,
    TranscriptionPlan,
    build_ladder,
    remaining_after,
    select_plan,
)
from app.utils.audio_slice import slice_wav_tail, wav_duration_seconds
from app.utils.file_utils import cleanup_path

ProgressCallback = Callable[[str, float | None], None]
CancelCallback = Callable[[], bool]
VramReader = Callable[[], Any]

# Rewind slightly past the last finished segment so the decoder re-enters with
# a little acoustic context instead of starting cold on the next word.
RESUME_REWIND_SECONDS = 0.5

# A run that produced nothing at all has not proven the plan works, so resuming
# from its "progress" would loop on the same failure.
_MINIMUM_RESUME_SECONDS = 1.0


@dataclass
class _Checkpoint:
    """Everything finished so far, across however many plans it took."""

    segments: list[TranscriptionSegment]
    language: str | None = None
    language_probability: float | None = None

    @property
    def resume_at_seconds(self) -> float:
        """The point a later attempt should restart from."""
        if not self.segments:
            return 0.0
        return max(0.0, self.segments[-1].end_seconds - RESUME_REWIND_SECONDS)

    def accept(self, segment: TranscriptionSegment) -> None:
        """Record one finished segment, dropping anything already covered."""
        # The rewind replays a little audio the previous attempt already
        # covered; anything wholly inside that overlap is a duplicate.
        if self.segments and segment.end_seconds <= self.segments[-1].end_seconds:
            return
        self.segments.append(segment)


class ResilientTranscriber:
    """Run a transcription across a ladder of progressively cheaper plans."""

    def __init__(
        self,
        whisper: WhisperAdapter,
        *,
        vram_reader: VramReader = read_vram,
        headroom_bytes: int = DEFAULT_HEADROOM_BYTES,
        allow_model_downgrade: bool = True,
        allow_word_timestamp_loss: bool = True,
        allow_cpu_fallback: bool = True,
    ) -> None:
        self._whisper = whisper
        self._vram_reader = vram_reader
        self._headroom = headroom_bytes
        self._allow_model_downgrade = allow_model_downgrade
        self._allow_word_timestamp_loss = allow_word_timestamp_loss
        self._allow_cpu_fallback = allow_cpu_fallback

    def transcribe(
        self,
        audio_path: Path,
        *,
        model_name: str,
        device: str,
        compute_type: str,
        beam_size: int,
        word_timestamps: bool,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
        **engine_options: Any,
    ) -> TranscriptionResult:
        """Transcribe, stepping down the ladder for as long as memory forces it."""
        notify = progress or (lambda _message, _percent: None)
        is_cancelled = cancelled or (lambda: False)
        log = get_logger()

        requested = TranscriptionPlan(
            model_name=model_name,
            device=self._whisper.select_device(device),
            compute_type=WhisperAdapter.select_compute_type(
                compute_type, self._whisper.select_device(device)
            ),
            beam_size=beam_size,
            word_timestamps=word_timestamps,
        )
        ladder = build_ladder(
            requested,
            allow_word_timestamp_loss=self._allow_word_timestamp_loss,
            allow_model_downgrade=self._allow_model_downgrade,
            allow_cpu_fallback=self._allow_cpu_fallback,
        )
        start = self._preflight(ladder, requested, log)
        attempts = (start, *remaining_after(ladder, start))

        checkpoint = _Checkpoint(segments=[])
        total_duration = wav_duration_seconds(audio_path)
        slices: list[Path] = []
        last_failure: str | None = None
        try:
            for position, plan in enumerate(attempts):
                if is_cancelled():
                    raise TranscriptionCancelledError("Transcription was cancelled.")
                if position:
                    log.warning(
                        "Stepping down after GPU memory pressure plan={} "
                        "resume_at_seconds={:.1f} committed_segments={}",
                        plan.describe(),
                        checkpoint.resume_at_seconds,
                        len(checkpoint.segments),
                    )
                    notify(f"Retrying with {plan.model_name} on {plan.device}", None)
                source, offset = self._source_for(plan, audio_path, checkpoint, slices)
                try:
                    result = self._run(
                        source,
                        plan,
                        checkpoint,
                        offset,
                        total_duration,
                        notify,
                        is_cancelled,
                        engine_options,
                    )
                except GpuMemoryError as exc:
                    # Keep only the text. Holding the exception would keep its
                    # traceback alive, that traceback holds the engine frames,
                    # and those frames hold the model whose memory the next rung
                    # needs. This is why an in-place `except: retry` never works.
                    last_failure = exc.details or exc.message
                    del exc
                    gc.collect()
                    continue
                except EmptyTranscriptionError:
                    if not checkpoint.segments:
                        raise
                    # A resumed tail that is pure silence still leaves a
                    # complete transcript from the earlier attempts.
                    result = None
                return self._assemble(checkpoint, plan, result, total_duration)
        finally:
            for path in slices:
                with suppress(Exception):
                    cleanup_path(path)

        raise GpuMemoryError(
            "Every GPU configuration ran out of memory, and the CPU fallback "
            "could not be used either. Free device memory, or run with "
            "--device cpu.",
            details=last_failure,
        )

    def _preflight(
        self,
        ladder: tuple[TranscriptionPlan, ...],
        requested: TranscriptionPlan,
        log: Any,
    ) -> TranscriptionPlan:
        """Choose a starting rung from the memory the driver reports free."""
        if requested.device != "cuda":
            return ladder[0]
        snapshot = self._vram_reader()
        free = getattr(snapshot, "free_bytes", None)
        chosen = select_plan(ladder, free, headroom=self._headroom)
        if free is None:
            log.debug("GPU memory could not be read; starting on the requested plan")
        elif chosen != requested:
            log.info(
                "Pre-flight downgrade free_vram={} requested={} selected={}",
                format_bytes(free),
                requested.describe(),
                chosen.describe(),
            )
        else:
            log.debug(
                "Pre-flight fit free_vram={} plan={}",
                format_bytes(free),
                chosen.describe(),
            )
        return chosen

    def _source_for(
        self,
        plan: TranscriptionPlan,
        audio_path: Path,
        checkpoint: _Checkpoint,
        slices: list[Path],
    ) -> tuple[Path, float]:
        """Return the audio this attempt should read, and its time offset."""
        resume_at = checkpoint.resume_at_seconds
        if resume_at < _MINIMUM_RESUME_SECONDS:
            return audio_path, 0.0
        destination = audio_path.parent / f"resume-{len(slices)}-{audio_path.name}"
        actual = slice_wav_tail(audio_path, destination, resume_at)
        slices.append(destination)
        return destination, actual

    def _run(
        self,
        source: Path,
        plan: TranscriptionPlan,
        checkpoint: _Checkpoint,
        offset: float,
        total_duration: float | None,
        notify: ProgressCallback,
        is_cancelled: CancelCallback,
        engine_options: dict[str, Any],
    ) -> TranscriptionResult:
        """Execute exactly one plan, checkpointing every segment it produces."""
        options = dict(engine_options)
        if not plan.word_timestamps:
            # The adapter turns alignment back on whenever this threshold is
            # set, which would undo the rung that just dropped it.
            options["hallucination_silence_threshold"] = None
        return self._whisper.transcribe(
            source,
            model_name=plan.model_name,
            device=plan.device,
            compute_type=plan.compute_type,
            beam_size=plan.beam_size,
            word_timestamps=plan.word_timestamps,
            time_offset_seconds=offset,
            first_segment_index=len(checkpoint.segments) + 1,
            total_duration_seconds=total_duration,
            progress=notify,
            cancelled=is_cancelled,
            on_segment=checkpoint.accept,
            **options,
        )

    @staticmethod
    def _assemble(
        checkpoint: _Checkpoint,
        plan: TranscriptionPlan,
        result: TranscriptionResult | None,
        total_duration: float | None,
    ) -> TranscriptionResult:
        """Join every attempt's committed segments into one renumbered result.

        The checkpoint is a streaming optimisation, not the source of truth: an
        adapter that returns its segments without announcing them one by one
        must still produce a complete transcript, so anything the final result
        carries beyond the checkpoint is folded back in here.
        """
        for segment in result.segments if result is not None else ():
            checkpoint.accept(segment)
        segments = tuple(
            segment.model_copy(update={"index": position})
            for position, segment in enumerate(checkpoint.segments, start=1)
        )
        if not segments:
            raise EmptyTranscriptionError(
                "Whisper found no usable speech in the prepared audio."
            )
        language = (
            result.detected_language
            if result is not None
            else (segments[0].language or "unknown")
        )
        return TranscriptionResult(
            segments=segments,
            detected_language=language,
            language_probability=(
                result.language_probability if result is not None else None
            ),
            duration_seconds=total_duration
            or (result.duration_seconds if result is not None else None),
            model_name=plan.model_name,
            device=plan.device,
            compute_type=plan.compute_type,
        )


__all__ = ["ResilientTranscriber"]
