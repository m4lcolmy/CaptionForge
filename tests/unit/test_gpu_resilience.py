"""Coverage for GPU memory planning, degradation, and resume-after-failure."""

from __future__ import annotations

import contextlib
import wave
from pathlib import Path
from typing import Any

import pytest

from app.adapters.gpu_memory import VramSnapshot, estimate_vram_bytes, resolve_profile
from app.core.exceptions import (
    EmptyTranscriptionError,
    GpuMemoryError,
    TranscriptionCancelledError,
)
from app.models.transcription import TranscriptionResult, TranscriptionSegment
from app.services.resilient_transcription import ResilientTranscriber
from app.services.transcription_plan import (
    TranscriptionPlan,
    build_ladder,
    select_plan,
)
from app.utils.audio_slice import slice_wav_tail, wav_duration_seconds

GIBIBYTE = 1024 * 1024 * 1024

# A transcript spread over a two-minute recording, used to prove that a failure
# part-way through keeps what came before it.
TIMELINE: tuple[tuple[float, float, str], ...] = (
    (0.0, 10.0, "أول"),
    (12.0, 22.0, "ثاني"),
    (24.0, 34.0, "ثالث"),
    (36.0, 46.0, "رابع"),
    (48.0, 58.0, "خامس"),
)

REQUESTED = TranscriptionPlan(
    model_name="large-v3",
    device="cuda",
    compute_type="float16",
    beam_size=5,
    word_timestamps=True,
)


def write_silence(path: Path, seconds: float) -> Path:
    """Write a mono 16 kHz PCM WAV of the requested length."""
    with contextlib.closing(wave.open(str(path), "wb")) as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


class ScriptedWhisper:
    """A Whisper adapter whose every call either fails on a rung or completes.

    ``script`` holds one entry per expected call: an integer means "emit that
    many segments, then run out of GPU memory", and None means "transcribe the
    rest successfully".
    """

    def __init__(self, script: list[int | None]) -> None:
        self.script = script
        self.plans: list[dict[str, Any]] = []
        self.offsets: list[float] = []

    def select_device(self, requested: str) -> str:
        return "cuda" if requested in {"auto", "cuda"} else requested

    @staticmethod
    def select_compute_type(requested: str, device: str) -> str:
        return "float16" if requested == "auto" else requested

    def transcribe(self, audio: Path, **kwargs: Any) -> TranscriptionResult:
        self.plans.append(
            {
                "model_name": kwargs["model_name"],
                "device": kwargs["device"],
                "compute_type": kwargs["compute_type"],
                "beam_size": kwargs["beam_size"],
                "word_timestamps": kwargs["word_timestamps"],
            }
        )
        offset = kwargs["time_offset_seconds"]
        self.offsets.append(offset)
        on_segment = kwargs["on_segment"]
        budget = self.script.pop(0)

        emitted: list[TranscriptionSegment] = []
        remaining = [item for item in TIMELINE if item[0] >= offset - 0.001]
        for start, end, text in remaining:
            if budget is not None and len(emitted) >= budget:
                raise GpuMemoryError(
                    "The GPU ran out of memory.", details="CUDA out of memory"
                )
            segment = TranscriptionSegment(
                index=kwargs["first_segment_index"] + len(emitted),
                start_seconds=start,
                end_seconds=end,
                text=text,
                language="ar",
            )
            emitted.append(segment)
            on_segment(segment)
        if budget is not None:
            raise GpuMemoryError(
                "The GPU ran out of memory.", details="CUDA out of memory"
            )
        return TranscriptionResult(
            segments=tuple(emitted),
            detected_language="ar",
            language_probability=0.99,
            duration_seconds=60.0,
            model_name=kwargs["model_name"],
            device=kwargs["device"],
            compute_type=kwargs["compute_type"],
        )


def run(
    whisper: ScriptedWhisper,
    audio: Path,
    *,
    free_bytes: int | None = 24 * GIBIBYTE,
    **overrides: Any,
) -> TranscriptionResult:
    transcriber = ResilientTranscriber(
        whisper,  # type: ignore[arg-type]
        vram_reader=lambda: (
            None if free_bytes is None else VramSnapshot(24 * GIBIBYTE, free_bytes)
        ),
    )
    options: dict[str, Any] = {
        "model_name": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 5,
        "word_timestamps": True,
    }
    options.update(overrides)
    return transcriber.transcribe(audio, **options)


def test_ladder_gives_up_quality_in_order_and_ends_on_cpu() -> None:
    ladder = build_ladder(REQUESTED)
    assert ladder[0] == REQUESTED
    # Beam width is surrendered before precision, precision before word
    # timings, and the checkpoint itself before the model that produces it.
    assert [plan.beam_size for plan in ladder[:3]] == [5, 2, 1]
    assert [plan.compute_type for plan in ladder[3:5]] == ["int8_float16", "int8"]
    assert ladder[5].word_timestamps is False
    # Every large revision is one rung: swapping v3 for v2 frees no memory.
    assert [plan.model_name for plan in ladder[6:8]] == ["medium", "small"]
    # The terminal rung cannot exhaust device memory, and it restores the
    # checkpoint the caller actually asked for.
    assert ladder[-1].device == "cpu"
    assert ladder[-1].model_name == REQUESTED.model_name


def test_ladder_of_a_cpu_request_has_nothing_to_degrade() -> None:
    cpu = TranscriptionPlan("small", "cpu", "int8", 5, True)
    assert build_ladder(cpu) == (cpu,)


def test_preflight_starts_on_a_plan_that_fits_the_card() -> None:
    ladder = build_ladder(REQUESTED)
    roomy = select_plan(ladder, 24 * GIBIBYTE)
    cramped = select_plan(ladder, 4 * GIBIBYTE)
    assert roomy == REQUESTED
    assert cramped != REQUESTED
    assert cramped.estimated_vram_bytes < REQUESTED.estimated_vram_bytes


def test_preflight_treats_an_unreadable_driver_as_no_evidence() -> None:
    ladder = build_ladder(REQUESTED)
    assert select_plan(ladder, None) == REQUESTED


def test_out_of_memory_downgrades_instead_of_failing(tmp_path: Path) -> None:
    audio = write_silence(tmp_path / "prepared.wav", 60)
    whisper = ScriptedWhisper([0, None])
    result = run(whisper, audio)

    assert [item.text for item in result.segments] == [text for _, _, text in TIMELINE]
    # The first rung surrendered beam width and nothing else.
    assert whisper.plans[0]["beam_size"] == 5
    assert whisper.plans[1]["beam_size"] == 2
    assert whisper.plans[1]["model_name"] == "large-v3"


def test_a_late_failure_resumes_instead_of_restarting(tmp_path: Path) -> None:
    audio = write_silence(tmp_path / "prepared.wav", 60)
    whisper = ScriptedWhisper([3, None])
    result = run(whisper, audio)

    # Everything finished before the failure was kept, and the retry picked up
    # just behind it rather than from the beginning.
    assert whisper.offsets[0] == 0.0
    assert 33.0 <= whisper.offsets[1] <= 34.0
    assert [item.text for item in result.segments] == [text for _, _, text in TIMELINE]
    assert [item.index for item in result.segments] == [1, 2, 3, 4, 5]
    assert [item.start_seconds for item in result.segments] == [
        start for start, _, _ in TIMELINE
    ]


def test_the_resume_slice_leaves_no_temporary_files_behind(tmp_path: Path) -> None:
    audio = write_silence(tmp_path / "prepared.wav", 60)
    run(ScriptedWhisper([2, None]), audio)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["prepared.wav"]


def test_the_job_still_finishes_when_every_gpu_rung_is_exhausted(
    tmp_path: Path,
) -> None:
    audio = write_silence(tmp_path / "prepared.wav", 60)
    ladder_length = len(build_ladder(REQUESTED))
    # Fail on every CUDA rung; only the terminal CPU rung is allowed to work.
    whisper = ScriptedWhisper([0] * (ladder_length - 1) + [None])
    result = run(whisper, audio)

    assert result.device == "cpu"
    assert len(result.segments) == len(TIMELINE)
    assert whisper.plans[-1]["device"] == "cpu"


def test_a_silent_resumed_tail_still_returns_the_earlier_transcript(
    tmp_path: Path,
) -> None:
    class SilentTail(ScriptedWhisper):
        def transcribe(self, audio: Path, **kwargs: Any) -> TranscriptionResult:
            if kwargs["time_offset_seconds"] > 0:
                raise EmptyTranscriptionError("no speech")
            return super().transcribe(audio, **kwargs)

    audio = write_silence(tmp_path / "prepared.wav", 60)
    result = run(SilentTail([2, None]), audio)
    assert [item.text for item in result.segments] == ["أول", "ثاني"]


def test_cancellation_is_not_mistaken_for_memory_pressure(tmp_path: Path) -> None:
    class Cancelling(ScriptedWhisper):
        def transcribe(self, audio: Path, **kwargs: Any) -> TranscriptionResult:
            raise TranscriptionCancelledError("Transcription was cancelled.")

    audio = write_silence(tmp_path / "prepared.wav", 60)
    with pytest.raises(TranscriptionCancelledError):
        run(Cancelling([None]), audio)


def test_dropping_word_timings_also_drops_what_would_force_them_back_on(
    tmp_path: Path,
) -> None:
    seen: list[Any] = []

    class Recording(ScriptedWhisper):
        def transcribe(self, audio: Path, **kwargs: Any) -> TranscriptionResult:
            seen.append(
                (
                    kwargs["word_timestamps"],
                    kwargs["hallucination_silence_threshold"],
                )
            )
            return super().transcribe(audio, **kwargs)

    audio = write_silence(tmp_path / "prepared.wav", 60)
    ladder = build_ladder(REQUESTED)
    first_without_words = next(
        index for index, plan in enumerate(ladder) if not plan.word_timestamps
    )
    whisper = Recording([0] * first_without_words + [None])
    run(whisper, audio, hallucination_silence_threshold=2.0)

    assert seen[0] == (True, 2.0)
    assert seen[-1] == (False, None)


def test_slicing_a_prepared_wav_reports_the_exact_cut(tmp_path: Path) -> None:
    source = write_silence(tmp_path / "prepared.wav", 10)
    destination = tmp_path / "tail.wav"
    actual = slice_wav_tail(source, destination, 4.25)

    assert actual == pytest.approx(4.25, abs=1 / 16000)
    assert wav_duration_seconds(destination) == pytest.approx(10 - actual, abs=1e-3)


def test_duration_of_an_unreadable_file_is_unknown(tmp_path: Path) -> None:
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFF")
    assert wav_duration_seconds(broken) is None


def test_estimates_track_the_levers_the_ladder_pulls() -> None:
    def estimate(**overrides: Any) -> int:
        options: dict[str, Any] = {
            "compute_type": "float16",
            "beam_size": 5,
            "word_timestamps": True,
        }
        options.update(overrides)
        return estimate_vram_bytes("large-v3", **options)

    assert estimate(beam_size=1) < estimate()
    assert estimate(compute_type="int8") < estimate()
    assert estimate(word_timestamps=False) < estimate()
    assert (
        estimate_vram_bytes(
            "small", compute_type="float16", beam_size=5, word_timestamps=True
        )
        < estimate()
    )


def test_unknown_checkpoints_are_budgeted_as_the_largest_model() -> None:
    assert resolve_profile("/models/my-finetune") == resolve_profile("large-v3")
    assert resolve_profile("large-v3-turbo").decoder_layers == 4
