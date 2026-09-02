"""Guards for transcription decoding options and word-aligned splitting."""

import inspect
from pathlib import Path
from typing import Any

import pytest

from app.adapters.whisper_adapter import WhisperAdapter
from app.core.config import Config
from app.models.subtitle import SubtitleSegment
from app.models.transcription import WordTiming
from app.services.postprocessing_service import PostProcessingService


class FakeWord:
    def __init__(self, start: float, end: float, word: str) -> None:
        self.start = start
        self.end = end
        self.word = word
        self.probability = 0.9


class FakeSegment:
    def __init__(self, start: float, end: float, text: str, words: list[FakeWord]):
        self.start = start
        self.end = end
        self.text = text
        self.words = words
        self.avg_logprob = -0.2
        self.no_speech_prob = 0.01


class FakeInfo:
    language = "ar"
    language_probability = 0.99
    duration = 10.0


class RecordingModel:
    """Model stub that records the decode options it was given."""

    kwargs: dict[str, Any] = {}

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name

    def transcribe(self, _audio: str, **kwargs: Any) -> tuple[Any, Any]:
        RecordingModel.kwargs = kwargs
        words = [FakeWord(0.0, 1.0, "مرحبا"), FakeWord(4.0, 5.0, "بالعالم")]
        return iter([FakeSegment(0.0, 5.0, "مرحبا بالعالم", words)]), FakeInfo()


def _transcribe(**overrides: Any) -> Any:
    adapter = WhisperAdapter(model_factory=RecordingModel, cuda_detector=lambda: False)
    return adapter.transcribe(Path("audio.wav"), model_name="small", **overrides)


def test_long_form_decoding_defaults_are_applied() -> None:
    """Repetition/hallucination guards must reach the engine."""
    _transcribe(initial_prompt="الأندلس", hallucination_silence_threshold=2.0)

    sent = RecordingModel.kwargs
    assert sent["condition_on_previous_text"] is False
    assert sent["initial_prompt"] == "الأندلس"
    assert sent["hallucination_silence_threshold"] == 2.0
    assert sent["compression_ratio_threshold"] == 2.4
    assert sent["no_speech_threshold"] == 0.6


def test_hallucination_threshold_forces_word_timestamps() -> None:
    """The engine only honours the threshold alongside word alignment."""
    _transcribe(word_timestamps=False, hallucination_silence_threshold=2.0)

    assert RecordingModel.kwargs["word_timestamps"] is True


def test_vad_parameters_are_forwarded() -> None:
    """Padding and threshold matter for clipped word onsets."""
    _transcribe(vad_speech_pad_ms=250, vad_threshold=0.4)

    assert RecordingModel.kwargs["vad_parameters"] == {
        "min_silence_duration_ms": 500,
        "threshold": 0.4,
        "speech_pad_ms": 250,
    }


def test_word_timings_are_captured() -> None:
    """Engine alignments must survive into the domain model."""
    result = _transcribe()

    words = result.segments[0].words
    assert [word.text for word in words] == ["مرحبا", "بالعالم"]
    assert words[0].start_seconds == 0.0
    assert words[1].end_seconds == 5.0


def _segment(text: str, words: tuple[WordTiming, ...]) -> SubtitleSegment:
    return SubtitleSegment(
        index=1,
        start_seconds=0.0,
        end_seconds=10.0,
        text=text,
        language="ar",
        words=words,
    )


def _four_word_segment(words: tuple[WordTiming, ...]) -> SubtitleSegment:
    """One cue whose words are spoken in two bursts around a four-second pause."""
    return _segment("بسم الله الرحمن الرحيم", words)


ALIGNED_WORDS = tuple(
    WordTiming(start_seconds=s, end_seconds=e, text=t)
    for s, e, t in [
        (0.0, 0.5, "بسم"),
        (0.5, 1.0, "الله"),
        (5.0, 5.5, "الرحمن"),
        (5.5, 6.0, "الرحيم"),
    ]
)


def test_split_lands_on_real_word_boundaries() -> None:
    """Cues must sit on the spoken words and leave the pause empty."""
    service = PostProcessingService(Config(maximum_characters_per_line=7))

    result = service.process((_four_word_segment(ALIGNED_WORDS),))

    assert len(result) == 2
    assert (result[0].start_seconds, result[0].end_seconds) == (0.0, 1.0)
    assert (result[1].start_seconds, result[1].end_seconds) == (5.0, 6.0)
    # The four-second silence is preserved, not filled by a stretched cue.
    assert result[1].start_seconds > result[0].end_seconds


def test_split_falls_back_to_proportional_timing_without_alignment() -> None:
    """Without word data the old behaviour applies: contiguous, guessed cuts."""
    service = PostProcessingService(Config(maximum_characters_per_line=7))

    result = service.process((_four_word_segment(()),))

    assert len(result) == 2
    assert result[0].end_seconds == result[1].start_seconds
    assert result[1].start_seconds < 5.0
    assert all(item.words == () for item in result)


def test_split_ignores_alignment_that_does_not_match_the_text() -> None:
    """A token-count mismatch must degrade, never misplace text."""
    stale = (WordTiming(start_seconds=0.0, end_seconds=0.5, text="بسم"),)
    service = PostProcessingService(Config(maximum_characters_per_line=7))

    result = service.process((_four_word_segment(stale),))

    assert result[0].end_seconds == result[1].start_seconds
    assert all(item.words == () for item in result)


def test_merged_cues_keep_a_consistent_alignment() -> None:
    """Merging must either combine both alignments or discard them."""
    service = PostProcessingService(Config(minimum_subtitle_duration=2.0))
    first = SubtitleSegment(
        index=1,
        start_seconds=0.0,
        end_seconds=0.4,
        text="بسم",
        language="ar",
        words=(WordTiming(start_seconds=0.0, end_seconds=0.4, text="بسم"),),
    )
    second = SubtitleSegment(
        index=2,
        start_seconds=0.5,
        end_seconds=0.9,
        text="الله",
        language="ar",
        words=(WordTiming(start_seconds=0.5, end_seconds=0.9, text="الله"),),
    )

    result = service.process((first, second))

    assert len(result) == 1
    assert len(result[0].words) == len(result[0].text.split())


def test_decode_options_match_the_installed_engine_signature() -> None:
    """Catch API drift: every option we send must exist on faster-whisper."""
    engine = pytest.importorskip("faster_whisper.transcribe")
    accepted = set(inspect.signature(engine.WhisperModel.transcribe).parameters)

    _transcribe(initial_prompt="x", hallucination_silence_threshold=2.0)

    unknown = set(RecordingModel.kwargs) - accepted
    assert not unknown, f"faster-whisper does not accept: {sorted(unknown)}"
    assert set(engine.VadOptions.__dataclass_fields__) >= set(
        RecordingModel.kwargs["vad_parameters"]
    )
