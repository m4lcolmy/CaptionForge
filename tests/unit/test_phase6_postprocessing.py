"""Phase 6 conservative Arabic subtitle post-processing coverage."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.core.config import Config
from app.interfaces.cli import app
from app.models.subtitle import (
    RawSubtitle,
    SubtitleSegment,
    SubtitleSourceType,
    SubtitleTrack,
)
from app.models.transcription import WordTiming
from app.services.postprocessing_service import PostProcessingService
from app.services.subtitle_service import SubtitleService


def segment(
    index: int,
    start: float,
    end: float,
    text: str,
    **kwargs: object,
) -> SubtitleSegment:
    return SubtitleSegment(
        index=index,
        start_seconds=start,
        end_seconds=end,
        text=text,
        language="ar",
        **kwargs,
    )


def test_arabic_whitespace_and_punctuation_preserve_mixed_text() -> None:
    result = PostProcessingService().process(
        (segment(1, 0, 2, "  مَرْحَبًا   يا OpenAI  ، كيف حالك ؟  "),)
    )
    assert result[0].text == "مَرْحَبًا يا OpenAI، كيف حالك؟"


def test_latin_punctuation_spacing_is_safe() -> None:
    result = PostProcessingService().process(
        (segment(1, 0, 2, "Hello ,world! Version 3.14"),)
    )
    assert result[0].text == "Hello, world! Version 3.14"


def _repetition_candidates() -> list[tuple[float, float, str]]:
    return [
        (0, 1, " "),
        (1, 2, "شكرا لكم شكرا لكم"),
        (2, 3, "شكرا لكم"),
    ]


def test_empty_and_redundant_cues_are_removed_without_rewriting_speech() -> None:
    """Blank and fully-contained cues go; repeated words are left alone."""
    result = PostProcessingService().process_candidates(_repetition_candidates(), "ar")

    assert [item.text for item in result] == ["شكرا لكم شكرا لكم"]


def test_repeated_phrase_collapse_is_opt_in() -> None:
    """Collapsing repetition rewrites speech, so it must be requested."""
    service = PostProcessingService(Config(collapse_repeated_phrases=True))

    result = service.process_candidates(_repetition_candidates(), "ar")

    assert [item.text for item in result] == ["شكرا لكم"]


def test_overlap_invalid_duration_and_short_fragment_merging() -> None:
    service = PostProcessingService(
        Config(minimum_subtitle_duration=0.8, subtitle_merge_threshold=0.5)
    )
    result = service.process_candidates(
        [(0, 0.2, "هذا نص"), (0.3, 0.5, "قصير"), (0.4, 0.2, "ومفيد")],
        "ar",
    )
    assert result[0].text == "هذا نص قصير ومفيد"
    assert result[0].end_seconds > result[0].start_seconds
    assert result[0].end_seconds - result[0].start_seconds >= 0.8


def test_long_segments_split_and_use_at_most_two_short_lines() -> None:
    config = Config(
        maximum_characters_per_line=16,
        maximum_subtitle_lines=2,
        maximum_subtitle_duration=3,
        minimum_subtitle_duration=0.2,
    )
    text = "هذه جملة عربية طويلة جدا. وتحتوي على كلمات كثيرة لتقسيمها بأمان."
    result = PostProcessingService(config).process((segment(1, 0, 8, text),))
    assert len(result) >= 3
    assert all(item.end_seconds - item.start_seconds <= 3 for item in result)
    assert all(len(item.text.splitlines()) <= 2 for item in result)
    assert all(len(line) <= 16 for item in result for line in item.text.splitlines())


def test_quranic_names_diacritics_and_digits_preserved_by_default() -> None:
    text = "قال الله تعالى: إِنَّا أَعْطَيْنَاكَ الْكَوْثَرَ، وقال مُحَمَّد ١٢ مرة."
    result = PostProcessingService().process((segment(1, 0, 5, text),))
    assert "إِنَّا أَعْطَيْنَاكَ الْكَوْثَرَ" in result[0].text.replace("\n", " ")
    assert "مُحَمَّد ١٢" in " ".join(item.text.replace("\n", " ") for item in result)


def test_risky_normalization_is_explicitly_configurable() -> None:
    service = PostProcessingService(
        Config(
            remove_diacritics=True,
            normalize_arabic_letters=True,
            normalize_arabic_indic_digits=True,
        )
    )
    result = service.process((segment(1, 0, 2, "إِنَّ ١٢"),))
    assert result[0].text == "ان 12"


def test_safe_silence_hallucination_removal() -> None:
    result = PostProcessingService().process(
        (
            segment(1, 0, 1, "موسيقى", no_speech_probability=0.98),
            segment(2, 1, 2, "الحمد لله", no_speech_probability=0.98),
        )
    )
    assert [item.text for item in result] == ["الحمد لله"]


def test_postprocessing_can_be_disabled_for_parsed_captions() -> None:
    track = SubtitleTrack(
        language_code="ar",
        normalized_language_code="ar",
        source_type=SubtitleSourceType.MANUAL,
        is_automatic=False,
    )
    segments = SubtitleService().parse_and_clean(
        RawSubtitle(
            format="srt",
            content="1\n00:00:00,000 --> 00:00:01,000\nمرحبا  ،  بالعالم\n",
        ),
        track,
        postprocess=False,
    )
    assert segments[0].text == "مرحبا ، بالعالم"
    assert segments[0].end_seconds == 1


@pytest.mark.parametrize("extension,header", [("srt", ""), ("vtt", "WEBVTT\n\n")])
def test_clean_command_supports_srt_and_vtt(
    tmp_path: Path, extension: str, header: str
) -> None:
    separator = "," if extension == "srt" else "."
    source = tmp_path / f"input.{extension}"
    timing = f"00:00:00{separator}000 --> 00:00:01{separator}000"
    source.write_text(
        f"{header}1\n{timing}\nمرحبا  ،  بالعالم\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["clean", str(source)])
    destination = tmp_path / f"input.cleaned.{extension}"
    assert result.exit_code == 0, result.output
    assert destination.exists()
    assert "مرحبا، بالعالم" in destination.read_text(encoding="utf-8")


def _timed_words(count: int, step: float = 0.6) -> tuple[WordTiming, ...]:
    """Words that run end to end, so the true speech end is unambiguous."""
    return tuple(
        WordTiming(
            text=f"w{index}",
            start_seconds=index * step,
            end_seconds=index * step + step - 0.05,
        )
        for index in range(count)
    )


def test_long_utterance_is_split_not_truncated() -> None:
    """A cue longer than the maximum must be divided, keeping its real end.

    Clamping the end before _split_long ran left the caption ending seconds
    before the speaker stopped, with the whole text crammed into one short cue.
    """
    words = _timed_words(20)
    speech_ends = words[-1].end_seconds
    source = SubtitleSegment(
        index=1,
        start_seconds=0.0,
        end_seconds=12.0,
        text=" ".join(word.text for word in words),
        language="ar",
        words=words,
    )

    result = PostProcessingService(Config()).process([source])

    assert len(result) > 1
    assert result[-1].end_seconds >= speech_ends
    for cue in result:
        assert cue.end_seconds - cue.start_seconds <= Config().maximum_subtitle_duration


def test_split_cues_land_on_real_word_boundaries() -> None:
    """Every boundary should match a word timing, not a character estimate."""
    words = _timed_words(20)
    source = SubtitleSegment(
        index=1,
        start_seconds=0.0,
        end_seconds=12.0,
        text=" ".join(word.text for word in words),
        language="ar",
        words=words,
    )

    result = PostProcessingService(Config()).process([source])
    boundaries = {round(word.end_seconds, 2) for word in words}

    assert round(result[0].end_seconds, 2) in boundaries


def test_short_cue_ends_on_the_last_word_not_the_padded_segment() -> None:
    """A cue within the limit stays whole and tightens onto real speech.

    VAD padding leaves the engine's segment end slightly after the last word,
    so snapping to the word boundary is the more accurate ending.
    """
    words = _timed_words(4)
    source = SubtitleSegment(
        index=1,
        start_seconds=0.0,
        end_seconds=2.4,
        text=" ".join(word.text for word in words),
        language="ar",
        words=words,
    )

    result = PostProcessingService(Config()).process([source])

    assert len(result) == 1
    assert result[0].end_seconds == pytest.approx(words[-1].end_seconds)
    assert result[0].end_seconds <= 2.4
