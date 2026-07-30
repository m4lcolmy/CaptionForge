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


def test_empty_exact_duplicates_and_repeated_whisper_phrases() -> None:
    result = PostProcessingService().process_candidates(
        [
            (0, 1, " "),
            (1, 2, "شكرا لكم شكرا لكم"),
            (2, 3, "شكرا لكم"),
        ],
        "ar",
    )
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
