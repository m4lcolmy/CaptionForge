"""Tests for safe UTF-8 multi-format subtitle export."""

import json
from pathlib import Path

import pytest

from app.core.exceptions import ExportError
from app.models.subtitle import SubtitleSegment
from app.services.export_service import ExportService
from app.utils.file_utils import sanitize_filename
from tests.conftest import make_track


@pytest.fixture
def segments() -> tuple[SubtitleSegment, ...]:
    return (
        SubtitleSegment(
            index=1,
            start_seconds=1.25,
            end_seconds=2.5,
            text="مرحباً بالعالم",
            language="ar",
        ),
        SubtitleSegment(
            index=2,
            start_seconds=3,
            end_seconds=4,
            text="السطر الثاني",
            language="ar",
        ),
    )


def test_filename_sanitization_preserves_unicode() -> None:
    assert sanitize_filename('  فيديو: "تجربة" / test?  ') == "فيديو_ _تجربة_ _ test_"
    assert sanitize_filename("CON") == "captions"


def test_all_exports_and_arabic_utf8(
    tmp_path: Path, video_metadata: object, segments: tuple[SubtitleSegment, ...]
) -> None:
    video = video_metadata.model_copy(update={"title": "فيديو عربي"})  # type: ignore[attr-defined]
    paths = ExportService().export(
        video, make_track("ar"), segments, ("srt", "vtt", "txt", "json"), tmp_path
    )
    outputs = {path.suffix: path.read_text(encoding="utf-8") for path in paths}

    assert "00:00:01,250 --> 00:00:02,500" in outputs[".srt"]
    assert outputs[".vtt"].startswith("WEBVTT\n")
    assert outputs[".txt"] == "مرحباً بالعالم\nالسطر الثاني\n"
    payload = json.loads(outputs[".json"])
    assert payload["video"]["video_id"] == video.video_id
    assert payload["selected_language"] == "ar"
    assert payload["caption_source_type"] == "manual"
    assert payload["segments"][0]["text"] == "مرحباً بالعالم"


def test_timestamped_txt(
    tmp_path: Path, video_metadata: object, segments: tuple[SubtitleSegment, ...]
) -> None:
    path = ExportService().export(
        video_metadata,  # type: ignore[arg-type]
        make_track("ar"),
        segments,
        ("txt",),
        tmp_path,
        timestamped_txt=True,
    )[0]

    assert path.read_text(encoding="utf-8").startswith("[00:00:01.250]\t")


def test_existing_file_requires_overwrite(
    tmp_path: Path, video_metadata: object, segments: tuple[SubtitleSegment, ...]
) -> None:
    service = ExportService()
    service.export(
        video_metadata,
        make_track("ar"),
        segments,
        ("srt",),
        tmp_path,  # type: ignore[arg-type]
    )
    with pytest.raises(ExportError, match="already exists"):
        service.export(
            video_metadata,  # type: ignore[arg-type]
            make_track("ar"),
            segments,
            ("srt",),
            tmp_path,
        )
    service.export(
        video_metadata,  # type: ignore[arg-type]
        make_track("ar"),
        segments,
        ("srt",),
        tmp_path,
        overwrite=True,
    )


def test_invalid_output_directory(
    tmp_path: Path, video_metadata: object, segments: tuple[SubtitleSegment, ...]
) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("x")

    with pytest.raises(ExportError, match="directory"):
        ExportService().export(
            video_metadata,  # type: ignore[arg-type]
            make_track("ar"),
            segments,
            ("srt",),
            file_path,
        )
