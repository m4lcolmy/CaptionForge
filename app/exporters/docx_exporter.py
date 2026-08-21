"""Word (.docx) plain-text transcript exporter."""

import re
from collections.abc import Sequence
from importlib import import_module
from io import BytesIO
from typing import Any

from app.core.exceptions import DocxNotInstalledError, ExportError
from app.models.subtitle import SubtitleSegment
from app.models.video import VideoMetadata
from app.utils.language_utils import is_right_to_left

PARAGRAPH_GAP_SECONDS = 4
SOFT_CHARACTER_LIMIT = 450
HARD_CHARACTER_LIMIT = 1200
_SENTENCE_ENDINGS = (".", "!", "?", "؟", "۔", "…", "。", "！", "？")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WHITESPACE = re.compile(r"\s+")


def group_paragraphs(
    segments: Sequence[SubtitleSegment],
    *,
    gap_seconds: float = PARAGRAPH_GAP_SECONDS,
    soft_limit: int = SOFT_CHARACTER_LIMIT,
    hard_limit: int = HARD_CHARACTER_LIMIT,
) -> tuple[str, ...]:
    """Join cue-sized segments into readable prose paragraphs.

    A paragraph ends at a silent gap, at a sentence boundary once it is long
    enough to read, or at a hard length limit for text without punctuation.
    """
    paragraphs: list[str] = []
    buffer: list[str] = []
    length = 0
    previous_end: float | None = None

    def flush() -> None:
        nonlocal buffer, length
        if buffer:
            paragraphs.append(" ".join(buffer))
        buffer = []
        length = 0

    for segment in segments:
        text = _normalize(segment.text)
        if not text:
            continue
        if (
            buffer
            and previous_end is not None
            and segment.start_seconds - previous_end >= gap_seconds
        ):
            flush()
        buffer.append(text)
        length += len(text) + 1
        previous_end = segment.end_seconds
        if length >= hard_limit or (
            length >= soft_limit and text.endswith(_SENTENCE_ENDINGS)
        ):
            flush()
    flush()
    return tuple(paragraphs)


def render_docx(
    video: VideoMetadata,
    segments: Sequence[SubtitleSegment],
    *,
    language: str | None = None,
    include_title: bool = True,
) -> bytes:
    """Render segments as a Word document containing only plain transcript text."""
    docx = _require_docx()
    shared = import_module("docx.shared")
    alignment = import_module("docx.enum.text").WD_ALIGN_PARAGRAPH
    resolved_language = language or (segments[0].language if segments else None)
    right_to_left = is_right_to_left(resolved_language)

    document = docx.Document()
    _configure_normal_style(document, shared.Pt, right_to_left)
    if include_title:
        heading = document.add_heading(_normalize(video.title) or "Transcript", level=1)
        _apply_direction(heading, right_to_left, alignment)
    for text in group_paragraphs(segments):
        _apply_direction(document.add_paragraph(text), right_to_left, alignment)
    document.core_properties.title = video.title

    buffer = BytesIO()
    try:
        document.save(buffer)
    except (OSError, ValueError) as exc:
        raise ExportError(
            "The Word document could not be generated.", details=str(exc)
        ) from exc
    return buffer.getvalue()


def _require_docx() -> Any:
    """Import python-docx, translating a missing dependency into a clear error."""
    try:
        return import_module("docx")
    except ImportError as exc:
        raise DocxNotInstalledError(
            "Word (.docx) export needs the python-docx package. "
            "Install it with: pip install python-docx",
            details=str(exc),
        ) from exc


def _normalize(value: str) -> str:
    """Collapse cue line breaks and drop characters Word cannot store."""
    return _WHITESPACE.sub(" ", _CONTROL_CHARACTERS.sub("", value)).strip()


def _configure_normal_style(document: Any, pt: Any, right_to_left: bool) -> None:
    """Give the transcript a readable body style, including complex scripts."""
    style = document.styles["Normal"]
    style.font.size = pt(12)
    style.paragraph_format.space_after = pt(10)
    if right_to_left:
        properties = style.element.get_or_add_rPr()
        _append_property(properties, "w:rtl", "1")
        _append_property(properties, "w:szCs", "24")


def _apply_direction(paragraph: Any, right_to_left: bool, alignment: Any) -> None:
    """Mark a paragraph and its runs right-to-left so Arabic renders correctly."""
    if not right_to_left:
        return
    _append_property(paragraph._p.get_or_add_pPr(), "w:bidi", "1")
    paragraph.alignment = alignment.RIGHT
    for run in paragraph.runs:
        _append_property(run._r.get_or_add_rPr(), "w:rtl", "1")


def _append_property(properties: Any, tag: str, value: str) -> None:
    """Append one raw WordprocessingML property element."""
    oxml = import_module("docx.oxml")
    qualified_name = import_module("docx.oxml.ns").qn
    element = oxml.OxmlElement(tag)
    element.set(qualified_name("w:val"), value)
    properties.append(element)
