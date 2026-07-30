"""Conservative subtitle post-processing shared by every source."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.core.config import Config
from app.models.subtitle import SubtitleSegment
from app.utils.arabic_text_utils import clean_caption_text, comparison_key
from app.utils.time_utils import distribute_duration

_SENTENCE_END = re.compile(r"[.!?؟؛…][\"'»”)]?$")
_SAFE_SILENCE_CUE = re.compile(
    r"^\s*[\[(]?\s*(?:music|applause|silence|موسيقى|تصفيق|صمت)\s*[\])]?\s*$",
    re.IGNORECASE,
)


@dataclass
class _Item:
    start: float
    end: float
    text: str
    language: str
    speaker: str | None = None
    confidence: float | None = None
    no_speech_probability: float | None = None


class PostProcessingService:
    """Clean timing and text without rewriting the speaker's words."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def process(
        self, segments: Sequence[SubtitleSegment]
    ) -> tuple[SubtitleSegment, ...]:
        """Return one normalized, exporter-ready subtitle sequence."""
        items = [
            _Item(
                item.start_seconds,
                item.end_seconds,
                item.text,
                item.language,
                item.speaker,
                item.confidence,
                item.no_speech_probability,
            )
            for item in segments
        ]
        return self._process(items)

    def process_candidates(
        self,
        candidates: Iterable[tuple[float, float, str]],
        language: str,
    ) -> tuple[SubtitleSegment, ...]:
        """Clean parser output before strict subtitle model validation."""
        return self._process(
            [_Item(start, end, text, language) for start, end, text in candidates]
        )

    def _process(self, items: list[_Item]) -> tuple[SubtitleSegment, ...]:
        items = self._clean_text(items)
        items = self._remove_duplicates(items)
        items = self._repair_timing(items, enforce_duration=False)
        items = self._merge_short(items)
        items = self._split_long(items)
        items = self._repair_timing(items)
        return tuple(
            SubtitleSegment(
                index=index,
                start_seconds=item.start,
                end_seconds=item.end,
                text=self._wrap(item.text),
                language=item.language,
                speaker=item.speaker,
                confidence=item.confidence,
                no_speech_probability=item.no_speech_probability,
            )
            for index, item in enumerate(items, 1)
        )

    def _clean_text(self, items: list[_Item]) -> list[_Item]:
        result = []
        for item in items:
            text = clean_caption_text(
                item.text,
                remove_diacritics=self.config.remove_diacritics,
                normalize_arabic_letters=self.config.normalize_arabic_letters,
                normalize_arabic_indic_digits=self.config.normalize_arabic_indic_digits,
            )
            if not text:
                continue
            text = self._collapse_repeated_phrase(text)
            if (
                item.no_speech_probability is not None
                and item.no_speech_probability >= 0.9
                and _SAFE_SILENCE_CUE.fullmatch(text)
            ):
                continue
            item.text = text
            result.append(item)
        return result

    @staticmethod
    def _collapse_repeated_phrase(text: str) -> str:
        """Collapse an adjacent repeated phrase of two or more words."""
        words = text.split()
        for size in range(len(words) // 2, 1, -1):
            index = 0
            while index + size * 2 <= len(words):
                first = [word.casefold() for word in words[index : index + size]]
                second = [
                    word.casefold() for word in words[index + size : index + size * 2]
                ]
                if first == second:
                    del words[index + size : index + size * 2]
                else:
                    index += 1
        return " ".join(words)

    def _remove_duplicates(self, items: list[_Item]) -> list[_Item]:
        result: list[_Item] = []
        for item in items:
            if not result:
                result.append(item)
                continue
            previous = result[-1]
            current_key = comparison_key(item.text)
            previous_key = comparison_key(previous.text)
            if current_key == previous_key:
                previous.end = max(previous.end, item.end)
                continue
            similarity = SequenceMatcher(None, previous_key, current_key).ratio()
            overlap = self._phrase_overlap(previous.text, item.text)
            if (
                similarity >= self.config.duplicate_detection_threshold
                or overlap is not None
            ):
                if overlap:
                    item.text = overlap
                elif len(item.text) <= len(previous.text):
                    previous.end = max(previous.end, item.end)
                    continue
            result.append(item)
        return result

    @staticmethod
    def _phrase_overlap(previous: str, current: str) -> str | None:
        """Remove a repeated leading phrase only when at least two words match."""
        left, right = previous.split(), current.split()
        maximum = min(len(left), len(right))
        for size in range(maximum, 1, -1):
            if [word.casefold() for word in left[-size:]] == [
                word.casefold() for word in right[:size]
            ]:
                remainder = right[size:]
                return " ".join(remainder) if remainder else None
        return None

    def _repair_timing(
        self, items: list[_Item], *, enforce_duration: bool = True
    ) -> list[_Item]:
        result: list[_Item] = []
        for item in items:
            item.start = max(0.0, item.start)
            if result and item.start < result[-1].end:
                if item.start > result[-1].start:
                    result[-1].end = item.start
                else:
                    item.start = result[-1].end
            minimum = (
                self.config.minimum_subtitle_duration if enforce_duration else 0.001
            )
            item.end = max(item.end, item.start + minimum)
            item.end = min(item.end, item.start + self.config.maximum_subtitle_duration)
            result.append(item)
        for index in range(len(result) - 1):
            if result[index].end > result[index + 1].start:
                result[index].end = max(
                    result[index].start + 0.001, result[index + 1].start
                )
        return result

    def _merge_short(self, items: list[_Item]) -> list[_Item]:
        result: list[_Item] = []
        limit = (
            self.config.maximum_characters_per_line * self.config.maximum_subtitle_lines
        )
        for item in items:
            if result:
                previous = result[-1]
                gap = item.start - previous.end
                combined = f"{previous.text} {item.text}"
                should_merge = (
                    gap <= self.config.subtitle_merge_threshold
                    and len(combined) <= limit
                    and (
                        previous.end - previous.start
                        < self.config.minimum_subtitle_duration
                        or item.end - item.start < self.config.minimum_subtitle_duration
                        or len(previous.text.split()) == 1
                        or len(item.text.split()) == 1
                    )
                    and not _SENTENCE_END.search(previous.text)
                )
                if should_merge:
                    previous.text = combined
                    previous.end = min(
                        max(previous.end, item.end),
                        previous.start + self.config.maximum_subtitle_duration,
                    )
                    continue
            result.append(item)
        return result

    def _split_long(self, items: list[_Item]) -> list[_Item]:
        result: list[_Item] = []
        limit = (
            self.config.maximum_characters_per_line * self.config.maximum_subtitle_lines
        )
        for item in items:
            parts = self._text_parts(item.text, limit)
            max_duration = self.config.maximum_subtitle_duration
            duration_parts = max(1, int((item.end - item.start) / max_duration + 0.999))
            target_parts = max(len(parts), duration_parts)
            if target_parts > len(parts):
                target_length = max(1, len(item.text) // target_parts)
                parts = self._text_parts(item.text, target_length)
            timings = distribute_duration(
                item.start, item.end, [max(1, len(part)) for part in parts]
            )
            for part, (start, end) in zip(parts, timings, strict=True):
                result.append(
                    _Item(
                        start,
                        end,
                        part,
                        item.language,
                        item.speaker,
                        item.confidence,
                        item.no_speech_probability,
                    )
                )
        return result

    @staticmethod
    def _text_parts(text: str, limit: int) -> list[str]:
        if len(text) <= limit:
            return [text]
        words = text.split()
        parts: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if current and len(candidate) > limit:
                parts.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
            near_limit = len(" ".join(current)) >= limit // 2
            if current and _SENTENCE_END.search(current[-1]) and near_limit:
                parts.append(" ".join(current))
                current = []
        if current:
            fits_previous = parts and len(parts[-1]) + len(current[0]) + 1 <= limit
            if len(current) == 1 and fits_previous:
                parts[-1] += f" {current[0]}"
            else:
                parts.append(" ".join(current))
        return parts

    def _wrap(self, text: str) -> str:
        width = self.config.maximum_characters_per_line
        if len(text) <= width:
            return text
        words = text.split()
        best = min(
            range(1, len(words)),
            key=lambda index: abs(
                len(" ".join(words[:index])) - len(" ".join(words[index:]))
            ),
            default=0,
        )
        first_fits = best and len(" ".join(words[:best])) <= width
        second_fits = best and len(" ".join(words[best:])) <= width
        if first_fits and second_fits:
            return f"{' '.join(words[:best])}\n{' '.join(words[best:])}"
        return text
