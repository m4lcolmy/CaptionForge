"""Language-code normalization and display helpers."""

import re

_LANGUAGE_PART = re.compile(r"^[A-Za-z]{2,3}$")
_REGION_PART = re.compile(r"^(?:[A-Za-z]{2}|\d{3})$")
_RIGHT_TO_LEFT_LANGUAGES = frozenset(
    {"ar", "arc", "az", "dv", "fa", "he", "iw", "ku", "ps", "sd", "ug", "ur", "yi"}
)
_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "tr": "Turkish",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}


def normalize_language_code(value: str | None) -> str | None:
    """Normalize a language identifier without raising on malformed input."""
    if not isinstance(value, str):
        return None
    parts = value.strip().replace("_", "-").split("-")
    if not parts or not _LANGUAGE_PART.fullmatch(parts[0]):
        return None
    normalized = [parts[0].lower()]
    for position, part in enumerate(parts[1:], start=1):
        if not part or not part.isalnum():
            return None
        if position == 1 and _REGION_PART.fullmatch(part):
            normalized.append(part.upper())
        elif len(part) == 4 and part.isalpha():
            normalized.append(part.title())
        else:
            normalized.append(part.lower())
    return "-".join(normalized)


def base_language(value: str | None) -> str | None:
    """Return the normalized base language from a language identifier."""
    normalized = normalize_language_code(value)
    return normalized.split("-", maxsplit=1)[0] if normalized else None


def language_name(value: str | None) -> str | None:
    """Return a small built-in display name for a language code."""
    base = base_language(value)
    return _LANGUAGE_NAMES.get(base) if base else None


def is_right_to_left(value: str | None) -> bool:
    """Return whether a language identifier is written right-to-left."""
    base = base_language(value)
    return base in _RIGHT_TO_LEFT_LANGUAGES if base else False
