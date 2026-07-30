"""Conservative caption text cleanup that preserves Arabic orthography."""

import html
import re

_TAG = re.compile(r"<[^>]+>")
_BRACKETED_CUE = re.compile(r"^\s*(?:\[[^\]]+\]|\([^)]+\))\s*$")
_WHITESPACE = re.compile(r"\s+")
_ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([،؛؟,.!?:;٪%])")
_SPACE_AFTER_PUNCTUATION = re.compile(r"([،؛؟,.!?:;])(?=[^\s،؛؟,.!?:;])")
_ARABIC_LETTER_TRANSLATION = str.maketrans(
    {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"}
)
_ARABIC_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def clean_caption_text(
    value: str,
    *,
    remove_diacritics: bool = False,
    normalize_arabic_letters: bool = False,
    normalize_arabic_indic_digits: bool = False,
    fix_punctuation: bool = True,
) -> str:
    """Remove caption markup and normalize whitespace without changing letters."""
    text = html.unescape(value.replace("\u200b", ""))
    text = _TAG.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if _BRACKETED_CUE.fullmatch(text):
        return ""
    if fix_punctuation:
        text = fix_punctuation_spacing(text)
    if remove_diacritics:
        text = _ARABIC_DIACRITICS.sub("", text)
    if normalize_arabic_letters:
        text = text.translate(_ARABIC_LETTER_TRANSLATION)
    if normalize_arabic_indic_digits:
        text = text.translate(_ARABIC_DIGIT_TRANSLATION)
    return text


def fix_punctuation_spacing(value: str) -> str:
    """Normalize safe spacing around common Arabic and Latin punctuation."""
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", value)

    def add_space(match: re.Match[str]) -> str:
        punctuation = match.group(1)
        next_character = text[match.end()]
        previous_character = text[match.start() - 1] if match.start() else ""
        if (
            punctuation in {".", ",", ":"}
            and previous_character.isdigit()
            and next_character.isdigit()
        ):
            return punctuation
        return f"{punctuation} "

    return _SPACE_AFTER_PUNCTUATION.sub(add_space, text)


def comparison_key(value: str, *, remove_diacritics: bool = False) -> str:
    """Build a conservative key for duplicate comparison."""
    text = clean_caption_text(value)
    if remove_diacritics:
        text = _ARABIC_DIACRITICS.sub("", text)
    return text.casefold()
