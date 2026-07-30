"""Tests for language identifier utilities."""

import pytest

from app.utils.language_utils import (
    base_language,
    language_name,
    normalize_language_code,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ar", "ar"),
        ("AR", "ar"),
        ("ar_eg", "ar-EG"),
        ("ar-eg", "ar-EG"),
        ("en_us", "en-US"),
        (" AR-sa ", "ar-SA"),
    ],
)
def test_normalization(value: str, expected: str) -> None:
    """Language and region casing should be canonical."""
    assert normalize_language_code(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("ar-EG", "ar"), ("en-US", "en"), ("ar", "ar")],
)
def test_base_language(value: str, expected: str) -> None:
    """Base language extraction should work for generic and regional codes."""
    assert base_language(value) == expected


@pytest.mark.parametrize("value", [None, "", "1", "a", "ar--EG", "@@"])
def test_malformed_language_values_are_safe(value: str | None) -> None:
    """Malformed codes should return None rather than raising."""
    assert normalize_language_code(value) is None
    assert base_language(value) is None


def test_known_and_unknown_language_names() -> None:
    """The small built-in display mapping should permit unknown codes."""
    assert language_name("ar-EG") == "Arabic"
    assert language_name("zu") is None
