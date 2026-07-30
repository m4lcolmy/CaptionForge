"""Tests for deterministic subtitle selection."""

import pytest

from app.models.subtitle import SubtitleSourceType
from app.services.subtitle_service import SubtitleService
from tests.conftest import make_track


@pytest.mark.parametrize(
    ("manual_codes", "automatic_codes", "preferred", "expected", "source"),
    [
        (["ar-EG"], [], "ar-EG", "ar-EG", SubtitleSourceType.MANUAL),
        (["ar-SA"], [], "ar-EG", "ar-SA", SubtitleSourceType.MANUAL),
        ([], ["ar-EG"], "ar-EG", "ar-EG", SubtitleSourceType.AUTOMATIC),
        ([], ["ar-SA"], "ar-EG", "ar-SA", SubtitleSourceType.AUTOMATIC),
        (
            ["ar-SA"],
            ["ar-EG"],
            "ar-EG",
            "ar-SA",
            SubtitleSourceType.MANUAL,
        ),
        (["en"], ["tr"], "ar", None, None),
        ([], [], "ar", None, None),
    ],
)
def test_selection_priority(
    manual_codes: list[str],
    automatic_codes: list[str],
    preferred: str,
    expected: str | None,
    source: SubtitleSourceType | None,
) -> None:
    """Selection must honor manual-first exact/base priorities."""
    manual = [make_track(code) for code in manual_codes]
    automatic = [
        make_track(code, SubtitleSourceType.AUTOMATIC) for code in automatic_codes
    ]

    selected, _ = SubtitleService().select_track(manual, automatic, preferred)

    assert (selected.normalized_language_code if selected else None) == expected
    assert (selected.source_type if selected else None) == source


def test_generic_base_code_wins_equal_priority() -> None:
    """A generic base track should beat regional alternatives."""
    tracks = [make_track("ar-SA"), make_track("ar"), make_track("ar-EG")]

    selected, _ = SubtitleService().select_track(tracks, [], "ar-IQ")

    assert selected and selected.normalized_language_code == "ar"


def test_equal_regional_tracks_sort_deterministically() -> None:
    """Regional ties should use normalized language-code ordering."""
    tracks = [make_track("ar-SA"), make_track("ar-EG"), make_track("ar-AE")]

    selected, _ = SubtitleService().select_track(tracks, [], "ar-IQ")

    assert selected and selected.normalized_language_code == "ar-AE"


def test_case_and_underscore_preference_is_normalized() -> None:
    """Preferred language input normalization should be selection-independent."""
    selected, match = SubtitleService().select_track([make_track("ar-EG")], [], "AR_eg")

    assert selected and selected.normalized_language_code == "ar-EG"
    assert match == "exact"
