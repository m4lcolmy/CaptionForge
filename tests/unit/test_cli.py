"""Tests for the Phase 1 command-line interface."""

import json

import pytest
from typer.testing import CliRunner

from app.core.constants import VERSION
from app.core.exceptions import MetadataRetrievalError
from app.interfaces import cli
from app.models.subtitle import SubtitleDiscoveryResult
from tests.conftest import VIDEO_URL, make_track

runner = CliRunner()


def test_version_command() -> None:
    """The version command should expose the package version."""
    result = runner.invoke(cli.app, ["version"])

    assert result.exit_code == 0
    assert VERSION in result.stdout


def test_doctor_reports_audio_dependencies(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doctor should check FFmpeg, yt-dlp, and the temporary folder."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "FFmpeg installed" in result.stdout
    assert "FFmpeg version" in result.stdout
    assert "yt-dlp installed" in result.stdout
    assert "Writable temporary folder" in result.stdout


def test_inspect_help() -> None:
    """The inspect command should document its Phase 2 options."""
    result = runner.invoke(cli.app, ["inspect", "--help"])

    assert result.exit_code == 0
    assert "--language" in result.stdout
    assert "--json" in result.stdout


class StubVideoService:
    """CLI test service returning a result or raising a prepared error."""

    def __init__(
        self,
        result: SubtitleDiscoveryResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error

    def inspect(self, url: str, preferred_language: str) -> SubtitleDiscoveryResult:
        if self.error:
            raise self.error
        assert url == VIDEO_URL
        assert preferred_language == "ar"
        assert self.result is not None
        return self.result


def test_valid_mocked_inspection(
    video_metadata: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Human output should display metadata, tracks, and selection."""
    track = make_track("ar")
    result_model = SubtitleDiscoveryResult(
        video=video_metadata,
        manual_tracks=(track,),
        selected_track=track,
        preferred_language="ar",
        selection_reason="Selected manual Arabic caption using an exact match.",
    )
    monkeypatch.setattr(
        cli, "_create_video_service", lambda: StubVideoService(result_model)
    )

    result = runner.invoke(cli.app, ["inspect", VIDEO_URL])

    assert result.exit_code == 0
    assert "Example video" in result.stdout
    assert "Selected subtitle" in result.stdout
    assert "Arabic" in result.stdout


def test_json_output_is_clean_and_valid(
    video_metadata: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON mode should emit exactly one parseable result document to stdout."""
    result_model = SubtitleDiscoveryResult(
        video=video_metadata,
        preferred_language="ar",
    )
    monkeypatch.setattr(
        cli, "_create_video_service", lambda: StubVideoService(result_model)
    )

    result = runner.invoke(cli.app, ["inspect", VIDEO_URL, "--json"])
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["video"]["video_id"] == "qJFbKl6RjLU"
    assert payload["selected_track"] is None
    assert "No subtitle" not in result.stdout


def test_adapter_failure_has_metadata_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata failures should be concise and use stable exit code four."""
    monkeypatch.setattr(
        cli,
        "_create_video_service",
        lambda: StubVideoService(
            error=MetadataRetrievalError("Metadata unavailable", details="secret")
        ),
    )

    result = runner.invoke(cli.app, ["inspect", VIDEO_URL])

    assert result.exit_code == 4
    assert "Metadata unavailable" in result.stderr
    assert "secret" not in result.stderr


def test_invalid_url_uses_input_exit_code() -> None:
    """Invalid input should fail before any network request."""
    result = runner.invoke(cli.app, ["inspect", "https://example.com/video"])

    assert result.exit_code == 2
    assert "valid YouTube" in result.stderr
