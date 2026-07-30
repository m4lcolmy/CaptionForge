"""Offline tests for the Phase 3 extract command."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.interfaces import cli
from app.models.subtitle import RawSubtitle, SubtitleDiscoveryResult
from tests.conftest import VIDEO_URL, make_track

runner = CliRunner()


class StubAdapter:
    """Adapter returning prepared caption content."""

    def __init__(self, raw: RawSubtitle) -> None:
        self.raw = raw

    def download_subtitle(self, video_id: str, track: object) -> RawSubtitle:
        return self.raw


class StubVideoService:
    """Inspection stub used to isolate the CLI from YouTube."""

    result: SubtitleDiscoveryResult

    def __init__(self, adapter: object, subtitle_service: object) -> None:
        pass

    def inspect(self, url: str, language: str) -> SubtitleDiscoveryResult:
        return self.result


def test_extract_multiple_formats(
    tmp_path: Path, video_metadata: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    track = make_track("ar")
    StubVideoService.result = SubtitleDiscoveryResult(
        video=video_metadata,  # type: ignore[arg-type]
        manual_tracks=(track,),
        selected_track=track,
        preferred_language="ar",
    )
    monkeypatch.setattr(
        cli,
        "YtDlpAdapter",
        lambda: StubAdapter(
            RawSubtitle(
                format="vtt",
                content="WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nمرحبا\n",
            )
        ),
    )
    monkeypatch.setattr(cli, "VideoService", StubVideoService)

    result = runner.invoke(
        cli.app,
        [
            "extract",
            VIDEO_URL,
            "--format",
            "srt",
            "--format",
            "txt",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Success" in result.stdout
    assert (tmp_path / "Example video.srt").exists()
    assert (tmp_path / "Example video.txt").exists()


def test_extract_no_matching_caption(
    video_metadata: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    StubVideoService.result = SubtitleDiscoveryResult(
        video=video_metadata,  # type: ignore[arg-type]
        preferred_language="ar",
    )
    monkeypatch.setattr(cli, "YtDlpAdapter", lambda: object())
    monkeypatch.setattr(cli, "VideoService", StubVideoService)

    result = runner.invoke(cli.app, ["extract", VIDEO_URL])

    assert result.exit_code == 1
    assert "No captions matching" in result.stderr
    assert "later phase" in result.stderr
