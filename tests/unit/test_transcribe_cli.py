"""CLI coverage for the Phase 5 transcribe command."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from app.interfaces import cli
from tests.conftest import VIDEO_URL


def test_transcribe_cli_options_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.srt"
    output.write_text("subtitle", encoding="utf-8")
    captured: dict[str, Any] = {}

    class StubService:
        def __init__(self, *_args: Any) -> None:
            pass

        def process(self, url: str, **kwargs: Any) -> Any:
            captured.update(url=url, **kwargs)
            kwargs["progress"]("Transcribing", 50.0)
            return SimpleNamespace(
                paths=(output,),
                used_existing_captions=False,
                prepared_audio_path=None,
                transcription=SimpleNamespace(
                    detected_language="ar",
                    language_probability=0.98,
                    model_name="medium",
                    device="cpu",
                    compute_type="int8",
                ),
            )

    monkeypatch.setattr(cli, "TranscriptionService", StubService)
    result = CliRunner().invoke(
        cli.app,
        [
            "transcribe",
            VIDEO_URL,
            "--language",
            "ar",
            "--model",
            "medium",
            "--device",
            "cpu",
            "--compute-type",
            "int8",
            "--format",
            "srt",
            "--output",
            str(tmp_path),
            "--keep-audio",
            "--force",
            "--no-postprocess",
        ],
    )

    assert result.exit_code == 0
    assert captured["url"] == VIDEO_URL
    assert captured["language"] == "ar"
    assert captured["model_name"] == "medium"
    assert captured["force"] is True
    assert captured["keep_audio"] is True
    assert captured["postprocess"] is False
    assert "Transcribing (50%)" in result.stdout
    assert str(output.resolve()) in result.stdout


def test_transcribe_help_lists_phase5_options() -> None:
    result = CliRunner().invoke(cli.app, ["transcribe", "--help"])
    assert result.exit_code == 0
    for option in (
        "--language",
        "--model",
        "--device",
        "--compute-type",
        "--format",
        "--output",
        "--keep-audio",
        "--force",
        "--no-postprocess",
    ):
        assert option in result.stdout
