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
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
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
                    engine="whisper",
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


DEEPGRAM_KEY = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def key_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A private settings folder, with no key in the environment."""
    from app.core.deepgram_key import ENVIRONMENT_NAMES

    monkeypatch.setenv("CAPTIONFORGE_CONFIG_FILE", str(tmp_path / "cfg" / "c.json"))
    for name in ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path / "cfg" / "deepgram.key"


def test_transcribe_can_use_deepgram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.srt"
    output.write_text("subtitle", encoding="utf-8")
    captured: dict[str, Any] = {}

    class StubService:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured["adapter"] = kwargs.get("deepgram")

        def process(self, url: str, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                paths=(output,),
                used_existing_captions=False,
                prepared_audio_path=None,
                transcription=SimpleNamespace(
                    detected_language="tr",
                    language_probability=0.9,
                    model_name="nova-3",
                    device="deepgram",
                    compute_type="cloud",
                    engine="deepgram",
                ),
            )

    monkeypatch.setattr(cli, "TranscriptionService", StubService)
    result = CliRunner().invoke(
        cli.app,
        ["transcribe", VIDEO_URL, "--engine", "deepgram", "--model", "nova-3"],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["engine"] == "deepgram" and captured["model_name"] == "nova-3"
    assert captured["adapter"] is not None
    assert "Completed using Deepgram transcription." in result.stdout
    # Device and compute type describe this computer, not Deepgram's.
    assert "model: nova-3" in result.stdout and "device:" not in result.stdout


def test_the_key_command_checks_then_saves(
    key_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.adapters.deepgram_adapter import DeepgramAdapter

    monkeypatch.setattr(DeepgramAdapter, "verify_key", lambda self, key: True)
    result = CliRunner().invoke(cli.app, ["deepgram-key"], input=f"{DEEPGRAM_KEY}\n")

    assert result.exit_code == 0, result.stdout
    assert key_home.read_text(encoding="utf-8").strip() == DEEPGRAM_KEY
    # The prompt hides what is typed, and the reply shows only the end.
    assert DEEPGRAM_KEY not in result.stdout and "…4567" in result.stdout


def test_the_key_command_saves_nothing_deepgram_refuses(
    key_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.adapters.deepgram_adapter import DeepgramAdapter

    monkeypatch.setattr(DeepgramAdapter, "verify_key", lambda self, key: False)
    result = CliRunner().invoke(cli.app, ["deepgram-key"], input=f"{DEEPGRAM_KEY}\n")

    assert result.exit_code != 0
    assert "did not accept" in result.stderr
    assert not key_home.exists()


def test_the_key_can_be_forgotten_and_doctor_says_so(key_home: Path) -> None:
    from app.core.deepgram_key import save_key

    save_key(DEEPGRAM_KEY, key_home)
    doctor = CliRunner().invoke(cli.app, ["doctor"])
    assert "Deepgram key" in doctor.stdout and "…4567" in doctor.stdout

    result = CliRunner().invoke(cli.app, ["deepgram-key", "--forget"])

    assert result.exit_code == 0
    assert "Removed the saved Deepgram key." in result.stdout
    assert not key_home.exists()
