"""Offline tests for Phase 7 reliability primitives and CLI behavior."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.core.config import Config
from app.core.exceptions import (
    InvalidYouTubeUrlError,
    MetadataRetrievalError,
)
from app.core.retry import retry_call
from app.interfaces import cli
from app.models.job import Job, JobStatus
from app.utils.file_utils import atomic_write_text


def test_retry_succeeds_after_temporary_failures() -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise MetadataRetrievalError("Temporary", details="timeout")
        return "ok"

    assert (
        retry_call(
            operation,
            attempts=3,
            delay_seconds=0.25,
            operation_name="test",
            sleep=sleeps.append,
        )
        == "ok"
    )
    assert calls == 3
    assert sleeps == [0.25, 0.25]


def test_retry_does_not_repeat_non_retryable_errors() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise InvalidYouTubeUrlError("Invalid")

    with pytest.raises(InvalidYouTubeUrlError):
        retry_call(
            operation,
            attempts=5,
            delay_seconds=0,
            operation_name="test",
        )
    assert calls == 1


def test_configuration_persistence_and_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.json"
    Config(logging_level="DEBUG", retry_count=4).persist(path)
    monkeypatch.setenv("CAPTIONFORGE_RETRY_COUNT", "2")

    loaded = Config.load(env_file=tmp_path / "missing", config_file=path)

    assert loaded.logging_level == "DEBUG"
    assert loaded.retry_count == 2


def test_invalid_persisted_values_recover_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"retry_count": -5, "logging_level": "DEBUG"}),
        encoding="utf-8",
    )

    loaded = Config.load(env_file=tmp_path / "missing", config_file=path)

    assert loaded.retry_count == Config().retry_count
    assert loaded.logging_level == "DEBUG"


def test_atomic_write_replaces_and_leaves_no_partial_files(tmp_path: Path) -> None:
    destination = tmp_path / "captions.txt"
    destination.write_text("old", encoding="utf-8")

    atomic_write_text(destination, "new", overwrite=True)

    assert destination.read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob("*.partial"))


def test_job_state_transitions_measure_duration() -> None:
    job = Job(source_url="https://youtu.be/qJFbKl6RjLU")
    job.transition(JobStatus.RUNNING, stage="metadata")
    job.transition(JobStatus.FAILED, stage="caption_download")

    assert job.started_at is not None
    assert job.finished_at is not None
    assert job.failure_stage == "caption_download"
    assert job.duration_seconds is not None


def test_config_cli_set_show_and_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.json"
    monkeypatch.setenv("CAPTIONFORGE_CONFIG_FILE", str(path))
    runner = CliRunner()

    saved = runner.invoke(cli.app, ["config", "set", "retry_count", "4"])
    shown = runner.invoke(cli.app, ["config", "show"])
    reset = runner.invoke(cli.app, ["config", "reset"])

    assert saved.exit_code == shown.exit_code == reset.exit_code == 0
    assert "retry_count" in shown.stdout
    assert "4" in shown.stdout
    assert not path.exists()


def test_unexpected_cli_error_is_friendly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_create_video_service", lambda: 1 / 0)

    result = CliRunner().invoke(cli.app, ["inspect", "https://youtu.be/qJFbKl6RjLU"])

    assert result.exit_code == 1
    assert "See the log" in result.stderr
    assert "ZeroDivisionError" not in result.stderr
