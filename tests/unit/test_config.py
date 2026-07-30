"""Tests for application configuration."""

from pathlib import Path

from app.core.config import Config


def test_config_has_sensible_defaults() -> None:
    """Default settings should support a local Phase 1 installation."""
    config = Config()

    assert config.default_language == "ar"
    assert config.default_output_folder == Path("output")
    assert config.logging_level == "INFO"
    assert config.default_output_formats == ("srt", "vtt")
    assert config.audio_format == "wav"
    assert config.audio_sample_rate == 16000
    assert config.audio_channels == 1
    assert config.temp_directory == Path("temp")
    assert config.keep_temp_files is False
    assert config.ffmpeg_executable == Path("ffmpeg")


def test_config_loads_prefixed_environment(monkeypatch: object) -> None:
    """Prefixed environment variables should override defaults."""
    monkeypatch.setenv("CAPTIONFORGE_LOGGING_LEVEL", "debug")  # type: ignore[attr-defined]
    monkeypatch.setenv("CAPTIONFORGE_AUDIO_SAMPLE_RATE", "22050")  # type: ignore[attr-defined]

    config = Config.load(env_file=Path("__missing__"))
    assert config.logging_level == "DEBUG"
    assert config.audio_sample_rate == 22050
