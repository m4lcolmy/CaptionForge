"""Typed application configuration loaded from defaults and the environment."""

import json
import os
from pathlib import Path
from typing import Any, ClassVar

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.constants import ENV_FILE, ENV_PREFIX, SUPPORTED_OUTPUT_FORMATS
from app.core.exceptions import ConfigurationError


class Config(BaseModel):
    """Validated runtime settings for CaptionForge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_language: str = Field(default="ar", min_length=2)
    default_output_folder: Path = Path("output")
    default_whisper_model: str = Field(default="small", min_length=1)
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_beam_size: int = Field(default=5, ge=1, le=100)
    whisper_vad_enabled: bool = True
    whisper_min_silence_duration_ms: int = Field(default=500, ge=0)
    whisper_language: str | None = None
    whisper_model_download_directory: Path | None = None
    logging_level: str = "INFO"
    temp_directory: Path = Path("temp")
    keep_temp_files: bool = False
    audio_format: str = Field(default="wav", min_length=1)
    audio_sample_rate: int = Field(default=16000, ge=8000, le=192000)
    audio_channels: int = Field(default=1, ge=1, le=8)
    ffmpeg_executable: Path = Path("ffmpeg")
    default_output_formats: tuple[str, ...] = ("srt", "vtt")
    maximum_characters_per_line: int = Field(default=42, ge=1, le=500)
    maximum_subtitle_lines: int = Field(default=2, ge=1, le=2)
    minimum_subtitle_duration: float = Field(default=0.8, gt=0, le=10)
    maximum_subtitle_duration: float = Field(default=7.0, gt=0, le=60)
    subtitle_merge_threshold: float = Field(default=1.0, ge=0, le=10)
    duplicate_detection_threshold: float = Field(default=0.9, ge=0, le=1)
    remove_diacritics: bool = False
    normalize_arabic_letters: bool = False
    normalize_arabic_indic_digits: bool = False
    retry_count: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0, le=60)
    minimum_free_disk_bytes: int = Field(default=100 * 1024 * 1024, ge=0)

    _field_names: ClassVar[frozenset[str]] = frozenset(
        {
            "default_language",
            "default_output_folder",
            "default_whisper_model",
            "whisper_device",
            "whisper_compute_type",
            "whisper_beam_size",
            "whisper_vad_enabled",
            "whisper_min_silence_duration_ms",
            "whisper_language",
            "whisper_model_download_directory",
            "logging_level",
            "temp_directory",
            "keep_temp_files",
            "audio_format",
            "audio_sample_rate",
            "audio_channels",
            "ffmpeg_executable",
            "default_output_formats",
            "maximum_characters_per_line",
            "maximum_subtitle_lines",
            "minimum_subtitle_duration",
            "maximum_subtitle_duration",
            "subtitle_merge_threshold",
            "duplicate_detection_threshold",
            "remove_diacritics",
            "normalize_arabic_letters",
            "normalize_arabic_indic_digits",
            "retry_count",
            "retry_delay_seconds",
            "minimum_free_disk_bytes",
        }
    )

    @field_validator("maximum_subtitle_duration")
    @classmethod
    def validate_subtitle_durations(cls, value: float, info: Any) -> float:
        minimum = info.data.get("minimum_subtitle_duration", 0)
        if value < minimum:
            raise ValueError(
                "maximum_subtitle_duration must not be less than "
                "minimum_subtitle_duration"
            )
        return value

    @field_validator("whisper_device")
    @classmethod
    def validate_whisper_device(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise ValueError("whisper_device must be auto, cpu, or cuda")
        return normalized

    @field_validator("whisper_compute_type")
    @classmethod
    def validate_whisper_compute_type(cls, value: str) -> str:
        normalized = value.lower()
        allowed = {
            "auto",
            "default",
            "int8",
            "int8_float16",
            "int8_float32",
            "int16",
            "float16",
            "float32",
            "bfloat16",
        }
        if normalized not in allowed:
            raise ValueError(f"whisper_compute_type must be one of {sorted(allowed)}")
        return normalized

    @field_validator(
        "whisper_language", "whisper_model_download_directory", mode="before"
    )
    @classmethod
    def empty_whisper_values_are_none(cls, value: Any) -> Any:
        """Treat blank optional environment settings as unset."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("logging_level")
    @classmethod
    def validate_logging_level(cls, value: str) -> str:
        """Normalize and validate the configured log level."""
        normalized = value.upper()
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"logging_level must be one of {sorted(allowed)}")
        return normalized

    @field_validator("default_output_formats", mode="before")
    @classmethod
    def parse_output_formats(cls, value: Any) -> Any:
        """Accept comma-separated environment values or an iterable of formats."""
        if isinstance(value, str):
            return tuple(
                item.strip().lower() for item in value.split(",") if item.strip()
            )
        return value

    @field_validator("default_output_formats")
    @classmethod
    def validate_output_formats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Ensure at least one known output format is configured."""
        normalized = tuple(item.lower() for item in value)
        unknown = set(normalized) - SUPPORTED_OUTPUT_FORMATS
        if not normalized:
            raise ValueError("at least one output format is required")
        if unknown:
            raise ValueError(f"unsupported output formats: {sorted(unknown)}")
        return normalized

    @classmethod
    def user_config_path(cls) -> Path:
        """Return the platform-appropriate per-user JSON configuration path."""
        override = os.environ.get("CAPTIONFORGE_CONFIG_FILE")
        if override:
            return Path(override).expanduser()
        base = os.environ.get("XDG_CONFIG_HOME")
        root = Path(base).expanduser() if base else Path.home() / ".config"
        return root / "captionforge" / "config.json"

    @classmethod
    def load(
        cls, env_file: Path | None = None, config_file: Path | None = None
    ) -> "Config":
        """Load settings from defaults, an optional dotenv file, and the environment."""
        source_file = env_file if env_file is not None else Path(ENV_FILE)
        dotenv_data = dotenv_values(source_file) if source_file.is_file() else {}
        combined: dict[str, Any] = {
            key: value for key, value in dotenv_data.items() if value is not None
        }
        persisted_path = config_file or cls.user_config_path()
        if persisted_path.is_file():
            try:
                persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
                if isinstance(persisted, dict):
                    combined.update(
                        {str(key): value for key, value in persisted.items()}
                    )
            except (OSError, json.JSONDecodeError):
                # A damaged optional user file must never make the app unusable.
                pass
        combined.update(os.environ)

        values = {
            field: combined.get(f"{ENV_PREFIX}{field.upper()}", combined.get(field))
            for field in cls._field_names
            if f"{ENV_PREFIX}{field.upper()}" in combined or field in combined
        }
        try:
            return cls.model_validate(values)
        except ValueError as exc:
            # Recover field-by-field so one stale value does not block every command.
            valid: dict[str, Any] = {}
            for name, value in values.items():
                try:
                    cls.model_validate({name: value})
                    valid[name] = value
                except ValueError:
                    continue
            try:
                return cls.model_validate(valid)
            except ValueError as fallback_exc:
                raise ConfigurationError(
                    "Unable to load CaptionForge configuration",
                    details=str(fallback_exc),
                ) from exc

    def persist(self, path: Path | None = None) -> Path:
        """Atomically save explicit settings to the user configuration file."""
        from app.utils.file_utils import atomic_write_text

        destination = path or self.user_config_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            destination,
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)
            + "\n",
            overwrite=True,
        )
        return destination

    @classmethod
    def parse_setting(cls, key: str, value: str) -> Any:
        """Validate a single CLI setting and return its normalized JSON value."""
        if key not in cls._field_names:
            raise ConfigurationError(f"Unknown configuration key: {key}")
        try:
            candidate = cls.model_validate({key: value})
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid value for configuration key '{key}'", details=str(exc)
            ) from exc
        return candidate.model_dump(mode="json")[key]
