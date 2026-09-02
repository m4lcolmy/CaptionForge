"""Per-user web interface choices, remembered between sessions.

The server picks a free port on each run, so the page's origin changes and
browser storage would be empty almost every session. These live on disk beside
the CLI configuration instead, in their own file: they are interface choices,
not validated application settings, and must never change what the CLI does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.config import Config
from app.core.constants import SUPPORTED_OUTPUT_FORMATS
from app.core.logging_config import get_logger

PREFERENCES_FILE_NAME = "web-preferences.json"


class WebPreferences(BaseModel):
    """The controls the page restores on the next visit."""

    model_config = ConfigDict(extra="ignore")

    language: str | None = Field(default=None, max_length=32)
    formats: tuple[str, ...] = ()
    model: str | None = Field(default=None, max_length=256)
    device: str | None = None
    prompt: str | None = Field(default=None, max_length=2048)
    force: bool = False
    overwrite: bool = False
    keep_audio: bool = False
    timestamped_txt: bool = False
    postprocess: bool = True
    allow_translated: bool = False

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Drop formats this build no longer supports rather than failing."""
        return tuple(item for item in value if item in SUPPORTED_OUTPUT_FORMATS)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str | None) -> str | None:
        """Accept only devices the Whisper adapter can resolve."""
        if value is None:
            return None
        normalized = value.lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        return normalized


def preferences_path() -> Path:
    """Return the file holding the interface choices, beside config.json."""
    return Config.user_config_path().parent / PREFERENCES_FILE_NAME


def load_preferences(path: Path | None = None) -> WebPreferences:
    """Read saved choices, falling back per field and never raising."""
    destination = path or preferences_path()
    try:
        raw = json.loads(destination.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return WebPreferences()
    except (OSError, json.JSONDecodeError) as exc:
        get_logger().warning(
            "Ignoring unreadable web preferences technical_cause={}", exc
        )
        return WebPreferences()
    if not isinstance(raw, dict):
        return WebPreferences()
    try:
        return WebPreferences(**raw)
    except ValidationError:
        return _salvage(raw)


def save_preferences(preferences: WebPreferences, path: Path | None = None) -> Path:
    """Atomically write the choices, so a crash cannot truncate the file."""
    from app.utils.file_utils import atomic_write_text

    destination = path or preferences_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        destination,
        json.dumps(preferences.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n",
        overwrite=True,
    )
    return destination


def resolve(preferences: WebPreferences, config: Config) -> dict[str, Any]:
    """Merge saved choices over configuration to get the page's start state."""
    return {
        "language": preferences.language or config.default_language,
        "formats": list(preferences.formats) or list(config.default_output_formats[:1]),
        "model": preferences.model or config.default_whisper_model,
        "device": preferences.device or config.whisper_device,
        "prompt": preferences.prompt or "",
        "force": preferences.force,
        "overwrite": preferences.overwrite,
        "keep_audio": preferences.keep_audio,
        "timestamped_txt": preferences.timestamped_txt,
        "postprocess": preferences.postprocess,
        "allow_translated": preferences.allow_translated,
    }


def _salvage(raw: dict[str, Any]) -> WebPreferences:
    """Keep every individually valid field and discard only what is broken."""
    valid: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in WebPreferences.model_fields:
            continue
        try:
            WebPreferences(**{key: value})
        except ValidationError:
            get_logger().warning("Ignoring invalid web preference {}", key)
            continue
        valid[key] = value
    return WebPreferences(**valid)
