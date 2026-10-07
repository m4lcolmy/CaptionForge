"""Where the Deepgram API key comes from, and the one file that may hold it.

The key is deliberately not a :class:`~app.core.config.Config` setting:
``config show`` prints every setting and ``persist`` writes them all to
``config.json``, and a secret must reach neither. It is read from the
environment first, then from a file beside the configuration that only this
user can read. Nothing outside this module ever sees more than its last four
characters.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from app.core.config import Config
from app.core.constants import ENV_FILE
from app.core.exceptions import ConfigurationError, ExportError, ValidationError

KEY_FILE_NAME = "deepgram.key"
# The first is CaptionForge's own; the second is the name Deepgram's tools use.
ENVIRONMENT_NAMES = ("CAPTIONFORGE_DEEPGRAM_API_KEY", "DEEPGRAM_API_KEY")
_MINIMUM_LENGTH = 20
_MAXIMUM_LENGTH = 256


@dataclass(frozen=True)
class DeepgramKey:
    """A usable key and where it was found."""

    value: str
    # "environment" when a variable or the .env file set it, "file" when it
    # was saved from the page, the window, or ``captionforge deepgram-key``.
    source: str

    @property
    def hint(self) -> str:
        """The last four characters, which is all an interface may show."""
        return f"…{self.value[-4:]}"

    def __repr__(self) -> str:
        # A key must not land in a log line through an f-string or a traceback.
        return f"DeepgramKey(source={self.source!r}, hint={self.hint!r})"


def key_path() -> Path:
    """The file a saved key lives in, beside ``config.json``."""
    return Config.user_config_path().parent / KEY_FILE_NAME


def normalize_key(value: str) -> str:
    """Accept a key pasted with spaces or a ``Token`` prefix; reject the rest."""
    candidate = value.strip()
    if candidate.lower().startswith("token "):
        candidate = candidate[6:].strip()
    if (
        len(candidate) < _MINIMUM_LENGTH
        or len(candidate) > _MAXIMUM_LENGTH
        or any(character.isspace() for character in candidate)
        or not candidate.isascii()
    ):
        raise ValidationError("That does not look like a Deepgram API key.")
    return candidate


def load_key(
    path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = None,
) -> DeepgramKey | None:
    """Find the key: the environment, then the .env file, then the saved file."""
    variables = os.environ if environ is None else environ
    dotenv_path = env_file if env_file is not None else Path(ENV_FILE)
    dotenv = dotenv_values(dotenv_path) if dotenv_path.is_file() else {}
    for source in (variables, dotenv):
        for name in ENVIRONMENT_NAMES:
            raw = source.get(name)
            if raw and raw.strip():
                with suppress(ValidationError):
                    return DeepgramKey(normalize_key(raw), "environment")
    destination = path or key_path()
    try:
        raw = destination.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return DeepgramKey(normalize_key(raw), "file")
    except ValidationError:
        return None


def save_key(value: str, path: Path | None = None) -> DeepgramKey:
    """Write the key where only this user can read it, and return it."""
    from app.utils.file_utils import atomic_write_text

    key = normalize_key(value)
    destination = path or key_path()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp creates the temporary file as 0600, and the replace keeps
        # that file, so the key is never readable by anyone else, even briefly.
        atomic_write_text(destination, key + "\n", overwrite=True)
        os.chmod(destination, 0o600)
    except (OSError, ExportError) as exc:
        raise ConfigurationError(
            "The Deepgram key could not be saved.", details=str(exc)
        ) from exc
    return DeepgramKey(key, "file")


def forget_key(path: Path | None = None) -> bool:
    """Delete the saved key; return whether there was one."""
    destination = path or key_path()
    try:
        destination.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ConfigurationError(
            "The saved Deepgram key could not be removed.", details=str(exc)
        ) from exc
    return True


def describe_key(key: DeepgramKey | None) -> dict[str, object]:
    """What an interface may know about the key: whether, where, and a hint."""
    return {
        "saved": key is not None,
        "source": key.source if key is not None else None,
        "hint": key.hint if key is not None else None,
    }


__all__ = [
    "DeepgramKey",
    "describe_key",
    "forget_key",
    "key_path",
    "load_key",
    "normalize_key",
    "save_key",
]
