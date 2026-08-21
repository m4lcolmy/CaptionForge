"""Safe output-directory and filename utilities."""

import os
import re
import shutil
import tempfile
import unicodedata
from contextlib import suppress
from pathlib import Path
from uuid import UUID

from app.core.exceptions import (
    CleanupError,
    ExportError,
    InsufficientDiskSpaceError,
    TemporaryDirectoryError,
)

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_SPACES = re.compile(r"\s+")
_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_filename(title: str, *, fallback: str = "captions") -> str:
    """Create a portable filename stem while preserving useful Unicode."""
    value = unicodedata.normalize("NFC", title)
    value = _UNSAFE.sub("_", value)
    value = _SPACES.sub(" ", value).strip(" .")
    value = value[:180].rstrip(" .")
    if not value or value.upper() in _WINDOWS_NAMES:
        value = fallback
    return value


def ensure_output_directory(directory: Path) -> Path:
    """Create and validate a writable output directory."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(
            f"Output directory could not be created: {directory}", details=str(exc)
        ) from exc
    if not directory.is_dir() or not os.access(directory, os.W_OK):
        raise ExportError(f"Output path is not a writable directory: {directory}")
    return directory


def create_job_directory(base: Path, job_id: UUID) -> Path:
    """Create a private writable directory for one audio preparation job."""
    try:
        base.mkdir(parents=True, exist_ok=True)
        if not base.is_dir() or not os.access(base, os.W_OK):
            raise PermissionError(str(base))
        job_directory = base / f"captionforge-{job_id}"
        job_directory.mkdir(mode=0o700)
        probe = job_directory / ".write-test"
        probe.touch()
        probe.unlink()
        return job_directory
    except OSError as exc:
        raise TemporaryDirectoryError(
            f"Temporary directory is not writable: {base}", details=str(exc)
        ) from exc


def ensure_disk_space(directory: Path, required_bytes: int) -> None:
    """Ensure the filesystem has enough room for download and conversion."""
    try:
        free = shutil.disk_usage(directory).free
    except OSError as exc:
        raise TemporaryDirectoryError(
            f"Temporary directory could not be inspected: {directory}",
            details=str(exc),
        ) from exc
    if free < required_bytes:
        raise InsufficientDiskSpaceError(
            "There is not enough free disk space to prepare this audio."
        )


def atomic_write_text(path: Path, content: str, *, overwrite: bool = False) -> None:
    """Write UTF-8 text and replace the destination only after a complete flush."""
    atomic_write_bytes(path, content.encode("utf-8"), overwrite=overwrite)


def atomic_write_bytes(path: Path, content: bytes, *, overwrite: bool = False) -> None:
    """Write bytes and replace the destination only after a complete flush."""
    if path.exists() and not overwrite:
        raise ExportError(f"Output file already exists: {path}. Use --overwrite.")
    temporary: Path | None = None
    try:
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".partial", dir=path.parent
        )
        temporary = Path(raw_path)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as exc:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise ExportError(
            "Output file could not be written.", details=str(exc)
        ) from exc


def cleanup_path(path: Path) -> None:
    """Remove one temporary file or directory with a domain-friendly failure."""
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except OSError as exc:
        raise CleanupError(
            f"Temporary files could not be cleaned up: {path}", details=str(exc)
        ) from exc
