"""Copies of files the page hands over, kept only while they can still be used.

A browser never tells a page where a chosen file lives, so the page sends the
file's bytes and CaptionForge keeps the copy in its temporary folder. Pasting
the file's path instead skips the copy entirely. A copy is removed when a newer
one replaces it and no job still needs it, and every copy goes when the server
stops; copies a crashed session left behind are swept on the next start.
"""

from __future__ import annotations

import re
import shutil
import threading
import time
from collections.abc import AsyncIterator, Collection
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from app.core.exceptions import (
    InsufficientDiskSpaceError,
    TemporaryDirectoryError,
    UnreadableMediaFileError,
)
from app.core.logging_config import get_logger
from app.utils.file_utils import ensure_disk_space, sanitize_filename

UPLOAD_FOLDER = "uploads"
# Old enough that no server still running could be using it.
STALE_AFTER_SECONDS = 24 * 60 * 60
_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,10}$")


class UploadStore:
    """Receives files from the page into private folders under ``temp/``."""

    def __init__(self, temp_directory: Path, minimum_free_bytes: int = 0) -> None:
        # Absolute, because the page sends this path back as the file to read.
        self._root = (temp_directory / UPLOAD_FOLDER).resolve()
        self._minimum_free = minimum_free_bytes
        self._folders: list[Path] = []
        self._lock = threading.Lock()

    async def receive(
        self, name: str, expected_bytes: int | None, chunks: AsyncIterator[bytes]
    ) -> Path:
        """Write one file and return where it landed, or leave nothing behind."""
        folder = self._root / uuid4().hex
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            folder.mkdir(mode=0o700)
        except OSError as exc:
            raise TemporaryDirectoryError(
                f"Temporary directory is not writable: {self._root}",
                details=str(exc),
            ) from exc
        with self._lock:
            self._folders.append(folder)
        destination = folder / safe_name(name)
        written = 0
        try:
            try:
                ensure_disk_space(folder, (expected_bytes or 0) + self._minimum_free)
            except InsufficientDiskSpaceError as exc:
                raise InsufficientDiskSpaceError(
                    "There is not enough free disk space for a copy of that file. "
                    "Paste the file's path instead; that needs no copy."
                ) from exc
            with destination.open("xb") as handle:
                async for chunk in chunks:
                    handle.write(chunk)
                    written += len(chunk)
        except BaseException:
            # Cancelled, disconnected or failed: a partial copy is useless.
            self._remove(folder)
            raise
        if not written:
            self._remove(folder)
            raise UnreadableMediaFileError("That file is empty.")
        get_logger().info("Received a file from the page bytes={}", written)
        return destination

    def prune(self, keep: Collection[Path]) -> None:
        """Remove every copy this session made, except the ones in ``keep``."""
        wanted = {path.parent for path in keep}
        with self._lock:
            doomed = [folder for folder in self._folders if folder not in wanted]
        for folder in doomed:
            self._remove(folder)

    def discard_all(self) -> None:
        """Remove every copy this session made; the server is stopping."""
        self.prune(())

    def sweep_stale(self) -> None:
        """Remove copies a session that never stopped cleanly left behind."""
        if not self._root.is_dir():
            return
        cutoff = time.time() - STALE_AFTER_SECONDS
        for folder in self._root.iterdir():
            with suppress(OSError):
                if folder.is_dir() and folder.stat().st_mtime < cutoff:
                    shutil.rmtree(folder)

    def _remove(self, folder: Path) -> None:
        with self._lock:
            if folder in self._folders:
                self._folders.remove(folder)
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass
        except OSError as exc:
            # A job may still hold the file open on Windows; it goes next start.
            get_logger().warning("Could not remove an uploaded copy: {}", exc)


def safe_name(name: str) -> str:
    """Keep the person's file name, minus anything a path could abuse."""
    original = Path(name.replace("\\", "/")).name
    suffix = Path(original).suffix
    if not _SUFFIX.fullmatch(suffix):
        suffix = ""
    stem = original[: len(original) - len(suffix)] if suffix else original
    return f"{sanitize_filename(stem, fallback='upload')}{suffix.lower()}"


__all__ = ["UPLOAD_FOLDER", "UploadStore", "safe_name"]
