"""Tell a file on this computer apart from a link, whatever form it was typed in.

People paste a path in every form their system copies it: plain, wrapped in
quotes by Windows' "Copy as path", as a ``file://`` URI from a file manager or
a drag, or starting with ``~``. All of them name the same file, and none of
them should be mistaken for a malformed YouTube link.
"""

import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from app.core.constants import LOCAL_MEDIA_EXTENSIONS
from app.core.exceptions import LocalFileNotFoundError, UnreadableMediaFileError

_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
# file:///C:/Users/x parses to the path /C:/Users/x.
_URI_DRIVE = re.compile(r"^/[A-Za-z]:/")
_PATH_PREFIXES = ("/", "~", "./", "../", ".\\", "..\\", "\\\\")
_QUOTES = ("'", '"')


def local_media_path(text: str) -> Path | None:
    """Return the file a typed or pasted input names, or None when it is a link.

    The path is returned as written, not checked: a path that names nothing
    should be reported as a missing file, never as an invalid link.
    """
    candidate = _strip_quotes(text.strip())
    if not candidate:
        return None
    if candidate.lower().startswith("file:"):
        parsed = urlparse(candidate)
        # file:///x and file://localhost/x are this computer; any other host
        # is not something CaptionForge can open.
        if parsed.netloc not in ("", "localhost"):
            return None
        location = unquote(parsed.path)
        if _URI_DRIVE.match(location):
            location = location[1:]
        return Path(location)
    if _SCHEME.match(candidate):
        return None
    path = Path(candidate).expanduser()
    if (
        candidate.startswith(_PATH_PREFIXES)
        or _WINDOWS_DRIVE.match(candidate)
        or path.suffix.lower().lstrip(".") in LOCAL_MEDIA_EXTENSIONS
        or _is_file(path)
    ):
        return path
    return None


def resolve_media_file(path: Path) -> Path:
    """Return the file as an absolute path, or say plainly why it cannot be used.

    Absolute matters beyond tidiness: FFprobe reads a bare argument that starts
    with ``-`` as an option, and never a path that starts with ``/``.
    """
    try:
        resolved = path.expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise UnreadableMediaFileError(
            f"CaptionForge could not open {path}.", details=str(exc)
        ) from exc
    if resolved.is_dir():
        raise LocalFileNotFoundError(
            f"{resolved} is a folder. Choose one video or audio file inside it."
        )
    if not resolved.is_file():
        raise LocalFileNotFoundError(f"No file was found at {resolved}.")
    if not os.access(resolved, os.R_OK):
        raise UnreadableMediaFileError(
            f"CaptionForge is not allowed to read {resolved}."
        )
    return resolved


def _strip_quotes(text: str) -> str:
    """Remove one pair of matching quotes around the whole input."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in _QUOTES:
        return text[1:-1].strip()
    return text


def _is_file(path: Path) -> bool:
    """Whether the path names an existing file, treating odd names as not."""
    try:
        return path.is_file()
    except (OSError, ValueError):
        return False


__all__ = ["local_media_path", "resolve_media_file"]
