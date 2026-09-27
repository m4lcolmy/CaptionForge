"""Find newer releases of CaptionForge's packages, and install the chosen ones.

Nothing is installed unless the person picks it. A check lists what could be
updated and what each package does for CaptionForge; an install upgrades only
the packages named, within the versions CaptionForge supports, and loads a new
yt-dlp into the running process so an open page can use it straight away.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from app.core.exceptions import PackageUpdateError, ValidationError
from app.core.logging_config import get_logger

# A check found nothing new a moment ago, so the page does not ask pip again
# every time it is opened. Asking for a fresh check always runs pip.
CHECK_MAX_AGE_SECONDS = 6 * 3600.0
# A hard stop, so a stalled download cannot hold the page or a command forever.
PIP_TIMEOUT_SECONDS = 600.0

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class Package:
    """One of CaptionForge's own dependencies, and why it is there."""

    name: str
    module: str
    requirement: str
    mission: str
    reloads: bool = False


# The versions mirror pyproject.toml, so an update never leaves the range
# CaptionForge was written against. A test keeps the two in step.
PACKAGES: tuple[Package, ...] = (
    Package(
        "yt-dlp",
        "yt_dlp",
        ">=2026.8.19",
        "Reads YouTube: video details, captions, audio and video. "
        "YouTube stops accepting old releases.",
        reloads=True,
    ),
    Package(
        "faster-whisper",
        "faster_whisper",
        ">=1.1,<2.0",
        "Transcribes the audio on this computer when a video has no captions.",
    ),
    Package("python-docx", "docx", ">=1.1,<2.0", "Writes transcripts as Word files."),
    Package("fastapi", "fastapi", ">=0.115,<1.0", "Answers the page's requests."),
    Package(
        "uvicorn", "uvicorn", ">=0.30,<1.0", "Runs the local server the page talks to."
    ),
    Package(
        "pywebview",
        "webview",
        ">=5.0,<7.0",
        "Opens CaptionForge in a window of its own.",
    ),
    Package(
        "typer",
        "typer",
        ">=0.12,<1.0",
        "Understands the captionforge commands typed in a terminal.",
    ),
    Package(
        "rich",
        "rich",
        ">=13.7,<15.0",
        "Draws the tables, colours and progress lines in a terminal.",
    ),
    Package(
        "pydantic",
        "pydantic",
        ">=2.7,<3.0",
        "Checks settings and video details before anything uses them.",
    ),
    Package("loguru", "loguru", ">=0.7,<1.0", "Writes the log files in logs/."),
    Package(
        "python-dotenv", "dotenv", ">=1.0,<2.0", "Reads settings from a .env file."
    ),
)


@dataclass(frozen=True)
class AvailableUpdate:
    """A newer release of one package, within the range CaptionForge supports."""

    package: Package
    installed: str
    latest: str

    @property
    def name(self) -> str:
        return self.package.name

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.package.name,
            "installed": self.installed,
            "latest": self.latest,
            "mission": self.package.mission,
        }


@dataclass(frozen=True)
class InstalledUpdate:
    """What one install changed, and whether it is already in use."""

    name: str
    version: str
    restart_needed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "restart_needed": self.restart_needed,
        }


class PackageUpdater:
    """Check CaptionForge's packages for updates and install the chosen ones."""

    def __init__(
        self,
        packages: Sequence[Package] = PACKAGES,
        *,
        runner: Runner | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_age_seconds: float = CHECK_MAX_AGE_SECONDS,
    ) -> None:
        self._packages = tuple(packages)
        self._runner = runner or _run
        self._clock = clock
        self._max_age = max_age_seconds
        # One pip at a time: a check and an install, or two installs, would
        # otherwise race over the same site-packages.
        self._lock = threading.Lock()
        self._checked: tuple[float, tuple[AvailableUpdate, ...]] | None = None

    @staticmethod
    def can_update() -> bool:
        """Whether this installation has a pip that updates can run through."""
        if getattr(sys, "frozen", False):
            # A bundled executable carries its packages inside itself.
            return False
        return importlib.util.find_spec("pip") is not None

    @staticmethod
    def installed_version(name: str) -> str | None:
        """The version on disk, which is not always the one loaded."""
        importlib.invalidate_caches()
        try:
            return package_version(name)
        except PackageNotFoundError:
            return None

    def check(self, *, fresh: bool = False) -> tuple[AvailableUpdate, ...]:
        """List newer releases of the installed packages, installing nothing."""
        with self._lock:
            now = self._clock()
            if (
                not fresh
                and self._checked is not None
                and now - self._checked[0] < self._max_age
            ):
                return self._checked[1]
            updates = self._check()
            self._checked = (now, updates)
            return updates

    def install(self, names: Iterable[str]) -> tuple[InstalledUpdate, ...]:
        """Upgrade exactly the named packages, and nothing the person did not pick."""
        wanted = set(names)
        chosen = [package for package in self._packages if package.name in wanted]
        unknown = wanted - {package.name for package in chosen}
        if unknown:
            # Only names from the list reach pip, never whatever a request sent.
            raise ValidationError(
                f"CaptionForge does not update {', '.join(sorted(unknown))}."
            )
        if not chosen:
            return ()
        with self._lock:
            self._require_pip()
            before = {
                package.name: self.installed_version(package.name) for package in chosen
            }
            log = get_logger()
            log.info("Updating packages the user chose: {}", sorted(wanted))
            completed = self._pip("install", "--upgrade", "--quiet", *_specs(chosen))
            if completed.returncode != 0:
                log.warning("pip install failed: {}", _tail(completed))
                raise PackageUpdateError(
                    "The update could not be installed. Check your internet "
                    "connection and try again.",
                    details=_tail(completed),
                )
            installed: list[InstalledUpdate] = []
            for package in chosen:
                version = self.installed_version(package.name)
                if version is None or version == before[package.name]:
                    continue
                in_use = self._activate(package)
                installed.append(InstalledUpdate(package.name, version, not in_use))
                log.info(
                    "Updated {} from {} to {}",
                    package.name,
                    before[package.name],
                    version,
                )
            if self._checked is not None:
                remaining = tuple(
                    update for update in self._checked[1] if update.name not in wanted
                )
                self._checked = (self._checked[0], remaining)
            return tuple(installed)

    def _check(self) -> tuple[AvailableUpdate, ...]:
        self._require_pip()
        present = {
            package.name: version
            for package in self._packages
            if (version := self.installed_version(package.name)) is not None
        }
        candidates = [package for package in self._packages if package.name in present]
        if not candidates:
            return ()
        completed = self._pip(
            "install",
            "--upgrade",
            "--dry-run",
            "--quiet",
            "--report",
            "-",
            *_specs(candidates),
        )
        if completed.returncode != 0:
            get_logger().warning(
                "pip could not check for updates: {}", _tail(completed)
            )
            raise PackageUpdateError(
                "CaptionForge could not check for updates. Check your internet "
                "connection and try again.",
                details=_tail(completed),
            )
        try:
            report = json.loads(completed.stdout)
            newest = {
                _normalize(entry["metadata"]["name"]): str(entry["metadata"]["version"])
                for entry in report.get("install", [])
                if entry.get("requested")
            }
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise PackageUpdateError(
                "CaptionForge could not read pip's answer about updates.",
                details=str(exc),
            ) from exc
        return tuple(
            AvailableUpdate(package, present[package.name], newest[key])
            for package in candidates
            if (key := _normalize(package.name)) in newest
            and newest[key] != present[package.name]
        )

    def _require_pip(self) -> None:
        if not self.can_update():
            raise PackageUpdateError(
                "This installation of CaptionForge has no pip to update with."
            )

    def _pip(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        # pip retries a dead connection five times, 15 seconds apiece, by
        # default. One short attempt already says enough: not right now.
        command = (
            sys.executable,
            "-m",
            "pip",
            *arguments,
            "--disable-pip-version-check",
            "--no-input",
            "--retries",
            "1",
            "--timeout",
            "20",
        )
        try:
            return self._runner(command)
        except (OSError, subprocess.SubprocessError) as exc:
            raise PackageUpdateError(
                "pip could not be started to handle the update.", details=str(exc)
            ) from exc

    @staticmethod
    def _activate(package: Package) -> bool:
        """Whether the new release is in use now, or only after a restart."""
        if package.module not in sys.modules:
            # Nothing has imported it yet, so the next import reads the new one.
            return True
        if package.reloads:
            return _reload(package.module)
        return False


def _reload(module: str) -> bool:
    """Swap a loaded module for the release now on disk.

    Objects already made from the old release keep working; anything created
    from here on comes from the new one.
    """
    prefix = f"{module}."
    previous = {
        name: loaded
        for name, loaded in sys.modules.items()
        if name == module or name.startswith(prefix)
    }
    for name in previous:
        del sys.modules[name]
    importlib.invalidate_caches()
    try:
        importlib.import_module(module)
    except Exception:
        get_logger().exception(
            "The updated {} could not be loaded; keeping the running release", module
        )
        for name in [
            name for name in sys.modules if name == module or name.startswith(prefix)
        ]:
            del sys.modules[name]
        sys.modules.update(previous)
        return False
    return True


def _specs(packages: Iterable[Package]) -> list[str]:
    return [f"{package.name}{package.requirement}" for package in packages]


def _normalize(name: str) -> str:
    """Compare distribution names the way pip does (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _tail(completed: subprocess.CompletedProcess[str]) -> str:
    return (completed.stderr or completed.stdout or "").strip()[-2000:]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=PIP_TIMEOUT_SECONDS,
    )


# One per process, so the page, its jobs, and the terminal share one lock and
# one remembered check.
UPDATER = PackageUpdater()
