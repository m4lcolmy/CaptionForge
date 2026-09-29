"""Install CaptionForge into the desktop, so it starts from an icon.

Every platform gets the same thing: an entry in the applications list that runs
``captionforge desktop`` from the folder CaptionForge was installed in, with no
terminal window in between.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.constants import APP_NAME, VERSION
from app.core.exceptions import ConfigurationError
from app.core.logging_config import get_logger

ENTRY_NAME = "captionforge"
ICON_SOURCE = Path(__file__).parent / "assets" / "captionforge.svg"
SUMMARY = "Download YouTube captions, transcribe locally, save MP4 or MP3"


@dataclass(frozen=True)
class Installation:
    """What an install wrote, and what the entry will run."""

    paths: tuple[Path, ...]
    workdir: Path
    command: tuple[str, ...]


def launch_command() -> tuple[str, ...]:
    """The command a desktop entry runs, using this very interpreter."""
    return (interpreter(), "-m", "app", "desktop")


def interpreter() -> str:
    """Prefer the console-free interpreter where one exists (Windows)."""
    executable = Path(sys.executable)
    quiet = executable.with_name("pythonw.exe")
    if os.name == "nt" and quiet.is_file():
        return str(quiet)
    return str(executable)


def install(workdir: Path | None = None) -> Installation:
    """Add CaptionForge to this computer's applications."""
    folder = (workdir or Path.cwd()).expanduser().resolve()
    command = launch_command()
    system = platform.system()
    if system == "Linux":
        paths = _install_linux(folder, command)
    elif system == "Darwin":
        paths = _install_macos(folder, command)
    elif system == "Windows":
        paths = _install_windows(folder, command)
    else:
        raise ConfigurationError(
            f"CaptionForge cannot add a desktop entry on {system}. "
            "Run 'captionforge desktop' to open the app instead."
        )
    get_logger().info("Installed the desktop entry into {}", folder)
    return Installation(paths=paths, workdir=folder, command=command)


def uninstall() -> tuple[Path, ...]:
    """Remove whatever a previous install wrote, and report what went."""
    removed: list[Path] = []
    for path in _installed_paths():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
        elif path.exists():
            path.unlink()
            removed.append(path)
    _refresh_linux_caches()
    return tuple(removed)


def _installed_paths() -> tuple[Path, ...]:
    """Every path an install on this platform is responsible for."""
    system = platform.system()
    if system == "Linux":
        return (_desktop_entry_path(), _icon_path())
    if system == "Darwin":
        return (_bundle_path(),)
    if system == "Windows":
        return (_shortcut_path(), _shortcut_path().with_suffix(".cmd"))
    return ()


# ---------- Linux ----------


def _data_home() -> Path:
    """The user's data directory, honouring XDG_DATA_HOME."""
    base = os.environ.get("XDG_DATA_HOME")
    return Path(base).expanduser() if base else Path.home() / ".local" / "share"


def _desktop_entry_path() -> Path:
    """Where the applications list looks for this user's own entries."""
    return _data_home() / "applications" / f"{ENTRY_NAME}.desktop"


def _icon_path() -> Path:
    """Where a themed icon named ``captionforge`` is found."""
    return (
        _data_home() / "icons" / "hicolor" / "scalable" / "apps" / f"{ENTRY_NAME}.svg"
    )


def desktop_entry(workdir: Path, command: tuple[str, ...]) -> str:
    """Render the .desktop file that starts CaptionForge without a terminal."""
    return "\n".join(
        (
            "[Desktop Entry]",
            "Type=Application",
            f"Name={APP_NAME}",
            "GenericName=Subtitles and transcripts",
            f"Comment={SUMMARY}",
            f"Exec={_exec_line(command)}",
            f"Path={workdir}",
            f"Icon={ENTRY_NAME}",
            "Terminal=false",
            "Categories=AudioVideo;Audio;Video;",
            "Keywords=subtitle;caption;transcript;youtube;whisper;srt;vtt;",
            "StartupNotify=true",
            # The window's WM_CLASS instance, so the dock shows this icon for it.
            f"StartupWMClass={ENTRY_NAME}",
            # One window only: a second launch brings the first one forward.
            "SingleMainWindow=true",
            f"X-CaptionForge-Version={VERSION}",
            "",
        )
    )


def _exec_line(command: tuple[str, ...]) -> str:
    """Quote a command the way the desktop-entry specification expects."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _install_linux(workdir: Path, command: tuple[str, ...]) -> tuple[Path, ...]:
    """Write the icon and the .desktop entry, then refresh the menu caches."""
    icon = _icon_path()
    icon.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ICON_SOURCE, icon)

    entry = _desktop_entry_path()
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(desktop_entry(workdir, command), encoding="utf-8")
    entry.chmod(0o755)
    _refresh_linux_caches()
    return (entry, icon)


def _refresh_linux_caches() -> None:
    """Ask the desktop to notice the change now instead of at next login."""
    if platform.system() != "Linux":
        return
    _run_quietly(["update-desktop-database", str(_desktop_entry_path().parent)])
    _run_quietly(
        ["gtk-update-icon-cache", "-f", "-t", str(_data_home() / "icons" / "hicolor")]
    )


def _run_quietly(command: list[str]) -> None:
    """Run an optional refresh tool, ignoring both absence and failure."""
    if shutil.which(command[0]) is None:
        return
    try:
        subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        get_logger().debug("Cache refresh skipped: {}", exc)


# ---------- macOS ----------


def _bundle_path() -> Path:
    """Where a per-user application bundle belongs."""
    return Path.home() / "Applications" / f"{APP_NAME}.app"


def _install_macos(workdir: Path, command: tuple[str, ...]) -> tuple[Path, ...]:
    """Write the smallest bundle Finder and Spotlight will accept."""
    bundle = _bundle_path()
    macos = bundle / "Contents" / "MacOS"
    resources = bundle / "Contents" / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    (bundle / "Contents" / "Info.plist").write_text(_info_plist(), encoding="utf-8")
    script = macos / APP_NAME
    quoted = " ".join(f'"{part}"' for part in command)
    script.write_text(
        f'#!/bin/sh\ncd "{workdir}" || exit 1\nexec {quoted}\n', encoding="utf-8"
    )
    script.chmod(0o755)
    shutil.copyfile(ICON_SOURCE, resources / f"{ENTRY_NAME}.svg")
    return (bundle,)


def _info_plist() -> str:
    """The bundle metadata Finder reads to name and version the app."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        f"  <key>CFBundleName</key><string>{APP_NAME}</string>\n"
        f"  <key>CFBundleDisplayName</key><string>{APP_NAME}</string>\n"
        f"  <key>CFBundleExecutable</key><string>{APP_NAME}</string>\n"
        f"  <key>CFBundleIdentifier</key><string>org.captionforge.app</string>\n"
        f"  <key>CFBundleShortVersionString</key><string>{VERSION}</string>\n"
        "  <key>CFBundlePackageType</key><string>APPL</string>\n"
        "  <key>LSMinimumSystemVersion</key><string>11.0</string>\n"
        "  <key>NSHighResolutionCapable</key><true/>\n"
        "</dict>\n"
        "</plist>\n"
    )


# ---------- Windows ----------


def _start_menu() -> Path:
    """The current user's Start Menu programs folder."""
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return root / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def _shortcut_path() -> Path:
    """Where the Start Menu shortcut is written."""
    return _start_menu() / f"{APP_NAME}.lnk"


def _install_windows(workdir: Path, command: tuple[str, ...]) -> tuple[Path, ...]:
    """Create a Start Menu shortcut, falling back to a batch file."""
    shortcut = _shortcut_path()
    shortcut.parent.mkdir(parents=True, exist_ok=True)
    arguments = subprocess.list2cmdline(command[1:])
    script = (
        "$link = (New-Object -ComObject WScript.Shell)"
        f".CreateShortcut('{shortcut}');"
        f"$link.TargetPath = '{command[0]}';"
        f"$link.Arguments = '{arguments}';"
        f"$link.WorkingDirectory = '{workdir}';"
        f"$link.Description = '{SUMMARY}';"
        "$link.Save()"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0 and shortcut.exists():
        return (shortcut,)

    batch = shortcut.with_suffix(".cmd")
    batch.write_text(
        f'@echo off\r\ncd /d "{workdir}"\r\nstart "" "{command[0]}" {arguments}\r\n',
        encoding="utf-8",
    )
    return (batch,)
