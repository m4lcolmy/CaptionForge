"""Run CaptionForge as a desktop application: a native window, no terminal.

The window is drawn with Qt and calls CaptionForge's services directly, in the
same process: no local server, no port, no browser. It looks and behaves like
the page ``captionforge web`` serves, section for section, and it starts from
the applications list once ``captionforge install-desktop`` has added it.

Importing this package never imports Qt, so the rest of CaptionForge works
without the ``desktop`` extra installed.
"""

from __future__ import annotations

from app.core.config import Config
from app.core.exceptions import ConfigurationError

QT_PACKAGES = frozenset({"PySide6", "shiboken6"})


def launch(config: Config) -> bool:
    """Open the window and return once it is done.

    Returns True when CaptionForge was already open and its window was brought
    forward instead of opening a second one.
    """
    try:
        from app.interfaces.desktop.application import run
    except ImportError as exc:
        if (exc.name or "").split(".")[0] not in QT_PACKAGES:
            raise
        raise ConfigurationError(
            "The desktop app needs Qt. Install it with "
            'pip install "captionforge[desktop]"',
            details=str(exc),
        ) from exc
    return run(config)


def main() -> None:
    """Entry point for the ``captionforge-desktop`` launcher."""
    from app.core.logging_config import configure_logging

    config = Config.load()
    configure_logging(config)
    launch(config)


__all__ = ["launch", "main"]
