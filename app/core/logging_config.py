"""Centralized Loguru configuration."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from app.core.constants import LOG_FILE_NAME, LOG_RETENTION, LOG_ROTATION
from app.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from app.core.config import Config

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def configure_logging(config: "Config", logs_directory: Path = Path("logs")) -> Any:
    """Configure console and rotating file handlers and return the shared logger."""
    try:
        logs_directory.mkdir(parents=True, exist_ok=True)
        logger.remove()
        logger.add(
            logs_directory / LOG_FILE_NAME,
            level=config.logging_level,
            format=LOG_FORMAT,
            rotation=LOG_ROTATION,
            retention=LOG_RETENTION,
            encoding="utf-8",
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
    except OSError as exc:
        raise ConfigurationError(
            "Unable to configure application logging", details=str(exc)
        ) from exc
    return logger


def get_logger() -> Any:
    """Return the shared Loguru logger for use by application modules."""
    return logger
