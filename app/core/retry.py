"""Selective retry support for temporary domain failures."""

import time
from collections.abc import Callable

from app.core.exceptions import CaptionForgeError
from app.core.logging_config import get_logger


def retry_call[T](
    operation: Callable[[], T],
    *,
    attempts: int,
    delay_seconds: float,
    operation_name: str,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run an operation again only when its translated error is retryable."""
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except CaptionForgeError as exc:
            if not exc.retryable or attempt >= attempts:
                raise
            get_logger().warning(
                "Temporary failure operation={} retry_attempt={} max_attempts={} "
                "technical_cause={}",
                operation_name,
                attempt,
                attempts,
                exc.details or type(exc).__name__,
            )
            sleep(delay_seconds)
    raise AssertionError("unreachable")
