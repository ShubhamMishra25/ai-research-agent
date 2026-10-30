"""Bounded retry helpers for transient tool and LLM failures."""
import time
from typing import Callable, TypeVar

from app.utils.logging_config import setup_logger

logger = setup_logger("utils.retry")

T = TypeVar("T")


def with_retries(
    operation: Callable[[], T],
    *,
    attempts: int,
    backoff_seconds: float,
    description: str,
    retryable: Callable[[Exception], bool] = lambda _exc: True,
) -> T:
    """Run ``operation`` with bounded exponential backoff.

    ``attempts`` is the total number of tries (including the first one).
    The original exception is re-raised once retries are exhausted or the
    failure is classified as non-retryable.
    """
    total_attempts = max(1, attempts)
    last_exc: Exception | None = None

    for attempt in range(1, total_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - callers decide retryability
            last_exc = exc
            if attempt >= total_attempts or not retryable(exc):
                raise
            delay = backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Retrying %s after %s (attempt %d/%d) in %.2fs",
                description,
                type(exc).__name__,
                attempt,
                total_attempts,
                delay,
            )
            time.sleep(delay)

    # Defensive: the loop always returns or raises.
    assert last_exc is not None
    raise last_exc
