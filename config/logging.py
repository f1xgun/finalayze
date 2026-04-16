"""Structured logging configuration.

See docs/architecture/OVERVIEW.md for logging conventions.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

from config.modes import WorkMode


def _drop_grpc_poller_noise(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor: drop benign gRPC PollerCompletionQueue BlockingIOError.

    These fire dozens of times per strategy cycle on every event loop that
    gRPC touches (including uvicorn's main loop where we cannot install a
    custom asyncio exception handler).  They are harmless EAGAIN retries.
    """
    event = event_dict.get("event", "")
    if "PollerCompletionQueue" in str(event) and "BlockingIOError" in str(
        event_dict.get("exc_info", "")
    ):
        raise structlog.DropEvent
    return event_dict


class _GrpcPollerFilter(logging.Filter):
    """Stdlib fallback: suppress BlockingIOError on the asyncio logger."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            exc = record.exc_info[1]
            if isinstance(exc, BlockingIOError):
                return False
        msg = record.getMessage()
        return "PollerCompletionQueue" not in msg or "BlockingIOError" not in msg


def setup_logging(mode: WorkMode) -> None:
    """Configure structured JSON logging based on work mode."""
    level = logging.DEBUG if mode == WorkMode.DEBUG else logging.INFO

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _drop_grpc_poller_noise,  # type: ignore[list-item]
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Stdlib fallback for gRPC noise that bypasses structlog
    logging.getLogger("asyncio").addFilter(_GrpcPollerFilter())
