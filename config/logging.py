"""Structured logging configuration.

See docs/architecture/OVERVIEW.md for logging conventions.
"""

from __future__ import annotations

import logging

import structlog

from config.modes import WorkMode


def setup_logging(mode: WorkMode) -> None:
    """Configure structured JSON logging based on work mode."""
    level = logging.DEBUG if mode == WorkMode.DEBUG else logging.INFO

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
