"""Tests for 6D.8: Verify all 8 modules use structlog instead of stdlib logging."""

from __future__ import annotations

import logging

import structlog

_MIGRATED_MODULES = [
    "finalayze.core.trading_loop",
    "finalayze.core.alerts",
    "finalayze.ml.loader",
    "finalayze.strategies.ml_strategy",
    "finalayze.execution.retry",
    "finalayze.data.normalizer",
    "finalayze.api.v1.system",
    "finalayze.api.v1.portfolio",
]


def test_no_stdlib_logging_getlogger() -> None:
    """None of the migrated modules should use logging.getLogger."""
    import importlib

    for module_name in _MIGRATED_MODULES:
        mod = importlib.import_module(module_name)
        # Check that no module-level variable is a stdlib Logger
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, logging.Logger):
                msg = (
                    f"{module_name}.{attr_name} is a stdlib logging.Logger, "
                    "should be a structlog logger"
                )
                raise AssertionError(msg)


def test_structlog_loggers_present() -> None:
    """Each migrated module should have a structlog-backed _log."""
    import importlib

    for module_name in _MIGRATED_MODULES:
        mod = importlib.import_module(module_name)
        log_attr_names = ["_log", "logger"]
        found = False
        for name in log_attr_names:
            if hasattr(mod, name):
                log_obj = getattr(mod, name)
                # structlog loggers are NOT stdlib Logger instances
                assert not isinstance(log_obj, logging.Logger), (
                    f"{module_name}.{name} is still a stdlib Logger"
                )
                found = True
        assert found, f"{module_name} has no _log or logger attribute"
