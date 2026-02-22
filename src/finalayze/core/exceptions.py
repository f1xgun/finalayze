"""Exception hierarchy for Finalayze (Layer 0).

All custom exceptions inherit from ``FinalayzeError``.
Exception names end in ``Error`` per ruff N818.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations


class FinalayzeError(Exception):
    """Base exception for all Finalayze errors."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ConfigurationError(FinalayzeError):
    """Raised for invalid or missing configuration values."""


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------
class MarketError(FinalayzeError):
    """Base class for market-related errors."""


class MarketNotFoundError(MarketError):
    """Market ID not found in the market registry."""


class InstrumentNotFoundError(MarketError):
    """Symbol / instrument not found in the market registry."""


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
class DataFetchError(FinalayzeError):
    """Data fetching from an external provider failed."""


class RateLimitError(DataFetchError):
    """External API rate limit was hit."""


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class StrategyError(FinalayzeError):
    """Strategy computation or validation error."""


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
class RiskCheckError(FinalayzeError):
    """Trade blocked by a risk management check."""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
class ExecutionError(FinalayzeError):
    """Order submission or execution failed."""


class BrokerError(ExecutionError):
    """Broker-specific error (e.g. insufficient funds, order rejected)."""


# ---------------------------------------------------------------------------
# Mode management
# ---------------------------------------------------------------------------
class ModeError(FinalayzeError):
    """Invalid mode transition or missing confirmation for real trading."""


# ---------------------------------------------------------------------------
# Legacy aliases kept for backward-compat with existing tests in test_core.py
# ---------------------------------------------------------------------------
class RiskViolationError(FinalayzeError):
    """Order rejected by risk management (legacy alias)."""


class CircuitBreakerError(FinalayzeError):
    """Trading halted by circuit breaker (legacy alias)."""


class InsufficientDataError(FinalayzeError):
    """Not enough data to compute indicator or make decision (legacy alias)."""


class MarketClosedError(FinalayzeError):
    """Attempted operation outside market hours (legacy alias)."""
