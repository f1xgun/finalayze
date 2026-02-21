"""Exception hierarchy for Finalayze (Layer 0).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations


class FinalayzeError(Exception):
    """Base exception for all Finalayze errors."""


class ConfigurationError(FinalayzeError):
    """Invalid configuration."""


class DataFetchError(FinalayzeError):
    """Error fetching data from an external source."""


class BrokerError(FinalayzeError):
    """Error communicating with a broker."""


class RiskViolationError(FinalayzeError):
    """Order rejected by risk management."""


class CircuitBreakerError(FinalayzeError):
    """Trading halted by circuit breaker."""


class InsufficientDataError(FinalayzeError):
    """Not enough data to compute indicator or make decision."""


class MarketClosedError(FinalayzeError):
    """Attempted operation outside market hours."""
