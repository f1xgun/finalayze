"""Unit tests for core modules."""

from __future__ import annotations

import pytest

from finalayze import __version__
from finalayze.core.exceptions import (
    BrokerError,
    CircuitBreakerError,
    ConfigurationError,
    DataFetchError,
    FinalayzeError,
    InsufficientDataError,
    MarketClosedError,
    RiskViolationError,
)


class TestVersion:
    def test_version_is_set(self) -> None:
        assert __version__ == "0.1.0"


class TestExceptions:
    def test_base_exception(self) -> None:
        with pytest.raises(FinalayzeError):
            raise FinalayzeError("test error")

    def test_configuration_error_inherits(self) -> None:
        with pytest.raises(FinalayzeError):
            raise ConfigurationError("bad config")

    def test_data_fetch_error_inherits(self) -> None:
        with pytest.raises(FinalayzeError):
            raise DataFetchError("fetch failed")

    def test_broker_error_inherits(self) -> None:
        with pytest.raises(FinalayzeError):
            raise BrokerError("broker down")

    def test_risk_violation_inherits(self) -> None:
        with pytest.raises(FinalayzeError):
            raise RiskViolationError("position too large")

    def test_circuit_breaker_inherits(self) -> None:
        with pytest.raises(FinalayzeError):
            raise CircuitBreakerError("L2 triggered")

    def test_insufficient_data_inherits(self) -> None:
        with pytest.raises(FinalayzeError):
            raise InsufficientDataError("need more candles")

    def test_market_closed_inherits(self) -> None:
        with pytest.raises(FinalayzeError):
            raise MarketClosedError("MOEX closed")
