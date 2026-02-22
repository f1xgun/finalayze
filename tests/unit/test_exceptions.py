"""Unit tests for the Finalayze exception hierarchy.

Tests verify:
- Inheritance relationships (all exceptions descend from FinalayzeError)
- Message storage
- Subclass groupings (e.g., MarketError, ExecutionError)
"""

from __future__ import annotations

import pytest

from finalayze.core.exceptions import (
    BrokerError,
    ConfigurationError,
    DataFetchError,
    ExecutionError,
    FinalayzeError,
    InstrumentNotFoundError,
    MarketError,
    MarketNotFoundError,
    ModeError,
    RateLimitError,
    RiskCheckError,
    StrategyError,
)

# ---------------------------------------------------------------------------
# Constants (no magic strings in tests)
# ---------------------------------------------------------------------------
MSG_BASE = "base error occurred"
MSG_CONFIG = "invalid database URL"
MSG_MARKET = "market operation failed"
MSG_MARKET_NOT_FOUND = "market 'XNAS' not found in registry"
MSG_INSTRUMENT_NOT_FOUND = "symbol 'AAPL' not in registry"
MSG_DATA_FETCH = "HTTP request to data provider failed"
MSG_RATE_LIMIT = "rate limit exceeded: 429 Too Many Requests"
MSG_STRATEGY = "RSI calculation failed: insufficient data"
MSG_RISK_CHECK = "order blocked: position exceeds max_position_pct"
MSG_EXECUTION = "order submission returned status 500"
MSG_BROKER = "Alpaca broker rejected order: insufficient funds"
MSG_MODE = "cannot transition to REAL mode without confirmation"


# ---------------------------------------------------------------------------
# FinalayzeError (base)
# ---------------------------------------------------------------------------
class TestFinalayzeError:
    def test_is_exception(self) -> None:
        err = FinalayzeError(MSG_BASE)
        assert isinstance(err, Exception)

    def test_message_stored(self) -> None:
        err = FinalayzeError(MSG_BASE)
        assert str(err) == MSG_BASE

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(FinalayzeError, match=MSG_BASE):
            raise FinalayzeError(MSG_BASE)


# ---------------------------------------------------------------------------
# ConfigurationError
# ---------------------------------------------------------------------------
class TestConfigurationError:
    def test_is_finalayze_error(self) -> None:
        err = ConfigurationError(MSG_CONFIG)
        assert isinstance(err, FinalayzeError)

    def test_message_stored(self) -> None:
        err = ConfigurationError(MSG_CONFIG)
        assert str(err) == MSG_CONFIG

    def test_caught_as_finalayze_error(self) -> None:
        with pytest.raises(FinalayzeError):
            raise ConfigurationError(MSG_CONFIG)


# ---------------------------------------------------------------------------
# MarketError (base for market-related errors)
# ---------------------------------------------------------------------------
class TestMarketError:
    def test_is_finalayze_error(self) -> None:
        err = MarketError(MSG_MARKET)
        assert isinstance(err, FinalayzeError)

    def test_message_stored(self) -> None:
        err = MarketError(MSG_MARKET)
        assert str(err) == MSG_MARKET


# ---------------------------------------------------------------------------
# MarketNotFoundError
# ---------------------------------------------------------------------------
class TestMarketNotFoundError:
    def test_is_market_error(self) -> None:
        err = MarketNotFoundError(MSG_MARKET_NOT_FOUND)
        assert isinstance(err, MarketError)

    def test_is_finalayze_error(self) -> None:
        err = MarketNotFoundError(MSG_MARKET_NOT_FOUND)
        assert isinstance(err, FinalayzeError)

    def test_message_stored(self) -> None:
        err = MarketNotFoundError(MSG_MARKET_NOT_FOUND)
        assert str(err) == MSG_MARKET_NOT_FOUND

    def test_caught_as_market_error(self) -> None:
        with pytest.raises(MarketError):
            raise MarketNotFoundError(MSG_MARKET_NOT_FOUND)


# ---------------------------------------------------------------------------
# InstrumentNotFoundError
# ---------------------------------------------------------------------------
class TestInstrumentNotFoundError:
    def test_is_market_error(self) -> None:
        err = InstrumentNotFoundError(MSG_INSTRUMENT_NOT_FOUND)
        assert isinstance(err, MarketError)

    def test_is_finalayze_error(self) -> None:
        err = InstrumentNotFoundError(MSG_INSTRUMENT_NOT_FOUND)
        assert isinstance(err, FinalayzeError)

    def test_message_stored(self) -> None:
        err = InstrumentNotFoundError(MSG_INSTRUMENT_NOT_FOUND)
        assert str(err) == MSG_INSTRUMENT_NOT_FOUND

    def test_caught_as_market_error(self) -> None:
        with pytest.raises(MarketError):
            raise InstrumentNotFoundError(MSG_INSTRUMENT_NOT_FOUND)


# ---------------------------------------------------------------------------
# DataFetchError
# ---------------------------------------------------------------------------
class TestDataFetchError:
    def test_is_finalayze_error(self) -> None:
        err = DataFetchError(MSG_DATA_FETCH)
        assert isinstance(err, FinalayzeError)

    def test_message_stored(self) -> None:
        err = DataFetchError(MSG_DATA_FETCH)
        assert str(err) == MSG_DATA_FETCH


# ---------------------------------------------------------------------------
# RateLimitError
# ---------------------------------------------------------------------------
class TestRateLimitError:
    def test_is_data_fetch_error(self) -> None:
        err = RateLimitError(MSG_RATE_LIMIT)
        assert isinstance(err, DataFetchError)

    def test_is_finalayze_error(self) -> None:
        err = RateLimitError(MSG_RATE_LIMIT)
        assert isinstance(err, FinalayzeError)

    def test_message_stored(self) -> None:
        err = RateLimitError(MSG_RATE_LIMIT)
        assert str(err) == MSG_RATE_LIMIT

    def test_caught_as_data_fetch_error(self) -> None:
        with pytest.raises(DataFetchError):
            raise RateLimitError(MSG_RATE_LIMIT)


# ---------------------------------------------------------------------------
# StrategyError
# ---------------------------------------------------------------------------
class TestStrategyError:
    def test_is_finalayze_error(self) -> None:
        err = StrategyError(MSG_STRATEGY)
        assert isinstance(err, FinalayzeError)

    def test_message_stored(self) -> None:
        err = StrategyError(MSG_STRATEGY)
        assert str(err) == MSG_STRATEGY


# ---------------------------------------------------------------------------
# RiskCheckError
# ---------------------------------------------------------------------------
class TestRiskCheckError:
    def test_is_finalayze_error(self) -> None:
        err = RiskCheckError(MSG_RISK_CHECK)
        assert isinstance(err, FinalayzeError)

    def test_message_stored(self) -> None:
        err = RiskCheckError(MSG_RISK_CHECK)
        assert str(err) == MSG_RISK_CHECK


# ---------------------------------------------------------------------------
# ExecutionError
# ---------------------------------------------------------------------------
class TestExecutionError:
    def test_is_finalayze_error(self) -> None:
        err = ExecutionError(MSG_EXECUTION)
        assert isinstance(err, FinalayzeError)

    def test_message_stored(self) -> None:
        err = ExecutionError(MSG_EXECUTION)
        assert str(err) == MSG_EXECUTION


# ---------------------------------------------------------------------------
# BrokerError
# ---------------------------------------------------------------------------
class TestBrokerError:
    def test_is_execution_error(self) -> None:
        err = BrokerError(MSG_BROKER)
        assert isinstance(err, ExecutionError)

    def test_is_finalayze_error(self) -> None:
        err = BrokerError(MSG_BROKER)
        assert isinstance(err, FinalayzeError)

    def test_message_stored(self) -> None:
        err = BrokerError(MSG_BROKER)
        assert str(err) == MSG_BROKER

    def test_caught_as_execution_error(self) -> None:
        with pytest.raises(ExecutionError):
            raise BrokerError(MSG_BROKER)


# ---------------------------------------------------------------------------
# ModeError
# ---------------------------------------------------------------------------
class TestModeError:
    def test_is_finalayze_error(self) -> None:
        err = ModeError(MSG_MODE)
        assert isinstance(err, FinalayzeError)

    def test_message_stored(self) -> None:
        err = ModeError(MSG_MODE)
        assert str(err) == MSG_MODE

    def test_caught_as_finalayze_error(self) -> None:
        with pytest.raises(FinalayzeError):
            raise ModeError(MSG_MODE)
