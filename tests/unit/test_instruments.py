"""Unit tests for the instrument registry (Layer 2)."""

from __future__ import annotations

import pytest

from finalayze.core.exceptions import InstrumentNotFoundError
from finalayze.markets.instruments import (
    Instrument,
    InstrumentRegistry,
    build_default_registry,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AAPL_SYMBOL = "AAPL"
MSFT_SYMBOL = "MSFT"
US_MARKET = "us"
UNKNOWN_SYMBOL = "UNKN"
UNKNOWN_MARKET = "unknown"

EXPECTED_DEFAULT_SYMBOLS = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ"}
EXPECTED_DEFAULT_COUNT = 7
EXPECTED_COUNT_AFTER_TWO = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_instrument(
    symbol: str = AAPL_SYMBOL,
    market_id: str = US_MARKET,
    name: str = "Apple Inc.",
) -> Instrument:
    return Instrument(symbol=symbol, market_id=market_id, name=name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_register_and_get() -> None:
    registry = InstrumentRegistry()
    instrument = make_instrument()
    registry.register(instrument)
    result = registry.get(AAPL_SYMBOL, US_MARKET)
    assert result is instrument


def test_get_raises_for_unknown() -> None:
    registry = InstrumentRegistry()
    with pytest.raises(InstrumentNotFoundError):
        registry.get(UNKNOWN_SYMBOL, UNKNOWN_MARKET)


def test_list_by_market_returns_sorted() -> None:
    registry = InstrumentRegistry()
    # Register in reverse order
    registry.register(make_instrument(symbol=MSFT_SYMBOL, name="Microsoft Corporation"))
    registry.register(make_instrument(symbol=AAPL_SYMBOL, name="Apple Inc."))
    results = registry.list_by_market(US_MARKET)
    assert [i.symbol for i in results] == [AAPL_SYMBOL, MSFT_SYMBOL]


def test_list_by_market_excludes_inactive() -> None:
    registry = InstrumentRegistry()
    active = Instrument(symbol=AAPL_SYMBOL, market_id=US_MARKET, name="Apple Inc.", is_active=True)
    inactive = Instrument(
        symbol=MSFT_SYMBOL, market_id=US_MARKET, name="Microsoft", is_active=False
    )
    registry.register(active)
    registry.register(inactive)
    results = registry.list_by_market(US_MARKET)
    symbols = [i.symbol for i in results]
    assert AAPL_SYMBOL in symbols
    assert MSFT_SYMBOL not in symbols


def test_build_default_registry_has_us_instruments() -> None:
    registry = build_default_registry()
    for symbol in EXPECTED_DEFAULT_SYMBOLS:
        instrument = registry.get(symbol, US_MARKET)
        assert instrument.market_id == US_MARKET


def test_build_default_registry_has_expected_count() -> None:
    registry = build_default_registry()
    assert len(registry) == EXPECTED_DEFAULT_COUNT


def test_register_overwrites() -> None:
    registry = InstrumentRegistry()
    original = Instrument(symbol=AAPL_SYMBOL, market_id=US_MARKET, name="Apple Original")
    updated = Instrument(symbol=AAPL_SYMBOL, market_id=US_MARKET, name="Apple Updated")
    registry.register(original)
    registry.register(updated)
    result = registry.get(AAPL_SYMBOL, US_MARKET)
    assert result.name == "Apple Updated"


def test_len() -> None:
    registry = InstrumentRegistry()
    assert len(registry) == 0
    registry.register(make_instrument(symbol=AAPL_SYMBOL))
    assert len(registry) == 1
    registry.register(make_instrument(symbol=MSFT_SYMBOL))
    assert len(registry) == EXPECTED_COUNT_AFTER_TWO
