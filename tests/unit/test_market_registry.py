"""Unit tests for market registry (Layer 2)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, time

import pytest

from finalayze.core.exceptions import MarketNotFoundError
from finalayze.markets.registry import (
    MarketDefinition,
    MarketRegistry,
    default_registry,
)

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────

EXPECTED_MARKET_COUNT = 2

US_OPEN_HOUR = 9
US_OPEN_MINUTE = 30
US_CLOSE_HOUR = 16
US_CLOSE_MINUTE = 0

MOEX_OPEN_HOUR = 10
MOEX_OPEN_MINUTE = 0
MOEX_CLOSE_HOUR = 18
MOEX_CLOSE_MINUTE = 40

# UTC datetimes for is_market_open tests
# Monday 2024-01-15 15:00 UTC = 10:00 ET -> US is open (after 9:30)
US_OPEN_DT = datetime(2024, 1, 15, 15, 0, tzinfo=UTC)
# Monday 2024-01-15 22:00 UTC = 17:00 ET -> US is closed (after 16:00)
US_CLOSED_DT = datetime(2024, 1, 15, 22, 0, tzinfo=UTC)
# Saturday 2024-01-13 15:00 UTC -> always closed (weekend)
WEEKEND_DT = datetime(2024, 1, 13, 15, 0, tzinfo=UTC)


# ── MarketDefinition ────────────────────────────────────────────────────


class TestMarketDefinition:
    def test_create_market(self) -> None:
        m = MarketDefinition(
            id="test",
            name="Test Market",
            currency="TST",
            timezone="UTC",
            open_time=time(9, 0),
            close_time=time(17, 0),
        )
        assert m.id == "test"
        assert m.name == "Test Market"
        assert m.currency == "TST"
        assert m.timezone == "UTC"
        assert m.open_time == time(9, 0)
        assert m.close_time == time(17, 0)

    def test_market_is_frozen(self) -> None:
        m = MarketDefinition(
            id="test",
            name="Test Market",
            currency="TST",
            timezone="UTC",
            open_time=time(9, 0),
            close_time=time(17, 0),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.id = "changed"  # type: ignore[misc]


# ── MarketRegistry ──────────────────────────────────────────────────────


class TestMarketRegistry:
    @pytest.fixture
    def registry(self) -> MarketRegistry:
        return default_registry()

    def test_get_us_market(self, registry: MarketRegistry) -> None:
        us = registry.get_market("us")
        assert us.id == "us"
        assert us.name == "US Stock Market"
        assert us.currency == "USD"
        assert us.timezone == "America/New_York"
        assert us.open_time == time(US_OPEN_HOUR, US_OPEN_MINUTE)
        assert us.close_time == time(US_CLOSE_HOUR, US_CLOSE_MINUTE)

    def test_get_moex_market(self, registry: MarketRegistry) -> None:
        moex = registry.get_market("moex")
        assert moex.id == "moex"
        assert moex.name == "Moscow Exchange"
        assert moex.currency == "RUB"
        assert moex.timezone == "Europe/Moscow"
        assert moex.open_time == time(MOEX_OPEN_HOUR, MOEX_OPEN_MINUTE)
        assert moex.close_time == time(MOEX_CLOSE_HOUR, MOEX_CLOSE_MINUTE)

    def test_get_unknown_market_raises(self, registry: MarketRegistry) -> None:
        with pytest.raises(MarketNotFoundError):
            registry.get_market("unknown")

    def test_list_markets(self, registry: MarketRegistry) -> None:
        markets = registry.list_markets()
        assert len(markets) == EXPECTED_MARKET_COUNT

    def test_is_market_open_during_hours(self, registry: MarketRegistry) -> None:
        assert registry.is_market_open("us", at=US_OPEN_DT) is True

    def test_is_market_closed_outside_hours(self, registry: MarketRegistry) -> None:
        assert registry.is_market_open("us", at=US_CLOSED_DT) is False

    def test_is_market_closed_on_weekend(self, registry: MarketRegistry) -> None:
        assert registry.is_market_open("us", at=WEEKEND_DT) is False
