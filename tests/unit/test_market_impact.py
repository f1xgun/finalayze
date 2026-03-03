"""Tests for square-root market impact model (execution/impact.py)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.execution.broker_base import OrderRequest
from finalayze.execution.impact import compute_market_impact, should_reject_trade
from finalayze.execution.simulated_broker import SimulatedBroker

# --- Constants to avoid magic numbers (ruff PLR2004) ---
DAILY_VOL_2PCT = 0.02
SHARES_10K = 10_000.0
ADV_1M = 1_000_000.0
DEFAULT_COEFF = 0.1
EXPECTED_IMPACT_BASIC = DAILY_VOL_2PCT * math.sqrt(SHARES_10K / ADV_1M) * DEFAULT_COEFF


class TestComputeMarketImpact:
    """Tests for compute_market_impact."""

    def test_impact_basic(self) -> None:
        """Known-value test: 2% vol, 10k shares, 1M ADV, coeff=0.1."""
        result = compute_market_impact(
            daily_vol=DAILY_VOL_2PCT,
            shares=SHARES_10K,
            adv=ADV_1M,
            impact_coeff=DEFAULT_COEFF,
        )
        assert result == pytest.approx(EXPECTED_IMPACT_BASIC, rel=1e-9)

    def test_impact_zero_adv(self) -> None:
        """ADV <= 0 should return 0.0."""
        assert compute_market_impact(daily_vol=DAILY_VOL_2PCT, shares=SHARES_10K, adv=0.0) == 0.0
        assert compute_market_impact(daily_vol=DAILY_VOL_2PCT, shares=SHARES_10K, adv=-1.0) == 0.0

    def test_impact_zero_shares(self) -> None:
        """shares <= 0 should return 0.0."""
        assert compute_market_impact(daily_vol=DAILY_VOL_2PCT, shares=0.0, adv=ADV_1M) == 0.0
        assert compute_market_impact(daily_vol=DAILY_VOL_2PCT, shares=-100.0, adv=ADV_1M) == 0.0

    def test_impact_proportional_to_sqrt_participation(self) -> None:
        """Doubling shares should increase impact by sqrt(2), not 2x.

        impact(2N) / impact(N) = sqrt(2N/ADV) / sqrt(N/ADV) = sqrt(2)
        """
        shares_n = 10_000.0
        shares_2n = 20_000.0
        impact_n = compute_market_impact(daily_vol=DAILY_VOL_2PCT, shares=shares_n, adv=ADV_1M)
        impact_2n = compute_market_impact(daily_vol=DAILY_VOL_2PCT, shares=shares_2n, adv=ADV_1M)
        ratio = impact_2n / impact_n
        assert ratio == pytest.approx(math.sqrt(2), rel=1e-9)


class TestShouldRejectTrade:
    """Tests for should_reject_trade."""

    def test_should_reject_high_impact(self) -> None:
        """Large order relative to ADV should be rejected (>50bps)."""
        # Force a huge participation rate to exceed 50bps
        result = should_reject_trade(
            daily_vol=0.10,  # 10% daily vol
            shares=500_000.0,
            adv=100_000.0,  # 500% participation
            impact_coeff=DEFAULT_COEFF,
            max_impact_bps=50.0,
        )
        assert result is True

    def test_should_not_reject_small_impact(self) -> None:
        """Small order relative to ADV should pass."""
        result = should_reject_trade(
            daily_vol=DAILY_VOL_2PCT,
            shares=100.0,
            adv=ADV_1M,
            impact_coeff=DEFAULT_COEFF,
            max_impact_bps=50.0,
        )
        assert result is False


# --- Helper ---
_TS = datetime(2024, 1, 2, tzinfo=UTC)


def _make_candle(
    symbol: str = "AAPL",
    open_: str = "100.00",
    high: str = "101.00",
    low: str = "99.00",
    close: str = "100.50",
) -> Candle:
    return Candle(
        symbol=symbol,
        market_id="us",
        timeframe="1d",
        timestamp=_TS,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=1_000_000,
    )


class TestSimulatedBrokerImpactIntegration:
    """Tests for impact model integration in SimulatedBroker."""

    def test_buy_price_increased_by_impact(self) -> None:
        """BUY fill price should be higher than candle open when impact is applied."""
        broker = SimulatedBroker(
            Decimal(2000000),
            use_impact_model=True,
            adv={"AAPL": 1_000_000.0},
            daily_vol={"AAPL": 0.02},
        )
        candle = _make_candle()
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(10000))
        result = broker.submit_order(order, fill_candle=candle)

        assert result.filled is True
        assert result.fill_price is not None
        # Impact should push the buy fill_price above the candle open
        assert result.fill_price > Decimal("100.00")

    def test_sell_price_decreased_by_impact(self) -> None:
        """SELL fill price should be lower than candle open when impact is applied."""
        broker = SimulatedBroker(
            Decimal(1000000),
            use_impact_model=True,
            adv={"AAPL": 1_000_000.0},
            daily_vol={"AAPL": 0.02},
        )
        # First buy some shares (without impact for simplicity -- use a no-impact buy)
        broker._positions["AAPL"] = Decimal(10000)
        candle = _make_candle()
        order = OrderRequest(symbol="AAPL", side="SELL", quantity=Decimal(10000))
        result = broker.submit_order(order, fill_candle=candle)

        assert result.filled is True
        assert result.fill_price is not None
        assert result.fill_price < Decimal("100.00")

    def test_impact_rejects_high_impact_order(self) -> None:
        """Order with impact exceeding max_impact_bps should be rejected."""
        broker = SimulatedBroker(
            Decimal(10000000),
            use_impact_model=True,
            adv={"AAPL": 100_000.0},
            daily_vol={"AAPL": 0.10},
            max_impact_bps=50.0,
        )
        candle = _make_candle()
        # 500k shares vs 100k ADV = 500% participation -> huge impact
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(500000))
        result = broker.submit_order(order, fill_candle=candle)

        assert result.filled is False
        assert "impact" in result.reason.lower()

    def test_no_impact_when_disabled(self) -> None:
        """When use_impact_model=False (default), fill at exact candle open."""
        broker = SimulatedBroker(Decimal(1000000))
        candle = _make_candle()
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(100))
        result = broker.submit_order(order, fill_candle=candle)

        assert result.filled is True
        assert result.fill_price == Decimal("100.00")
