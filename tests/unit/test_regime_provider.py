"""Tests for RegimeProvider protocol and BacktestEngine regime integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.risk.regime import (
    MarketRegime,
    RegimeState,
    StaticRegimeProvider,
)
from finalayze.strategies.base import BaseStrategy

_INITIAL_CASH = Decimal(100_000)
_CANDLE_COUNT = 40
_BUY_BAR = 30
_SELL_BAR = 35


def _make_candles(count: int = _CANDLE_COUNT) -> list[Candle]:
    """Create an upward-trending candle series suitable for backtest."""
    base = Decimal(100)
    return [
        Candle(
            symbol="TEST",
            market_id="us",
            timeframe="1d",
            timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=i),
            open=base + Decimal(i),
            high=base + Decimal(i) + Decimal(2),
            low=base + Decimal(i) - Decimal(2),
            close=base + Decimal(i) + Decimal(1),
            volume=1_000_000,
        )
        for i in range(count)
    ]


class _BuySellStrategy(BaseStrategy):
    """Emits BUY at bar _BUY_BAR, SELL at bar _SELL_BAR."""

    @property
    def name(self) -> str:
        return "buy_sell_stub"

    def supported_segments(self) -> list[str]:
        return ["us_large_cap"]

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        **kwargs: object,
    ) -> Signal | None:
        idx = len(candles) - 1
        if idx == _BUY_BAR:
            return Signal(
                strategy_name=self.name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.BUY,
                confidence=0.8,
                features={"m": 1.0},
                reasoning="buy",
            )
        if idx == _SELL_BAR:
            return Signal(
                strategy_name=self.name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.SELL,
                confidence=0.8,
                features={"m": -1.0},
                reasoning="sell",
            )
        return None


# ---------------------------------------------------------------------------
# 1. StaticRegimeProvider unit tests
# ---------------------------------------------------------------------------


class TestStaticRegimeProvider:
    """Tests for StaticRegimeProvider."""

    def test_static_regime_provider_returns_fixed(self) -> None:
        """StaticRegimeProvider always returns the configured state."""
        state = RegimeState(
            regime=MarketRegime.ELEVATED,
            allow_new_longs=True,
            position_scale=0.5,
        )
        provider = StaticRegimeProvider(state)
        candles = _make_candles(5)

        result = provider.get_regime(candles, 0)
        assert result is state

        result2 = provider.get_regime(candles, 4)
        assert result2 is state

    def test_regime_state_normal_factory(self) -> None:
        """RegimeState.normal() returns expected defaults."""
        state = RegimeState.normal()
        assert state.regime == MarketRegime.NORMAL
        assert state.allow_new_longs is True
        assert state.position_scale == 1.0

    def test_regime_state_crisis_factory(self) -> None:
        """RegimeState.crisis() returns expected defaults."""
        state = RegimeState.crisis()
        assert state.regime == MarketRegime.CRISIS
        assert state.allow_new_longs is False
        assert state.position_scale == Decimal("0.10")


# ---------------------------------------------------------------------------
# 2. BacktestEngine with no regime provider (unchanged behavior)
# ---------------------------------------------------------------------------


class TestEngineWithoutRegimeProvider:
    """Engine works normally when regime_provider is None."""

    def test_engine_without_regime_provider(self) -> None:
        """Without regime_provider, BUY signals execute normally."""
        strategy = _BuySellStrategy()
        engine = BacktestEngine(strategy, initial_cash=_INITIAL_CASH)
        candles = _make_candles()

        trades, _snapshots = engine.run("TEST", "us_large_cap", candles)

        # Should have at least 1 trade (the BUY/SELL pair)
        assert len(trades) >= 1
        # Verify a buy actually happened by checking entry prices
        buy_trades = [t for t in trades if t.entry_price > 0]
        assert len(buy_trades) >= 1


# ---------------------------------------------------------------------------
# 3. BacktestEngine with crisis regime (blocks buys)
# ---------------------------------------------------------------------------


class TestEngineWithCrisisRegime:
    """Crisis regime blocks BUY signals."""

    def test_engine_with_crisis_regime_blocks_buys(self) -> None:
        """StaticRegimeProvider(CRISIS) prevents BUY signals from executing."""
        strategy = _BuySellStrategy()
        crisis_state = RegimeState.crisis()
        provider = StaticRegimeProvider(crisis_state)
        engine = BacktestEngine(
            strategy,
            initial_cash=_INITIAL_CASH,
            regime_provider=provider,
        )
        candles = _make_candles()

        trades, _snapshots = engine.run("TEST", "us_large_cap", candles)

        # No trades should occur because BUY was blocked and SELL has no
        # position to close.
        assert len(trades) == 0

    def test_crisis_regime_preserves_cash(self) -> None:
        """When crisis blocks buys, cash should remain at initial level."""
        strategy = _BuySellStrategy()
        provider = StaticRegimeProvider(RegimeState.crisis())
        engine = BacktestEngine(
            strategy,
            initial_cash=_INITIAL_CASH,
            regime_provider=provider,
        )
        candles = _make_candles()

        _trades, snapshots = engine.run("TEST", "us_large_cap", candles)

        # Cash unchanged since no trades occurred
        assert snapshots[-1].cash == _INITIAL_CASH


# ---------------------------------------------------------------------------
# 4. BacktestEngine with normal regime (allows buys)
# ---------------------------------------------------------------------------


class TestEngineWithNormalRegime:
    """Normal regime allows trading normally."""

    def test_engine_with_normal_regime_allows_buys(self) -> None:
        """StaticRegimeProvider(NORMAL) allows BUY signals to execute."""
        strategy = _BuySellStrategy()
        normal_state = RegimeState.normal()
        provider = StaticRegimeProvider(normal_state)
        engine = BacktestEngine(
            strategy,
            initial_cash=_INITIAL_CASH,
            regime_provider=provider,
        )
        candles = _make_candles()

        trades, _snapshots = engine.run("TEST", "us_large_cap", candles)

        # At least one trade should have occurred
        assert len(trades) >= 1


# ---------------------------------------------------------------------------
# 5. Regime position scaling
# ---------------------------------------------------------------------------


class TestRegimePositionScaling:
    """Regime position_scale affects position sizing."""

    def test_elevated_regime_scales_position_down(self) -> None:
        """An ELEVATED regime with position_scale=0.5 results in smaller
        positions compared to NORMAL with position_scale=1.0."""
        strategy = _BuySellStrategy()
        candles = _make_candles()

        # Run with normal regime (scale=1.0)
        normal_provider = StaticRegimeProvider(RegimeState.normal())
        engine_normal = BacktestEngine(
            strategy,
            initial_cash=_INITIAL_CASH,
            regime_provider=normal_provider,
        )
        trades_normal, _ = engine_normal.run("TEST", "us_large_cap", candles)

        # Run with elevated regime (scale=0.5, still allows longs)
        elevated_state = RegimeState(
            regime=MarketRegime.ELEVATED,
            allow_new_longs=True,
            position_scale=0.5,
        )
        elevated_provider = StaticRegimeProvider(elevated_state)
        engine_elevated = BacktestEngine(
            strategy,
            initial_cash=_INITIAL_CASH,
            regime_provider=elevated_provider,
        )
        trades_elevated, _ = engine_elevated.run("TEST", "us_large_cap", candles)

        # Both should have trades
        assert len(trades_normal) >= 1
        assert len(trades_elevated) >= 1

        # Elevated regime should have smaller quantity
        normal_qty = trades_normal[0].quantity
        elevated_qty = trades_elevated[0].quantity
        assert elevated_qty < normal_qty


# ---------------------------------------------------------------------------
# 6. Portfolio-level regime integration
# ---------------------------------------------------------------------------


class TestPortfolioRegimeIntegration:
    """Regime provider works in run_portfolio() as well."""

    def test_portfolio_crisis_blocks_buys(self) -> None:
        """Crisis regime blocks buys in run_portfolio() too."""
        strategy = _BuySellStrategy()
        provider = StaticRegimeProvider(RegimeState.crisis())
        engine = BacktestEngine(
            strategy,
            initial_cash=_INITIAL_CASH,
            regime_provider=provider,
        )
        candles = _make_candles()

        trades, _ = engine.run_portfolio(
            symbols=["TEST"],
            segment_id="us_large_cap",
            candles_by_symbol={"TEST": candles},
        )

        assert len(trades) == 0
