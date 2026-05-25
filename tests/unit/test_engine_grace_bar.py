"""Unit tests for grace bar + catastrophic override in BacktestEngine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from finalayze.backtest.engine import _CATASTROPHIC_DROP_PCT, _NO_ENTRY_BAR, BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

INITIAL_CASH = Decimal(100_000)
SYMBOL = "TEST"
SEGMENT = "us_large_cap"
ENTRY_PRICE = Decimal(100)
# BUY_DAY=20: signal fires at i=20, fill on candle[21] (Jan 22 = Monday, weekday).
# Entry_bar=20, grace bar at i=21 (hold_bars=1).
# Fill price = ENTRY_PRICE + 21 = 121. ATR ≈ 4. Stop ≈ 121 - 3*4 = 109.
# Catastrophic threshold = 121 * 0.85 ≈ 102.85.
BUY_DAY = 20
CANDLE_COUNT = 40
# Fill candle low that is below stop (~109) but above catastrophic threshold (~102.85)
_GRACE_BAR_LOW = Decimal(105)


def _make_candle(
    day: int,
    *,
    price: Decimal = ENTRY_PRICE,
    low: Decimal | None = None,
    symbol: str = SYMBOL,
) -> Candle:
    """Build a single candle at `day` offset from 2024-01-01."""
    effective_low = low if low is not None else price - Decimal(2)
    return Candle(
        symbol=symbol,
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=day),
        open=price,
        high=price + Decimal(2),
        low=effective_low,
        close=price + Decimal(1),
        volume=1_000_000,
    )


def _make_candle_series(
    count: int = CANDLE_COUNT,
    *,
    fill_candle_low: Decimal | None = None,
    symbol: str = SYMBOL,
) -> list[Candle]:
    """Create candles; optionally override the fill candle (BUY_DAY + 1) low."""
    candles: list[Candle] = []
    for i in range(count):
        price = ENTRY_PRICE + Decimal(i)
        low = None
        # The fill candle is the one AFTER the signal candle
        if fill_candle_low is not None and i == BUY_DAY + 1:
            low = fill_candle_low
        candles.append(_make_candle(i, price=price, low=low, symbol=symbol))
    return candles


class _BuyOnceStrategy(BaseStrategy):
    """Emits BUY at candle index BUY_DAY only."""

    @property
    def name(self) -> str:
        return "buy_once"

    def supported_segments(self) -> list[str]:
        return [SEGMENT]

    def generate_signal(  # type: ignore[override]
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        **kwargs: object,
    ) -> Signal | None:
        idx = len(candles) - 1
        if idx == BUY_DAY:
            return Signal(
                strategy_name=self.name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.BUY,
                confidence=0.8,
                strategy_payload={"test": 1.0},
                reasoning="test buy",
            )
        return None

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


# ---------------------------------------------------------------------------
# Single-symbol run() tests
# ---------------------------------------------------------------------------


class TestGraceBarSkipsStop:
    """Grace bar prevents stop-loss on the fill candle."""

    def test_grace_bar_skips_stop_on_fill_candle(self) -> None:
        """Fill candle with low < stop but above catastrophic -> grace bar holds."""
        # low=105 is below trailing stop (~109) but above catastrophic (~102.85)
        candles = _make_candle_series(fill_candle_low=_GRACE_BAR_LOW)
        engine = BacktestEngine(
            strategy=_BuyOnceStrategy(),
            initial_cash=INITIAL_CASH,
            atr_multiplier=Decimal("1.5"),
        )
        trades, _ = engine.run(symbol=SYMBOL, segment_id=SEGMENT, candles=candles)
        # Position should survive past the fill candle (hold_bars > 1)
        assert len(trades) >= 1, "Should have at least one trade"
        for t in trades:
            assert t.hold_bars is not None and t.hold_bars > 1, (
                f"Grace bar should prevent stop on fill candle, got hold_bars={t.hold_bars}"
            )


class TestCatastrophicOverride:
    """15% drop overrides grace bar."""

    def test_grace_bar_catastrophic_override_triggers(self) -> None:
        """20% drop on fill candle -> stop fires despite grace bar (hold_bars=1)."""
        entry_signal_price = ENTRY_PRICE + Decimal(BUY_DAY)  # 120
        catastrophic_low = entry_signal_price * (Decimal(1) - Decimal("0.20"))  # 96
        candles = _make_candle_series(fill_candle_low=catastrophic_low)
        engine = BacktestEngine(
            strategy=_BuyOnceStrategy(),
            initial_cash=INITIAL_CASH,
            atr_multiplier=Decimal("1.5"),
        )
        trades, _ = engine.run(symbol=SYMBOL, segment_id=SEGMENT, candles=candles)
        # Should be stopped on the grace bar (hold_bars=1) due to catastrophic drop
        grace_bar_stops = [t for t in trades if t.hold_bars == 1]
        assert len(grace_bar_stops) >= 1, "Catastrophic drop should override grace bar"

    def test_grace_bar_no_override_at_14pct(self) -> None:
        """14% drop on fill candle -> grace bar holds (below 15% threshold)."""
        entry_signal_price = ENTRY_PRICE + Decimal(BUY_DAY)  # 120
        # 14% drop from signal price; still below catastrophic for fill price (121)
        mild_low = entry_signal_price * (Decimal(1) - Decimal("0.14"))  # ~103.2
        candles = _make_candle_series(fill_candle_low=mild_low)
        engine = BacktestEngine(
            strategy=_BuyOnceStrategy(),
            initial_cash=INITIAL_CASH,
            atr_multiplier=Decimal("1.5"),
        )
        trades, _ = engine.run(symbol=SYMBOL, segment_id=SEGMENT, candles=candles)
        # Grace bar should hold -- no stop on fill candle (hold_bars=1)
        fill_candle_stops = [t for t in trades if t.hold_bars is not None and t.hold_bars <= 1]
        assert len(fill_candle_stops) == 0, "14% drop should not override grace bar"


# ---------------------------------------------------------------------------
# Portfolio mode tests
# ---------------------------------------------------------------------------


class TestPortfolioGraceBar:
    """Grace bar works in portfolio mode too."""

    def test_portfolio_grace_bar_skips_stop(self) -> None:
        """Portfolio mode: fill candle low < stop but above catastrophic -> grace bar holds."""
        candles = _make_candle_series(fill_candle_low=_GRACE_BAR_LOW)
        engine = BacktestEngine(
            strategy=_BuyOnceStrategy(),
            initial_cash=INITIAL_CASH,
            atr_multiplier=Decimal("1.5"),
        )
        trades, _ = engine.run_portfolio(
            symbols=[SYMBOL],
            candles_by_symbol={SYMBOL: candles},
            segment_id=SEGMENT,
        )
        assert len(trades) >= 1, "Should have at least one trade"
        for t in trades:
            assert t.hold_bars is not None and t.hold_bars > 1, (
                f"Portfolio grace bar should prevent stop on fill candle, "
                f"got hold_bars={t.hold_bars}"
            )

    def test_portfolio_catastrophic_override_triggers(self) -> None:
        """Portfolio mode: 20% drop on fill candle -> stop fires despite grace bar."""
        entry_signal_price = ENTRY_PRICE + Decimal(BUY_DAY)
        catastrophic_low = entry_signal_price * (Decimal(1) - Decimal("0.20"))
        candles = _make_candle_series(fill_candle_low=catastrophic_low)
        engine = BacktestEngine(
            strategy=_BuyOnceStrategy(),
            initial_cash=INITIAL_CASH,
            atr_multiplier=Decimal("1.5"),
        )
        trades, _ = engine.run_portfolio(
            symbols=[SYMBOL],
            candles_by_symbol={SYMBOL: candles},
            segment_id=SEGMENT,
        )
        grace_bar_stops = [t for t in trades if t.hold_bars == 1]
        assert len(grace_bar_stops) >= 1, "Portfolio catastrophic drop should override grace bar"


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestCatastrophicDropConstant:
    """Verify the catastrophic drop constant value."""

    def test_catastrophic_drop_is_15pct(self) -> None:
        assert Decimal("0.15") == _CATASTROPHIC_DROP_PCT

    def test_no_entry_bar_sentinel(self) -> None:
        assert _NO_ENTRY_BAR == -2
        # Grace bar math: _NO_ENTRY_BAR + 1 = -1, never matches valid bar index
        assert _NO_ENTRY_BAR + 1 < 0


class TestPortfolioUsesPublicUpdateStop:
    """Verify broker.update_stop_loss() is used, not broker._stop_states."""

    def test_no_private_stop_states_access(self) -> None:
        """Engine source should not access broker._stop_states directly."""
        import inspect

        from finalayze.backtest import engine

        source = inspect.getsource(engine)
        assert "_stop_states" not in source, (
            "engine.py should use broker.update_stop_loss() not broker._stop_states"
        )
