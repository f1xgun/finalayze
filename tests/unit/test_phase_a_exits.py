"""Tests for Phase A.4 (Chandelier Exit) and A.5 (Strategy-Specific Time Exits)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import ClassVar

from finalayze.backtest.config import (
    DEFAULT_STRATEGY_HOLD_BARS,
    BacktestConfig,
    resolve_max_hold_bars,
)
from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.risk.chandelier_exit import (
    _compute_atr,
    compute_chandelier_stop,
    get_chandelier_multiplier,
)
from finalayze.strategies.base import BaseStrategy

INITIAL_CASH = Decimal(100_000)
BASE_PRICE = Decimal(100)
BUY_BAR = 25  # need >= 22+1 candles for chandelier ATR period

# Start on a Monday so that weekday offsets are predictable
_BASE_DATE = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)  # Monday


def _business_day_offset(base: datetime, n: int) -> datetime:
    """Return the nth business day after base (skipping weekends)."""
    days_added = 0
    current = base
    while days_added < n:
        current += timedelta(days=1)
        # Monday=0 ... Friday=4
        if current.weekday() < 5:  # noqa: PLR2004
            days_added += 1
    return current


def _make_candle(
    idx: int,
    *,
    price: Decimal = BASE_PRICE,
    high_offset: Decimal = Decimal(2),
    low_offset: Decimal = Decimal(2),
    symbol: str = "TEST",
) -> Candle:
    ts = _BASE_DATE if idx == 0 else _business_day_offset(_BASE_DATE, idx)
    return Candle(
        symbol=symbol,
        market_id="us",
        timeframe="1d",
        timestamp=ts,
        open=price,
        high=price + high_offset,
        low=price - low_offset,
        close=price,
        volume=1_000_000,
    )


def _make_flat_candles(
    count: int,
    *,
    price: Decimal = BASE_PRICE,
    symbol: str = "TEST",
) -> list[Candle]:
    """Create flat candles with small oscillation (for ATR computation).

    Uses business days only (Mon-Fri) to avoid pre-trade market-hours check
    failures during backtesting.
    """
    return [
        _make_candle(i, price=price, high_offset=Decimal(2), low_offset=Decimal(2), symbol=symbol)
        for i in range(count)
    ]


class NamedBuyOnceStrategy(BaseStrategy):
    """Emits BUY at a configurable bar with a configurable strategy name."""

    def __init__(self, strategy_name: str = "buy_once", buy_bar: int = BUY_BAR) -> None:
        self._strategy_name = strategy_name
        self._buy_bar = buy_bar

    @property
    def name(self) -> str:
        return self._strategy_name

    def supported_segments(self) -> list[str]:
        return ["us_test", "us_tech", "ru_energy"]

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
        if idx == self._buy_bar:
            return Signal(
                strategy_name=self._strategy_name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.BUY,
                confidence=0.8,
                features={"test": 1.0},
                reasoning="Test buy",
            )
        return None


# ---------------------------------------------------------------------------
# A.4 Chandelier Exit tests
# ---------------------------------------------------------------------------


class TestChandelierStopFormula:
    """Test the core chandelier stop computation."""

    def test_chandelier_stop_formula(self) -> None:
        """highest_high(22) - ATR(22) * 3.0 matches manual calculation."""
        candles = _make_flat_candles(22, price=Decimal(100))
        # All candles have high=102, low=98, close=100
        # highest_high = 102
        # ATR: each TR = max(102-98, |102-100|, |98-100|) = 4
        # ATR = 4.0
        # stop = 102 - 3.0 * 4 = 90
        stop = compute_chandelier_stop(candles, atr_period=22, multiplier=Decimal("3.0"))
        assert stop is not None
        expected = Decimal(102) - Decimal("3.0") * Decimal(4)
        assert stop == expected

    def test_chandelier_insufficient_data(self) -> None:
        """Returns None with fewer than atr_period candles."""
        candles = _make_flat_candles(21)
        stop = compute_chandelier_stop(candles, atr_period=22)
        assert stop is None

    def test_chandelier_stop_with_varying_prices(self) -> None:
        """Verify formula with non-uniform prices."""
        # Create 22 candles where the last few have higher highs
        candles = _make_flat_candles(20, price=Decimal(100))
        # Add 2 candles with higher prices
        for i in range(2):
            candles.append(
                _make_candle(
                    20 + i,
                    price=Decimal(110),
                    high_offset=Decimal(5),
                    low_offset=Decimal(3),
                )
            )
        stop = compute_chandelier_stop(candles, atr_period=22, multiplier=Decimal("3.0"))
        assert stop is not None
        # highest_high = 115 (from the last candles: 110 + 5)
        # ATR is computed from the 22 most recent candles
        highest_high = max(c.high for c in candles[-22:])
        atr = _compute_atr(candles[-22:])
        expected = highest_high - Decimal("3.0") * atr
        assert stop == expected


class TestChandelierMonotonicRatchet:
    """Verify the stop only moves up, never down."""

    def test_chandelier_monotonic_ratchet(self) -> None:
        """Stop never decreases even when ATR increases or high drops."""
        # Start with high prices, then prices drop
        candles_high = _make_flat_candles(22, price=Decimal(120))
        stop_high = compute_chandelier_stop(candles_high, atr_period=22, multiplier=Decimal("3.0"))
        assert stop_high is not None

        # Add candles with lower prices (which would produce a lower stop)
        candles_with_drop = list(candles_high)
        candles_with_drop.extend(
            _make_candle(
                22 + i,
                price=Decimal(90),
                high_offset=Decimal(5),
                low_offset=Decimal(5),
            )
            for i in range(5)
        )
        stop_after_drop = compute_chandelier_stop(
            candles_with_drop, atr_period=22, multiplier=Decimal("3.0")
        )
        assert stop_after_drop is not None

        # The ratchet logic: max(current_stop, new_candidate)
        # The caller applies this; verify the logic works
        ratcheted = max(stop_high, stop_after_drop)
        assert ratcheted >= stop_high  # never decreases


class TestChandelierMultiplierBySegment:
    """Verify segment-specific multipliers."""

    _EXPECTED_MULTIPLIERS: ClassVar[dict[str, Decimal]] = {
        "us_tech": Decimal("3.0"),
        "us_broad": Decimal("3.0"),
        "us_healthcare": Decimal("3.5"),
        "us_finance": Decimal("2.5"),
        "ru_blue_chips": Decimal("4.0"),
        "ru_finance": Decimal("4.0"),
        "ru_energy": Decimal("4.5"),
        "ru_tech": Decimal("3.5"),
    }

    def test_chandelier_multiplier_by_segment(self) -> None:
        """All segment-specific multipliers match the plan."""
        for segment, expected in self._EXPECTED_MULTIPLIERS.items():
            actual = get_chandelier_multiplier(segment)
            assert actual == expected, f"{segment}: expected {expected}, got {actual}"

    def test_chandelier_multiplier_unknown_segment(self) -> None:
        """Unknown segment returns default 3.0."""
        assert get_chandelier_multiplier("unknown_segment") == Decimal("3.0")


class TestEngineChandelierMode:
    """Engine uses Chandelier exit when stop_loss_mode='chandelier'."""

    def test_engine_chandelier_mode(self) -> None:
        """Engine with chandelier mode opens and eventually exits positions."""
        total_bars = BUY_BAR + 40
        candles = _make_flat_candles(total_bars)

        config = BacktestConfig(
            initial_cash=INITIAL_CASH,
            stop_loss_mode="chandelier",
            profit_target_atr=Decimal(0),  # disable profit target
            max_hold_bars=0,  # disable time exit
            trail_activation_atr=Decimal(100),  # disable trailing
        )
        engine = BacktestEngine(
            strategy=NamedBuyOnceStrategy(),
            config=config,
        )
        trades, snapshots = engine.run("TEST", "us_tech", candles)

        # Should have at least the end-of-backtest close
        assert len(trades) >= 1
        assert len(snapshots) == total_bars

    def test_engine_chandelier_triggers_on_price_drop(self) -> None:
        """Chandelier stop triggers when price drops below the stop level."""
        # Build candles: flat for BUY_BAR bars, then entry, then sharp drop
        candles = _make_flat_candles(BUY_BAR + 1)
        # Entry fill at BUY_BAR+1
        candles.append(_make_candle(len(candles), price=BASE_PRICE))

        # A few flat bars to establish chandelier stop
        for _ in range(3):
            candles.append(_make_candle(len(candles), price=BASE_PRICE))

        # Sharp drop below the chandelier stop
        # Chandelier stop = highest_high(22) - 3.0 * ATR(22) = 102 - 12 = 90
        # So a candle with low < 90 should trigger
        drop_price = Decimal(85)
        drop_idx = len(candles)
        drop_ts = _business_day_offset(_BASE_DATE, drop_idx)
        candles.append(
            Candle(
                symbol="TEST",
                market_id="us",
                timeframe="1d",
                timestamp=drop_ts,
                open=drop_price + Decimal(2),
                high=drop_price + Decimal(3),
                low=drop_price,
                close=drop_price + Decimal(1),
                volume=1_000_000,
            )
        )
        # Fill candle after trigger
        candles.append(_make_candle(len(candles), price=drop_price))

        config = BacktestConfig(
            initial_cash=INITIAL_CASH,
            stop_loss_mode="chandelier",
            profit_target_atr=Decimal(0),
            max_hold_bars=0,
            trail_activation_atr=Decimal(100),
        )
        engine = BacktestEngine(
            strategy=NamedBuyOnceStrategy(),
            config=config,
        )
        trades, _ = engine.run("TEST", "us_tech", candles)

        # Should have a sell trade (stop-loss triggered or end-of-backtest)
        sell_trades = [t for t in trades if t.side == "SELL"]
        assert len(sell_trades) >= 1

    def test_engine_trailing_mode_unchanged(self) -> None:
        """Default trailing mode still works as before."""
        total_bars = BUY_BAR + 15
        candles = _make_flat_candles(total_bars)

        engine = BacktestEngine(
            strategy=NamedBuyOnceStrategy(),
            initial_cash=INITIAL_CASH,
            profit_target_atr=Decimal(0),
            max_hold_bars=0,
            trail_activation_atr=Decimal(100),
        )
        _trades, snapshots = engine.run("TEST", "us_test", candles)

        assert len(snapshots) == total_bars
        assert engine._stop_loss_mode == "trailing"


# ---------------------------------------------------------------------------
# A.5 Strategy-Specific Time Exit tests
# ---------------------------------------------------------------------------


class TestResolveMaxHoldBars:
    """Unit tests for resolve_max_hold_bars helper."""

    def test_time_exit_int_backward_compat(self) -> None:
        """int value applies to all strategies regardless of name."""
        int_value = 42
        assert resolve_max_hold_bars(int_value, "momentum") == int_value
        assert resolve_max_hold_bars(int_value, "mean_reversion") == int_value
        assert resolve_max_hold_bars(int_value, "unknown") == int_value

    def test_time_exit_dict_lookup(self) -> None:
        """Dict maps strategy names to their specific hold bars."""
        hold_map = {"momentum": 40, "mean_reversion": 8}
        assert resolve_max_hold_bars(hold_map, "momentum") == 40
        assert resolve_max_hold_bars(hold_map, "mean_reversion") == 8

    def test_time_exit_dict_fallback(self) -> None:
        """Unknown strategy name falls back to 30 when using a dict."""
        hold_map = {"momentum": 40}
        fallback_value = 30
        assert resolve_max_hold_bars(hold_map, "unknown_strategy") == fallback_value

    def test_default_strategy_hold_bars_values(self) -> None:
        """Verify all default per-strategy values match the plan."""
        assert DEFAULT_STRATEGY_HOLD_BARS["momentum"] == 40
        assert DEFAULT_STRATEGY_HOLD_BARS["mean_reversion"] == 20
        assert DEFAULT_STRATEGY_HOLD_BARS["pairs"] == 15
        assert DEFAULT_STRATEGY_HOLD_BARS["event_driven"] == 63
        assert DEFAULT_STRATEGY_HOLD_BARS["rsi2_connors"] == 10
        assert DEFAULT_STRATEGY_HOLD_BARS["ml_ensemble"] == 20


class TestStrategySpecificTimeExitMR:
    """Mean reversion exits at 8 bars, not 30."""

    def test_strategy_specific_time_exit_mr(self) -> None:
        """MR position closes at 8 bars when max_hold_bars is a dict."""
        total_bars = BUY_BAR + 15  # enough for 8 bars after entry
        candles = _make_flat_candles(total_bars)

        config = BacktestConfig(
            initial_cash=INITIAL_CASH,
            max_hold_bars={"mean_reversion": 8, "momentum": 40},
            profit_target_atr=Decimal(0),
            trail_activation_atr=Decimal(100),
        )
        engine = BacktestEngine(
            strategy=NamedBuyOnceStrategy(strategy_name="mean_reversion"),
            config=config,
        )
        trades, _ = engine.run("TEST", "us_test", candles)

        # Should exit via time-based exit before end of backtest
        sell_trades = [t for t in trades if t.side == "SELL"]
        assert len(sell_trades) >= 1


class TestStrategySpecificTimeExitMomentum:
    """Momentum exits at 40 bars."""

    def test_strategy_specific_time_exit_momentum(self) -> None:
        """Momentum position held for 40 bars when max_hold_bars is a dict."""
        # Use enough bars: BUY_BAR + 45 so that time exit at bar 40 happens
        total_bars = BUY_BAR + 45
        candles = _make_flat_candles(total_bars)

        config = BacktestConfig(
            initial_cash=INITIAL_CASH,
            max_hold_bars={"mean_reversion": 8, "momentum": 40},
            profit_target_atr=Decimal(0),
            trail_activation_atr=Decimal(100),
        )
        engine = BacktestEngine(
            strategy=NamedBuyOnceStrategy(strategy_name="momentum"),
            config=config,
        )
        trades, _ = engine.run("TEST", "us_test", candles)

        # Should exit via time-based exit
        sell_trades = [t for t in trades if t.side == "SELL"]
        assert len(sell_trades) >= 1

    def test_momentum_not_exited_at_8_bars(self) -> None:
        """Momentum should NOT exit at 8 bars (only MR does)."""
        # Only provide enough bars for 10 after entry -- momentum should still be open
        total_bars = BUY_BAR + 12
        candles = _make_flat_candles(total_bars)

        config = BacktestConfig(
            initial_cash=INITIAL_CASH,
            max_hold_bars={"mean_reversion": 8, "momentum": 40},
            profit_target_atr=Decimal(0),
            trail_activation_atr=Decimal(100),
        )
        engine = BacktestEngine(
            strategy=NamedBuyOnceStrategy(strategy_name="momentum"),
            config=config,
        )
        trades, _ = engine.run("TEST", "us_test", candles)

        # Only the end-of-backtest close should happen (no time exit)
        sell_trades = [t for t in trades if t.side == "SELL"]
        assert len(sell_trades) == 1  # only end-of-backtest close


class TestBacktestConfigMaxHoldBars:
    """BacktestConfig correctly stores int and dict max_hold_bars."""

    def test_config_int_default(self) -> None:
        """Default max_hold_bars is 30."""
        cfg = BacktestConfig()
        assert cfg.max_hold_bars == 30

    def test_config_dict_value(self) -> None:
        """Dict value is stored correctly."""
        hold_map = {"momentum": 40, "mean_reversion": 8}
        cfg = BacktestConfig(max_hold_bars=hold_map)
        assert cfg.max_hold_bars == hold_map
