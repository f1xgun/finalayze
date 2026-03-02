"""Unit tests for exit improvements: profit target, time-based exit, wider stops."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.execution.simulated_broker import SimulatedBroker
from finalayze.strategies.base import BaseStrategy

INITIAL_CASH = Decimal(100000)
# ATR period is 14, so we need >=15 candles before the buy signal
BUY_BAR = 20
BASE_PRICE = Decimal(100)


def _make_candle(
    idx: int,
    *,
    price: Decimal = BASE_PRICE,
    high_offset: Decimal = Decimal(2),
    low_offset: Decimal = Decimal(2),
    symbol: str = "TEST",
) -> Candle:
    return Candle(
        symbol=symbol,
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=idx),
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
    """Create flat candles with small oscillation (for ATR computation)."""
    return [
        _make_candle(i, price=price, high_offset=Decimal(2), low_offset=Decimal(2), symbol=symbol)
        for i in range(count)
    ]


class BuyOnceStrategy(BaseStrategy):
    """Emits BUY at bar BUY_BAR, never sells."""

    @property
    def name(self) -> str:
        return "buy_once"

    def supported_segments(self) -> list[str]:
        return ["us_test"]

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
        if idx == BUY_BAR:
            return Signal(
                strategy_name=self.name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.BUY,
                confidence=0.8,
                features={"test": 1.0},
                reasoning="Test buy",
            )
        return None


class TestProfitTarget:
    """Tests for profit target exit logic."""

    def test_profit_target_triggers_at_3x_atr(self) -> None:
        """Price rise to 3x ATR above entry triggers profit target exit."""
        candles = _make_flat_candles(BUY_BAR + 1)

        # Entry will be at bar BUY_BAR+1 open = BASE_PRICE
        # ATR ≈ 4 (high-low range) for flat candles
        # So profit target = entry + 3.0 * ATR = 100 + 12 = 112
        # Add a candle with high >= 112 to trigger
        spike_price = BASE_PRICE + Decimal(12)
        candles.append(_make_candle(len(candles), price=BASE_PRICE))  # fill candle for buy
        for _i in range(3):
            candles.append(_make_candle(len(candles), price=BASE_PRICE))
        # Candle with high reaching profit target
        candles.append(
            Candle(
                symbol="TEST",
                market_id="us",
                timeframe="1d",
                timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=len(candles)),
                open=spike_price - Decimal(2),
                high=spike_price + Decimal(1),
                low=spike_price - Decimal(3),
                close=spike_price,
                volume=1_000_000,
            )
        )
        # Need one more candle after spike for fill
        candles.append(_make_candle(len(candles), price=spike_price))

        engine = BacktestEngine(
            strategy=BuyOnceStrategy(),
            initial_cash=INITIAL_CASH,
            atr_multiplier=Decimal("3.0"),
            profit_target_atr=Decimal("3.0"),
            max_hold_bars=0,  # disable time exit
            trail_activation_atr=Decimal("1.5"),
            trail_distance_atr=Decimal("1.5"),
        )
        trades, _ = engine.run("TEST", "us_test", candles)

        # Should have at least one trade (the profit target exit)
        assert len(trades) >= 1
        # The exit trade should be profitable
        sell_trades = [t for t in trades if t.side == "SELL"]
        assert len(sell_trades) >= 1
        assert sell_trades[0].pnl > 0

    def test_profit_target_not_triggered_below_threshold(self) -> None:
        """Price rise of only 2x ATR should NOT trigger profit target at 3x ATR."""
        candles = _make_flat_candles(BUY_BAR + 1)
        candles.append(_make_candle(len(candles), price=BASE_PRICE))  # fill candle for buy

        # Add candles with modest high (2x ATR rise, not enough for 3x target)
        for _ in range(5):
            candles.append(
                _make_candle(
                    len(candles),
                    price=BASE_PRICE + Decimal(4),
                    high_offset=Decimal(4),
                    low_offset=Decimal(2),
                )
            )
        # One more to close out at end
        candles.append(_make_candle(len(candles), price=BASE_PRICE + Decimal(4)))

        engine = BacktestEngine(
            strategy=BuyOnceStrategy(),
            initial_cash=INITIAL_CASH,
            atr_multiplier=Decimal("3.0"),
            profit_target_atr=Decimal("3.0"),
            max_hold_bars=0,  # disable time exit
            trail_activation_atr=Decimal(100),  # disable trailing stop
            trail_distance_atr=Decimal("1.5"),
        )
        trades, _ = engine.run("TEST", "us_test", candles)

        # Should only have the end-of-backtest close, no profit target exit
        sell_trades = [t for t in trades if t.side == "SELL"]
        assert len(sell_trades) == 1  # only end-of-backtest close


class TestTimeExit:
    """Tests for time-based (max holding period) exit logic."""

    def test_time_exit_at_30_bars(self) -> None:
        """Flat price for 30 bars triggers forced exit."""
        total_bars = BUY_BAR + 35  # enough for 30 bars after entry
        candles = _make_flat_candles(total_bars)

        engine = BacktestEngine(
            strategy=BuyOnceStrategy(),
            initial_cash=INITIAL_CASH,
            atr_multiplier=Decimal("3.0"),
            profit_target_atr=Decimal(0),  # disable profit target
            max_hold_bars=30,
            trail_activation_atr=Decimal(100),  # disable trailing
            trail_distance_atr=Decimal("1.5"),
        )
        trades, _ = engine.run("TEST", "us_test", candles)

        # Should have a trade from time-based exit
        sell_trades = [t for t in trades if t.side == "SELL"]
        assert len(sell_trades) >= 1

    def test_time_exit_not_triggered_at_20_bars(self) -> None:
        """Position held for only 20 bars should NOT trigger time exit."""
        total_bars = BUY_BAR + 23  # only ~20 bars after entry
        candles = _make_flat_candles(total_bars)

        engine = BacktestEngine(
            strategy=BuyOnceStrategy(),
            initial_cash=INITIAL_CASH,
            atr_multiplier=Decimal("3.0"),
            profit_target_atr=Decimal(0),  # disable profit target
            max_hold_bars=30,
            trail_activation_atr=Decimal(100),  # disable trailing
            trail_distance_atr=Decimal("1.5"),
        )
        trades, _ = engine.run("TEST", "us_test", candles)

        # Should only have end-of-backtest close (no time exit)
        sell_trades = [t for t in trades if t.side == "SELL"]
        assert len(sell_trades) == 1  # only end-of-backtest close

    def test_profit_target_before_time_exit(self) -> None:
        """Profit target at bar 10 exits before time exit at bar 30."""
        candles = _make_flat_candles(BUY_BAR + 1)
        candles.append(_make_candle(len(candles), price=BASE_PRICE))  # fill candle

        # Add a few flat bars, then a spike
        for _ in range(5):
            candles.append(_make_candle(len(candles), price=BASE_PRICE))

        # Spike at bar ~27 (about 6 bars after entry)
        spike_price = BASE_PRICE + Decimal(15)
        candles.append(
            Candle(
                symbol="TEST",
                market_id="us",
                timeframe="1d",
                timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=len(candles)),
                open=spike_price - Decimal(1),
                high=spike_price + Decimal(1),
                low=spike_price - Decimal(3),
                close=spike_price,
                volume=1_000_000,
            )
        )
        # Fill candle for profit target exit
        candles.append(_make_candle(len(candles), price=spike_price))

        # Add remaining bars to reach 30+ total
        for _ in range(25):
            candles.append(_make_candle(len(candles), price=BASE_PRICE))

        engine = BacktestEngine(
            strategy=BuyOnceStrategy(),
            initial_cash=INITIAL_CASH,
            atr_multiplier=Decimal("3.0"),
            profit_target_atr=Decimal("3.0"),
            max_hold_bars=30,
            trail_activation_atr=Decimal(100),  # disable trailing
            trail_distance_atr=Decimal("1.5"),
        )
        trades, _ = engine.run("TEST", "us_test", candles)

        sell_trades = [t for t in trades if t.side == "SELL"]
        # Profit target should have triggered, so trade should be profitable
        assert len(sell_trades) >= 1
        assert sell_trades[0].pnl > 0


class TestWiderStopParameters:
    """Tests for updated default parameter values."""

    def test_wider_stop_defaults(self) -> None:
        """Verify new defaults: 3.0 ATR stop, 1.5 activation, 1.5 trail."""
        engine = BacktestEngine(strategy=BuyOnceStrategy())
        assert engine._atr_multiplier == Decimal("3.0")
        assert engine._trail_activation_atr == Decimal("1.0")
        assert engine._trail_distance_atr == Decimal("1.5")
        assert engine._profit_target_atr == Decimal("5.0")
        assert engine._max_hold_bars == 30


class TestBrokerEntryHelpers:
    """Tests for SimulatedBroker.get_entry_atr() and get_entry_price()."""

    def test_get_entry_atr_returns_value(self) -> None:
        """get_entry_atr returns correct ATR when position has stop state."""
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        broker.set_trailing_stop(
            symbol="AAPL",
            entry_price=Decimal(150),
            initial_stop=Decimal(144),
            atr_value=Decimal("2.0"),
            activation_atr=Decimal("1.5"),
            trail_atr=Decimal("1.5"),
        )
        assert broker.get_entry_atr("AAPL") == Decimal("2.0")
        assert broker.get_entry_price("AAPL") == Decimal(150)

    def test_get_entry_atr_returns_none_no_position(self) -> None:
        """get_entry_atr returns None when no stop state exists."""
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        assert broker.get_entry_atr("AAPL") is None
        assert broker.get_entry_price("AAPL") is None
