"""Tests for SignalExecutor.process_from_candles.

Public seam introduced so signal-threshold logic is testable without wiring
up a fetcher, DataNormalizer, or staleness check (Candidate 1 refactor).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.orchestration.cycle_stats import CycleStats
from finalayze.orchestration.signal_executor import SignalExecutor, _SignalContext

_NOW = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)
_CLOSE = Decimal(150)
_SYMBOL = "SBER"
_SEG_ID = "ru_blue_chips"
_MARKET = "moex"


def _make_candle(close: Decimal = _CLOSE) -> Candle:
    return Candle(
        symbol=_SYMBOL,
        market_id=_MARKET,
        timeframe="1d",
        timestamp=_NOW,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000,
    )


def _make_signal(
    direction: SignalDirection = SignalDirection.BUY, confidence: float = 0.75
) -> Signal:
    return Signal(
        strategy_name="momentum",
        symbol=_SYMBOL,
        market_id=_MARKET,
        segment_id=_SEG_ID,
        direction=direction,
        confidence=confidence,
        features={},
        reasoning="test",
    )


def _make_executor(
    *,
    signal_return: Signal | None = None,
    has_open_position: bool = False,
    exited_symbols: set[str] | None = None,
) -> SignalExecutor:
    executor = SignalExecutor.__new__(SignalExecutor)

    tracker = MagicMock()
    tracker.exited_symbols = exited_symbols or set()
    tracker.check_stop_losses.return_value = None
    tracker.maybe_register_retroactive_stop.return_value = None
    executor._position_tracker = tracker

    sentiment = MagicMock()
    sentiment.get_sentiment.return_value = 0.0
    executor._sentiment_mgr = sentiment

    broker = MagicMock()
    broker.has_position.return_value = has_open_position
    router = MagicMock()
    router.route.return_value = broker
    executor._broker_router = router

    strategy = MagicMock()
    strategy.generate_signal.return_value = signal_return
    executor._strategy = strategy

    executor._persistence = None
    executor._metrics = None

    return executor


class TestSignalBelowThreshold:
    def test_none_signal_sets_dropped_below_threshold(self) -> None:
        executor = _make_executor(signal_return=None)

        result = executor.process_from_candles([_make_candle()], _SYMBOL, _SEG_ID, _MARKET)

        assert isinstance(result, CycleStats)
        assert result.dropped_below_threshold == 1
        assert result.signals_generated == 0

    def test_none_signal_matches_factory(self) -> None:
        executor = _make_executor(signal_return=None)

        result = executor.process_from_candles([_make_candle()], _SYMBOL, _SEG_ID, _MARKET)

        assert result == CycleStats.signal_dropped_threshold()


class TestValidSignal:
    def test_valid_buy_returns_signal_context(self) -> None:
        signal = _make_signal()
        executor = _make_executor(signal_return=signal)

        result = executor.process_from_candles([_make_candle()], _SYMBOL, _SEG_ID, _MARKET)

        assert isinstance(result, _SignalContext)
        assert result.signal is signal
        assert result.symbol == _SYMBOL
        assert result.seg_id == _SEG_ID

    def test_sell_signal_proceeds_when_already_positioned(self) -> None:
        signal = _make_signal(direction=SignalDirection.SELL)
        executor = _make_executor(signal_return=signal, has_open_position=True)

        result = executor.process_from_candles([_make_candle()], _SYMBOL, _SEG_ID, _MARKET)

        assert isinstance(result, _SignalContext)


class TestEarlyExitGuards:
    def test_stopped_out_symbol_skipped(self) -> None:
        executor = _make_executor(
            signal_return=_make_signal(),
            exited_symbols={_SYMBOL},
        )

        result = executor.process_from_candles([_make_candle()], _SYMBOL, _SEG_ID, _MARKET)

        assert isinstance(result, CycleStats)
        assert result.signals_generated == 0

    def test_buy_skipped_when_already_positioned(self) -> None:
        executor = _make_executor(
            signal_return=_make_signal(direction=SignalDirection.BUY),
            has_open_position=True,
        )

        result = executor.process_from_candles([_make_candle()], _SYMBOL, _SEG_ID, _MARKET)

        assert isinstance(result, CycleStats)
        assert result.signals_generated == 0

    def test_retroactive_stop_registered_when_positioned(self) -> None:
        executor = _make_executor(signal_return=None, has_open_position=True)
        candles = [_make_candle()]

        executor.process_from_candles(candles, _SYMBOL, _SEG_ID, _MARKET)

        executor._position_tracker.maybe_register_retroactive_stop.assert_called_once_with(
            _SYMBOL, candles, _MARKET
        )

    def test_retroactive_stop_not_called_when_not_positioned(self) -> None:
        executor = _make_executor(signal_return=None, has_open_position=False)

        executor.process_from_candles([_make_candle()], _SYMBOL, _SEG_ID, _MARKET)

        executor._position_tracker.maybe_register_retroactive_stop.assert_not_called()
