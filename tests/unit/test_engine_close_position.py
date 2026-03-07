"""Unit tests for BacktestEngine._close_position() helper and _NO_ENTRY_BAR constant."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

from finalayze.backtest.engine import _NO_ENTRY_BAR, BacktestEngine
from finalayze.core.schemas import Candle, Signal, TradeResult
from finalayze.strategies.base import BaseStrategy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INITIAL_CASH = Decimal(100_000)
ENTRY_PRICE = Decimal("150.00")
EXIT_PRICE = Decimal("160.00")
QUANTITY = Decimal(10)
ENTRY_BAR = 5
EXIT_BAR = 12
EXPECTED_HOLD_BARS = EXIT_BAR - ENTRY_BAR  # 7
SYMBOL = "TEST"
SEGMENT = "us_large_cap"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _SilentStrategy(BaseStrategy):
    """Strategy that never emits signals -- used to construct BacktestEngine."""

    @property
    def name(self) -> str:
        return "silent"

    def supported_segments(self) -> list[str]:
        return [SEGMENT]

    def generate_signal(  # type: ignore[override]
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        **kwargs: object,
    ) -> Signal | None:
        return None

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


def _make_engine(
    transaction_costs: object | None = None,
) -> BacktestEngine:
    """Build a minimal BacktestEngine for unit-testing _close_position."""
    return BacktestEngine(
        strategy=_SilentStrategy(),
        initial_cash=INITIAL_CASH,
        transaction_costs=transaction_costs,  # type: ignore[arg-type]
    )


def _make_state_dicts(
    *,
    entry_price: Decimal = ENTRY_PRICE,
    entry_bar: int = ENTRY_BAR,
    strategy_name: str = "momentum",
    chandelier_stop: Decimal = Decimal("145.00"),
) -> tuple[dict[str, Decimal], dict[str, int], dict[str, str], dict[str, Decimal]]:
    """Return the four mutable tracking dicts pre-populated for SYMBOL."""
    return (
        {SYMBOL: entry_price},
        {SYMBOL: entry_bar},
        {SYMBOL: strategy_name},
        {SYMBOL: chandelier_stop},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestClosePositionComputesPnl:
    """Verify PnL = (exit - entry) * quantity when no transaction costs."""

    def test_positive_pnl(self) -> None:
        engine = _make_engine()
        entry_prices, entry_bars, entry_strategies, chandelier_stops = _make_state_dicts()
        trades: list[TradeResult] = []

        trade = engine._close_position(
            symbol=SYMBOL,
            exit_price=EXIT_PRICE,
            quantity=QUANTITY,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=EXIT_BAR,
            trades=trades,
        )

        expected_pnl = (EXIT_PRICE - ENTRY_PRICE) * QUANTITY  # (160-150)*10 = 100
        assert trade.pnl == expected_pnl
        assert trade.entry_price == ENTRY_PRICE
        assert trade.exit_price == EXIT_PRICE
        assert trade.quantity == QUANTITY
        assert trade.hold_bars == EXPECTED_HOLD_BARS

    def test_negative_pnl(self) -> None:
        engine = _make_engine()
        low_exit = Decimal("140.00")
        entry_prices, entry_bars, entry_strategies, chandelier_stops = _make_state_dicts()
        trades: list[TradeResult] = []

        trade = engine._close_position(
            symbol=SYMBOL,
            exit_price=low_exit,
            quantity=QUANTITY,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=EXIT_BAR,
            trades=trades,
        )

        expected_pnl = (low_exit - ENTRY_PRICE) * QUANTITY  # (140-150)*10 = -100
        assert trade.pnl == expected_pnl

    def test_pnl_pct(self) -> None:
        engine = _make_engine()
        entry_prices, entry_bars, entry_strategies, chandelier_stops = _make_state_dicts()
        trades: list[TradeResult] = []

        trade = engine._close_position(
            symbol=SYMBOL,
            exit_price=EXIT_PRICE,
            quantity=QUANTITY,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=EXIT_BAR,
            trades=trades,
        )

        expected_pnl_pct = (EXIT_PRICE - ENTRY_PRICE) / ENTRY_PRICE
        assert trade.pnl_pct == expected_pnl_pct


class TestClosePositionWithTransactionCosts:
    """Verify that transaction costs are deducted from PnL."""

    def test_costs_deducted(self) -> None:
        from finalayze.backtest.costs import TransactionCosts

        costs = TransactionCosts()
        engine = _make_engine(transaction_costs=costs)
        entry_prices, entry_bars, entry_strategies, chandelier_stops = _make_state_dicts()
        trades: list[TradeResult] = []

        trade = engine._close_position(
            symbol=SYMBOL,
            exit_price=EXIT_PRICE,
            quantity=QUANTITY,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=EXIT_BAR,
            trades=trades,
        )

        raw_pnl = (EXIT_PRICE - ENTRY_PRICE) * QUANTITY
        txn_cost = costs.total_cost(EXIT_PRICE, QUANTITY)
        assert txn_cost > 0, "Transaction cost must be positive for this test to be meaningful"
        assert trade.pnl == raw_pnl - txn_cost

    def test_no_costs_when_none(self) -> None:
        engine = _make_engine(transaction_costs=None)
        entry_prices, entry_bars, entry_strategies, chandelier_stops = _make_state_dicts()
        trades: list[TradeResult] = []

        trade = engine._close_position(
            symbol=SYMBOL,
            exit_price=EXIT_PRICE,
            quantity=QUANTITY,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=EXIT_BAR,
            trades=trades,
        )

        expected_pnl = (EXIT_PRICE - ENTRY_PRICE) * QUANTITY
        assert trade.pnl == expected_pnl


class TestClosePositionCleansUpState:
    """Verify that all four tracking dicts are cleaned up after close."""

    def test_all_dicts_popped(self) -> None:
        engine = _make_engine()
        entry_prices, entry_bars, entry_strategies, chandelier_stops = _make_state_dicts()
        trades: list[TradeResult] = []

        engine._close_position(
            symbol=SYMBOL,
            exit_price=EXIT_PRICE,
            quantity=QUANTITY,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=EXIT_BAR,
            trades=trades,
        )

        assert SYMBOL not in entry_prices
        assert SYMBOL not in entry_bars
        assert SYMBOL not in entry_strategies
        assert SYMBOL not in chandelier_stops

    def test_other_symbols_untouched(self) -> None:
        """Closing SYMBOL should not affect OTHER_SYM."""
        engine = _make_engine()
        entry_prices, entry_bars, entry_strategies, chandelier_stops = _make_state_dicts()
        other = "OTHER"
        entry_prices[other] = Decimal("200.00")
        entry_bars[other] = 3
        entry_strategies[other] = "rsi2_connors"
        chandelier_stops[other] = Decimal("195.00")
        trades: list[TradeResult] = []

        engine._close_position(
            symbol=SYMBOL,
            exit_price=EXIT_PRICE,
            quantity=QUANTITY,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=EXIT_BAR,
            trades=trades,
        )

        # SYMBOL removed, OTHER intact
        assert SYMBOL not in entry_prices
        assert other in entry_prices
        assert other in entry_bars
        assert other in entry_strategies
        assert other in chandelier_stops

    def test_missing_keys_do_not_raise(self) -> None:
        """If symbol is not in some dicts (edge case), pop with default should not raise."""
        engine = _make_engine()
        entry_prices: dict[str, Decimal] = {SYMBOL: ENTRY_PRICE}
        entry_bars: dict[str, int] = {}  # deliberately empty
        entry_strategies: dict[str, str] = {}
        chandelier_stops: dict[str, Decimal] = {}
        trades: list[TradeResult] = []

        # Should not raise
        trade = engine._close_position(
            symbol=SYMBOL,
            exit_price=EXIT_PRICE,
            quantity=QUANTITY,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=EXIT_BAR,
            trades=trades,
        )

        # hold_bars defaults to bar_index - bar_index = 0 when entry_bars has no key
        assert trade.hold_bars == 0


class TestClosePositionRecordsTrade:
    """Verify _record_trade is called and trade is appended to the list."""

    def test_trade_appended_to_list(self) -> None:
        engine = _make_engine()
        entry_prices, entry_bars, entry_strategies, chandelier_stops = _make_state_dicts()
        trades: list[TradeResult] = []

        trade = engine._close_position(
            symbol=SYMBOL,
            exit_price=EXIT_PRICE,
            quantity=QUANTITY,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=EXIT_BAR,
            trades=trades,
        )

        assert len(trades) == 1
        assert trades[0] is trade

    def test_record_trade_called(self) -> None:
        engine = _make_engine()
        entry_prices, entry_bars, entry_strategies, chandelier_stops = _make_state_dicts()
        trades: list[TradeResult] = []

        with patch.object(engine, "_record_trade") as mock_record:
            trade = engine._close_position(
                symbol=SYMBOL,
                exit_price=EXIT_PRICE,
                quantity=QUANTITY,
                entry_prices=entry_prices,
                entry_bars=entry_bars,
                entry_strategies=entry_strategies,
                chandelier_stops=chandelier_stops,
                bar_index=EXIT_BAR,
                trades=trades,
            )
            mock_record.assert_called_once_with(trade)

    def test_trade_side_is_sell(self) -> None:
        engine = _make_engine()
        entry_prices, entry_bars, entry_strategies, chandelier_stops = _make_state_dicts()
        trades: list[TradeResult] = []

        trade = engine._close_position(
            symbol=SYMBOL,
            exit_price=EXIT_PRICE,
            quantity=QUANTITY,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=EXIT_BAR,
            trades=trades,
        )

        assert trade.side == "SELL"
        assert trade.symbol == SYMBOL


class TestClosePositionZeroEntry:
    """Edge case: entry_price=0 should not cause DivisionByZeroError."""

    def test_zero_entry_no_divide_by_zero(self) -> None:
        engine = _make_engine()
        entry_prices, entry_bars, entry_strategies, chandelier_stops = _make_state_dicts(
            entry_price=Decimal(0),
        )
        trades: list[TradeResult] = []

        trade = engine._close_position(
            symbol=SYMBOL,
            exit_price=EXIT_PRICE,
            quantity=QUANTITY,
            entry_prices=entry_prices,
            entry_bars=entry_bars,
            entry_strategies=entry_strategies,
            chandelier_stops=chandelier_stops,
            bar_index=EXIT_BAR,
            trades=trades,
        )

        # pnl_pct should be 0 when entry is 0 (guard against ZeroDivisionError)
        assert trade.pnl_pct == Decimal(0)
        # pnl = (160 - 0) * 10 = 1600
        assert trade.pnl == EXIT_PRICE * QUANTITY


class TestNoEntryBarConstant:
    """Verify _NO_ENTRY_BAR is exported and has the expected value."""

    def test_value(self) -> None:
        assert _NO_ENTRY_BAR == -2

    def test_grace_bar_math(self) -> None:
        """The grace bar logic uses entry_bar + 1 == i.
        With _NO_ENTRY_BAR = -2, entry_bar + 1 = -1, which never matches
        a valid bar index (0+), so stop checks are NOT skipped -- correct
        for symbols without an open position.
        """
        assert _NO_ENTRY_BAR + 1 < 0
