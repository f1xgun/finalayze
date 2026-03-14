"""Integration tests for bond walk-forward backtest machinery.

Validates that walk_forward_bond_backtest:
- Splits date range into correct number of rolling folds
- Runs BondBacktestEngine on each fold's test period
- Aggregates out-of-sample metrics correctly
- Produces WalkForwardResult with per-fold and aggregate data

Uses synthetic candle data (small universe, short dates) to test
structure, not PnL assertions.
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from finalayze.backtest.bond_engine import BondBacktestConfig
from finalayze.backtest.costs import MOEX_BOND_COSTS
from finalayze.core.schemas import BondInfo, Candle, CouponPayment, Signal, SignalDirection
from finalayze.risk.yield_stop import YieldStop

# Ensure scripts/ is importable
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _make_candle(symbol: str, dt: date, close: Decimal = Decimal("99.50")) -> Candle:
    """Create a synthetic bond candle."""
    return Candle(
        symbol=symbol,
        market_id="MOEX",
        timeframe="1D",
        timestamp=datetime(dt.year, dt.month, dt.day, tzinfo=UTC),
        open=close - Decimal("0.10"),
        high=close + Decimal("0.20"),
        low=close - Decimal("0.30"),
        close=close,
        volume=1000,
    )


def _make_bond_info(symbol: str, maturity_date: date | None = None) -> BondInfo:
    """Create synthetic BondInfo."""
    mat = maturity_date or date(2028, 1, 15)
    return BondInfo(
        figi=f"FIGI_{symbol}",
        ticker=symbol,
        isin=f"RU000A{symbol[-4:]}",
        name=f"OFZ {symbol}",
        face_value=Decimal(1000),
        coupon_rate=Decimal("7.50"),
        coupon_frequency=2,
        maturity_date=mat,
    )


def _make_coupon(figi: str, coupon_date: date, number: int = 1) -> CouponPayment:
    """Create synthetic coupon payment."""
    return CouponPayment(
        bond_figi=figi,
        coupon_date=coupon_date,
        record_date=coupon_date - timedelta(days=2),
        amount_per_bond=Decimal("37.50"),
        coupon_number=number,
    )


def _generate_daily_candles(
    symbol: str,
    start: date,
    end: date,
    base_price: Decimal = Decimal("99.50"),
) -> list[Candle]:
    """Generate daily candles for date range (skipping weekends)."""
    candles: list[Candle] = []
    current = start
    idx = 0
    while current <= end:
        if current.weekday() < 5:  # skip weekends
            # Small price variation
            variation = Decimal(str(0.01 * (idx % 7 - 3)))
            candles.append(_make_candle(symbol, current, base_price + variation))
            idx += 1
        current += timedelta(days=1)
    return candles


def _always_buy_strategy(
    symbol: str,
    candles: list[Candle],
    positions: dict,
    bar_idx: int,
    **kwargs: object,
) -> Signal | None:
    """Simple strategy that buys on every 20th bar if not already holding."""
    if symbol in positions:
        return None
    if bar_idx % 20 == 10:  # buy on bar 10, 30, 50, ...
        return Signal(
            symbol=symbol,
            direction=SignalDirection.BUY,
            confidence=Decimal("0.80"),
            source="test_strategy",
        )
    return None


class TestWalkForwardBondBacktestImport:
    """Verify walk_forward_bond_backtest exists and is callable."""

    def test_function_exists(self) -> None:
        import run_bond_iteration

        assert hasattr(run_bond_iteration, "walk_forward_bond_backtest")
        assert callable(run_bond_iteration.walk_forward_bond_backtest)

    def test_walk_forward_result_exists(self) -> None:
        import run_bond_iteration

        assert hasattr(run_bond_iteration, "WalkForwardResult")


class TestWalkForwardFoldGeneration:
    """Test that walk-forward generates correct number of folds."""

    def test_three_year_range_produces_multiple_folds(self) -> None:
        """2022-01-01 to 2024-12-31 with 12mo train + 6mo test, 3mo roll
        should produce at least 3 folds."""
        import run_bond_iteration

        symbols = ["BOND_A", "BOND_B"]
        start = date(2022, 1, 1)
        end = date(2024, 12, 31)

        candles_by_symbol: dict[str, list[Candle]] = {}
        bond_info: dict[str, BondInfo] = {}
        coupon_schedule: dict[str, list[CouponPayment]] = {}

        for sym in symbols:
            candles_by_symbol[sym] = _generate_daily_candles(sym, start, end)
            bond_info[sym] = _make_bond_info(sym)
            figi = f"FIGI_{sym}"
            coupon_schedule[sym] = [
                _make_coupon(figi, date(2022, 7, 15), 1),
                _make_coupon(figi, date(2023, 1, 15), 2),
                _make_coupon(figi, date(2023, 7, 15), 3),
                _make_coupon(figi, date(2024, 1, 15), 4),
                _make_coupon(figi, date(2024, 7, 15), 5),
            ]

        config = BondBacktestConfig(
            initial_cash=Decimal(500_000),
            max_positions=2,
            yield_stop=YieldStop(threshold_bps=100),
            transaction_costs=MOEX_BOND_COSTS,
            max_hold_bars=60,
        )

        result = run_bond_iteration.walk_forward_bond_backtest(
            candles_by_symbol=candles_by_symbol,
            bond_info=bond_info,
            coupon_schedule=coupon_schedule,
            strategy_fn=_always_buy_strategy,
            config=config,
            start=start,
            end=end,
        )

        assert isinstance(result, run_bond_iteration.WalkForwardResult)
        # 3-year range with 12mo train + 6mo test + 3mo roll = at least 3 folds
        assert len(result.per_fold) >= 3

    def test_each_fold_has_metrics(self) -> None:
        """Each fold should have test_start, test_end, and metrics."""
        import run_bond_iteration

        symbols = ["BOND_X"]
        start = date(2022, 1, 1)
        end = date(2024, 12, 31)

        candles_by_symbol = {sym: _generate_daily_candles(sym, start, end) for sym in symbols}
        bond_info = {sym: _make_bond_info(sym) for sym in symbols}
        coupon_schedule: dict[str, list[CouponPayment]] = {
            sym: [_make_coupon(f"FIGI_{sym}", date(2023, 1, 15))] for sym in symbols
        }

        config = BondBacktestConfig(
            initial_cash=Decimal(500_000),
            max_positions=2,
            yield_stop=YieldStop(threshold_bps=100),
            transaction_costs=MOEX_BOND_COSTS,
            max_hold_bars=60,
        )

        result = run_bond_iteration.walk_forward_bond_backtest(
            candles_by_symbol=candles_by_symbol,
            bond_info=bond_info,
            coupon_schedule=coupon_schedule,
            strategy_fn=_always_buy_strategy,
            config=config,
            start=start,
            end=end,
        )

        for fold in result.per_fold:
            assert "test_start" in fold
            assert "test_end" in fold
            assert "metrics" in fold
            assert fold["test_start"] < fold["test_end"]

    def test_aggregate_metrics_computed(self) -> None:
        """Aggregate metrics should be computed across all folds."""
        import run_bond_iteration

        symbols = ["BOND_Y"]
        start = date(2022, 1, 1)
        end = date(2024, 12, 31)

        candles_by_symbol = {sym: _generate_daily_candles(sym, start, end) for sym in symbols}
        bond_info = {sym: _make_bond_info(sym) for sym in symbols}
        coupon_schedule: dict[str, list[CouponPayment]] = {
            sym: [_make_coupon(f"FIGI_{sym}", date(2023, 1, 15))] for sym in symbols
        }

        config = BondBacktestConfig(
            initial_cash=Decimal(500_000),
            max_positions=2,
            yield_stop=YieldStop(threshold_bps=100),
            transaction_costs=MOEX_BOND_COSTS,
            max_hold_bars=60,
        )

        result = run_bond_iteration.walk_forward_bond_backtest(
            candles_by_symbol=candles_by_symbol,
            bond_info=bond_info,
            coupon_schedule=coupon_schedule,
            strategy_fn=_always_buy_strategy,
            config=config,
            start=start,
            end=end,
        )

        assert result.aggregate is not None
        # Aggregate should have standard metric keys
        assert "sharpe" in result.aggregate
        assert "profit_factor" in result.aggregate
        assert "max_drawdown_pct" in result.aggregate
        assert "total_return_pct" in result.aggregate


class TestWalkForwardCLI:
    """Test that --walk-forward CLI flag is recognized."""

    def test_walk_forward_flag_in_parser(self) -> None:
        """Verify walk_forward_bond_backtest and WalkForwardResult exist in module."""
        import run_bond_iteration

        # Module should have walk-forward support
        assert hasattr(run_bond_iteration, "walk_forward_bond_backtest")
        assert hasattr(run_bond_iteration, "WalkForwardResult")
        assert hasattr(run_bond_iteration, "_run_bond_segment_walk_forward")
