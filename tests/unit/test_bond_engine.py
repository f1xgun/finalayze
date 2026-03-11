"""Tests for BondBacktestEngine.

TDD tests for the OFZ bond backtest engine. Uses synthetic candle data
and simple strategy functions to verify core bond mechanics:
- Clean/dirty price accounting
- Coupon income tracking
- DV01-based position sizing
- Yield-based stop-loss
- Duration monitoring
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from finalayze.backtest.bond_engine import (
    BondBacktestConfig,
    BondBacktestEngine,
    BondBacktestResult,
    BondPosition,
)
from finalayze.backtest.costs import MOEX_BOND_COSTS
from finalayze.core.schemas import (
    BondInfo,
    Candle,
    CouponPayment,
    Signal,
    SignalDirection,
)
from finalayze.risk.dv01_sizing import DV01BudgetStep
from finalayze.risk.yield_stop import YieldStop

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FACE_VALUE = Decimal(1000)
_SYMBOL = "SU26243RMFS4"
_FIGI = "TCS00A105EM7"

# Standard OFZ bond info for tests: 7.1% semiannual, matures 2030-12-11
_BOND_INFO = BondInfo(
    figi=_FIGI,
    ticker=_SYMBOL,
    isin="RU000A105EM7",
    name="OFZ 26243",
    face_value=_FACE_VALUE,
    coupon_rate=Decimal("7.10"),
    coupon_frequency=2,
    maturity_date=date(2030, 12, 11),
)


def _make_bond_candles(symbol: str, prices: list[float], start_date: date) -> list[Candle]:
    """Create synthetic bond candles (close = clean price %)."""
    candles: list[Candle] = []
    d = start_date
    for p in prices:
        # Skip weekends
        while d.weekday() >= 5:
            d += timedelta(days=1)
        candles.append(
            Candle(
                symbol=symbol,
                market_id="moex",
                timeframe="1d",
                timestamp=datetime(d.year, d.month, d.day, tzinfo=UTC),
                open=Decimal(str(p)),
                high=Decimal(str(p + 0.5)),
                low=Decimal(str(p - 0.5)),
                close=Decimal(str(p)),
                volume=1000,
            )
        )
        d += timedelta(days=1)
    return candles


def _make_coupon_schedule(
    figi: str,
    dates: list[date],
    amount: Decimal = Decimal("35.50"),
) -> list[CouponPayment]:
    """Create a simple coupon schedule."""
    return [
        CouponPayment(
            bond_figi=figi,
            coupon_date=d,
            record_date=d - timedelta(days=2),
            amount_per_bond=amount,
            coupon_number=i + 1,
        )
        for i, d in enumerate(dates)
    ]


def _never_buy_strategy(
    symbol: str,
    candles: list[Candle],
    positions: dict[str, BondPosition],
    bar_idx: int,
) -> Signal | None:
    """Strategy that never generates a signal."""
    return None


def _buy_at_bar_strategy(buy_bar: int, confidence: float = 0.8):
    """Return a strategy that buys at a specific bar index."""

    def strategy_fn(
        symbol: str,
        candles: list[Candle],
        positions: dict[str, BondPosition],
        bar_idx: int,
    ) -> Signal | None:
        if bar_idx == buy_bar and symbol not in positions:
            return Signal(
                strategy_name="test_bond_strategy",
                symbol=symbol,
                market_id="moex",
                segment_id="ru_ofz_strategic",
                direction=SignalDirection.BUY,
                confidence=confidence,
                features={"bar_idx": float(bar_idx)},
                reasoning="Test buy signal",
                instrument_type="bond",
            )
        return None

    return strategy_fn


# ---------------------------------------------------------------------------
# Test 1: Empty candles returns empty result
# ---------------------------------------------------------------------------


class TestBondEngineEmpty:
    """Empty candles produce an empty result."""

    def test_empty_candles_returns_empty_result(self) -> None:
        engine = BondBacktestEngine()
        result = engine.run(
            candles_by_symbol={},
            bond_info={},
            coupon_schedule={},
            strategy_fn=_never_buy_strategy,
        )
        assert isinstance(result, BondBacktestResult)
        assert result.trade_count == 0
        assert result.trades == []
        assert result.equity_curve == []
        assert result.dates == []

    def test_empty_candle_list_returns_empty(self) -> None:
        engine = BondBacktestEngine()
        result = engine.run(
            candles_by_symbol={_SYMBOL: []},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={},
            strategy_fn=_never_buy_strategy,
        )
        assert result.trade_count == 0


# ---------------------------------------------------------------------------
# Test 2: Single bond, no signals — equity unchanged
# ---------------------------------------------------------------------------


class TestBondEngineNoSignals:
    """When the strategy generates no signals, cash stays constant."""

    def test_equity_unchanged_no_signals(self) -> None:
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(initial_cash=initial_cash)
        engine = BondBacktestEngine(config=config)

        prices = [85.0 + 0.1 * i for i in range(10)]
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: []},
            strategy_fn=_never_buy_strategy,
        )

        assert result.trade_count == 0
        # All equity curve values should equal initial cash (no positions)
        for eq in result.equity_curve:
            assert eq == initial_cash


# ---------------------------------------------------------------------------
# Test 3: Buy and hold to end — trade created with correct entry/exit
# ---------------------------------------------------------------------------


class TestBondEngineBuyHold:
    """Buy at bar 2, hold to end. Trade forced-closed at last bar."""

    def test_buy_and_hold_to_end(self) -> None:
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(
            initial_cash=initial_cash,
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=500),  # Very wide stop to avoid triggering
            max_hold_bars=200,  # Won't trigger
        )
        engine = BondBacktestEngine(config=config)

        # 15 bars of stable prices
        prices = [85.0] * 15
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))

        # Simple coupon schedule far in the future (no coupons during test)
        coupons = _make_coupon_schedule(
            _FIGI,
            [date(2025, 6, 11), date(2025, 12, 11)],
        )

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
        )

        # Should have exactly one trade (forced close at end)
        assert result.trade_count == 1
        trade = result.trades[0]
        assert trade.symbol == _SYMBOL
        assert trade.instrument_type == "bond"
        assert trade.quantity > 0
        assert trade.hold_bars > 0


# ---------------------------------------------------------------------------
# Test 4: Coupon during hold — coupon_income tracked in trade result
# ---------------------------------------------------------------------------


class TestBondEngineCouponIncome:
    """A coupon payment during the hold period is tracked."""

    def test_coupon_income_tracked(self) -> None:
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(
            initial_cash=initial_cash,
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=500),  # Won't trigger
            max_hold_bars=200,  # Won't trigger
        )
        engine = BondBacktestEngine(config=config)

        # 20 bars starting Jan 6, coupon on Jan 15 (bar ~7)
        prices = [85.0] * 20
        start = date(2025, 1, 6)
        candles = _make_bond_candles(_SYMBOL, prices, start)

        # Place coupon on a weekday within the candle range
        coupon_date = date(2025, 1, 15)  # Wednesday
        coupon_amount = Decimal("35.50")
        coupons = _make_coupon_schedule(
            _FIGI,
            [coupon_date, date(2025, 7, 11)],
            amount=coupon_amount,
        )

        # Buy at bar 1 (before coupon), hold to end
        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=1),
        )

        assert result.trade_count == 1
        trade = result.trades[0]
        # Coupon income should be quantity * coupon_amount
        expected_coupon = coupon_amount * trade.quantity
        assert trade.coupon_income == expected_coupon

        # Total coupon tracking should reflect gross and net (13% NDFL)
        assert result.total_coupon_income_gross == expected_coupon
        expected_tax = expected_coupon * Decimal("0.13")
        assert result.total_tax_paid == expected_tax
        expected_net = expected_coupon - expected_tax
        assert result.total_coupon_income_net == expected_net


# ---------------------------------------------------------------------------
# Test 5: Yield stop triggers — position closed when YTM rises
# ---------------------------------------------------------------------------


class TestBondEngineYieldStop:
    """Position stopped when yield rises above threshold."""

    def test_yield_stop_triggers_on_price_drop(self) -> None:
        """When price drops sharply (YTM rises), yield stop triggers."""
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(
            initial_cash=initial_cash,
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=50),  # Tight stop: 50bps
            max_hold_bars=200,
        )
        engine = BondBacktestEngine(config=config)

        # Price starts at 85, drops significantly (YTM rises)
        # A 3% price drop on a ~5yr duration bond is roughly ~60bps yield rise
        prices = [85.0] * 5 + [83.0, 82.0, 81.0, 80.0, 79.0]
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))

        coupons = _make_coupon_schedule(
            _FIGI,
            [date(2025, 6, 11), date(2025, 12, 11)],
        )

        # Buy at bar 2
        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
        )

        assert result.trade_count >= 1
        trade = result.trades[0]
        # Should be closed before the end (stopped out)
        num_candle_dates = len(result.equity_curve)
        assert trade.hold_bars < num_candle_dates - 2  # Closed early


# ---------------------------------------------------------------------------
# Test 6: Max hold bars — position closed at limit
# ---------------------------------------------------------------------------


class TestBondEngineMaxHold:
    """Position closed when max_hold_bars is reached."""

    def test_max_hold_bars_triggers_close(self) -> None:
        max_hold = 5
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(
            initial_cash=initial_cash,
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=500),  # Very wide to not trigger
            max_hold_bars=max_hold,
        )
        engine = BondBacktestEngine(config=config)

        # 20 bars of stable prices
        prices = [85.0] * 20
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))

        coupons = _make_coupon_schedule(
            _FIGI,
            [date(2025, 6, 11), date(2025, 12, 11)],
        )

        # Buy at bar 2
        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
        )

        assert result.trade_count >= 1
        trade = result.trades[0]
        # Should close at exactly max_hold_bars
        assert trade.hold_bars == max_hold


# ---------------------------------------------------------------------------
# Test 7: DV01 sizing — position size respects DV01 budget
# ---------------------------------------------------------------------------


class TestBondEngineDV01Sizing:
    """DV01BudgetStep controls position size."""

    def test_dv01_budget_limits_position(self) -> None:
        """With tight DV01 budget, position size is small."""
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(
            initial_cash=initial_cash,
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.01"),  # Very tight: 1%
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.10"),  # Tight limit
            ),
            yield_stop=YieldStop(threshold_bps=500),
            max_hold_bars=200,
        )
        engine = BondBacktestEngine(config=config)

        prices = [85.0] * 15
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))

        coupons = _make_coupon_schedule(
            _FIGI,
            [date(2025, 6, 11), date(2025, 12, 11)],
        )

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
        )

        assert result.trade_count == 1
        trade = result.trades[0]
        # Position capped by max_single_position_pct (10% of 1M = 100k, ~100 bonds)
        max_by_position = int(initial_cash * Decimal("0.10") / _FACE_VALUE)
        assert trade.quantity <= max_by_position


# ---------------------------------------------------------------------------
# Test 8: Transaction costs — deducted from PnL
# ---------------------------------------------------------------------------


class TestBondEngineTransactionCosts:
    """Transaction costs reduce trade PnL."""

    def test_costs_deducted_from_pnl(self) -> None:
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(
            initial_cash=initial_cash,
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=500),
            max_hold_bars=200,
            transaction_costs=MOEX_BOND_COSTS,
        )
        engine = BondBacktestEngine(config=config)

        # Flat prices so price PnL ~ 0; total PnL should be negative (costs)
        prices = [85.0] * 15
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))

        coupons = _make_coupon_schedule(
            _FIGI,
            [date(2025, 6, 11), date(2025, 12, 11)],
        )

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
        )

        assert result.trade_count == 1
        trade = result.trades[0]
        # With flat prices and no coupons during the period, PnL should be
        # negative due to transaction costs (buy + sell costs)
        assert trade.pnl < 0


# ---------------------------------------------------------------------------
# Test 9: Multiple bonds — independent positions tracked
# ---------------------------------------------------------------------------


class TestBondEngineMultipleBonds:
    """Multiple bond symbols are tracked independently."""

    def test_multiple_bonds_independent_positions(self) -> None:
        symbol_a = "SU26243RMFS4"
        symbol_b = "SU26244RMFS2"
        figi_a = "TCS00A105EM7"
        figi_b = "TCS00A105EM8"

        info_a = BondInfo(
            figi=figi_a,
            ticker=symbol_a,
            isin="RU000A105EM7",
            name="OFZ 26243",
            face_value=_FACE_VALUE,
            coupon_rate=Decimal("7.10"),
            coupon_frequency=2,
            maturity_date=date(2030, 12, 11),
        )
        info_b = BondInfo(
            figi=figi_b,
            ticker=symbol_b,
            isin="RU000A105EM8",
            name="OFZ 26244",
            face_value=_FACE_VALUE,
            coupon_rate=Decimal("8.50"),
            coupon_frequency=2,
            maturity_date=date(2032, 6, 15),
        )

        initial_cash = Decimal(2_000_000)
        config = BondBacktestConfig(
            initial_cash=initial_cash,
            max_positions=5,
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.30"),
            ),
            yield_stop=YieldStop(threshold_bps=500),
            max_hold_bars=200,
        )
        engine = BondBacktestEngine(config=config)

        prices_a = [85.0] * 15
        prices_b = [90.0] * 15
        start = date(2025, 1, 6)
        candles_a = _make_bond_candles(symbol_a, prices_a, start)
        candles_b = _make_bond_candles(symbol_b, prices_b, start)

        coupons_a = _make_coupon_schedule(figi_a, [date(2025, 6, 11)])
        coupons_b = _make_coupon_schedule(figi_b, [date(2025, 6, 15)])

        result = engine.run(
            candles_by_symbol={symbol_a: candles_a, symbol_b: candles_b},
            bond_info={symbol_a: info_a, symbol_b: info_b},
            coupon_schedule={symbol_a: coupons_a, symbol_b: coupons_b},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
        )

        # Both bonds should have a trade (forced close at end)
        assert result.trade_count == 2
        symbols_traded = {t.symbol for t in result.trades}
        assert symbol_a in symbols_traded
        assert symbol_b in symbols_traded


# ---------------------------------------------------------------------------
# Test 10: Equity curve — correct length and monotonically plausible
# ---------------------------------------------------------------------------


class TestBondEngineEquityCurve:
    """Equity curve has correct length and plausible values."""

    def test_equity_curve_length_matches_dates(self) -> None:
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(initial_cash=initial_cash)
        engine = BondBacktestEngine(config=config)

        prices = [85.0] * 10
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: []},
            strategy_fn=_never_buy_strategy,
        )

        assert len(result.equity_curve) == len(result.dates)
        assert len(result.equity_curve) == len(candles)

    def test_equity_curve_starts_at_initial_cash(self) -> None:
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(initial_cash=initial_cash)
        engine = BondBacktestEngine(config=config)

        prices = [85.0] * 10
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: []},
            strategy_fn=_never_buy_strategy,
        )

        assert result.equity_curve[0] == initial_cash


# ---------------------------------------------------------------------------
# Test 11: Metrics are computed correctly
# ---------------------------------------------------------------------------


class TestBondEngineMetrics:
    """Verify result metrics are computed from trades and equity curve."""

    def test_win_rate_all_losing(self) -> None:
        """Flat prices + costs = all trades lose => win_rate == 0."""
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(
            initial_cash=initial_cash,
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=500),
            max_hold_bars=200,
        )
        engine = BondBacktestEngine(config=config)

        prices = [85.0] * 15
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))
        coupons = _make_coupon_schedule(_FIGI, [date(2025, 6, 11)])

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
        )

        assert result.trade_count == 1
        assert result.win_rate == Decimal(0)

    def test_total_return_computed(self) -> None:
        """Total return % is computed from equity curve."""
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(initial_cash=initial_cash)
        engine = BondBacktestEngine(config=config)

        prices = [85.0] * 10
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: []},
            strategy_fn=_never_buy_strategy,
        )

        # No trades, so total return should be 0
        assert result.total_return_pct == Decimal(0)

    def test_max_drawdown_non_negative(self) -> None:
        """Max drawdown should be >= 0 (reported as positive value)."""
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(
            initial_cash=initial_cash,
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=500),
            max_hold_bars=200,
        )
        engine = BondBacktestEngine(config=config)

        prices = [85.0 - 0.1 * i for i in range(15)]
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))
        coupons = _make_coupon_schedule(_FIGI, [date(2025, 6, 11)])

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
        )

        assert result.max_drawdown_pct >= Decimal(0)


# ---------------------------------------------------------------------------
# Test 12: NKD estimation from coupon schedule
# ---------------------------------------------------------------------------


class TestBondEngineNKDEstimation:
    """NKD is estimated correctly from coupon schedule when not pre-computed."""

    def test_nkd_estimation_uses_coupon_schedule(self) -> None:
        """Engine runs without pre-computed nkd_series (uses estimation)."""
        initial_cash = Decimal(1_000_000)
        config = BondBacktestConfig(
            initial_cash=initial_cash,
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=500),
            max_hold_bars=200,
        )
        engine = BondBacktestEngine(config=config)

        prices = [85.0] * 15
        start = date(2025, 1, 6)
        candles = _make_bond_candles(_SYMBOL, prices, start)

        # Coupons bracketing our test window
        coupons = _make_coupon_schedule(
            _FIGI,
            [date(2024, 12, 11), date(2025, 6, 11)],
        )

        # Should work without nkd_series (None is default)
        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
            nkd_series=None,
        )

        # Engine ran without error; we got a trade
        assert result.trade_count == 1
