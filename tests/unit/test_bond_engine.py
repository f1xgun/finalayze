"""Tests for BondBacktestEngine.

TDD tests for the OFZ bond backtest engine. Uses synthetic candle data
and simple strategy functions to verify core bond mechanics:
- Clean/dirty price accounting
- Coupon income tracking
- DV01-based position sizing
- Yield-based stop-loss
- Duration monitoring
- Macro context wiring to strategies
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

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
    ExitReason,
    Signal,
    SignalDirection,
)
from finalayze.data.fetchers.cbr import MacroContextProvider, MacroSnapshot
from finalayze.execution.bond_simulated_broker import BondSimulatedBroker
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
                strategy_payload={"bar_idx": float(bar_idx)},
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


# ---------------------------------------------------------------------------
# Test 13: Macro kwargs forwarded to strategy_fn
# ---------------------------------------------------------------------------


class TestBondEngineMacroWiring:
    """When macro_provider is supplied, macro kwargs reach the strategy."""

    def test_strategy_receives_macro_kwargs(self) -> None:
        """Strategy is called with key_rate, ruonia_7d_avg, cpi_yoy, last_cbr_decision."""
        received_kwargs: list[dict[str, Any]] = []

        def capturing_strategy(
            symbol: str,
            candles: list[Candle],
            positions: dict[str, BondPosition],
            bar_idx: int,
            **kwargs: Any,
        ) -> Signal | None:
            if kwargs:
                received_kwargs.append(dict(kwargs))
            if bar_idx == 2 and symbol not in positions:
                return Signal(
                    strategy_name="test",
                    symbol=symbol,
                    market_id="moex",
                    segment_id="ru_ofz_pd",
                    direction=SignalDirection.BUY,
                    confidence=0.8,
                    strategy_payload={},
                    reasoning="test",
                    instrument_type="bond",
                )
            return None

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

        # Use dates in 2024 where macro data is available
        prices = [85.0] * 10
        candles = _make_bond_candles(_SYMBOL, prices, date(2024, 6, 10))
        coupons = _make_coupon_schedule(_FIGI, [date(2024, 12, 11)])

        macro_provider = MacroContextProvider()
        engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=capturing_strategy,
            macro_provider=macro_provider,
        )

        # Strategy should have received kwargs on every call
        assert len(received_kwargs) > 0
        sample = received_kwargs[0]
        assert "key_rate" in sample
        assert "ruonia_7d_avg" in sample
        assert "cpi_yoy" in sample
        assert "last_cbr_decision" in sample
        # key_rate should be a Decimal or None (we know 2024-06 has data)
        assert isinstance(sample["key_rate"], Decimal)

    def test_no_macro_kwargs_without_provider(self) -> None:
        """Without macro_provider, strategy receives no extra kwargs."""
        received_kwargs: list[dict[str, Any]] = []

        def capturing_strategy(
            symbol: str,
            candles: list[Candle],
            positions: dict[str, BondPosition],
            bar_idx: int,
            **kwargs: Any,
        ) -> Signal | None:
            received_kwargs.append(dict(kwargs))
            return None

        engine = BondBacktestEngine()
        prices = [85.0] * 5
        candles = _make_bond_candles(_SYMBOL, prices, date(2024, 6, 10))

        engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: []},
            strategy_fn=capturing_strategy,
            # No macro_provider
        )

        # All calls should have empty kwargs
        for kw in received_kwargs:
            assert kw == {}

    def test_macro_snapshot_no_look_ahead(self) -> None:
        """Macro data at bar T must reflect only information available at T."""
        snapshots_by_date: dict[date, dict[str, Any]] = {}

        def capturing_strategy(
            symbol: str,
            candles: list[Candle],
            positions: dict[str, BondPosition],
            bar_idx: int,
            **kwargs: Any,
        ) -> Signal | None:
            if candles:
                d = candles[-1].timestamp.date()
                snapshots_by_date[d] = dict(kwargs)
            return None

        engine = BondBacktestEngine()

        # Span across Oct 2024 hike: before = 19.00, after = 21.00
        # Meeting: 2024-10-25
        prices = [85.0] * 15
        # Start from 2024-10-20 to span the meeting
        candles = _make_bond_candles(_SYMBOL, prices, date(2024, 10, 20))

        macro_provider = MacroContextProvider()
        engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: []},
            strategy_fn=capturing_strategy,
            macro_provider=macro_provider,
        )

        # Before meeting (2024-10-24): rate should be 19.00 (from Sep 13 hike)
        pre_meeting = date(2024, 10, 24)
        if pre_meeting in snapshots_by_date:
            assert snapshots_by_date[pre_meeting]["key_rate"] == Decimal("19.00")

        # After meeting (2024-10-25 or later): rate should be 21.00
        post_meeting = date(2024, 10, 25)
        if post_meeting in snapshots_by_date:
            assert snapshots_by_date[post_meeting]["key_rate"] == Decimal("21.00")


# ---------------------------------------------------------------------------
# Test 14: OFZ rotation wiring in BondBacktestEngine
# ---------------------------------------------------------------------------


class TestBondEngineOFZRotation:
    """OFZ rotation wiring: apply_ofz_rotation called when layer_configs provided."""

    def test_ofz_rotation_inactive_by_default(self) -> None:
        """Without layer_configs, ofz_rotation_active is False."""
        engine = BondBacktestEngine()
        result = engine.run(
            candles_by_symbol={},
            bond_info={},
            coupon_schedule={},
            strategy_fn=_never_buy_strategy,
        )
        assert result.ofz_rotation_active is False

    def test_ofz_rotation_inactive_no_cuts(self, monkeypatch: Any) -> None:
        """With layer_configs but no CBR cuts, ofz_rotation_active is False."""
        from finalayze.core.schemas import DEFAULT_LAYER_CONFIGS

        # Monkeypatch CBR_MEETINGS to empty (no cuts at all)
        monkeypatch.setattr("finalayze.data.fetchers.cbr.CBR_MEETINGS", ())

        engine = BondBacktestEngine()
        prices = [85.0] * 10
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))
        coupons = _make_coupon_schedule(_FIGI, [date(2025, 6, 11)])

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_never_buy_strategy,
            layer_configs=DEFAULT_LAYER_CONFIGS,
            as_of_date=date(2025, 1, 20),
        )
        assert result.ofz_rotation_active is False

    def test_ofz_rotation_active_two_consecutive_cuts(self, monkeypatch: Any) -> None:
        """With 2 consecutive CBR cuts before as_of_date, ofz_rotation_active is True."""
        from finalayze.core.schemas import DEFAULT_LAYER_CONFIGS
        from finalayze.data.fetchers.cbr import CBRMeeting

        # Monkeypatch CBR_MEETINGS with two consecutive cuts
        fake_meetings = (
            CBRMeeting(date(2024, 6, 1), "core", "cut", Decimal("15.00")),
            CBRMeeting(date(2024, 7, 1), "core", "cut", Decimal("14.00")),
        )
        monkeypatch.setattr("finalayze.data.fetchers.cbr.CBR_MEETINGS", fake_meetings)

        engine = BondBacktestEngine()
        prices = [85.0] * 10
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))
        coupons = _make_coupon_schedule(_FIGI, [date(2025, 6, 11)])

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_never_buy_strategy,
            layer_configs=DEFAULT_LAYER_CONFIGS,
            as_of_date=date(2025, 1, 20),
        )
        assert result.ofz_rotation_active is True


# ---------------------------------------------------------------------------
# Test 15: exit_reason / entry_strategy wiring on closed bond TradeResults
# (EXITDIAG-02 / D-04) — RED until bond_engine threads the close reason.
# ---------------------------------------------------------------------------

_ENTRY_BAR = 0
_CLOSE_BAR = 3
_QTY = 100
_CARRY_STRATEGY = "bond_carry"
_DURATION_STRATEGY = "bond_duration_rotation"
_MAX_HOLD_FOR_MAPPING = 2


def _make_open_position(
    *,
    symbol: str = _SYMBOL,
    quantity: int = _QTY,
    entry_clean_pct: Decimal = Decimal("85.0"),
    entry_ytm_pct: Decimal = Decimal("7.5"),
    entry_bar_idx: int = _ENTRY_BAR,
    entry_date: date = date(2025, 1, 6),
    entry_strategy: str | None = None,
) -> BondPosition:
    """Build a synthetic open BondPosition for direct _close_position drives."""
    return BondPosition(
        symbol=symbol,
        quantity=quantity,
        entry_clean_pct=entry_clean_pct,
        entry_nkd=Decimal(0),
        entry_ytm_pct=entry_ytm_pct,
        entry_bar_idx=entry_bar_idx,
        entry_date=entry_date,
        entry_strategy=entry_strategy,
    )


def _single_candle_lookup(
    symbol: str,
    close_pct: Decimal,
    when: date,
) -> dict[str, dict[date, Candle]]:
    """A candle_lookup with a single bar so _close_position can read an exit price."""
    candle = Candle(
        symbol=symbol,
        market_id="moex",
        timeframe="1d",
        timestamp=datetime(when.year, when.month, when.day, tzinfo=UTC),
        open=close_pct,
        high=close_pct + Decimal("0.5"),
        low=close_pct - Decimal("0.5"),
        close=close_pct,
        volume=1000,
    )
    return {symbol: {when: candle}}


class TestBondExitReasonWiring:
    """Closed bond TradeResults must carry exit_reason + entry_strategy (EXITDIAG-02)."""

    def _close_with(
        self,
        *,
        exit_reason: object | None,
        entry_strategy: str | None,
    ) -> Any:
        """Drive _close_position directly with synthetic state and return the trade."""
        close_date = date(2025, 1, 9)
        engine = BondBacktestEngine()
        broker = BondSimulatedBroker(
            initial_cash=Decimal(1_000_000),
            coupon_schedule={_SYMBOL: []},
            face_value=_FACE_VALUE,
        )
        # Open a real broker lot so the closing sell has something to sell.
        broker.buy_bond(_SYMBOL, _QTY, Decimal("85.0"), Decimal(0), Decimal(0))
        pos = _make_open_position(entry_strategy=entry_strategy)
        candle_lookup = _single_candle_lookup(_SYMBOL, Decimal("85.0"), close_date)
        return engine._close_position(  # noqa: SLF001
            _SYMBOL,
            pos,
            close_date,
            _CLOSE_BAR,
            candle_lookup,
            None,
            broker,
            {_SYMBOL: _BOND_INFO},
            {_SYMBOL: []},
            exit_reason=exit_reason,
            entry_strategy=entry_strategy,
        )

    def test_close_position_records_stop_exit_reason(self) -> None:
        """yield_stop maps to ExitReason.STOP on the closed TradeResult."""
        trade = self._close_with(exit_reason=ExitReason.STOP, entry_strategy=_CARRY_STRATEGY)
        assert trade is not None
        assert trade.exit_reason == ExitReason.STOP.value

    def test_close_position_records_time_exit_reason(self) -> None:
        """max_hold maps to ExitReason.TIME on the closed TradeResult."""
        trade = self._close_with(exit_reason=ExitReason.TIME, entry_strategy=_DURATION_STRATEGY)
        assert trade is not None
        assert trade.exit_reason == ExitReason.TIME.value

    def test_close_position_records_force_close_exit_reason(self) -> None:
        """last-bar force-close maps to ExitReason.FORCE_CLOSE."""
        trade = self._close_with(exit_reason=ExitReason.FORCE_CLOSE, entry_strategy=_CARRY_STRATEGY)
        assert trade is not None
        assert trade.exit_reason == ExitReason.FORCE_CLOSE.value

    def test_close_position_records_entry_strategy(self) -> None:
        """entry_strategy threads through to the closed TradeResult."""
        trade = self._close_with(exit_reason=ExitReason.STOP, entry_strategy=_CARRY_STRATEGY)
        assert trade is not None
        assert trade.entry_strategy == _CARRY_STRATEGY

    def test_full_run_force_close_at_last_bar_sets_force_close(self) -> None:
        """A held-to-end position is force-closed with ExitReason.FORCE_CLOSE."""
        config = BondBacktestConfig(
            initial_cash=Decimal(1_000_000),
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=500),  # won't trigger
            max_hold_bars=200,  # won't trigger
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
        assert result.trades[0].exit_reason == ExitReason.FORCE_CLOSE.value

    def test_full_run_max_hold_maps_to_time(self) -> None:
        """A position closed by max_hold_bars carries ExitReason.TIME."""
        config = BondBacktestConfig(
            initial_cash=Decimal(1_000_000),
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=500),  # won't trigger
            max_hold_bars=_MAX_HOLD_FOR_MAPPING,
        )
        engine = BondBacktestEngine(config=config)
        prices = [85.0] * 20
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))
        coupons = _make_coupon_schedule(_FIGI, [date(2025, 6, 11)])

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
        )

        assert result.trade_count >= 1
        time_closed = result.trades[0]
        assert time_closed.hold_bars == _MAX_HOLD_FOR_MAPPING
        assert time_closed.exit_reason == ExitReason.TIME.value

    def test_full_run_yield_stop_maps_to_stop(self) -> None:
        """A position closed by yield_stop carries ExitReason.STOP."""
        config = BondBacktestConfig(
            initial_cash=Decimal(1_000_000),
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=50),  # tight stop
            max_hold_bars=200,
        )
        engine = BondBacktestEngine(config=config)
        prices = [85.0] * 5 + [83.0, 82.0, 81.0, 80.0, 79.0]
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))
        coupons = _make_coupon_schedule(_FIGI, [date(2025, 6, 11)])

        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
        )

        assert result.trade_count >= 1
        assert result.trades[0].exit_reason == ExitReason.STOP.value

    def test_full_run_entry_strategy_recorded(self) -> None:
        """The opening strategy name is recorded on the closed TradeResult."""
        config = BondBacktestConfig(
            initial_cash=Decimal(1_000_000),
            dv01_sizer=DV01BudgetStep(
                max_dd_pct=Decimal("0.05"),
                expected_max_rate_move_bps=200,
                max_single_position_pct=Decimal("0.50"),
            ),
            yield_stop=YieldStop(threshold_bps=500),
            max_hold_bars=200,
        )
        engine = BondBacktestEngine(config=config)
        prices = [85.0] * 12
        candles = _make_bond_candles(_SYMBOL, prices, date(2025, 1, 6))
        coupons = _make_coupon_schedule(_FIGI, [date(2025, 6, 11)])

        # _buy_at_bar_strategy emits strategy_name="test_bond_strategy".
        result = engine.run(
            candles_by_symbol={_SYMBOL: candles},
            bond_info={_SYMBOL: _BOND_INFO},
            coupon_schedule={_SYMBOL: coupons},
            strategy_fn=_buy_at_bar_strategy(buy_bar=2),
        )

        assert result.trade_count == 1
        assert result.trades[0].entry_strategy == "test_bond_strategy"


# ---------------------------------------------------------------------------
# Test 16: PnL-inertness of the exit_reason / entry_strategy wiring
# (T-69-07) — populating metadata must not change trade economics.
# ---------------------------------------------------------------------------


class TestBondExitReasonPnLInert:
    """exit_reason / entry_strategy only populate metadata; PnL is unchanged."""

    def _close_once(
        self,
        *,
        exit_reason: object | None,
        entry_strategy: str | None,
    ) -> Any:
        close_date = date(2025, 1, 9)
        engine = BondBacktestEngine()
        broker = BondSimulatedBroker(
            initial_cash=Decimal(1_000_000),
            coupon_schedule={_SYMBOL: []},
            face_value=_FACE_VALUE,
        )
        broker.buy_bond(_SYMBOL, _QTY, Decimal("85.0"), Decimal(0), Decimal(0))
        pos = _make_open_position(entry_strategy=entry_strategy)
        candle_lookup = _single_candle_lookup(_SYMBOL, Decimal("84.0"), close_date)
        return engine._close_position(  # noqa: SLF001
            _SYMBOL,
            pos,
            close_date,
            _CLOSE_BAR,
            candle_lookup,
            None,
            broker,
            {_SYMBOL: _BOND_INFO},
            {_SYMBOL: []},
            exit_reason=exit_reason,
            entry_strategy=entry_strategy,
        )

    def test_pnl_unchanged_with_vs_without_exit_reason(self) -> None:
        """Closing with metadata vs without yields identical economics."""
        with_meta = self._close_once(exit_reason=ExitReason.STOP, entry_strategy=_CARRY_STRATEGY)
        without_meta = self._close_once(exit_reason=None, entry_strategy=None)

        assert with_meta is not None
        assert without_meta is not None
        # Economics identical -- only metadata differs.
        assert with_meta.pnl == without_meta.pnl
        assert with_meta.pnl_pct == without_meta.pnl_pct
        assert with_meta.exit_price == without_meta.exit_price
        assert with_meta.hold_bars == without_meta.hold_bars
        # Metadata is the only difference.
        assert with_meta.exit_reason == ExitReason.STOP.value
        assert without_meta.exit_reason is None
        assert with_meta.entry_strategy == _CARRY_STRATEGY
        assert without_meta.entry_strategy is None
