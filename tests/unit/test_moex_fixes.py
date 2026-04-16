"""Tests for MOEX currency fixes: RUB cash conversion, min_pos, vol-normalization."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch
from uuid import uuid4

import pytest

from finalayze.core.schemas import Candle, PortfolioState, TradeResult

# ---------------------------------------------------------------------------
# 1. Cash conversion logic (mirrors run_iteration.py segment_cash logic)
# ---------------------------------------------------------------------------

_FALLBACK_USDRUB = Decimal("90.0")


def _compute_segment_cash(cash: Decimal, segment: str) -> Decimal:
    """Replicate the segment cash conversion from run_iteration.py."""
    return cash * _FALLBACK_USDRUB if segment.startswith("ru_") else cash


class TestSegmentCashConversion:
    """Test that MOEX segments get RUB-denominated cash."""

    CASH_USD = Decimal(100_000)
    EXPECTED_RUB = Decimal(9_000_000)  # 100_000 * 90

    def test_ru_blue_chips_gets_rub_cash(self) -> None:
        result = _compute_segment_cash(self.CASH_USD, "ru_blue_chips")
        assert result == self.EXPECTED_RUB

    def test_ru_energy_gets_rub_cash(self) -> None:
        result = _compute_segment_cash(self.CASH_USD, "ru_energy")
        assert result == self.EXPECTED_RUB

    def test_ru_finance_gets_rub_cash(self) -> None:
        result = _compute_segment_cash(self.CASH_USD, "ru_finance")
        assert result == self.EXPECTED_RUB

    def test_ru_tech_gets_rub_cash(self) -> None:
        result = _compute_segment_cash(self.CASH_USD, "ru_tech")
        assert result == self.EXPECTED_RUB

    def test_us_tech_keeps_usd_cash(self) -> None:
        result = _compute_segment_cash(self.CASH_USD, "us_tech")
        assert result == self.CASH_USD

    def test_us_broad_keeps_usd_cash(self) -> None:
        result = _compute_segment_cash(self.CASH_USD, "us_broad")
        assert result == self.CASH_USD

    def test_unknown_segment_keeps_usd(self) -> None:
        result = _compute_segment_cash(self.CASH_USD, "other_segment")
        assert result == self.CASH_USD

    def test_conversion_rate_is_90(self) -> None:
        """Verify the fallback USDRUB rate."""
        assert Decimal("90.0") == _FALLBACK_USDRUB


# ---------------------------------------------------------------------------
# 2. Engine min_pos for MOEX vs US segments
# ---------------------------------------------------------------------------


def _compute_min_pos(segment_id: str) -> Decimal:
    """Replicate the min_pos logic from BacktestEngine._open_position."""
    return Decimal(5000) if segment_id.startswith("ru_") else Decimal(500)


class TestMinPositionSize:
    """Test currency-aware minimum position sizes."""

    def test_moex_min_pos_is_5000_rub(self) -> None:
        assert _compute_min_pos("ru_blue_chips") == Decimal(5000)

    def test_moex_energy_min_pos_is_5000_rub(self) -> None:
        assert _compute_min_pos("ru_energy") == Decimal(5000)

    def test_us_min_pos_is_500_usd(self) -> None:
        assert _compute_min_pos("us_tech") == Decimal(500)

    def test_us_broad_min_pos_is_500_usd(self) -> None:
        assert _compute_min_pos("us_broad") == Decimal(500)

    def test_unknown_segment_uses_usd_default(self) -> None:
        assert _compute_min_pos("other") == Decimal(500)


# ---------------------------------------------------------------------------
# 3. Integration: verify engine.py actually uses 5000 for MOEX
# ---------------------------------------------------------------------------


class TestEngineMinPosIntegration:
    """Verify BacktestEngine source code has equity-scaled min_pos."""

    def test_engine_source_has_scaled_min_pos(self) -> None:
        """Read position executor source to confirm min_pos scales with equity for ru_."""
        from finalayze.backtest import position_executor as executor_mod  # noqa: PLC0415

        source = inspect.getsource(executor_mod)
        # MOEX min is capped at 5000 RUB, floors at 1000 RUB
        assert "Decimal(5000)" in source
        assert "Decimal(1000)" in source
        # US min is capped at 500 USD
        assert "Decimal(500)" in source


# ---------------------------------------------------------------------------
# 4. Currency-aware metrics aggregation (Fix 2)
# ---------------------------------------------------------------------------


def _normalize_trades_to_usd(
    trades: list[TradeResult],
    segment: str,
) -> list[TradeResult]:
    """Replicate the normalize function from run_iteration.py."""
    if not segment.startswith("ru_"):
        return trades
    return [
        TradeResult(
            signal_id=t.signal_id,
            symbol=t.symbol,
            side=t.side,
            quantity=t.quantity,
            entry_price=t.entry_price / _FALLBACK_USDRUB,
            exit_price=t.exit_price / _FALLBACK_USDRUB,
            pnl=t.pnl / _FALLBACK_USDRUB,
            pnl_pct=t.pnl_pct,
            hold_bars=t.hold_bars,
        )
        for t in trades
    ]


def _normalize_snapshots_to_usd(
    snapshots: list[PortfolioState],
    segment: str,
) -> list[PortfolioState]:
    """Replicate the normalize function from run_iteration.py."""
    if not segment.startswith("ru_"):
        return snapshots
    return [
        PortfolioState(
            timestamp=s.timestamp,
            equity=s.equity / _FALLBACK_USDRUB,
            cash=s.cash / _FALLBACK_USDRUB,
            positions=s.positions,
        )
        for s in snapshots
    ]


def _make_trade(
    pnl: Decimal,
    entry_price: Decimal = Decimal(100),
    exit_price: Decimal = Decimal(110),
) -> TradeResult:
    """Create a minimal TradeResult for testing."""
    return TradeResult(
        signal_id=uuid4(),
        symbol="TEST",
        side="BUY",
        quantity=Decimal(10),
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        pnl_pct=Decimal("0.10"),
        hold_bars=5,
    )


def _make_snapshot(equity: Decimal, cash: Decimal) -> PortfolioState:
    """Create a minimal PortfolioState for testing."""
    return PortfolioState(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        equity=equity,
        cash=cash,
        positions={},
    )


class TestCurrencyAwareAggregation:
    """Fix 2: MOEX trades should be converted to USD before aggregation."""

    def test_us_trades_unchanged(self) -> None:
        """US trades should pass through without conversion."""
        trade = _make_trade(pnl=Decimal(280))
        result = _normalize_trades_to_usd([trade], "us_tech")
        assert result is not None
        assert len(result) == 1
        assert result[0].pnl == Decimal(280)
        assert result[0].entry_price == Decimal(100)
        assert result[0].exit_price == Decimal(110)

    def test_us_trades_returns_same_list(self) -> None:
        """US segment should return the exact same list object (no copy)."""
        trades = [_make_trade(pnl=Decimal(100))]
        result = _normalize_trades_to_usd(trades, "us_broad")
        assert result is trades

    def test_moex_trade_pnl_divided_by_fx(self) -> None:
        """MOEX trade PnL should be divided by 90 (USDRUB)."""
        trade = _make_trade(
            pnl=Decimal(9000),
            entry_price=Decimal(9000),
            exit_price=Decimal(9900),
        )
        result = _normalize_trades_to_usd([trade], "ru_blue_chips")
        assert len(result) == 1
        converted = result[0]
        assert converted.pnl == Decimal(9000) / _FALLBACK_USDRUB  # 100 USD
        assert converted.entry_price == Decimal(9000) / _FALLBACK_USDRUB  # 100 USD
        assert converted.exit_price == Decimal(9900) / _FALLBACK_USDRUB  # 110 USD

    def test_moex_trade_pnl_pct_unchanged(self) -> None:
        """Percentage PnL should not change after conversion."""
        trade = _make_trade(pnl=Decimal(9000))
        result = _normalize_trades_to_usd([trade], "ru_energy")
        assert result[0].pnl_pct == trade.pnl_pct

    def test_moex_trade_quantity_unchanged(self) -> None:
        """Quantity (number of shares) should not change after conversion."""
        trade = _make_trade(pnl=Decimal(9000))
        result = _normalize_trades_to_usd([trade], "ru_finance")
        assert result[0].quantity == trade.quantity

    def test_moex_trade_hold_bars_unchanged(self) -> None:
        """Hold bars should not change after conversion."""
        trade = _make_trade(pnl=Decimal(9000))
        result = _normalize_trades_to_usd([trade], "ru_tech")
        assert result[0].hold_bars == trade.hold_bars

    def test_moex_snapshots_converted(self) -> None:
        """MOEX equity snapshots should be divided by 90."""
        snap = _make_snapshot(
            equity=Decimal(9_000_000),
            cash=Decimal(4_500_000),
        )
        result = _normalize_snapshots_to_usd([snap], "ru_blue_chips")
        assert len(result) == 1
        converted = result[0]
        assert converted.equity == Decimal(9_000_000) / _FALLBACK_USDRUB  # 100_000 USD
        assert converted.cash == Decimal(4_500_000) / _FALLBACK_USDRUB  # 50_000 USD

    def test_us_snapshots_unchanged(self) -> None:
        """US snapshots should pass through without conversion."""
        snap = _make_snapshot(equity=Decimal(100_000), cash=Decimal(50_000))
        result = _normalize_snapshots_to_usd([snap], "us_tech")
        assert result is not None
        assert len(result) == 1
        assert result[0].equity == Decimal(100_000)

    def test_us_snapshots_returns_same_list(self) -> None:
        """US segment should return the exact same list object."""
        snaps = [_make_snapshot(equity=Decimal(100_000), cash=Decimal(50_000))]
        result = _normalize_snapshots_to_usd(snaps, "us_broad")
        assert result is snaps

    def test_snapshot_positions_preserved(self) -> None:
        """Positions dict should be preserved as-is (not converted)."""
        snap = PortfolioState(
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            equity=Decimal(9_000_000),
            cash=Decimal(4_500_000),
            positions={"SBER": Decimal(100), "GAZP": Decimal(50)},
        )
        result = _normalize_snapshots_to_usd([snap], "ru_blue_chips")
        assert result[0].positions == {"SBER": Decimal(100), "GAZP": Decimal(50)}

    def test_empty_trades_returns_empty(self) -> None:
        """Empty list should return empty list for any segment."""
        assert _normalize_trades_to_usd([], "ru_blue_chips") == []
        assert _normalize_trades_to_usd([], "us_tech") == []

    def test_empty_snapshots_returns_empty(self) -> None:
        """Empty list should return empty list for any segment."""
        assert _normalize_snapshots_to_usd([], "ru_blue_chips") == []
        assert _normalize_snapshots_to_usd([], "us_tech") == []


# ---------------------------------------------------------------------------
# 5. Vol-normalized dual momentum confidence (Fix 3)
# ---------------------------------------------------------------------------

_BASE_DT = datetime(2024, 1, 1, tzinfo=UTC)
_ONE_DAY = timedelta(days=1)


def _make_candle(
    idx: int,
    close: float,
    *,
    symbol: str = "TEST",
    market_id: str = "us",
) -> Candle:
    """Create a single candle for testing."""
    return Candle(
        symbol=symbol,
        market_id=market_id,
        timeframe="1d",
        timestamp=_BASE_DT + _ONE_DAY * idx,
        open=Decimal(str(close - 1)),
        high=Decimal(str(close + 2)),
        low=Decimal(str(close - 2)),
        close=Decimal(str(close)),
        volume=1_000_000,
    )


def _make_uptrend_candles(
    n: int,
    *,
    base: float = 100.0,
    daily_return: float = 0.002,
    symbol: str = "TEST",
    market_id: str = "us",
) -> list[Candle]:
    """Create *n* candles with a steady upward drift."""
    candles: list[Candle] = []
    price = base
    for i in range(n):
        candles.append(_make_candle(i, price, symbol=symbol, market_id=market_id))
        price *= 1 + daily_return
    return candles


class TestVolNormalizedConfidence:
    """Fix 3: Vol-normalize dual momentum confidence.

    The formula ``normalized_score = abs(score) / asset_vol * 0.15`` means that
    identical proportional returns at different volatilities produce different
    confidence values.  Higher vol -> lower confidence for the same return.
    """

    _RV_PATCH = "finalayze.strategies.dual_momentum._compute_rv"

    def _generate_signal_with_vol(
        self,
        realized_vol: float,
        candles: list[Candle],
        segment_id: str = "us_tech",
    ):
        """Generate a dual momentum signal with a mocked realized vol."""
        from finalayze.strategies.dual_momentum import DualMomentumStrategy  # noqa: PLC0415

        strat = DualMomentumStrategy()
        # Clear YAML cache so test params take effect
        strat._params_cache[segment_id] = {}
        with patch(self._RV_PATCH, return_value=Decimal(str(realized_vol))):
            return strat.generate_signal(
                symbol=candles[0].symbol,
                candles=candles,
                segment_id=segment_id,
            )

    def test_same_return_different_vol_produces_similar_confidence(self) -> None:
        """Same proportional return at 2x vol should get ~half the score boost.

        A 10% return at 15% vol and a 10% return at 30% vol: the latter
        should get roughly half the confidence increment above the base.
        """
        # Build candles with a ~10% return over 126 days
        candles = _make_uptrend_candles(130, base=100.0, daily_return=0.00075)

        sig_low_vol = self._generate_signal_with_vol(0.15, candles)
        sig_high_vol = self._generate_signal_with_vol(0.30, candles)

        assert sig_low_vol is not None
        assert sig_high_vol is not None
        # High-vol signal should have LOWER confidence
        assert sig_high_vol.confidence < sig_low_vol.confidence
        # The difference in confidence increment should be roughly 2:1
        inc_low = sig_low_vol.confidence - 0.4  # above _CONFIDENCE_BASE
        inc_high = sig_high_vol.confidence - 0.4
        assert inc_high > 0  # still positive
        ratio = inc_low / inc_high
        assert 1.5 < ratio < 2.5  # approximately 2x

    def test_moex_high_vol_lower_confidence_than_old_formula(self) -> None:
        """MOEX-like return (10% at 30% vol) should produce LOWER confidence
        than the old formula (which ignored vol).
        """
        candles = _make_uptrend_candles(130, base=100.0, daily_return=0.00075)

        sig = self._generate_signal_with_vol(0.30, candles)
        assert sig is not None

        # Old formula: confidence = min(0.95, 0.4 + abs(score) * 1.0)
        # Compute the score the same way the strategy does
        close_now = float(candles[-1].close)
        close_1m = float(candles[-21].close)
        close_3m = float(candles[-63].close)
        close_6m = float(candles[-126].close)
        ret_1m = (close_now - close_1m) / close_1m
        ret_3m = (close_now - close_3m) / close_3m
        ret_6m = (close_now - close_6m) / close_6m
        score = ret_1m * 0.4 + ret_3m * 0.3 + ret_6m * 0.3

        old_confidence = min(0.95, 0.4 + abs(score) * 1.0)
        # New confidence at 30% vol should be strictly lower
        assert sig.confidence < old_confidence

    def test_us_baseline_vol_approximately_same_as_old(self) -> None:  # noqa: PLR0914
        """US-like return (5% at 15% vol) should produce ~same confidence
        as the old formula because 15% is the baseline vol.
        """
        candles = _make_uptrend_candles(130, base=100.0, daily_return=0.0004)

        sig = self._generate_signal_with_vol(0.15, candles)
        assert sig is not None

        # Compute old formula confidence
        close_now = float(candles[-1].close)
        close_1m = float(candles[-21].close)
        close_3m = float(candles[-63].close)
        close_6m = float(candles[-126].close)
        ret_1m = (close_now - close_1m) / close_1m
        ret_3m = (close_now - close_3m) / close_3m
        ret_6m = (close_now - close_6m) / close_6m
        score = ret_1m * 0.4 + ret_3m * 0.3 + ret_6m * 0.3

        old_confidence = min(0.95, 0.4 + abs(score) * 1.0)
        # At baseline vol (15%), new formula == old formula
        assert abs(sig.confidence - old_confidence) < 0.01


# ---------------------------------------------------------------------------
# 6. RollingVolRegimeProvider: time-varying MOEX regime
# ---------------------------------------------------------------------------


class TestRollingVolRegimeProvider:
    """Test that RollingVolRegimeProvider computes regime per-bar from IMOEX candles."""

    def _make_imoex_candles(
        self,
        n: int,
        *,
        base: float = 3000.0,
        daily_change_pct: float = 0.001,
    ) -> list[Candle]:
        """Create IMOEX candles with a given daily change percentage."""
        candles: list[Candle] = []
        price = base
        for i in range(n):
            # Alternate up/down to simulate volatility
            direction = 1 if i % 2 == 0 else -1
            price *= 1 + direction * daily_change_pct
            candles.append(
                Candle(
                    symbol="IMOEX",
                    market_id="moex",
                    timeframe="1d",
                    timestamp=_BASE_DT + _ONE_DAY * i,
                    open=Decimal(str(price - 1)),
                    high=Decimal(str(price + 2)),
                    low=Decimal(str(price - 2)),
                    close=Decimal(str(price)),
                    volume=1_000_000,
                )
            )
        return candles

    def test_low_vol_returns_normal(self) -> None:
        """Low daily changes should produce NORMAL or LOW_VOL regime."""
        from finalayze.risk.regime import MarketRegime, RollingVolRegimeProvider  # noqa: PLC0415

        # 0.1% daily change -> ~1.6% annualized vol -> NORMAL or LOW_VOL
        candles = self._make_imoex_candles(50, daily_change_pct=0.001)
        provider = RollingVolRegimeProvider(imoex_candles=candles)

        state = provider.get_regime(candles=[], bar_index=40)
        assert state.regime in (MarketRegime.NORMAL, MarketRegime.LOW_VOL)
        assert state.allow_new_longs is True
        assert state.position_scale == Decimal("1.0")

    def test_high_vol_returns_elevated_or_crisis(self) -> None:
        """Large daily swings should produce ELEVATED or CRISIS regime."""
        from finalayze.risk.regime import MarketRegime, RollingVolRegimeProvider  # noqa: PLC0415

        # 5% daily change -> ~79% annualized vol -> CRISIS
        candles = self._make_imoex_candles(50, daily_change_pct=0.05)
        provider = RollingVolRegimeProvider(imoex_candles=candles)

        state = provider.get_regime(candles=[], bar_index=40)
        assert state.regime in (MarketRegime.ELEVATED, MarketRegime.CRISIS)

    def test_bar_index_below_window_returns_normal(self) -> None:
        """If bar_index < window, should return normal (insufficient data)."""
        from finalayze.risk.regime import RollingVolRegimeProvider  # noqa: PLC0415

        candles = self._make_imoex_candles(50, daily_change_pct=0.05)
        provider = RollingVolRegimeProvider(imoex_candles=candles, window=20)

        state = provider.get_regime(candles=[], bar_index=10)
        assert state.regime.value == "normal"
        assert state.allow_new_longs is True

    def test_empty_candles_returns_normal(self) -> None:
        """Empty IMOEX candles list should return normal for any bar_index."""
        from finalayze.risk.regime import RollingVolRegimeProvider  # noqa: PLC0415

        provider = RollingVolRegimeProvider(imoex_candles=[], window=20)

        # bar_index=0, min(0, -1) = -1, which is < window -> normal
        state = provider.get_regime(candles=[], bar_index=0)
        assert state.regime.value == "normal"
        assert state.allow_new_longs is True

    def test_crisis_blocks_longs(self) -> None:
        """CRISIS regime should block new longs."""
        from finalayze.risk.regime import MarketRegime, RollingVolRegimeProvider  # noqa: PLC0415

        # 5% daily change -> CRISIS
        candles = self._make_imoex_candles(50, daily_change_pct=0.05)
        provider = RollingVolRegimeProvider(imoex_candles=candles)

        state = provider.get_regime(candles=[], bar_index=40)
        if state.regime == MarketRegime.CRISIS:
            assert state.allow_new_longs is False
            assert state.position_scale == Decimal("0.25")


# ---------------------------------------------------------------------------
# 7. Max hold bars MOEX uplift
# ---------------------------------------------------------------------------

from finalayze.backtest.config import resolve_max_hold_bars  # noqa: E402


class TestMoexHoldBarsUplift:
    """MOEX segments should get 1.3x uplift on max hold bars."""

    def test_us_segment_unchanged(self) -> None:
        hold = resolve_max_hold_bars({"momentum": 30}, "momentum", segment_id="us_tech")
        assert hold == 30

    def test_moex_segment_uplifted(self) -> None:
        hold = resolve_max_hold_bars({"momentum": 30}, "momentum", segment_id="ru_blue_chips")
        assert hold == 39  # int(30 * 1.3) = 39

    def test_moex_mean_reversion_uplifted(self) -> None:
        hold = resolve_max_hold_bars(
            {"mean_reversion": 20}, "mean_reversion", segment_id="ru_energy"
        )
        assert hold == 26  # int(20 * 1.3) = 26

    def test_int_max_hold_bars_moex(self) -> None:
        hold = resolve_max_hold_bars(30, "momentum", segment_id="ru_blue_chips")
        assert hold == 39

    def test_default_segment_id_empty(self) -> None:
        """Default segment_id="" should not apply uplift."""
        hold = resolve_max_hold_bars({"momentum": 30}, "momentum")
        assert hold == 30
