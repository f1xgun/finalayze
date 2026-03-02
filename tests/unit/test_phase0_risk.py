"""Phase 0 risk module tests: regime, position sizing pipeline, Kelly kill switch, cross-market."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from finalayze.risk.circuit_breaker import CircuitLevel
from finalayze.risk.kelly import RollingKelly, TradeRecord
from finalayze.risk.position_sizing_pipeline import (
    HardCapsStep,
    KellyStep,
    PositionSizingPipeline,
    RegimeStep,
    SizingContext,
    VolTargetStep,
)
from finalayze.risk.pre_trade_check import PreTradeChecker
from finalayze.risk.regime import (
    MarketRegime,
    RegimeState,
    compute_regime_state,
)

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────────

_VIX_LOW = Decimal(12)
_VIX_NORMAL = Decimal(18)
_VIX_ELEVATED = Decimal(25)
_VIX_CRISIS = Decimal(35)

_SCALE_FULL = Decimal("1.0")
_SCALE_HALF = Decimal("0.5")
_SCALE_QUARTER = Decimal("0.25")

_EQUITY = Decimal(100000)
_BASE_POSITION = Decimal(10000)
_MAX_POSITION_PCT = Decimal("0.20")
_MIN_POSITION_SIZE = Decimal(500)
_ASSET_VOL = Decimal("0.25")
_TARGET_VOL = Decimal("0.15")

_GOOD_WIN_PNL = Decimal(150)
_GOOD_WIN_PCT = Decimal("0.03")
_SMALL_LOSS_PNL = Decimal(-50)
_SMALL_LOSS_PCT = Decimal("-0.01")
_LARGE_LOSS_PNL = Decimal(-200)
_LARGE_LOSS_PCT = Decimal("-0.05")

_KILL_THRESHOLD = 3
_KELLY_WINDOW = 50
_NEGATIVE_EXPECTANCY_WINS = 5
_NEGATIVE_EXPECTANCY_LOSSES = 45

_MARKET_OPEN_DT = datetime(2026, 2, 25, 15, 0, tzinfo=UTC)
_PORTFOLIO_EQUITY = Decimal(100000)
_AVAILABLE_CASH = Decimal(50000)
_ORDER_VALUE = Decimal(5000)
_OPEN_POSITIONS = 3


# ════════════════════════════════════════════════════════════════════════════
# Task 0.1: Regime logic tests
# ════════════════════════════════════════════════════════════════════════════


class TestRegimeLogic:
    """Tests for compute_regime_state with correct allow_longs logic."""

    def test_crisis_blocks_longs_regardless_of_sma(self) -> None:
        """CRISIS always blocks longs, even when price is above SMA200."""
        state = compute_regime_state(vix_value=_VIX_CRISIS, sma200_above=True)
        assert state.regime == MarketRegime.CRISIS
        assert state.allow_new_longs is False
        assert state.position_scale == _SCALE_QUARTER

    def test_crisis_below_sma_blocks_longs(self) -> None:
        """CRISIS + below SMA200 also blocks longs."""
        state = compute_regime_state(vix_value=_VIX_CRISIS, sma200_above=False)
        assert state.regime == MarketRegime.CRISIS
        assert state.allow_new_longs is False

    def test_elevated_below_sma_blocks_longs(self) -> None:
        """ELEVATED + below SMA200 blocks longs."""
        state = compute_regime_state(vix_value=_VIX_ELEVATED, sma200_above=False)
        assert state.regime == MarketRegime.ELEVATED
        assert state.allow_new_longs is False
        assert state.position_scale == _SCALE_HALF

    def test_elevated_above_sma_allows_longs(self) -> None:
        """ELEVATED + above SMA200 allows longs."""
        state = compute_regime_state(vix_value=_VIX_ELEVATED, sma200_above=True)
        assert state.regime == MarketRegime.ELEVATED
        assert state.allow_new_longs is True

    def test_normal_always_allows_longs(self) -> None:
        """NORMAL and LOW_VOL always allow longs."""
        normal = compute_regime_state(vix_value=_VIX_NORMAL, sma200_above=False)
        assert normal.regime == MarketRegime.NORMAL
        assert normal.allow_new_longs is True
        assert normal.position_scale == _SCALE_FULL

        low_vol = compute_regime_state(vix_value=_VIX_LOW, sma200_above=False)
        assert low_vol.regime == MarketRegime.LOW_VOL
        assert low_vol.allow_new_longs is True
        assert low_vol.position_scale == _SCALE_FULL


# ════════════════════════════════════════════════════════════════════════════
# Task 0.3: PositionSizingPipeline tests
# ════════════════════════════════════════════════════════════════════════════


def _make_context(
    *,
    regime_scale: Decimal = Decimal("1.0"),
    asset_vol: Decimal = _ASSET_VOL,
    target_vol: Decimal = _TARGET_VOL,
) -> SizingContext:
    return SizingContext(
        equity=_EQUITY,
        base_position=_BASE_POSITION,
        max_position_pct=_MAX_POSITION_PCT,
        min_position_size=_MIN_POSITION_SIZE,
        asset_vol=asset_vol,
        target_vol=target_vol,
        regime_scale=regime_scale,
        correlation_scale=Decimal("1.0"),
    )


class TestPositionSizingPipeline:
    """Tests for unified position sizing pipeline."""

    def test_pipeline_crisis_scenario(self) -> None:
        """In crisis (scale=0.25), position is reduced but stays above floor."""
        pipeline = PositionSizingPipeline()
        ctx = _make_context(regime_scale=_SCALE_QUARTER)
        result = pipeline.compute(ctx)
        # Regime step applies 0.25 scale, floor at 0.10 does not kick in
        # Result should be > 0 (not eliminated)
        assert result > Decimal(0)
        # Should be less than base position
        assert result < _BASE_POSITION

    def test_pipeline_normal_scenario(self) -> None:
        """Normal regime with vol targeting produces a reasonable size."""
        pipeline = PositionSizingPipeline()
        ctx = _make_context(regime_scale=_SCALE_FULL)
        result = pipeline.compute(ctx)
        assert result > Decimal(0)
        # Should not exceed max position cap
        max_cap = _EQUITY * _MAX_POSITION_PCT
        assert result <= max_cap

    def test_regime_and_vol_target_both_applied(self) -> None:
        """Regime step and vol target step both reduce position size."""
        pipeline = PositionSizingPipeline()
        # High vol asset with crisis regime
        ctx_crisis = _make_context(
            regime_scale=_SCALE_QUARTER,
            asset_vol=Decimal("0.40"),
        )
        ctx_normal = _make_context(
            regime_scale=_SCALE_FULL,
            asset_vol=Decimal("0.40"),
        )
        crisis_size = pipeline.compute(ctx_crisis)
        normal_size = pipeline.compute(ctx_normal)
        assert crisis_size < normal_size

    def test_pipeline_steps_order(self) -> None:
        """Pipeline applies steps in defined order: Kelly, VolTarget, Regime, HardCaps."""
        pipeline = PositionSizingPipeline()
        expected_types = [KellyStep, VolTargetStep, RegimeStep, HardCapsStep]
        actual_types = [type(s) for s in pipeline.steps]
        assert actual_types == expected_types

    def test_below_min_position_returns_zero(self) -> None:
        """Positions below min_position_size are eliminated."""
        pipeline = PositionSizingPipeline()
        # Very small base position that will shrink below min
        ctx = SizingContext(
            equity=_EQUITY,
            base_position=Decimal(100),  # very small
            max_position_pct=_MAX_POSITION_PCT,
            min_position_size=_MIN_POSITION_SIZE,
            asset_vol=Decimal("0.50"),
            target_vol=_TARGET_VOL,
            regime_scale=_SCALE_QUARTER,
            correlation_scale=Decimal("1.0"),
        )
        result = pipeline.compute(ctx)
        assert result == Decimal(0)


# ════════════════════════════════════════════════════════════════════════════
# Task 0.8: Kelly kill switch tests
# ════════════════════════════════════════════════════════════════════════════


def _fill_negative_window(kelly: RollingKelly) -> None:
    """Fill one full window with negative expectancy trades."""
    for _ in range(_NEGATIVE_EXPECTANCY_WINS):
        kelly.update(TradeRecord(pnl=Decimal(80), pnl_pct=Decimal("0.02")))
    for _ in range(_NEGATIVE_EXPECTANCY_LOSSES):
        kelly.update(TradeRecord(pnl=_LARGE_LOSS_PNL, pnl_pct=_LARGE_LOSS_PCT))


class TestKellyKillSwitch:
    """Tests for negative expectancy kill switch in RollingKelly."""

    def test_kill_switch_activates_after_3_windows(self) -> None:
        """After 3 consecutive full-window negative expectancy, should_halt is True."""
        kelly = RollingKelly(window=_KELLY_WINDOW)
        # Fill 3 consecutive negative windows, calling optimal_fraction after each
        # to trigger the counter increment
        for i in range(_KILL_THRESHOLD):
            _fill_negative_window(kelly)
            result = kelly.optimal_fraction()
            if i < _KILL_THRESHOLD - 1:
                # Not yet at threshold: should return reduced fixed fractional
                assert kelly.should_halt is False
            else:
                # At threshold: should halt
                assert result == Decimal(0)
                assert kelly.should_halt is True

    def test_kill_switch_resets_on_positive(self) -> None:
        """A positive expectancy window resets the consecutive counter."""
        kelly = RollingKelly(window=_KELLY_WINDOW)
        # 2 negative windows, calling optimal_fraction to increment counter
        negative_windows = 2
        for _ in range(negative_windows):
            _fill_negative_window(kelly)
            kelly.optimal_fraction()

        assert kelly.should_halt is False  # not yet at threshold

        # One positive window (good trades replace the bad via deque rolling)
        wins = 40
        losses = _KELLY_WINDOW - wins
        for _ in range(wins):
            kelly.update(TradeRecord(pnl=_GOOD_WIN_PNL, pnl_pct=_GOOD_WIN_PCT))
        for _ in range(losses):
            kelly.update(TradeRecord(pnl=_SMALL_LOSS_PNL, pnl_pct=_SMALL_LOSS_PCT))

        kelly.optimal_fraction()  # triggers reset
        assert kelly.should_halt is False

    def test_kill_switch_returns_zero_fraction(self) -> None:
        """When kill switch is active, optimal_fraction returns exactly zero."""
        kelly = RollingKelly(window=_KELLY_WINDOW)
        for _ in range(_KILL_THRESHOLD):
            _fill_negative_window(kelly)
            kelly.optimal_fraction()  # increment counter each window

        assert kelly.should_halt is True
        # Calling optimal_fraction after halt returns zero
        assert kelly.optimal_fraction() == Decimal(0)
        # Calling again still returns zero
        assert kelly.optimal_fraction() == Decimal(0)


# ════════════════════════════════════════════════════════════════════════════
# Task 0.9: Cross-market exposure fail-closed tests
# ════════════════════════════════════════════════════════════════════════════


def _cross_market_base_kwargs() -> dict:
    return {
        "order_value": _ORDER_VALUE,
        "portfolio_equity": _PORTFOLIO_EQUITY,
        "available_cash": _AVAILABLE_CASH,
        "open_position_count": _OPEN_POSITIONS,
        "market_id": "us",
        "dt": _MARKET_OPEN_DT,
        "circuit_breaker_level": CircuitLevel.NORMAL,
    }


class TestCrossMarketFailClosed:
    """Tests for fail-closed cross-market exposure check."""

    def test_fails_closed_when_exposure_not_provided(self) -> None:
        """Multiple markets active but cross_market_exposure_pct is None -> violation."""
        checker = PreTradeChecker()
        kwargs = _cross_market_base_kwargs()
        kwargs["markets_active"] = ["us", "moex"]
        kwargs["cross_market_exposure_pct"] = None
        kwargs["max_cross_market_exposure_pct"] = Decimal("0.60")
        result = checker.check(**kwargs)
        assert not result.passed
        violation_text = " ".join(result.violations)
        assert "cross-market" in violation_text.lower() or "cross_market" in violation_text.lower()

    def test_passes_when_exposure_provided(self) -> None:
        """Multiple markets with valid exposure percentage passes."""
        checker = PreTradeChecker()
        kwargs = _cross_market_base_kwargs()
        kwargs["markets_active"] = ["us", "moex"]
        kwargs["cross_market_exposure_pct"] = Decimal("0.40")
        kwargs["max_cross_market_exposure_pct"] = Decimal("0.60")
        result = checker.check(**kwargs)
        # No cross-market violation (other checks may still fail/pass)
        cross_violations = [
            v for v in result.violations if "cross" in v.lower() or "market" in v.lower()
        ]
        assert len(cross_violations) == 0

    def test_single_market_skips_check(self) -> None:
        """Single market active skips cross-market check even with None exposure."""
        checker = PreTradeChecker()
        kwargs = _cross_market_base_kwargs()
        kwargs["markets_active"] = ["us"]
        kwargs["cross_market_exposure_pct"] = None
        kwargs["max_cross_market_exposure_pct"] = Decimal("0.60")
        result = checker.check(**kwargs)
        cross_violations = [v for v in result.violations if "cross" in v.lower()]
        assert len(cross_violations) == 0
