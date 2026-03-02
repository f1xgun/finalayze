"""Phase A regime tests: VIX provider, MOEX regime, vol targeting, realized vol."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.risk.position_sizing_pipeline import (
    PositionSizingPipeline,
    SizingContext,
    VolTargetStep,
)
from finalayze.risk.regime import (
    MarketRegime,
    RegimeState,
    VIXRegimeProvider,
    compute_moex_regime_state,
    compute_realized_vol,
    compute_regime_state,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_BASE_DT = datetime(2024, 6, 1, 14, 30, tzinfo=UTC)
_ONE_DAY = timedelta(days=1)

_VIX_LOW = Decimal(12)
_VIX_ELEVATED = Decimal(25)
_VIX_CRISIS = Decimal(35)
_VIX_NORMAL_HIGH = Decimal(22)

_SCALE_FULL = Decimal("1.0")
_SCALE_HALF = Decimal("0.5")
_SCALE_QUARTER = Decimal("0.25")

_EQUITY = Decimal(100_000)
_BASE_POSITION = Decimal(10_000)
_MAX_POSITION_PCT = Decimal("0.20")
_MIN_POSITION_SIZE = Decimal(500)

_VOL_WINDOW = 20
_ANNUALIZATION_FACTOR = 252


def _make_candle(
    symbol: str,
    close: Decimal,
    idx: int,
    *,
    market_id: str = "us",
    base_dt: datetime = _BASE_DT,
) -> Candle:
    """Create a single candle with the given close price."""
    return Candle(
        symbol=symbol,
        market_id=market_id,
        timeframe="1d",
        timestamp=base_dt + _ONE_DAY * idx,
        open=close - Decimal(1),
        high=close + Decimal(2),
        low=close - Decimal(2),
        close=close,
        volume=1_000_000,
    )


def _make_vix_candles(
    values: list[Decimal],
    *,
    base_dt: datetime = _BASE_DT,
) -> list[Candle]:
    """Create VIX candle series from a list of close values."""
    return [
        _make_candle("VIX", v, i, market_id="us", base_dt=base_dt) for i, v in enumerate(values)
    ]


def _make_spy_candles(
    values: list[Decimal],
    *,
    base_dt: datetime = _BASE_DT,
) -> list[Candle]:
    """Create SPY candle series for SMA200 calculation."""
    return [
        _make_candle("SPY", v, i, market_id="us", base_dt=base_dt) for i, v in enumerate(values)
    ]


def _make_asset_candles(
    closes: list[Decimal],
    *,
    symbol: str = "TEST",
    market_id: str = "us",
) -> list[Candle]:
    """Create asset candle series from close prices."""
    return [_make_candle(symbol, c, i, market_id=market_id) for i, c in enumerate(closes)]


# ═══════════════════════════════════════════════════════════════════════════════
# A.1: VIXRegimeProvider tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestVIXRegimeProvider:
    """Tests for VIXRegimeProvider."""

    def test_vix_regime_provider_low_vol(self) -> None:
        """VIX=12 maps to LOW_VOL with scale=1.0."""
        vix_count = 10
        vix_candles = _make_vix_candles([_VIX_LOW] * vix_count)
        provider = VIXRegimeProvider(vix_candles=vix_candles)

        # Asset candles with matching timestamps
        asset_candles = _make_asset_candles([Decimal(100 + i) for i in range(vix_count)])

        result = provider.get_regime(asset_candles, vix_count - 1)
        assert result.regime == MarketRegime.LOW_VOL
        assert result.position_scale == _SCALE_FULL

    def test_vix_regime_provider_elevated(self) -> None:
        """VIX=25 maps to ELEVATED with scale=0.5."""
        vix_count = 10
        vix_candles = _make_vix_candles([_VIX_ELEVATED] * vix_count)
        provider = VIXRegimeProvider(vix_candles=vix_candles)

        asset_candles = _make_asset_candles([Decimal(100 + i) for i in range(vix_count)])

        result = provider.get_regime(asset_candles, vix_count - 1)
        assert result.regime == MarketRegime.ELEVATED
        assert result.position_scale == _SCALE_HALF

    def test_vix_regime_provider_crisis(self) -> None:
        """VIX=35 maps to CRISIS with scale=0.25, no longs."""
        vix_count = 10
        vix_candles = _make_vix_candles([_VIX_CRISIS] * vix_count)
        provider = VIXRegimeProvider(vix_candles=vix_candles)

        asset_candles = _make_asset_candles([Decimal(100 + i) for i in range(vix_count)])

        result = provider.get_regime(asset_candles, vix_count - 1)
        assert result.regime == MarketRegime.CRISIS
        assert result.position_scale == _SCALE_QUARTER
        assert result.allow_new_longs is False

    def test_vix_momentum_upgrade(self) -> None:
        """VIX=22 but rising 7pts in 5 days upgrades to ELEVATED."""
        # VIX starts at 15, jumps to 22 over last 5 bars (delta > 5)
        vix_values = [Decimal(15)] * 5 + [Decimal(22)]
        vix_candles = _make_vix_candles(vix_values)
        provider = VIXRegimeProvider(vix_candles=vix_candles)

        asset_candles = _make_asset_candles([Decimal(100 + i) for i in range(len(vix_values))])

        result = provider.get_regime(asset_candles, len(vix_values) - 1)
        # VIX=22 is ELEVATED by threshold (20-30), and momentum confirms
        assert result.regime == MarketRegime.ELEVATED

    def test_vix_momentum_upgrade_from_normal(self) -> None:
        """VIX=19 (NORMAL) but rising >5pts in 5 days upgrades to ELEVATED."""
        # VIX starts at 13, jumps to 19 over last 5 bars (delta=6 > 5)
        vix_values = [Decimal(13)] * 5 + [Decimal(19)]
        vix_candles = _make_vix_candles(vix_values)
        provider = VIXRegimeProvider(vix_candles=vix_candles)

        asset_candles = _make_asset_candles([Decimal(100 + i) for i in range(len(vix_values))])

        result = provider.get_regime(asset_candles, len(vix_values) - 1)
        # Normally VIX=19 would be NORMAL, but momentum upgrade -> ELEVATED
        assert result.regime == MarketRegime.ELEVATED

    def test_vix_forward_fill(self) -> None:
        """Missing VIX bar uses latest available (forward-fill)."""
        vix_count = 5
        vix_candles = _make_vix_candles([_VIX_LOW] * vix_count)
        provider = VIXRegimeProvider(vix_candles=vix_candles)

        # Asset candles extend beyond VIX data
        extended_count = 10
        asset_candles = _make_asset_candles([Decimal(100 + i) for i in range(extended_count)])

        # bar_index=9 is beyond VIX data, should forward-fill last VIX=12
        result = provider.get_regime(asset_candles, extended_count - 1)
        assert result.regime == MarketRegime.LOW_VOL
        assert result.vix_value == _VIX_LOW


# ═══════════════════════════════════════════════════════════════════════════════
# A.1: MOEX regime tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMOEXRegime:
    """Tests for compute_moex_regime_state using realized volatility."""

    def test_moex_regime_normal(self) -> None:
        """Realized vol 20% maps to NORMAL."""
        state = compute_moex_regime_state(realized_vol=Decimal("0.20"))
        assert state.regime == MarketRegime.NORMAL
        assert state.allow_new_longs is True

    def test_moex_regime_elevated(self) -> None:
        """Realized vol 30% maps to ELEVATED."""
        state = compute_moex_regime_state(realized_vol=Decimal("0.30"))
        assert state.regime == MarketRegime.ELEVATED
        assert state.position_scale == _SCALE_HALF

    def test_moex_regime_high_vol_crisis(self) -> None:
        """Realized vol 45% maps to CRISIS."""
        state = compute_moex_regime_state(realized_vol=Decimal("0.45"))
        assert state.regime == MarketRegime.CRISIS
        assert state.allow_new_longs is False
        assert state.position_scale == _SCALE_QUARTER

    def test_moex_regime_low_vol(self) -> None:
        """Realized vol 10% maps to LOW_VOL (below 25% threshold)."""
        state = compute_moex_regime_state(realized_vol=Decimal("0.10"))
        assert state.regime == MarketRegime.LOW_VOL
        assert state.position_scale == _SCALE_FULL


# ═══════════════════════════════════════════════════════════════════════════════
# A.2: compute_realized_vol tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeRealizedVol:
    """Tests for compute_realized_vol helper."""

    def test_compute_realized_vol(self) -> None:
        """Verify annualized vol calculation from daily closes."""
        # Create 21 candles (need 20 returns from 21 closes)
        # Constant 1% daily return -> daily vol = 0, but let's use alternating
        closes = []
        for i in range(21):
            price = Decimal(100) if i % 2 == 0 else Decimal(102)
            closes.append(price)

        candles = _make_asset_candles(closes)
        vol = compute_realized_vol(candles, window=_VOL_WINDOW)

        # Should be a positive annualized volatility
        assert vol > Decimal(0)
        # Daily returns alternate between ~+2% and ~-2%
        # Daily vol should be about 2%, annualized ~31%
        assert Decimal("0.10") < vol < Decimal("0.60")

    def test_compute_realized_vol_insufficient_data(self) -> None:
        """Returns 0 when insufficient candles for the window."""
        candles = _make_asset_candles([Decimal(100)] * 5)
        vol = compute_realized_vol(candles, window=_VOL_WINDOW)
        assert vol == Decimal(0)


# ═══════════════════════════════════════════════════════════════════════════════
# A.2: VolTargetStep tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_sizing_context(
    *,
    asset_vol: Decimal = Decimal("0.20"),
    target_vol: Decimal = Decimal("0.10"),
    regime_scale: Decimal = Decimal("1.0"),
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


class TestVolTargetStep:
    """Tests for VolTargetStep in position sizing pipeline."""

    def test_vol_target_step_scales_position(self) -> None:
        """target_vol=10%, asset_vol=20% -> 0.5x scaling."""
        step = VolTargetStep()
        ctx = _make_sizing_context(
            target_vol=Decimal("0.10"),
            asset_vol=Decimal("0.20"),
        )
        result = step.adjust(_BASE_POSITION, ctx)
        expected = _BASE_POSITION * Decimal("0.5")
        assert result == expected

    def test_vol_target_step_bounds_upper(self) -> None:
        """Scaling capped at 2.0x when asset_vol is very low."""
        step = VolTargetStep()
        ctx = _make_sizing_context(
            target_vol=Decimal("0.30"),
            asset_vol=Decimal("0.05"),  # ratio = 6.0, capped to 2.0
        )
        result = step.adjust(_BASE_POSITION, ctx)
        expected = _BASE_POSITION * Decimal("2.0")
        assert result == expected

    def test_vol_target_step_bounds_lower(self) -> None:
        """Scaling floored at 0.25x when asset_vol is very high."""
        step = VolTargetStep()
        ctx = _make_sizing_context(
            target_vol=Decimal("0.05"),
            asset_vol=Decimal("0.80"),  # ratio = 0.0625, capped to 0.25
        )
        result = step.adjust(_BASE_POSITION, ctx)
        expected = _BASE_POSITION * Decimal("0.25")
        assert result == expected

    def test_vol_target_step_zero_asset_vol(self) -> None:
        """Zero asset vol passes through unchanged."""
        step = VolTargetStep()
        ctx = _make_sizing_context(
            target_vol=Decimal("0.10"),
            asset_vol=Decimal(0),
        )
        result = step.adjust(_BASE_POSITION, ctx)
        assert result == _BASE_POSITION


# ═══════════════════════════════════════════════════════════════════════════════
# A.3: Confirm no VIX-rank step
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoVIXRankStep:
    """Confirm VIX-rank is NOT a separate step in the pipeline."""

    def test_pipeline_has_no_vix_rank_step(self) -> None:
        """Default pipeline should not contain a VIX-rank step."""
        pipeline = PositionSizingPipeline()
        step_names = [type(s).__name__ for s in pipeline.steps]
        for name in step_names:
            assert "vix" not in name.lower() or name == "VolTargetStep"
            assert "rank" not in name.lower()
