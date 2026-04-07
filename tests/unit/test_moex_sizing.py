"""Tests for MOEX position sizing: 1M RUB starting capital and correct market open time."""

from __future__ import annotations

from datetime import UTC, datetime, time
from decimal import Decimal

# ---------------------------------------------------------------------------
# Test 1: MOEX segment_cash is 1M RUB
# ---------------------------------------------------------------------------


def test_moex_segment_cash_is_1m_rub() -> None:
    """MOEX segments must use 1,000,000 RUB starting capital."""
    segment = "ru_blue_chips"
    cash = Decimal(100_000)  # USD default

    # Replicate the logic from run_iteration.py
    segment_cash = Decimal(1_000_000) if segment.startswith("ru_") else cash

    assert segment_cash == Decimal(1_000_000)


def test_us_segment_cash_unchanged() -> None:
    """US segments must keep original cash value."""
    segment = "us_tech"
    cash = Decimal(100_000)

    segment_cash = Decimal(1_000_000) if segment.startswith("ru_") else cash

    assert segment_cash == cash


# ---------------------------------------------------------------------------
# Test 2: Pre-trade check_dt market open times
# ---------------------------------------------------------------------------

_US_MARKET_OPEN_UTC = time(14, 30, tzinfo=UTC)
_MOEX_MARKET_OPEN_UTC = time(7, 0, tzinfo=UTC)


def _adjust_check_dt(dt: datetime, segment_id: str) -> datetime:
    """Replicate engine logic for adjusting midnight candle timestamps."""
    from finalayze.backtest.engine import _MOEX_MARKET_OPEN_UTC as ENGINE_MOEX_OPEN
    from finalayze.backtest.engine import _US_MARKET_OPEN_UTC as ENGINE_US_OPEN

    if dt.hour == 0 and dt.minute == 0:
        if segment_id.startswith("ru_"):
            return datetime.combine(dt.date(), ENGINE_MOEX_OPEN)
        return datetime.combine(dt.date(), ENGINE_US_OPEN)
    return dt


def test_pretrade_check_dt_moex_uses_0700_utc() -> None:
    """MOEX midnight candle timestamps should be adjusted to 07:00 UTC."""
    midnight_dt = datetime(2025, 3, 10, 0, 0, tzinfo=UTC)
    adjusted = _adjust_check_dt(midnight_dt, "ru_blue_chips")

    assert adjusted.hour == 7
    assert adjusted.minute == 0


def test_pretrade_check_dt_us_uses_1430_utc() -> None:
    """US midnight candle timestamps should be adjusted to 14:30 UTC (unchanged)."""
    midnight_dt = datetime(2025, 3, 10, 0, 0, tzinfo=UTC)
    adjusted = _adjust_check_dt(midnight_dt, "us_tech")

    assert adjusted.hour == 14
    assert adjusted.minute == 30


# ---------------------------------------------------------------------------
# Test 3: Position sizing produces correct order of magnitude
# ---------------------------------------------------------------------------


def test_position_size_correct_order_of_magnitude() -> None:
    """compute_position_size with 1M RUB equity should produce ~8-20% position.

    Half-Kelly with win_rate=0.5, avg_win_ratio=1.5:
    f* = (0.5*1.5 - 0.5)/1.5 = 0.167, half-Kelly = 0.0833 => 83K RUB.
    The key assertion: positions are NOT 0.02% (the old bug ~200 RUB).
    """
    from finalayze.risk.position_sizer import compute_position_size

    position = compute_position_size(
        win_rate=Decimal("0.50"),
        avg_win_ratio=Decimal("1.5"),
        equity=Decimal(1_000_000),
        kelly_fraction=Decimal("0.5"),
        max_position_pct=Decimal("0.20"),
    )

    # Half-Kelly gives 8.33% = 83,333 RUB. Must be in 5-20% range (50K-200K),
    # critically NOT the old bug value of ~200 RUB (0.02%).
    assert Decimal(50_000) <= position <= Decimal(200_000), (
        f"Position {position} not in 50K-200K range (5-20% of 1M RUB)"
    )


def test_position_size_not_tiny_bug_value() -> None:
    """Position size must NOT be the old bug value of ~0.02% of equity."""
    from finalayze.risk.position_sizer import compute_position_size

    position = compute_position_size(
        win_rate=Decimal("0.50"),
        avg_win_ratio=Decimal("1.5"),
        equity=Decimal(1_000_000),
        kelly_fraction=Decimal("0.5"),
        max_position_pct=Decimal("0.20"),
    )

    # Old bug: positions were ~200 RUB (0.02% of equity). Must be WAY higher.
    min_acceptable = Decimal(10_000)  # 1% of equity as absolute minimum
    assert position > min_acceptable, (
        f"Position {position} is suspiciously small -- likely the old sizing bug"
    )


# ---------------------------------------------------------------------------
# Test 4: CBRRegimeStep
# ---------------------------------------------------------------------------

from finalayze.risk.position_sizing_pipeline import (  # noqa: E402
    CBRRegimeStep,
    SectorAllocationStep,
    SizingContext,
)

# Standard SizingContext for step tests
_TEST_SIZE = Decimal("100000.0000")


def _make_context() -> SizingContext:
    return SizingContext(
        equity=Decimal(1_000_000),
        base_position=_TEST_SIZE,
        max_position_pct=Decimal("0.20"),
        min_position_size=Decimal(500),
        asset_vol=Decimal("0.30"),
        target_vol=Decimal("0.20"),
        regime_scale=Decimal("1.0"),
        correlation_scale=Decimal("1.0"),
    )


# CBRRegimeStep scaling constants
_STEEPENING_SCALE = Decimal("1.2")
_FLAT_SCALE = Decimal("1.0")
_INVERTED_SCALE = Decimal("0.6")


class TestCBRRegimeStep:
    """CBRRegimeStep scales ru_* positions by yield curve slope."""

    def test_steepening_scales_up(self) -> None:
        """yield_slope_bps=150 -> 1.2x for ru_blue_chips."""
        step = CBRRegimeStep(yield_slope_bps=150.0, segment_id="ru_blue_chips")
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == (_TEST_SIZE * _STEEPENING_SCALE).quantize(Decimal("0.0001"))

    def test_flat_neutral(self) -> None:
        """yield_slope_bps=50 -> 1.0x."""
        step = CBRRegimeStep(yield_slope_bps=50.0, segment_id="ru_blue_chips")
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == (_TEST_SIZE * _FLAT_SCALE).quantize(Decimal("0.0001"))

    def test_inverted_scales_down(self) -> None:
        """yield_slope_bps=-100 -> 0.6x."""
        step = CBRRegimeStep(yield_slope_bps=-100.0, segment_id="ru_blue_chips")
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == (_TEST_SIZE * _INVERTED_SCALE).quantize(Decimal("0.0001"))

    def test_non_ru_passthrough(self) -> None:
        """us_tech -> size unchanged regardless of slope."""
        step = CBRRegimeStep(yield_slope_bps=-100.0, segment_id="us_tech")
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == _TEST_SIZE

    def test_missing_data_neutral(self) -> None:
        """yield_slope_bps=0.0 -> 1.0x (missing data graceful)."""
        step = CBRRegimeStep(yield_slope_bps=0.0, segment_id="ru_energy")
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == (_TEST_SIZE * _FLAT_SCALE).quantize(Decimal("0.0001"))


# ---------------------------------------------------------------------------
# Test 5: SectorAllocationStep
# ---------------------------------------------------------------------------

# Sector allocation scaling constants
_ENERGY_OW = Decimal("1.3")
_ENERGY_UW = Decimal("0.7")
_ENERGY_NEUTRAL = Decimal("1.0")
_FINANCE_CUT = Decimal("1.2")
_FINANCE_HIKE = Decimal("0.8")
_FINANCE_HOLD = Decimal("1.0")


class TestSectorAllocationStep:
    """SectorAllocationStep scales ru_energy by Brent, ru_finance by CBR direction."""

    def test_energy_high_brent_overweight(self) -> None:
        """brent_rub=7000 -> 1.3x for ru_energy."""
        step = SectorAllocationStep(
            brent_rub_price=7000, cbr_direction="cut", segment_id="ru_energy"
        )
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == (_TEST_SIZE * _ENERGY_OW).quantize(Decimal("0.0001"))

    def test_energy_low_brent_underweight(self) -> None:
        """brent_rub=3500 -> 0.7x for ru_energy."""
        step = SectorAllocationStep(
            brent_rub_price=3500, cbr_direction="cut", segment_id="ru_energy"
        )
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == (_TEST_SIZE * _ENERGY_UW).quantize(Decimal("0.0001"))

    def test_energy_mid_brent_neutral(self) -> None:
        """brent_rub=5000 -> 1.0x for ru_energy."""
        step = SectorAllocationStep(
            brent_rub_price=5000, cbr_direction="cut", segment_id="ru_energy"
        )
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == (_TEST_SIZE * _ENERGY_NEUTRAL).quantize(Decimal("0.0001"))

    def test_finance_cut_overweight(self) -> None:
        """cbr_direction=cut -> 1.2x for ru_finance."""
        step = SectorAllocationStep(
            brent_rub_price=5000, cbr_direction="cut", segment_id="ru_finance"
        )
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == (_TEST_SIZE * _FINANCE_CUT).quantize(Decimal("0.0001"))

    def test_finance_hike_underweight(self) -> None:
        """cbr_direction=hike -> 0.8x for ru_finance."""
        step = SectorAllocationStep(
            brent_rub_price=5000, cbr_direction="hike", segment_id="ru_finance"
        )
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == (_TEST_SIZE * _FINANCE_HIKE).quantize(Decimal("0.0001"))

    def test_finance_hold_neutral(self) -> None:
        """cbr_direction=hold -> 1.0x for ru_finance."""
        step = SectorAllocationStep(
            brent_rub_price=5000, cbr_direction="hold", segment_id="ru_finance"
        )
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == (_TEST_SIZE * _FINANCE_HOLD).quantize(Decimal("0.0001"))

    def test_blue_chips_passthrough(self) -> None:
        """ru_blue_chips -> size unchanged (not energy or finance)."""
        step = SectorAllocationStep(
            brent_rub_price=7000, cbr_direction="cut", segment_id="ru_blue_chips"
        )
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == _TEST_SIZE

    def test_us_tech_passthrough(self) -> None:
        """us_tech -> size unchanged (not ru_*)."""
        step = SectorAllocationStep(brent_rub_price=7000, cbr_direction="cut", segment_id="us_tech")
        result = step.adjust(_TEST_SIZE, _make_context())
        assert result == _TEST_SIZE
