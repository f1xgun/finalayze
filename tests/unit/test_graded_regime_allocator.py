"""Unit tests for the graded-regime SAA weight interpolation (core.allocation)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.core.allocation import graded_regime_weights
from finalayze.core.schemas import AssetClass

# balanced profile vectors (config/allocation_profiles.yaml)
_HR = {
    AssetClass.DEPOSIT: Decimal("0.60"),
    AssetClass.OFZ_PK: Decimal("0.10"),
    AssetClass.EQUITY: Decimal("0.30"),
}
_EA = {
    AssetClass.DEPOSIT: Decimal("0.25"),
    AssetClass.OFZ_PK: Decimal("0.40"),
    AssetClass.EQUITY: Decimal("0.35"),
}
_PEAK = Decimal(21)
_NEUTRAL = Decimal("7.5")


def _sum(w: dict[AssetClass, Decimal]) -> Decimal:
    return sum(w.values(), Decimal(0))


def test_at_peak_returns_high_rate() -> None:
    w = graded_regime_weights(_HR, _EA, _PEAK, _PEAK, _NEUTRAL)
    assert w == _HR


def test_at_neutral_returns_easing() -> None:
    w = graded_regime_weights(_HR, _EA, _NEUTRAL, _PEAK, _NEUTRAL)
    assert w == _EA


def test_above_peak_clamps_to_high_rate() -> None:
    w = graded_regime_weights(_HR, _EA, Decimal(25), _PEAK, _NEUTRAL)
    assert w == _HR


def test_below_neutral_clamps_to_easing() -> None:
    w = graded_regime_weights(_HR, _EA, Decimal(5), _PEAK, _NEUTRAL)
    assert w == _EA


def test_midpoint_is_average() -> None:
    mid = (_PEAK + _NEUTRAL) / 2  # t = 0.5
    w = graded_regime_weights(_HR, _EA, mid, _PEAK, _NEUTRAL)
    assert w[AssetClass.DEPOSIT] == (Decimal("0.60") + Decimal("0.25")) / 2  # 0.425
    assert w[AssetClass.OFZ_PK] == (Decimal("0.10") + Decimal("0.40")) / 2  # 0.25
    assert w[AssetClass.EQUITY] == (Decimal("0.30") + Decimal("0.35")) / 2  # 0.325


def test_current_1425_partial_shift() -> None:
    # today's 14.25%: t = (21-14.25)/(21-7.5) = 0.5 -> deposit 42.5% (NOT the binary 25%)
    w = graded_regime_weights(_HR, _EA, Decimal("14.25"), _PEAK, _NEUTRAL)
    assert abs(w[AssetClass.DEPOSIT] - Decimal("0.425")) < Decimal("1e-9")


def test_sums_to_one_across_the_path() -> None:
    # A convex combo with a repeating-decimal t sums to 1.0 only up to Decimal rounding.
    for rate in ("21", "16", "14.25", "13", "10", "8", "7.5"):
        w = graded_regime_weights(_HR, _EA, Decimal(rate), _PEAK, _NEUTRAL)
        assert abs(_sum(w) - Decimal("1.0")) < Decimal("1e-20")
        assert all(v >= 0 for v in w.values())


def test_monotone_deposit_falls_as_rate_falls() -> None:
    rates = [Decimal(r) for r in ("21", "18", "16", "14.25", "12", "10", "8")]
    deposits = [
        graded_regime_weights(_HR, _EA, r, _PEAK, _NEUTRAL)[AssetClass.DEPOSIT] for r in rates
    ]
    assert deposits == sorted(deposits, reverse=True)  # non-increasing


def test_raises_when_peak_not_above_neutral() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        graded_regime_weights(_HR, _EA, Decimal(10), Decimal("7.5"), Decimal("7.5"))


def test_raises_on_mismatched_classes() -> None:
    bad = {AssetClass.DEPOSIT: Decimal("1.0")}
    with pytest.raises(ValueError, match="same asset classes"):
        graded_regime_weights(_HR, bad, Decimal(10), _PEAK, _NEUTRAL)
