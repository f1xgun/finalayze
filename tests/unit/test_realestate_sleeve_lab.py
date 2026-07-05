"""Unit tests for the real-estate sleeve lab (Phase C, beyond-MOEX-edge R&D).

The two NEW production helpers real estate needs and gold/ZO did not:

- ``accrue_rental_yield`` — MREDC is a PRICE-only index; real estate's whole income
  point is rent, so the cert overlays a labelled *net* rental accrual. This is the
  tested primitive.
- ``bars_per_year`` — quantifies sampling frequency so the cert can FLAG MREDC's
  weekly-smoothed low volatility as an artifact (a daily-traded asset is ~252/yr; a
  weekly appraisal index is ~52/yr).
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from finalayze.backtest.realestate_sleeve_lab import accrue_rental_yield, bars_per_year

_DAY = timedelta(days=1)


def test_zero_yield_returns_price_curve_unchanged() -> None:
    curve = [
        (date(2024, 1, 1), Decimal(100)),
        (date(2024, 6, 1), Decimal(110)),
        (date(2024, 12, 1), Decimal(105)),
    ]
    out = accrue_rental_yield(curve, Decimal(0))
    assert out == curve  # no rent -> identical price path


def test_flat_price_one_year_accrues_the_net_yield() -> None:
    # A single 365-day step over a flat price at 4% net -> final ~= 1.04x.
    curve = [(date(2024, 1, 1), Decimal(100)), (date(2024, 12, 31), Decimal(100))]
    out = accrue_rental_yield(curve, Decimal(4))
    factor = out[-1][1] / out[0][1]
    # 365-day ACT/365 accrual at 4% -> exactly 1.04 (0-day base excluded).
    assert abs(factor - Decimal("1.04")) < Decimal("0.0005")


def test_first_point_is_the_base_and_dates_preserved() -> None:
    curve = [(date(2024, 1, 1), Decimal(250)), (date(2024, 2, 1), Decimal(255))]
    out = accrue_rental_yield(curve, Decimal(5))
    assert out[0] == curve[0]  # base point unchanged
    assert [d for d, _ in out] == [d for d, _ in curve]  # dates preserved


def test_rent_arm_dominates_price_arm_pointwise_for_positive_yield() -> None:
    curve = [(date(2024, 1, 1) + i * 30 * _DAY, Decimal(100 + i)) for i in range(6)]
    priced = accrue_rental_yield(curve, Decimal(0))
    rented = accrue_rental_yield(curve, Decimal(4))
    # rent can only add value; every non-base point is strictly higher
    assert all(r[1] >= p[1] for r, p in zip(rented, priced, strict=True))
    assert rented[-1][1] > priced[-1][1]


def test_empty_curve_is_empty() -> None:
    assert accrue_rental_yield([], Decimal(4)) == []


def test_bars_per_year_flags_weekly_vs_daily() -> None:
    start = date(2024, 1, 1)
    weekly = [start + i * 7 * _DAY for i in range(52)]  # ~1yr weekly
    daily = [start + i * _DAY for i in range(365)]  # ~1yr daily
    wk = bars_per_year(weekly)
    dl = bars_per_year(daily)
    assert 40 < wk < 65  # ~52/yr
    assert 240 < dl < 400  # ~365 calendar bars/yr
    assert wk < dl  # weekly index is structurally lower-frequency (smoothed)


def test_bars_per_year_degenerate_inputs() -> None:
    assert bars_per_year([]) == 0.0
    assert bars_per_year([date(2024, 1, 1)]) == 0.0
