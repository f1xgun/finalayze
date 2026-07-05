"""Real-estate sleeve lab — beyond-MOEX-edge R&D, Phase C (diagnostic, backtest-only).

Real estate is the third and final "new asset class" candidate (after gold -> Phase A,
ZO -> Phase B). Unlike those two it is the ONE candidate that pays *income* (rent), which
is exactly what the operator's income goal asks for — so the cert must model that income,
not just price appreciation. But it comes with two honesty-critical traps that gold/ZO did
not have, and this module holds the two tested primitives for them:

- :func:`accrue_rental_yield` — the MREDC index is PRICE-only (residential sale price per
  sq.m); real estate's whole point is the rental coupon. The cert overlays a labelled *net*
  (post-cost, post-NDFL) rental accrual on top of the price path. This is that primitive —
  pure Decimal, ACT/365, compounding.
- :func:`bars_per_year` — MREDC updates ~WEEKLY and is a transaction/appraisal index, so
  its measured volatility and drawdown are structurally UNDERSTATED vs a daily-traded asset.
  The cert uses this to FLAG the smoothing (a real rental ZPIF wrapper carries the market
  volatility + illiquidity + fees the index hides).

The blend/verdict machinery is reused unchanged from :mod:`finalayze.backtest.gold_sleeve_lab`
(``blend_portfolio``, ``diversification_verdict``, ``forward_align_legs``, ``master_axis``).
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

_DAYS_PER_YEAR = Decimal(365)
_ONE = Decimal(1)
_HUNDRED = Decimal(100)
_MIN_POINTS = 2  # a frequency needs at least two dated points to span an interval


def accrue_rental_yield(
    curve: list[tuple[date, Decimal]], net_annual_pct: Decimal
) -> list[tuple[date, Decimal]]:
    """Overlay a net rental-income accrual on a price-level curve (ACT/365, compounding).

    ``curve`` is a ``(date, price_level)`` series (for the cert: the MREDC price path,
    already net of price-appreciation NDFL). ``net_annual_pct`` is the assumed rental yield
    **net** of vacancy, management, property tax, repairs AND income NDFL — a single honest
    input the cert sweeps for sensitivity, never a measured number.

    Each step multiplies the running total-return level by the price return
    ``price[i]/price[i-1]`` AND a rental factor ``1 + y*days/365`` where ``y =
    net_annual_pct/100``. The first point is the unchanged base (a 0-day step accrues no
    rent), so a zero yield returns the price curve verbatim and a flat price over exactly
    365 days at ``y`` returns ``1+y``. Rent only ever *adds* value, so the rented curve
    dominates the price curve pointwise for a positive yield.
    """
    if not curve:
        return []
    y = net_annual_pct / _HUNDRED
    out: list[tuple[date, Decimal]] = [curve[0]]
    running = curve[0][1]
    for i in range(1, len(curve)):
        d0, p0 = curve[i - 1]
        d1, p1 = curve[i]
        price_ret = (p1 / p0) if p0 > 0 else _ONE
        days = Decimal((d1 - d0).days)
        rent_factor = _ONE + y * days / _DAYS_PER_YEAR
        running = running * price_ret * rent_factor
        out.append((d1, running))
    return out


def bars_per_year(dates: list[date]) -> float:
    """Sampling frequency of a date series in bars/calendar-year (the smoothing gauge).

    A daily-traded asset is ~252 trading (~365 calendar) bars/yr; a weekly appraisal index
    like MREDC is ~52. The cert flags a candidate whose frequency is far below a daily
    instrument's as SMOOTHED — its measured volatility/drawdown are then structurally
    understated and must not be read as a real (tradeable) low-risk profile.

    Returns ``0.0`` for a degenerate series (fewer than two dates or a zero-length span).
    """
    if len(dates) < _MIN_POINTS:
        return 0.0
    span_days = (dates[-1] - dates[0]).days
    if span_days <= 0:
        return 0.0
    return len(dates) / (span_days / 365.25)
