"""PEAD sleeve lab -- pure NAV construction for the post-earnings-drift deposit gate.

Post-Earnings-Announcement-Drift (PEAD): stocks are said to drift in the direction of an
earnings surprise for weeks after the report. The operator asked to test that edge through
the deposit gate. MOEX has NO consensus EPS feed and no eps_ttm history (D-01), so this
tests the PRICE-REACTION variant (Chan-Jegadeesh-Lakonishok): the surprise is proxied by
the announcement-window abnormal return, and the drift is the tradeable part measured after
a realistic post-announcement entry.

This module holds the PURE, tested NAV primitives. A PEAD LONG sleeve parks idle capital in
the deposit and goes long a positive-surprise name over its drift window, netting the
round-trip retail cost + NDFL-on-gain per window. The resulting NET total-return curve is
fed to the ``instrument_integration_gate`` as a Candidate (the formal deposit gate) and
compared to holding 100% deposit.

NO NETWORK, NO I/O, NO real money. The surprise/drift arithmetic reuses
:mod:`finalayze.backtest.event_study_lab`. See docs/architecture/DEPENDENCY_LAYERS.md.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

_ONE = Decimal(1)
_ZERO = Decimal(0)


def net_window_factor(
    entry: Decimal, exit_: Decimal, per_side_cost: Decimal, ndfl: Decimal
) -> Decimal:
    """Net total growth factor of ONE long drift window (round-trip cost + NDFL on a gain).

    ``(exit_ / entry) * (1 - c) ** 2`` is the gross-of-tax factor after a per-side cost on
    both legs; a positive gain is then taxed at ``ndfl`` (a loss is never given a
    negative-tax windfall). A non-positive entry yields the identity factor ``1``.
    """
    if entry <= _ZERO:
        return _ONE
    gross = (exit_ / entry) * (_ONE - per_side_cost) * (_ONE - per_side_cost)
    gain = gross - _ONE
    if gain > _ZERO:
        gross = _ONE + gain * (_ONE - ndfl)
    return gross


def spread_factor(total_factor: Decimal, n_bars: int) -> Decimal:
    """Equal geometric per-bar factor whose product over ``n_bars`` equals ``total_factor``.

    Distributes a window's total net factor smoothly across its active bars so concurrently
    active windows can be averaged per bar. ``n_bars <= 0`` or a non-positive total yields
    the identity ``1`` (a window with no active bars contributes nothing).
    """
    if n_bars <= 0 or total_factor <= _ZERO:
        return _ONE
    return total_factor ** (Decimal(1) / Decimal(n_bars))


def blend_pead_nav(
    axis: list[date],
    deposit_daily: dict[date, Decimal],
    active_per_bar: dict[date, list[Decimal]],
    initial_nav: Decimal = _ONE,
) -> list[tuple[date, Decimal]]:
    """Daily NAV: equal-weight the active drift windows on active bars, else earn the deposit.

    ``deposit_daily`` maps each axis bar to that day's deposit growth factor (the idle base --
    a retail investor parks uninvested cash in the deposit). ``active_per_bar`` maps a bar to
    the per-bar net factors of every drift window active that day; on such a bar the NAV grows
    by their EQUAL-WEIGHT average (an equal-weight basket of the currently-drifting names). A
    bar with no active window earns the deposit factor (default ``1`` if missing).

    With ``active_per_bar`` empty the NAV reproduces the compounded deposit exactly -- the
    control: a PEAD sleeve that never trades IS the deposit.
    """
    nav = initial_nav
    out: list[tuple[date, Decimal]] = [(axis[0], nav)]
    for d in axis[1:]:
        facs = active_per_bar.get(d)
        r = (sum(facs, _ZERO) / Decimal(len(facs))) if facs else deposit_daily.get(d, _ONE)
        nav = nav * r
        out.append((d, nav))
    return out


def realpath_window(
    active_dates: list[date], name_daily: dict[date, Decimal], target_total: Decimal
) -> dict[date, Decimal]:
    """Per-bar factors that ride the name's REAL daily returns but net to ``target_total``.

    ``spread_factor`` flattens a window to one constant per-bar factor, which erases the
    name's true daily volatility -- a sleeve built that way hands the integration gate a
    fictitiously smooth curve (fake low vol / near-zero correlation, the same smoothing
    artifact real estate's weekly index had). Instead ride each active bar's REAL daily
    factor ``name_daily[d]`` and absorb the round-trip cost + NDFL + entry-open detail into
    the LAST bar so the window's PRODUCT equals ``target_total`` (from
    :func:`net_window_factor`) while every interior bar keeps its real move. A non-positive
    raw product or empty window returns the raw factors unscaled.
    """
    if not active_dates:
        return {}
    out = {d: name_daily.get(d, _ONE) for d in active_dates}
    product = _ONE
    for d in active_dates:
        product = product * out[d]
    if product <= _ZERO:
        return out
    last = active_dates[-1]
    out[last] = out[last] * (target_total / product)
    return out


def daily_factors(level_curve: list[tuple[date, Decimal]]) -> dict[date, Decimal]:
    """Map each bar (after the first) to its gross daily factor ``level[i] / level[i-1]``.

    A non-positive prior level yields a flat ``1`` for that bar (defensive, never divide by
    zero). Used to turn the deposit LEVEL curve into the idle per-bar growth the sleeve rides.
    """
    out: dict[date, Decimal] = {}
    for i in range(1, len(level_curve)):
        prev = level_curve[i - 1][1]
        cur = level_curve[i][1]
        out[level_curve[i][0]] = (cur / prev) if prev > _ZERO else _ONE
    return out
