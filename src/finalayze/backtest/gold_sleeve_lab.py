"""Diagnostic gold-sleeve allocation simulator (beyond-MOEX-edge R&D, Phase A).

NOT production trading code. A transparent, deterministic multi-sleeve allocation
blender that answers ONE honest question opened by the strategic-direction pivot
(docs/research/strategic_direction_review.md) AFTER active equity SELECTION was
proven to have no net-of-retail edge (0/113 + three HARD_FAIL tilt experiments):

  does adding a small GOLD sleeve (GLDRUB spot — an asset priced largely in USD,
  so an intrinsic ruble-devaluation hedge that the all-ruble deposit/OFZ/equity
  stack structurally lacks) REDUCE tail drawdown in the 2022 ruble/geopolitical
  crash, WITHOUT materially hurting risk-adjusted return in the calm 16-21%
  high-rate regime where gold (zero yield) is pure drag?

This is a DIVERSIFICATION test, NEVER an alpha claim. Pre-registered honesty
(mirrors the allocation-gate discipline):

- the FULL-window and HIGH-RATE regime are EXPECTED to HARD_FAIL the deposit bar
  (gold pays no coupon; against a ~18% deposit it is a drag) — that is the correct,
  expected result and part of the deliverable;
- the REAL deliverable is the 2022-crash sub-window: gold PASSES as a diversifier
  ONLY if it cuts portfolio MaxDD by a material absolute margin AND improves the
  crash-window excess-Sortino, AFTER the ETF-TER + spot-spread cost haircut. Any
  positive is labelled DIVERSIFICATION / tail-MaxDD, with the N=1 single-crash
  caveat — never alpha.

Modeling choices (transparent, auditable):

- Each sleeve is a NET total-return curve on a shared, forward-filled date axis:
  deposit via :func:`allocation_gate.accrue_real_risk_free_leg` (CBR key - 1pp,
  net-NDFL); equity = MCFTRR (already net); gold = GLDRUB_TOM spot netted via
  :func:`allocation_gate.net_index_returns` (NDFL on the daily positive mark —
  CONSERVATIVE: it taxes unrealized gains, over-stating gold's tax drag) PLUS a
  continuous ETF-TER haircut (:func:`apply_ter_drag`, the wrapper holding cost).
- The OFZ floater (RUFLBITR) is DROPPED here: it has no data before 2023 so it
  cannot cover the 2022-crash deliverable, and its role was already settled in
  Phase 76. Its rate-anchor role folds into the cost-free deposit leg.
- A fixed-weight, quarterly-rebalanced blend (:func:`blend_portfolio`): between
  boundaries each sleeve drifts with its own return; at each boundary the weights
  reset to target, charging a per-side cost on the traded turnover of the non-free
  (equity, gold) legs — the deposit leg is cost-free (allocation_gate D-09).

During the 27-day 2022 equity halt the MCFTRR leg has no bars; the forward-fill
holds its last mark flat while gold (which kept trading on the currency market)
and the deposit move, then equity gaps down on reopen — the realistic "could not
sell equity; gold cushioned" path.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import bisect
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

_TRADING_DAYS_PER_YEAR = 252
_ONE = Decimal(1)
_ZERO = Decimal(0)

# Pre-registered cost stack (NOT fitted). The gold ETF wrapper's annual holding cost
# (TER) — a continuous drag applied to the gold leg before blending. ~0.8%/yr is the
# mid of RU gold ETFs (TGLD/SBGD/AKGD ~0.5-1.2%).
_GOLD_TER_ANNUAL_PCT = Decimal("0.8")
# Per-side rebalance cost (retail "Инвестор" tariff): 0.30% commission + 0.15% half-spread
# + 0.10% slippage = 0.55%/side, mirroring backtest.costs.MOEX_RETAIL_COSTS.
_RETAIL_PER_SIDE_COST = Decimal("0.0055")
# Pre-registered diversification bar: gold must cut portfolio MaxDD by at least this many
# absolute percentage points in the crash window (AND not worsen excess-Sortino).
_MAXDD_CUT_MIN_PP = Decimal("3.0")


def apply_ter_drag(
    curve: list[tuple[date, Decimal]],
    annual_ter_pct: Decimal = _GOLD_TER_ANNUAL_PCT,
) -> list[tuple[date, Decimal]]:
    """Apply a continuous annual holding cost (ETF TER) to a net TR curve.

    Each daily step's gross return is multiplied by the daily TER factor
    ``(1 - ter/100) ** (1/252)``. The drag compounds PER AXIS-BAR (the blend's union
    trading calendar), so a leg held over ``N`` bars loses ~``ter% * N/252``; on a
    currency-market axis with more bars than calendar trading days the realized
    annualized drag is marginally heavier than a strict calendar-year ``ter%`` (a
    transparent continuous-per-bar convention, not a calendar day-count). With
    ``annual_ter_pct == 0`` the curve is returned UNCHANGED (the identity control:
    a free wrapper must not move the leg). The curve opens at its base value.
    """
    if not curve:
        return []
    if annual_ter_pct == _ZERO:
        return list(curve)
    annual_factor = 1.0 - float(annual_ter_pct) / 100.0
    daily_factor = Decimal(str(annual_factor ** (1.0 / _TRADING_DAYS_PER_YEAR)))
    out: list[tuple[date, Decimal]] = [curve[0]]
    value = curve[0][1]
    prev = curve[0][1]
    for d, level in curve[1:]:
        if prev > _ZERO:
            gross_ret = level / prev
            value = value * gross_ret * daily_factor
        prev = level
        out.append((d, value))
    return out


def master_axis(legs: dict[str, list[tuple[date, Decimal]]]) -> list[date]:
    """The sorted union of every leg's trading dates — the blend's common calendar.

    The currency-market gold leg trades on days the equity index does not (e.g.
    through the 2022 halt); the union axis keeps those days so the hedge is visible,
    with each leg forward-filled by :func:`forward_align_legs`.
    """
    dates: set[date] = set()
    for series in legs.values():
        dates.update(d for d, _ in series)
    return sorted(dates)


def forward_align_legs(
    legs: dict[str, list[tuple[date, Decimal]]],
    axis: list[date],
) -> dict[str, list[Decimal]]:
    """Forward-fill each leg onto ``axis`` (last known value on/before each date).

    A date before a leg's first bar takes that first bar's value (no look-behind
    fabrication beyond the opening level). A gap inside the series holds the prior
    value — the realistic "frozen mark" during the equity halt.
    """
    aligned: dict[str, list[Decimal]] = {}
    for name, series in legs.items():
        ordered = sorted(series, key=lambda p: p[0])
        leg_dates = [d for d, _ in ordered]
        leg_vals = [v for _, v in ordered]
        out: list[Decimal] = []
        for d in axis:
            idx = bisect.bisect_right(leg_dates, d) - 1
            out.append(leg_vals[idx] if idx >= 0 else leg_vals[0])
        aligned[name] = out
    return aligned


def blend_portfolio(
    *,
    legs: dict[str, list[Decimal]],
    dates: list[date],
    target_weights: dict[str, Decimal],
    rebalance_dates: list[date],
    per_side_cost: Decimal = _RETAIL_PER_SIDE_COST,
    free_legs: set[str],
    initial_nav: Decimal = _ONE,
    weight_schedule: dict[date, dict[str, Decimal]] | None = None,
) -> list[tuple[date, Decimal]]:
    """Simulate a fixed-weight, quarterly-rebalanced multi-sleeve portfolio NAV.

    ``legs`` are NET value series already aligned to ``dates`` (same length). Per day,
    in order: each holding drifts by its leg's daily return (``value[i]/value[i-1]``);
    on a ``rebalance_dates`` day every leg resets to ``nav * target_weight`` and a
    per-side cost is charged on the traded turnover ``|target - drifted|`` of each
    leg NOT in ``free_legs`` (the deposit leg is cost-free). The cost is deducted from
    NAV and the post-cost NAV re-scaled across legs. The opening allocation on
    ``dates[0]`` is cost-free (the initial buy-in is not part of the rebalance drag).

    ``weight_schedule`` (optional) maps a rebalance date to the target-weight vector to
    use ON THAT date — the seam for a CONDITIONAL / regime-dependent overlay (e.g. hold a
    hedge leg only when a trailing stress flag is on). A date absent from the schedule (or
    ``weight_schedule=None``) falls back to the static ``target_weights``; a leg absent from
    a schedule vector is targeted at 0. ``names`` (the tradeable leg set) is always
    ``target_weights`` keys, so a conditionally-held leg must appear there (at weight 0).

    A single 100%-weight leg with no other leg reproduces that leg's net return path
    to ~26 significant digits (turnover 0 -> cost is exactly 0; the only deviation is
    sub-1e-26 Decimal-context rounding from the reset round-trip): the data-correctness
    control. The cert's ``zero_ok`` check compares two blends on the IDENTICAL code path,
    so that rounding cancels and the 0%-gold-vs-baseline equality is exact.
    """
    names = list(target_weights)
    rebal = set(rebalance_dates)

    def _targets(d: date) -> dict[str, Decimal]:
        if weight_schedule is not None and d in weight_schedule:
            return weight_schedule[d]
        return target_weights

    init_tw = _targets(dates[0])
    holdings = {n: initial_nav * init_tw.get(n, _ZERO) for n in names}
    out: list[tuple[date, Decimal]] = [(dates[0], initial_nav)]

    for i in range(1, len(dates)):
        for n in names:
            prev = legs[n][i - 1]
            cur = legs[n][i]
            if prev > _ZERO:
                holdings[n] = holdings[n] * (cur / prev)
        nav = sum(holdings.values(), _ZERO)

        if dates[i] in rebal and nav > _ZERO:
            tw = _targets(dates[i])
            total_cost = _ZERO
            for n in names:
                target = nav * tw.get(n, _ZERO)
                if n not in free_legs:
                    total_cost += abs(target - holdings[n]) * per_side_cost
                holdings[n] = target
            nav_after = nav - total_cost
            scale = nav_after / nav
            for n in names:
                holdings[n] = holdings[n] * scale
            nav = nav_after

        out.append((dates[i], nav))
    return out


def slice_curve(
    curve: list[tuple[date, Decimal]], start: date, end: date
) -> list[tuple[date, Decimal]]:
    """Inclusive date slice of a ``(date, Decimal)`` curve (a pure filter, no re-net)."""
    return [(d, v) for d, v in curve if start <= d <= end]


def diversification_verdict(
    *,
    baseline_maxdd_pct: float,
    gold_maxdd_pct: float,
    baseline_sortino: float,
    gold_sortino: float,
    maxdd_cut_min_pp: Decimal = _MAXDD_CUT_MIN_PP,
) -> dict[str, object]:
    """The pre-registered crash-window diversification verdict (NOT an alpha test).

    Gold ``diversifies`` IFF **both** hold (conjunctive, mirroring the gate's
    Sharpe∧Sortino∧MaxDD discipline):

    1. it cuts portfolio MaxDD by at least ``maxdd_cut_min_pp`` absolute percentage
       points (``baseline_maxdd - gold_maxdd >= bar``), AND
    2. it does not worsen the crash-window excess-Sortino
       (``gold_sortino >= baseline_sortino``).

    A big MaxDD cut bought with a worse risk-adjusted return is NOT a win; neither is
    a Sortino bump with no real drawdown relief. Returns the components for the report.
    """
    maxdd_cut_pp = Decimal(str(baseline_maxdd_pct)) - Decimal(str(gold_maxdd_pct))
    maxdd_ok = maxdd_cut_pp >= maxdd_cut_min_pp
    sortino_ok = gold_sortino >= baseline_sortino
    return {
        "diversifies": bool(maxdd_ok and sortino_ok),
        "maxdd_cut_pp": float(maxdd_cut_pp),
        "maxdd_cut_min_pp": float(maxdd_cut_min_pp),
        "maxdd_ok": maxdd_ok,
        "sortino_ok": sortino_ok,
    }
