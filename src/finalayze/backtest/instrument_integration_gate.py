"""Instrument Integration Gate — the standard "should this instrument join the SAA?" measurement.

The reusable L5 measurement layer for the autonomous budget-diversification program
(docs/research/instrument_integration_program.md). Given ANY candidate net-TR sleeve plus its
metadata (risk_tier, intended_role), it measures the candidate's MARGINAL CONTRIBUTION to the
crash-inclusive deposit-anchored core (deposit 40% + equity 60% over the candidate's full window,
which for crash-capable instruments includes the 2022 tail) and emits a PRE-REGISTERED 3-tier
verdict: INTEGRATE / PROBATION / REJECT (+ INSUFFICIENT_DATA).

Design choices (honest-by-construction):

- **Marginal deltas are basis-robust.** Every binding metric is ``aug - base`` (the book WITH the
  candidate minus the book WITHOUT), both run through the SAME net-of-cost/NDFL blender, so the
  fixed-15% RUONIA basis (apt only for the high-rate era) cancels — sidestepping the metric trap
  the gold cert documented.
- **Crash-inclusive.** A diversification gate that cannot see a crash is hollow, so the base book
  spans the candidate's full window (2022+ for gold-class instruments). The 2022 tail is the
  discriminator: an instrument whose tail is IN-window and which RAISES the crash-year drawdown is
  REJECTED (tested & failed); one whose tail POSTDATES the crash is capped at PROBATION (a sound
  hedge whose payoff cannot be proven — a small forward-looking toe-hold).
- **Two evaluation weights, no circularity.** Marginal deltas for INTEGRATE/REJECT are measured at
  a meaningful 10% weight; PROBATION viability at the 3% toe-hold it would actually receive.
- **Pre-registered thresholds, never moved to fit a candidate** (the anti-overfit pin lives in
  :func:`classify`, a pure function).

Reuses the reviewed ``gold_sleeve_lab`` blender + ``_metrics``/``regime_split``/
``accrue_real_risk_free_leg``. It authorizes a CONFIG weight, NEVER an order — real money is a
hard stop. See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from finalayze.backtest.allocation_gate import accrue_real_risk_free_leg, regime_split
from finalayze.backtest.equity_tilt_experiment import _metrics, _slice
from finalayze.backtest.equity_tilt_lab import quarter_end_dates
from finalayze.backtest.gold_sleeve_lab import blend_portfolio, forward_align_legs, master_axis
from finalayze.core.ndfl import YtdTaxAccumulator

# ── Pre-registered constants (named; NEVER moved to fit a candidate) ──────────
_MIN_BARS = 300
_W_EVAL = Decimal("0.10")  # meaningful-allocation eval weight for INTEGRATE/REJECT
_W_TOEHOLD = Decimal("0.03")  # the PROBATION toe-hold weight
_BASE_DEPOSIT_W = Decimal("0.4")
_BASE_EQUITY_W = Decimal("0.6")
_DEPOSIT_SPREAD_PP = Decimal("1.0")
_INTEGRATE_SHARPE_MIN = 0.10
_INTEGRATE_SORTINO_MIN = 0.0
_MAXDD_CUT_MIN_PP = 3.0  # mirrors gold_sleeve_lab._MAXDD_CUT_MIN_PP
_REJECT_SORTINO_LINE = -0.10
_PROBATION_SORTINO_FLOOR = -0.10
_PROBATION_CORR_EQUITY = 0.35
_PROBATION_CORR_DEPOSIT = 0.20
_INTEGRATE_CORR_CEIL = 0.60
_PROBATION_MAXDD_MIN_PP = 1.0
_TIER_NOMINAL_CAPS: dict[str, Decimal] = {
    "low": Decimal("0.10"),
    "medium": Decimal("0.08"),
    "high": Decimal("0.04"),
}
_PROBATION_NOMINAL_CAP = Decimal("0.03")
# An instrument whose first bar is on/before this date covers the 2022 MOEX crash (tail-testable).
_CRASH_REFERENCE_DATE = date(2022, 3, 1)
_CRASH_YEAR_START = date(2022, 1, 1)
_CRASH_YEAR_END = date(2022, 12, 31)
_BINDING_END = date(2026, 6, 10)  # look-ahead clamp (mirrors allocation_gate._BINDING_END)
_MIN_CORR_PAIRS = 2
_DEPOSIT = "deposit"
_EQUITY = "equity"
_CAND = "candidate"


@dataclass(frozen=True)
class Candidate:
    """A candidate instrument: a NET total-return curve + integration metadata."""

    name: str
    net_curve: list[tuple[date, Decimal]]
    risk_tier: str  # low | medium | high
    intended_role: str  # cash | carry | hedge | diversifier | growth


@dataclass(frozen=True)
class Scorecard:
    """The standardized marginal-contribution scorecard (all deltas are aug - base)."""

    window_bars: int
    regimes_covered: int
    tail_backtestable: bool
    marginal_sharpe_delta: float
    marginal_sortino_delta: float
    marginal_maxdd_delta_pp: float  # positive = candidate CUT book MaxDD
    crash_year_maxdd_delta_pp: float  # positive = candidate RAISED the crash-year drawdown
    toehold_sortino_delta: float  # at the 3% toe-hold
    corr_to_legs: dict[str, float]  # signed Pearson corr of daily returns
    max_corr_to_existing_legs: float  # max |corr|
    anti_hollow_ok: bool


@dataclass(frozen=True)
class IntegrationVerdict:
    """The gate's decision for one candidate."""

    name: str
    tier: str  # INTEGRATE | PROBATION | REJECT | INSUFFICIENT_DATA
    proposed_weight: Decimal
    carved_from: str
    scorecard: Scorecard
    n1_caveat: bool
    reasons: list[str]


def daily_returns(curve: list[tuple[date, Decimal]]) -> list[float]:
    """Daily simple returns of a ``(date, Decimal)`` curve (prior value must be positive)."""
    vals = [float(v) for _, v in curve]
    return [vals[i] / vals[i - 1] - 1.0 for i in range(1, len(vals)) if vals[i - 1] > 0]


def leg_correlations(
    cand_curve: list[tuple[date, Decimal]],
    base_legs: dict[str, list[tuple[date, Decimal]]],
) -> dict[str, float]:
    """Signed Pearson corr of the candidate's daily returns vs each leg's, on common dates."""
    cm = {d: float(v) for d, v in cand_curve if v > 0}
    out: dict[str, float] = {}
    for name, leg in base_legs.items():
        lm = {d: float(v) for d, v in leg if v > 0}
        common = sorted(set(cm) & set(lm))
        cr: list[float] = []
        lr: list[float] = []
        for i in range(1, len(common)):
            d0, d1 = common[i - 1], common[i]
            cr.append(cm[d1] / cm[d0] - 1.0)
            lr.append(lm[d1] / lm[d0] - 1.0)
        if len(cr) >= _MIN_CORR_PAIRS and _has_spread(cr) and _has_spread(lr):
            out[name] = statistics.correlation(cr, lr)
        else:
            out[name] = 0.0
    return out


def _has_spread(xs: list[float]) -> bool:
    return len(xs) >= _MIN_CORR_PAIRS and statistics.pstdev(xs) > 0


def _blend(
    legs: dict[str, list[Decimal]],
    axis: list[date],
    weights: dict[str, Decimal],
    rebal: list[date],
) -> list[tuple[date, Decimal]]:
    return blend_portfolio(
        legs={k: legs[k] for k in weights},
        dates=axis,
        target_weights=weights,
        rebalance_dates=rebal,
        free_legs={_DEPOSIT},
    )


def _maxdd_pp(nav: list[tuple[date, Decimal]], start: date, end: date) -> float:
    return _metrics(_slice([d for d, _ in nav], [v for _, v in nav], start, end)).maxdd_pct


def compute_scorecard(
    *,
    axis: list[date],
    deposit_curve: list[tuple[date, Decimal]],
    equity_curve: list[tuple[date, Decimal]],
    candidate: Candidate,
) -> Scorecard:
    """Measure the candidate's marginal contribution to the deposit+equity core (on ``axis``)."""
    carve_from = _DEPOSIT if candidate.intended_role == "cash" else _EQUITY
    cand_aligned = forward_align_legs({_CAND: candidate.net_curve}, axis)[_CAND]
    legs = {
        _DEPOSIT: [v for _, v in deposit_curve],
        _EQUITY: [v for _, v in equity_curve],
        _CAND: cand_aligned,
    }
    rebal = sorted({axis[0], *quarter_end_dates(axis)})

    base = _blend(legs, axis, {_DEPOSIT: _BASE_DEPOSIT_W, _EQUITY: _BASE_EQUITY_W}, rebal)

    def _aug(weight: Decimal) -> list[tuple[date, Decimal]]:
        w = {_DEPOSIT: _BASE_DEPOSIT_W, _EQUITY: _BASE_EQUITY_W}
        w[carve_from] = w[carve_from] - weight
        w[_CAND] = weight
        return _blend(legs, axis, w, rebal)

    aug = _aug(_W_EVAL)
    aug_toe = _aug(_W_TOEHOLD)

    # Anti-hollow: the verdict must come from a candidate that ACTUALLY moves the book. A 10%
    # allocation that leaves the NAV curve byte-identical to the base is a degenerate/hollow input
    # (a flat or no-op curve) and cannot be judged -> INSUFFICIENT_DATA via this flag.
    anti_hollow_ok = legs_to_vals(aug) != legs_to_vals(base)

    start, end = axis[0], axis[-1]
    base_m = _metrics(_slice(axis, legs_to_vals(base), start, end))
    aug_m = _metrics(_slice(axis, legs_to_vals(aug), start, end))
    aug_toe_m = _metrics(_slice(axis, legs_to_vals(aug_toe), start, end))

    tail_backtestable = candidate.net_curve[0][0] <= _CRASH_REFERENCE_DATE
    crash_delta = 0.0
    if tail_backtestable:
        base_crash = _maxdd_pp(base, _CRASH_YEAR_START, _CRASH_YEAR_END)
        aug_crash = _maxdd_pp(aug, _CRASH_YEAR_START, _CRASH_YEAR_END)
        crash_delta = aug_crash - base_crash  # positive = raised the crash drawdown

    corr = leg_correlations(candidate.net_curve, {_DEPOSIT: deposit_curve, _EQUITY: equity_curve})
    max_corr = max((abs(c) for c in corr.values()), default=0.0)

    return Scorecard(
        window_bars=len(axis),
        regimes_covered=len(regime_split(axis)),
        tail_backtestable=tail_backtestable,
        marginal_sharpe_delta=aug_m.sharpe - base_m.sharpe,
        marginal_sortino_delta=aug_m.sortino - base_m.sortino,
        marginal_maxdd_delta_pp=base_m.maxdd_pct - aug_m.maxdd_pct,
        crash_year_maxdd_delta_pp=crash_delta,
        toehold_sortino_delta=aug_toe_m.sortino - base_m.sortino,
        corr_to_legs=corr,
        max_corr_to_existing_legs=max_corr,
        anti_hollow_ok=anti_hollow_ok,
    )


def legs_to_vals(nav: list[tuple[date, Decimal]]) -> list[Decimal]:
    """The value column of a ``(date, Decimal)`` NAV curve (small named helper for _slice)."""
    return [v for _, v in nav]


def classify(sc: Scorecard) -> tuple[str, list[str]]:  # noqa: PLR0911 — flat tier ladder is clearest
    """The PRE-REGISTERED 3-tier verdict (pure; the anti-overfit pin). Returns (tier, reasons)."""
    if sc.window_bars < _MIN_BARS or not sc.anti_hollow_ok:
        return "INSUFFICIENT_DATA", [
            f"cannot judge: window_bars={sc.window_bars} (<{_MIN_BARS}) or anti-hollow failed"
        ]

    if (
        sc.tail_backtestable
        and sc.regimes_covered >= 2  # noqa: PLR2004 — both regimes present
        and sc.marginal_sharpe_delta >= _INTEGRATE_SHARPE_MIN
        and sc.marginal_sortino_delta >= _INTEGRATE_SORTINO_MIN
        and sc.marginal_maxdd_delta_pp >= _MAXDD_CUT_MIN_PP
        and sc.crash_year_maxdd_delta_pp <= 0.0
        and sc.max_corr_to_existing_legs <= _INTEGRATE_CORR_CEIL
    ):
        return "INTEGRATE", ["free improvement: cuts MaxDD>=3pp, Sortino not worsened, tail-tested"]

    # REJECT hard vetoes (checked before PROBATION so a tested-and-failed hedge cannot sneak in).
    if sc.max_corr_to_existing_legs > _INTEGRATE_CORR_CEIL:
        return "REJECT", [
            f"redundant factor: max |corr| {sc.max_corr_to_existing_legs:.2f} "
            f"> {_INTEGRATE_CORR_CEIL}"
        ]
    if sc.tail_backtestable and sc.crash_year_maxdd_delta_pp > 0.0:
        return "REJECT", [
            f"tail tested & FAILED: raised crash-year MaxDD by {sc.crash_year_maxdd_delta_pp:.2f}pp"
        ]
    if sc.tail_backtestable and sc.marginal_sortino_delta < _REJECT_SORTINO_LINE:
        return "REJECT", [
            f"tail tested & materially worsens risk-adjusted return "
            f"(ΔSortino {sc.marginal_sortino_delta:.2f})"
        ]

    # PROBATION: a structurally-sound, uncorrelated hedge whose tail cannot be proven.
    if (
        abs(sc.corr_to_legs.get(_EQUITY, 1.0)) <= _PROBATION_CORR_EQUITY
        and abs(sc.corr_to_legs.get(_DEPOSIT, 1.0)) <= _PROBATION_CORR_DEPOSIT
        and sc.marginal_maxdd_delta_pp >= _PROBATION_MAXDD_MIN_PP
        and sc.toehold_sortino_delta >= _PROBATION_SORTINO_FLOOR
        and (not sc.tail_backtestable or sc.regimes_covered < 2)  # noqa: PLR2004
    ):
        return "PROBATION", [
            "structurally-sound uncorrelated hedge; tail un-backtestable -> 3% forward toe-hold"
        ]

    return "REJECT", ["no measurable net benefit over the deposit+equity core"]


def propose_weight(tier: str, risk_tier: str, _scorecard: Scorecard) -> Decimal:
    """The proposed CONFIG weight (carved from the role leg) — capped by tier; never a solver."""
    if tier in ("REJECT", "INSUFFICIENT_DATA"):
        return Decimal(0)
    if tier == "PROBATION":
        return _PROBATION_NOMINAL_CAP
    return _TIER_NOMINAL_CAPS.get(risk_tier, _TIER_NOMINAL_CAPS["high"])


def run_integration_gate(
    candidate: Candidate,
    equity_curve: list[tuple[date, Decimal]],
) -> IntegrationVerdict:
    """Public entry: build the deposit+equity core over the candidate's window, score, classify.

    The deposit leg is accrued from the REAL CBR archive on the master axis (union of the candidate
    and equity dates, look-ahead-clamped). ``equity_curve`` is MCFTRR net (already net). The verdict
    is computed entirely from the real blended curves — no hook, no pre-baked literal.
    """
    start = max(candidate.net_curve[0][0], equity_curve[0][0])
    axis = [
        d
        for d in master_axis({_CAND: candidate.net_curve, _EQUITY: equity_curve})
        if start <= d <= _BINDING_END
    ]
    deposit_curve = accrue_real_risk_free_leg(
        axis, Decimal(1), spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=YtdTaxAccumulator()
    )
    equity_vals = forward_align_legs({_EQUITY: equity_curve}, axis)[_EQUITY]
    equity_aligned = list(zip(axis, equity_vals, strict=True))
    sc = compute_scorecard(
        axis=axis, deposit_curve=deposit_curve, equity_curve=equity_aligned, candidate=candidate
    )
    tier, reasons = classify(sc)
    weight = propose_weight(tier, candidate.risk_tier, sc)
    carved_from = _DEPOSIT if candidate.intended_role == "cash" else _EQUITY
    n1_caveat = (not sc.tail_backtestable) or sc.regimes_covered < 2  # noqa: PLR2004
    return IntegrationVerdict(
        name=candidate.name,
        tier=tier,
        proposed_weight=weight,
        carved_from=carved_from,
        scorecard=sc,
        n1_caveat=n1_caveat,
        reasons=reasons,
    )
