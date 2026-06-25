"""Deposit-ladder optimizer (Layer 5) -- RECOMMENDATION ONLY.

Given a budget, a REAL offered deposit term-structure (committed snapshot or operator
input), and a set of CBR key-rate scenarios, this ranks candidate deposit ladders by a
robust expected after-tax terminal value and -- the load-bearing output -- reports whether
any "lock long vs roll short" edge is REAL or whether the offered curve already prices the
expected cuts. It moves no money, constructs no orders, opens no live broker/channel/session,
holds no token, and never calls any ``BrokerBase`` order method. It only drives the SIMULATED
``DepositSimulatedBroker`` for arithmetic (mandatory: NDFL is path/YTD-dependent and not
separable, so the whole ladder must run through one broker = one YTD allowance pool).

Honesty rails (see docs/design/deposit_ladder_optimizer.md):
- Offered rates are REAL snapshot/operator data; the loader FAILS CLOSED when absent and NEVER
  synthesizes a term structure from ``deposit_rate_as_of`` (that would be a fixture, not a
  measurement -- the Phase-87 anti-hollow lesson).
- The lock-in edge is a simulated diff vs an actually-rolled ALL_SHORT baseline, anchored on a
  DERIVED curve-implied breakeven scenario; the default easing-regime outcome
  ``NO_EDGE_CURVE_PRICES_CUTS`` is a first-class success, and a win that exists only because the
  modeled path disagrees with the curve is reported as ``REGIME_BET_NOT_EDGE``, not alpha.
- Every projected number (roll rates, future-year allowance, spread persistence) is labeled
  projected; the rate path is an ASSUMPTION, carried with an N=1 caveat (the committed CBR
  calendar is one realized easing cycle).

This module NEVER imports a live broker, router, channel, token, or session, and never wires
into the frozen ``AllocationOrchestrator`` / ``plan_rebalance`` real-money path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from dateutil.relativedelta import relativedelta

from finalayze.core.constants import (
    ASV_CAP_PER_BANK,
    ASV_RAISED_TIER_2_8M,
    ASV_RAISED_TIER_2M,
    DEPOSIT_DEMAND_RATE,
    MAX_OFFER_STALENESS_DAYS,
    NDFL_PROGRESSIVE_THRESHOLD,
)
from finalayze.core.exceptions import ConfigurationError
from finalayze.core.schemas import DepositTranche
from finalayze.data.fetchers.cbr import deposit_rate_as_of
from finalayze.execution.deposit_broker import (
    DepositSimulatedBroker,
    split_across_banks,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_PCT = Decimal(100)
_BPS = Decimal(10_000)
_DAYS_PER_YEAR = Decimal(365)
_THIRTY_SIX_MONTHS = 36
_SATURDAY = 5  # date.weekday() >= 5 is a weekend
_MIN_FOR_STDEV = 2
_PATH_FRAGILE_PCT = Decimal("0.005")  # >50bps scenario dispersion -> path-dependent outcome
_TERM_STRUCTURE_PATH = Path(__file__).resolve().parent / "data" / "deposit_term_structure.json"


# ============================================================================
# INPUTS
# ============================================================================


@dataclass(frozen=True)
class TermOffer:
    """One point on the REAL offered term structure (snapshot/operator, never synthetic)."""

    term_months: int
    annual_rate: Decimal  # offered fraction, e.g. Decimal("0.19")
    roll_spread_pp: Decimal  # SIGNED per-term spread vs key rate (REQUIRED; loader fails closed)
    instrument_type: str = "deposit"  # "deposit" | "irrevocable_cert" -> ASV tier
    bank_id: str | None = None


@dataclass(frozen=True)
class SpreadBand:
    """Forward-spread perturbation envelope for the short roll leg (the as-of spread is a FACT;
    its forward persistence is an ASSUMPTION, so it is swept, not frozen)."""

    widen_pp: Decimal = Decimal("0.5")  # spread widens -> rolled short earns LESS
    tighten_pp: Decimal = Decimal("-0.5")  # spread tightens -> rolled short earns MORE


@dataclass(frozen=True)
class LadderConstraints:
    allowed_terms: tuple[int, ...] = (3, 6, 12, 36)
    min_liquid_fraction: Decimal = Decimal(0)
    liquidity_horizon_months: int = 6
    grid_step: Decimal = Decimal("0.5")
    max_candidates: int = 48
    robustness_lambda: Decimal = Decimal("0.5")  # 0=mean .. 1=worst-case
    noise_bps: Decimal = Decimal(5)  # de-minimis lock-in edge
    tie_bps: Decimal = Decimal(5)
    uninsured_penalty_lambda: Decimal = Decimal(0)  # 0 -> uninsured is a raw axis, not penalized
    spread_band: SpreadBand = SpreadBand()


class AsvTier(StrEnum):
    """ASV insurance tiers -- VERIFIED against Minfin (raised limits "from 18 Dec 2025"; the
    effective date/legal force should be re-verified). Boundary = strictly >3yr, so a 36mo
    (=3yr) instrument stays STANDARD. ASV here is a SOFT reported metric -- a mis-tiered cap
    only mislabels a non-blocking risk number, never excludes a candidate or moves money.
    """

    STANDARD = "1.4M"  # ordinary deposit, term <= 3yr (36mo inclusive)
    RAISED_2M = "2.0M"  # ordinary deposit strictly >3yr  OR irrevocable cert 1-3yr
    RAISED_2_8M = "2.8M"  # irrevocable cert strictly >3yr


@dataclass(frozen=True)
class RatePathScenario:
    scenario_id: str
    key_rate_at: Callable[[date], Decimal]  # key rate in PERCENTAGE POINTS
    weight: Decimal = Decimal(1)
    is_realized_anchor: bool = False
    is_curve_implied: bool = False


@dataclass(frozen=True)
class TermStructure:
    """The committed, dated, provenanced offered-rate snapshot (fail-closed loaded)."""

    as_of: date
    source: str
    git_sha: str | None
    snapshot_mode: str  # "backtest" (staleness-exempt) | "forward" (staleness-gated)
    horizon_months: int
    offers: tuple[TermOffer, ...]
    raw_scenarios: Mapping[
        str, object
    ]  # parsed lazily into RatePathScenario by make_default_scenarios


@dataclass(frozen=True)
class OptimizerRequest:
    budget: Decimal
    start: date
    horizon_months: int
    term_structure: TermStructure
    constraints: LadderConstraints = LadderConstraints()
    ytd_other_taxable_income: Decimal = Decimal(0)  # seeds the cross-sleeve 2.4M band lower bound


# ============================================================================
# RESULTS
# ============================================================================


@dataclass(frozen=True)
class LadderCandidate:
    candidate_id: str
    archetype: str
    weights: Mapping[int, Decimal]  # term_months -> fraction, sums to 1


@dataclass(frozen=True)
class SimResult:
    candidate_id: str
    scenario_id: str
    terminal_value: Decimal
    liquidation_floor: Decimal
    gross_interest: Decimal
    net_interest: Decimal
    tax_paid: Decimal
    effective_after_tax_yield: Decimal
    roll_count: int
    locked_value_fraction: Decimal
    min_liquidity_fraction: Decimal
    max_uninsured_excess: Decimal
    banks_needed: int
    progressive_band_caveat: bool


class LockinVerdict(StrEnum):
    NO_EDGE_CURVE_PRICES_CUTS = "no_edge_curve_prices_cuts"
    REAL_LOCKIN_EDGE = "real_lockin_edge"
    TAX_SMOOTHING_EDGE = "tax_smoothing_edge"
    LIQUIDITY_COST = "liquidity_cost"
    REGIME_BET_NOT_EDGE = "regime_bet_not_edge"
    NO_ROBUST_EDGE = "no_robust_edge"


@dataclass(frozen=True)
class RegimeLockinReport:
    baseline_id: str
    candidate_id: str
    per_scenario_bps: Mapping[str, Decimal]
    curve_implied_bps: Decimal
    mean_lockin_bps: Decimal
    min_lockin_bps: Decimal
    max_lockin_bps: Decimal
    spread_band_min_bps: Decimal
    spread_band_max_bps: Decimal
    spread_sign_flips: bool
    curve_slope_bps: Decimal
    curve_inverted: bool
    edge_source: str
    verdict: LockinVerdict
    worst_case_scenario_id: str
    honest_message: str
    spread_held_constant: bool = False
    scenario_set_degenerate: bool = False
    n1_caveat: bool = True


@dataclass(frozen=True)
class RankedLadder:
    candidate: LadderCandidate
    mean_eatv: Decimal
    min_eatv: Decimal
    max_eatv: Decimal
    std_eatv: Decimal
    rank_key: Decimal
    path_fragile: bool
    per_scenario: Mapping[str, SimResult]
    banks_needed: int
    uninsured_at_horizon: Decimal


@dataclass(frozen=True)
class LadderPlan:
    """RECOMMENDATION-ONLY result. Holds NO orders, NO broker handle -- only Decimal metrics."""

    budget: Decimal
    start: date
    horizon_months: int
    scenarios_used: tuple[str, ...]
    ranked: tuple[RankedLadder, ...]
    recommended: RankedLadder
    lockin_report: RegimeLockinReport
    snapshot_provenance: str
    spread_held_constant: bool
    n1_caveat: bool
    progressive_band_caveat: bool
    recommendation_caveat: str  # reconciles the recommendation with the lock-in verdict (B1)


# ============================================================================
# LOADER (fail-closed)
# ============================================================================


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def load_term_structure(
    path: Path = _TERM_STRUCTURE_PATH, *, today: date | None = None
) -> TermStructure:
    """Load the committed offered-rate snapshot, FAILING CLOSED on any defect.

    Raises ``ConfigurationError`` on: missing/corrupt file, missing key, non-positive rate,
    an offer missing an explicit signed ``roll_spread_pp`` (NEVER falls back to a flat default),
    and -- for ``snapshot_mode == "forward"`` -- a stale ``as_of`` (older than
    ``MAX_OFFER_STALENESS_DAYS``). Never synthesizes a term structure from ``deposit_rate_as_of``.
    """
    _require(path.exists(), f"deposit term-structure snapshot not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"deposit term-structure snapshot is corrupt JSON: {exc}") from exc

    for key in (
        "as_of",
        "source",
        "snapshot_mode",
        "horizon_months",
        "offers",
        "key_rate_scenarios",
    ):
        _require(key in raw, f"deposit term-structure snapshot missing key: {key}")

    snapshot_mode = str(raw["snapshot_mode"])
    _require(
        snapshot_mode in ("backtest", "forward"),
        f"snapshot_mode must be 'backtest' or 'forward', got {snapshot_mode!r}",
    )
    as_of = date.fromisoformat(str(raw["as_of"]))

    if snapshot_mode == "forward":
        ref = today if today is not None else datetime.now(tz=UTC).date()
        age = (ref - as_of).days
        _require(
            age <= MAX_OFFER_STALENESS_DAYS,
            f"forward snapshot is stale: as_of {as_of} is {age}d old "
            f"(> {MAX_OFFER_STALENESS_DAYS}d) -- refusing to recommend on stale offers",
        )

    offers_raw = raw["offers"]
    _require(
        isinstance(offers_raw, list) and len(offers_raw) > 0, "offers must be a non-empty list"
    )
    offers: list[TermOffer] = []
    for o in offers_raw:
        for k in ("term_months", "annual_rate", "roll_spread_pp"):
            _require(k in o, f"offer missing required field {k!r}: {o}")
        rate = Decimal(str(o["annual_rate"]))
        _require(rate > 0, f"offer annual_rate must be positive: {o}")
        offers.append(
            TermOffer(
                term_months=int(o["term_months"]),
                annual_rate=rate,
                roll_spread_pp=Decimal(str(o["roll_spread_pp"])),
                instrument_type=str(o.get("instrument_type", "deposit")),
                bank_id=o.get("bank_id"),
            )
        )

    return TermStructure(
        as_of=as_of,
        source=str(raw["source"]),
        git_sha=raw.get("git_sha"),
        snapshot_mode=snapshot_mode,
        horizon_months=int(raw["horizon_months"]),
        offers=tuple(offers),
        raw_scenarios=raw["key_rate_scenarios"],
    )


# ============================================================================
# SCENARIOS
# ============================================================================


def _realized_key_rate(d: date) -> Decimal:
    """Committed-CBR-calendar key rate (pp) as-of d -- the real easing path, look-ahead-safe.

    Resolves the static committed meeting calendar (no live HTTP fetch). For dates beyond the
    last DECIDED meeting it holds the last known rate flat (the honest 'what we know' anchor).
    """
    return deposit_rate_as_of(d, spread_pp=Decimal(0)) * _PCT


def _step_key_rate(points: tuple[tuple[date, Decimal], ...]) -> Callable[[date], Decimal]:
    """Build a piecewise-constant key-rate(pp) function from sorted (date, rate-pp) points."""
    ordered = tuple(sorted(points, key=lambda p: p[0]))

    def fn(d: date) -> Decimal:
        rate = ordered[0][1]
        for pt_date, pt_rate in ordered:
            if pt_date <= d:
                rate = pt_rate
            else:
                break
        return rate

    return fn


def _flat_key_rate(rate_pp: Decimal) -> Callable[[date], Decimal]:
    def fn(_d: date) -> Decimal:
        return rate_pp

    return fn


def _parse_snapshot_scenarios(ts: TermStructure) -> list[RatePathScenario]:
    out: list[RatePathScenario] = []
    for sid, val in ts.raw_scenarios.items():
        if isinstance(val, str):
            _require(
                val == "use_committed_cbr_calendar",
                f"scenario {sid!r} sentinel must be 'use_committed_cbr_calendar', got {val!r}",
            )
            out.append(
                RatePathScenario(
                    scenario_id=sid, key_rate_at=_realized_key_rate, is_realized_anchor=True
                )
            )
        elif isinstance(val, list):
            points = tuple((date.fromisoformat(p[0]), Decimal(str(p[1]))) for p in val)
            out.append(RatePathScenario(scenario_id=sid, key_rate_at=_step_key_rate(points)))
        else:  # pragma: no cover - defensive
            raise ConfigurationError(f"scenario {sid!r} has unsupported value type: {type(val)}")
    return out


def make_default_scenarios(req: OptimizerRequest) -> list[RatePathScenario]:
    """REALIZED (committed calendar) + the snapshot's literal scenarios + a DERIVED CURVE_IMPLIED
    breakeven anchor. Enforces exactly one realized anchor and one curve-implied anchor (E4/E5)."""
    scenarios = _parse_snapshot_scenarios(req.term_structure)
    realized = [s for s in scenarios if s.is_realized_anchor]
    _require(
        len(realized) == 1,
        "scenario set must contain exactly one REALIZED anchor "
        "(use_committed_cbr_calendar) -- the real-anchor leg can never be optimized away",
    )
    scenarios.append(derive_curve_implied_scenario(req))
    return scenarios


# ============================================================================
# ASV TIER
# ============================================================================


def asv_tier_cap(offer: TermOffer) -> Decimal:
    """Minfin-verified ASV cap (RUB). Boundary = strictly >3yr; 36mo stays STANDARD."""
    over_3yr = offer.term_months > _THIRTY_SIX_MONTHS
    if offer.instrument_type == "irrevocable_cert":
        return ASV_RAISED_TIER_2_8M if over_3yr else ASV_RAISED_TIER_2M
    # ordinary deposit
    return ASV_RAISED_TIER_2M if over_3yr else ASV_CAP_PER_BANK


# ============================================================================
# CANDIDATES
# ============================================================================


def _norm(weights: dict[int, Decimal]) -> dict[int, Decimal]:
    total = sum(weights.values(), Decimal(0))
    if total == 0:
        return weights
    return {t: w / total for t, w in weights.items() if w > 0}


def generate_candidates(req: OptimizerRequest) -> list[LadderCandidate]:
    """<=8 archetypes + a capped coarse simplex grid over allowed_terms; de-duplicated."""
    terms = tuple(t for t in req.constraints.allowed_terms)
    short, long = min(terms), max(terms)
    archetypes: dict[str, dict[int, Decimal]] = {
        "ALL_SHORT": {short: Decimal(1)},
        "ALL_LONG": {long: Decimal(1)},
        "BARBELL": {short: Decimal("0.5"), long: Decimal("0.5")},
        "EVEN_SPREAD": {t: Decimal(1) / len(terms) for t in terms},
        "FRONT_WEIGHTED": _front_back(terms, front=True),
        "BACK_WEIGHTED": _front_back(terms, front=False),
    }
    candidates: list[LadderCandidate] = []
    seen: set[tuple[tuple[int, str], ...]] = set()

    def add(archetype: str, weights: dict[int, Decimal]) -> None:
        normed = _norm(dict(weights))
        if not normed:
            return
        key = tuple(sorted((t, str(w.quantize(Decimal("0.0001")))) for t, w in normed.items()))
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            LadderCandidate(
                candidate_id=f"{archetype}#{len(candidates)}", archetype=archetype, weights=normed
            )
        )

    for name, w in archetypes.items():
        add(name, w)
    for grid_weights in _simplex_grid(terms, req.constraints.grid_step):
        if len(candidates) >= req.constraints.max_candidates:
            break
        add("GRID", grid_weights)
    return candidates


def _front_back(terms: tuple[int, ...], *, front: bool) -> dict[int, Decimal]:
    ordered = sorted(terms, reverse=not front)
    # linearly decreasing weights over the ordered terms
    n = len(ordered)
    raw = {t: Decimal(n - i) for i, t in enumerate(ordered)}
    return _norm(raw)


def _simplex_grid(terms: tuple[int, ...], step: Decimal) -> list[dict[int, Decimal]]:
    """Coarse weight vectors on `step` over `terms`, summing to 1."""
    steps = int((Decimal(1) / step).to_integral_value())
    out: list[dict[int, Decimal]] = []

    def recurse(idx: int, remaining: int, acc: dict[int, Decimal]) -> None:
        if idx == len(terms) - 1:
            acc2 = dict(acc)
            acc2[terms[idx]] = Decimal(remaining) * step
            out.append(acc2)
            return
        for k in range(remaining + 1):
            acc2 = dict(acc)
            acc2[terms[idx]] = Decimal(k) * step
            recurse(idx + 1, remaining - k, acc2)

    recurse(0, steps, {})
    return out


# ============================================================================
# SIMULATION
# ============================================================================


def _business_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < _SATURDAY:  # Mon-Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def _annualize(ratio: Decimal, horizon_days: int) -> Decimal:
    """(terminal/budget) compounded to an annual fraction. ratio>0, horizon_days>0."""
    if horizon_days <= 0 or ratio <= 0:
        return Decimal(0)
    exponent = _DAYS_PER_YEAR / Decimal(horizon_days)
    return ratio**exponent - Decimal(1)


def _expand_tranches(
    candidate: LadderCandidate, req: OptimizerRequest, offer_by_term: dict[int, TermOffer]
) -> list[DepositTranche]:
    tranches: list[DepositTranche] = []
    for term, weight in candidate.weights.items():
        if weight <= 0:
            continue
        offer = offer_by_term[term]
        principal = req.budget * weight
        tranches.append(
            DepositTranche(
                principal=principal,
                term_months=term,
                annual_rate=offer.annual_rate,  # the LOCKED offered rate for the first term
                open_date=req.start,
                maturity_date=req.start + relativedelta(months=term),
            )
        )
    return tranches


def simulate_candidate(
    candidate: LadderCandidate,
    scenario: RatePathScenario,
    req: OptimizerRequest,
    offer_by_term: dict[int, TermOffer],
    *,
    roll_spread_delta_pp: Decimal = Decimal(0),
) -> SimResult:
    """Drive ONE fresh DepositSimulatedBroker day-by-day to the horizon. At each maturity the
    DRIVER rebuilds the rolled tranche at the SCENARIO rate (no broker edit). Pure arithmetic;
    never calls any order method."""
    tranches = _expand_tranches(candidate, req, offer_by_term)
    broker = DepositSimulatedBroker(initial_cash=Decimal(0), tranches=tranches)
    horizon_end = req.start + relativedelta(months=req.horizon_months)
    horizon_days = (horizon_end - req.start).days
    rolled_indices: set[int] = set()  # positions that ever rolled (lockedness, no id() hazard, W4)
    roll_count = 0

    for d in _business_days(req.start, horizon_end):
        # Roll matured tranches BEFORE accruing (B3): the original tranches accrue on req.start, so
        # a rolled tranche must likewise accrue on its OPEN day -- otherwise every roll silently
        # skips one accrual day, biasing the lock-in diff toward the never-rolling long lock.
        # Rolling first keeps coverage continuous: the old tranche accrued through the prior
        # business day, the fresh one accrues from d (each calendar day is accrued exactly once).
        if d < horizon_end:
            for i, tr in enumerate(tranches):
                if tr.broken or tr.maturity_date > d:
                    continue
                offer = offer_by_term[tr.term_months]
                rolled_rate = (
                    scenario.key_rate_at(d) + offer.roll_spread_pp + roll_spread_delta_pp
                ) / _PCT
                tranches[i] = DepositTranche(
                    principal=tr.principal + tr.accrued_net,
                    term_months=tr.term_months,
                    annual_rate=max(Decimal(0), rolled_rate),
                    open_date=d,
                    maturity_date=d + relativedelta(months=tr.term_months),
                )
                rolled_indices.add(i)
                roll_count += 1
        broker.accrue(d)

    terminal = broker.deposit_value()
    # locked value = value from positions that NEVER rolled (tracked by index, not id()).
    locked = sum(
        (tr.principal + tr.accrued_net for i, tr in enumerate(tranches) if i not in rolled_indices),
        Decimal(0),
    )
    locked_fraction = (locked / terminal) if terminal > 0 else Decimal(0)
    # liquidation floor: still-locked tranches broken to demand rate; matured ones fully liquid
    floor = sum(
        (
            (t.principal + t.accrued_net)
            if t.maturity_date <= horizon_end
            else t.principal * (Decimal(1) + DEPOSIT_DEMAND_RATE)
            for t in tranches
        ),
        Decimal(0),
    )
    min_liq = sum(
        (
            w
            for term, w in candidate.weights.items()
            if term <= req.constraints.liquidity_horizon_months
        ),
        Decimal(0),
    )
    # ASV: banks needed for full insurance + uninsured if held in ONE bank (operator's behaviour)
    tier_cap = min(asv_tier_cap(offer_by_term[t]) for t in candidate.weights)
    banks = split_across_banks(terminal, tier_cap)
    uninsured_one_bank = max(Decimal(0), terminal - tier_cap)
    yield_ = _annualize(terminal / req.budget, horizon_days) if req.budget > 0 else Decimal(0)
    progressive = (
        req.ytd_other_taxable_income + broker.interest_income_gross
    ) > NDFL_PROGRESSIVE_THRESHOLD

    return SimResult(
        candidate_id=candidate.candidate_id,
        scenario_id=scenario.scenario_id,
        terminal_value=terminal,
        liquidation_floor=floor,
        gross_interest=broker.interest_income_gross,
        net_interest=broker.interest_income_net,
        tax_paid=broker.tax_paid,
        effective_after_tax_yield=yield_,
        roll_count=roll_count,
        locked_value_fraction=locked_fraction,
        min_liquidity_fraction=min_liq,
        max_uninsured_excess=uninsured_one_bank,
        banks_needed=len(banks),
        progressive_band_caveat=progressive,
    )


# ============================================================================
# CURVE-IMPLIED BREAKEVEN (the EMH anchor)
# ============================================================================


def derive_curve_implied_scenario(req: OptimizerRequest) -> RatePathScenario:
    """Bisection-solve the FLAT forward key-rate (pp) under which rolling the shortest term to the
    horizon ties locking the longest term -- i.e. the market's breakeven. This IS the
    curve-prices-cuts anchor: any lock-in 'edge' that vanishes under it is not a structural edge.
    """
    offer_by_term = {o.term_months: o for o in req.term_structure.offers}
    terms = tuple(req.constraints.allowed_terms)
    short, long = min(terms), max(terms)
    all_short = LadderCandidate("CI_SHORT", "ALL_SHORT", {short: Decimal(1)})
    all_long = LadderCandidate("CI_LONG", "ALL_LONG", {long: Decimal(1)})

    def diff(rate_pp: Decimal) -> Decimal:
        scen = RatePathScenario("ci_probe", _flat_key_rate(rate_pp))
        long_t = simulate_candidate(all_long, scen, req, offer_by_term).terminal_value
        short_t = simulate_candidate(all_short, scen, req, offer_by_term).terminal_value
        return long_t - short_t  # decreasing in rate_pp (short rolls richer as rate rises)

    lo, hi = Decimal(0), Decimal(40)
    f_lo, f_hi = diff(lo), diff(hi)
    if f_lo * f_hi > 0:
        # no sign change in [0,40]: the long lock dominates/loses across all flat rates; clamp to
        # the endpoint nearest breakeven (still a valid, honest anchor).
        solved = lo if abs(f_lo) < abs(f_hi) else hi
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            f_mid = diff(mid)
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        solved = (lo + hi) / 2
    return RatePathScenario(
        scenario_id="CURVE_IMPLIED", key_rate_at=_flat_key_rate(solved), is_curve_implied=True
    )


# ============================================================================
# RANKING
# ============================================================================


def _std(values: list[Decimal], mean: Decimal) -> Decimal:
    if len(values) < _MIN_FOR_STDEV:
        return Decimal(0)
    var = sum(((v - mean) ** 2 for v in values), Decimal(0)) / Decimal(len(values))
    return var.sqrt()


def rank_ladders(
    req: OptimizerRequest,
    candidates: list[LadderCandidate],
    scenarios: list[RatePathScenario],
) -> list[RankedLadder]:
    """Multi-scenario robustness ranking. Raises if no REALIZED anchor is present (E4)."""
    _require(
        any(s.is_realized_anchor for s in scenarios),
        "rank_ladders requires a REALIZED anchor scenario (E4)",
    )
    offer_by_term = {o.term_months: o for o in req.term_structure.offers}
    lam = req.constraints.robustness_lambda
    real_world = [s for s in scenarios if not s.is_curve_implied]

    sims_by_cand: dict[str, dict[str, SimResult]] = {}
    for cand in candidates:
        per: dict[str, SimResult] = {}
        for scen in scenarios:
            per[scen.scenario_id] = simulate_candidate(cand, scen, req, offer_by_term)
        sims_by_cand[cand.candidate_id] = per

    # ALL_SHORT min_eatv reference for the min_s gate (overfitting fix e)
    short_cand = next((c for c in candidates if c.archetype == "ALL_SHORT"), None)
    short_min = Decimal(0)
    if short_cand is not None:
        short_min = min(
            sims_by_cand[short_cand.candidate_id][s.scenario_id].terminal_value for s in real_world
        )
    tie_abs = req.budget * req.constraints.tie_bps / _BPS

    ranked: list[RankedLadder] = []
    for cand in candidates:
        per = sims_by_cand[cand.candidate_id]
        evs = [per[s.scenario_id].terminal_value for s in real_world]
        mean_ev = sum(evs, Decimal(0)) / Decimal(len(evs))
        min_ev, max_ev = min(evs), max(evs)
        base = (Decimal(1) - lam) * mean_ev + lam * min_ev
        uninsured = per[real_world[0].scenario_id].max_uninsured_excess
        rank_key = base - req.constraints.uninsured_penalty_lambda * uninsured
        # min_s gate: a candidate that trails ALL_SHORT's worst case by > tie can't win on averaging
        if lam < 1 and min_ev < short_min - tie_abs:
            rank_key -= short_min - min_ev
        # path_fragile: the outcome depends materially (> _PATH_FRAGILE_PCT of budget) on which rate
        # path realizes -- a locked ladder has ~zero dispersion; a rolling one swings with the path.
        path_fragile = (max_ev - min_ev) > req.budget * _PATH_FRAGILE_PCT
        ranked.append(
            RankedLadder(
                candidate=cand,
                mean_eatv=mean_ev,
                min_eatv=min_ev,
                max_eatv=max_ev,
                std_eatv=_std(evs, mean_ev),
                rank_key=rank_key,
                path_fragile=path_fragile,
                per_scenario=per,
                banks_needed=per[real_world[0].scenario_id].banks_needed,
                uninsured_at_horizon=uninsured,
            )
        )
    ranked.sort(key=lambda r: (r.rank_key, r.min_eatv), reverse=True)
    return ranked


# ============================================================================
# LOCK-IN ASSESSMENT
# ============================================================================


def _lockin_bps(long_t: Decimal, short_t: Decimal, budget: Decimal, horizon_days: int) -> Decimal:
    long_y = _annualize(long_t / budget, horizon_days)
    short_y = _annualize(short_t / budget, horizon_days)
    return (long_y - short_y) * _BPS


def assess_lockin(
    req: OptimizerRequest,
    scenarios: list[RatePathScenario],
) -> RegimeLockinReport:
    """Lock-in diff (ALL_LONG vs the actually-rolled ALL_SHORT baseline), anchored on
    CURVE_IMPLIED, swept across the spread band. Raises if CURVE_IMPLIED is absent (E5)."""
    _require(
        any(s.is_curve_implied for s in scenarios),
        "assess_lockin requires a CURVE_IMPLIED breakeven anchor (E5)",
    )
    offer_by_term = {o.term_months: o for o in req.term_structure.offers}
    terms = tuple(req.constraints.allowed_terms)
    short, long = min(terms), max(terms)
    all_short = LadderCandidate("ASSESS_SHORT", "ALL_SHORT", {short: Decimal(1)})
    all_long = LadderCandidate("ASSESS_LONG", "ALL_LONG", {long: Decimal(1)})
    horizon_end = req.start + relativedelta(months=req.horizon_months)
    horizon_days = (horizon_end - req.start).days
    noise = req.constraints.noise_bps

    per_scenario_bps: dict[str, Decimal] = {}
    long_sims: dict[str, SimResult] = {}
    short_sims: dict[str, SimResult] = {}
    for scen in scenarios:
        ls = simulate_candidate(all_long, scen, req, offer_by_term)
        ss = simulate_candidate(all_short, scen, req, offer_by_term)
        long_sims[scen.scenario_id] = ls
        short_sims[scen.scenario_id] = ss
        per_scenario_bps[scen.scenario_id] = _lockin_bps(
            ls.terminal_value, ss.terminal_value, req.budget, horizon_days
        )

    curve_implied_bps = next(
        per_scenario_bps[s.scenario_id] for s in scenarios if s.is_curve_implied
    )
    real_world_ids = [s.scenario_id for s in scenarios if not s.is_curve_implied]
    rw_bps = {sid: per_scenario_bps[sid] for sid in real_world_ids}
    mean_bps = sum(rw_bps.values(), Decimal(0)) / Decimal(len(rw_bps))
    min_bps = min(rw_bps.values())
    max_bps = max(rw_bps.values())
    worst_id = min(rw_bps, key=lambda k: rw_bps[k])

    # B4: a structural-edge win could come from NDFL floor/band smoothing rather than a higher
    # locked coupon. At the worst real-world scenario (smallest long edge), if the long advantage
    # is explained by paying LESS tax rather than earning MORE gross, it is a tax effect.
    wl, ws_ = long_sims[worst_id], short_sims[worst_id]
    gross_delta = wl.gross_interest - ws_.gross_interest
    tax_delta = ws_.tax_paid - wl.tax_paid  # > 0 -> the long leg pays less tax
    tax_driven = tax_delta > 0 and gross_delta <= tax_delta

    # spread-band sweep (under the curve-implied anchor): does the verdict sign survive?
    ci = next(s for s in scenarios if s.is_curve_implied)
    band = req.constraints.spread_band
    band_bps: list[Decimal] = []
    for delta in (Decimal(0), band.widen_pp, band.tighten_pp):
        lt = simulate_candidate(
            all_long, ci, req, offer_by_term, roll_spread_delta_pp=delta
        ).terminal_value
        st = simulate_candidate(
            all_short, ci, req, offer_by_term, roll_spread_delta_pp=delta
        ).terminal_value
        band_bps.append(_lockin_bps(lt, st, req.budget, horizon_days))
    spread_min, spread_max = min(band_bps), max(band_bps)
    spread_sign_flips = spread_min < -noise < noise < spread_max

    # falsifiability (E6): the scenario set must SPAN rate directions -- include at least one
    # cutting path AND at least one non-cutting (hold/hike) path -- so it could have disproved
    # the recommendation either way. A set that only cuts (or only holds) cannot falsify, so no
    # robust claim is made. This is a property of the rate PATHS, not of the outcome signs.
    real_world = [s for s in scenarios if not s.is_curve_implied]
    has_cut = any(s.key_rate_at(horizon_end) < s.key_rate_at(req.start) for s in real_world)
    has_non_cut = any(s.key_rate_at(horizon_end) >= s.key_rate_at(req.start) for s in real_world)
    degenerate = not (has_cut and has_non_cut)

    curve_slope_bps = (offer_by_term[long].annual_rate - offer_by_term[short].annual_rate) * _BPS
    curve_inverted = curve_slope_bps < 0

    verdict, edge_source, message = _classify_lockin(
        curve_implied_bps=curve_implied_bps,
        min_bps=min_bps,
        max_bps=max_bps,
        worst_id=worst_id,
        noise=noise,
        spread_sign_flips=spread_sign_flips,
        degenerate=degenerate,
        curve_inverted=curve_inverted,
        curve_slope_bps=curve_slope_bps,
        tax_driven=tax_driven,
    )

    return RegimeLockinReport(
        baseline_id="ALL_SHORT",
        candidate_id="ALL_LONG",
        per_scenario_bps=per_scenario_bps,
        curve_implied_bps=curve_implied_bps,
        mean_lockin_bps=mean_bps,
        min_lockin_bps=min_bps,
        max_lockin_bps=max_bps,
        spread_band_min_bps=spread_min,
        spread_band_max_bps=spread_max,
        spread_sign_flips=spread_sign_flips,
        curve_slope_bps=curve_slope_bps,
        curve_inverted=curve_inverted,
        edge_source=edge_source,
        verdict=verdict,
        worst_case_scenario_id=worst_id,
        honest_message=message,
        spread_held_constant=False,
        scenario_set_degenerate=degenerate,
        n1_caveat=True,
    )


def _classify_lockin(  # noqa: PLR0911 - explicit verdict decision tree, one return per outcome
    *,
    curve_implied_bps: Decimal,
    min_bps: Decimal,
    max_bps: Decimal,
    worst_id: str,
    noise: Decimal,
    spread_sign_flips: bool,
    degenerate: bool,
    curve_inverted: bool,
    curve_slope_bps: Decimal,
    tax_driven: bool,
) -> tuple[LockinVerdict, str, str]:
    inverted_note = (
        f" The offered curve is inverted (long < short by {abs(curve_slope_bps):.0f}bps), "
        "so the market is already pricing cuts."
        if curve_inverted
        else ""
    )
    # W1: only claim "~0 as expected" when the bisection actually solved a breakeven; on a steep
    # curve it clamps and curve_implied_bps is large -- never call a large number "~0".
    if abs(curve_implied_bps) <= noise:
        breakeven_note = (
            f" (Curve-implied breakeven lock-in = {curve_implied_bps:.0f}bps, ~0 as expected: the "
            "offered curve internally prices the cuts.)"
        )
    else:
        breakeven_note = (
            f" (Curve-implied breakeven did not solve to ~0 ({curve_implied_bps:.0f}bps): no flat "
            "forward rate offsets this curve's slope, which is what makes the lock structural.)"
        )
    # E6: a scenario set that does not SPAN rate directions (cut + non-cut) could not have
    # falsified the recommendation -> make no robust claim.
    if degenerate:
        return (
            LockinVerdict.NO_EDGE_CURVE_PRICES_CUTS,
            "none",
            "No edge claimed: the scenario set does not span both a cutting and a non-cutting "
            "rate path, so it could not have falsified the recommendation." + inverted_note,
        )
    # Structural edge: locking long wins in EVERY real-world scenario (including a rate HOLD).
    if min_bps > noise:
        if spread_sign_flips:
            return (
                LockinVerdict.NO_ROBUST_EDGE,
                "none",
                f"No robust edge: locking long wins (worst +{min_bps:.0f}bps) but the sign flips "
                "within a plausible forward-spread band -- an artifact of an assumed roll spread, "
                "not a measured edge." + inverted_note,
            )
        if tax_driven:
            return (
                LockinVerdict.TAX_SMOOTHING_EDGE,
                "tax_smoothing",
                f"Tax-smoothing edge: locking long wins (worst +{min_bps:.0f}bps) but the gain is "
                "from paying LESS NDFL (gross kept under the annual tax-free floor), not a higher "
                "coupon -- a tax effect, not a rate forecast." + inverted_note,
            )
        return (
            LockinVerdict.REAL_LOCKIN_EDGE,
            "rate_lock",
            "Real lock-in edge: locking long beats rolling short in EVERY modeled scenario "
            f"(worst +{min_bps:.0f}bps) -- a structural edge, not a rate bet."
            + breakeven_note
            + inverted_note,
        )
    # Locks long loses in every modeled scenario -> a pure liquidity/opportunity cost.
    if max_bps < -noise:
        return (
            LockinVerdict.LIQUIDITY_COST,
            "none",
            f"Liquidity cost: locking long loses in every scenario (up to {min_bps:.0f}bps, "
            f"{worst_id}) -- prefer the shorter ladder for liquidity." + inverted_note,
        )
    # Sign depends on the rate path -> a directional regime bet, not an edge.
    if min_bps < -noise < noise < max_bps:
        return (
            LockinVerdict.REGIME_BET_NOT_EDGE,
            "regime_bet",
            "Regime bet, not an edge: locking long wins only if cuts run deeper than the curve "
            f"prices (best +{max_bps:.0f}bps) and loses if rates hold (worst {min_bps:.0f}bps, "
            f"scenario {worst_id}) -- a directional rate forecast, not alpha."
            + breakeven_note
            + inverted_note,
        )
    return (
        LockinVerdict.NO_EDGE_CURVE_PRICES_CUTS,
        "none",
        "No lock-in edge: locking long neither beats nor materially trails rolling short under the "
        "modeled scenarios -- the term structure already prices the expected CBR cuts. Choose the "
        "shorter ladder for liquidity at no yield cost." + breakeven_note + inverted_note,
    )


# ============================================================================
# TOP-LEVEL
# ============================================================================


def _recommendation_caveat(
    recommended: RankedLadder,
    lockin: RegimeLockinReport,
    ranked: list[RankedLadder],
    req: OptimizerRequest,
) -> str:
    """Reconcile the robust-value recommendation with the honest lock-in verdict (B1): if the
    recommendation locks long but the verdict says that lock is NOT a structural edge, say so
    prominently and name the liquid alternative -- never let the recommendation silently
    contradict the verdict it ships beside."""
    shortest = min(req.constraints.allowed_terms)
    avg_term = sum((Decimal(t) * w for t, w in recommended.candidate.weights.items()), Decimal(0))
    locks_long = avg_term > Decimal(shortest)
    if not locks_long:
        return ""
    short = next((r for r in ranked if r.candidate.archetype == "ALL_SHORT"), None)
    alt = (
        f" Liquid all-short alternative robust value: {short.mean_eatv:,.0f} RUB." if short else ""
    )
    if lockin.verdict in (LockinVerdict.REGIME_BET_NOT_EDGE, LockinVerdict.NO_ROBUST_EDGE):
        return (
            "The recommended ladder out-ranks the rest only by betting on the modeled cut path; "
            f"the lock-in verdict is {lockin.verdict.value.upper()} -- NOT a structural edge. "
            f"Prefer a shorter, liquid ladder unless you will bet rates keep falling.{alt}"
        )
    if lockin.verdict == LockinVerdict.LIQUIDITY_COST:
        return (
            "WARNING: the recommended ladder locks long but the lock-in verdict is LIQUIDITY_COST "
            f"(long loses in every modeled scenario) -- prefer the shorter ladder.{alt}"
        )
    return ""


def optimize_deposit_ladder(req: OptimizerRequest) -> LadderPlan:
    """RECOMMENDATION ONLY: candidates -> scenarios -> rank -> lock-in -> recommend.

    Constructs no orders, touches no real-money path, calls no BrokerBase order method, performs
    no live HTTP fetch (the committed CBR calendar resolves the REALIZED scenario)."""
    offered_terms = {o.term_months for o in req.term_structure.offers}
    missing = [t for t in req.constraints.allowed_terms if t not in offered_terms]
    _require(not missing, f"allowed_terms {missing} have no offer in the snapshot (E1)")
    scenarios = make_default_scenarios(req)
    # W2: enforce min_liquid_fraction -- drop candidates that do not keep enough principal maturing
    # within the liquidity horizon (a stated constraint must bind, not be silently ignored).
    horizon = req.constraints.liquidity_horizon_months
    floor = req.constraints.min_liquid_fraction
    candidates = [
        c
        for c in generate_candidates(req)
        if sum((w for term, w in c.weights.items() if term <= horizon), Decimal(0)) >= floor
    ]
    _require(
        bool(candidates),
        f"no candidate ladder meets min_liquid_fraction={floor} within {horizon}mo",
    )
    ranked = rank_ladders(req, candidates, scenarios)
    lockin = assess_lockin(req, scenarios)
    ts = req.term_structure
    provenance = (
        f"{_TERM_STRUCTURE_PATH.name} | as_of={ts.as_of} | mode={ts.snapshot_mode} "
        f"| git_sha={ts.git_sha} | source={ts.source[:60]}..."
    )
    # B2: the tax-band caveat is a LOWER BOUND -- it must fire if ANY real-world scenario crosses
    # the 2.4M band (keying it off the lowest-interest REALIZED path alone would under-warn).
    progressive = any(
        ranked[0].per_scenario[s.scenario_id].progressive_band_caveat
        for s in scenarios
        if not s.is_curve_implied
    )
    return LadderPlan(
        budget=req.budget,
        start=req.start,
        horizon_months=req.horizon_months,
        scenarios_used=tuple(s.scenario_id for s in scenarios),
        ranked=tuple(ranked),
        recommended=ranked[0],
        lockin_report=lockin,
        snapshot_provenance=provenance,
        spread_held_constant=False,
        n1_caveat=True,
        progressive_band_caveat=progressive,
        recommendation_caveat=_recommendation_caveat(ranked[0], lockin, ranked, req),
    )
