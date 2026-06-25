# Deposit-Ladder Optimizer — Final Implementation-Ready Design

> Status: implementation-ready. Every refutation finding/fix below is integrated.
> Recommendation-only. No real-money path. No orders. No live broker, router, token, or session.
> All load-bearing facts verified against source (paths + line numbers cited inline).

---

## 1. Goal & non-goals

### Goal

A **recommendation-only** ranker that, given a budget, a *real* offered deposit term-structure,
an operator's constraints, and a set of CBR key-rate scenarios, simulates candidate deposit
ladders through the **existing `DepositSimulatedBroker`** (`src/finalayze/execution/deposit_broker.py`)
and reports which split maximizes a robust expected after-tax terminal value — **and whether any
lock-in edge is real or whether the offered curve already prices the cuts.**

The single most important output is the honest `lock long vs roll short` comparison, anchored to a
**derived curve-implied breakeven scenario** (the real efficient-market test). The default outcome in
an easing/inverted-curve regime is **"no edge — the curve prices the cuts,"** printed as a first-class
success.

This is a near-deterministic instrument: no alpha, no fill, no slippage. The tool is a
**scenario-robustness + tax/liquidity trade-off ranker**, not a return search.

### Non-goals (hard boundaries)

- **No real-money path.** Moves no money, constructs no orders, opens no live broker/channel/session,
  holds no token, touches no DB. The only path that ever reaches a real broker remains
  `scripts/run_rebalance.py`; this tool never imports it.
- **No order objects.** Result types hold only `Decimal` metrics + weight maps — never `OrderRequest`,
  `PlannedLeg`, a broker handle, or `submit()`.
- **No frozen-path edits.** Zero edits to `deposit_broker.py`, `ndfl.py`, `orchestration/allocation.py`,
  `rebalance_planner.py`, `allocation_gate.py`, the frozen gate/cert, or the real-money path.
- **No fabricated data.** No synthetic term structure, no invented bank-default probability, no
  hard-coded tax allowance, no flat-1.0pp spread fallback. Fail closed when real inputs are absent.
- **No SAA wire-up.** Does not inject tranche detail into `AllocationOrchestrator`/`plan_rebalance`
  (that path is frozen and gate-measured; its deposit leg is a single mark-only `ManualAction`).
  The future read-only seam is documented, not built.

---

## 2. Inputs & outputs (frozen dataclasses + public signatures)

New module `src/finalayze/orchestration/deposit_ladder.py` (Layer 5 — sibling of the existing L5
`orchestration/allocation.py`; imports L0 schemas/ndfl/constants + the L5 simulated broker from
`execution/`; never imports L6). Money = `Decimal` RUB; rates = `Decimal` annual fractions
(matching `DepositTranche.annual_rate` / `deposit_rate_as_of`). Key-rate path values are in
**percentage points** (e.g. `Decimal("20.00")` = 20%), matching the CBR calendar convention
(cbr.py: `key_rate` is pp). All result dataclasses are `frozen=True` — every field DERIVED.

```python
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

# ============================================================================
# INPUTS (operator-supplied or committed snapshot)
# ============================================================================

@dataclass(frozen=True)
class TermOffer:
    """One point on the REAL offered term structure (snapshot/operator, never synthetic)."""
    term_months: int                    # 3 / 6 / 12 / 36 ...
    annual_rate: Decimal                # offered fraction, e.g. Decimal("0.19")
    roll_spread_pp: Decimal             # SIGNED per-term spread vs key rate, REQUIRED (no default).
                                        #   The loader FAILS CLOSED if absent (data-honesty fix #1):
                                        #   never falls back to _DEFAULT_DEPOSIT_SPREAD_PP / flat 1.0pp.
    instrument_type: str = "deposit"    # "deposit" | "irrevocable_cert" -> drives ASV tier
    bank_id: str | None = None          # optional; enables per-bank ASV reporting

@dataclass(frozen=True)
class SpreadBand:
    """Forward-spread perturbation envelope for the short roll leg (efficient-market fix #2).

    The as-of roll_spread_pp is a SNAPSHOT fact; its FORWARD persistence is an assumption.
    We sweep it instead of freezing it. pp deltas applied to every roll's spread along the path.
    """
    widen_pp: Decimal = Decimal("0.5")    # spread widens (rolled short earns LESS) under easing
    tighten_pp: Decimal = Decimal("-0.5") # spread tightens (rolled short earns MORE)

@dataclass(frozen=True)
class LadderConstraints:
    allowed_terms: tuple[int, ...] = (3, 6, 12, 36)
    min_liquid_fraction: Decimal = Decimal("0")     # frac that must mature <= liquidity_horizon
    liquidity_horizon_months: int = 6
    grid_step: Decimal = Decimal("0.25")            # coarse simplex step
    max_candidates: int = 64                        # hard cap; coarsen grid_step until under it
    robustness_lambda: Decimal = Decimal("0.5")     # 0=mean .. 1=worst-case (maximin)
    noise_bps: Decimal = Decimal("5")               # de-minimis lock-in edge (annualized bps)
    tie_bps: Decimal = Decimal("5")                 # robustness tie band
    uninsured_tolerance_rub: Decimal = Decimal("0") # operator's accepted uninsured excess
    uninsured_penalty_lambda: Decimal = Decimal("0")# 0 => uninsured reported as raw axis, not penalized
    max_offer_staleness_days: int = 14              # fail-closed staleness gate (new L0 constant)
    spread_band: SpreadBand = SpreadBand()          # forward-spread sweep band (efficient-market fix)

class AsvTier(StrEnum):
    """ASV insurance tiers — VERIFIED against Minfin (CORRECTS the consilium refuter, which
    wrongly claimed ordinary deposits never get a raised tier; the operator's own RF-fact
    challenge prompted re-verification at source).

    Source: minfin.gov.ru press-center id_4=40134 + id_4=39825 (raised limits "from 18 Dec 2025").
    NOTE: this source frames the ordinary-deposit>3yr raise partly as a законопроект; the
    effective date / legal force should be RE-VERIFIED before relying on a raised tier. ASV here
    is a SOFT, non-blocking REPORTED metric (uninsured_penalty_lambda=0) — a mis-tiered cap only
    mislabels a risk number, never excludes a candidate or moves money. The boundary is "свыше
    3 лет" = STRICTLY >3yr, so a 36mo (=3yr exactly) instrument stays STANDARD.
    """
    STANDARD = "1.4M"          # ordinary deposit, term <= 3yr (36mo INCLUSIVE)
    RAISED_2M = "2.0M"         # ordinary deposit strictly >3yr  OR  irrevocable cert 1-3yr
    RAISED_2_8M = "2.8M"       # irrevocable cert strictly >3yr
                               # (escrow accounts -> 30M: out of scope for this tool)

@dataclass(frozen=True)
class RatePathScenario:
    scenario_id: str                                # "REALIZED"|"CURVE_IMPLIED"|"FAST_CUTS"|...
    key_rate_at: Callable[[date], Decimal]          # key rate in PP; perturbation of the REAL calendar
    weight: Decimal = Decimal("1")                  # prior weight for the mean functional
    is_realized_anchor: bool = False                # exactly one True (the committed CBR calendar)
    is_curve_implied: bool = False                  # exactly one True (the breakeven anchor)

@dataclass(frozen=True)
class TermStructure:
    """The committed, dated, provenanced offered-rate snapshot (fail-closed loaded)."""
    as_of: date
    source: str                                     # bank + where the rate was read (provenance)
    git_sha: str | None                             # snapshot provenance (mirrors gate snapshot)
    offers: tuple[TermOffer, ...]
    scenarios: tuple[RatePathScenario, ...]         # named key-rate paths anchored to the real calendar

@dataclass(frozen=True)
class OptimizerRequest:
    budget: Decimal
    start: date
    horizon_months: int
    term_structure: TermStructure                   # REQUIRED; no synthetic fallback
    constraints: LadderConstraints = LadderConstraints()
    ytd_other_taxable_income: Decimal = Decimal("0") # operator-supplied dividends+coupons+other
                                                     #   (tax-insurance fix A: seeds the cross-sleeve
                                                     #    2.4M progressive-band lower bound)

# ============================================================================
# CANDIDATE / RESULT
# ============================================================================

@dataclass(frozen=True)
class BankAllocation:        # re-exported shape; see execution.deposit_broker.BankAllocation
    bank_id: str
    principal: Decimal
    accrued_net: Decimal

@dataclass(frozen=True)
class LadderCandidate:
    candidate_id: str
    archetype: str                                  # "ALL_SHORT"|"ALL_LONG"|"BARBELL"|"EVEN_SPREAD"|
                                                    #   "FRONT_WEIGHTED"|"BACK_WEIGHTED"|"LIQUIDITY_FLOOR"|"GRID"
    weights: Mapping[int, Decimal]                  # term_months -> fraction, sums to 1

@dataclass(frozen=True)
class SimResult:
    candidate_id: str
    scenario_id: str
    terminal_value: Decimal              # hold-to-maturity after-tax mark at horizon (broker.deposit_value())
    liquidation_floor: Decimal           # value if all still-locked tranches broken at horizon (risk line)
    gross_interest: Decimal
    net_interest: Decimal
    tax_paid: Decimal
    effective_after_tax_yield: Decimal   # (terminal/budget)**(365/horizon_days) - 1
    roll_count: int                      # how much of terminal is re-priced vs locked
    locked_value_fraction: Decimal       # share of terminal from REAL observed offer rates vs scenario rolls
    min_liquidity_fraction: Decimal      # min over horizon of frac maturing within liq_horizon
    max_uninsured_excess: Decimal        # tier-resolved, principal+accrued, path-max
    progressive_band_caveat: bool        # DEPOSIT-SLEEVE-ONLY YTD (seeded by ytd_other_taxable_income)
                                         #   crossed 2.4M in some year (flat-13% under-tax); LOWER BOUND only

class LockinVerdict(StrEnum):
    NO_EDGE_CURVE_PRICES_CUTS = "no_edge_curve_prices_cuts"  # default success
    REAL_LOCKIN_EDGE          = "real_lockin_edge"           # survives CURVE_IMPLIED + spread band
    TAX_SMOOTHING_EDGE        = "tax_smoothing_edge"
    LIQUIDITY_COST            = "liquidity_cost"
    REGIME_BET_NOT_EDGE       = "regime_bet_not_edge"        # wins only vs CURVE_IMPLIED -> a rate forecast
    NO_ROBUST_EDGE            = "no_robust_edge"             # sign flips within the spread band / degenerate set

@dataclass(frozen=True)
class RegimeLockinReport:
    baseline_id: str                     # ALL_SHORT (realized roll baseline)
    candidate_id: str                    # best non-short / ALL_LONG
    per_scenario_bps: Mapping[str, Decimal]
    curve_implied_bps: Decimal           # lock-in benefit under the DERIVED breakeven path (the EMH anchor)
    mean_lockin_bps: Decimal
    min_lockin_bps: Decimal
    max_lockin_bps: Decimal
    spread_band_min_bps: Decimal         # worst lock-in across the forward-spread sweep
    spread_band_max_bps: Decimal         # best  lock-in across the forward-spread sweep
    spread_sign_flips: bool              # verdict sign flips within the spread band -> NO_ROBUST_EDGE
    curve_slope_bps: Decimal             # long_rate(start) - short_rate(start)
    curve_inverted: bool
    edge_source: str                     # "rate_lock" | "tax_smoothing" | "regime_bet" | "none"
    verdict: LockinVerdict
    worst_case_scenario_id: str          # the scenario behind min_lockin_bps (label on the bps number)
    honest_message: str                  # verbatim operator-facing sentence
    spread_held_constant: bool = False   # True only if no spread band swept (explicit assumption flag)
    scenario_set_degenerate: bool = False# set could not falsify the recommendation -> edge demoted
    n1_caveat: bool = True               # committed calendar = one realized easing cycle

@dataclass(frozen=True)
class RankedLadder:
    candidate: LadderCandidate
    mean_eatv: Decimal
    min_eatv: Decimal
    max_eatv: Decimal
    std_eatv: Decimal
    rank_key: Decimal
    path_fragile: bool                   # wins on REALIZED but worst on another scenario
    per_scenario: Mapping[str, SimResult]
    bank_split: tuple[BankAllocation, ...]   # via split_across_banks at the TIER cap, for ASV reporting
    uninsured_at_horizon: Decimal

@dataclass(frozen=True)
class LadderPlan:                         # RECOMMENDATION-ONLY: holds NO orders, NO broker handle
    request_echo: OptimizerRequest
    scenarios_used: tuple[str, ...]
    ranked: tuple[RankedLadder, ...]      # best-first by rank_key, tie-broken on liquidity+ASV
    recommended: RankedLadder
    lockin_report: RegimeLockinReport
    snapshot_provenance: str             # snapshot path + as_of + git_sha (reproducibility)
    spread_held_constant: bool
    n1_caveat: bool
    progressive_band_caveat: bool

# ============================================================================
# ENTRY POINTS
# ============================================================================

_TERM_STRUCTURE: Path  # src/finalayze/orchestration/data/deposit_term_structure.json

def load_term_structure(path: Path = _TERM_STRUCTURE) -> TermStructure:
    """Fail-closed (ConfigurationError). Raises on: missing/corrupt file, missing key,
    non-positive rate, a TermOffer missing an explicit signed roll_spread_pp (NEVER falls
    back to _DEFAULT_DEPOSIT_SPREAD_PP/flat 1.0pp), missing/stale as_of (> max_offer_staleness_days),
    a requested allowed_term with no offer, a scenario bar post-dating as_of+horizon, and a
    scenario set that lacks a REALIZED anchor or a CURVE_IMPLIED breakeven, or that cannot
    FALSIFY the recommendation (no scenario where ALL_LONG wins AND none where ALL_SHORT wins).
    NO synthetic fallback (NEVER deposit_rate_as_of for offered rates)."""

def make_default_scenarios(start: date, horizon_months: int,
                           term_structure: TermStructure) -> list[RatePathScenario]:
    """REALIZED (real CBR calendar via deposit_rate_as_of, is_realized_anchor=True)
    + CURVE_IMPLIED (DERIVED breakeven, is_curve_implied=True)
    + FAST_CUTS/SLOW_CUTS/HOLD perturbations whose amplitude BRACKETS the curve-implied path."""

def derive_curve_implied_scenario(term_structure: TermStructure, start: date,
                                  horizon_months: int) -> RatePathScenario:
    """Back out the market-implied forward short-rate path from the offered term structure
    (no-arbitrage on after-tax compounded values: the long lock and the rolled-short path must
    deliver equal terminal value under this path). This IS the curve-prices-cuts anchor."""

def generate_candidates(req: OptimizerRequest) -> list[LadderCandidate]:
    """<=8 archetypes + capped coarse simplex grid over allowed_terms; de-dup; <= max_candidates."""

def asv_tier_cap(offer: TermOffer) -> Decimal:
    """VERIFIED against Minfin (see AsvTier). Boundary = strictly >3yr (36mo stays STANDARD):
      deposit,  term <= 36mo  -> 1.4M
      deposit,  term  > 36mo  -> 2.0M   (raised tier; effective-date caveat per AsvTier docstring)
      irrevocable_cert, 12..36mo -> 2.0M
      irrevocable_cert, term > 36mo -> 2.8M
    Soft reported metric only — never excludes a candidate."""

def simulate_candidate(candidate: LadderCandidate, scenario: RatePathScenario,
                       req: OptimizerRequest, *, roll_spread_delta_pp: Decimal = Decimal("0")) -> SimResult:
    """Build opening tranches at REAL offer rates; drive ONE fresh DepositSimulatedBroker day-by-day
    to the horizon; at each maturity the DRIVER rebuilds the rolled tranche at the SCENARIO rate
    (= scenario.key_rate_at(roll_date)/100 + offer.roll_spread_pp + roll_spread_delta_pp).
    NDFL/floor handled INSIDE accrue. Pure: never calls submit_order/cancel_order."""

def rank_ladders(req: OptimizerRequest, candidates: list[LadderCandidate],
                 scenarios: list[RatePathScenario]) -> list[RankedLadder]:
    """Multi-scenario robustness ranking. Raises ConfigurationError if no REALIZED anchor present."""

def assess_lockin(req: OptimizerRequest, ranked: list[RankedLadder],
                  scenarios: list[RatePathScenario]) -> RegimeLockinReport:
    """Lock-in diff vs the realized-roll ALL_SHORT baseline, anchored on CURVE_IMPLIED, swept
    across the spread band. Raises ConfigurationError if CURVE_IMPLIED anchor absent."""

def optimize_deposit_ladder(req: OptimizerRequest) -> LadderPlan:
    """Top-level: candidates -> scenarios(default if none) -> rank -> lockin -> recommend.
    RECOMMENDATION ONLY; constructs no orders, touches no real-money path, calls no BrokerBase
    order method, performs no live HTTP fetch (deposit_rate_as_of resolves the committed calendar)."""
```

---

## 3. Data-source contract — real snapshot + fail-closed loader

The codebase has **no offered-rate-by-term data**: `deposit_rate_as_of` (cbr.py:782) synthesizes
`(key_rate − spread_pp)/100` for *every* term with `spread_pp` defaulting to
`_DEFAULT_DEPOSIT_SPREAD_PP = Decimal("1.0")` (cbr.py:778). So any per-term structure not sourced
from the operator IS a fabrication. The optimizer treats offered rates as **committed-snapshot /
operator-supplied data and FAILS CLOSED when absent** — mirroring `allocation_gate._load_gate_snapshot`
+ `ConfigurationError` (`finalayze.core.exceptions:19`, verified) discipline, and the
`generated_at` / `git_sha` / `window` provenance fields already in
`backtest/data/allocation_gate_snapshot.json` (verified).

### Committed fixture — `src/finalayze/orchestration/data/deposit_term_structure.json`

`roll_spread_pp` is **REQUIRED per offer** (no JSON default; loader fails closed if absent).

```json
{
  "generated_at": "2026-06-20T00:00:00+00:00",
  "as_of": "2026-06-20",
  "git_sha": "<git sha at snapshot time>",
  "source": "operator-curated offered rates (bank X retail; read 2026-06-20)",
  "offers": [
    {"term_months": 3,  "annual_rate": "0.18",  "roll_spread_pp": "-2.0", "instrument_type": "deposit"},
    {"term_months": 6,  "annual_rate": "0.185", "roll_spread_pp": "-1.5", "instrument_type": "deposit"},
    {"term_months": 12, "annual_rate": "0.19",  "roll_spread_pp": "-1.0", "instrument_type": "deposit"},
    {"term_months": 36, "annual_rate": "0.16",  "roll_spread_pp": "-4.0", "instrument_type": "deposit"}
  ],
  "key_rate_scenarios": {
    "REALIZED": "use_committed_cbr_calendar",
    "FAST_CUTS": [["2026-06-20","20.00"], ["2026-07-25","18.00"], ["2026-09-12","16.00"]],
    "SLOW_CUTS": [["2026-06-20","20.00"], ["2026-09-12","19.00"], ["2026-12-19","18.00"]],
    "HOLD":      [["2026-06-20","20.00"]]
  }
}
```

`CURVE_IMPLIED` is **not stored as literals** — it is DERIVED at load/optimize time from `offers`
(no-arbitrage back-out), so it can never disagree with the committed curve.

### Fail-closed rules (E1–E6)

- **E1 — no fabricated term structure, ever.** Missing file / incomplete (a requested `allowed_term`
  has no offer) / **stale** (`today − as_of > max_offer_staleness_days`, default 14) → `ConfigurationError`,
  refuse to rank. Never interpolate, never reuse `deposit_rate_as_of`, never carry a neighboring term's
  rate. Cite the missing field and (for staleness) the age in days.
- **E2 — snapshot is committed, dated, provenanced** (`as_of`, `source`, `git_sha`, per-`(term[,bank])`
  rate, `instrument_type` for the ASV tier). `snapshot_provenance` recorded in the result for
  reproducibility (path + `as_of` + `git_sha`), matching the gate snapshot discipline.
- **E3 — `roll_spread_pp` REQUIRED, no default** (data-honesty fix #1). The loader raises
  `ConfigurationError` if any allowed-term offer lacks an explicit signed `roll_spread_pp`. It NEVER
  falls back to `_DEFAULT_DEPOSIT_SPREAD_PP` / a flat 1.0pp. A regression test asserts this.
- **E4 — REALIZED is a hard invariant** (data-honesty fix #2). The scenario set fed to
  `rank_ladders`/`assess_lockin` MUST contain exactly one `is_realized_anchor=True` scenario bound to
  the committed CBR calendar; `rank_ladders`/`assess_lockin` raise `ConfigurationError` if it is absent.
  The real-anchor leg can never be optimized away.
- **E5 — CURVE_IMPLIED is a hard invariant** (efficient-market fix #1). The set MUST contain exactly one
  `is_curve_implied=True` breakeven scenario derived from `offers`; `assess_lockin` raises
  `ConfigurationError` if absent. It is the decision anchor for any lock-in claim.
- **E6 — scenario set must be able to FALSIFY the recommendation** (overfitting fix a). The set must
  contain at least one scenario under which `ALL_LONG` wins the lock-in diff AND at least one under which
  `ALL_SHORT` wins. If every scenario agrees, the loader stamps `scenario_set_degenerate=True` and
  `assess_lockin` demotes any `REAL_LOCKIN_EDGE`/`TAX_SMOOTHING_EDGE` to `NO_EDGE_CURVE_PRICES_CUTS`
  (you cannot claim a robust edge from a set that could not have disproved it). Perturbation amplitude
  is DERIVED so the `FAST/SLOW` envelope brackets the curve-implied path (overfitting fix b).
- **Every projected number marked projected** (roll rates, future-year allowances, spread persistence,
  uninsured EV) — `locked_value_fraction`/`roll_count` surface the assumption share (H6).

### New L0 constant

`MAX_OFFER_STALENESS_DAYS = 14` in `src/finalayze/core/constants.py` (additive). Optional named raised-cert
tiers `ASV_RAISED_CERT_1_3Y = Decimal(2_000_000)` and `ASV_RAISED_CERT_OVER_3Y = Decimal(2_800_000)` if the
tier resolver needs them; default path uses the verified `ASV_CAP_PER_BANK = Decimal(1_400_000)` (constants.py:24).

---

## 4. Optimization algorithm

### (a) Candidate generation — bounded, not combinatorial

A ladder = fractional weights over `allowed_terms` summing to 1.

- **Archetypes (≤8):** `ALL_SHORT` (= the realized-roll baseline), `ALL_LONG` (lock-in candidate),
  `BARBELL` (short+long), `EVEN_SPREAD`, `FRONT_WEIGHTED`/`BACK_WEIGHTED`,
  `LIQUIDITY_FLOOR` (min long given `min_liquid_fraction`).
- **Coarse simplex grid:** weight vectors on step `g` (default 0.25). For `T=4, g=0.25` → C(7,3)=35;
  `T=3` → 15. **Hard cap `max_candidates=64`; auto-coarsen `g` (0.25→0.334→0.5) until under cap.**
  De-dup grid points coinciding with archetypes. Total ≈ 15–64, each cheap.
- **Bank-split is DERIVED, never searched** — `split_across_banks()` at the tier cap, post-hoc,
  for the ASV report only. No term×bank Cartesian product.

### (b) Per-candidate simulation (the whole-ladder unit)

One **fresh** `DepositSimulatedBroker(initial_cash=Decimal(0), tranches=expand(candidate))` per
(candidate, scenario, spread-delta). Driver day loop over business days `[start, start+horizon]`:

1. `broker.accrue(d)` — daily-compound net interest, NDFL R-2 floor applied **inside** against the
   shared YTD accumulators (verified accrue:115–148). **One broker = one YTD allowance pool consumed
   across ALL tranches** — `self._ytd_deposit_gross` is broker-level (accrue:117,143), reset on the
   Jan-1 boundary (accrue:116–119). This is the dominant double-count hazard; a hard invariant +
   regression test. Idempotent per date (accrue:110–113, WR-04).
2. At each maturity `d < horizon_end`, the **driver** replaces the matured tranche with a freshly-built
   `DepositTranche` reproducing the exact `roll_at_maturity` construction (verified deposit_broker.py:187–195)
   but substituting the scenario rate:
   `annual_rate = scenario.key_rate_at(d)/100 + offer.roll_spread_pp + roll_spread_delta_pp`
   — **scenario injection without touching the broker** (no `rate_provider` param, no broker edit).
3. At `horizon_end`: `terminal = broker.deposit_value()` (deposit_broker.py:150, hold-to-maturity,
   after-tax, NOT broken). Additionally compute `liquidation_floor` (value if every still-locked
   tranche were `break_tranche`'d — resets accrued to `principal × DEPOSIT_DEMAND_RATE`,
   deposit_broker.py:159–175) as a **reported risk line, not the objective.**

**Whole-ladder-as-a-unit is mandatory** because NDFL is path- and YTD-dependent (one floor + one
progressive band per calendar year shared across tranches) — terminal value is **not separable**;
a closed form would silently misprice the tax and fabricate a false ladder edge.

**Objective:** `EATV(L,s) = SimResult.terminal_value`. Secondary metrics reported, not in the scalar.

### (c) Multi-scenario robustness ranking (anti-overfit)

Never rank on one path. For each candidate aggregate `EATV` over the scenario set:

```
base(L)     = (1-λ)·mean_s EATV(L,s) + λ·min_s EATV(L,s)        # λ default 0.5
rank_key(L) = base(L)
            − uninsured_penalty_lambda · uninsured_at_horizon   # 0 by default (raw axis)
            − penalty(liquidity_breach)
```

- Report `mean/min/max/std` per candidate; flag `path_fragile` if it wins on REALIZED but is worst on
  another scenario.
- **min_s gate (overfitting fix e):** when `λ < 1`, a candidate may NOT rank #1 if its `min_s EATV`
  trails `ALL_SHORT`'s `min_s EATV` by more than `tie_bps` — so a mean-strong / worst-catastrophic
  ladder can never win on averaging.
- **Tie band** `tie_bps`: among near-ties, prefer **more liquidity and less uninsured exposure**
  (lexicographic) — the tool never "recommends" a lock-in that beats short by a rounding error.

**Scenario set (CBR-anchored, derived/perturbations of the real calendar — not free invention):**
`REALIZED` (committed calendar), `CURVE_IMPLIED` (derived breakeven — the decision anchor),
`FAST_CUTS`/`SLOW_CUTS` (amplitude brackets the curve-implied path), `HOLD` (frozen at start rate),
optional `HIKE_TAIL`. Plus, for the lock-in assessment, the **forward-spread sweep**
(`roll_spread_delta_pp ∈ {0, widen_pp, tighten_pp}`) applied to the short roll leg.

---

## 5. HONESTY RAILS (baked in)

- **H1 — lock-in is a simulated diff, never asserted.** `lockin_benefit(L,s) = EATV(L,s) − EATV(ALL_SHORT, s)`,
  where `ALL_SHORT` is the realized all-short roll baseline (100% shortest term, *actually rolled* at every
  maturity at `scenario.key_rate_at(roll_date)/100 + roll_spread_pp + delta`). Both legs run the **same
  broker, same NDFL state machine, same scenario, same spread delta** — apples-to-apples on tax. Default
  easing-regime output is `NO_EDGE_CURVE_PRICES_CUTS`, printed verbatim and prominently as a **success**.
- **H2 — CURVE_IMPLIED is the EMH anchor (efficient-market fix #1).** A derived breakeven scenario whose
  forward short-rate path makes a roll-short ladder's after-tax EATV exactly equal the locked long-rung
  EATV is run FIRST. If lock-in ≈ 0 under `CURVE_IMPLIED`, that IS the curve-prices-cuts proof, stated as
  the primary success. `REAL_LOCKIN_EDGE` may be asserted ONLY when `min_lockin_bps > noise_bps` even under
  `CURVE_IMPLIED` AND across the spread band; otherwise the "edge" is the modeled path disagreeing with the
  market → `REGIME_BET_NOT_EDGE`, flagged as a directional rate forecast, not alpha.
- **H3 — inverted-curve prior stated BEFORE sim, confirmed after, quantified in bps.**
  `curve_slope = long_rate(start) − short_rate(start)`. If `≤ 0`, the report leads with:
  *"Deposit curve is inverted (long < short by X bps) — the market is pricing cuts; a long lock-in starts
  behind and must be earned back by realized cuts being deeper than priced."*
- **H4 — spread persistence is swept, never frozen (efficient-market + data-honesty + overfitting fix).**
  `roll_spread_pp` is a snapshot FACT; its forward persistence is an ASSUMPTION. The short leg's roll spread
  is swept over `SpreadBand` (constant, +widen, −tighten). If the verdict sign flips within the band,
  `spread_sign_flips=True` and the verdict is demoted to `NO_ROBUST_EDGE`. If no band is swept,
  `spread_held_constant=True` is surfaced as an explicit assumption flag beside `n1_caveat`.
- **H5 — tax-smoothing vs rate-lock disambiguated.** A "win" surviving only via tax-smoothing of gross
  interest under the annual floor/band (detected by comparing `tax_paid` at near-equal gross) →
  `TAX_SMOOTHING_EDGE`, labeled a tax effect, never a rate forecast.
- **H6 — robustness over a CBR-anchored, falsifiable scenario set.** Worst-case awareness +
  `path_fragile`/`std` dispersion + `scenario_set_degenerate` demotion; `n1_caveat` carried throughout
  (committed calendar = one realized easing cycle; the scenario set is the ONLY source of path diversity —
  we never imply statistical confidence the data cannot support).
- **H7 — locked vs roll-leg split surfaced** (`locked_value_fraction`, `roll_count`): a long-horizon
  all-short ladder is *mostly assumption*; a horizon-matched long-rung ladder is *mostly locked* — that
  locked share IS the lock-in being measured. Roll rates, future-year allowances, and spread persistence
  are explicitly **projected/assumption**, not fact.
- **H8 — whole-ladder simulation through the real broker is mandatory** (NDFL non-separability).
- **H9 — NDFL floor stays on the committed live key-rate** (verified accrue:123 reads
  `deposit_rate_as_of(current_date, spread_pp=_ZERO_SPREAD_PP)` from the committed calendar, independent
  of the scenario). The scenario levers are the **offered term-structure rates** (`tranche.annual_rate`)
  and the **roll rate** — documented explicitly, not silently forked. A perturbed scenario's future-year
  allowance is therefore approximate; flagged as a known limitation.
- **H10 — progressive band is DEPOSIT-SLEEVE-ONLY, a LOWER BOUND (tax-insurance fix A).** The broker
  deposit path is flat 13% (`ndfl_on_deposit_interest` ends in `* NDFL_RATE`, ndfl.py:84). The statutory
  13/15%-above-2.4M band (`ndfl_marginal`/`YtdTaxAccumulator`, threshold 2.4M) is **cross-sleeve**
  (the YtdTaxAccumulator's `ytd_before` is the cumulative taxable income across ALL sleeves). This tool
  sees deposits in isolation, so `progressive_band_caveat` can only **lower-bound** the band crossing.
  It is seeded by `OptimizerRequest.ytd_other_taxable_income` (operator dividends+coupons+other) so it
  fires correctly when the cross-sleeve total crosses 2.4M; the under-tax magnitude is estimated post-hoc
  via `ndfl_marginal`. The materiality rationale is **TOTAL cross-sleeve portfolio income ≪ 2.4M at
  ~2.5M capital** (≈475k deposit interest; crossing 2.4M would need ~96% portfolio yield, impossible) —
  NOT the incomplete "deposit interest ≪ 2.4M" rationale. Wiring the progressive band into the deposit
  broker is a separate, broker-touching, re-securing change — out of scope.
- **H11 — no fabricated bank-default probability** (data-honesty). Uninsured RUB surfaced as a raw risk
  axis; converted to EV only if the operator supplies `p`/λ.
- **H12 — ASV is a soft axis, never a hard cap** (operator legitimately holds >1.4M, modeled as
  `uninsured_excess` FLAG only, deposit_broker.py:234). Tier resolver is **VERIFIED against Minfin
  (corrects the consilium refuter's RF-fact error — re-checked at source after the operator's own
  challenge):** boundary is strictly >3yr, so `deposit` ≤36mo → `1.4M`, `deposit` >36mo → `2.0M`;
  `irrevocable_cert` 1-3yr → `2.0M`, >3yr → `2.8M`. Effective-date/legal-force of the raised
  ordinary-deposit tier should be re-verified (source frames it partly as a законопроект "from
  18 Dec 2025"); since ASV is non-blocking and the fixture's deposits are all ≤3yr, the nuance is
  immaterial to any recommendation — a mis-tiered cap only mislabels a risk metric, never excludes
  a candidate, never moves money. **LESSON: do not trust a consilium refuter on RF legal facts —
  verify against Minfin/АСВ at source.**

### Decision rule (the product's spine)

1. lock-in ≈ 0 under `CURVE_IMPLIED` (within `noise_bps`) → **`NO_EDGE_CURVE_PRICES_CUTS`**, printed
   verbatim and prominently as a **success:** *"No lock-in edge: the deposit term structure already
   prices the expected CBR cuts. Locking long at today's rate does not beat rolling short under the
   curve-implied path; choose the shorter ladder for liquidity at no yield cost."*
2. `min_lockin_bps < 0` (long loses in ≥1 scenario, typical when curve inverted) → **`LIQUIDITY_COST`**,
   quantified in bps, recommend short.
3. `min_lockin_bps > noise_bps` under `CURVE_IMPLIED` AND across the spread band, gain from a higher
   locked coupon (gross-interest paths, not fewer taxable crossings) → **`REAL_LOCKIN_EDGE`**, quantified,
   with `worst_case_scenario_id` labeling the bps and the scenario-dependence caveat.
4. Advantage survives only via tax-smoothing → **`TAX_SMOOTHING_EDGE`**.
5. Edge appears only on a modeled path that `CURVE_IMPLIED` rules out → **`REGIME_BET_NOT_EDGE`**
   (directional rate forecast, not alpha).
6. Verdict sign flips within the spread band, or `scenario_set_degenerate` → **`NO_ROBUST_EDGE`**.

---

## 6. Module / file plan (reuse vs net-new)

| File | New/Edit | Reuse vs net-new |
|---|---|---|
| `src/finalayze/orchestration/deposit_ladder.py` | **NEW (L5)** | **net-new:** candidate enumerator + ranking driver + scenario-roll wrapper + curve-implied derivation + spread-band sweep + fail-closed loader + lock-in assessor. **Reuses:** `DepositSimulatedBroker` (accrue/roll/break/deposit_value/income+tax props), `split_across_banks`/`uninsured_excess`, `ndfl_on_deposit_interest` (indirect via accrue) + `ndfl_marginal` (post-hoc caveat only), `ASV_CAP_PER_BANK`/`DEPOSIT_DEMAND_RATE`/`NDFL_PROGRESSIVE_THRESHOLD`, `DepositTranche`/`BankAllocation` schemas, `deposit_rate_as_of` (REALIZED + NDFL floor), `ConfigurationError`. |
| `src/finalayze/orchestration/data/deposit_term_structure.json` | **NEW (committed fixture)** | operator-curated real offered rates + REQUIRED signed per-term roll spreads + key-rate scenarios + provenance (`as_of`/`git_sha`/`source`). Mirrors `backtest/data/allocation_gate_snapshot.json`. |
| `src/finalayze/core/constants.py` | **EDIT (additive)** | add `MAX_OFFER_STALENESS_DAYS = 14` (+ optional `ASV_RAISED_CERT_1_3Y`/`ASV_RAISED_CERT_OVER_3Y`). |
| `scripts/recommend_deposit_ladder.py` | **NEW (token-free)** | thin CLI: load snapshot → optimize → print ranked ladder + lock-in verdict + caveats. No token, no DB, no `--mode sandbox/live`, no `--confirm`, no submit. |
| `tests/unit/test_deposit_ladder.py` | **NEW** | TDD (failing-test-first). See §8. |
| *(deferred, optional)* `api/v1/saa.py` + `dashboard/pages/saa_allocation.py` | — | additive read-only route + render fn — **only if/when those modules exist** (they do NOT in this worktree). |

**Untouched (zero blast radius):** `deposit_broker.py`, `orchestration/allocation.py`,
`rebalance_planner.py`, `allocation_gate.py`, `run_rebalance.py`, `ndfl.py`, the frozen gate/cert,
and the real-money path.

### Recommendation-only safety boundary (3 structural layers)

1. **Inert result.** `LadderPlan`/`RankedLadder`/`SimResult`/`RegimeLockinReport` hold only `Decimal`
   metrics + weight maps — **no `OrderRequest`, no `PlannedLeg`, no broker handle, no `submit()`.**
   (Stronger than `RebalancePlan`, which at least holds order requests.)
2. **No execution-coupling imports.** The module imports `DepositSimulatedBroker` (pure-arithmetic
   *simulated* broker, no I/O on the accrue/roll/break/value surface), `split_across_banks`/
   `uninsured_excess`, L0 `schemas`/`constants`/`ndfl`, `ConfigurationError`, and `deposit_rate_as_of`
   (for `REALIZED` + NDFL floor) **only**. Module-docstring invariant (mirrors `rebalance_planner` L-01):
   *"Imports the SIMULATED deposit broker by symbol only; NEVER imports a live broker, router, channel,
   token, or session; NEVER calls any `BrokerBase` order method (`submit_order`/`cancel_order`/
   `get_portfolio`); only drives `DepositSimulatedBroker` for simulation. No real-money path exists here."*
   `DepositSimulatedBroker` subclasses `SimulatedBroker(BrokerBase)`, so it INHERITS a live-shaped
   `submit_order`/`cancel_order` surface — in-memory only — but the wrapper never invokes it (asserted
   by a negative-call test, §8 T13).
3. **Read-only token-free surfaces.** `scripts/recommend_deposit_ladder.py` — loads the committed
   snapshot, runs `optimize_deposit_ladder`, prints the ranked ladder + lock-in verdict + caveats.
   **No `--mode sandbox/live`, no `--confirm`, no token, no DB.** (Optional, deferred: additive
   `GET /api/v1/saa/deposit-ladder` + `render_deposit_ladder` dashboard block — only if/when the SAA
   API/dashboard modules exist; they do NOT in this worktree.)

The real-money hard stop is preserved by construction: the only path that ever touches a real broker
remains `scripts/run_rebalance.py`; this tool never imports it.

### Break recommendations

Default = **never break a live tranche** (a break collapses accrued interest to the demand rate,
verified break_tranche:159–175). A break is surfaced ONLY when a forward-looking sim shows
`terminal(break) − terminal(hold) > 0` net of forfeited interest (essentially only in a *rising*-rate,
low-accrued case), OR to meet a `min_liquid_fraction` cash need — and then as an **honest cost line**
(`liquidation_floor`, forfeited-interest RUB), preferring a shorter-rung ladder up front over breaking.
The break's tax effect flows through the sim (resetting accrued interest reduces that year's taxable
deposit interest) — never a separate tax adjustment.

### SAA wire-up — documented, NOT built

Do not wire into `AllocationOrchestrator`/`plan_rebalance` — that path is frozen and gate-measured, and
its deposit leg is a single mark-only `ManualAction`; injecting tranche detail would invalidate the cert.
The honest future seam: `LadderPlan.recommended.candidate.weights` is exactly the split an operator would
type into the deposit `ManualAction` — a future rebalance preview could *display* it read-only beside
"place X RUB on a bank deposit." One-line note in the module docstring; costs nothing now.

### Key reused signatures (verbatim, verified against source)

- `DepositSimulatedBroker.__init__(self, initial_cash: Decimal, tranches: list[DepositTranche], tax_rate: Decimal = NDFL_RATE)` (deposit_broker.py:70)
- `.accrue(current_date: date) -> Decimal` — daily net accrual, NDFL R-2 floor internal, **broker-level YTD pool**, idempotent per date (deposit_broker.py:90; floor read :123; YTD :117/:143; idempotency :110–113)
- `.roll_at_maturity(tranche, current_date) -> DepositTranche` — rolls at `deposit_rate_as_of(current_date)`; **the driver substitutes the scenario rate by rebuilding the tranche** per :187–195 (deposit_broker.py:177)
- `.deposit_value() -> Decimal` (:150); `.interest_income_net`/`.interest_income_gross`/`.tax_paid` (:198/:203/:208)
- `.break_tranche(tranche, current_date) -> None` — resets `accrued_net = principal × DEPOSIT_DEMAND_RATE` (:159, :175)
- `split_across_banks(total_principal, cap=ASV_CAP_PER_BANK) -> list[BankAllocation]` (:213); `uninsured_excess(bank, cap=ASV_CAP_PER_BANK) -> Decimal` (:234) — pass the **tier-resolved cap**
- `deposit_rate_as_of(as_of, spread_pp=_DEFAULT_DEPOSIT_SPREAD_PP=Decimal("1.0"))` — synthesizes one rate per date, returns `Decimal(0)` before the first meeting (cbr.py:782; constant :778). **Used ONLY for the REALIZED scenario + the NDFL floor; NEVER for offered rates.**
- `ndfl_on_deposit_interest(gross, ytd_deposit_gross_before, running_floor) -> Decimal` — **flat 13%** (ndfl.py:84); `ndfl_marginal`/`YtdTaxAccumulator` (ndfl.py:32/:46, threshold `NDFL_PROGRESSIVE_THRESHOLD=2.4M`) = the cross-sleeve progressive path used only post-hoc for the `progressive_band_caveat` lower-bound estimate
- `ConfigurationError` — `from finalayze.core.exceptions import ConfigurationError` (exceptions.py:19)

---

## 7. Acceptance criteria (testable)

1. **A1 — fail-closed loader.** `load_term_structure` raises `ConfigurationError` (citing the field) on:
   missing/corrupt file; a requested `allowed_term` with no offer; an offer missing `roll_spread_pp`
   (never falls back to `_DEFAULT_DEPOSIT_SPREAD_PP`/1.0pp); a non-positive `annual_rate`; a stale
   `as_of` (`today − as_of > MAX_OFFER_STALENESS_DAYS`), citing the age in days.
2. **A2 — no synthetic term structure.** No code path produces a per-term offered rate from
   `deposit_rate_as_of`; offered rates come only from the committed snapshot / operator request.
3. **A3 — REALIZED + CURVE_IMPLIED invariants.** `rank_ladders` raises `ConfigurationError` if the
   scenario set lacks a `is_realized_anchor=True` leg; `assess_lockin` raises if it lacks a
   `is_curve_implied=True` leg. The curve-implied leg is derived from `offers`, not stored as literals.
4. **A4 — falsifiability.** A scenario set where every scenario agrees on the lock-in sign yields
   `scenario_set_degenerate=True` and demotes any edge verdict to `NO_EDGE_CURVE_PRICES_CUTS`.
5. **A5 — curve-prices-cuts on the fixture.** On the committed (inverted) fixture, the verdict under
   `CURVE_IMPLIED` is `NO_EDGE_CURVE_PRICES_CUTS`, and flipping the short roll-spread band does NOT
   silently produce `REAL_LOCKIN_EDGE` (sign-stable → no spurious edge).
6. **A6 — inverted-curve → LIQUIDITY_COST.** On a clearly inverted curve where the long leg loses in
   ≥1 scenario, the verdict is `LIQUIDITY_COST`, quantified in bps, recommending short.
7. **A7 — flat-curve robustness changes the answer.** On a near-flat curve where `mean_s` and `min_s`
   disagree, the recommendation is flagged `path_fragile`, the lock-in bps carries its
   `worst_case_scenario_id` label, and the `min_s` gate prevents a mean-strong / worst-catastrophic
   ladder from ranking #1 (overfitting fix c+e).
8. **A8 — single-broker one-allowance-pool.** A ladder of N tranches consumes ONE YTD tax-free
   allowance pool (broker-level), not N — terminal value matches a single-broker simulation and differs
   from the (wrong) per-tranche-pool computation.
9. **A9 — ASV tier Minfin-verified.** `asv_tier_cap` returns `1.4M` for `deposit` at 36mo (=3yr,
   boundary stays STANDARD); `2.0M` for `deposit` at 60mo (>3yr); `2.0M` for `irrevocable_cert` 1-3yr;
   `2.8M` for `irrevocable_cert` strictly >3yr. ASV is reported (`uninsured_at_horizon`), never used to
   exclude a candidate. (Soft metric; raised-deposit-tier effective date carries a re-verify caveat.)
10. **A10 — progressive band cross-sleeve lower bound.** `progressive_band_caveat` fires when
    `ytd_other_taxable_income + deposit YTD taxable interest` crosses 2.4M in some year, and does NOT
    fire at the operator's ~2.5M capital with `ytd_other_taxable_income=0` (immaterial-but-honest).
11. **A11 — recommendation-only scope.** `optimize_deposit_ladder` returns a `LadderPlan` containing no
    `OrderRequest`, no `PlannedLeg`, no broker handle; the import graph contains none of
    `{alpaca_broker, tinkoff_broker, broker_router, BrokerRouter, AsyncClient, AsyncSandboxClient}`;
    the wrapper never calls `submit_order`/`cancel_order` on its broker.
12. **A12 — deterministic + provenanced.** Running the optimizer twice on the committed fixture yields
    byte-identical metrics; `snapshot_provenance` records path + `as_of` + `git_sha`. No live HTTP fetch
    occurs (`deposit_rate_as_of` resolves the committed calendar).
13. **A13 — green gates.** `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy src/`
    stay green; the new tests pass.

---

## 8. Ordered TDD test list (RED-first) — `tests/unit/test_deposit_ladder.py`

Each test is written failing-first, then made green. Anti-hollow tests are driven by the REAL committed
fixture, not literals invented in the test body.

1. **T01 — loader happy path (RED).** `load_term_structure()` on the committed fixture returns a
   `TermStructure` with 4 offers, each carrying an explicit signed `roll_spread_pp`, plus `as_of`,
   `source`, `git_sha`. (A2)
2. **T02 — fail-closed: missing roll_spread_pp.** A fixture variant with one offer missing
   `roll_spread_pp` → `ConfigurationError` citing the field; asserts the loader never substitutes
   `_DEFAULT_DEPOSIT_SPREAD_PP` / 1.0pp. (A1, E3)
3. **T03 — fail-closed: missing allowed term.** Request `allowed_terms` includes a term with no offer →
   `ConfigurationError`. (A1)
4. **T04 — fail-closed: stale as_of.** `as_of` older than `MAX_OFFER_STALENESS_DAYS` → `ConfigurationError`
   citing the age in days. (A1)
5. **T05 — fail-closed: non-positive / corrupt rate / missing key.** Each → `ConfigurationError`. (A1)
6. **T06 — REALIZED invariant.** A scenario set without `is_realized_anchor=True` → `rank_ladders`
   raises `ConfigurationError`. (A3, E4)
7. **T07 — CURVE_IMPLIED derivation + invariant.** `derive_curve_implied_scenario` returns a scenario
   under which `EATV(ALL_LONG) ≈ EATV(ALL_SHORT)` within `noise_bps` on the fixture; a set without it →
   `assess_lockin` raises `ConfigurationError`. (A3, E5)
8. **T08 — falsifiability / degenerate set.** A scenario set where every scenario agrees on the lock-in
   sign → `scenario_set_degenerate=True` and any edge verdict demoted to `NO_EDGE_CURVE_PRICES_CUTS`. (A4, E6)
9. **T09 — no-edge-when-curve-prices-cuts (ANTI-HOLLOW).** On the committed inverted fixture, the verdict
   under `CURVE_IMPLIED` is `NO_EDGE_CURVE_PRICES_CUTS`; flipping the short roll-spread band
   (`widen_pp`/`tighten_pp`) does NOT flip it to `REAL_LOCKIN_EDGE`. (A5)
10. **T10 — inverted-curve → LIQUIDITY_COST.** A clearly inverted curve where ALL_LONG loses in ≥1
    scenario → verdict `LIQUIDITY_COST`, bps quantified, recommend short; `curve_inverted=True`,
    `curve_slope_bps < 0`. (A6)
11. **T11 — flat-curve robustness (the case the inverted fixture hides).** A near-flat curve where `mean_s`
    and `min_s` disagree → recommendation flagged `path_fragile`, lock-in bps labeled with
    `worst_case_scenario_id`, and the `min_s` gate prevents a mean-strong/worst-catastrophic ladder from
    ranking #1. (A7)
12. **T12 — spread-band sign flip → NO_ROBUST_EDGE.** A curve where the verdict sign flips within the
    spread band → `spread_sign_flips=True`, verdict `NO_ROBUST_EDGE`. (A5, H4)
13. **T13 — single-broker one-allowance-pool invariant (ANTI-HOLLOW).** A multi-tranche ladder's
    terminal value equals a single-broker reference run and differs from a (wrong) per-tranche-pool
    computation — proving one YTD allowance is shared across tranches. (A8)
14. **T14 — ASV tier Minfin-verified.** `asv_tier_cap`: `deposit`@36mo → 1.4M (boundary stays STANDARD);
    `deposit`@60mo → 2.0M (>3yr raised tier); `irrevocable_cert` 1-3yr → 2.0M; `irrevocable_cert` >3yr →
    2.8M. ASV never excludes a candidate. (A9)
15. **T15 — progressive band cross-sleeve lower bound.** With `ytd_other_taxable_income` seeded high
    enough that the cross-sleeve total crosses 2.4M, `progressive_band_caveat=True`; at operator scale
    with `ytd_other_taxable_income=0`, it stays `False`. (A10)
16. **T16 — REGIME_BET_NOT_EDGE.** A scenario set where ALL_LONG wins ONLY on a path `CURVE_IMPLIED`
    rules out → verdict `REGIME_BET_NOT_EDGE`, `edge_source="regime_bet"`. (decision rule 5)
17. **T17 — import-graph / scope lock (ANTI-HOLLOW).** Walk the new module's transitive imports (or assert
    on a hardcoded set) and FAIL if any of `{alpaca_broker, tinkoff_broker, broker_router, BrokerRouter,
    AsyncClient, AsyncSandboxClient}` appears. Assert `LadderPlan` and nested types expose no `OrderRequest`/
    `PlannedLeg`/broker handle. (A11)
18. **T18 — negative-call guard (ANTI-HOLLOW).** Monkeypatch `DepositSimulatedBroker.submit_order` and
    `.cancel_order` to raise `AssertionError`; run `optimize_deposit_ladder` end-to-end on the fixture and
    assert it COMPLETES — proving the order surface is never touched. (A11)
19. **T19 — no live HTTP / deterministic.** Run the optimizer twice on the committed fixture → byte-identical
    metrics; assert (via monkeypatch / no-network) that `deposit_rate_as_of` resolves the committed calendar
    with no live `httpx` fetch on the recommendation path. (A12)
20. **T20 — provenance stamped.** `LadderPlan.snapshot_provenance` contains the snapshot path, `as_of`,
    and `git_sha`. (A12, E2)
21. **T21 — CLI smoke (token-free).** `scripts/recommend_deposit_ladder.py` runs on the committed fixture,
    prints the ranked ladder + verdict + caveats, exposes no `--mode sandbox/live`, no `--confirm`, no token,
    no DB. (A11)

---

### Verified-fact appendix (paths + lines, for the implementer)

- Broker lives in `src/finalayze/execution/deposit_broker.py` (NOT `orchestration/`) — original draft
  mis-pathed it; corrected here.
- `accrue` (deposit_broker.py:90): broker-level `self._ytd_deposit_gross` (:117, :143), Jan-1 reset
  (:116–119), floor via `deposit_rate_as_of(d, spread_pp=_ZERO_SPREAD_PP)` (:123), idempotent (:110–113).
- `roll_at_maturity` construction (:187–195) is the exact template the driver reproduces with the
  scenario rate.
- `break_tranche` resets `accrued_net = principal × DEPOSIT_DEMAND_RATE` (:175).
- `deposit_value` = sum(principal + accrued_net) (:157).
- `split_across_banks` (:213), `uninsured_excess` (:234).
- `ndfl_on_deposit_interest` flat 13% (ndfl.py:84); `ndfl_marginal` (:32) + `YtdTaxAccumulator` (:46)
  = cross-sleeve progressive path.
- `deposit_rate_as_of` (cbr.py:782) default `spread_pp=_DEFAULT_DEPOSIT_SPREAD_PP=Decimal("1.0")`
  (cbr.py:778) — the flat-1.0pp fabrication vector the loader must never reach for offered rates.
- `ConfigurationError` (core/exceptions.py:19); gate snapshot provenance fields `generated_at`/`git_sha`/
  `window` (backtest/data/allocation_gate_snapshot.json) — mirror for the new fixture.
- `ASV_CAP_PER_BANK=Decimal(1_400_000)` (constants.py:24), `DEPOSIT_DEMAND_RATE=Decimal("0.0001")`
  (constants.py:27), `NDFL_PROGRESSIVE_THRESHOLD=Decimal(2_400_000)` (constants.py:21).

---

## 9. Code-review fixes applied (post-implementation, adversarial pass)

An adversarial review (gsd-code-reviewer) found 4 BLOCKER + 4 WARNING defects; all fixed:

- **B1 (recommend ≠ verdict):** the ranker maximizes robust after-tax value over a cut-heavy
  scenario set and could recommend the exact long lock the lock-in report calls "a regime bet,
  not alpha." `LadderPlan.recommendation_caveat` now reconciles the two — when the verdict is
  `REGIME_BET_NOT_EDGE`/`NO_ROBUST_EDGE`/`LIQUIDITY_COST` and the recommendation locks long, it
  prints an explicit reconciliation and names the liquid all-short alternative's robust value.
- **B2 (tax caveat false-negative):** `progressive_band_caveat` was keyed off the lowest-interest
  REALIZED scenario, breaking the documented LOWER-BOUND guarantee. Now `any(...)` across the
  real-world scenarios (fires if any plausible path crosses the 2.4M band).
- **B3 (skipped open-day bias):** rolled tranches never accrued on their open day (originals do on
  `start`), losing ~1 day per roll — a directional bias (~26 bps > the 5 bps noise floor) that
  inflated the apparent lock-long edge. The driver now rolls BEFORE accruing, so coverage is
  continuous and the ALL_SHORT baseline is unbiased. Regression test asserts a rolling ladder ties
  a locked equivalent at the same flat rate.
- **B4 (dead honesty rail):** `TAX_SMOOTHING_EDGE` was defined but never returned, so a tax-driven
  win would misclassify as `REAL_LOCKIN_EDGE`. Now detected at the worst-case scenario (long edge
  driven by lower NDFL, not higher gross) and labeled a tax effect.
- **W1:** the "~0 as expected" breakeven note is now conditional — when the bisection clamps on a
  steep curve it states the clamp honestly instead of calling a large number "~0."
- **W2:** `min_liquid_fraction` is now ENFORCED (candidates failing it are dropped; an impossible
  constraint fails closed); the truly-dead `uninsured_tolerance_rub` field was removed.
- **W3:** `path_fragile` is documented as terminal-value dispersion (not a lock-decision
  robustness measure); the regime-bet warning now rides on `recommendation_caveat` regardless.
- **W4:** `locked_value_fraction` no longer keys off `id()` of discarded objects (id-reuse hazard);
  lockedness is tracked by stable index.

INFOs (I1–I4) accepted as documented simplifications: NDFL floor uses the real calendar under
hypothetical scenarios (H9, diff-neutral); committed snapshot `git_sha=null` (provenance carried by
`source`); ASV `min`-cap for mixed cert/deposit ladders (conservative, soft metric); the import-graph
guard is static (the runtime negative-call guard T18 covers the order surface).
