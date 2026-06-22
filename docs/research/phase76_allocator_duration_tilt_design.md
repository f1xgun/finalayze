# Phase 76 (v11.2 W1) — Allocator Redesign: Fixed-Coupon OFZ Duration + Regime Tilt

**Status:** DESIGN (consilium-approved, operator forks locked)
**Branch:** `gsd/phase-76-allocator-duration-tilt-redesign` (off `origin/main` @ c5847dc / #273)
**Skill gate:** `backtest-iteration` mandatory (bond-leg + weight-rule change).

---

## 1. Context & locked decisions

v11.1 closed with an honest **HARD_FAIL×3**: the frozen allocator does not beat a near-riskless
~18% deposit even in the easing regime. `regime_verdict_decision.md §4` named the precise economic
hypothesis for the easing loss: the bond sleeve is a **floater** (RUFLBITR) with **no price upside as
rates fall**. Phase 76 answers that deferred question on **real data** against the **best** duration
construction, then re-runs the FROZEN measurement-only gate.

Operator decisions (this session):
- **Strategic gate (build-or-stop):** `build all, deposit-anchored` — the product's value is
  regime-adaptive allocation ("mostly deposit at high rates, shift to equity/duration as rates fall").
  A HARD_FAIL in the current 16–21% regime is an acceptable, honest outcome, **not** a blocker.
- **OQ-A (bond leg):** `SWAP` — replace RUFLBITR floater with **RGBITR** (fixed-coupon OFZ
  total-return index). 3-leg shape preserved; zero hardwired-chokepoint relaxation.
- **OQ-B (tilt weights):** accept the proposed **deposit-anchored** table (§5).

## 2. Goal / Why / Expected results

**Goal:** swap the OFZ leg RUFLBITR→RGBITR (captures bond price appreciation as rates fall) + add a
**deterministic, look-ahead-safe regime tilt**, then re-run the strict-conjunctive allocation gate on
real net-of-tax data and record an **honest** per-regime cert. The deliverable is a working redesigned
leg + a reviewable real-data cert **whatever its verdict** — NOT a PASS.

**Why:** answers the `§4` deferred-redesign question against the strongest candidate, so the operator's
deposit-anchor decision is finally made vs the best duration construction, not a strawman floater.
Either easing finally PASSes (first regime-level PASS ever) or the deposit-anchor is confirmed.

**Honest expectation:** **most likely still HARD_FAIL.** The binding metric is risk-adjusted
(Sharpe ∧ Sortino ∧ MaxDD-cap) and the deposit's near-zero volatility at 13–22% nominal sets an
essentially unbeatable bar. Live probe: RGBITR easing **+18.3% gross** (vs floater ~16–19%,
deposit_net +13.6%) but only **+7.4% in high_rate** (vs deposit +22.5%) — duration adds upside ONLY
in easing and bleeds in high_rate, so **the tilt is load-bearing**. Worth building because the answer
(against the best candidate) is the input to the v11.2+ fork, and the build cost is small (one secid
constant + a fixed YAML tilt table + a per-boundary selector).

## 3. Scope

**In:** RGBITR secid swap + snapshot key rename; net RGBITR gross→net through the **existing**
`net_fixed_income_legs_interleaved` (stays 2-leg: deposit + OFZ); regime tilt as a **fixed table
lookup** in YAML selected per quarterly boundary by a look-ahead-safe CBR selector; refresh the
committed snapshot via a real RGBITR fetch (≥300 bars, one shared MCFTRR axis, clamp ≤ 2026-06-10);
re-run gate + record real per-regime verdicts / derived escalation / `n1_caveat`; before/after diff;
`backtest-iteration`; ruff + mypy green.

**Out:** any gate softening (cap/threshold/`>=`→`>`/AND→OR — `verdict_for_profile:269` frozen); any
optimizer/solver/search (tilt weights are fixed config, chosen on a documented thesis BEFORE the cert,
never fitted to the verdict); re-enabling active stock-picking (MCFTRR stays passive, SAA-04);
re-netting per regime slice (`_slice_leg` pure date filter, Phase-74 CR-01 contract); the **4-leg ADD**
variant (deferred — operator chose SWAP); statistical-robustness claims on the N=1 easing cycle.

## 4. Design — exact change sites (SWAP, 3-leg)

The orchestrator currently picks **one** `_LegWeights` per run (`run()` L399–401) and applies it at
every quarterly boundary (`_apply_allocation_and_rebalancing` L557–580). The tilt makes weight
selection **per-boundary** by regime. Backward-compatible: a profile with no `regime_weights` keeps
today's static behavior, so naive injected legs and the D-13 legacy path are untouched.

| # | File | Change |
|---|------|--------|
| C1 | `core/schemas.py:56` `AllocationProfile` | add optional field `regime_weights: dict[str, dict[AssetClass, Decimal]] \| None = None` (keys `high_rate`/`easing`). Default `None` ⇒ naive legs stay static. |
| C2 | `config/allocation_profiles.yaml` | add a `regime_weights:` block (high_rate + easing vectors) to each of the 3 real profiles; keep base `weights` (fallback + backward compat). |
| C3 | `config/allocation_profiles.py` loader | parse + validate the optional `regime_weights`: each (profile×regime) vector non-negative, all 3 classes present, sums to exactly `Decimal('1.0')`, fail-closed `ConfigurationError`; no renormalization; no solver import. |
| C4 | `data/fetchers/cbr.py` | add `rate_regime_as_of(as_of) -> str` (look-ahead-safe): `"easing"` if `get_last_cbr_decision(as_of).decision == "cut"` else `"high_rate"`. Reuses the existing meeting calendar (anti-pattern-5: no new fetcher). |
| C5 | `orchestration/allocation.py` | new `_target_weights(when, static_weights, *, legacy_cadence)`: legacy⇒static; `regime_weights is None`⇒static; else select `regime_weights[rate_regime_as_of(when)]`. In the boundary block compute `boundary_weights = self._target_weights(d, weights, legacy_cadence=...)` and drive `target_dep/ofz/eq` + `_should_rebalance` off it. |
| C6 | `scripts/run_allocation_gate.py:143,167` | `_OFZ_SECID "RUFLBITR"→"RGBITR"`; `_SNAP_LEG_OFZ "ofz_ruflbitr_net"→"ofz_rgbitr_net"`. |
| C7 | `backtest/allocation_gate.py` | snapshot leg-key constant `_SNAPSHOT_LEG_KEYS` / `_SNAP_LEG_OFZ` rename `ofz_ruflbitr_net→ofz_rgbitr_net`; `net_fixed_income_legs_interleaved` stays **2-leg**. No verdict/tighten/regime_split change. |
| C8 | `src/finalayze/backtest/data/allocation_gate_snapshot.json` | regenerate via real RGBITR `--refresh-snapshot` (T-6); new `ofz_rgbitr_net` leg, same shared axis. |

**Untouched invariants (must stay byte-identical in behavior):** `AssetClass` (3 members),
`tighten()` (drains EQUITY→DEPOSIT, OFZ flat), `verdict_for_profile` (Sharpe∧Sortino∧MaxDD, inclusive
`>=`, WR-02 sentinel-tie fails), `gate_with_autotighten` (single freeze), `regime_split` /
`regime_verdicts` (slice already-netted curves via `_slice_leg`, **no re-net**), `derive_escalation`
(deposit_anchor_vs_redesign IFF both regimes HARD_FAIL), `n1_caveat` (always-on separate metadata).

## 4a. Regime-selector decision (C4)

Rule: **first-cut trigger** — `easing` once the most recent CBR decision on/before `as_of` is a `cut`,
else `high_rate`. Rationale: (1) **look-ahead-safe** — only reads meetings ≤ as_of (test: `2025-06-05`
→ `high_rate`, the `2025-06-06` first cut is in the future); (2) **aligns** the candidate's easing
window with the gate's `REGIME_SPLIT_BOUNDARY = 2025-06-06`, so the cert reads "in the easing
sub-window the candidate held easing weights"; (3) **forward-general** for the live product (no magic
date). Alternative `is_cutting_cycle` (2 consecutive cuts) lags the gate boundary and muddies the
easing read; noted as a future tuning lever, not used now.

## 5. Locked tilt table (OQ-B) — `(deposit / ofz_rgbitr / equity)`, Σ=1.0

| profile | high_rate (16–21%) | easing (rates ↓) | MaxDD cap |
|---|---|---|---|
| conservative | 0.75 / 0.10 / 0.15 | 0.45 / 0.35 / 0.20 | 0.08 |
| balanced | 0.60 / 0.10 / 0.30 | 0.25 / 0.40 / 0.35 | 0.15 |
| growth | 0.40 / 0.10 / 0.50 | 0.10 / 0.40 / 0.50 | 0.25 |

Thesis: high_rate ⇒ deposit dominates, duration minimal (RGBITR bleeds at high rates); easing ⇒ shift
into OFZ duration + equity. Frozen BEFORE the cert (never fitted to the verdict).

## 6. TDD subtasks

- **T-1 RED:** OFZ fetch secid is RGBITR (not RUFLBITR), routes through `net_fixed_income_legs_interleaved`; MCFTRR still never netted (Pitfall-1); `load_mcftr_series('RGBITR')` ≥300 bars, CLOSE in ~600–700 TR range (not ~110 price-only RGBI), fail-closed on short fetch.
- **T-2 GREEN:** C6 + C7 secid/key rename; make T-1 green.
- **T-3 RED+GREEN:** `rate_regime_as_of` look-ahead-safe (2025-06-05→high_rate); regime tilt is a table lookup with **zero trainable params** (two runs byte-identical); C1/C2/C4/C5.
- **T-4 RED+GREEN:** loader validates every (profile×regime) vector sums to exactly 1.0, non-negative, 3 classes; `ConfigurationError` on 0.95/1.05, no renormalization; C3.
- **T-5 RED:** invariant regressions — `verdict_for_profile` unchanged (incl. WR-02 sentinel, TRAP-A %→frac); `gate_with_autotighten` single-freeze (a persistent breach is a binding HARD_FAIL); `regime_verdicts` slices the **tilted** netted curve via `_slice_leg`, never instantiates `YtdTaxAccumulator`; `net_fixed_income_legs_interleaved` called exactly once at full-window creation.
- **T-6:** real RGBITR `--refresh-snapshot` (≥300 bars ~620, one shared MCFTRR axis via `_forward_align`, clamp ≤ 2026-06-10, fail-closed); commit the fixture; deterministic offline replay reproduces the verdict byte-identically.
- **T-7 anti-hollow:** per-regime verdicts + `derive_escalation` + `n1_caveat=true` produced by the REAL frozen path (greppable from the JSON cert, **not** a literal/hook); `rebalance_cost>0` and reconciled NDFL FIRE on the real run (Phase-72 hollow-GREEN lesson).
- **T-8:** before/after RUFLBITR-static vs RGBITR-tilt per-regime table → `docs/research`; `backtest-iteration`; ruff + mypy green; cert recorded verbatim (HARD_FAIL acceptable & shippable).

## 7. Success criteria (phase DONE regardless of pass/fail)

A recorded, honest, offline-reproducible per-regime cert (full_window/high_rate/easing × 3 profiles)
with a DERIVED escalation and `n1_caveat`, against the RGBITR-tilt candidate, with all §4 invariants
proven unchanged by tests, `backtest-iteration` run, lint/type green, and a reviewable before/after
diff. PASS/HARD_FAIL are both complete, shippable outcomes — the phase MUST NOT require beating the
deposit to be done. `exit 0` = artifacts written; nonzero = harness failure only.

---

## 8. OUTCOME (post-execution) — duration REJECTED, tilt KEPT

The §1 plan was a SWAP (RGBITR duration). Execution answered the deferred question on real data and
the answer changed the deliverable: **the duration swap is falsified; the regime tilt is kept on the
RUFLBITR floater.** Shipped allocator = **floater + regime tilt** (operator decision, this session).

### 8.1 Ablation (single-harness 2×2, candidate Sharpe — balanced, full / high_rate / easing)

|                | static                 | tilt                   |
|----------------|------------------------|------------------------|
| floater (RUFLBITR) | −0.847 / −0.80 / −0.86 | −0.859 / −0.78 / −0.90 |
| duration (RGBITR)  | −1.002 / −1.01 / −0.91 | −0.971 / −0.89 / −1.02 |

- **DURATION is the robust culprit:** RGBITR costs **~0.11–0.16 Sharpe vs the floater in every cell**,
  every regime — more volatile, uncompensated in this 16–21%→easing window (equity *fell*). Falsified.
- **The TILT is a full-window wash but directionally right:** it HELPS in high_rate (more deposit — the
  deposit-anchoring direction) and HURTS in easing only because this **single (N=1)** easing cycle had
  *falling* equity; in a typical easing it would help. Thesis-correct → kept as the product's
  regime-adaptive mechanism.

### 8.2 CRITICAL bug found + fixed — the tilt never reached the binding gate

The gate built its candidate via `gate_with_autotighten(base_weights=profile.weights, …)` — passing
ONLY the static base weights. On a failing candidate it fell through to the auto-tighten path and
returned the **static** re-gate, discarding the (uncomputed) tilt. So **every cert measured a static
allocator**; only the standalone ablation exercised the tilt. Fix: thread `regime_weights` through
`_naive_orchestrator → _run_and_score → gate_with_autotighten → regime_verdicts → run_gate`, and a
failing **tilted** candidate is an honest HARD_FAIL with its REAL tilted metrics (a static
auto-tighten cannot de-risk a per-regime tilt; it never fires on the binding data anyway). Guarded by
`test_gate_candidate_applies_tilt_not_just_orchestrator`.

### 8.3 Binding cert (floater + tilt, real net-of-tax committed snapshot)

HARD_FAIL×3 — conservative **−0.9032** / balanced **−0.8589** / growth **−0.8215** vs best-naive
**−0.6506** (the near-vol-free deposit); per-regime high_rate AND easing HARD_FAIL; phase verdict
HARD_FAIL; escalation `deposit_anchor_vs_redesign`; `n1_caveat` true — all DERIVED from the REAL gate
path. The deposit anchor holds against the best floater+tilt construction. This is the honest,
shippable deliverable; large real capital only after a regime-level PASS (deferred).

