# ML Failure Root-Cause — Phase 70 Findings & Decision (MLDIAG-04)

**Date:** 2026-06-02
**Phase:** 70 (ML failure root-cause spike — diagnostic, NOT a retrain)
**Requirements:** MLDIAG-04 (synthesizes MLDIAG-01 / MLDIAG-02 / MLDIAG-03)
**Status:** findings/decision deliverable — operator review/sign-off pending

This is the spike's single deliverable: a ranked root-cause finding and ONE concrete
pursue/abandon recommendation, synthesized from the Plan 01–03 evidence.

> **Scope discipline (D-05 — read first).** This document **PROPOSES only**. It enables NO
> segment, ships NO model, applies NO gate change, and edits NO production code or constant.
> The "abandon" branch's ML-debt cleanup and any future "pursue" change are **SEPARATE future
> phases** — not this spike. `MOEX_MIN_PASSING_FOLDS_RATIO` (the highest curve-fit-risk lever)
> was **never edited** (guarded by `tests/unit/test_quality_gates.py::TestAuditedConstants`).
> No KEEP is fabricated: every verdict cited here is traceable to
> `docs/quality/ml_gate_truth_table.md` and the honest JSONL logs in `results/experiments/`.

**Legend:** `KEEP = overall_passed:true AND not force_saved` · `DISC = honest fail
(overall_passed:false)` · `FORCED = force_saved` (never shipped; structurally impossible in
the Phase-70 harness — it has no `--force-save` path).

---

## 1. Evidence per hypothesis (H1–H4)

All four root-cause dimensions (D-02) were probed. Numbers below are MEASURED — from the
Plan 01 H1 reporter (`scripts/diagnose_ml_data_sufficiency.py`), the Plan 02 codified H2
audit (`tests/unit/test_wf_gate_discrepancy.py` + `TestAuditedConstants`), and the Plan 03
honest E1–E5 experiment matrix (truth table + `results/experiments/*.jsonl`, 28 records, 0
force-saved).

### H1 — Data / sample sufficiency (leading hypothesis)

| Segment | symbols (live, post-Phase-66) | raw candles | triple-barrier samples | class balance | WF folds | binding sub-gate behaviour |
|---------|-------------------------------|-------------|------------------------|---------------|----------|----------------------------|
| ru_blue_chips | **SFIN only (1 symbol)** — Phase-66 liquidity selector collapsed the universe | 855 | 634 | 56.0% positive | 10 | accuracy clears only **2/10** folds (< 0.34 min-ratio) |
| ru_energy | 7 (LKOH/ROSN/NVTK/TATN/TRNFP/SIBN/RNFT) | ~855–861 each | 4512 | 52.0% positive | 10 | accuracy **2/10**, brier **1–2/10** |

- **ru_blue_chips is structurally crippled by breadth, not just depth.** The Phase-66
  liquidity selector resolves the live ru_blue_chips universe to **SFIN only** — a single
  symbol. 634 samples come from one instrument, so the model learns one stock's idiosyncratic
  path with no cross-sectional signal. This is a *breadth* deficiency the accuracy/Brier gates
  cannot be argued around: there is effectively one asset's worth of independent information.
- **ru_energy has ample depth (4512 samples, 7 symbols) and is NOT data-starved** in the
  Phase-62 "insufficient-data" sense (no <500-day skips, no zero-barrier symbols). Its failure
  is therefore a *signal*-quality failure, not a sample-count failure (see H3).
- **Small effective-N raises the accuracy bar by design.** The accuracy gate threshold is
  `min(0.50 + 2.5·sqrt(0.25/n_eff), 0.55 + 0.10·(1−exp(−n_eff/200)))` and the Brier gate has a
  two-regime small-N floor — both were ALREADY relaxed for MOEX-sized folds (H2). So the gate
  is not punishing small N unfairly; it is asking for a separation neither segment produces.
- **Context vs Phase-62 (3 folds):** the prior legitimate retrain saw `n_folds:3`; the
  Phase-70 harness produced **10 folds** on the same segments. More folds did not help — the
  binding sub-gates still clear far below the 0.34 (≈1/3) ratio, which corroborates that the
  constraint is signal, not fold count.

**H1 verdict:** real but *segment-specific*. ru_blue_chips fails primarily on **breadth**
(single-symbol universe); ru_energy is data-rich and fails elsewhere. H1 alone does not
explain both segments.

### H2 — Gate calibration / logic

- **RESOLVED — the 62-04 discrepancy is a REPORTING ARTIFACT, not a logic bug.** Codified as a
  permanent regression guard in `tests/unit/test_wf_gate_discrepancy.py`: the binding
  `evaluate_walk_forward` (per-gate fold-ratio gate, `quality_gates.py:278`) returns
  `overall_passed=False` while the NON-binding synthetic-binomtest BH
  (`scripts/training/walk_forward.py:557` `apply_bh_across_segments`, the
  `int(acc*n_folds*100)` fabrication at :581) can print `p=… [PASS]` to stdout. The verdict
  keys strictly off `overall_passed` (`legit_pass = overall_passed is True and not
  force_saved`) — the stdout `[PASS]` is informational noise that can never enable a model.
- **Audited constants verified and drift-guarded** (`TestAuditedConstants`): `BH_FDR=0.10`,
  `MOEX_MIN_PASSING_FOLDS_RATIO=0.34`, MOEX WF windows `(8,1,3,2)`, `MOEX_PURGE_GAP_DAYS=40` —
  all byte-identical to source; no production gate code modified in Plan 02 (`git diff` = test
  files only).
- **The "too strict" story is pre-empted.** The accuracy and Brier gates are ALREADY
  small-N-aware (smooth caps / two-regime floor added precisely so MOEX folds aren't held to
  an unreachable bar). The specific binding failures are accuracy and Brier clearing ≤ 2/10
  folds — i.e., the model is wrong/under-confident on most folds, not that the bar is unfair.
- **The one genuinely unsound element is NON-binding.** The synthetic-binomtest's fabricated
  `n_folds×100` trials is statistically improper, but it does not touch `overall_passed`, so
  "fixing" it would change zero verdicts. Flagging it as cleanup is honest; recalibrating it to
  flip a verdict would be tune-to-pass (forbidden, D-05).

**H2 verdict:** **NOT a cause.** The gate is doing its job; the discrepancy is explained and
guarded. No principled recalibration of the binding gate would make either segment honestly
pass on the current data (it would only lower the evidentiary bar).

### H3 — Feature signal / no-alpha

- **ru_blue_chips is sub-random.** Across E1 (baseline + ablation), E2 (feature subsets), E3
  (barrier horizon), E4 (ensemble weights), E5 (regularization), `best_acc` stays in
  **0.4056–0.4864 — clearly BELOW the 0.50 random floor.** A model that cannot beat a coin flip
  on its own (single-symbol) data has no separable predictive edge.
- **ru_energy is marginal, never separating.** `best_acc` hovers **0.5051–0.5344 — only just
  above 0.50** and never enough to clear any binding sub-gate. Accuracy clears 2/10 folds and
  Brier 1–2/10 regardless of lever.
- **E2 (feature noise) did not help.** Dropping single features (`month_cos`, `min_ret_20d`,
  `max_ret_20d`, `hist_vol_20`) moved `best_acc` only marginally and crossed no gate. Feature
  noise is not the binding constraint.

**H3 verdict:** **strongly supported.** Neither segment exhibits a separable, gate-crossing
edge; ru_blue_chips is below random and ru_energy is statistically indistinguishable from it.
This is the "no alpha" finding, corroborated (not chased) across a fixed bounded matrix.

### H4 — Labeling horizon + model class

- **E3 (triple-barrier horizon):** varied the one param `_TB_MAX_HOLD` (20→10, 20→40 bars) via
  the harness's own dataset build. Both shorter and longer horizons stayed DISC on both
  segments (ru_blue_chips 0.4655/0.4864; ru_energy 0.5051/0.5100). The 40-bar run lost one WF
  fold (9 vs 10) as the longer label window consumes more tail history — expected, not a defect.
  Horizon mismatch is not the binding constraint.
- **E4 (model class / ensemble weights):** Cat-heavy mixes (0.6–0.7) raised raw profit-factor
  (ru_energy pf 2.18 / 3.60) but `best_acc` stayed below the gate (0.478–0.530) and
  accuracy/Brier still cleared ≤ 2/10 folds. The raw-PF bump is regime luck on a handful of
  trades, not a calibrated edge. Ensemble overfit is not the savior.
- **E5 (hyperparameter regularization):** heavier regularization (XGB max_depth 5→3, 5→4) did
  NOT lift OOS accuracy (ru_blue_chips flat at 0.4112; ru_energy ~0.534) — no gate crossing.
  Regularization is not the binding constraint.

**H4 verdict:** **ruled out as a lever.** No labeling-horizon, model-class, or regularization
change crossed a binding sub-gate on either segment. H4 changes operate on top of H3's missing
signal and cannot manufacture one.

---

## 2. Ranking H1–H4 by evidence

Ordered strongest-supported root cause first:

1. **H3 — No separable predictive alpha (DOMINANT).** Both data-richest MOEX segments fail to
   produce a gate-crossing edge across every bounded lever: ru_blue_chips sub-random
   (0.41–0.49), ru_energy marginal (0.51–0.53). 22 honest experiments, 0 KEEP. This is the
   common, lever-invariant failure mode for both segments.
2. **H1 — Sample/breadth sufficiency (strong, segment-specific amplifier).** ru_blue_chips is
   collapsed to a single symbol (SFIN) by the Phase-66 liquidity selector — a breadth deficiency
   that compounds H3 into a sub-random result. ru_energy, by contrast, is data-rich (4512
   samples, 7 symbols) yet still fails, which is *why H3 outranks H1*: ample data did not rescue
   the signal. H1 explains the *severity* on ru_blue_chips but not the failure on ru_energy.
3. **H4 — Labeling horizon + model class (null lever).** Every H4 experiment (E3/E4/E5) stayed
   DISC. H4 is downstream of H3: with no signal to shape, horizon/model/regularization changes
   have nothing to amplify.
4. **H2 — Gate calibration/logic (NOT a cause; resolved).** The 62-04 discrepancy is a
   reporting artifact (codified test), the binding constants are sound and small-N-aware, and the
   one unsound element (synthetic-binomtest) is non-binding. Relaxing the gate would not produce
   an *honest* pass — it would only lower the bar (forbidden, D-05). H2 is ruled out.

---

## 3. Dominant root cause

**Dominant root cause: H3 — there is no separable, gate-crossing predictive alpha in the
current MOEX feature/label setup for either data-richest segment.**

- ru_blue_chips: **sub-random** (`best_acc` 0.41–0.49, below the 0.50 floor), and structurally
  starved of breadth (single-symbol SFIN universe — H1 amplifier).
- ru_energy: **marginal and non-separating** (`best_acc` 0.51–0.53) *despite* ample data (4512
  samples, 7 symbols), failing every binding sub-gate (accuracy 2/10, Brier 1–2/10).
- No bounded, principled lever — feature subset (E2), barrier horizon (E3), ensemble weights
  (E4), or regularization (E5) — moved either segment across a binding sub-gate. **22 honest
  experiments, 0 KEEP, 0 force-saved.**

The gate (H2) is correctly rejecting models that genuinely lack edge; the data depth (H1) is
sufficient on ru_energy yet still produces no signal; and the model/label levers (H4) have no
latent edge to amplify. The binding constraint is the **absence of predictive signal itself**.

---

## 4. Single recommendation — ABANDON (accept rule-based-only)

### Operational go/no-go bar (D-04, made explicit)

> **Pursue is justified ONLY if** at least one E1–E5 experiment produced an HONEST
> `overall_passed:true` with `force_saved:false`, **OR** there is a principled, bounded change
> with a *credible mechanism* to reach an honest pass **within the existing gate** (NEVER by
> relaxing the gate, NEVER a fabricated/overstated KEEP).

### Evaluation against the bar

- **No honest pass exists.** 0 of 22 Phase-70 experiments (and 0 of the seeded Phase-62/64
  verdicts) produced `overall_passed:true`. Every record is `discard` / `overall_passed:false`,
  `force_saved` absent.
- **No credible bounded mechanism exists.** ru_blue_chips is sub-random (you cannot regularize
  or re-horizon your way past a coin flip), and ru_energy is non-separating despite ample data —
  the levers most likely to help (feature subset, horizon, model class, regularization) were all
  tried and all stayed DISC. The only "path to pass" would be relaxing
  `MOEX_MIN_PASSING_FOLDS_RATIO` or the binding accuracy/Brier gates, which is exactly the
  tune-to-pass / curve-fit move D-05 forbids and `TestAuditedConstants` guards against.

**Decision: ABANDON the current MOEX ML-ensemble approach for these segments. Accept
rule-based-only trading. The walk-forward gate is working correctly; it is honestly reporting
that the present feature/label/model recipe has no edge on MOEX.**

### Deferred ML-debt cleanup — handed to a SEPARATE follow-up phase (NOT done here)

This recommendation enables nothing and cleans up nothing in this phase. The following items
are enumerated for a dedicated follow-up phase (the "abandon-branch cleanup", deferred per D-05
/ CONTEXT Deferred Ideas):

1. **Quarantine stale force-saved `.pkl` artefacts.** The pre-Phase-62 (Mar-7 / Mar-21)
   force-saved model files in `models/<segment>/` are inert (no preset enables `ml_ensemble`,
   so the loader never loads them) but should be moved/removed so they cannot be loaded by
   accident.
2. **Silence / remove the `ml_force_saved_artifact_loaded` warning path** once the stale
   artefacts are quarantined, so boot logs are clean.
3. **Mark all MOEX `ru_*` segments DISC** in the canonical status sources (truth table is
   already current; reconcile any preset/doc that still implies a latent enable).
4. **Define a revisit trigger** — the condition under which MOEX ML is reconsidered (e.g., a
   materially broader/deeper liquid universe that restores ru_blue_chips breadth, a genuinely
   new feature family with an out-of-sample edge hypothesis, or substantially more usable daily
   history). Absent such a trigger, do not re-run the matrix (no tune-until-pass).
5. **Track the existing GAPS entries** (`G-008` ML accuracy, `G-011` `ml_ensemble` disabled,
   `G-013` WF Sharpe) under the abandon decision rather than as open "to fix" debt.

> None of items 1–5 are performed in Phase 70. They are the scope of a future cleanup phase.

### Why not "pursue"

A pursue recommendation would have to name a single bounded change with a credible mechanism to
reach an honest pass. The evidence offers none: every principled lever was tried and stayed
DISC, ru_blue_chips is below random, and ru_energy is data-rich yet signal-poor. Manufacturing
a pursue case (e.g., re-broadening ru_blue_chips and *hoping* SFIN→multi-symbol restores an
edge) is speculative, not credible-mechanism-backed by this evidence — and it is precisely the
out-of-scope "ML approach change" deferred to a future phase. If breadth restoration is ever
pursued, it belongs in the revisit-trigger of the cleanup phase, gated on a *new* honest
experiment, not asserted here.

---

## 5. Experiment truth-table rows (reference)

All per-experiment honest verdicts cited above live in **`docs/quality/ml_gate_truth_table.md`**
(the living WF-gate truth source), reconciled in Task 1 against the JSONL logs:

- **Phase 70 experiment rows** — 22 one-knob experiments (11 per segment: E1×2, E2×3, E3×2,
  E4×2, E5×2), every row `overall_passed:false` / `force_saved:false` / `discard` / DISC. Plus
  the closing tally: *"Phase 70: 22 experiments, 0 honest KEEP / 22 DISC, 0 force-saved."*
- **Seeded prior verdicts** — Phase 62 (Stage-1 legitimate retrain: 0/17 `ru_*` enabled, 0
  force-saved) and Phase 64 (Stage-3 fundamental A/B Round-1: KEEP = NONE, 0/2).
- **Honest JSONL logs** — `results/experiments/ru_blue_chips_experiment_log.jsonl` (14 records)
  and `results/experiments/ru_energy_experiment_log.jsonl` (14 records); `force_saved` key
  absent from all 28 records (`grep -c '"force_saved": true' results/experiments/*.jsonl` = 0).
- **H1 evidence** — `scripts/diagnose_ml_data_sufficiency.py` (Plan 01) produces the per-symbol
  sample/fold/`n_effective` counts; the harness E1 fold/sample counts (10 WF folds;
  ru_blue_chips ~634 samples single-symbol SFIN; ru_energy ~4512 samples / 7 symbols) are the
  H1 measurement used in §1.
- **H2 evidence** — `tests/unit/test_wf_gate_discrepancy.py` + `TestAuditedConstants`
  (`tests/unit/test_quality_gates.py`) codify the reporting-artifact resolution and the audited
  constants (`BH_FDR=0.10`, `MOEX_MIN_PASSING_FOLDS_RATIO=0.34`, `overall_passed` semantics).

---

*Phase: 70-ml-re-enablement-decision-and-legitimate-retrain (reframed → ML failure root-cause spike)*
*Deliverable: MLDIAG-04 findings/decision doc — proposes only; enables nothing, ships nothing, recalibrates nothing.*
*Completed: 2026-06-02 (operator sign-off pending)*
