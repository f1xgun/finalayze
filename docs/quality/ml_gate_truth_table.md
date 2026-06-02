# ML Gate Truth Table — Phase 61 Stage 0 Audit

**Date:** 2026-05-30
**Phase:** 61 (Stage 0 of the v10.3 retrain milestone)
**Requirements:** RETRAIN-AUDIT-01, RETRAIN-AUDIT-02, RETRAIN-AUDIT-03

Stage 0 (Phase 61) audit reconciling each MOEX `ru_*` preset's `ml_ensemble.enabled`
flag against the model's recorded walk-forward gate status. **Disable-only: ML is never
enabled here.** Any segment whose model did not *legitimately* pass the gate (absent
model dir, absent or null/false gate status, or a force-saved artefact) has its
`ml_ensemble` disabled (`enabled: false`, `weight: 0.00`) in the preset YAML. This
document is a committed, reusable input for Stage 1 (Phase 62) retraining. It is distinct
from the gitignored execution `SUMMARY.md`.

## Legitimacy rule (fail-closed)

A model is treated as having legitimately passed the gate **only if all** hold:

1. `models/<segment>/` exists, and
2. `models/<segment>/wf_gate_results.json` exists, and
3. it has a truthy `gate_passed` **or** `overall_passed` is exactly `true`, and
4. it is **not** `force_saved` (a truthy `force_saved` ⇒ not legitimate).

Anything else — absent dir, absent file, absent/null gate keys, `overall_passed: false`,
or `force_saved: true` — is read as **"not passed"**, which (when `ml_ensemble.enabled`
is `true`) requires the preset to be disabled. This invariant is enforced in code by
`tests/unit/test_ml_gate_reconciliation.py::test_no_ru_preset_enables_ml_on_unpassed_model`
and resolves the models dir via the `FINALAYZE_MODELS_DIR` env override (default
project-root `models/`).

> **Live re-read note (2026-05-30):** values below were read from the primary checkout's
> real artefacts at `/Users/f1xgun/finalayze/models/`. The worktree itself has no `models/`
> tree; under the fail-closed rule absent-dir alone already mandates disable, so the
> invariant holds in either environment.

## Truth table (MOEX `ru_*` segments)

| segment       | model dir exists? | gate_passed | force_saved | verdict                      | ml_ensemble.enabled (before) | action taken                                                                 | ml_ensemble.enabled (after) |
|---------------|-------------------|-------------|-------------|------------------------------|------------------------------|------------------------------------------------------------------------------|-----------------------------|
| ru_blue_chips | yes               | absent      | absent      | NOT PASSED (`overall_passed: false`) | true (weight 0.10)           | **disabled**; `weight: 0.00`. `model_weights.json` is all-zero (`xgboost/lightgbm/catboost = 0.0`) — contributed nothing even while "on". normalize_mode `firing`, other weights left as-is. | false (weight 0.00)         |
| ru_tech       | **no**            | absent      | absent      | NOT PASSED (no artefacts)    | true (weight 0.25)           | **disabled**; `weight: 0.00`. No `models/ru_tech/` directory exists at all. normalize_mode `total`; remaining enabled weights rescaled proportionally (momentum 0.20→0.31, mean_reversion 0.35→0.54, event_driven pinned 0.15) so the declared budget re-sums to 1.00. | false (weight 0.00)         |
| ru_energy     | yes               | absent      | absent      | no recorded gate (v9.1 KEEP) | false (weight 0.00)          | **no edit** — already disabled. KEEP-but-disabled discrepancy documented below. | false (weight 0.00)         |
| ru_finance    | yes               | absent      | absent      | no recorded gate             | false (weight 0.00)          | **no edit** — already conservative (disabled, no gate file to justify enabling). | false (weight 0.00)         |
| ru_ofz_pd     | no                | n/a         | n/a         | n/a (bond preset)            | n/a — no `ml_ensemble` block | **no action** — bond preset declares no `ml_ensemble`.                        | n/a                         |
| ru_ofz_pk     | no                | n/a         | n/a         | n/a (bond preset)            | n/a — no `ml_ensemble` block | **no action** — bond preset declares no `ml_ensemble`.                        | n/a                         |

`gate_passed` / `force_saved` cells read **"absent"** where the key or the whole file is
missing — these are not fabricated as `true`/`false`. ru_blue_chips' `wf_gate_results.json`
records `overall_passed: false`, `bh_passed: false`, `n_folds: 3`, `best_accuracy: 0.471`
and carries no `gate_passed` / `force_saved` / `verdict` keys at all.

### US segments (informational only — out of scope)

US training is paused under the MOEX-first focus; these rows are read-only context and no
action was taken. All four have a `wf_gate_results.json` with `overall_passed: false` and
no `gate_passed` / `force_saved` keys.

| segment       | model dir exists? | gate_passed | force_saved | verdict                | action |
|---------------|-------------------|-------------|-------------|------------------------|--------|
| us_broad      | yes               | absent      | absent      | NOT PASSED (`overall_passed: false`) | out of scope / paused (MOEX-first) |
| us_finance    | yes               | absent      | absent      | NOT PASSED (`overall_passed: false`) | out of scope / paused (MOEX-first) |
| us_healthcare | yes               | absent      | absent      | NOT PASSED (`overall_passed: false`) | out of scope / paused (MOEX-first) |
| us_tech       | yes               | absent      | absent      | NOT PASSED (`overall_passed: false`) | out of scope / paused (MOEX-first) |

## ru_energy KEEP-but-disabled discrepancy

The v9.1 milestone recorded a **KEEP** verdict for `ru_energy` (its first KEEP verdict),
yet the live preset has `ml_ensemble.enabled: false` and `models/ru_energy/` contains
**no** `wf_gate_results.json` to justify re-enabling it under the current (tightened) gate.

There are two ways to read this discrepancy, and Stage 0 resolves it conservatively:

- The v9.1 KEEP predates the gate-tightening and stable feature-selection work; per the
  audit-corrected ML enablement status, all shipped artefacts were ultimately
  `--force-save`d with `gate_passed=False`. A KEEP under the *old* regime is not evidence
  of a pass under the *current* gate.
- With no recorded `wf_gate_results.json`, there is no machine-readable pass to trust.

**Conclusion:** leave `ru_energy` **disabled** in Stage 0 (D-02 disable-only — this phase
never enables ML). Re-enabling `ru_energy` is **deferred to Stage 1 (Phase 62)**, gated on
a legitimate retrain that writes a recorded passing `wf_gate_results.json`. No flip here.
`ru_finance` is in the same posture (disabled, no recorded gate) and is likewise left as-is.

## Backtest-iteration certificate

Per CLAUDE.md invariant #4, disabling `ml_ensemble` is a strategy-behaviour change and must
be cleared by a backtest-iteration on an affected segment. The gate here is **trade_count > 0**
(the disable did not break or fatally under-weight the preset); a REJECT verdict is acceptable
and reflects baseline MOEX economics, not a defect.

**Segment:** `ru_blue_chips` (affected — `ml_ensemble` disabled by this phase)

**Command:**

```bash
export GRPC_DNS_RESOLVER=native
export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=certs/grpc_roots.pem   # Phase-59 TLS unblock
# FINALAYZE_TINKOFF_TOKEN loaded from .env (operator-supplied; MOEX = Tinkoff gRPC only)
uv run python scripts/run_iteration.py --segments ru_blue_chips \
  --start-date 2024-01-01 --end-date 2024-12-31 \
  --name phase61-stage0-ru_blue_chips-ml-disabled \
  --description "Phase 61 Stage 0: ml_ensemble disabled on ru_blue_chips (gate not passed); prove preset still loads + trades" \
  --baseline none --output results/iterations
```

**history.jsonl record (verbatim):**

```json
{"name": "phase61-stage0-ru_blue_chips-ml-disabled", "created_at": "2026-05-30T17:55:18.685594+00:00", "git_sha": "ea8d6b997dbaeab3822da89bf811d1a4b68b3118", "verdict": "REJECT", "wf_sharpe": 0.0131, "wf_max_drawdown": 0.02, "trade_count": 12}
```

| field | value |
|-------|-------|
| name | phase61-stage0-ru_blue_chips-ml-disabled |
| git_sha | ea8d6b9 (clean) |
| verdict | REJECT |
| wf_sharpe | 0.0131 |
| wf_max_drawdown | 0.02 |
| **trade_count** | **12** |

**Interpretation:** Disable did not break or under-weight the preset — the combiner pool
loaded and produced trades (trade_count = 12 > 0; firing strategies observed:
`dual_momentum`, `mean_reversion`, `momentum`, `rsi2_connors`, with **no** `ml_ensemble`
in the pool, confirming the disable took effect). The REJECT verdict reflects baseline
MOEX economics, not a defect (consistent with the prior `phase60-ru_energy-wired` REJECT).
The Stage-0 gate criterion — preset still loads and trades after the ML disable — is met.

---

## Phase 62 — Stage 1 verdicts (legitimate retrain, no force-save)

**Date:** 2026-05-30
**Phase:** 62 (Stage 1 of the v10.3 retrain milestone)
**Requirements:** RETRAIN-01, RETRAIN-02, RETRAIN-03

Stage 0 (Phase 61, above) was **disable-only** and is preserved verbatim as the *before*
state. Stage 1 (this section) is the one place ML is permitted to be turned back ON — but
only for a segment whose model **legitimately passes** the walk-forward gate with **NO
`--force-save`**. Each of the 4 trainable `ru_*` segments was retrained through the fixed
pipeline (`triple_barrier --walk-forward --excess-returns --sequential-bootstrap`, **no
`--force-save`**); the 13 sector/bond segments have no `SEGMENT_SYMBOLS` mapping and are
recorded as "not trainable", not as failures.

**Honest framing (D-04):** success = legitimate verdicts recorded without force-save, even
if **zero** segments pass. v9.1 found ru_finance / ru_tech DISC due to data limitation and
ru_energy KEEP under the *old* (pre-tightening, ultimately force-saved) regime. The Stage-1
re-run produced **DISC for all four** under the current tightened gate. **Net result: zero
segments enabled; zero force-save debt remaining** for any MOEX segment.

## What un-degraded `--excess-returns` (the enabling fix)

Plan 62-02 found and fixed a blocking bug: `scripts/training/data_loader.py::fetch_moex_benchmark`
routed the **IMOEX index** through `TinkoffFetcher` (share-registry FIGI lookup), which does
not resolve a non-tradeable index → the benchmark fetch silently failed and the pipeline fell
back to **absolute returns** (Pitfall 2 — NOT the fixed pipeline). The fix routes IMOEX
through `MoexISSFetcher` (the MOEX-native ISS source the production `MarketDataLoader._load_moex`
already uses; distinct from yfinance, which is never used for MOEX). After the fix, every
Stage-1 run printed `Fetched 509 benchmark candles (IMOEX)` and computed genuine
**excess-return** triple-barrier labels — so all four verdicts below are honest fixed-pipeline
results (committed in `f55a920`).

## Stage-1 verdict table — all 17 `ru_*` segments

Columns: trainable (has `SEGMENT_SYMBOLS`)? · preset YAML? · in `run_iteration` UNIVERSE
(backtest-gateable)? · data source used · n_folds · `overall_passed` · `force_saved` ·
Stage-1 verdict · `ml_ensemble.enabled` before (Phase 61) · action taken · `ml_ensemble.enabled`
after · conservative weight (only if enabled).

| segment | trainable? | preset? | in UNIVERSE? | data source used | n_folds | overall_passed | force_saved | Stage-1 verdict | enabled (before) | action taken | enabled (after) | cons. weight |
|---------|-----------|---------|--------------|------------------|---------|----------------|-------------|-----------------|------------------|--------------|-----------------|--------------|
| **ru_blue_chips** | yes | yes | yes | live Tinkoff gRPC (7/8 syms; TCSG skipped <500d) + IMOEX via MoexISS | 3 | **false** | false | **DISC** (model-quality; BH p=0.1632, best_acc 0.5307) | false (0.00) | retrained, no force-save; gate failed → no enable | false (0.00) | — |
| **ru_energy** | yes | yes | yes | live Tinkoff gRPC (6/6 syms) + IMOEX via MoexISS | 3 | **false** | false | **DISC** (model-quality; BH p=0.9984, best_acc 0.4167; v9.1 KEEP does NOT survive tightened gate) | false (0.00) | retrained, no force-save; gate failed → no enable | false (0.00) | — |
| **ru_finance** | yes | yes | yes | live Tinkoff gRPC (3/5 syms contribute; TCSG skipped <500d, VTBR 0 barriers) + IMOEX via MoexISS | 3 | **false** | false | **DISC** (mixed insufficient-data + model-quality; per-gate min_ratio 34% not met, best_acc 0.5297; gate-file bh_passed:false despite stdout BH p=0.0162) | false (0.00) | retrained, no force-save; gate failed → no enable | false (0.00) | — |
| **ru_tech** | yes | yes | **no** | live Tinkoff gRPC (2/4 syms: OZON 515c→197s, VKCO 606c→419s; YNDX 11c & CIAN 0c skipped <500d) + IMOEX via MoexISS | 3 | **false** | false | **DISC (train-only)** — not in UNIVERSE → not backtest-gateable; best_acc 0.3243, BH p=1.0000 | false (0.00) | retrained train-only, no force-save; **not enabled** (DISC + no UNIVERSE entry); preset untouched | false (0.00) | — |
| ru_metals | no | no | no | — | — | — | — | not trainable (no SEGMENT_SYMBOLS) | n/a | none | n/a | — |
| ru_consumer | no | no | no | — | — | — | — | not trainable (no SEGMENT_SYMBOLS) | n/a | none | n/a | — |
| ru_telecom | no | no | no | — | — | — | — | not trainable (no SEGMENT_SYMBOLS) | n/a | none | n/a | — |
| ru_utilities | no | no | no | — | — | — | — | not trainable (no SEGMENT_SYMBOLS) | n/a | none | n/a | — |
| ru_construction | no | no | no | — | — | — | — | not trainable (no SEGMENT_SYMBOLS) | n/a | none | n/a | — |
| ru_chemicals | no | no | no | — | — | — | — | not trainable (no SEGMENT_SYMBOLS) | n/a | none | n/a | — |
| ru_transport | no | no | no | — | — | — | — | not trainable (no SEGMENT_SYMBOLS) | n/a | none | n/a | — |
| ru_ofz_pd | no | yes (bond) | no | — | — | — | — | bond preset, n/a (no `ml_ensemble` block) | n/a | none | n/a | — |
| ru_ofz_pk | no | yes (bond) | no | — | — | — | — | bond preset, n/a (no `ml_ensemble` block) | n/a | none | n/a | — |

> The 13 non-trainable rows (7 sector segments + 2 bond presets, plus the 4 trainable rows
> = the full `DEFAULT_SEGMENTS` `ru_*` set) are classified for completeness. Bond presets
> (`ru_ofz_pd`, `ru_ofz_pk`) declare no `ml_ensemble` block at all (guarded by
> `test_ru_ofz_presets_have_no_ml_ensemble`); the 7 sector segments lack a `SEGMENT_SYMBOLS`
> mapping in `scripts/training/cli.py` and pass `[]` → "No samples, skipping".

## Per-segment notes

### ru_blue_chips → DISC (Plan 62-02)
Retrained; `overall_passed:false`, BH-corrected p=0.1632, best_accuracy 0.5307, n_folds 3.
7 of 8 symbols survived the 500-day gate (TCSG skipped — delisted/rebranded TCS→T, no FIGI).
Fold accuracies 0.563 / 0.535 / 0.531; accuracy & brier each cleared only 1/3 folds.
`ml_ensemble` stays disabled (was already `false` weight 0.00 from Phase 61). No preset edit.

### ru_energy → DISC (Plan 62-03)
The only segment with an asymmetric `SEGMENT_BARRIER_CONFIG = (1.5, 2.0)` (×1.2 MOEX uplift
⇒ (1.8, 2.4), applied automatically by the dataset builder — no CLI flag). Retrained;
`overall_passed:false`, BH-corrected p=0.9984, best_accuracy 0.4167, n_folds 3. All 6/6
symbols survived the 500-day gate → a genuine **model-quality** DISC, not insufficient-data.
**This is the headline reconciliation:** v9.1's sole KEEP predated the gate-tightening and was
ultimately force-saved with `gate_passed=False`; under the current gate ru_energy does **not**
pass. The Phase-61 "KEEP-but-disabled discrepancy" is now resolved as a legitimate DISC.
`ml_ensemble` stays disabled. No preset edit.

### ru_finance → DISC (Plan 62-04)
Retrained; `overall_passed:false`, per-gate min_ratio 34% not met (accuracy / brier_score /
profit_factor each cleared only 1/3 folds), best_accuracy 0.5297, n_folds 3. 3 of 5 symbols
contributed usable triple-barrier samples (SBER, MOEX, CBOM); **TCSG skipped** (<500d,
delisted) and **VTBR produced 0 triple-barrier samples** (illiquid kopeck-priced, near-flat
excess returns) → a **mixed insufficient-data + model-quality** DISC. Note: the stdout BH
block printed `p=0.0162 [PASS]`, but the authoritative gate file records `overall_passed:false`
AND `bh_passed:false`; `legit_pass` keys strictly off `overall_passed`, so the segment stays
DISC. `ml_ensemble` stays disabled. No preset edit.

### ru_tech → DISC, train-only (Plan 62-05, this plan)
**Train-only** by design: ru_tech has a preset and is trainable, but is **not in the
`run_iteration` UNIVERSE**, so it cannot be backtest-gated (RETRAIN-02 N/A). It is therefore
never a candidate for enable in Stage 1 and stays `ml_ensemble.enabled:false` (its Phase-61
state, which already satisfies the reconciliation invariant by staying disabled).
Despite its lead symbol **YNDX being delisted (now YDEX, 11 candles → skipped <500d)** and
**CIAN unavailable (0 candles → skipped <500d)**, the segment was **NOT** a no-samples skip:
the two surviving newer listings trained it —
**OZON** (515 candles → 197 triple-barrier samples, 69.5% positive) and
**VKCO** (606 candles → 419 triple-barrier samples, 41.3% positive),
616 market-neutral labels (50.3% positive). The model legitimately **FAILED** the gate:
`overall_passed:false`, `force_saved:false`, best_accuracy 0.3243, n_folds 3, BH p=1.0000.
Per-gate pass-rates: accuracy 33.3% [FAIL], brier_score 33.3% [FAIL], class_balance 33.3%
[FAIL], degenerate_predictor 33.3% [FAIL], profit_factor 66.7% [PASS], sensitivity 66.7%
[PASS], signal_count 100% [PASS], specificity 66.7% [PASS]. Fold accuracies 0.558 / 0.909 /
0.324 — the strong middle fold did not survive the min-ratio / BH correction.
**Deferred (out of this phase):** adding ru_tech to UNIVERSE and fixing the delisted
YNDX→YDEX symbol in `SEGMENT_SYMBOLS`. The preset, UNIVERSE, and symbol map were left untouched.

## Stage-1 conclusion (per D-04)

- **Segments enabled:** **0.** No model passed the tightened walk-forward gate.
- **Force-save debt remaining:** **0.** Every Stage-1 verdict was produced with NO
  `--force-save`; each `models/ru_*/wf_gate_results.json` records `force_saved:false`.
  The stale Mar-7/Mar-21 force-saved `.pkl` artefacts were **not** overwritten (gate failed,
  force-save not passed) and are inert — every preset keeps `ml_ensemble.enabled:false`, so
  the loader never loads them. Cleaning the stale `.pkl`s is a later housekeeping concern.
- **Reconciliation invariant:** still green — no preset enables `ml_ensemble` on an
  unpassed/force-saved model (`tests/unit/test_ml_gate_reconciliation.py`).
- **Loader legitimacy:** no enabled segment, so `ml_force_saved_artifact_loaded` cannot fire
  for any live ru_ segment.
- **Backtest-iteration:** no enable occurred, so no Stage-1 backtest-iteration was required
  (CLAUDE.md #4 fires only on an `ml_ensemble` enable). The only ru_ backtest-iteration on
  record is the Phase-61 Stage-0 `phase61-stage0-ru_blue_chips-ml-disabled` (REJECT,
  trade_count 12), documented above.

**This is the honest Stage-1 outcome and a VALID success per D-04:** legitimate per-segment
gate verdicts recorded across all 17 `ru_*` segments, with zero segments force-passed and
zero force-save debt remaining. The MOEX ML truth is now re-established honestly; any future
enable must earn a genuine gate pass.

---

## Phase 64 — Stage 3 Round-1 verdicts (fundamental-features A/B, no force-save)

**Date:** 2026-05-31
**Phase:** 64 (Stage 3 — fundamental features in ML, gated on data maturity)
**Requirements:** FUNDML-02, FUNDML-03

Stage 3 Round-1 asks a single causal question: **does real, now-populated fundamental DATA
move the walk-forward gate?** Plan 64-01 made the loader/feature slice carry fundamentals
(proven LIVE by its GREEN guard test — `compute_features` emits a non-zero `earnings_yield`
on a backfilled `as_of`). Round-1 changes **only the data** — same `label_mode`, WF params,
`--sequential-bootstrap`, `--excess-returns`, feature selection, and gate thresholds as
Phase 62. Two segments were retrained against the live `finalayze-db` (real MOEX candles via
Tinkoff gRPC — SBER=606 bars etc.) to a **separate** `models_fund/` dir with **NO
`--force-save`**; the Phase-62 baseline in `models/` was left UNTOUCHED (D-05/D-07/D-08).

**Honest framing (D-09):** confirming fundamentals do **not** move the gate is a valid,
valuable result and is recorded here either way. **Net Round-1 result: zero segments passed,
zero segments kept, zero preset edits, zero force-save debt.**

### Per-segment A/B — Phase-62 baseline (`models/`) vs Round-1 fundamental run (`models_fund/`)

| segment | baseline overall_passed / best_accuracy | fundamental-run overall_passed / bh_passed / best_accuracy | n_folds | key gate_pass_rates (fundamental run) | selected fundamental features | KEEP/DISC | force-save debt |
|---------|------------------------------------------|------------------------------------------------------------|---------|----------------------------------------|-------------------------------|-----------|-----------------|
| **ru_blue_chips** | false / 0.5307 | **false** / (stdout BH p=0.0033 [PASS]; `bh_passed` NOT persisted in gate file) / 0.5818 | 3 | accuracy 0.333, brier_score 0.333, profit_factor 0.333 (each < 0.34 min_ratio → FAIL); class_balance 1.0, degenerate_predictor 1.0, sensitivity 1.0, signal_count 1.0, specificity 1.0 | NOT written — model only saved on a gate pass; gate FAILED so `selected_features.json` was not emitted. Fundamental data path proven LIVE by 64-01 GREEN guard test, not from this artefact. | **DISC** | 0 (`force_saved: false`) |
| **ru_energy** | false / 0.4167 | **false** / **false** (stdout BH p=0.9935 [FAIL]) / 0.4310 | 3 | accuracy 0.0, brier_score 0.0 (FAIL); profit_factor 0.667, sensitivity 0.333; class_balance 1.0, degenerate_predictor 1.0, signal_count 1.0, specificity 1.0 | NOT written — gate FAILED (same as above). | **DISC** | 0 (`force_saved: false`) |

### KEEP rule (D-07) application

KEEP iff `overall_passed` flips false→true **AND** `bh_passed: true` (DISC→KEEP), or a
passing segment improves without regression. **Neither segment met the rule:**

- **ru_blue_chips** — `overall_passed` stayed **false**. `best_accuracy` improved
  0.5307 → 0.5818 and the standalone stdout BH printed `p=0.0033 [PASS]`, but the
  authoritative WF gate still **FAILS**: accuracy / brier_score / profit_factor each clear
  only 1/3 folds (pass-rate 0.333 < `MOEX_MIN_PASSING_FOLDS_RATIO` 0.34). Per Pitfall 4
  (only 3 MOEX folds), a marginal accuracy bump that does **not** flip the gate is **not a
  keep** — the gate, not the headline accuracy, is the arbiter. **DISC.**
- **ru_energy** — `overall_passed` stayed **false** and `bh_passed: false` (BH p=0.9935).
  accuracy / brier_score pass-rates are 0.0 (0/3 folds). Clear model-quality **DISC**.

### Data-maturity context (Assumption A1 — per-symbol `fundamental_snapshots` depth)

Operator DB query (`SELECT symbol, count(*), min(as_of), max(as_of) FROM fundamental_snapshots GROUP BY symbol`):

- **ru_blue_chips peers (7/8 have fundamentals):** SBER=10, LKOH=9, GMKN=9, ROSN=9,
  NVTK=7, MGNT=8, TATN=9, **TCSG=MISSING (0 rows)**.
- **ru_energy peers (6/6):** ROSN=9, TATN=9, NVTK=7, LKOH=9, SNGS=9, SIBN=9.
- Range ~2022-02 to ~2026-04; **7–10 annual/semi-annual snapshots per symbol (thin)**.

The ≥4-peer co-existence requirement for the z-score features (Pitfall 2) is satisfied for
both segments, so the DISC is **not** a "too few peers" artefact — it is a genuine
no-improvement result on thin (annual/semi-annual) point-in-time fundamental history.

### Round-1 conclusion (per D-09)

- **Segments kept (Round-1 KEEP):** **0.** Neither `ru_blue_chips` nor `ru_energy` flipped
  the gate false→true.
- **Preset edits:** **none.** Both `src/finalayze/strategies/presets/ru_blue_chips.yaml` and
  `ru_energy.yaml` keep `ml_ensemble.enabled: false`, `weight: 0.00` — revert by omission.
  No artefact was copied from `models_fund/` into `models/`.
- **Force-save debt:** **0.** Both Round-1 runs used NO `--force-save`; each
  `models_fund/<seg>/wf_gate_results.json` records `force_saved: false`. `models_fund/` is
  throwaway A/B evidence and is **not** committed.
- **Reconciliation invariant:** still green (`tests/unit/test_ml_gate_reconciliation.py`,
  5 passed) — no preset enables `ml_ensemble`, so the unpassed-model invariant holds trivially.
- **Loader legitimacy:** no enabled segment → `ml_force_saved_artifact_loaded` cannot fire;
  `tests/unit -k force_saved` green (6 passed).
- **Backtest-iteration:** no segment enabled → no backtest-iteration required (CLAUDE.md #4
  fires only on an `ml_ensemble` enable). `results/iterations/history.jsonl` is unchanged.

**This is the honest Round-1 outcome and a VALID result per D-09:** real fundamental data,
proven live in the feature path, did **not** move either WF gate past the threshold on the
current thin point-in-time history. **Round-1 KEEP = NONE (0/2 segments passed the WF gate.)**
This gates the conditional Plan 03 (Round-2 derived fundamental features) **OUT** — Plan 03
executes only if Round 1 kept ≥1 segment.

### Phase 64 Round 2 — deferred (per D-01)

Phase 64 Round 2 (derived multi-year fundamental features + schema `4`→`5`) is **deferred per
D-01** — Round 1 found no segment improvement (**KEEP = NONE**), so Round 2 was **not attempted**
in Phase 64. The conditional Plan 64-03 took its documented-skip branch: **no source under
`src/` was modified** and `FEATURE_SCHEMA_VERSION` **stays at `4`** (bumping to `5` with no
passing v5 artefact would deadlock the loader — every v4 artefact would be rejected at load with
no replacement to load). The unchanged schema-4 feature set stays green
(`tests/unit/test_fundamental_features.py`, 10 passed). Revisit Round 2 only if/when a future
fundamental retrain (e.g. on deeper fundamental history) produces a Round-1 KEEP; it is not an
automatic continuation.

---

## Phase 70 — ML failure root-cause spike (MLDIAG-03 controlled experiments)

**Date:** 2026-06-02
**Phase:** 70 (ML failure root-cause spike — diagnostic, NOT a retrain)
**Requirements:** MLDIAG-03
**Plan:** 70-03 (H3+H4 bounded controlled-experiment matrix on the two data-richest segments)

This phase enables **NO** segment and ships **NO** model. It runs a **FIXED** bounded
experiment matrix (one variable per row) via the existing `scripts/auto_ml_research.py`
harness — which has **NO `--force-save` path** (the flag does not exist; the honesty
guardrail is structurally enforced). Every run records its **honest** `overall_passed` /
`gate_pass_rates` / `status` (`keep` / `discard` / `crash`). DISC / no-alpha is a VALID and
likely outcome (D-05). The matrix is interpreted, not chased — no tune-until-pass loop.

The Phase-61/62/64 rows above are the **seeded** truth source this phase builds on:
- **Phase 62 — Stage-1 legitimate retrain: 0/17 `ru_*` segments passed the WF gate, 0 enabled, 0 force-saved.** (`ru_blue_chips` DISC best_acc 0.5307; `ru_energy` DISC best_acc 0.4167; `ru_finance` DISC; `ru_tech` DISC train-only.)
- **Phase 64 — Stage-3 fundamental A/B Round-1: KEEP = NONE (0/2).** Fundamentals do not move the WF gate on thin point-in-time history.

### Legend

`KEEP = overall_passed:true AND not force_saved` · `DISC = honest fail (overall_passed:false)` · `FORCED = force_saved (never shipped — structurally impossible in this harness)`

### Fixed experiment matrix (one knob per row; E1 is a GATING checkpoint)

| # | Knob (one variable) | Hypothesis | Harness path | Gating |
|---|---------------------|------------|--------------|--------|
| E1 | Baseline / no-signal floor | Is `best_acc` distinguishable from the 0.50 random floor? | `--strategy ablation --max-experiments 1` baseline + 0.50/permutation comparison | **GATING** — if `best_acc` ≈ 0.50 for BOTH segments, H3 no-alpha is largely confirmed and E2-E5 become low-value (short-circuit) |
| E2 | Feature subset / `max_features` | Does feature noise hurt small-N? | `--strategy ablation --max-experiments 3` | mandatory |
| E3 | Triple-barrier horizon / barriers | Is `TB_MAX_HOLD` / ATR-mult mismatched to the MOEX regime? | controlled `SEGMENT_BARRIER_CONFIG` / `TB_*` variation | mandatory |
| E4 | Model class / ensemble weights | Is the XGB+LGBM+CatBoost ensemble overfit for small N? | `--strategy ensemble_weights --max-experiments 2` | mandatory |
| E5 | Hyperparameter regularization | Does heavier regularization lift OOS accuracy? | `--strategy hyperparameter --max-experiments 2` | **CONDITIONAL** on E1 showing non-random signal (proceed-full); SKIPPED if short-circuit |

### Phase 70 experiment rows

Columns: `Segment | Experiment | knob varied | n_folds | best_acc (avg_accuracy) | overall_passed | binding sub-gate fails | force_saved | status | Verdict`. Every run is appended here with its honest harness verdict. `force_saved` is always `false` (no force-save path exists). `best_acc` is compared against the **0.50 random floor** and the H1 `n_effective` context from Plan 01's reporter (small `n_eff` raises the accuracy-gate threshold sharply).

<!-- E1 rows appended by Task 2 (gating checkpoint). E2-E5 rows appended by Task 3 after the operator gating decision. -->

| Segment | Experiment | knob varied | n_folds | best_acc | overall_passed | binding sub-gate fails (pass-rate) | force_saved | status | Verdict |
|---------|-----------|-------------|---------|----------|----------------|------------------------------------|-------------|--------|---------|
| ru_blue_chips | E1 baseline | standard MI selection, default MOEX hparams | 10 | 0.4056 | false | accuracy 0.2 / brier 0.2 / class_balance 0.1 / pf 0.3 / specificity 0.2 / degenerate 0.2 (all < 0.34) | false | discard | **DISC** |
| ru_blue_chips | E1 ablate-month_cos | drop `month_cos` (1 feature) | 10 | 0.4318 | false | accuracy 0.2 / brier 0.2 / class_balance 0.1 / pf 0.3 / specificity 0.2 (all < 0.34) | false | discard | **DISC** |
| ru_energy | E1 baseline | standard MI selection, default MOEX hparams | 10 | 0.5154 | false | accuracy 0.2 / brier 0.1 (< 0.34); class_balance 0.6, degenerate 0.6, pf 0.5, sensitivity 0.5, specificity 0.9 PASS | false | discard | **DISC** |
| ru_energy | E1 ablate-max_ret_20d | drop `max_ret_20d` (1 feature) | 10 | 0.5302 | false | accuracy 0.2 (< 0.34); brier 0.2; class_balance 0.6, degenerate 0.8, pf 0.5, specificity 0.9 PASS | false | discard | **DISC** |
| ru_blue_chips | E2 ablate-month_cos | drop `month_cos` (feature subset) | 10 | 0.4318 | false | accuracy 0.2 / brier 0.2 / class_balance 0.1 / degenerate 0.1 / pf 0.3 / specificity 0.2 (all < 0.34) | false | discard | **DISC** |
| ru_blue_chips | E2 ablate-min_ret_20d | drop `min_ret_20d` (feature subset) | 10 | 0.4226 | false | accuracy 0.3 / brier 0.2 / class_balance 0.1 / degenerate 0.1 / pf 0.2 / specificity 0.3 (all < 0.34) | false | discard | **DISC** |
| ru_blue_chips | E2 ablate-max_ret_20d | drop `max_ret_20d` (feature subset) | 10 | 0.4111 | false | accuracy 0.2 / brier 0.2 / class_balance 0.2 / degenerate 0.2 / pf 0.3 / specificity 0.3 (all < 0.34) | false | discard | **DISC** |
| ru_blue_chips | E3 barrier-horizon TB_MAX_HOLD=10 | shorter label horizon (20→10 bars) | 10 | 0.4655 | false | accuracy 0.3 / brier 0.1 / class_balance 0.3 / degenerate 0.3 / pf 0.2 (all < 0.34) | false | discard | **DISC** |
| ru_blue_chips | E3 barrier-horizon TB_MAX_HOLD=40 | longer label horizon (20→40 bars) | 9 | 0.4864 | false | accuracy 0.222 / brier 0.222 / class_balance 0.222 / degenerate 0.222 / specificity 0.0 (all < 0.34) | false | discard | **DISC** |
| ru_blue_chips | E4 ew-0.1-0.2-0.7 | ensemble weights XGB=0.1 LGBM=0.2 Cat=0.7 | 10 | 0.4801 | false | brier 0.3 / class_balance 0.2 / degenerate 0.3 / pf 0.3 (all < 0.34; accuracy 0.4 PASS) | false | discard | **DISC** |
| ru_blue_chips | E4 ew-0.1-0.3-0.6 | ensemble weights XGB=0.1 LGBM=0.3 Cat=0.6 | 10 | 0.4777 | false | brier 0.2 / class_balance 0.2 / degenerate 0.2 / pf 0.3 (all < 0.34) | false | discard | **DISC** |
| ru_blue_chips | E5 hp-xgb_max_depth=3 | XGB max_depth 5→3 (regularize) | 10 | 0.4112 | false | accuracy 0.2 / brier 0.2 / class_balance 0.1 / degenerate 0.2 / pf 0.2 / specificity 0.2 (all < 0.34) | false | discard | **DISC** |
| ru_blue_chips | E5 hp-xgb_max_depth=4 | XGB max_depth 5→4 (regularize) | 10 | 0.4112 | false | accuracy 0.2 / brier 0.2 / class_balance 0.1 / degenerate 0.2 / pf 0.2 / specificity 0.2 (all < 0.34) | false | discard | **DISC** |
| ru_energy | E2 ablate-max_ret_20d | drop `max_ret_20d` (feature subset) | 10 | 0.5191 | false | brier 0.2 (< 0.34); accuracy 0.4, class_balance 0.7, degenerate 0.9, pf 0.5, specificity 0.9 PASS | false | discard | **DISC** |
| ru_energy | E2 ablate-min_ret_20d | drop `min_ret_20d` (feature subset) | 10 | 0.5329 | false | accuracy 0.3 / brier 0.1 (< 0.34); class_balance 0.9, degenerate 1.0, specificity 0.9 PASS | false | discard | **DISC** |
| ru_energy | E2 ablate-hist_vol_20 | drop `hist_vol_20` (feature subset) | 10 | 0.5340 | false | brier 0.3 (< 0.34); accuracy 0.4, class_balance 0.7, degenerate 0.9, pf 0.5, specificity 0.8 PASS | false | discard | **DISC** |
| ru_energy | E3 barrier-horizon TB_MAX_HOLD=10 | shorter label horizon (20→10 bars) | 10 | 0.5051 | false | accuracy 0.0 / brier 0.0 / pf 0.3 (all < 0.34) | false | discard | **DISC** |
| ru_energy | E3 barrier-horizon TB_MAX_HOLD=40 | longer label horizon (20→40 bars) | 9 | 0.5100 | false | accuracy 0.222 / brier 0.0 / pf 0.333 (all < 0.34) | false | discard | **DISC** |
| ru_energy | E4 ew-0.1-0.2-0.7 | ensemble weights XGB=0.1 LGBM=0.2 Cat=0.7 | 10 | 0.5299 | false | accuracy 0.2 / brier 0.2 (< 0.34); pf 2.18 raw | false | discard | **DISC** |
| ru_energy | E4 ew-0.1-0.3-0.6 | ensemble weights XGB=0.1 LGBM=0.3 Cat=0.6 | 10 | 0.5289 | false | accuracy 0.2 / brier 0.2 (< 0.34); pf 3.60 raw | false | discard | **DISC** |
| ru_energy | E5 hp-xgb_max_depth=3 | XGB max_depth 5→3 (regularize) | 10 | 0.5344 | false | accuracy 0.3 / brier 0.1 (< 0.34) | false | discard | **DISC** |
| ru_energy | E5 hp-xgb_max_depth=4 | XGB max_depth 5→4 (regularize) | 10 | 0.5341 | false | accuracy 0.2 / brier 0.1 / pf 0.3 (all < 0.34) | false | discard | **DISC** |

> **Harness-mandatory baselines:** `run_research_loop` always runs a `baseline` experiment
> before the strategy batch, so the JSONL also carries one `baseline` record per E2/E4/E5 invocation
> (identical to the E1 baseline — `ru_blue_chips` acc 0.4056, `ru_energy` acc 0.5148). These are NOT
> tune-until-pass retries; they are the harness's fixed control. The E3 runs ARE the baseline (the
> one varied variable is `_TB_MAX_HOLD`, set at runtime; `--max-experiments 0` ⇒ baseline-only).

### E2–E5 results (proceed-full decision, 2026-06-02)

The operator returned **proceed-full** on the Task 2 E1 gating checkpoint, so the FULL fixed matrix
ran — E2, E3, E4 mandatory **and E5** (the hyperparameter-regularization grid) — on BOTH segments via
the existing `scripts/auto_ml_research.py` harness, one variable per row, **NO `--force-save`** (no such
flag). DB-first cached MOEX candles (token in `os.environ`, never logged — T-70-06). **Every one of the
18 Task-3 experiment runs landed `overall_passed:false` / DISC; zero crossed any binding sub-gate.**

- **E2 (feature noise, H3):** dropping single features moved `best_acc` only marginally
  (ru_blue_chips 0.41–0.43 — sub-random; ru_energy 0.52–0.53 — barely above 0.50). No subset crossed the
  gate. Feature noise is not the binding constraint.
- **E3 (triple-barrier horizon, H4):** varied the **one** param `_TB_MAX_HOLD` (20→10, 20→40 bars) at
  runtime through the harness's own `build_full_dataset` (no CLI knob exists — documented per plan). Both
  shorter and longer horizons stayed DISC on both segments (ru_blue_chips 0.4655/0.4864; ru_energy
  0.5051/0.5100). Horizon mismatch is not the binding constraint. *(Note: the 40-bar run lost one WF fold
  — 9 vs 10 — as the longer label window consumes more tail history; expected, not a defect.)*
- **E4 (model class / ensemble weights, H4):** Cat-heavy mixes (0.6–0.7) raised raw profit-factor
  (ru_energy pf 2.18/3.60) but `best_acc` stayed below the gate (ru_blue_chips 0.478/0.480; ru_energy
  0.529/0.530) and accuracy/brier still cleared only ≤ 2/10 folds. Ensemble overfit is not the savior.
- **E5 (hyperparameter regularization, H4):** heavier regularization (XGB max_depth 5→3, 5→4) did **not**
  lift OOS accuracy (ru_blue_chips flat at 0.4112; ru_energy 0.534) — no gate crossing. Regularization is
  not the binding constraint.

**Net Task-3 result: 0/18 experiment runs passed; KEEP = NONE on both segments.** Across E1–E5 the
binding accuracy and brier sub-gates never cleared the `MOEX_MIN_PASSING_FOLDS_RATIO = 0.34` threshold on
either segment, regardless of which single lever was varied. This is the honest, expected outcome (D-05):
no separable predictive edge was found within the bounded, principled matrix — strong evidence for H3
"no alpha" (corroborated, not chased). No tune-until-pass loop was run; the matrix was fixed up front and
every verdict recorded as-is. This feeds Plan 04's H1–H4 ranking and the pursue/abandon recommendation.

### E1 gating evidence (read against the 0.50 random floor)

Both E1 runs used the existing `auto_ml_research.py` harness with `--strategy ablation --max-experiments 1 --seed 42`, live Tinkoff/MOEX-ISS data (token via `os.environ`, never logged — T-70-06), and **NO `--force-save`** (no such flag exists; `force_saved` key absent from every JSONL record). Per-experiment honest JSONL: `results/experiments/ru_blue_chips_experiment_log.jsonl` (2 records) and `results/experiments/ru_energy_experiment_log.jsonl` (2 records).

- **ru_blue_chips** — live universe resolves to **SFIN-only** (Phase-66 liquidity selector): 855 candles → 634 triple-barrier samples (56.0% positive), 10 WF folds. **best_acc 0.4056–0.4318 is clearly BELOW the 0.50 random floor.** No separable signal; the accuracy sub-gate clears only 2/10 folds. H3 "no alpha" strongly supported.
- **ru_energy** — 7 symbols (LKOH/ROSN/NVTK/TATN/TRNFP/SIBN/RNFT, ~855–861 candles each) → 4512 triple-barrier samples (52.0% positive), 10 WF folds. **best_acc 0.5154–0.5302 sits only marginally ABOVE 0.50.** The accuracy sub-gate still clears only 2/10 folds (< 0.34 min-ratio) and brier clears 1–2/10 — `overall_passed:false`. Marginal, not clearly separable.

**Gating recommendation (evidence only — operator decides):** ru_blue_chips is sub-random (no alpha). ru_energy is marginally above 0.50 but does not separate enough to pass any binding sub-gate. This is closer to **short-circuit** (H3 no-alpha largely confirmed; E2–E5 low-value, skip E5) than to a clear **proceed-full** signal. The operator returns the gating decision; Task 3 then runs accordingly.
