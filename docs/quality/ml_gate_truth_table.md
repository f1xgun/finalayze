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
