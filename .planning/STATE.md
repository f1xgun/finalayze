---
gsd_state_version: 1.0
milestone: v9.1
milestone_name: MOEX ML Model Quality
status: defining_requirements
stopped_at: Milestone started, defining requirements
last_updated: "2026-04-14"
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-14)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** v9.1 — MOEX ML Model Quality

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-04-14 — Milestone v9.1 started

## Performance Metrics

**Velocity:**

- Total plans completed: 0 (v9.1)
- Average duration: —
- Total execution time: —

## Accumulated Context

### Key Findings from Agent Analysis (v9.1 motivation)

- ru_blue_chips: 9/51 experiments pass quality gates after brier/accuracy/mean_uniqueness fixes
- ru_energy: 0/51 — sensitivity=0.30, model can't predict UP (missing commodity features)
- ru_tech: 0/51 — degenerate predictions (HEAD ~370d, YDEX ~450d, insufficient history)
- ru_finance: 0/51 — poor accuracy/brier (SBER+SBERP correlation, T relisted ~500d)
- Quality gates already adapted: brier two-regime threshold, smooth accuracy cap, mean_uniqueness=1/hold_bars, adaptive min_passing_folds_ratio
- XGBoost always applies scale_pos_weight=n_neg/n_pos even with sample_weight; LightGBM sets 1.0 — inconsistent

### Key Architectural Decisions (v9.0 → v9.1)

- All v9.0 changes concentrated in `scripts/auto_ml_research.py` + `quality_gates.py` — no new modules
- Quality gates adapted for MOEX: brier floor=0.24 for n_eff<40, smooth accuracy cap, min_ratio=0.50 for >=8 folds
- MOEX lookback increased from 730 to 1095 days (10 folds instead of 4)
- mean_uniqueness = 1/avg_hold_bars (AFML Ch.4) — was hardcoded 1.0

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-04-14
Stopped at: Defining requirements for v9.1
Resume file: None
