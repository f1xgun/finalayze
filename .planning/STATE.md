---
gsd_state_version: 1.0
milestone: v9.1
milestone_name: MOEX ML Model Quality
status: Phase complete — ready for verification
stopped_at: Completed 47-02-PLAN.md
last_updated: "2026-04-14T11:48:11.741Z"
progress:
  total_phases: 21
  completed_phases: 3
  total_plans: 5
  completed_plans: 5
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-14)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** Phase 47 — Cross-Asset Features & Asymmetric Barriers

## Current Position

Phase: 47 (Cross-Asset Features & Asymmetric Barriers) — EXECUTING
Plan: 2 of 2

## Performance Metrics

**Velocity:**

- Total plans completed: 0 (v9.1)
- Average duration: —
- Total execution time: —

## Accumulated Context

### Key Findings from Agent Analysis (v9.1 motivation)

- ru_energy: 0/51 experiments pass — sensitivity=0.30, model can't predict UP (missing Brent features)
- ru_tech: 0/51 — degenerate predictions (HEAD ~370d, YDEX ~450d, insufficient history)
- ru_finance: 0/51 — poor accuracy/brier (SBER+SBERP rho>0.95, T relisted ~500d)
- Root cause 1: depth=5 on ~850 samples → severe overfitting (ml-engineer finding)
- Root cause 2: per-fold MI-based feature selection → feature set changes fold-to-fold
- Root cause 3: XGBoost applies scale_pos_weight AND sample_weight → double-rebalancing

### Decisions

- Phase 45 before 46: fix complexity first, then measure feature stability improvement
- Phase 46 before 47: stable feature selection makes it meaningful to add new features
- Phase 47 before 48: features and barriers must be in place before final validation run
- SBERP removal goes in Phase 48 (segment config), not Phase 45 (model config)
- [Phase 45]: XGBoost scale_pos_weight=1.0 when sample_weight provided — prevents double-rebalancing with class ratio
- [Phase 45]: CatBoost auto_class_weights=None when sample_weight provided — same consistency pattern as XGBoost/LightGBM
- [Phase 45]: MOEX segments use reduced hyperparameters (depth=3, n_estimators=100, min_child_weight=20) via _get_hparams() helper routing; _DEFAULT_HPARAMS unchanged for US segments
- [Phase 46]: select_features_efficient called once on union of all training indices (pre-fold) to eliminate feature churn
- [Phase 46]: New ExperimentConfig created with feature_subset set — caller's config not mutated (T-46-02 mitigation)
- [Phase 47]: Independent per-feature fallback: each Brent feature computes in its own if-block; partial data yields partial features
- [Phase 47]: Clip bounds scale with horizon: 1d=[-0.15,0.15], 5d=[-0.30,0.30], 21d=[-0.50,0.50]
- [Phase 47]: _SEGMENT_BARRIER_CONFIG is a dict lookup — adding new segments requires only a new entry, no if/else logic
- [Phase 47]: MOEX uplift applied in _get_barrier_params() not in the config dict — keeps pre-uplift values readable

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-04-14T11:48:11.738Z
Stopped at: Completed 47-02-PLAN.md
Resume file: None
