---
gsd_state_version: 1.0
milestone: v9.1
milestone_name: MOEX ML Model Quality
status: unknown
stopped_at: Completed 46-01-PLAN.md
last_updated: "2026-04-14T11:28:49.078Z"
progress:
  total_phases: 21
  completed_phases: 2
  total_plans: 3
  completed_plans: 3
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-14)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** Phase 46 — Feature Selection Stability

## Current Position

Phase: 47
Plan: Not started

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

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-04-14T11:26:03.134Z
Stopped at: Completed 46-01-PLAN.md
Resume file: None
