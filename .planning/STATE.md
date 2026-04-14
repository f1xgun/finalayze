---
gsd_state_version: 1.0
milestone: v9.1
milestone_name: MOEX ML Model Quality
status: ready_to_plan
stopped_at: Roadmap created, Phase 45 ready to plan
last_updated: "2026-04-14"
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-14)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** v9.1 — Phase 45: Model Complexity & Ensemble Consistency

## Current Position

Phase: 45 of 48 (Model Complexity & Ensemble Consistency)
Plan: — of — (not yet planned)
Status: Ready to plan
Last activity: 2026-04-14 — Roadmap created for v9.1, 4 phases (45-48)

Progress: [░░░░░░░░░░] 0%

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

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-04-14
Stopped at: Roadmap written. Next: `/gsd:plan-phase 45`
Resume file: None
