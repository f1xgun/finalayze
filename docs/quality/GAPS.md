# Technical Debt & Gaps Tracker

## Active Gaps

| ID | Module | Description | Priority | Added |
|----|--------|-------------|----------|-------|
| G-002 | config/ | No unit tests for `settings.py` | Medium | 2026-02-21 |
| G-005 | all | `import-linter` or custom ruff rule for layer enforcement not configured | Low | 2026-02-21 |
| G-008 | ml/ | ML model accuracy suboptimal (~57% best fold). 16 features added but models need retraining with tuned hyperparameters | High | 2026-03-08 |
| G-009 | strategies/ | Trade count low (626 vs 1300 target). ADX routing + confidence thresholds filter most signals | Medium | 2026-03-06 |
| G-010 | analysis/ | `event_driven` strategy disabled — no real-time news feed integration | Medium | 2026-03-01 |
| G-011 | ml/ | `ml_ensemble` disabled in all presets pending accuracy improvements | High | 2026-03-08 |
| G-012 | strategies/ | `pead` strategy disabled — needs earnings calendar data source | Low | 2026-03-01 |
| G-013 | backtest/ | Walk-forward Sharpe still negative (-0.004). OOS performance needs improvement | High | 2026-03-06 |

## Resolved Gaps

| ID | Module | Description | Resolved | Phase |
|----|--------|-------------|----------|-------|
| G-001 | core/ | `schemas.py`, `models.py`, `events.py`, `clock.py` are empty stubs | 2026-02-22 | Phase 1 |
| G-003 | all | No Alembic migration files yet | 2026-02-22 | Phase 1 |
| G-004 | all | No integration test infrastructure (DB fixtures, Redis fixtures) | 2026-02-28 | Phase 4 |
| G-006 | core/ | Structured logging not integrated | 2026-02-25 | Phase 3 |
| G-007 | backtest/ | No full backtest integration test against real DB | 2026-02-28 | Phase 4 |
