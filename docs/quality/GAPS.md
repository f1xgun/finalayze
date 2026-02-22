# Technical Debt & Gaps Tracker

## Active Gaps

| ID | Module | Description | Priority | Added |
|----|--------|-------------|----------|-------|
| G-002 | config/ | No unit tests for `settings.py` | Medium | 2026-02-21 |
| G-004 | all | No integration test infrastructure (DB fixtures, Redis fixtures) | Medium | 2026-02-21 |
| G-005 | all | `import-linter` or custom ruff rule for layer enforcement not configured | Low | 2026-02-21 |
| G-006 | core/ | Structured logging (`structlog`) configured but not integrated into any module | Low | 2026-02-21 |
| G-007 | backtest/ | Integration test: full backtest run against real PostgreSQL + TimescaleDB not yet implemented (requires docker-compose) | Medium | 2026-02-22 |

## Resolved Gaps

| ID | Module | Description | Resolved | Phase |
|----|--------|-------------|----------|-------|
| G-001 | core/ | `schemas.py`, `models.py`, `events.py`, `clock.py` are empty stubs | 2026-02-22 | Phase 1 Backtest Slice |
| G-003 | all | No Alembic migration files yet | 2026-02-22 | Phase 1 Backtest Slice |
