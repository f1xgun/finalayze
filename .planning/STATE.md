---
gsd_state_version: 1.0
milestone: v9.0
milestone_name: ML AutoResearch & MOEX Adaptation
status: ready_to_plan
stopped_at: Roadmap created — Phase 40 ready to plan
last_updated: "2026-04-13"
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-13)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** Phase 40 — MOEX Data Adapter & Macro Features

## Current Position

Phase: 40 of 44 (MOEX Data Adapter & Macro Features)
Plan: — (not yet planned)
Status: Ready to plan
Last activity: 2026-04-13 — v9.0 roadmap created, Phase 40 ready to plan

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0 (v9.0)
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

*Updated after each plan completion*

## Accumulated Context

### Key Architectural Decisions (v8.0 → v9.0)

- All v9.0 changes concentrated in `scripts/auto_ml_research.py` + `quality_gates.py` — no new modules
- TinkoffFetcher sync-async bridge via `_run_async()` — no nest_asyncio needed in script context
- `sandbox=False` is mandatory for training — sandbox endpoint has no historical candles
- ExperimentManager integration is opt-in via `--experiment-id` flag — existing JSONL invocations unaffected
- Macro series must be `shift(1)` before feature join — look-ahead bias prevention

### Pending Todos

None.

### Blockers/Concerns

- Phase 40 research flag: verify actual MOEX candle counts per ru_* segment empirically on first run
- Phase 41 research flag: MOEX walk-forward fold constants are analytically derived — validate against real data
- Phase 44 research flag: feature engineering domain rules and permutation test threshold need explicit design decision before implementation

## Session Continuity

Last session: 2026-04-13
Stopped at: Roadmap written — 5 phases (40-44), 11 requirements mapped
Resume file: None
