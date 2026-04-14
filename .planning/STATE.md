---
gsd_state_version: 1.0
milestone: v10.0
milestone_name: Runtime LLM Trading Agents
status: defining_requirements
stopped_at: Milestone started, research pending
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
**Current focus:** v10.0 — Runtime LLM Trading Agents

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-04-14 — Milestone v10.0 started

## Performance Metrics

**Velocity:**

- Total plans completed: 0 (v10.0)
- Average duration: —
- Total execution time: —

## Accumulated Context

### Key Architectural Decisions (v9.0 → v10.0)

- All v9.0 changes concentrated in `scripts/auto_ml_research.py` + `quality_gates.py` — no new modules
- TinkoffFetcher sync-async bridge via `_run_async()` — no nest_asyncio needed in script context
- `sandbox=False` is mandatory for training — sandbox endpoint has no historical candles
- ExperimentManager integration is opt-in via `--experiment-id` flag — existing JSONL invocations unaffected
- Macro series must be `shift(1)` before feature join — look-ahead bias prevention

### Expert Debate Results (v10.0 scoping)

- 2 rounds, 5 domain agents (Quant, Risk, Architect, Portfolio, ML Engineer)
- **APPROVED:** News Pipeline, EventDriven activation, Portfolio Review Agent, Anomaly Interpreter, Sentiment ML features infra
- **REJECTED (unanimous):** Pre-Trade Reasoning Agent — non-determinism in sizing pipeline, uncalibrated output, irreproducible backtests
- **DEFERRED:** Cached Reasoning Overlay (only if ML features < 0.55 AUC), live A/B testing
- Key conditions: credibility cap 0.7, 5s hard timeout, advisory-only agents, Haiku for volume / Sonnet for reasoning

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-04-14
Stopped at: Milestone v10.0 started, research pending
Resume file: None
