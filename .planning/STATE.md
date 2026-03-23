---
gsd_state_version: 1.0
milestone: v5.0
milestone_name: Data Flow Correctness & Live-Backtest Parity
status: defining_requirements
stopped_at: null
last_updated: "2026-03-23T10:00:00.000Z"
last_activity: 2026-03-23 -- Milestone v5.0 started
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-23)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** v5.0 Data Flow Correctness — defining requirements

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-03-23 — Milestone v5.0 started

## Accumulated Context

### Decisions

Decisions from v1.0-v4.0 are archived in milestones/.
Key carry-forward decisions for v5.0:

- sys.modules shims for backward-compat module moves (v4.0)
- MetricsCollector via constructor DI (v4.0)
- asyncio.Lock for async, threading.Lock for sync paths (v4.0)
- GARCH rolling vol fallback over NaN (v4.0)
- Fire-and-forget DB persistence — never crash the trading loop

### Pending Todos

None yet.

### Blockers/Concerns

- PositionSizingPipeline wiring in live requires careful integration — pipeline was designed for backtest engine
- Trailing stop in live needs state management across APScheduler cycles (not single-pass like backtest)
- News pipeline disable must preserve the option to re-enable event_driven later
- SELL sizing fix must handle partial positions and lot rounding for MOEX

## Session Continuity

Last session: 2026-03-23
Stopped at: Defining requirements
Resume file: None
