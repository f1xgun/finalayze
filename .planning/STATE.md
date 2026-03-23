---
gsd_state_version: 1.0
milestone: v5.0
milestone_name: Data Flow Correctness & Live-Backtest Parity
status: Phase complete — ready for verification
stopped_at: Completed 24-02-PLAN.md
last_updated: "2026-03-23T19:56:06.227Z"
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 7
  completed_plans: 3
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-23)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** Phase 24 — live-backtest-parity

## Current Position

Phase: 24 (live-backtest-parity) — EXECUTING
Plan: 2 of 2

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: --
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

## Accumulated Context

| Phase 23 P01 | 5min | 2 tasks | 2 files |
| Phase 24 P01 | 8min | 1 tasks | 8 files |
| Phase 24 P02 | 5min | 2 tasks | 3 files |

### Decisions

Decisions from v1.0-v4.0 are archived in milestones/.
Key carry-forward decisions for v5.0:

- sys.modules shims for backward-compat module moves (v4.0)
- MetricsCollector via constructor DI (v4.0)
- asyncio.Lock for async, threading.Lock for sync paths (v4.0)
- GARCH rolling vol fallback over NaN (v4.0)
- Fire-and-forget DB persistence -- never crash the trading loop
- [Phase 23]: SELL orders skip Kelly sizing and CAUTION reduction, sell entire held position
- [Phase 23]: Segment min_confidence loaded from same YAML presets as StrategyCombiner, cached per segment
- [Phase 24]: StopLossState reused from simulated_broker.py as canonical trailing stop state for both backtest and live
- [Phase 24]: ATR value derived algebraically from stop formula instead of adding compute_atr_value function
- [Phase 24]: Correlations return empty dict for graceful degradation in live pre-trade check 14
- [Phase 24]: Pipeline includes Copula+EVT steps matching backtest even with empty returns_history

### Pending Todos

None yet.

### Blockers/Concerns

- PositionSizingPipeline wiring in live requires careful integration -- pipeline was designed for backtest engine
- Trailing stop in live needs state management across APScheduler cycles (not single-pass like backtest)
- News pipeline disable must preserve the option to re-enable event_driven later
- SELL sizing fix must handle partial positions and lot rounding for MOEX

## Session Continuity

Last session: 2026-03-23T19:56:06.224Z
Stopped at: Completed 24-02-PLAN.md
Resume file: None
