---
gsd_state_version: 1.0
milestone: v5.0
milestone_name: Data Flow Correctness & Live-Backtest Parity
status: unknown
stopped_at: Completed 25-02-PLAN.md
last_updated: "2026-03-24T07:47:09.387Z"
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 7
  completed_plans: 7
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-23)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** Phase 25 — data-validation-and-infrastructure

## Current Position

Phase: 26
Plan: Not started

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
| Phase 26 P01 | 5min | 1 tasks | 3 files |
| Phase 25 P01 | 5min | 1 tasks | 3 files |
| Phase 25 P02 | 8min | 2 tasks | 5 files |

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
- [Phase 26]: Sentiment decay uses time.monotonic() with 4h half-life, cached once per process
- [Phase 25]: 48h staleness threshold (2x daily timeframe) as module-level constant
- [Phase 25]: Bond async methods use _get_services_async for persistent gRPC channel
- [Phase 25]: Brent cache uses Candle model class with BZ_F cache_id

### Roadmap Evolution

- Phase 27 added: Intelligent News Impact Analysis — sector-aware LLM pipeline replacing naive ticker extraction

### Pending Todos

None yet.

### Blockers/Concerns

- NewsImpactAnalyzer prompt must handle Russian + English articles with sector vocabulary
- SectorTickerMapper must stay in sync with InstrumentRegistry — stale mapping = missed tickers
- Per-ticker sentiment cache changes _sentiment_cache structure — event_driven strategy must adapt
- LLM response quality for sector impact prediction needs validation against real news corpus

## Session Continuity

Last session: 2026-03-24T07:38:47.006Z
Stopped at: Completed 25-02-PLAN.md
Resume file: None
