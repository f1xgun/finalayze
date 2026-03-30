---
gsd_state_version: 1.0
milestone: v6.0
milestone_name: Sandbox Stability & Observability
status: requirements
stopped_at: null
last_updated: "2026-03-30"
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-30)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** Defining requirements for v6.0

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-03-30 — Milestone v6.0 started

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: --
- Total execution time: 0 hours

## Accumulated Context

Carried from v5.0:

- Fire-and-forget DB persistence — never crash the trading loop
- Per-ticker sentiment cache with (segment_id, ticker) tuple keys
- Persistent gRPC channel for TinkoffFetcher bond methods
- sys.modules shims for backward-compat module moves

### Sandbox Analysis Findings (2026-03-30)

Key issues from week-long sandbox run (March 20-30):
- gRPC BlockingIOError floods asyncio loop → 127 missed scheduler jobs, strategy cycle drift up to 60 min
- T-Bank API error 70001 → 62 portfolio_fetch_failed, multi-hour outage windows during market hours
- DB persistence broken → orders/signals/news_articles/sentiment_scores all 0 rows
- Loki pipeline non-functional → Promtail not shipping logs, 0 entries ever stored
- FX rate = 0.0 (gRPC failure, no fallback)
- Strategy cycles fire outside MOEX hours with 0 instruments
- 5 stale tickers in config (FIVE, FIXP, POLY, YNDX, HHRU)
- 35 LLM fallback activations/day due to article duplication
- Only 1 trade executed in 62 hours (BSPB BUY, rsi2_connors)
- Equity: 2,498,817 → 2,502,576 RUB (+0.15%) over 5 days

### Pending Todos

None yet.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-03-30
Stopped at: Defining requirements
Resume file: None
