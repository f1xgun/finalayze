---
gsd_state_version: 1.0
milestone: v4.0
milestone_name: Architecture Hardening
status: Ready to execute
stopped_at: Completed 20-02-PLAN.md
last_updated: "2026-03-22T20:46:44.509Z"
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 5
  completed_plans: 4
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-22)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** Phase 20 — async-correctness-and-resource-management

## Current Position

Phase: 20 (async-correctness-and-resource-management) — EXECUTING
Plan: 3 of 3

## Performance Metrics

**Velocity:**

- Total plans completed: 0 (v4.0) / 48 (all milestones)
- Average duration: --
- Total execution time: --

## Accumulated Context

### Decisions

Decisions from v1.0-v3.0 are archived in milestones/.
Key carry-forward decisions for v4.0:

- Monitoring services standalone (not embedded in TradingLoop)
- KillSwitch uses deferred imports for CircuitLevel/AlertPriority to maintain layer boundaries
- File-based kill flag works even when DB is down
- Fire-and-forget DB persistence for metrics -- never crash the trading loop
- [Phase 19]: Keep _client_lock as threading.Lock for sync _get_client (APScheduler compat); separate _loop_init_lock for event loop init guard
- [Phase 19]: Stop-loss check-and-sell made atomic under single lock hold to prevent double-sell TOCTOU race
- [Phase 19]: Replaced getattr indirection with direct call for critical monitoring APIs
- [Phase 20]: Idempotent TelegramAlerter.close() via _closed flag; both instances closed in lifespan shutdown
- [Phase 20]: Used default ThreadPoolExecutor for run_in_executor in portfolio API -- appropriate for I/O-bound broker calls
- [Phase 20]: Split close() cleanup into separate try/except blocks for __aexit__ and loop.stop -- independent failure handling
- [Phase 20]: Default gRPC timeout 60s for TinkoffFetcher -- balances MOEX latency with hang prevention

### Pending Todos

None yet.

### Blockers/Concerns

- trading_loop.py is ~1800 lines god-object -- extract carefully to avoid breaking APScheduler wiring
- 140+ except Exception clauses -- tightening too aggressively may crash the trading loop
- Event bus removal must not break bond_discovery.py coupon publishing

## Session Continuity

Last session: 2026-03-22T20:46:44.506Z
Stopped at: Completed 20-02-PLAN.md
Resume file: None
