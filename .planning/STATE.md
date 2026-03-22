---
gsd_state_version: 1.0
milestone: v4.0
milestone_name: Architecture Hardening
status: unknown
stopped_at: Completed 21-02-PLAN.md
last_updated: "2026-03-22T21:07:56.544Z"
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 7
  completed_plans: 7
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-22)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** Phase 21 — error-handling-hardening

## Current Position

Phase: 22
Plan: Not started

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
- [Phase 20]: Used _stop_event.wait(timeout=) for gRPC reconnect delay instead of time.sleep
- [Phase 20]: Used asyncio.iscoroutine() in aexecute for dual sync/async callable support
- [Phase 20]: Lazy background event loop thread for SandboxMonitor persistence replacing asyncio.run()
- [Phase 21]: GARCH returns NaN only for < 2 data points; all other failures use rolling vol fallback
- [Phase 21]: EventBus uses try/except redis.ResponseError instead of contextlib.suppress(Exception)
- [Phase 21]: POST /kill endpoint now requires X-API-Key authentication
- [Phase 21]: Used AlertPriority.CRITICAL for consecutive failure alerts; per-layer error tracking in BondCycleProcessor; threshold of 3 consecutive failures

### Pending Todos

None yet.

### Blockers/Concerns

- trading_loop.py is ~1800 lines god-object -- extract carefully to avoid breaking APScheduler wiring
- 140+ except Exception clauses -- tightening too aggressively may crash the trading loop
- Event bus removal must not break bond_discovery.py coupon publishing

## Session Continuity

Last session: 2026-03-22T21:03:46.178Z
Stopped at: Completed 21-02-PLAN.md
Resume file: None
