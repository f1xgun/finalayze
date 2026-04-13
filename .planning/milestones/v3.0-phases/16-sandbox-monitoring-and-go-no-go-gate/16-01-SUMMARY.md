---
phase: 16-sandbox-monitoring-and-go-no-go-gate
plan: 01
subsystem: monitoring
tags: [timescaledb, anomaly-detection, z-score, dataclass, orm, alembic]

requires:
  - phase: 15-schemas-config-rollout-foundation
    provides: "ORM model patterns, migration conventions, TelegramAlerter"
provides:
  - "CycleMetrics frozen dataclass for per-cycle metric capture"
  - "SandboxMonitorService with record_slippage, on_cycle_complete, DB persistence"
  - "AnomalyDetector with z-score drawdown, fill rate, slippage threshold checks"
  - "SandboxMetricRow ORM model mapped to sandbox_metrics hypertable"
  - "Alembic migration 005 creating sandbox_metrics TimescaleDB hypertable"
affects: [16-02-go-no-go-reporter, 16-03-trading-loop-wiring]

tech-stack:
  added: []
  patterns: ["fire-and-forget async persistence via _run_async_safe", "per-metric cooldown with monotonic clock"]

key-files:
  created:
    - src/finalayze/monitoring/__init__.py
    - src/finalayze/monitoring/sandbox_monitor.py
    - src/finalayze/monitoring/anomaly_detector.py
    - alembic/versions/005_sandbox_metrics.py
    - tests/unit/test_sandbox_monitor.py
    - tests/unit/test_anomaly_detector.py
  modified:
    - src/finalayze/core/models.py

key-decisions:
  - "Frozen dataclass for CycleMetrics -- immutable per-cycle snapshots, no mutation after creation"
  - "Fire-and-forget DB persistence via asyncio.run wrapped in try/except -- metrics never crash the loop"
  - "AnomalyDetector uses deferred import for AlertPriority to avoid circular dependency"

patterns-established:
  - "Monitoring service pattern: standalone service with on_cycle_complete callback"
  - "Per-metric cooldown: dict[str, float] with time.monotonic for independent cooldown tracking"

requirements-completed: [MON-01, MON-02, MON-04]

duration: 4min
completed: 2026-03-21
---

# Phase 16 Plan 01: Monitoring Foundation Summary

**CycleMetrics dataclass + SandboxMonitorService with DB persistence + AnomalyDetector with z-score/threshold alerting and 30-min cooldown**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-21T20:47:34Z
- **Completed:** 2026-03-21T20:51:42Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- CycleMetrics frozen dataclass with 11 fields capturing all per-cycle trading metrics
- SandboxMonitorService with slippage buffer, cycle counting, async DB persistence, and anomaly detection integration
- AnomalyDetector with drawdown z-score (>2 sigma), fill rate (<0.90), and slippage (>50bps) checks
- 30-minute per-metric independent cooldown preventing alert spam
- SandboxMetricRow ORM model with composite PK (timestamp, market_id)
- Alembic migration 005 creating sandbox_metrics TimescaleDB hypertable
- 28 unit tests across both test files

## Task Commits

Each task was committed atomically:

1. **Task 1: Create SandboxMonitorService with CycleMetrics and DB persistence** - `bcec477` (feat)
2. **Task 2: Create AnomalyDetector with z-score and threshold alerting** - `f476e43` (feat)

## Files Created/Modified
- `src/finalayze/monitoring/__init__.py` - Package init exporting CycleMetrics, SandboxMonitorService, AnomalyDetector
- `src/finalayze/monitoring/sandbox_monitor.py` - CycleMetrics dataclass and SandboxMonitorService class
- `src/finalayze/monitoring/anomaly_detector.py` - AnomalyDetector with z-score and threshold checks
- `src/finalayze/core/models.py` - Added SandboxMetricRow ORM model
- `alembic/versions/005_sandbox_metrics.py` - TimescaleDB hypertable migration
- `tests/unit/test_sandbox_monitor.py` - 14 tests for monitor service
- `tests/unit/test_anomaly_detector.py` - 14 tests for anomaly detector

## Decisions Made
- Used frozen dataclass (not Pydantic) for CycleMetrics -- consistent with RolloutLimits pattern from Phase 15
- Fire-and-forget persistence via asyncio.run wrapped in try/except -- metrics should never crash the trading loop
- Deferred import of AlertPriority inside _fire_alert to avoid circular dependency with core.alerts

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Test for drawdown z-score needed more baseline entries (10 instead of 3) to produce a z-score > 2.0 when spike is added to the rolling window. Adjusted test data accordingly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Monitoring module ready for GoNoGoReporter (Plan 02) to query SandboxMetricRow
- SandboxMonitorService ready for TradingLoop wiring (Plan 03) via on_cycle_complete callback
- AnomalyDetector fully functional with optional TelegramAlerter integration

---
*Phase: 16-sandbox-monitoring-and-go-no-go-gate*
*Completed: 2026-03-21*
