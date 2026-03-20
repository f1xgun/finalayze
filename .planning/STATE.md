---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: MOEX Profitability
status: completed
stopped_at: Completed 08-03-PLAN.md (Phase 08 complete)
last_updated: "2026-03-20T08:02:25.680Z"
last_activity: 2026-03-20 -- Completed 08-03 MOEX Dividend Calendar Expansion
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 3
  completed_plans: 3
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-20)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** Phase 8 -- Data Foundation

## Current Position

Phase: 8 of 12 (Data Foundation) -- first phase of v2.0
Plan: 3 of 3 in current phase (08-01, 08-02, 08-03 complete)
Status: Phase 08 Complete
Last activity: 2026-03-20 -- Completed 08-03 MOEX Dividend Calendar Expansion

Progress: [##########] 100% (phase 08 complete)

## Performance Metrics

**Velocity (v1.0):**

- Total plans completed: 22
- Average duration: ~45 min
- Total execution time: ~16.5 hours

**Velocity (v2.0):**

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 08    | 01   | 2min     | 2     | 6     |
| 08    | 02   | 6min     | 2     | 6     |
| 08    | 03   | 3min     | 3     | 4     |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v2.0]: MOEX-only focus -- US market deferred
- [v2.0]: Universe surgery first -- toxic symbols account for ~60% negative PnL
- [v2.0]: Dividend gap as primary alpha -- 70%+ documented closure rate on blue chips
- [v2.0]: Sector rotation MUST be in sizing pipeline, NOT combiner (architectural constraint)
- [v1.0]: OFZ-PK carry ENABLED (Sharpe +1.14) -- portfolio foundation for v2.0
- [08-01]: vol_target 0.40 for MOEX (was 0.19-0.22) -- matches 35-45% annualized vol
- [08-01]: Toxic symbols removed (GAZP, VTBR, SNGS, SNGSP, IRAO, ALRS) -- ~60% negative PnL
- [08-02]: exclude_periods as tuple of string pairs for JSON serializability and frozen dataclass compatibility
- [08-02]: filter_candles_by_exclusion in stop_loss.py (Layer 4) reused by ATR and chandelier computations
- [08-03]: DividendEntry status defaults to "paid" for backward compatibility
- [08-03]: Only "paid" dividends trigger BUY signals -- cancelled/reduced skipped
- [08-03]: T-Invest API lacks cancelled dividend data -- manual overrides required

### Pending Todos

None yet.

### Blockers/Concerns

- MOEX sector index tickers (MOEXOG, MOEXFN) need live API validation before Phase 10
- OFZ yield curve slope data source unclear -- research needed before Phase 10
- Preferred share cointegration must be validated on post-2022 data before Phase 11

## Session Continuity

Last session: 2026-03-20
Stopped at: Completed 08-03-PLAN.md (Phase 08 complete)
Resume file: None
