---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: MOEX Profitability
status: completed
stopped_at: Completed 10-01-PLAN.md (Phase 10 complete)
last_updated: "2026-03-20T09:24:16.622Z"
last_activity: 2026-03-20 -- Completed 10-01 CBR Regime and Sector Allocation Steps
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 7
  completed_plans: 7
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-20)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** Phase 10 -- Macro Regime

## Current Position

Phase: 10 of 12 (Macro Regime) -- third phase of v2.0
Plan: 2 of 2 in current phase (10-01, 10-02 complete)
Status: Phase 10 Complete
Last activity: 2026-03-20 -- Completed 10-01 CBR Regime and Sector Allocation Steps

Progress: [##########] 100% (phase 10 complete)

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
| 09    | 01   | 7min     | 2     | 7     |
| 09    | 02   | 5min     | 3     | 5     |
| 10    | 01   | 5min     | 2     | 7     |
| 10    | 02   | 2min     | 1     | 2     |

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
- [09-01]: Event strategies bypass ADX implicitly + explicit is_event flag for clarity
- [09-01]: Engine hold bar safety ceiling for dividend_gap set to 60 (max of all yield tiers)
- [09-01]: Event confidence floor 0.40 applied only when event strategy fires
- [10-01]: Yield curve slope data is static dict for backtest reproducibility
- [10-01]: CBRRegimeStep inserted after BrentGate, before Copula/EVT/MetaLabel/HardCaps
- [10-01]: SectorAllocationStep handles only ru_energy and ru_finance; others pass through
- [10-02]: OFZ rotation uses relative shift (+/-0.15) for capital conservation invariant
- [10-02]: Deferred CBR_MEETINGS import inside apply_ofz_rotation to avoid circular dependency
- [09-02]: Pipeline built per-run (not at init) because MOEX steps need segment_id at run() time
- [09-02]: rub_oil_regime_signal typed as object in BacktestConfig to avoid circular import
- [09-02]: FXRate objects converted to synthetic Candle objects for RubOilRegimeSignal correlation

### Pending Todos

None yet.

### Blockers/Concerns

- MOEX sector index tickers (MOEXOG, MOEXFN) need live API validation before Phase 10
- OFZ yield curve slope data source unclear -- research needed before Phase 10
- Preferred share cointegration must be validated on post-2022 data before Phase 11

## Session Continuity

Last session: 2026-03-20
Stopped at: Completed 10-01-PLAN.md (Phase 10 complete)
Resume file: None
