---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: MOEX Profitability
status: defining_requirements
stopped_at: null
last_updated: "2026-03-20T12:00:00.000Z"
last_activity: 2026-03-20 -- Milestone v2.0 started
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-20)

**Core value:** Autonomous profitable MOEX trading (stocks + bonds + coupons) with risk limits, operating without human intervention
**Current focus:** v2.0 MOEX Profitability — make MOEX equity profitable through universe cleanup, MOEX-native strategies, and ML

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-03-20 — Milestone v2.0 started

Progress: [░░░░░░░░░░] 0%

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
v1.0 decisions carried forward:

- [v1.0]: ou_mean_reversion disabled on all MOEX segments (negative Sharpe: -0.28, -0.11, -0.55)
- [v1.0]: All 8 candidate pairs failed cointegration (p>0.05) — pairs strategy disabled on all MOEX
- [v1.0]: Individual symbols profitable (YNDX +0.88, ROSN +0.65) but segment averages dragged by losing symbols
- [v1.0]: OFZ-PK carry ENABLED (Sharpe +1.14, PF 25.22), OFZ-PD duration rotation DISABLED (Sharpe -0.16)
- [v1.0]: event_driven at 0.15 weight, 0 backtest trades (needs live news)
- [v1.0]: Three-quarter Kelly (0.75) for MOEX — 3x position sizes vs default
- [v2.0-CEO]: GAZP, VTBR, SNGS, IRAO, ALRS identified as toxic symbols (~60% of negative PnL)
- [v2.0-CEO]: Dividend gap closure documented as primary MOEX alpha source (70%+ gap closure within 30-60 days)
- [v2.0-CEO]: OFZ-PK carry is portfolio foundation (20% annual at 21% CBR rate)
- [v2.0-CEO]: US market development deferred — MOEX-only focus

### Pending Todos

None yet.

### Blockers/Concerns

- MOEX equity WF Sharpe consistently negative (-0.01 to -0.09 across all iterations)
- moex_dividends.yaml has only 43 events across 6 symbols — data gap for dividend strategy
- rub_oil_regime.py exists but not wired into equity sizing pipeline
- 2022 sanctions regime makes backtest data noisy (structural break)

## Session Continuity

Last session: 2026-03-20
Stopped at: Defining v2.0 requirements
