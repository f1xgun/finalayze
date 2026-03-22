---
gsd_state_version: 1.0
milestone: v4.0
milestone_name: Architecture Hardening
status: defining_requirements
stopped_at: null
last_updated: "2026-03-22T12:00:00.000Z"
last_activity: 2026-03-22 -- Milestone v4.0 started
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-22)

**Core value:** Autonomous profitable MOEX trading with acceptable risk limits
**Current focus:** v4.0 Architecture Hardening — defining requirements

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-03-22 — Milestone v4.0 started

## Accumulated Context

### Decisions

Decisions from v1.0-v3.0 are archived in milestones/.
Key carry-forward decisions for v4.0:

- Monitoring services standalone (not embedded in TradingLoop)
- Go/no-go is advisory report, not automated promotion
- KillSwitch uses deferred imports for CircuitLevel/AlertPriority to maintain layer boundaries
- File-based kill flag works even when DB is down
- Fire-and-forget DB persistence for metrics — never crash the trading loop

### Pending Todos

None yet.

### Blockers/Concerns

- trading_loop.py is ~1800 lines god-object — extract carefully to avoid breaking APScheduler wiring
- 140+ except Exception clauses — tightening too aggressively may crash the trading loop
- Event bus removal must not break bond_discovery.py coupon publishing

## Session Continuity

Last session: 2026-03-22
Stopped at: Defining requirements
Resume file: None
