---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
stopped_at: Phase 5 context gathered
last_updated: "2026-03-14T19:33:15.211Z"
last_activity: 2026-03-14 -- Completed 04-03 (Bond walk-forward validation)
progress:
  total_phases: 7
  completed_phases: 4
  total_plans: 11
  completed_plans: 11
  percent: 55
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-14)

**Core value:** Autonomous profitable MOEX trading (stocks + bonds + coupons) with risk limits, operating without human intervention
**Current focus:** Phase 4 - Bond Execution

## Current Position

Phase: 4 of 7 (Bond Execution)
Plan: 3 of 3 in current phase (COMPLETE)
Status: Phase 4 Complete
Last activity: 2026-03-14 -- Completed 04-03 (Bond walk-forward validation)

Progress: [██████░░░░] 55%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 18min
- Total execution time: 1.2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | 24min | 12min |
| 02 | 2 | 49min | 25min |

**Recent Trend:**
- Last 5 plans: 01-01 (6min), 01-02 (18min), 02-01 (11min), 02-02 (38min)
- Trend: stable

*Updated after each plan completion*
| Phase 02 P03 | 4min | 3 tasks | 5 files |
| Phase 03 P02 | 5min | 3 tasks | 6 files |
| Phase 03 P01 | 9min | 2 tasks | 6 files |
| Phase 03 P03 | 9min | 3 tasks | 8 files |
| Phase 04 P01 | 7min | 2 tasks | 9 files |
| Phase 04 P02 | 11min | 2 tasks | 5 files |
| Phase 04 P03 | 25min | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: 7 phases derived from 32 requirements; strict dependency chain (sizing -> equity validation -> bond data -> bond execution -> integration -> sandbox -> news+go-live)
- [Roadmap]: Phases 2 and 3 both depend only on Phase 1 (parallel-capable but sequential recommended)
- [Roadmap]: News pipeline deferred to Phase 7 (differentiator, not table-stake for autonomous operation)
- [01-01]: Transferred holidays as static per-year frozensets (government decrees are static)
- [01-01]: is_moex_holiday expanded to check both fixed and transferred (backward-compatible)
- [01-01]: Lazy import of moex_calendar in trading_loop to maintain dependency layering
- [01-02]: MOEX starting capital fixed at 1M RUB (not USD * FX rate)
- [01-02]: Half-Kelly with default params gives 8.33% position size (not 10-20% as initially expected)
- [02-01]: ou_mean_reversion disabled on all MOEX segments (negative Sharpe on all 3: -0.28, -0.11, -0.55)
- [02-01]: Weights redistributed proportionally after OU disable; all presets sum to 1.00
- [02-01]: ru_finance added to UNIVERSE (7 symbols in run_iteration, 4 in isolation)
- [02-02]: All 8 candidate pairs failed cointegration (p>0.05) -- pairs strategy disabled on all MOEX segments
- [02-02]: Optuna-tuned MOEX-specific params (ADX thresholds, BB std_dev, confidence levels) distinct from US defaults
- [02-02]: Walk-forward targets not fully met (avg Sharpe negative) -- best-effort accepted after 5 iterations per plan
- [02-02]: Individual symbols profitable (YNDX Sharpe +0.88, ROSN +0.65) but segment averages dragged by losing symbols
- [Phase 02]: All pruned MOEX symbols restored for future news/sentiment integration (Phase 7)
- [Phase 02]: Three-quarter Kelly (0.75) kept for MOEX segments -- 3x position sizes vs default
- [Phase 03]: Yield curve parsed from CBR HTML using lxml.html; async DB persistence via asyncio boundary in sync refresh()
- [Phase 03]: QuantLib cleanPrice/bondYield use % of face (MOEX convention), not absolute RUB
- [Phase 03]: liquidity_flag as proxy for 10M RUB/day turnover (T-Invest API limitation)
- [Phase 03]: OFZ classified by class_code (TQOB/TQOD) or sector containing "government"
- [Phase 03]: CouponEvent emitted on record_date match; bond candle prices in % of face value
- [Phase 03]: Cross-validation tolerance 100bps (not 1bps) due to business-day vs calendar schedule difference
- [Phase 03]: FloatingRateBond requires 1-year historical fixings backfill with flat rate (MVP)
- [Phase 04]: face_value renamed to unit_cost (default 1000) for DV01 sizing backward compat
- [Phase 04]: make_bond_broker shares AsyncClient (single gRPC channel) with equity broker
- [Phase 04]: reconcile_with_broker adds unknown bonds to Core layer with zeroed entry data
- [Phase 04]: Limit orders (not market) for bond execution; 2-min fill timeout with 2s polling
- [Phase 04]: Partial fills kept in ledger; remainder cancelled; no retry (next cycle tries again)
- [Phase 04]: Transaction costs estimated from MOEX bond cost constants (0.05% + 5bps + 3bps)
- [04-03]: ru_ofz_pk carry strategy ENABLED (Sharpe +1.14, PF 25.22, DD 1.0%, Win Rate 78.6%)
- [04-03]: ru_ofz_pd duration rotation DISABLED (Sharpe -0.16, negative PnL in 2022-2025 hiking cycle)
- [04-03]: Raw Sharpe (rf=0) used for bond acceptance checks

### Pending Todos

None yet.

### Blockers/Concerns

- ~~RUB position sizing bug~~ FIXED in 01-02 (1M RUB starting capital, 8% position sizing)
- ~~MOEX-specific ADX threshold calibration~~ DONE in 02-02 (Optuna-tuned per segment)
- OFZ-PK floater duration formula needs validation during Phase 4 planning
- Russian news RSS URLs have MEDIUM confidence -- validate at Phase 7 implementation

## Session Continuity

Last session: 2026-03-14T19:33:15.208Z
Stopped at: Phase 5 context gathered
Resume file: .planning/phases/05-integration-and-telegram/05-CONTEXT.md
