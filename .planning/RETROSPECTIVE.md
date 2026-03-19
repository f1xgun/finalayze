# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — MOEX MVP

**Shipped:** 2026-03-19
**Phases:** 7 | **Plans:** 22 | **Timeline:** 22 days

### What Was Built
- Complete MOEX equity trading system with RUB-native sizing, 5 tuned strategies, MOEX holidays
- Full bond pipeline: QuantLib bond math, BondCycleProcessor, OFZ carry strategy (Sharpe +1.14)
- Autonomous TradingLoop with equity + bond + news cycles, crash recovery
- Telegram monitoring system with priority queue, trade/CBR/coupon alerts, /status + /stop commands
- Russian news pipeline: RSS (3 sources) + Telegram channel reader + LLM entity extraction
- Sandbox validation infrastructure with Docker stack and validation reporting
- Go-live configuration with real_confirmed safety guard

### What Worked
- GSD autonomous workflow executed entire milestone (discuss → plan → execute per phase) with minimal human intervention
- Wave-based parallel execution saved time on independent plans (e.g., RSS fetcher + Telegram reader in parallel)
- Plan checker caught 2 blockers (missing backtest gate, missing AUT-05 implementation) before execution — saved rework
- TDD approach produced 3,651 tests, catching issues early (e.g., YAML weight redistribution)
- Existing codebase provided strong foundation — most work was wiring, not rewriting

### What Was Inefficient
- MOEX walk-forward Sharpe still negative on aggregate despite profitable individual symbols — significant tuning effort for marginal gains
- Phase 2 (equity validation) needed 3 plans including gap closure — underestimated initial calibration complexity
- Integration checker agent didn't exist in registry — had to do cross-phase audit manually
- Some SUMMARY.md frontmatter fields (one_liner) were empty, requiring manual extraction during milestone completion

### Patterns Established
- `real_confirmed: bool = False` guard in Settings prevents accidental real-money deployment
- Bond math uses QuantLib with % of face value convention (MOEX standard, not absolute RUB)
- News pipeline uses independent error handling per source (RSS failure doesn't block Telegram and vice versa)
- Event-driven strategy at 15% weight — meaningful but technical signals dominate
- APScheduler jobs with stable IDs and replace_existing=True for crash recovery

### Key Lessons
1. Backtest validation gates must be explicit in plans — the checker caught a missing backtest-iteration run that would have shipped unvalidated preset changes
2. OFZ carry strategy works; OFZ duration rotation doesn't in hiking cycle — market regime matters more than theoretical diversification
3. event_driven shows 0 trades in backtests by design (needs live news) — can't validate via standard backtest, need sandbox with live feeds
4. MOEX data MUST come from T-Invest API — yfinance cannot fetch MOEX tickers (confirmed multiple times across phases)

### Cost Observations
- Model mix: primarily opus for orchestration + sonnet for verification/checking agents
- GSD workflow reduced manual coordination overhead significantly
- Notable: parallel Wave 1 execution (2 agents simultaneously) saved ~50% wall-clock time vs sequential

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Timeline | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 22 days | 7 | GSD autonomous workflow; plan checker verification loop |

### Cumulative Quality

| Milestone | Tests | Plans | Phases |
|-----------|-------|-------|--------|
| v1.0 | 3,651 | 22 | 7 |

### Top Lessons (Verified Across Milestones)

1. Plan checker verification prevents shipping gaps — always run before execution
2. Parallel wave execution is safe when file sets don't overlap
3. External service validation (Telegram, T-Invest, RSS) must be human-verified — can't automate without live credentials
