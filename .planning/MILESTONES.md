# Milestones

## v3.0 Production Readiness (Shipped: 2026-03-22)

**Phases completed:** 4 phases, 10 plans, 0 tasks

**Key accomplishments:**
- (none recorded)

---

## v2.0 MOEX Profitability (Shipped: 2026-03-21)

**Phases completed:** 7 phases, 16 plans, 0 tasks

**Key accomplishments:**
- (none recorded)

---

## v1.0 MOEX MVP (Shipped: 2026-03-19)

**Phases completed:** 7 phases, 22 plans
**Timeline:** 22 days (2026-02-22 → 2026-03-15)
**Codebase:** 35,199 LOC Python, 3,651 tests

**Key accomplishments:**
- MOEX equity trading with RUB-native Half-Kelly sizing, MOEX holiday calendar, 5 tuned strategies across 4 segments
- Full bond trading pipeline with QuantLib (YTM, duration, convexity), BondCycleProcessor, OFZ carry strategy (Sharpe +1.14)
- Autonomous TradingLoop with APScheduler equity + bond + news cycles, crash recovery, graceful error handling
- Telegram monitoring: priority message queue, trade/coupon/CBR alerts, /status + /stop commands, daily P&L
- Sandbox validation infrastructure: Docker stack, validation report generator, autonomous operation criteria
- Russian news pipeline: RSS fetcher (RBC, Interfax, TASS), Telegram reader, LLM entity extraction, event_driven at 15% weight

---

