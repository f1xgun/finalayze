# Roadmap: Finalayze

## Milestones

- ✅ **v1.0 MOEX MVP** -- Phases 1-7 (shipped 2026-03-19)
- ✅ **v2.0 MOEX Profitability** -- Phases 8-14 (shipped 2026-03-21)
- ✅ **v3.0 Production Readiness** -- Phases 15-18 (shipped 2026-03-22)
- ✅ **v4.0 Architecture Hardening** -- Phases 19-22 (shipped 2026-03-22)
- ✅ **v5.0 Data Flow Correctness** -- Phases 23-27 (shipped 2026-03-24)
- 🚧 **v6.0 Sandbox Stability & Observability** -- Phases 28-31 (in progress)

## Phases

<details>
<summary>✅ v1.0 MOEX MVP (Phases 1-7) -- SHIPPED 2026-03-19</summary>

- [x] Phase 1: MOEX Equity Foundation (2/2 plans) -- completed 2026-03-14
- [x] Phase 2: MOEX Equity Validation (3/3 plans) -- completed 2026-03-14
- [x] Phase 3: Bond Data Pipeline (3/3 plans) -- completed 2026-03-14
- [x] Phase 4: Bond Execution (3/3 plans) -- completed 2026-03-14
- [x] Phase 5: Integration and Telegram (4/4 plans) -- completed 2026-03-14
- [x] Phase 6: Sandbox Validation (4/4 plans) -- completed 2026-03-15
- [x] Phase 7: News Pipeline and Go-Live (3/3 plans) -- completed 2026-03-15

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

<details>
<summary>✅ v2.0 MOEX Profitability (Phases 8-14) -- SHIPPED 2026-03-21</summary>

- [x] Phase 8: Data Foundation (3/3 plans) -- completed 2026-03-20
- [x] Phase 9: Strategy Wiring (2/2 plans) -- completed 2026-03-20
- [x] Phase 10: Macro Regime (2/2 plans) -- completed 2026-03-20
- [x] Phase 11: Advanced Strategies and ML (4/4 plans) -- completed 2026-03-21
- [x] Phase 12: Portfolio Assembly (2/2 plans) -- completed 2026-03-21
- [x] Phase 13: Script Wiring Fixes (1/1 plan) -- completed 2026-03-21 (gap closure)
- [x] Phase 14: Bond Backtest and Portfolio CLI (2/2 plans) -- completed 2026-03-21 (gap closure)

Full details: `.planning/milestones/v2.0-ROADMAP.md`

</details>

<details>
<summary>✅ v3.0 Production Readiness (Phases 15-18) -- SHIPPED 2026-03-22</summary>

- [x] Phase 15: Schemas, Config, and Rollout Foundation (2/2 plans) -- completed 2026-03-21
- [x] Phase 16: Sandbox Monitoring and Go/No-Go Gate (3/3 plans) -- completed 2026-03-21
- [x] Phase 17: Production Operations (3/3 plans) -- completed 2026-03-21
- [x] Phase 18: Dashboard and API Integration (2/2 plans) -- completed 2026-03-21

Full details: `.planning/milestones/v3.0-ROADMAP.md`

</details>

<details>
<summary>✅ v4.0 Architecture Hardening (Phases 19-22) -- SHIPPED 2026-03-22</summary>

- [x] Phase 19: Concurrency Safety and Integration Fixes (2/2 plans) -- completed 2026-03-22
- [x] Phase 20: Async Correctness and Resource Management (3/3 plans) -- completed 2026-03-22
- [x] Phase 21: Error Handling Hardening (2/2 plans) -- completed 2026-03-22
- [x] Phase 22: Dependency Layer Cleanup (3/3 plans) -- completed 2026-03-22

Full details: `.planning/milestones/v4.0-ROADMAP.md`

</details>

<details>
<summary>✅ v5.0 Data Flow Correctness (Phases 23-27) -- SHIPPED 2026-03-24</summary>

- [x] Phase 23: Order Sizing Bug Fixes (1/1 plan) -- completed 2026-03-23
- [x] Phase 24: Live-Backtest Parity (2/2 plans) -- completed 2026-03-23
- [x] Phase 25: Data Validation and Infrastructure (2/2 plans) -- completed 2026-03-24
- [x] Phase 26: News Pipeline Fixes (2/2 plans) -- completed 2026-03-24
- [x] Phase 27: Intelligent News Impact Analysis (2/2 plans) -- completed 2026-03-24

Full details: `.planning/milestones/v5.0-ROADMAP.md`

</details>

### v6.0 Sandbox Stability & Observability (In Progress)

**Milestone Goal:** Fix all critical issues discovered during week-long sandbox validation run (March 20-30) to make the system production-ready. Stable 5-min strategy cycles, complete audit trail, operational log pipeline, and resilient broker connectivity.

- [ ] **Phase 28: Operational Hygiene** - Fix stale tickers, add market-hours gate, LLM dedup, alerter resilience
- [ ] **Phase 29: Core Stability** - gRPC event loop isolation and Loki log pipeline fix
- [ ] **Phase 30: Broker Resilience** - gRPC 70001 reconnect, portfolio cache fallback, FX rate fallback
- [ ] **Phase 31: Data Capture** - DB persistence for orders, signals, news articles, sentiment scores

## Phase Details

### Phase 28: Operational Hygiene
**Goal**: Strategy cycles only fire during MOEX market hours with correct ticker symbols, LLM quota is not wasted on duplicate articles, and Telegram alerter failures do not block trading
**Depends on**: Nothing (zero-risk config and guard fixes)
**Requirements**: OPS-01, OPS-02, OPS-03, OPS-04
**Success Criteria** (what must be TRUE):
  1. Strategy cycle checks MOEX market hours before executing and skips the cycle with a log message when the market is closed -- no cycles processing 0 instruments outside 07:00-15:45 UTC
  2. config/segments.py contains only valid MOEX tickers -- FIVE, FIXP, POLY are removed; YNDX is replaced with YDEX; HHRU is replaced with HH (if valid on MOEX)
  3. News articles already seen within the last 24 hours are skipped before being sent to the LLM -- duplicate content does not consume LLM API quota
  4. If Telegram alerter fails to connect at startup, the trading loop launches normally and queues alerts for the next successful connection -- no startup crash from invalid or missing Telegram token
**Plans:** 1/2 plans executed
Plans:
- [x] 28-01-PLAN.md -- Market-hours gate and stale ticker fixes (OPS-01, OPS-02)
- [ ] 28-02-PLAN.md -- LLM article dedup and alerter resilience (OPS-03, OPS-04)

### Phase 29: Core Stability
**Goal**: Strategy cycles fire reliably within 5 minutes of scheduled time and all container logs are queryable in Grafana/Loki
**Depends on**: Phase 28
**Requirements**: GRPC-01, OBS-01, OBS-02
**Success Criteria** (what must be TRUE):
  1. gRPC calls (TinkoffBroker, TinkoffFetcher) run on a dedicated event loop thread isolated from the main asyncio loop -- no BlockingIOError from PollerCompletionQueue, strategy cycles complete within 5 minutes of their scheduled time
  2. Promtail ships Docker container logs from all 7 containers to Loki -- `/var/lib/docker/containers` is mounted and JSON log format is correctly parsed
  3. Grafana log dashboard queries return results for all containers with at least 30 days of retention -- logs from any container are searchable within seconds of being emitted
**Plans**: TBD

### Phase 30: Broker Resilience
**Goal**: Trading continues through T-Bank API failures and FX rate is always available for position sizing
**Depends on**: Phase 29 (requires gRPC event loop isolation to be in place before adding reconnect logic)
**Requirements**: GRPC-02, GRPC-03, OBS-03
**Success Criteria** (what must be TRUE):
  1. When TinkoffBroker receives StatusCode.INTERNAL (error 70001), it automatically resets the gRPC channel and retries -- recovery happens within one retry cycle without multi-hour outage windows
  2. When portfolio fetch fails, the strategy cycle continues using the last successfully fetched portfolio state -- positions, balances, and risk checks use cached data instead of skipping the entire cycle
  3. When gRPC FX rate fetch fails, USD/RUB rate is fetched from CBR XML API as a background job -- the `finalayze_usd_rub_rate` Prometheus metric is never zero during market hours
**Plans**: TBD

### Phase 31: Data Capture
**Goal**: Every trade, signal, news article, and sentiment score is persisted to the database for audit trail and future analysis
**Depends on**: Phase 29 (stable cycles produce meaningful data; 60-min drift would create misleading timestamps)
**Requirements**: PERSIST-01, PERSIST-02, PERSIST-03, PERSIST-04, PERSIST-05
**Success Criteria** (what must be TRUE):
  1. After an order is filled, a row appears in the `orders` table with symbol, side, quantity, fill_price, order_id, and timestamp -- every executed trade has a permanent record
  2. When a strategy generates a signal, a row appears in the `signals` table with strategy name, symbol, direction, confidence, and reasoning -- the decision-making trail is preserved
  3. When a news article is processed, a row appears in the `news_articles` table with title, source, published_at, and content hash -- all analyzed news is recorded
  4. When sentiment is computed for a ticker, a row appears in the `sentiment_scores` table with ticker, score, source, and timestamp -- sentiment history is queryable
  5. If any DB write fails, the failure is logged with structlog and a `db_write_failures` Prometheus counter is incremented -- the trading loop and consecutive error counter are never affected by DB issues
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 28 -> 29 -> 30 -> 31
Note: Phases 30 and 31 both depend on Phase 29 but are independent of each other -- they can run in parallel.

| Phase | Milestone | Plans | Status | Completed |
|-------|-----------|-------|--------|-----------|
| 1-7 | v1.0 | 22/22 | Complete | 2026-03-19 |
| 8-14 | v2.0 | 16/16 | Complete | 2026-03-21 |
| 15-18 | v3.0 | 10/10 | Complete | 2026-03-22 |
| 19-22 | v4.0 | 10/10 | Complete | 2026-03-22 |
| 23-27 | v5.0 | 9/9 | Complete | 2026-03-24 |
| 28. Operational Hygiene | v6.0 | 1/2 | In Progress|  |
| 29. Core Stability | v6.0 | 0/TBD | Not started | - |
| 30. Broker Resilience | v6.0 | 0/TBD | Not started | - |
| 31. Data Capture | v6.0 | 0/TBD | Not started | - |
