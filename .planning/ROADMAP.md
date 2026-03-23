# Roadmap: Finalayze

## Milestones

- ✅ **v1.0 MOEX MVP** -- Phases 1-7 (shipped 2026-03-19)
- ✅ **v2.0 MOEX Profitability** -- Phases 8-14 (shipped 2026-03-21)
- ✅ **v3.0 Production Readiness** -- Phases 15-18 (shipped 2026-03-22)
- ✅ **v4.0 Architecture Hardening** -- Phases 19-22 (shipped 2026-03-22)
- 🚧 **v5.0 Data Flow Correctness & Live-Backtest Parity** -- Phases 23-26 (in progress)

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

### v5.0 Data Flow Correctness & Live-Backtest Parity (In Progress)

**Milestone Goal:** Fix critical data flow bugs -- SELL order sizing, sector exposure calculation, live/backtest risk pipeline divergence, data validation gaps, and news pipeline no-op waste.

- [x] **Phase 23: Order Sizing Bug Fixes** - Fix money-losing SELL sizing, sector exposure, and CAUTION threshold bugs (completed 2026-03-23)
- [ ] **Phase 24: Live-Backtest Parity** - Wire PositionSizingPipeline, trailing stops, pre-trade checks, and re-entry guard in live path
- [ ] **Phase 25: Data Validation and Infrastructure** - Wire DataNormalizer, candle staleness, IMOEX volume fix, gRPC channel reuse, Brent caching
- [ ] **Phase 26: News Pipeline Fixes** - Disable no-op news cycle, add sentiment decay, fix ticker mismatch, deduplicate Telegram messages

## Phase Details

### Phase 23: Order Sizing Bug Fixes
**Goal**: SELL orders, sector exposure, and CAUTION thresholds produce correct values -- no over-sells, no cross-contaminated prices, no hardcoded thresholds
**Depends on**: Nothing (first v5.0 phase)
**Requirements**: SIZE-01, SIZE-02, SIZE-03
**Success Criteria** (what must be TRUE):
  1. When the system generates a SELL order, the quantity equals the actual held position for that symbol -- Kelly sizing is not applied to exits
  2. Sector exposure pre-trade check computes each position's notional value using that position's own last traded price -- not the price of the instrument currently being evaluated
  3. CAUTION confidence threshold is computed as `segment.min_combined_confidence * 1.2` and changes when segment config changes -- no literal 0.6 in the code path
**Plans:** 1/1 plans complete
Plans:
- [x] 23-01-PLAN.md -- Fix SELL sizing, sector exposure prices, and CAUTION threshold bugs

### Phase 24: Live-Backtest Parity
**Goal**: Live trading loop risk pipeline matches the backtest engine -- same sizing steps, same trailing stop behavior, same pre-trade checks, same re-entry guard
**Depends on**: Phase 23
**Requirements**: PARITY-01, PARITY-02, PARITY-03, PARITY-04
**Success Criteria** (what must be TRUE):
  1. Live trading loop instantiates PositionSizingPipeline with all steps (VolTarget, Regime, MetaLabel, Copula, EVT, HardCaps) and calls it for every BUY signal -- the same pipeline that backtest engine uses
  2. Live trailing stops ratchet the stop price upward after an activation threshold is reached, and never ratchet downward -- matching SimulatedBroker trailing stop state machine
  3. All 14 pre-trade checks in live path receive their required parameters (stop_loss_price, has_pending_order, regime_state, strategy_name, correlations) -- no check is skipped due to missing input
  4. When a symbol is stopped out in a given equity cycle, the same cycle does not re-enter that symbol -- a per-cycle exclusion set prevents immediate re-buy after stop-loss
**Plans**: TBD

### Phase 25: Data Validation and Infrastructure
**Goal**: Market data is validated before strategy consumption, stale data is detected and skipped, and data fetching is efficient with no redundant connections or downloads
**Depends on**: Nothing (data layer is independent of sizing/parity fixes)
**Requirements**: DATA-01, DATA-02, DATA-03, INFRA-01, INFRA-02
**Success Criteria** (what must be TRUE):
  1. DataNormalizer.validate() runs on every batch of fetched candles before they reach strategy processing -- candles with negative prices, low > high, or zero volume are rejected with a warning log
  2. When candle data for an instrument is older than 2x the expected timeframe interval (configurable), a warning is logged and the instrument is skipped for that cycle -- no trading on stale data
  3. IMOEX index candles store share volume (column index 5) not turnover value (column index 4) -- volume-based indicators on IMOEX produce correct readings
  4. TinkoffFetcher maintains a persistent gRPC channel that is reused across calls within the same session -- connection setup overhead is eliminated for consecutive data requests
  5. Brent crude candles are cached via _cached_fetch() in MarketDataLoader -- repeated backtest runs do not re-download Brent data from yfinance
**Plans**: TBD

### Phase 26: News Pipeline Fixes
**Goal**: News pipeline does not waste LLM tokens when unused, sentiment ages out properly, ticker extraction is correct, and Telegram messages are not processed twice
**Depends on**: Nothing (news pipeline is independent of other v5.0 work)
**Requirements**: NEWS-01, NEWS-02, NEWS-03, NEWS-04
**Success Criteria** (what must be TRUE):
  1. When no segment in the active configuration has event_driven strategy enabled, the news cycle is skipped entirely -- zero LLM API calls are made
  2. Cached sentiment scores decay exponentially with a configurable half-life (default 4 hours) -- a sentiment score cached 8 hours ago has decayed to 25% of its original value
  3. Entity extractor _VALID_TICKERS map contains "TCSG" (the MOEX ticker) and does not contain bare "T" -- news mentioning T-Bank resolves to TCSG
  4. Telegram reader tracks processed message URLs and skips duplicates within a configurable time window -- the same Telegram post is not sent to the LLM twice
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 23 -> 24 -> 25 -> 26
Note: Phases 25 and 26 have no dependency on 23/24 and could run in parallel after Phase 23 completes.

| Phase | Milestone | Plans | Status | Completed |
|-------|-----------|-------|--------|-----------|
| 1-7 | v1.0 | 22/22 | Complete | 2026-03-19 |
| 8-14 | v2.0 | 16/16 | Complete | 2026-03-21 |
| 15-18 | v3.0 | 10/10 | Complete | 2026-03-22 |
| 19-22 | v4.0 | 10/10 | Complete | 2026-03-22 |
| 23. Order Sizing Bug Fixes | v5.0 | 1/1 | Complete   | 2026-03-23 |
| 24. Live-Backtest Parity | v5.0 | 0/TBD | Not started | - |
| 25. Data Validation and Infrastructure | v5.0 | 0/TBD | Not started | - |
| 26. News Pipeline Fixes | v5.0 | 0/TBD | Not started | - |
