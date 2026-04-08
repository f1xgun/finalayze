# Roadmap: Finalayze

## Milestones

- ✅ **v1.0 MOEX MVP** -- Phases 1-7 (shipped 2026-03-19)
- ✅ **v2.0 MOEX Profitability** -- Phases 8-14 (shipped 2026-03-21)
- ✅ **v3.0 Production Readiness** -- Phases 15-18 (shipped 2026-03-22)
- ✅ **v4.0 Architecture Hardening** -- Phases 19-22 (shipped 2026-03-22)
- ✅ **v5.0 Data Flow Correctness** -- Phases 23-27 (shipped 2026-03-24)
- ✅ **v6.0 Sandbox Stability & Observability** -- Phases 28-31 (shipped 2026-03-30)
- 🚧 **v7.0 Agent Intelligence & Experiment Framework** -- Phases 32-35

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

- [x] **Phase 28: Operational Hygiene** - Fix stale tickers, add market-hours gate, LLM dedup, alerter resilience (completed 2026-03-30)
- [x] **Phase 29: Core Stability** - gRPC event loop isolation and Loki log pipeline fix (completed 2026-03-30)
- [x] **Phase 30: Broker Resilience** - gRPC 70001 reconnect, portfolio cache fallback, FX rate fallback (completed 2026-03-30)
- [x] **Phase 31: Data Capture** - DB persistence for orders, signals, news articles, sentiment scores (completed 2026-03-30)

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
**Plans:** 2/2 plans complete
Plans:
- [x] 28-01-PLAN.md -- Market-hours gate and stale ticker fixes (OPS-01, OPS-02)
- [x] 28-02-PLAN.md -- LLM article dedup and alerter resilience (OPS-03, OPS-04)

### Phase 29: Core Stability
**Goal**: Strategy cycles fire reliably within 5 minutes of scheduled time and all container logs are queryable in Grafana/Loki
**Depends on**: Phase 28
**Requirements**: GRPC-01, OBS-01, OBS-02
**Success Criteria** (what must be TRUE):
  1. gRPC calls (TinkoffBroker, TinkoffFetcher) run on a dedicated event loop thread isolated from the main asyncio loop -- no BlockingIOError from PollerCompletionQueue, strategy cycles complete within 5 minutes of their scheduled time
  2. Promtail ships Docker container logs from all 7 containers to Loki -- `/var/lib/docker/containers` is mounted and JSON log format is correctly parsed
  3. Grafana log dashboard queries return results for all containers with at least 30 days of retention -- logs from any container are searchable within seconds of being emitted
**Plans:** 2/2 plans complete
Plans:
- [x] 29-01-PLAN.md -- gRPC event loop isolation (GRPC-01)
- [x] 29-02-PLAN.md -- Loki log pipeline fix (OBS-01, OBS-02)

### Phase 30: Broker Resilience
**Goal**: Trading continues through T-Bank API failures and FX rate is always available for position sizing
**Depends on**: Phase 29 (requires gRPC event loop isolation to be in place before adding reconnect logic)
**Requirements**: GRPC-02, GRPC-03, OBS-03
**Success Criteria** (what must be TRUE):
  1. When TinkoffBroker receives StatusCode.INTERNAL (error 70001), it automatically resets the gRPC channel and retries -- recovery happens within one retry cycle without multi-hour outage windows
  2. When portfolio fetch fails, the strategy cycle continues using the last successfully fetched portfolio state -- positions, balances, and risk checks use cached data instead of skipping the entire cycle
  3. When gRPC FX rate fetch fails, USD/RUB rate is fetched from CBR XML API as a background job -- the `finalayze_usd_rub_rate` Prometheus metric is never zero during market hours
**Plans:** 2/2 plans complete
Plans:
- [x] 30-01-PLAN.md -- Portfolio cache fallback and 70001 auto-reconnect (GRPC-02, GRPC-03)
- [x] 30-02-PLAN.md -- FX rate CBR fallback and Prometheus metric (OBS-03)

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
**Plans**: 2 plans
Plans:
- [x] 31-01-PLAN.md -- Fire-and-forget helper, order and signal persistence (PERSIST-01, PERSIST-02, PERSIST-05)
- [x] 31-02-PLAN.md -- News article and sentiment score persistence (PERSIST-03, PERSIST-04)

### v7.0 Agent Intelligence & Experiment Framework

**Milestone Goal:** Build a scientific decision-making system: structured debate protocol for agent recommendations, experiment registry with pre-defined success criteria, backtest-based A/B testing with interaction effects, and Streamlit Experiment Lab UI for full lifecycle visibility.

- [x] **Phase 32: Critical Sandbox Fixes** - Fix _CANDLE_LOOKBACK=210, kill switch startup check, rollout default for sandbox mode (prerequisite for meaningful experiments) (completed 2026-04-07)
- [x] **Phase 33: Structured Debate Protocol** - Evidence-based agent output format (claim + source + prediction + risk), arbiter agent for fact-checking, debate state tracking (completed 2026-04-08)
- [x] **Phase 34: Experiment Registry & Runner** - Hypothesis lifecycle (define → criteria → run → compare → verdict), parameterized backtest runner with hypothesis_id, interaction testing (A, B, A+B), integration with history.jsonl (completed 2026-04-08)
- [ ] **Phase 35: Experiment Lab UI** - Streamlit app for experiment lifecycle: hypothesis context, pre-defined success criteria, execution status, results vs expectations, decision history

### Phase 32: Critical Sandbox Fixes
**Goal**: All strategies function correctly in MOEX sandbox mode, safety defaults prevent accidental production-level risk, news pipeline activated, and signal diagnostics available
**Depends on**: Nothing (zero-risk fixes)
**Requirements**: SANDBOX-FIX-01, SANDBOX-FIX-02, SANDBOX-FIX-03, SANDBOX-FIX-04, SANDBOX-FIX-05, SANDBOX-FIX-06, SANDBOX-FIX-07, SANDBOX-FIX-08, SANDBOX-FIX-09, SANDBOX-FIX-10
**Success Criteria** (what must be TRUE):
  1. `_CANDLE_LOOKBACK >= 210` in trading loop -- RSI2 Connors (needs SMA(200)), dual_momentum (needs 126 bars), and OU mean reversion (needs 126 bars) all receive sufficient data in live mode
  2. `TradingLoop.start()` checks `KillSwitch.is_killed` before starting scheduler -- a killed system does not resume trading on Docker restart
  3. When `FINALAYZE_MODE=sandbox` and `rollout_phase` is not explicitly set, the effective rollout phase is MINIMAL (not FULL) -- sandbox always starts with conservative risk limits
  4. Staleness threshold handles weekends (72h) and MOEX holidays -- Monday morning and post-New-Year cycles not blocked
  5. TinkoffFetcher wrapped in CachingFetcher and RateLimiter in sandbox mode -- no repeated API calls, no throttling
  6. event_driven enabled for ru_blue_chips, ru_energy, ru_finance with LLM setup documented -- news pipeline produces sentiment scores
  7. ValidationLogger tracks per-gate signal drops (no_bars, below_threshold, pre_trade_rejected) -- signal loss is diagnosable
  8. ML profit_factor gate computes actual PF from fold predictions -- gate no longer always fails
  9. ML Brier gate uses calibrated probabilities -- calibrator applied during walk-forward evaluation
**Plans:** 4/4 plans complete
Plans:
- [x] 32-01-PLAN.md -- Data pipeline and safety defaults (SANDBOX-FIX-01, SANDBOX-FIX-02, SANDBOX-FIX-03, SANDBOX-FIX-04)
- [x] 32-02-PLAN.md -- Sandbox data wiring, news pipeline, signal diagnostics (SANDBOX-FIX-05, SANDBOX-FIX-06, SANDBOX-FIX-07, SANDBOX-FIX-08)
- [x] 32-03-PLAN.md -- ML quality gate fixes (SANDBOX-FIX-09, SANDBOX-FIX-10)
- [x] 32-04-PLAN.md -- Gap closure: wire per-fold calibrator in walk-forward Brier evaluation (SANDBOX-FIX-10)

### Phase 33: Structured Debate Protocol
**Goal**: Agent recommendations include verifiable evidence, conflicts are detected automatically, and unresolved conflicts escalate to experiments
**Depends on**: Phase 32 (need working sandbox to validate experiment results)
**Requirements**: DEBATE-01, DEBATE-02, DEBATE-03
**Success Criteria** (what must be TRUE):
  1. Agent output schema enforces structured claims with source references (file:line or metric value) -- no unsourced assertions in agent recommendations
  2. An arbiter agent can take two conflicting agent outputs and produce a fact-check report showing which claims are verified, which are contradicted, and which are untestable
  3. Debate state (claims, conflicts, resolutions) is persisted in `.planning/debates/` for audit trail -- every multi-agent decision has a traceable history
**Plans:** 2/2 plans complete
Plans:
- [x] 33-01-PLAN.md — Debate protocol schemas (TDD): Claim, AgentOutput, FactCheckReport, DebateState (DEBATE-01, DEBATE-02, DEBATE-03)
- [x] 33-02-PLAN.md — Arbiter agent, DebateManager CRUD, debates directory (DEBATE-02, DEBATE-03)

### Phase 34: Experiment Registry & Runner
**Goal**: Hypotheses are defined with success criteria before execution, backtest experiments test proposals in isolation and combination, and results are structured for comparison
**Depends on**: Phase 33 (debate protocol identifies which conflicts need experiments)
**Requirements**: EXP-01, EXP-02, EXP-03, EXP-04
**Success Criteria** (what must be TRUE):
  1. Experiment registry stores hypothesis, success criteria (metric + threshold), status, and linked backtest results -- every experiment has a pre-registered definition
  2. `run_iteration.py --hypothesis <id>` runs a parameterized backtest and links results to the hypothesis -- experiment results are automatically associated with their hypothesis
  3. Interaction testing: given hypotheses A and B, the runner executes A-only, B-only, and A+B runs and compares all three -- combination effects are measured, not assumed
  4. Experiment verdicts (ACCEPT/REJECT/INCONCLUSIVE) are recorded with reasoning and linked to the debate that triggered them
**Plans:** 2/2 plans complete
Plans:
- [x] 34-01-PLAN.md — Experiment schemas (TDD) + ExperimentManager CRUD, verdict, debate linkage (EXP-01, EXP-04)
- [x] 34-02-PLAN.md — run_iteration.py --hypothesis extension + interaction test runner (EXP-02, EXP-03)

### Phase 35: Experiment Lab UI
**Goal**: Full experiment lifecycle is visible in a Streamlit web app -- from debate context through execution to final decision
**Depends on**: Phase 34 (needs experiment registry and results to display)
**Requirements**: UI-EXP-01, UI-EXP-02, UI-EXP-03
**Success Criteria** (what must be TRUE):
  1. Experiment list page shows all experiments with status (PENDING/RUNNING/COMPLETED), hypothesis summary, and key metrics -- at a glance, what experiments exist and their state
  2. Experiment detail page shows: debate context (why), success criteria (what we expect), backtest results with charts (what happened), and A vs B vs A+B comparison table -- complete decision context on one screen
  3. Decision history page shows accepted/rejected experiments with reasoning -- the team can review past decisions and understand why the system is configured the way it is
**Plans:** 2 plans
Plans:
- [x] 34-01-PLAN.md — Experiment schemas (TDD) + ExperimentManager CRUD, verdict, debate linkage (EXP-01, EXP-04)
- [ ] 34-02-PLAN.md — run_iteration.py --hypothesis extension + interaction test runner (EXP-02, EXP-03)

## Progress

**Execution Order:**

v6.0: 28 -> 29 -> 30 -> 31 (all complete)
v7.0: 32 -> 33 -> 34 -> 35

| Phase | Milestone | Plans | Status | Completed |
|-------|-----------|-------|--------|-----------|
| 1-7 | v1.0 | 22/22 | Complete | 2026-03-19 |
| 8-14 | v2.0 | 16/16 | Complete | 2026-03-21 |
| 15-18 | v3.0 | 10/10 | Complete | 2026-03-22 |
| 19-22 | v4.0 | 10/10 | Complete | 2026-03-22 |
| 23-27 | v5.0 | 9/9 | Complete | 2026-03-24 |
| 28. Operational Hygiene | v6.0 | 2/2 | Complete | 2026-03-30 |
| 29. Core Stability | v6.0 | 2/2 | Complete | 2026-03-30 |
| 30. Broker Resilience | v6.0 | 2/2 | Complete | 2026-03-30 |
| 31. Data Capture | v6.0 | 2/2 | Complete | 2026-03-30 |
| 32. Critical Sandbox Fixes | v7.0 | 4/4 | Complete    | 2026-04-07 |
| 33. Structured Debate Protocol | v7.0 | 2/2 | Complete    | 2026-04-08 |
| 34. Experiment Registry & Runner | v7.0 | 2/2 | Complete   | 2026-04-08 |
| 35. Experiment Lab UI | v7.0 | TBD | Pending | -- |
