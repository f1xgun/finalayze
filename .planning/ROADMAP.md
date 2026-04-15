# Roadmap: Finalayze

## Milestones

- ✅ **v1.0 MOEX MVP** -- Phases 1-7 (shipped 2026-03-19)
- ✅ **v2.0 MOEX Profitability** -- Phases 8-14 (shipped 2026-03-21)
- ✅ **v3.0 Production Readiness** -- Phases 15-18 (shipped 2026-03-22)
- ✅ **v4.0 Architecture Hardening** -- Phases 19-22 (shipped 2026-03-22)
- ✅ **v5.0 Data Flow Correctness** -- Phases 23-27 (shipped 2026-03-24)
- ✅ **v6.0 Sandbox Stability & Observability** -- Phases 28-31 (shipped 2026-03-30)
- ✅ **v7.0 Agent Intelligence & Experiment Framework** -- Phases 32-35 (shipped 2026-04-12)
- ✅ **v8.0 Agent Integration & Autonomous Decision Loop** -- Phases 36-39 (shipped 2026-04-12)
- ✅ **v9.0 ML AutoResearch & MOEX Adaptation** -- Phases 40-44 (shipped 2026-04-13)
- ✅ **v9.1 MOEX ML Model Quality** -- Phases 45-48 (shipped 2026-04-14)
- 🚧 **v10.0 Runtime LLM Trading Agents** -- Phases 49-53 (in progress)

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

<details>
<summary>✅ v6.0 Sandbox Stability & Observability (Phases 28-31) -- SHIPPED 2026-03-30</summary>

- [x] **Phase 28: Operational Hygiene** - Fix stale tickers, add market-hours gate, LLM dedup, alerter resilience (completed 2026-03-30)
- [x] **Phase 29: Core Stability** - gRPC event loop isolation and Loki log pipeline fix (completed 2026-03-30)
- [x] **Phase 30: Broker Resilience** - gRPC 70001 reconnect, portfolio cache fallback, FX rate fallback (completed 2026-03-30)
- [x] **Phase 31: Data Capture** - DB persistence for orders, signals, news articles, sentiment scores (completed 2026-03-30)

Full details in Phase Details section below (collapsed milestone).

</details>

<details>
<summary>✅ v7.0 Agent Intelligence & Experiment Framework (Phases 32-35) -- SHIPPED 2026-04-12</summary>

- [x] **Phase 32: Critical Sandbox Fixes** - Fix _CANDLE_LOOKBACK=210, kill switch startup check, rollout default for sandbox mode (completed 2026-04-07)
- [x] **Phase 33: Structured Debate Protocol** - Evidence-based agent output format, arbiter agent, debate state tracking (completed 2026-04-08)
- [x] **Phase 34: Experiment Registry & Runner** - Hypothesis lifecycle, parameterized backtest runner, interaction testing (completed 2026-04-08)
- [x] **Phase 35: Experiment Lab UI** - Streamlit experiment lifecycle: hypothesis context, success criteria, results, decision history (completed 2026-04-08)

</details>

<details>
<summary>✅ v8.0 Agent Integration & Autonomous Decision Loop (Phases 36-39) -- SHIPPED 2026-04-12</summary>

- [x] **Phase 36: Conflict Detection Foundation** - Agents emit structured AgentOutput; ConflictDetector with debouncing and severity scoring (completed 2026-04-12)
- [x] **Phase 37: Agent Orchestrator + Debate/Experiment REST API** - Full conflict→debate→arbiter→experiment→verdict pipeline with REST endpoints and snapshot safety (completed 2026-04-12)
- [x] **Phase 38: PresetApplicator + Auto-Apply Loop** - Atomic YAML write-back, circuit-breaker gate, position-ownership tracking, sandbox validation gate (completed 2026-04-12)
- [x] **Phase 39: REST Endpoint Hardening** - Wire real alerter, circuit breaker state, multi-debate response, and finalize endpoint into REST API (completed 2026-04-12)

</details>

<details>
<summary>✅ v9.0 ML AutoResearch & MOEX Adaptation (Phases 40-44) -- SHIPPED 2026-04-13</summary>

- [x] **Phase 40: MOEX Data Adapter & Macro Features** - Wire TinkoffFetcher and MOEX macro features into auto_ml_research for all ru_* segments (completed 2026-04-13)
- [x] **Phase 41: Adaptive Quality Gates** - Parametrize min_signals, add MOEX fold constants, add degenerate predictor guard (completed 2026-04-13)
- [x] **Phase 42: ExperimentManager Integration** - Opt-in --experiment-id flag with hypothesis lifecycle and backward-compatible JSONL audit trail (completed 2026-04-13)
- [x] **Phase 43: Ensemble Weight Optimization** - Bounded XGB/LGBM/CatBoost weight grid search with overfitting guard (completed 2026-04-13)
- [x] **Phase 44: New Search Strategies** - Cross-segment US→MOEX feature transfer and domain-motivated feature engineering (completed 2026-04-13)

</details>

<details>
<summary>✅ v9.1 MOEX ML Model Quality (Phases 45-48) — SHIPPED 2026-04-14</summary>

- [x] Phase 45: Model Complexity & Ensemble Consistency (2/2 plans) — completed 2026-04-14
- [x] Phase 46: Feature Selection Stability (1/1 plan) — completed 2026-04-14
- [x] Phase 47: Cross-Asset Features & Asymmetric Barriers (2/2 plans) — completed 2026-04-14
- [x] Phase 48: Segment Restructuring & Validation (2/2 plans) — completed 2026-04-14

</details>

### 🚧 v10.0 Runtime LLM Trading Agents (In Progress)

**Milestone Goal:** Add runtime LLM agents to the live trading pipeline — news ingestion hardened and activated, EventDrivenStrategy firing live signals, portfolio review and anomaly interpretation agents deployed, and sentiment data accumulating for future ML use.

- [x] **Phase 49: News Pipeline Hardening** (2/3 plans) - Fix latent bugs and add production safeguards before live news activation (completed 2026-04-15)
- [x] **Phase 50: EventDriven Activation** - Enable event_driven strategy on all ru_* segments with signal quality guards (completed 2026-04-15)
- [ ] **Phase 51: Anomaly Interpreter Agent** - LLM enrichment for anomaly alerts via fire-and-forget async dispatch
- [ ] **Phase 52: Portfolio Review Agent** - Daily advisory LLM portfolio analysis with structured Pydantic output
- [ ] **Phase 53: Sentiment ML Infrastructure** - TimescaleDB continuous aggregates and SentimentStore reader for future ML features

## Phase Details

### Phase 49: News Pipeline Hardening
**Goal**: News pipeline bugs fixed and production safeguards active before Phase 50 activates EventDrivenStrategy
**Depends on**: Nothing (fixes existing code)
**Requirements**: NEWS-01, NEWS-02, NEWS-03, NEWS-04, NEWS-05, NEWS-06
**Plans:** 3/3 plans complete

Plans:
- [x] 49-01-PLAN.md -- Bug fixes: parse_structured migration, per-article 5s timeout, threading.Lock fix, article budget cap
- [x] 49-02-PLAN.md -- Production safeguards: source credibility map, ticker validation, LLM liveness monitoring
- [x] 49-03-PLAN.md -- Gap closure: wire validate_tickers into production path, fix LLM liveness blind spot (NEWS-04, NEWS-05)

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
- [x] 33-01-PLAN.md -- Debate protocol schemas (TDD): Claim, AgentOutput, FactCheckReport, DebateState (DEBATE-01, DEBATE-02, DEBATE-03)
- [x] 33-02-PLAN.md -- Arbiter agent, DebateManager CRUD, debates directory (DEBATE-02, DEBATE-03)

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
- [x] 34-01-PLAN.md -- Experiment schemas (TDD) + ExperimentManager CRUD, verdict, debate linkage (EXP-01, EXP-04)
- [x] 34-02-PLAN.md -- run_iteration.py --hypothesis extension + interaction test runner (EXP-02, EXP-03)

### Phase 35: Experiment Lab UI
**Goal**: Full experiment lifecycle is visible in a Streamlit web app -- from debate context through execution to final decision
**Depends on**: Phase 34 (needs experiment registry and results to display)
**Requirements**: UI-EXP-01, UI-EXP-02, UI-EXP-03
**Success Criteria** (what must be TRUE):
  1. Experiment list page shows all experiments with status (PENDING/RUNNING/COMPLETED), hypothesis summary, and key metrics -- at a glance, what experiments exist and their state
  2. Experiment detail page shows: debate context (why), success criteria (what we expect), backtest results with charts (what happened), and A vs B vs A+B comparison table -- complete decision context on one screen
  3. Decision history page shows accepted/rejected experiments with reasoning -- the team can review past decisions and understand why the system is configured the way it is
**Plans:** 2/2 plans complete
Plans:
- [x] 35-01-PLAN.md -- Smoke tests + Experiments List page (UI-EXP-01)
- [x] 35-02-PLAN.md -- Experiment Detail page + Decision History page (UI-EXP-02, UI-EXP-03)

### Phase 36: Conflict Detection Foundation
**Goal**: Domain agents emit schema-validated AgentOutput with sourced Claim objects, and the ConflictDetector identifies contradictions deterministically with debouncing and severity scoring
**Depends on**: Phase 35 (builds on v7.0 AgentOutput/Claim schemas already shipped)
**Requirements**: AGOUT-01, AGOUT-02, CONF-01, CONF-02, CONF-03, CONF-04
**Success Criteria** (what must be TRUE):
  1. A domain agent invocation returns an `AgentOutput` object with at least one `Claim`, each claim carrying a mandatory `source` field (file:line or metric name+value) -- no unsourced assertions pass schema validation
  2. `AnthropicClient.parse_structured()` wraps `client.messages.parse()` and guarantees the returned object matches the target Pydantic model -- structured output is enforced by the SDK, not by post-hoc string parsing
  3. `ConflictDetector.detect(outputs)` returns a `ConflictReport` using deterministic rule-based similarity scoring -- no LLM call is made inside the detector, execution completes in under 50 ms per pair
  4. `ConflictReport` schema is defined in `core/schemas.py` with `conflict_type`, `severity`, and `involved_claims` fields -- downstream orchestration can read conflict details without parsing free-text
  5. Topic-level deduplication and a minimum confidence delta of >0.15 are enforced before a conflict is escalated -- the same disagreement on the same topic does not trigger multiple debate entries within a single session
**Plans:** 2/2 plans complete
Plans:
- [x] 36-01-PLAN.md -- ConflictReport schema + parse_structured() on all LLM clients (CONF-02, AGOUT-02)
- [x] 36-02-PLAN.md -- ConflictDetector + agent .md Output Format sections (CONF-01, CONF-03, CONF-04, AGOUT-01)

### Phase 37: Agent Orchestrator + Debate/Experiment REST API
**Goal**: The full conflict→debate→arbiter→experiment→verdict pipeline runs end-to-end, manually triggerable via REST, with snapshot safety preventing false contradiction verdicts after code changes
**Depends on**: Phase 36 (requires ConflictDetector and AgentOutput emission)
**Requirements**: ORCH-01, ORCH-02, ORCH-03, ORCH-04
**Success Criteria** (what must be TRUE):
  1. `AgentOrchestrator.run(outputs)` executes the full pipeline -- detected conflict triggers a DebateManager entry, arbiter produces a FactCheckReport, and an experiment is created with a verdict -- the entire flow completes without manual intervention
  2. `GET /api/v1/debates` and `GET /api/v1/debates/{id}` return debate list and detail; `POST /api/v1/debates` creates a debate manually -- the pipeline is invocable without writing Python
  3. `GET /api/v1/experiments` and `GET /api/v1/experiments/{id}` return experiment state and linked backtest results -- all experiment data is accessible via REST without filesystem access
  4. `FileLineSource` carries a `snapshot_sha` field; when the referenced file has changed since the claim was recorded, the arbiter marks that claim `UNTESTABLE` instead of `CONTRADICTED` -- stale source references do not trigger false conflict escalations
  5. The `.claude/agents/agent-orchestrator.md` definition exists and can be invoked as a Claude Code sub-agent to run a full orchestration cycle autonomously
**Plans**: 2 plans
Plans:
- [x] 37-01-PLAN.md -- snapshot_sha schema + AgentOrchestrator pipeline (ORCH-01, ORCH-03)
- [x] 37-02-PLAN.md -- Debates + Experiments REST API, agent-orchestrator.md (ORCH-02, ORCH-04)

### Phase 38: PresetApplicator + Auto-Apply Loop
**Goal**: Accepted experiment verdicts atomically update strategy YAML presets with full safety gates -- circuit breaker, position ownership check, and mandatory sandbox validation before any live apply
**Depends on**: Phase 37 (verdict must exist before apply; orchestrator must be wired before auto-apply is meaningful)
**Requirements**: APPLY-01, APPLY-02, APPLY-03, APPLY-04, APPLY-05, APPLY-06
**Success Criteria** (what must be TRUE):
  1. `PresetApplicator.apply(experiment_id)` writes `preset_overrides` to the target strategy YAML using atomic `os.replace()` rename with a timestamped backup -- no partial write is visible to the strategy cycle; the backup is queryable post-apply
  2. When `CircuitLevel != NORMAL`, `apply_verdict()` raises and logs a rejection -- no YAML is written while a circuit breaker is active, regardless of verdict status
  3. `TradingLoop._entry_strategy` tracks which strategy opened each open position; attempting to disable a strategy via auto-apply while it holds open positions results in a blocked apply with a Telegram alert sent -- no strategy is disabled under live exposure
  4. `combiner.invalidate_segment_cache()` is called immediately after atomic YAML rename -- the next strategy cycle reads fresh preset values without requiring a process restart
  5. An INCONCLUSIVE verdict sends a Telegram alert with experiment ID and metric summary and does not trigger any YAML write -- the operator is notified and retains full control
  6. A sandbox validation gate requires at least 3 trading days of sandbox metrics after an ACCEPT verdict before the live apply is permitted -- backtest acceptance alone is not sufficient for live promotion
**Plans:** 2/2 plans complete
Plans:
- [x] 38-01-PLAN.md -- PresetApplicator + SandboxGate + REST apply endpoint (APPLY-01, APPLY-02, APPLY-05, APPLY-06)
- [x] 38-02-PLAN.md -- _entry_strategy tracking + invalidate_segment_cache (APPLY-03, APPLY-04)

### Phase 39: REST Endpoint Hardening
**Goal**: REST API endpoints for debates and experiments have real safety gates wired — Telegram alerts fire on INCONCLUSIVE, circuit breaker state is injected, multi-debate responses return all debate IDs, and finalize_debate() is REST-accessible
**Depends on**: Phase 38 (needs PresetApplicator and AgentOrchestrator fully implemented)
**Requirements**: ORCH-01, ORCH-02, APPLY-02, APPLY-05
**Gap Closure**: Closes integration/flow gaps from v8.0 audit
**Success Criteria** (what must be TRUE):
  1. `POST /experiments/{id}/apply` with INCONCLUSIVE verdict sends a real Telegram alert — no-op alerter replaced with a real or injectable alerter in the REST context
  2. `POST /experiments/{id}/apply` checks live circuit breaker state from a shared source — empty `circuit_breakers={}` replaced with actual circuit breaker lookup
  3. `POST /debates` response includes `debate_ids: list[str]` containing all created debate IDs — multi-debate cases are fully represented
  4. `POST /debates/{id}/finalize` endpoint accepts a FactCheckReport and calls `AgentOrchestrator.finalize_debate()` — the arbiter-to-experiment loop is REST-triggerable
**Plans**: 1 plan
Plans:
- [x] 39-01-PLAN.md -- Wire real alerter, circuit breaker, multi-debate response, finalize endpoint (ORCH-01, ORCH-02, APPLY-02, APPLY-05)

### Phase 40: MOEX Data Adapter & Macro Features
**Goal**: auto_ml_research runs end-to-end on all four ru_* segments using TinkoffFetcher for candles and real MOEX macro features (CBR rate, USDRUB, IMOEX, Brent) in the feature pipeline
**Depends on**: Phase 39 (v8.0 complete; this starts v9.0)
**Requirements**: MOEX-01, MOEX-02, MOEX-03
**Success Criteria** (what must be TRUE):
  1. `python scripts/auto_ml_research.py --segment ru_blue_chips` completes data loading without error — candle counts are printed and TinkoffFetcher is used (not yfinance) for all ru_* segments
  2. `_SEGMENT_SYMBOLS` in auto_ml_research.py contains ru_blue_chips, ru_energy, ru_finance, ru_tech symbols that match the production universe in config/segments.py — no symbol lookup errors at runtime
  3. All 10 MOEX macro features (usdrub_zscore_60d, brent_zscore_60d, cbr_rate_level, cbr_rate_delta, real_rate_zscore, etc.) are non-zero in the feature matrix for any MOEX experiment run — macro context is actually flowing through build_full_dataset()
  4. Macro series are shift(1) aligned before join — a unit test with a synthetic macro series verifies no future value leaks into the feature vector (look-ahead bias absent)
**Plans:** 2/2 plans complete
Plans:
- [x] 40-01-PLAN.md -- MOEX segment symbols and TinkoffFetcher data loading (MOEX-01, MOEX-02)
- [x] 40-02-PLAN.md -- MOEX macro data fetching and MarketContext wiring (MOEX-03)

### Phase 41: Adaptive Quality Gates
**Goal**: MOEX experiments produce trustworthy walk-forward results — signal count gates are calibrated to MOEX dataset sizes, folds never collapse to fewer than 3, and degenerate all-BUY/all-SELL models are rejected automatically
**Depends on**: Phase 40 (needs working MOEX data flow to validate gate thresholds empirically)
**Requirements**: GATE-01, GATE-02, GATE-03
**Success Criteria** (what must be TRUE):
  1. `evaluate_fold(min_signals=15)` accepts a MOEX experiment with 15-30 signals per fold — the hardcoded _MIN_SIGNALS=50 no longer blocks all MOEX runs
  2. A 730-day MOEX dataset produces 3 or more valid walk-forward folds using MOEX-specific fold constants — the experiment does not trivially pass on a single fold
  3. A model that predicts BUY on 92% of samples fails the degenerate predictor gate and is logged as REJECTED with buy_ratio=0.92 — all-directional models cannot receive a verdict without this check
**Plans:** 2/2 plans complete
Plans:
- [x] 41-01-PLAN.md -- Adaptive min_signals + degenerate predictor gate in quality_gates.py (GATE-01, GATE-03)
- [x] 41-02-PLAN.md -- MOEX fold constants + min_signals wiring in auto_ml_research.py (GATE-02)

### Phase 42: ExperimentManager Integration
**Goal**: auto_ml_research research runs are tracked as named experiments with hypothesis lifecycle, verdicts, and backward-compatible JSONL audit trail when --experiment-id is not provided
**Depends on**: Phase 41 (quality gates must be reliable before experiment verdicts carry meaning)
**Requirements**: EXPINT-01, EXPINT-02
**Success Criteria** (what must be TRUE):
  1. Running `auto_ml_research.py --segment ru_blue_chips --experiment-id ru_blue_chips_baseline_20260413_1200` creates an ExperimentManager entry, links per-fold results, and records ACCEPT/REJECT/INCONCLUSIVE at completion — the experiment is queryable via ExperimentManager.get()
  2. Two concurrent segment runs with different --experiment-id values produce non-overlapping experiment files — no ID collision or shared-state corruption
  3. Running `auto_ml_research.py --segment ru_blue_chips` without --experiment-id completes normally with JSONL output only — existing invocations are not broken by the integration
**Plans:** 1/1 plans complete
Plans:
- [x] 42-01-PLAN.md -- ExperimentManager integration with --experiment-id flag (EXPINT-01, EXPINT-02)

### Phase 43: Ensemble Weight Optimization
**Goal**: A new ensemble_weights search strategy explores the XGB/LGBM/CatBoost weight simplex, enforces overfitting guards, and logs optimization gain separately from baseline
**Depends on**: Phase 42 (experiment tracking must be in place to compare weight configurations as named hypotheses)
**Requirements**: STRAT-01
**Success Criteria** (what must be TRUE):
  1. `auto_ml_research.py --strategy ensemble_weights --segment ru_blue_chips` evaluates at least 9 distinct weight configurations across the simplex — XGB, LGBM, CatBoost weights are explored in bounded combinations, each summing to 1.0
  2. No single model weight exceeds 0.7 in any evaluated configuration — the overfitting constraint is enforced at generation time, not post-hoc
  3. When fewer than 4 independent folds are available, equal weights (1/3 each) are used as the default and optimization is skipped with a logged warning — small dataset safety is automatic
**Plans:** 1/1 plans complete
Plans:
- [x] 43-01-PLAN.md -- Ensemble weight simplex generator, weighted averaging, small-fold guard (STRAT-01)

### Phase 44: New Search Strategies
**Goal**: Two new search strategies extend the research loop — cross-segment transfer validates US-learned features on MOEX, and feature engineering generates domain-motivated combinations with hard overfitting caps
**Depends on**: Phase 43 (stable MOEX baseline and reliable experiment tracking needed before adding high-complexity strategies)
**Requirements**: STRAT-02, STRAT-03
**Success Criteria** (what must be TRUE):
  1. `auto_ml_research.py --strategy cross_segment_transfer --segment ru_blue_chips` reads best US experiment features from JSONL history and filters to market-neutral intersection — VIX-only and MOEX-only features are excluded from the transfer set, and the filtered feature list is logged
  2. `auto_ml_research.py --strategy feature_engineering --segment ru_blue_chips` generates domain-motivated feature combinations (lag ratios, rolling z-scores, cross-feature interactions) with a hard cap of n_samples/20 candidates — no more than ~36 candidates are generated for a 730-day MOEX dataset
  3. Generated features that do not pass a permutation importance test are discarded before model training — feature engineering cannot add noise-only columns to the feature matrix
**Plans:** 1/1 plans complete
Plans:
- [x] 44-01-PLAN.md -- Cross-segment transfer + feature engineering strategies (STRAT-02, STRAT-03)

### Phase 49: News Pipeline Hardening
**Goal**: The news ingestion pipeline is production-safe — latent bugs fixed, cost safeguards in place, and signal quality guards active — before any live segment activation
**Depends on**: Phase 48 (v9.1 complete; this starts v10.0)
**Requirements**: NEWS-01, NEWS-02, NEWS-03, NEWS-04, NEWS-05, NEWS-06
**Success Criteria** (what must be TRUE):
  1. A news cycle processes up to 20 articles with a 5-second per-article LLM timeout — the APScheduler thread completes the cycle within 2 minutes regardless of LLM latency, and exceeding the budget logs a `news_budget_cap_hit` metric
  2. Sentiment scores written to the `sentiment_scores` table reflect the originating source credibility (RSS sources: 0.8, Telegram: 0.7) — inspecting DB rows shows a non-null `credibility` column populated from the per-source map
  3. An LLM-extracted ticker that does not appear in InstrumentRegistry is rejected with a structured log entry containing `entity_not_in_registry` and the rejected ticker — no ghost-ticker sentiment scores accumulate in the DB
  4. When the LLM API fails for 3 consecutive cycles, a Telegram alert fires and `llm_liveness_failures` Prometheus counter increments — LLM downtime is observable from the monitoring dashboard
  5. Calling `NewsAnalyzer.analyze()` on a news article returns a structured `SentimentResult` Pydantic object via `parse_structured()` — no `json.loads()` call remains in the news analysis path
**Plans**: 3 plans (2 complete, 1 gap closure)

Plans:
- [x] 49-01-PLAN.md -- Bug fixes: parse_structured migration, per-article 5s timeout, article budget cap
- [x] 49-02-PLAN.md -- Production safeguards: source credibility map, ticker validation, LLM liveness monitoring
- [ ] 49-03-PLAN.md -- Gap closure: wire validate_tickers into production path, fix LLM liveness blind spot (NEWS-04, NEWS-05)

### Phase 50: EventDriven Activation
**Goal**: The event_driven strategy fires live signals on all ru_* segments with CBR/dividend duplicate-signal protection and sentiment decay gated on market hours
**Depends on**: Phase 49 (news pipeline must be production-safe before enabling live strategy signals)
**Requirements**: EVNT-01, EVNT-02, EVNT-03
**Success Criteria** (what must be TRUE):
  1. All ru_* segment presets have `event_driven.enabled: true` with weight 0.15 — a sandbox strategy cycle that processes a news article generates at least one EventDrivenStrategy signal entry in the `signals` table
  2. When an article classified as a CBR announcement arrives and `cbr_calendar` strategy also has a pending signal for the same ticker, the combiner suppresses the double-weight — the final combined confidence for that tick does not exceed what a single strategy would contribute
  3. Between 18:50 MSK (MOEX close) and 09:50 MSK (MOEX open) the sentiment decay clock is frozen — the first article of the trading day produces a signal score within ±10% of the last signal from the previous session rather than spiking from a near-zero decayed baseline
**Plans**: 2 plans
Plans:
- [x] 50-01-PLAN.md — Dynamic sentiment TTL + event_type caching + ru_tech preset activation
- [x] 50-02-PLAN.md — Credibility threading + CBR/dividend dedup guard in combiner

### Phase 51: Anomaly Interpreter Agent
**Goal**: Detected anomalies receive an LLM explanation appended to the Telegram alert without ever delaying the raw statistical alert
**Depends on**: Phase 49 (fire-and-forget async pattern from news pipeline hardening; LLM liveness check establishes baseline reliability)
**Requirements**: ANMI-01, ANMI-02, ANMI-03
**Success Criteria** (what must be TRUE):
  1. When AnomalyDetector fires an alert, the raw statistical alert message is sent to Telegram immediately — a unit test asserting that `TelegramAlerter.send()` is called before any LLM await passes
  2. Within 30 seconds of the raw alert, a follow-up Telegram message arrives containing the LLM explanation labeled "AI interpretation (unverified)" — the enrichment is visible in the Telegram chat as a separate message after the original
  3. When the LLM call times out or raises an exception, the raw alert is still delivered and a `anomaly_llm_failure` structlog entry is emitted — suppressing the raw alert on LLM failure is impossible by design
**Plans**: 2 plans
Plans:
- [ ] 51-01-PLAN.md — AnomalyDetector class + AnomalyResult schema with TDD
- [ ] 51-02-PLAN.md — Wire anomaly detection into TradingLoop with LLM enrichment

### Phase 52: Portfolio Review Agent
**Goal**: A daily LLM portfolio analysis runs outside market hours, delivers a structured advisory report via Telegram, and has no write path to the order pipeline
**Depends on**: Phase 49 (LLM async patterns and credibility safeguards established; Phase 51 validates advisory-only Telegram dispatch pattern)
**Requirements**: PFRA-01, PFRA-02, PFRA-03
**Success Criteria** (what must be TRUE):
  1. A `PortfolioReviewResult` Pydantic schema exists in `core/schemas.py` with no `direction`, `confidence`, `symbol`+`market_id` combination that matches `Signal` or `OrderRequest` — a type-checker assertion at handler entry prevents accidental trade directive fields from being added
  2. Daily at 19:00 MSK (after MOEX close), a Telegram message arrives summarizing open positions, concentration risk, and any upcoming catalyst events — the message is structured, not free-form prose, and references specific ticker names
  3. The `PortfolioReviewAgent` handler writes only to `TelegramAlerter` — a code search for `BrokerRouter`, `place_order`, or `generate_signal` inside the agent handler returns zero results
**Plans**: 2 plans
Plans:
- [ ] 50-01-PLAN.md — Dynamic sentiment TTL + event_type caching + ru_tech preset activation
- [ ] 50-02-PLAN.md — Credibility threading + CBR/dividend dedup guard in combiner
**UI hint**: yes

### Phase 53: Sentiment ML Infrastructure
**Goal**: Rolling sentiment aggregations accumulate in TimescaleDB and a Layer 2 accessor provides the read API that the v11 ML feature pipeline will consume
**Depends on**: Phase 49 (sentiment_scores table must be populating from live news before aggregation is meaningful)
**Requirements**: STML-01, STML-02
**Success Criteria** (what must be TRUE):
  1. A TimescaleDB continuous aggregate view `sentiment_7d_avg` exists and auto-refreshes — `SELECT * FROM sentiment_7d_avg WHERE ticker = 'SBER' ORDER BY bucket DESC LIMIT 5` returns rows after one week of live sentiment data accumulation
  2. `SentimentStore.get_rolling(ticker, window='7d')` (Layer 2) returns a list of `(bucket, avg_score, article_count)` rows — a unit test with seeded `sentiment_scores` fixture data verifies the query returns correct aggregates without Python-side computation
  3. Querying `SentimentStore` on a ticker with no sentiment history returns an empty list without error — the v11 feature pipeline can call the accessor safely before data accumulates
**Plans**: 2 plans
Plans:
- [ ] 50-01-PLAN.md — Dynamic sentiment TTL + event_type caching + ru_tech preset activation
- [ ] 50-02-PLAN.md — Credibility threading + CBR/dividend dedup guard in combiner

## Progress

**Execution Order:**

v6.0: 28 -> 29 -> 30 -> 31 (all complete)
v7.0: 32 -> 33 -> 34 -> 35 (all complete)
v8.0: 36 -> 37 -> 38 -> 39 (all complete)
v9.0: 40 -> 41 -> 42 -> 43 -> 44 (all complete)
v9.1: 45 -> 46 -> 47 -> 48 (all complete)
v10.0: 49 -> 50 -> 51 -> 52 -> 53 (in progress)

| Phase | Milestone | Plans | Status | Completed |
|-------|-----------|-------|--------|-----------|
| 1-7 | v1.0 | 22/22 | Complete | 2026-03-19 |
| 8-14 | v2.0 | 16/16 | Complete | 2026-03-21 |
| 15-18 | v3.0 | 10/10 | Complete | 2026-03-22 |
| 19-22 | v4.0 | 10/10 | Complete | 2026-03-22 |
| 23-27 | v5.0 | 9/9 | Complete | 2026-03-24 |
| 28-31 | v6.0 | 8/8 | Complete | 2026-03-30 |
| 32-35 | v7.0 | 10/10 | Complete | 2026-04-08 |
| 36-39 | v8.0 | 7/7 | Complete | 2026-04-12 |
| 40-44 | v9.0 | 7/7 | Complete | 2026-04-13 |
| 45-48 | v9.1 | 7/7 | Complete | 2026-04-14 |
| 49. News Pipeline Hardening | v10.0 | 3/3 | Complete    | 2026-04-15 |
| 50. EventDriven Activation | v10.0 | 2/2 | Complete    | 2026-04-15 |
| 51. Anomaly Interpreter Agent | v10.0 | 0/TBD | Not started | - |
| 52. Portfolio Review Agent | v10.0 | 0/TBD | Not started | - |
| 53. Sentiment ML Infrastructure | v10.0 | 0/TBD | Not started | - |
