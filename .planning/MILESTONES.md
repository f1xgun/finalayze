# Milestones

## v10.0 Runtime LLM Trading Agents (Shipped: 2026-04-15)

**Phases completed:** 5 phases, 11 plans, 15 tasks

**Key accomplishments:**

- 1. [Rule 3 - Blocking] Files referenced in plan do not exist
- 1. [Rule 3 - Blocking] _persist_sentiment_batch_async does not exist
- 1. [Rule 2 - Missing Critical] Enabled event_driven on ru_blue_chips, ru_energy, ru_finance presets
- CBR/dividend duplicate-signal guard in StrategyCombiner with credibility threading and event_type_code in Signal.features
- Statistical anomaly detector with rolling 20-bar z-score for >3-sigma price moves and >2x volume spikes, TDD-driven with 8 unit tests
- Wire AnomalyDetector into TradingLoop._process_instrument with raw Telegram alert + fire-and-forget LLM enrichment via run_coroutine_threadsafe, verified by 8 integration tests
- 1. [Rule 1 - Bug] Docstring contained code-grep target strings
- Alembic migration 006 converting sentiment_scores to hypertable with sentiment_7d_avg continuous aggregate and hourly refresh policy
- SentimentStore Layer 2 accessor with get_rolling() querying sentiment_7d_avg view via text() named param bindings, with window allowlist validation and empty-list safety

---

## v9.1 MOEX ML Model Quality (Shipped: 2026-04-14)

**Phases completed:** 4 phases, 7 plans, 4 tasks

**Key accomplishments:**

- One-liner:
- One-liner:
- One-liner:
- Extended _compute_brent_return_features to return brent_ret_5d and brent_ret_21d alongside existing brent_return, with independent per-feature fallback logic and horizon-scaled clip bounds
- One-liner:
- 1. [Rule 1 - Bug] Existing test fixtures used 200/300 candles — below new 500-day gate

---

## v9.0 ML AutoResearch & MOEX Adaptation (Shipped: 2026-04-13)

**Phases completed:** 5 phases, 7 plans, 4 tasks

**Key accomplishments:**

- TinkoffFetcher adapter for all ru_* MOEX equity segments with symbols sourced from config/segments.py (single source of truth), IMOEX benchmark, and graceful FINALAYZE_TINKOFF_TOKEN skip
- MOEX macro features (CBR rate, USDRUB, IMOEX turnover, Brent) fetched once and wired via MoexMarketData into build_full_dataset() with 2-bar look-ahead bias prevention
- Adaptive min_signals parameter in evaluate_fold() (default 50 for US, 15 for MOEX) and degenerate predictor guard (buy_ratio 0.15-0.85 bounds) as 8th quality gate
- MOEX-specific walk-forward fold constants (8mo/1mo/3mo/21d/2mo) producing 3+ folds on 730-day MOEX data
- Opt-in --experiment-id flag wiring ExperimentManager lifecycle (create → link → verdict) with backward-compatible JSONL audit trail
- Ensemble weight optimization strategy (33 simplex configs, 0.7 cap, small-fold guard)
- Cross-segment transfer (US→MOEX market-neutral feature filtering) and feature engineering (domain-motivated combinations with n_samples/20 cap + permutation importance filter)

---

## v8.0 Agent Integration & Autonomous Decision Loop (Shipped: 2026-04-12)

**Phases completed:** 4 phases, 7 plans, ~14 tasks

**Key accomplishments:**

- ConflictDetector with deterministic rule-based contradiction detection (direction/metric/statement), 3-level severity scoring, SHA-256 topic deduplication, and confidence delta filtering
- parse_structured() on all 5 LLM clients (Anthropic, OpenAI, OpenRouter, Groq, DeepSeek) with BadRequestError fallback to JSON mode
- AgentOrchestrator pipeline coordinator: conflict → debate → arbiter → experiment → verdict, with snapshot_sha safety on FileLineSource
- REST API for debates (POST create, GET list/detail, POST finalize) and experiments (GET list/detail, POST apply) with X-API-Key auth
- PresetApplicator with 7-gate safety pipeline: circuit breaker first, INCONCLUSIVE Telegram routing, SandboxGate (3+ trading days), atomic os.replace() with timestamped backup
- _entry_strategy position-ownership tracking in TradingLoop, invalidate_segment_cache() on StrategyCombiner
- agent-orchestrator.md Claude Code sub-agent definition for autonomous pipeline runs
- 87+ new tests across all phases
- Fire-and-forget DB persistence for news articles (with SHA-256 content hash) and batch sentiment scores after LLM analysis
- CachingFetcher with 4 req/sec RateLimiter wired in both sandbox entry points, event_driven enabled for all MOEX segments, per-gate signal drop counters in CycleLogEntry
- Per-fold EnsembleCalibrator fitted on cal_idx and wired to _evaluate_fold_metrics so walk-forward Brier score uses calibrated probabilities
- One-liner:
- Experiment registry with Pydantic schemas (6-status lifecycle, operator whitelist, path-safe IDs) and ExperimentManager CRUD with automated ACCEPT/REJECT/INCONCLUSIVE verdict computation and bidirectional debate linkage
- Hypothesis-linked backtest runner with --hypothesis/--run-name flags and A/B/AB interaction test comparison orchestrator
- Streamlit Experiments List page with status/hypothesis filtering, gradient-colored Sharpe/PF metrics, and navigation to detail view
- Experiment detail page with debate context, A/B/AB grouped bar chart, and decision history page with reverse-chronological audit trail
- Status:
- 1. [Rule 1 - Bug] Test confidence values caused unintended filter suppression
- One-liner:
- One-liner:
- One-liner:
- One-liner:

---

## v7.0 Agent Intelligence & Experiment Framework (Shipped: 2026-04-12)

**Phases completed:** 8 phases, 18 plans, 28 tasks

**Key accomplishments:**

- Market-hours gate in strategy cycle to skip off-hours computation, and HHRU->HEAD ticker fix in ru_tech segment
- SHA-256 article deduplication before LLM calls (24h TTL, 5000-entry cap) and try/except-wrapped Telegram alerter at sandbox startup/shutdown
- Dedicated gRPC event loop isolating PollerCompletionQueue from HTTP/DB/Telegram work to eliminate 60-min strategy cycle drift
- Fixed Promtail->Loki log pipeline with container log volume mounts, __path__ relabeling, low-cardinality labels, and 30-day retention via compactor
- Portfolio cache fallback on T-Bank 70001 errors with auto-reconnect after 5 consecutive failures
- CBR XML FX rate fallback with in-memory cache and Prometheus metric wiring to prevent zero USD/RUB rate during gRPC outages
- Fire-and-forget DB persistence for orders and signals via _persist_to_db helper with Prometheus failure counter
- Fire-and-forget DB persistence for news articles (with SHA-256 content hash) and batch sentiment scores after LLM analysis
- CachingFetcher with 4 req/sec RateLimiter wired in both sandbox entry points, event_driven enabled for all MOEX segments, per-gate signal drop counters in CycleLogEntry
- Per-fold EnsembleCalibrator fitted on cal_idx and wired to _evaluate_fold_metrics so walk-forward Brier score uses calibrated probabilities
- One-liner:
- Experiment registry with Pydantic schemas (6-status lifecycle, operator whitelist, path-safe IDs) and ExperimentManager CRUD with automated ACCEPT/REJECT/INCONCLUSIVE verdict computation and bidirectional debate linkage
- Hypothesis-linked backtest runner with --hypothesis/--run-name flags and A/B/AB interaction test comparison orchestrator
- Streamlit Experiments List page with status/hypothesis filtering, gradient-colored Sharpe/PF metrics, and navigation to detail view
- Experiment detail page with debate context, A/B/AB grouped bar chart, and decision history page with reverse-chronological audit trail

---

## v6.0 Sandbox Stability & Observability (Shipped: 2026-03-30)

**Phases completed:** 4 phases, 8 plans, 15 tasks

**Key accomplishments:**

- Market-hours gate in strategy cycle to skip off-hours computation, and HHRU->HEAD ticker fix in ru_tech segment
- SHA-256 article deduplication before LLM calls (24h TTL, 5000-entry cap) and try/except-wrapped Telegram alerter at sandbox startup/shutdown
- Dedicated gRPC event loop isolating PollerCompletionQueue from HTTP/DB/Telegram work to eliminate 60-min strategy cycle drift
- Fixed Promtail->Loki log pipeline with container log volume mounts, __path__ relabeling, low-cardinality labels, and 30-day retention via compactor
- Portfolio cache fallback on T-Bank 70001 errors with auto-reconnect after 5 consecutive failures
- CBR XML FX rate fallback with in-memory cache and Prometheus metric wiring to prevent zero USD/RUB rate during gRPC outages
- Fire-and-forget DB persistence for orders and signals via _persist_to_db helper with Prometheus failure counter
- Fire-and-forget DB persistence for news articles (with SHA-256 content hash) and batch sentiment scores after LLM analysis

---

## v5.0 Data Flow Correctness (Shipped: 2026-03-24)

**Phases completed:** 4 phases, 7 plans, 10 tasks

**Key accomplishments:**

- Fixed SELL qty (held position), sector exposure (per-position prices), and CAUTION threshold (segment preset) in TradingLoop
- Trailing stop state machine with 5-step ratcheting logic and per-cycle re-entry guard wired into TradingLoop, matching SimulatedBroker behavior
- PositionSizingPipeline wired in live _build_order with all 14 pre-trade check parameters passed
- DataNormalizer candle validation, 48h staleness detection, and IMOEX volume column fix wired into live trading loop
- Persistent gRPC channel for all TinkoffFetcher bond methods and Brent crude caching via GenericFileCache
- News cycle skip guard when event_driven is disabled plus 4-hour half-life exponential sentiment decay
- Fixed T-Bank ticker mismatch (T -> TCSG) in entity extractor and added URL-based message deduplication to Telegram reader with 5000-entry LRU eviction

---

## v4.0 Architecture Hardening (Shipped: 2026-03-22)

**Phases completed:** 4 phases, 10 plans, 21 tasks

**Key accomplishments:**

- asyncio.Lock for async broker paths, threading.Lock double-check for loop init, async-with session scoping in macro_cache
- Atomic stop-loss under single lock hold preventing double-sell TOCTOU race, plus direct feed timestamp wiring and verified /gonogo import
- Fixed three async bugs: non-blocking gRPC reconnect via _stop_event.wait(), coroutine-aware aexecute() with iscoroutine guard, and thread-safe persistence via background event loop replacing asyncio.run()
- Non-blocking portfolio endpoint via run_in_executor, structured close() failure logging, and configurable 60s gRPC timeout for TinkoffFetcher
- Idempotent TelegramAlerter.close() wired into FastAPI lifespan shutdown for both alerter instances (trading loop + bot handler), preventing httpx.AsyncClient resource leaks
- GARCH NaN fallback with rolling vol + structlog warnings, EventBus exception narrowing to redis.ResponseError, and POST /kill authenticated via X-API-Key
- Structured error_type logging in TinkoffFetcher, consecutive failure counters with Telegram alerting in TradingLoop, and per-layer error escalation in BondCycleProcessor
- Moved 4 misplaced modules from core/ to their correct dependency layers: orchestration/ (L5) and api/ (L6), with sys.modules shims for zero-breakage backward compatibility
- MetricsCollector injected into TradingLoop via constructor, eliminating 6 deferred L6 imports; backtest/ and monitoring/ assigned definitive layers
- Removed 3 dead EventBus stream constants and 2 unused event models; converted 7 stub API endpoints from empty-200 to explicit 501 Not Implemented

---

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
