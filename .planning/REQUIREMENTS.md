# Requirements: Finalayze

**Defined:** 2026-03-30
**Core Value:** Autonomous profitable MOEX trading with acceptable risk limits

## v8.0 Requirements

Requirements for Agent Integration & Autonomous Decision Loop milestone.

### Agent Output

- [x] **AGOUT-01**: Domain agents emit `AgentOutput` with structured `Claim` objects and mandatory source references — no unsourced assertions in agent recommendations
- [x] **AGOUT-02**: `AnthropicClient.parse_structured()` wraps `client.messages.parse()` with Pydantic model schema derivation — structured output guaranteed by SDK

### Conflict Detection

- [x] **CONF-01**: `ConflictDetector` compares `list[AgentOutput]` and returns `ConflictReport` using deterministic rule-based logic — no LLM in the hot path
- [x] **CONF-02**: `ConflictReport` schema added to `core/schemas.py` with conflict type, severity, and involved claims
- [x] **CONF-03**: Debouncing: topic-level deduplication and minimum confidence delta (>0.15) before escalation — prevents debate storms
- [x] **CONF-04**: Conflict severity scoring ranks contradictions by impact to reduce noise

### Orchestration Pipeline

- [x] **ORCH-01**: `AgentOrchestrator` coordinates full pipeline: conflict → debate → arbiter → experiment → backtest → verdict
- [x] **ORCH-02**: REST API endpoints for debates (list, detail, create) and experiments (list, detail) — manual pipeline invocation
- [x] **ORCH-03**: `snapshot_sha` field on `FileLineSource` prevents false CONTRADICTED verdicts after code changes
- [x] **ORCH-04**: Claude Code `agent-orchestrator.md` definition enables autonomous pipeline runs

### Auto-Apply & Safety

- [x] **APPLY-01**: `PresetApplicator` writes experiment `preset_overrides` to strategy YAML with backup snapshot and atomic `os.replace()` rename
- [x] **APPLY-02**: Circuit-breaker gate blocks auto-apply when `CircuitLevel != NORMAL`
- [x] **APPLY-03**: `_entry_strategy` dict in `TradingLoop` tracks which strategy opened each position; blocks strategy-disable if positions exist
- [x] **APPLY-04**: `combiner.invalidate_segment_cache()` method forces preset reload after YAML write
- [x] **APPLY-05**: INCONCLUSIVE experiment verdicts route to Telegram alert (no auto-apply)
- [x] **APPLY-06**: Sandbox validation gate (≥3 trading days) required between ACCEPT verdict and live apply

## v6.0 Requirements

Requirements for Sandbox Stability & Observability milestone.

### gRPC Stability

- [x] **GRPC-01**: gRPC PollerCompletionQueue runs on a dedicated event loop isolated from APScheduler — no BlockingIOError flooding the main asyncio loop, strategy cycles fire within 5 min of scheduled time
- [x] **GRPC-02**: TinkoffBroker reconnects gRPC channel on StatusCode.INTERNAL (error 70001) — automatic recovery within 1 retry cycle, no multi-hour outage windows
- [x] **GRPC-03**: Portfolio fetch failure falls back to last-known portfolio state — strategy cycle continues with cached positions instead of skipping entirely

### Data Persistence

- [x] **PERSIST-01**: Executed orders persisted to `orders` table after fill — symbol, side, quantity, fill_price, order_id, timestamp stored
- [x] **PERSIST-02**: Generated signals persisted to `signals` table — strategy, symbol, direction, confidence, reasoning stored
- [x] **PERSIST-03**: Processed news articles persisted to `news_articles` table — title, source, published_at, content hash stored
- [x] **PERSIST-04**: Sentiment scores persisted to `sentiment_scores` table — ticker, score, source, timestamp stored
- [x] **PERSIST-05**: DB write failures are fire-and-forget with structured logging — never crash the trading loop or increment consecutive error counter

### Observability

- [x] **OBS-01**: Promtail ships Docker container logs to Loki — `/var/lib/docker/containers` mounted, JSON log format parsed correctly
- [x] **OBS-02**: Loki retains queryable logs for 30 days — dashboard queries return results for all 7 containers
- [x] **OBS-03**: FX rate (USD/RUB) fetched from CBR XML API as fallback when gRPC FX fetch fails — `finalayze_usd_rub_rate` metric is non-zero

### Operational Hygiene

- [x] **OPS-01**: Strategy cycle skips execution when MOEX market is closed — no wasted cycles with 0 instruments processed
- [x] **OPS-02**: Stale tickers removed/updated in config/segments.py — FIVE, FIXP, POLY removed; YNDX→YDEX; HHRU→HH (if valid)
- [x] **OPS-03**: LLM article deduplication via content hash — seen articles skipped within 24h TTL window, reducing rate-limit fallbacks
- [x] **OPS-04**: Telegram alerter startup failure does not block trading loop launch — alert sent on next successful cycle instead

## v7.0 Requirements

Requirements for Agent Intelligence & Experiment Framework milestone.

### Sandbox Signal Fixes

- [x] **SANDBOX-FIX-01**: `_CANDLE_LOOKBACK >= 210` in trading loop — RSI2 Connors (SMA-200), dual_momentum (126 bars), OU mean reversion (126 bars) all receive sufficient data in live mode
- [x] **SANDBOX-FIX-02**: `TradingLoop.start()` checks `KillSwitch.is_killed` before starting scheduler — killed system does not resume on Docker restart
- [x] **SANDBOX-FIX-03**: When `FINALAYZE_MODE=sandbox` and `rollout_phase` not explicitly set, effective rollout is MINIMAL — sandbox starts with conservative risk limits
- [x] **SANDBOX-FIX-04**: Staleness threshold handles weekends and MOEX holidays — Monday morning and post-holiday cycles not blocked by 48h threshold
- [x] **SANDBOX-FIX-05**: TinkoffFetcher wrapped in CachingFetcher in sandbox mode — repeated API calls for same data eliminated
- [x] **SANDBOX-FIX-06**: RateLimiter passed to TinkoffFetcher in sandbox — API throttling prevented for large instrument universes
- [x] **SANDBOX-FIX-07**: `FINALAYZE_LLM_API_KEY` documented and event_driven enabled for ru_blue_chips, ru_energy, ru_finance — news pipeline activated for MOEX
- [x] **SANDBOX-FIX-08**: Per-gate signal drop counters in ValidationLogger — instruments_no_bars, signals_below_threshold, signals_pre_trade_rejected tracked separately
- [x] **SANDBOX-FIX-09**: ML quality gate bug: profit_factor gate populated with actual PF from fold predictions — gate no longer always fails with default 1.0
- [x] **SANDBOX-FIX-10**: ML quality gate: Brier score evaluated on calibrated probabilities — calibrator applied during walk-forward evaluation

### Structured Debate Protocol

- [x] **DEBATE-01**: Agent output schema enforces structured claims with source references (file:line or metric value) — no unsourced assertions in agent recommendations
- [x] **DEBATE-02**: An arbiter agent can take two conflicting agent outputs and produce a fact-check report showing which claims are verified, which are contradicted, and which are untestable
- [x] **DEBATE-03**: Debate state (claims, conflicts, resolutions) is persisted in `.planning/debates/` for audit trail — every multi-agent decision has a traceable history

### Experiment Registry & Runner

- [x] **EXP-01**: Experiment registry stores hypothesis, success criteria (metric + threshold), status, and linked backtest results — every experiment has a pre-registered definition
- [x] **EXP-02**: `run_iteration.py --hypothesis <id>` runs a parameterized backtest and links results to the hypothesis — experiment results are automatically associated with their hypothesis
- [x] **EXP-03**: Interaction testing: given hypotheses A and B, the runner executes A-only, B-only, and A+B runs and compares all three — combination effects are measured, not assumed
- [x] **EXP-04**: Experiment verdicts (ACCEPT/REJECT/INCONCLUSIVE) are recorded with reasoning and linked to the debate that triggered them

### Experiment Lab UI

- [x] **UI-EXP-01**: Experiment list page shows all experiments with status (PENDING/RUNNING/COMPLETED), hypothesis summary, and key metrics — at a glance, what experiments exist and their state
- [x] **UI-EXP-02**: Experiment detail page shows: debate context (why), success criteria (what we expect), backtest results with charts (what happened), and A vs B vs A+B comparison table — complete decision context on one screen
- [x] **UI-EXP-03**: Decision history page shows accepted/rejected experiments with reasoning — the team can review past decisions and understand why the system is configured the way it is

## Future Requirements

### News Pipeline Enhancement

- **NEWS-F01**: Article persistence to database with queryable API endpoint
- **NEWS-F02**: Prompt injection sanitization for LLM inputs

### Code Quality

- **QUAL-01**: Migrate 99 test files from core.trading_loop shim to canonical imports
- **QUAL-02**: Inject _alerter_ref via TradingLoop constructor parameter

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full gRPC SDK replacement | t-tech-investments SDK works, only isolation needed |
| Real-time WebSocket data feeds | Current polling interval matches strategy timeframe |
| Multi-provider LLM orchestration | Single primary + fallback is sufficient |
| LLM-based semantic conflict detection | Nondeterministic, costly, adds latency to hot path |
| Agent consensus overriding backtest | LLM consensus cannot override empirical evidence |
| Real-time inline conflict detection | Adds latency to sub-minute trading cycles |
| Full LLM-to-LLM debate on every cycle | Cost/latency infeasible for daily operation |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| AGOUT-01 | Phase 36 | Complete |
| AGOUT-02 | Phase 36 | Complete |
| CONF-01 | Phase 36 | Complete |
| CONF-02 | Phase 36 | Complete |
| CONF-03 | Phase 36 | Complete |
| CONF-04 | Phase 36 | Complete |
| ORCH-01 | Phase 37 | Complete |
| ORCH-02 | Phase 37 | Complete |
| ORCH-03 | Phase 37 | Complete |
| ORCH-04 | Phase 37 | Complete |
| APPLY-01 | Phase 38 | Complete |
| APPLY-02 | Phase 38 | Complete |
| APPLY-03 | Phase 38 | Complete |
| APPLY-04 | Phase 38 | Complete |
| APPLY-05 | Phase 38 | Complete |
| APPLY-06 | Phase 38 | Complete |
| GRPC-01 | Phase 29 | Complete |
| GRPC-02 | Phase 30 | Complete |
| GRPC-03 | Phase 30 | Complete |
| PERSIST-01 | Phase 31 | Complete |
| PERSIST-02 | Phase 31 | Complete |
| PERSIST-03 | Phase 31 | Complete |
| PERSIST-04 | Phase 31 | Complete |
| PERSIST-05 | Phase 31 | Complete |
| OBS-01 | Phase 29 | Complete |
| OBS-02 | Phase 29 | Complete |
| OBS-03 | Phase 30 | Complete |
| OPS-01 | Phase 28 | Complete |
| OPS-02 | Phase 28 | Complete |
| OPS-03 | Phase 28 | Complete |
| OPS-04 | Phase 28 | Complete |
| SANDBOX-FIX-01 | Phase 32 | Complete |
| SANDBOX-FIX-02 | Phase 32 | Complete |
| SANDBOX-FIX-03 | Phase 32 | Complete |
| SANDBOX-FIX-04 | Phase 32 | Complete |
| SANDBOX-FIX-05 | Phase 32 | Complete |
| SANDBOX-FIX-06 | Phase 32 | Complete |
| SANDBOX-FIX-07 | Phase 32 | Complete |
| SANDBOX-FIX-08 | Phase 32 | Complete |
| SANDBOX-FIX-09 | Phase 32 | Complete |
| SANDBOX-FIX-10 | Phase 32 | Complete |
| DEBATE-01 | Phase 33 | Complete |
| DEBATE-02 | Phase 33 | Complete |
| DEBATE-03 | Phase 33 | Complete |
| EXP-01 | Phase 34 | Complete |
| EXP-02 | Phase 34 | Complete |
| EXP-03 | Phase 34 | Complete |
| EXP-04 | Phase 34 | Complete |
| UI-EXP-01 | Phase 35 | Complete |
| UI-EXP-02 | Phase 35 | Complete |
| UI-EXP-03 | Phase 35 | Complete |

**Coverage:**
- v8.0 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-30*
*Last updated: 2026-04-12 after v8.0 milestone requirements defined*
