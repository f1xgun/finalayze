# Finalayze MOEX MVP

## What This Is

AI-powered autonomous trading system for the Russian MOEX market via T-Invest (Tinkoff Invest) API.
The system ingests Russian news (RSS + Telegram) with LLM analysis, generates signals from
6 technical strategies + event-driven sentiment, manages risk with Half-Kelly sizing and circuit breakers,
and executes real trades in stocks and OFZ bonds — fully autonomously.

## Core Value

The system must autonomously execute profitable trades on MOEX with acceptable risk limits,
operating 24/7 without human intervention beyond initial configuration and monitoring.

## Current Milestone: v9.1 MOEX ML Model Quality

**Goal:** Raise ML model quality on failing MOEX segments (ru_energy, ru_tech, ru_finance) to pass quality gates, using insights from parallel agent analysis (quant-analyst, ml-engineer, data-quality).

**Target features:**
- Model complexity reduction for MOEX (depth=3, estimators=100)
- Brent crude cross-asset features for ru_energy (returns, momentum)
- XGBoost/LightGBM scale_pos_weight consistency fix
- Stable feature selection (once on full pre-test data, not per-fold)
- Segment restructuring (ru_finance SBERP removal, ru_tech min-history check)
- Asymmetric triple barrier for energy stocks (wider lower barrier)

**Agent analysis (2026-04-14):** 3 parallel agents diagnosed root causes:
- ml-engineer: overfitting (depth=5 on 850 samples), label imbalance from symmetric barriers, per-fold MI instability
- quant-analyst: intra-segment correlation kills diversity, ru_tech IPOs too recent, SBER+SBERP zero independent signal
- data-quality: HEAD ~370d, YDEX ~450d history; brier ~0.25 across all segments = poor calibration

## Next Milestone: v10.0 Runtime LLM Trading Agents (planned, not started)

**Goal:** Add runtime LLM agents to the live trading pipeline.

## Current State: v9.0 shipped, v9.1 in progress

v9.0 ML AutoResearch & MOEX Adaptation shipped 2026-04-13. 5 phases, 7 plans.
auto_ml_research.py now runs on all MOEX equity segments via TinkoffFetcher with macro features
(CBR rate, USDRUB, IMOEX, Brent) and 2-bar look-ahead bias prevention.
Adaptive quality gates: min_signals=15 for MOEX, degenerate predictor guard (0.15-0.85 bounds),
MOEX walk-forward fold constants (8/1/3/21/2mo) producing 3+ folds on 730-day data.
ExperimentManager integration via --experiment-id with ACCEPT/REJECT/INCONCLUSIVE verdict lifecycle.
3 new search strategies: ensemble_weights (33 simplex configs), cross_segment_transfer (US→MOEX),
feature_engineering (domain-motivated combos + permutation importance filter).

<details>
<summary>v8.0 Agent Integration & Autonomous Decision Loop (2026-04-12)</summary>

v8.0 shipped 2026-04-12. 4 phases, 7 plans.
Agents emit structured AgentOutput with sourced Claims via parse_structured() on all 5 LLM clients.
ConflictDetector with deterministic rule-based contradiction detection, 3-level severity, SHA-256 dedup.
AgentOrchestrator pipeline: conflict → debate → arbiter → experiment → verdict, with snapshot_sha safety.
REST API for debates and experiments with X-API-Key auth.
PresetApplicator with 7-gate safety pipeline. Position-ownership tracking in TradingLoop.
</details>

<details>
<summary>v7.0 Agent Intelligence & Experiment Framework (2026-04-12)</summary>

v7.0 shipped 2026-04-12. 4 phases, 10 plans.
Sandbox signal fixes, CachingFetcher + RateLimiter, ML quality gates.
Structured debate protocol, experiment registry with hypothesis lifecycle, Experiment Lab UI.
</details>

<details>
<summary>v6.0 Sandbox Stability & Observability (2026-03-31)</summary>

v6.0 Sandbox Stability & Observability shipped 2026-03-31. 4 phases, 8 plans.
Fixed gRPC event loop isolation (dedicated loop eliminates 60-min cycle drift).
T-Bank 70001 errors handled with portfolio cache fallback + auto-reconnect.
DB persistence wired for orders, signals, news articles, sentiment scores (fire-and-forget).
Loki log pipeline fixed (Promtail volume mount + JSON parsing + 30-day retention).
FX rate CBR XML fallback, market-hours gate, stale tickers cleaned, LLM article dedup added.
</details>

v4.0 Architecture Hardening shipped 2026-03-22. 4 phases, 10 plans.
Fixed concurrency bugs (stop-loss TOCTOU, async lock, session leak), async correctness
(non-blocking reconnect, coroutine-aware retry, run_in_executor), error handling
(GARCH NaN fallback, EventBus narrowed catch, consecutive error alerting, /kill auth),
and dependency layers (orchestration/ extraction, MetricsCollector DI, dead code removal).
v3.0 integration gaps closed (Telegram /gonogo import, HealthMonitor feed freshness).

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ Backtest engine with walk-forward validation — v1.0
- ✓ 5 technical strategies + event_driven news strategy — v1.0
- ✓ ADX regime routing (trend vs MR pool gating) — v1.0
- ✓ Strategy combiner with weighted signal aggregation — v1.0
- ✓ Half-Kelly position sizing + 11-check pre-trade pipeline — v1.0
- ✓ 3-level circuit breaker — v1.0
- ✓ ML ensemble (XGBoost + LightGBM + CatBoost + meta-learner) — existing (us_tech only)
- ✓ 45 technical features with feature selection pipeline — existing
- ✓ LLM client + NewsAnalyzer + EventClassifier + EntityExtractor — v1.0
- ✓ Tinkoff broker integration (gRPC, sandbox + live) — v1.0
- ✓ Tinkoff data fetcher (candles, dividends, instruments) — v1.0
- ✓ MOEX ISS + CBR fetchers (IMOEX, FX, key rate, turnover) — v1.0
- ✓ Instrument registry with FIGI mapping — v1.0
- ✓ Currency conversion (RUB/USD) — v1.0
- ✓ REST API (20+ endpoints) + Prometheus metrics — v1.0
- ✓ Streamlit dashboard — v1.0
- ✓ 4 work modes (debug/sandbox/test/real) — v1.0
- ✓ Structured logging (structlog) — v1.0
- ✓ RUB-native position sizing for MOEX — v1.0
- ✓ MOEX holiday calendar (transferred holidays) — v1.0
- ✓ MOEX costs (commissions, slippage) in backtest — v1.0
- ✓ MOEX-specific strategy parameters (Optuna-tuned) — v1.0
- ✓ Bond data pipeline (QuantLib YTM, duration, convexity) — v1.0
- ✓ Bond execution (BondCycleProcessor, limit orders, carry strategy) — v1.0
- ✓ Telegram monitoring (priority queue, trade/coupon/CBR alerts) — v1.0
- ✓ Telegram bot (/status, /breakers, /stop commands) — v1.0
- ✓ Autonomous TradingLoop (equity + bond + news cycles) — v1.0
- ✓ Sandbox validation infrastructure — v1.0
- ✓ Russian news RSS fetcher (RBC, Interfax, TASS) — v1.0
- ✓ Telegram channel reader (Telethon) — v1.0
- ✓ LLM entity extraction (news → MOEX tickers) — v1.0
- ✓ event_driven strategy enabled on all ru_* segments (15% weight) — v1.0
- ✓ Go-live configuration with real_confirmed guard — v1.0
- ✓ 3,651 tests — v1.0
- ✓ Universe cleanup: toxic symbols removed, confidence thresholds raised — v2.0
- ✓ Dividend gap closure strategy with expanded calendar (150+ events) — v2.0
- ✓ Preferred share arbitrage (SBER/SBERP, TATN/TATNP) with Kalman filter — v2.0
- ✓ CBR rate regime gating via yield curve slope — v2.0
- ✓ Brent price gate for energy sector (BrentGateStep) — v2.0
- ✓ RUB/oil decorrelation regime in sizing pipeline — v2.0
- ✓ OFZ PK→PD rotation on CBR cutting cycle — v2.0
- ✓ Sector allocation (energy Brent-tiered, financials CBR-sensitive) — v2.0
- ✓ ML ensemble for ru_blue_chips with 10 Russian macro features — v2.0
- ✓ Portfolio-level allocation (40% OFZ + 60% equity) with USDRUB crisis brake — v2.0
- ✓ PortfolioBacktestOrchestrator with walk-forward Sharpe — v2.0

- ✓ Sandbox monitoring dashboard with real-time metric collection — v3.0
- ✓ Automated go/no-go gate report with pass/fail thresholds — v3.0
- ✓ Gradual rollout configuration (tightened limits for minimal capital) — v3.0
- ✓ Production health monitoring and kill switch — v3.0
- ✓ Capital ladder validation at 50K/150K/500K/2.5M RUB tiers — v3.0

- ✓ Stop-loss atomicity (no double-sell TOCTOU race) — v4.0
- ✓ asyncio.Lock in async broker paths (no threading.Lock deadlock) — v4.0
- ✓ Thread-safe event loop initialization — v4.0
- ✓ macro_cache session scoping with async-with — v4.0
- ✓ Non-blocking gRPC reconnect (asyncio.sleep, not time.sleep) — v4.0
- ✓ Coroutine-aware RetryPolicy.aexecute() — v4.0
- ✓ Portfolio API run_in_executor (non-blocking FastAPI) — v4.0
- ✓ Async-safe sandbox monitor persistence — v4.0
- ✓ GARCH NaN fallback with rolling volatility — v4.0
- ✓ EventBus narrowed exception handling (redis.ResponseError only) — v4.0
- ✓ Structured TinkoffFetcher error logging — v4.0
- ✓ Consecutive error alerting in TradingLoop and BondCycle — v4.0
- ✓ POST /kill authentication via X-API-Key — v4.0
- ✓ Orchestration module extraction (core/ → orchestration/) — v4.0
- ✓ MetricsCollector dependency injection — v4.0
- ✓ Dead event bus streams removed — v4.0
- ✓ Stub API endpoints return 501 Not Implemented — v4.0
- ✓ TinkoffBroker.close() structured logging — v4.0
- ✓ TinkoffFetcher configurable gRPC timeout — v4.0
- ✓ httpx client lifecycle management — v4.0
- ✓ Telegram /gonogo import fixed — v4.0
- ✓ HealthMonitor feed freshness wired — v4.0

- ✓ gRPC event loop isolation — dedicated loop eliminates BlockingIOError — v6.0
- ✓ T-Bank 70001 resilience — portfolio cache fallback + auto-reconnect — v6.0
- ✓ DB persistence for orders, signals, news articles, sentiment scores (fire-and-forget) — v6.0
- ✓ Loki log pipeline operational — Promtail ships all 7 container logs — v6.0
- ✓ FX rate CBR XML API fallback — v6.0
- ✓ Market-hours gate in strategy cycle — v6.0
- ✓ Stale tickers cleaned (HHRU→HEAD, FIVE/FIXP/POLY removed, YNDX→YDEX) — v6.0
- ✓ LLM article deduplication (SHA-256, 24h TTL) — v6.0
- ✓ Telegram alerter startup resilience — v6.0

- ✓ Agents emit AgentOutput with structured Claims, parse_structured() on all LLM clients — v8.0
- ✓ ConflictDetector with deterministic rule-based contradiction detection + severity scoring — v8.0
- ✓ AgentOrchestrator pipeline: conflict → debate → arbiter → experiment → verdict — v8.0
- ✓ REST API for debates and experiments with X-API-Key auth — v8.0
- ✓ snapshot_sha safety on FileLineSource for stale claim detection — v8.0
- ✓ PresetApplicator with 7-gate safety pipeline (circuit breaker, sandbox gate, atomic write) — v8.0
- ✓ Position-ownership tracking (_entry_strategy) and INCONCLUSIVE Telegram routing — v8.0
- ✓ agent-orchestrator.md Claude Code sub-agent for autonomous pipeline runs — v8.0

- ✓ MOEX data adapter for auto_ml_research (TinkoffFetcher, symbols from segments.py) — v9.0
- ✓ MOEX macro features in ML pipeline (CBR rate, USDRUB, IMOEX, Brent) with 2-bar bias prevention — v9.0
- ✓ Adaptive quality gates for small MOEX datasets (min_signals=15, degenerate predictor guard) — v9.0
- ✓ MOEX walk-forward fold constants (8/1/3/21/2mo) producing 3+ folds on 730-day data — v9.0
- ✓ ExperimentManager integration via --experiment-id (hypotheses, verdicts, backward-compatible JSONL) — v9.0
- ✓ Ensemble weight optimization strategy (33 simplex configs, 0.7 cap, small-fold guard) — v9.0
- ✓ Cross-segment transfer strategy (US→MOEX market-neutral feature filtering) — v9.0
- ✓ Feature engineering strategy (domain-motivated combos, n_samples/20 cap, permutation filter) — v9.0

### Active

<!-- v10.0 Runtime LLM Trading Agents -->

- [ ] News ingestion pipeline with real MOEX sources (RSS, Telegram channels, T-Pulse)
- [ ] EventDrivenStrategy enabled on live news feed with credibility cap 0.7
- [ ] Portfolio Review Agent — daily LLM portfolio analysis, advisory only
- [ ] Anomaly Interpreter Agent — LLM explanation of anomalies, fire-and-forget
- [ ] Sentiment data persistence + rolling aggregation for future ML features

### Out of Scope

- Derivatives/futures trading — complexity too high
- High-frequency trading — system operates on daily/intraday bars
- Mobile app — Streamlit dashboard + Telegram alerts sufficient
- Cryptocurrency — not available on MOEX
- Custom ML model training UI — CLI scripts sufficient
- US market development — deferred, MOEX-only focus
- Multi-account support — deferred to v5.0+
- Tax optimization (NDFL, IIS) — deferred to v5.0+

## Context

### Current State (v6.0 shipped)

Codebase: ~480 Python files, ~40,000 LOC.
Tech stack: Python 3.12, FastAPI, SQLAlchemy 2.0 async, PostgreSQL+TimescaleDB, Redis,
XGBoost, LightGBM, CatBoost, PyTorch, pandas-ta, QuantLib, feedparser, Telethon.

v6.0 shipped: Sandbox stability complete. 4 phases (28-31), 8 plans, ~52 new tests.
gRPC isolated on dedicated event loop — strategy cycles no longer drift 60 min.
T-Bank 70001 errors handled with portfolio cache + auto-reconnect.
All 4 DB tables wired (orders, signals, news_articles, sentiment_scores) with fire-and-forget.
Loki pipeline fixed — all 7 containers ship logs with 30-day retention.
FX rate fallback via CBR XML, market-hours gate, LLM article dedup, alerter resilience.

### Known Issues
- ML quality gates fail for small MOEX datasets (accuracy cap at 0.55 for n_eff<20)
- event_driven strategy shows 0 backtest trades (expected — needs live news)
- Portfolio CLI requires FINALAYZE_TINKOFF_TOKEN for real data
- Walk-forward Sharpe on blended portfolio not yet validated with live data
- 99 test files still use core.trading_loop shim (functional but should migrate)
- _alerter_ref set via attribute mutation, not constructor injection

### Data Sources
- **Market data:** T-Invest gRPC API (candles, instruments, dividends)
- **Index/benchmark:** MOEX ISS REST API (IMOEX index)
- **FX/Macro:** CBR XML API (USD/RUB, key rate)
- **Commodities:** yfinance (Brent crude BZ=F)
- **News:** Russian media RSS (RBC, Interfax, TASS), Telegram channels
- **LLM:** OpenRouter (free model) for entity extraction and sentiment analysis

## Constraints

- **Broker:** T-Invest (Tinkoff Invest) gRPC API only
- **Capital:** 500K–2.5M RUB target range
- **Max Drawdown:** 10% hard limit
- **Tech stack:** Python 3.12, existing framework
- **Data:** MOEX data MUST come from T-Invest API (yfinance cannot fetch MOEX tickers)
- **Risk:** Full autopilot requires robust circuit breakers and risk limits

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| MOEX-only focus for MVP | US already works; MOEX needs fixing | ✓ Good — focused scope enabled 22-day delivery |
| Full autopilot (not semi-auto) | User wants hands-off operation | ✓ Good — TradingLoop autonomous with circuit breakers |
| All 3 instrument types (stocks, bonds, coupons) | User requirement — equal priority | ✓ Good — equity + OFZ bonds both operational |
| LLM news in MVP (not v2) | Core differentiator for MOEX | ✓ Good — RSS + Telegram + entity extraction shipped |
| Backtest → sandbox → real path | Safe deployment strategy | ✓ Good — validation infrastructure in place |
| Use existing strategy framework | Avoid rewrite, adapt for MOEX | ✓ Good — event_driven added to existing combiner |
| T-Invest + СМИ + Telegram for news | Multiple sources for coverage | ✓ Good — 3 RSS sources + Telegram channels |
| QuantLib for bond math | Accurate YTM/duration calculations | ✓ Good — cross-validated with manual calculations |
| OFZ carry strategy (not duration rotation) | Carry positive Sharpe, duration negative in hiking cycle | ✓ Good — ru_ofz_pk Sharpe +1.14 |
| OpenRouter free model for LLM | Cost efficiency for news analysis | ✓ Good — avoids per-call charges |
| Three-quarter Kelly for MOEX | Larger positions for less liquid market | ⚠️ Revisit — monitor in live trading |
| Rollout phases for gradual capital | Safety for first live deployment | ✓ Good — MINIMAL/STANDARD/FULL with env var override |
| Standalone monitoring services | Not embedded in TradingLoop (per research) | ✓ Good — clean separation, testable |
| Go/no-go is advisory, not automated | Human decides live promotion | ✓ Good — PROCEED/DEFER/ABORT report |
| File-based kill flag | Works even when DB is down | ✓ Good — persistent across restarts |

| MOEX-only focus for v2.0 | MOEX equity needs fixing; US already works | ✓ Good — focused delivery in 2 days |
| Universe surgery first | Toxic symbols (GAZP, VTBR, SNGS) account for 60% negative PnL | ✓ Good — removed from all segments |
| Dividend gap as primary alpha | Documented 70%+ gap closure on MOEX blue chips within 30-60 days | ✓ Good — yield-based hold bars + event bypass |
| OFZ carry as portfolio foundation | Sharpe +1.14, provides 20% base return at 21% CBR rate | ✓ Good — 40/60 allocation with crisis brake |
| ML reinforcer-only for MOEX | Quality gates infeasible for small datasets | ⚠️ Revisit — cap threshold helps but gates still strict |
| Sector allocation in sizing (not combiner) | Architectural constraint from requirements | ✓ Good — clean separation of concerns |
| sys.modules shims for backward compat | Avoid updating 99+ test imports during module move | ✓ Good — zero-breakage migration, shims transparent |
| MetricsCollector via constructor DI | Eliminate 6 deferred L6 imports in TradingLoop | ✓ Good — clean layer boundary |
| asyncio.Lock for async, threading.Lock for sync | Separate lock types for separate execution contexts | ✓ Good — eliminates latent deadlock |
| GARCH rolling vol fallback over NaN | NaN propagation through sizing pipeline is dangerous | ✓ Good — safe fallback with warning logged |
| 501 Not Implemented for stub endpoints | Empty 200 responses mislead API consumers | ✓ Good — clear signal that feature is pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-14 after v10.0 milestone started*
