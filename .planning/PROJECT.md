# Finalayze MOEX MVP

## What This Is

AI-powered autonomous trading system for the Russian MOEX market via T-Invest (Tinkoff Invest) API.
The system ingests Russian news (RSS + Telegram) with LLM analysis, generates signals from
6 technical strategies + event-driven sentiment, manages risk with Half-Kelly sizing and circuit breakers,
and executes real trades in stocks and OFZ bonds — fully autonomously.

## Core Value

The system must autonomously execute profitable trades on MOEX with acceptable risk limits,
operating 24/7 without human intervention beyond initial configuration and monitoring.

## Current Milestone: v4.0 Architecture Hardening

**Goal:** Fix critical architectural defects found in comprehensive audit — dependency layer violations, async correctness bugs, dead infrastructure, error handling gaps that risk money loss or silent degradation.

**Target areas:**
- Critical concurrency bugs (stop-loss double-sell, async retry, session leaks)
- Async correctness (blocking sleeps, threading.Lock in async, TOCTOU races)
- Dependency layer cleanup (extract orchestrators from core/, assign missing layers)
- Error handling hardening (GARCH NaN propagation, overly broad exception suppression)
- Dead code removal (unused event bus infrastructure)
- API correctness (blocking broker calls in async endpoints, unauthenticated /kill)

## Current State: v3.0 shipped

v3.0 Production Readiness shipped 2026-03-22. 4 phases, 10 plans, +5,097 LOC.
Sandbox monitoring, Go/No-Go gate, kill switch, health monitor, and dashboard all operational.
Two known integration bugs accepted (Telegram /gonogo import, HealthMonitor feed freshness wiring).

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

### Active

<!-- v4.0 Architecture Hardening -->

- [ ] Fix stop-loss TOCTOU race condition (potential double-sell)
- [ ] Fix async retry bugs (sync fn() in aexecute, blocking sleep in execute)
- [ ] Fix macro_cache session leak (no async with, no rollback)
- [ ] Fix TinkoffBroker threading.Lock in async code (latent deadlock)
- [ ] Fix TinkoffBroker TOCTOU race on event loop creation
- [ ] Fix GARCH NaN propagation into sizing pipeline
- [ ] Fix EventBus suppress(Exception) — narrow to suppress(ResponseError)
- [ ] Fix portfolio API endpoint blocking event loop (sync broker calls)
- [ ] Replace time.sleep(300) in gRPC reconnect with non-blocking approach
- [ ] Add authentication to POST /kill endpoint
- [ ] Extract trading_loop.py and bond_cycle.py from core/ to orchestration/
- [ ] Remove or productionize dead event bus infrastructure
- [ ] Assign layers to backtest/ and monitoring/ modules
- [ ] Fix Telegram /gonogo import bug (OPS-04 integration gap)
- [ ] Wire HealthMonitor.update_feed_timestamp() into TradingLoop (OPS-02 gap)

### Out of Scope

- Derivatives/futures trading — complexity too high
- High-frequency trading — system operates on daily/intraday bars
- Mobile app — Streamlit dashboard + Telegram alerts sufficient
- Cryptocurrency — not available on MOEX
- Custom ML model training UI — CLI scripts sufficient
- US market development — deferred, MOEX-only focus
- Multi-account support — deferred to v4.0+
- Tax optimization (NDFL, IIS) — deferred to v4.0+

## Context

### Current State (v3.0 shipped)

Codebase: ~460 Python files, ~42,000 LOC.
Tech stack: Python 3.12, FastAPI, SQLAlchemy 2.0 async, PostgreSQL+TimescaleDB, Redis,
XGBoost, LightGBM, CatBoost, PyTorch, pandas-ta, QuantLib, feedparser, Telethon.

v3.0 shipped: Production readiness infrastructure complete. 4 phases (15-18), 10 plans.
Rollout phase system (MINIMAL/STANDARD/FULL) with risk limit overrides.
SandboxMonitorService persists per-cycle metrics to TimescaleDB.
GoNoGoReporter evaluates 8 criteria with backtest-derived thresholds.
KillSwitch + HealthMonitor + Telegram /kill and /gonogo commands.
Streamlit sandbox dashboard + REST /sandbox/gonogo endpoint.

### Known Issues
- ML quality gates fail for small MOEX datasets (accuracy cap at 0.55 for n_eff<20)
- ML us_tech quality gates regressed after schema v3 bump (brier/class_balance strict)
- event_driven strategy shows 0 backtest trades (expected — needs live news)
- Portfolio CLI requires FINALAYZE_TINKOFF_TOKEN for real data
- Walk-forward Sharpe on blended portfolio not yet validated with live data

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

---
*Last updated: 2026-03-22 after v4.0 milestone started*
