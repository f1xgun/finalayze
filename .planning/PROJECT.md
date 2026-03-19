# Finalayze MOEX MVP

## What This Is

AI-powered autonomous trading system for the Russian MOEX market via T-Invest (Tinkoff Invest) API.
The system ingests Russian news (RSS + Telegram) with LLM analysis, generates signals from
6 technical strategies + event-driven sentiment, manages risk with Half-Kelly sizing and circuit breakers,
and executes real trades in stocks and OFZ bonds — fully autonomously.

## Core Value

The system must autonomously execute profitable trades on MOEX with acceptable risk limits,
operating 24/7 without human intervention beyond initial configuration and monitoring.

## Current Milestone: v2.0 MOEX Profitability

**Goal:** Transform MOEX equity from consistently negative Sharpe to profitable operation through universe cleanup, MOEX-native strategies (dividend gap, CBR regime, sector rotation), and ML with Russian macro features.

**Target features:**
- Universe surgery: remove toxic symbols (GAZP, VTBR, SNGS, IRAO, ALRS), raise confidence thresholds
- Dividend gap closure as primary MOEX equity alpha engine (expand from 43 to 150+ events)
- CBR rate and macro regime gating for equity positions
- MOEX-specific sector rotation (Brent-gated energy, CBR-sensitive financials)
- ML ensemble for MOEX with Russian macro features (CBR rate, USDRUB, Brent, IMOEX relative)
- Portfolio assembly: OFZ carry 40% + equity 60%, RUB crisis brake

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

### Active

<!-- v2.0 MOEX Profitability scope -->

- [ ] Universe cleanup: remove structurally broken symbols, raise confidence thresholds
- [ ] Dividend gap closure strategy as primary MOEX equity alpha engine
- [ ] Preferred share arbitrage (SBER/SBERP, TATN/TATNP, SNGS/SNGSP)
- [ ] CBR rate regime gating for equity and bond allocation
- [ ] Brent price gate for energy sector momentum
- [ ] RUB/oil decorrelation regime wiring into position sizing
- [ ] OFZ PK→PD rotation trigger (CBR cutting cycle detection)
- [ ] MOEX-specific sector rotation strategy (replace US-calibrated momentum)
- [ ] ML ensemble for ru_* segments with Russian macro features
- [ ] Portfolio-level allocation (40% OFZ + 60% equity) with RUB crisis brake

### Out of Scope

- Derivatives/futures trading — complexity too high
- High-frequency trading — system operates on daily/intraday bars
- Mobile app — Streamlit dashboard + Telegram alerts sufficient
- Cryptocurrency — not available on MOEX
- Custom ML model training UI — CLI scripts sufficient
- US market development — deferred, MOEX-only focus for v2.0
- Multi-account support — deferred to v3.0
- Tax optimization (NDFL, IIS) — deferred to v3.0

## Context

### Current State (v1.0 shipped)

Codebase: ~400 Python files, 35,199 LOC, 3,651 tests.
Tech stack: Python 3.12, FastAPI, SQLAlchemy 2.0 async, PostgreSQL+TimescaleDB, Redis,
XGBoost, LightGBM, CatBoost, PyTorch, pandas-ta, QuantLib, feedparser, Telethon.

All v1.0 MOEX MVP requirements shipped. System ready for sandbox deployment and controlled go-live.

### Known Issues
- MOEX walk-forward Sharpe still negative on aggregate (individual symbols profitable)
- event_driven strategy shows 0 backtest trades (expected — needs live news)
- RSS URLs (Interfax, TASS) need live validation at deployment
- Nyquist validation partial (3/7 phases fully compliant)

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

| MOEX-only focus for v2.0 | MOEX equity needs fixing; US already works | — Pending |
| Universe surgery first | Toxic symbols (GAZP, VTBR, SNGS) account for 60% negative PnL | — Pending |
| Dividend gap as primary alpha | Documented 70%+ gap closure on MOEX blue chips within 30-60 days | — Pending |
| OFZ carry as portfolio foundation | Sharpe +1.14, provides 20% base return at 21% CBR rate | ✓ Good |

---
*Last updated: 2026-03-20 after v2.0 milestone start*
