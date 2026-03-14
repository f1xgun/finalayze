# Finalayze MOEX MVP

## What This Is

AI-powered autonomous trading system focused on the Russian MOEX market via T-Invest (Tinkoff Invest) API.
The system analyzes news with LLMs, generates signals from technical/ML strategies, manages risk,
and executes real trades in stocks, bonds, and coupon instruments — fully autonomously.

## Core Value

The system must autonomously execute profitable trades on MOEX with acceptable risk limits,
operating 24/7 without human intervention beyond initial configuration and monitoring.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. Inferred from existing codebase. -->

- ✓ Backtest engine with walk-forward validation — existing (`src/finalayze/backtest/`)
- ✓ 5 technical strategies (momentum, dual_momentum, mean_reversion, rsi2_connors, ou_mean_reversion) — existing
- ✓ ADX regime routing (trend vs MR pool gating) — existing (`strategies/adx.py`)
- ✓ Strategy combiner with weighted signal aggregation — existing (`strategies/combiner.py`)
- ✓ Half-Kelly position sizing + 11-check pre-trade pipeline — existing (`risk/`)
- ✓ 3-level circuit breaker — existing (`risk/circuit_breaker.py`)
- ✓ ML ensemble (XGBoost + LightGBM + CatBoost + meta-learner) — existing (`ml/`)
- ✓ 45 technical features with feature selection pipeline — existing (`ml/features/`)
- ✓ LLM client + NewsAnalyzer + EventClassifier — existing (`analysis/`)
- ✓ Tinkoff broker integration (gRPC, sandbox + live) — existing (`execution/tinkoff_broker.py`)
- ✓ Tinkoff data fetcher (candles, dividends, instruments) — existing (`data/fetchers/tinkoff_data.py`)
- ✓ MOEX ISS + CBR fetchers (IMOEX, FX, key rate, turnover) — existing (`data/fetchers/`)
- ✓ Instrument registry with FIGI mapping — existing (`markets/instruments.py`)
- ✓ Currency conversion (RUB/USD) — existing (`markets/`)
- ✓ REST API (20+ endpoints) + Prometheus metrics — existing (`api/`)
- ✓ Streamlit dashboard — existing (`dashboard/`)
- ✓ 4 work modes (debug/sandbox/test/real) — existing (`config/modes.py`)
- ✓ Structured logging (structlog) — existing
- ✓ 2325+ tests — existing

### Active

<!-- Current scope. Building toward these for MOEX MVP. -->

- [ ] Autonomous MOEX stock trading (full autopilot with risk limits)
- [ ] Autonomous MOEX bond trading (yield curve, duration, credit analysis)
- [ ] Autonomous coupon/interest income collection and reinvestment
- [ ] LLM news analysis from T-Invest API in real-time
- [ ] LLM news analysis from Russian media (RBC, Interfax, TASS, Kommersant)
- [ ] LLM news analysis from Telegram financial channels
- [ ] News-driven signal generation (event impact → trading decision)
- [ ] MOEX-specific strategy tuning (parameters optimized for Russian market)
- [ ] MOEX backtests showing positive PnL with walk-forward validation
- [ ] T-Invest sandbox validation (autonomous trading for N days without critical errors)
- [ ] Real money deployment (small account, first real trades)
- [ ] MOEX position sizing in RUB (not USD)
- [ ] Telegram alerts for trades, P&L, circuit breaker events
- [ ] Bond-specific risk management (duration limits, credit risk, yield targets)

### Out of Scope

<!-- Explicit boundaries. Documented to prevent scope creep. -->

- US market trading in this milestone — already works, not MVP focus
- Derivatives/futures trading — complexity too high for MVP
- High-frequency trading — system operates on daily/intraday bars, not tick-level
- Mobile app — Streamlit dashboard + Telegram alerts sufficient
- Multi-account support — single T-Invest account for MVP
- Cryptocurrency — not available on MOEX
- Custom ML model training UI — CLI scripts sufficient

## Context

### Existing Infrastructure
The project already has a mature codebase (367 Python files, 2325+ tests) with working
backtesting, strategies, ML pipeline, and broker integrations. The MOEX-specific work builds
on top of this foundation.

### Known Blockers (from old MVP plan)
Several critical bugs were identified in the 2026-03-04 MVP plan. Some were fixed in weeks 1-5,
but MOEX-specific issues may remain:
- Position sizing in USD instead of RUB (MOEX positions ~0.02% instead of 15%)
- `event_driven` strategy disabled (no real-time news feed)
- MOEX costs not fully wired in backtests
- Daily P&L reports showing zero in Telegram

### Data Sources
- **Market data:** T-Invest gRPC API (candles, instruments, dividends)
- **Index/benchmark:** MOEX ISS REST API (IMOEX index)
- **FX/Macro:** CBR XML API (USD/RUB, key rate)
- **Commodities:** yfinance (Brent crude BZ=F)
- **News:** T-Invest API, Russian media RSS, Telegram channels
- **LLM:** Claude Sonnet for news analysis

### Bond Trading Context
Bond trading is fundamentally different from stocks:
- Yield-to-maturity and duration drive decisions (not momentum/mean-reversion)
- Coupon schedules matter for income strategy
- Credit risk assessment needed (government vs corporate bonds)
- Interest rate sensitivity (CBR key rate changes impact bond prices)
- The existing strategy framework may need bond-specific strategies

## Constraints

- **Broker:** T-Invest (Tinkoff Invest) gRPC API only — no other MOEX brokers
- **Capital:** 500K–2.5M RUB target range
- **Max Drawdown:** 10% hard limit
- **Tech stack:** Python 3.12, existing framework (no rewrites)
- **Data:** MOEX data MUST come from T-Invest API (yfinance cannot fetch MOEX tickers)
- **Timeline:** Go live only when system proves itself in sandbox
- **Risk:** Full autopilot requires robust circuit breakers and risk limits

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| MOEX-only focus for MVP | US already works; MOEX needs fixing | — Pending |
| Full autopilot (not semi-auto) | User wants hands-off operation | — Pending |
| All 3 instrument types (stocks, bonds, coupons) | User requirement — equal priority | — Pending |
| LLM news in MVP (not v2) | Core differentiator for MOEX | — Pending |
| Backtest → sandbox → real path | Safe deployment strategy | — Pending |
| Use existing strategy framework | Avoid rewrite, adapt for MOEX | — Pending |
| T-Invest + СМИ + Telegram for news | Multiple sources for coverage | — Pending |

---
*Last updated: 2026-03-14 after initialization*
