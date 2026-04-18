# src/finalayze/ — Package Node

Parent: [`src/AGENTS.md`](../AGENTS.md) · Graph root: [`/AGENTS.md`](../../AGENTS.md)

This is the sole production package. Every subfolder is a module node with its own `AGENTS.md`.

## Module index

| Module | Layer | One-liner | Node |
|---|---|---|---|
| `core/` | 0 | Pydantic schemas, exception hierarchy, Redis event bus, SQLAlchemy ORM, clock, work modes | [AGENTS.md](core/AGENTS.md) |
| `data/` | 2 | Market data fetchers (yfinance, Tinkoff gRPC, MOEX ISS, CBR, Finnhub, RSS, Telegram) + cache + normalizer | [AGENTS.md](data/AGENTS.md) |
| `markets/` | 2 | Market registry (US/MOEX), instrument registry with FIGI, FX service, trading calendar | [AGENTS.md](markets/AGENTS.md) |
| `analysis/` | 3 | LLM clients (Anthropic/OpenAI/OpenRouter), news sentiment, entity/event classification | [AGENTS.md](analysis/AGENTS.md) |
| `ml/` | 3 | 45-feature engineering, XGBoost/LightGBM/CatBoost/LSTM ensemble, calibration, meta-labeling | [AGENTS.md](ml/AGENTS.md) |
| `strategies/` | 4 | 8 strategies (5 enabled) + ADX regime routing + combiner with preset weights | [AGENTS.md](strategies/AGENTS.md) |
| `risk/` | 4 | Sizing pipeline (Kelly→VolTarget→...→HardCaps), 11-check pre-trade, 3-level circuit breaker | [AGENTS.md](risk/AGENTS.md) |
| `execution/` | 5 | Broker ABC + Alpaca/Tinkoff/Simulated brokers, retry policy, router | [AGENTS.md](execution/AGENTS.md) |
| `orchestration/` | 5 | APScheduler trading loop + bond cycle processor | [AGENTS.md](orchestration/AGENTS.md) |
| `backtest/` | 4–5 | Backtest engine, walk-forward, performance analyzer, iteration tracker, Monte Carlo | [AGENTS.md](backtest/AGENTS.md) |
| `api/` | 6 | FastAPI routers (v1), Prometheus metrics, Telegram bot/alerter | [AGENTS.md](api/AGENTS.md) |
| `monitoring/` | 6 | Health monitor, sandbox monitor, anomaly detector, go/no-go reporter | [AGENTS.md](monitoring/AGENTS.md) |
| `dashboard/` | 6 | Streamlit dashboard (no AGENTS.md yet — thin UI layer over the API) | — |

## Cross-cutting contracts (canonical for all modules)

- `from __future__ import annotations` in every file
- Pydantic v2 models are `ConfigDict(frozen=True)`
- Decimal for money, float for probabilities / ratios
- Exception names end in `Error` (ruff `N818`)
- Async boundaries: SQLAlchemy 2.0 async, `httpx`, broker gRPC / HTTP
- Timestamps: UTC-aware everywhere

## Dependency contract

Imports flow downward across layers only. `TYPE_CHECKING` is the only valid escape for
inverted references (orchestration → monitoring types, for example).
