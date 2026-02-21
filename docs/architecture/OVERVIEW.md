# Architecture Overview

## System Purpose

Finalayze is an AI-powered multi-market stock trading system that:

1. **Ingests** news (global and regional), social sentiment, and market data
   from multiple sources.
2. **Analyzes** using LLMs (Claude Sonnet) for sentiment scoring and
   fact-checking, plus traditional technical indicators.
3. **Predicts** price movements using an ML ensemble (XGBoost + LightGBM + LSTM)
   trained per segment.
4. **Executes** trades via Alpaca (US markets) and Tinkoff Invest (MOEX).

## High-Level Component Map

```
+-------------------+     +-------------------+     +-------------------+
|   Data Ingestion  |     |   News Pipeline   |     |  Market Data Feed |
|   (data/)         |     |   (analysis/)     |     |  (markets/)       |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                          |
         v                         v                          v
+--------+-------------------------+---------------------------+--------+
|                         Event Bus / Message Layer                     |
|                         (core/events.py)                             |
+--------+-------------------------+---------------------------+--------+
         |                         |                          |
         v                         v                          v
+--------+----------+     +--------+----------+     +--------+----------+
|   ML Pipeline     |     |  Strategy Engine  |     |  Risk Manager     |
|   (ml/)           |     |  (strategies/)    |     |  (risk/)          |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                          |
         v                         v                          v
+--------+-------------------------+---------------------------+--------+
|                         Execution Layer                               |
|                         (execution/)                                  |
+---+-------------------------------------------------------------------+
    |                                          |
    v                                          v
+---+---------------+              +-----------+-------+
|  Alpaca Broker    |              |  Tinkoff Broker   |
|  (US Markets)     |              |  (MOEX)           |
+-------------------+              +-------------------+
```

## Dependency Layers

See [DEPENDENCY_LAYERS.md](DEPENDENCY_LAYERS.md) for the full layering rules.

```
Layer 0: Types & Schemas       core/
Layer 1: Configuration          config/
Layer 2: Data / Repository      data/, markets/
Layer 3: Analysis / ML          analysis/, ml/
Layer 4: Strategy / Risk        strategies/, risk/
Layer 5: Execution              execution/
Layer 6: API / Dashboard        api/, dashboard/
```

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Web framework | FastAPI |
| Database | PostgreSQL + TimescaleDB |
| Cache | Redis |
| ORM | SQLAlchemy 2.0 (async) |
| ML - Gradient Boosting | XGBoost, LightGBM |
| ML - Deep Learning | PyTorch (LSTM) |
| Technical Analysis | pandas-ta |
| LLM | Claude Sonnet (via Anthropic API) |
| US Broker | Alpaca |
| MOEX Broker | Tinkoff Invest |
| Dashboard | Streamlit |
| Package Manager | uv |
| Linter/Formatter | ruff |
| Type Checker | mypy (strict) |
| Testing | pytest, pytest-cov, pytest-asyncio |

## Work Modes

The system supports four operational modes, controlled by `config/modes.py`:

| Mode | Broker | Data Source | Purpose |
|---|---|---|---|
| `debug` | Mock | Fixtures | Local development, verbose logging |
| `sandbox` | Alpaca Paper / Tinkoff Sandbox | Live feeds | Paper trading validation |
| `test` | Simulated | Historical | Automated integration testing, backtesting |
| `real` | Alpaca Live / Tinkoff Live | Live feeds | Production trading |

## Key Design Principles

1. **Layer isolation** -- imports flow downward only, never upward.
2. **Async-first** -- all I/O operations use async/await.
3. **Configuration-driven** -- no magic numbers; everything in config.
4. **Mode-aware** -- every component respects the active work mode.
5. **Segment-oriented** -- ML models and strategies are scoped per segment.
6. **Fail-safe** -- risk checks gate every order before execution.
7. **Observable** -- structured logging, metrics, health checks on every layer.

## Data Storage

- **PostgreSQL + TimescaleDB**: OHLCV data (hypertables), trade history,
  portfolio snapshots, news articles, sentiment scores.
- **Redis**: real-time caches (order book snapshots, latest prices, rate
  limiter counters, session state).
- **File system**: ML model artifacts, backtest results, configuration YAML.

## External Integrations

| Integration | Protocol | Purpose |
|---|---|---|
| Alpaca | REST + WebSocket | US market data and order execution |
| Tinkoff Invest | gRPC + REST | MOEX market data and order execution |
| Anthropic (Claude) | REST | News sentiment analysis, fact-checking |
| News APIs | REST | News article ingestion |
| Social APIs | REST | Social sentiment data |
