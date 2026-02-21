# API Endpoints

## Base URL

`/api/v1`

## System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health (all brokers) |
| GET | `/mode` | Current work mode |
| POST | `/mode` | Switch work mode |

## Markets & Segments

| Method | Path | Description |
|--------|------|-------------|
| GET | `/markets` | List markets (US, MOEX) |
| GET | `/markets/{id}/segments` | Segments in a market |
| GET | `/segments/{id}` | Segment details + strategy config |
| PUT | `/segments/{id}/strategies` | Update segment strategy config |

## Portfolio

| Method | Path | Description |
|--------|------|-------------|
| GET | `/portfolio` | Unified portfolio (all markets, base currency) |
| GET | `/portfolio/{market_id}` | Per-market portfolio |
| GET | `/portfolio/positions` | All positions across markets |
| GET | `/portfolio/history` | Historical snapshots |

## Trades

| Method | Path | Description |
|--------|------|-------------|
| GET | `/trades` | All trades (filterable by market/segment) |
| POST | `/trades/manual` | Manual trade (specify broker) |

## Signals & Strategies

| Method | Path | Description |
|--------|------|-------------|
| GET | `/signals` | Recent signals (filterable) |
| GET | `/strategies` | List all strategies |
| PUT | `/strategies/{name}/toggle` | Enable/disable per segment |

## Risk

| Method | Path | Description |
|--------|------|-------------|
| GET | `/risk/status` | Per-market + aggregate risk |
| POST | `/risk/emergency-stop` | Emergency halt (all or per-market) |

## Backtest

| Method | Path | Description |
|--------|------|-------------|
| POST | `/backtest/run` | Multi-market backtest |
| GET | `/backtest/{id}/results` | Results per segment |

## Data

| Method | Path | Description |
|--------|------|-------------|
| GET | `/data/candles/{market}/{symbol}` | OHLCV data |
| GET | `/data/news` | News (filterable by scope/language) |
| GET | `/data/currency/{pair}` | Currency rate history |

## ML

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ml/models` | Per-segment models |
| POST | `/ml/models/train` | Train (specific segment) |

## Status

**Phase 1:** Health, mode, basic portfolio endpoints.
**Phase 2:** Full endpoint set implemented.
