# src/ — Source Packages (Area Node)

Parent: [root AGENTS.md](../AGENTS.md) · Sibling areas: `config/`, `tests/`, `scripts/`, `docs/`

## Layout

Only one package lives here — `finalayze/`. Walk into it for the module graph.

→ [`src/finalayze/AGENTS.md`](finalayze/AGENTS.md)

## Root-level files

| File | Purpose |
|---|---|
| `finalayze/__init__.py` | Package init (intentionally empty for import speed) |
| `finalayze/main.py` | FastAPI app factory + uvicorn entry point |
| `finalayze/bootstrap.py` | Dependency wiring: constructs broker router, data loader, strategies, risk pipeline |
| `finalayze/py.typed` | PEP 561 marker (typed package) |

## Dependency layer cheat-sheet (enforced by import rules)

```
Layer 0  core/        schemas, exceptions, event bus, ORM, clock
Layer 1  config/      (lives at /config, not under src/)
Layer 2  data/, markets/
Layer 3  analysis/, ml/
Layer 4  strategies/, risk/
Layer 5  execution/, orchestration/
Layer 6  api/, dashboard/, monitoring/
```

Import rule: a module at layer N may import from layers `< N` only. Violations caught by
`systems-architect` agent review and the layer-check test suite.

## Typical "where do I edit?" routing

| Task keyword | Module |
|---|---|
| "add a strategy", "signal", "ADX routing" | [`strategies/`](finalayze/strategies/AGENTS.md) |
| "position sizing", "circuit breaker", "stop loss" | [`risk/`](finalayze/risk/AGENTS.md) |
| "walk-forward", "backtest run", "performance metrics" | [`backtest/`](finalayze/backtest/AGENTS.md) |
| "feature engineering", "model training", "ensemble" | [`ml/`](finalayze/ml/AGENTS.md) |
| "fetch candles", "MOEX data", "news feed" | [`data/`](finalayze/data/AGENTS.md) |
| "Alpaca", "Tinkoff broker", "order routing" | [`execution/`](finalayze/execution/AGENTS.md) |
| "REST endpoint", "Prometheus", "Telegram bot" | [`api/`](finalayze/api/AGENTS.md) |
| "instrument registry", "FX rates", "market hours" | [`markets/`](finalayze/markets/AGENTS.md) |
| "LLM prompt", "sentiment", "event classifier" | [`analysis/`](finalayze/analysis/AGENTS.md) |
| "trading loop", "bond cycle" | [`orchestration/`](finalayze/orchestration/AGENTS.md) |
| "health monitor", "go/no-go", "anomaly alert" | [`monitoring/`](finalayze/monitoring/AGENTS.md) |
| "Pydantic schema", "domain exception", "event bus" | [`core/`](finalayze/core/AGENTS.md) |
