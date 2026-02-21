# Roadmap

## Phase Overview

| Phase | Name | Duration | Status |
|-------|------|----------|--------|
| 0 | Code Quality Foundation | 2-3 days | COMPLETE |
| 1 | Foundation, US Market & Sandbox | Weeks 1-5 | NOT STARTED |
| 2 | MOEX + Tinkoff, Intelligence & Test Mode | Weeks 6-10 | NOT STARTED |
| 3 | Hardening & Advanced | Weeks 11-14 | NOT STARTED |
| 4 | Real Trading & Optimization | Weeks 15-18 | NOT STARTED |

## Phase 0: Code Quality Foundation (COMPLETE)

- [x] Project initialization with uv
- [x] pyproject.toml with full dependency spec
- [x] ruff configuration (full rule set)
- [x] mypy configuration (strict mode)
- [x] Pre-commit hooks
- [x] Docker Compose (PostgreSQL + TimescaleDB + Redis)
- [x] Documentation skeleton (all docs/ files)
- [x] CLAUDE.md, WORKFLOW.md, CHANGELOG.md
- [x] .env.example
- [x] .gitignore
- [x] Git initialized

## Phase 1: Foundation, US Market & Sandbox

See [PHASE_1.md](PHASE_1.md) for detailed execution plan.

- [ ] Database schema + Alembic migrations
- [ ] Core framework (mode manager, event bus, clock, logging)
- [ ] Markets framework (registry, segments, instruments)
- [ ] Data ingestion (Finnhub, yfinance)
- [ ] Strategies (momentum, mean reversion, combiner)
- [ ] Backtest engine + simulated broker
- [ ] Risk management (position sizing, stop-loss, pre-trade checks)
- [ ] FastAPI skeleton

## Phase 2: MOEX + Tinkoff, Intelligence & Test Mode

- [ ] Tinkoff gRPC integration
- [ ] MOEX historical data seeding
- [ ] Claude news analysis pipeline (EN + RU)
- [ ] ML pipeline (XGBoost + LightGBM per segment)
- [ ] Event-driven strategy
- [ ] Broker router (Alpaca + Tinkoff)
- [ ] Test mode (paper trading on both markets)
- [ ] Streamlit dashboard

## Phase 3: Hardening & Advanced

- [ ] Per-market circuit breakers
- [ ] Cross-market risk management
- [ ] LSTM per segment
- [ ] Pairs trading
- [ ] Alerting (Slack/Discord/Telegram)
- [ ] Integration + E2E tests

## Phase 4: Real Trading & Optimization

- [ ] Live broker integrations
- [ ] Performance optimization (Polars, caching)
- [ ] Prometheus + Grafana monitoring
- [ ] Walk-forward optimization
- [ ] Production deployment
