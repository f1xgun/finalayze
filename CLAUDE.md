# Finalayze - Agent Entry Point

Finalayze is an AI-powered multi-market stock trading system. It ingests news,
social sentiment, and market data; analyzes them with LLMs and ML ensembles;
and executes trades on US (Alpaca) and MOEX (Tinkoff Invest) markets.

## Table of Contents

| Document | Purpose |
|---|---|
| [docs/INDEX.md](docs/INDEX.md) | Master index of all documentation |
| [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md) | High-level system architecture |
| [docs/architecture/DEPENDENCY_LAYERS.md](docs/architecture/DEPENDENCY_LAYERS.md) | Import layering rules |
| [docs/architecture/DATA_FLOW.md](docs/architecture/DATA_FLOW.md) | Event flow diagrams |
| [docs/architecture/DECISIONS.md](docs/architecture/DECISIONS.md) | Architecture Decision Records |
| [docs/design/MARKETS.md](docs/design/MARKETS.md) | Multi-market design (US, MOEX) |
| [docs/design/SEGMENTS.md](docs/design/SEGMENTS.md) | Stock segment system |
| [docs/design/STRATEGIES.md](docs/design/STRATEGIES.md) | Trading strategy design |
| [docs/design/RISK.md](docs/design/RISK.md) | Risk management design |
| [docs/design/NEWS_PIPELINE.md](docs/design/NEWS_PIPELINE.md) | News ingestion pipeline |
| [docs/design/ML_PIPELINE.md](docs/design/ML_PIPELINE.md) | ML ensemble pipeline |
| [docs/api/ENDPOINTS.md](docs/api/ENDPOINTS.md) | API contract reference |
| [docs/quality/GRADES.md](docs/quality/GRADES.md) | Quality grades per domain |
| [docs/quality/GAPS.md](docs/quality/GAPS.md) | Tech debt tracker |
| [docs/quality/TEST_STRATEGY.md](docs/quality/TEST_STRATEGY.md) | Testing approach per module |
| [docs/operations/RUNBOOK.md](docs/operations/RUNBOOK.md) | Operational runbook |
| [docs/operations/DEPLOYMENT.md](docs/operations/DEPLOYMENT.md) | Deployment guide |
| [docs/operations/MONITORING.md](docs/operations/MONITORING.md) | Monitoring & alerting |
| [docs/plans/ROADMAP.md](docs/plans/ROADMAP.md) | Phase overview with status |
| [docs/plans/PHASE_1.md](docs/plans/PHASE_1.md) | Phase 1 execution plan |
| [WORKFLOW.md](WORKFLOW.md) | Development process conventions |
| [CHANGELOG.md](CHANGELOG.md) | Project changelog |

## Dependency Layering Rules

Imports must flow **downward only**. A module may import from its own layer
or any layer with a smaller number. Never import upward.

```
Layer 0: Types & Schemas       core/schemas.py, core/exceptions.py
Layer 1: Configuration          config/settings.py, config/modes.py, config/segments.py
Layer 2: Data / Repository      data/, markets/
Layer 3: Analysis / ML          analysis/, ml/
Layer 4: Strategy / Risk        strategies/, risk/
Layer 5: Execution              execution/
Layer 6: API / Dashboard        api/, dashboard/
```

## Active Coding Conventions

- Python 3.12, strict typing, `from __future__ import annotations`
- Formatter/linter: ruff (line-length 99), type checker: mypy (strict)
- Package manager: uv, lockfile committed
- Async-first: SQLAlchemy 2.0 async, httpx for HTTP
- Pydantic v2 for all schemas and settings
- Test files mirror source: `tests/<module>/test_<name>.py`
- Docstrings: Google style, required on all public functions
- 4 work modes: debug, sandbox, test, real

## Current Phase

**Phase 1 -- Foundation, US Market & Sandbox** (not started).
See [docs/plans/PHASE_1.md](docs/plans/PHASE_1.md) for the detailed plan.

## Quick Commands

```bash
uv sync                    # install dependencies
uv run pytest              # run tests
uv run ruff check .        # lint
uv run mypy src/           # type-check
```
