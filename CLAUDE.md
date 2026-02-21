# Finalayze - Agent Entry Point

Finalayze is an AI-powered multi-market stock trading system. It ingests news,
social sentiment, and market data; analyzes them with LLMs and ML ensembles;
and executes trades on US (Alpaca) and MOEX (Tinkoff Invest) markets.

## Superpowers Workflow (mandatory)

Follow this sequence for ALL work. Skills trigger automatically -- invoke them.

1. **Brainstorm** before any feature/change (design first, code never)
2. **Worktree** -- isolate work in a git worktree
3. **Write plan** -- save to `docs/plans/YYYY-MM-DD-<name>.md`
4. **Execute plan** via subagent-driven-development or executing-plans
5. **TDD** -- RED-GREEN-REFACTOR for all implementations
6. **Verify** before claiming completion (run tests, read output)
7. **Finish branch** -- merge, PR, or keep

## Documentation Map

| Document | Purpose |
|---|---|
| [docs/INDEX.md](docs/INDEX.md) | Master index of all documentation |
| [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md) | System architecture |
| [docs/architecture/DEPENDENCY_LAYERS.md](docs/architecture/DEPENDENCY_LAYERS.md) | Import layering rules |
| [docs/architecture/DATA_FLOW.md](docs/architecture/DATA_FLOW.md) | Event flow diagrams |
| [docs/design/](docs/design/) | MARKETS, SEGMENTS, STRATEGIES, RISK, NEWS, ML |
| [docs/api/ENDPOINTS.md](docs/api/ENDPOINTS.md) | API contract reference |
| [docs/quality/GRADES.md](docs/quality/GRADES.md) | Quality grades per domain |
| [docs/quality/GAPS.md](docs/quality/GAPS.md) | Tech debt tracker |
| [docs/plans/ROADMAP.md](docs/plans/ROADMAP.md) | Phase overview with status |
| [docs/plans/PHASE_1.md](docs/plans/PHASE_1.md) | Phase 1 execution plan |
| [WORKFLOW.md](WORKFLOW.md) | Development process conventions |

## Dependency Layering Rules

Imports must flow **downward only**. Never import upward.

```
Layer 0: Types & Schemas       core/schemas.py, core/exceptions.py
Layer 1: Configuration          config/settings.py, config/modes.py, config/segments.py
Layer 2: Data / Repository      data/, markets/
Layer 3: Analysis / ML          analysis/, ml/
Layer 4: Strategy / Risk        strategies/, risk/
Layer 5: Execution              execution/
Layer 6: API / Dashboard        api/, dashboard/
```

## Coding Conventions

- Python 3.12, strict typing, `from __future__ import annotations`
- Formatter/linter: ruff (line-length 100), type checker: mypy (strict)
- Package manager: uv, lockfile committed
- Async-first: SQLAlchemy 2.0 async, httpx for HTTP
- Pydantic v2 for all schemas and settings
- TDD mandatory: write failing test FIRST, then implement
- 4 work modes: debug, sandbox, test, real

## Current Phase

**Phase 1 -- Foundation, US Market & Sandbox** (not started).
See [docs/plans/PHASE_1.md](docs/plans/PHASE_1.md).

## Quick Commands

```bash
uv sync                    # install dependencies
uv run pytest              # run tests
uv run ruff check .        # lint
uv run mypy src/           # type-check
```
