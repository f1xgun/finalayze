# Finalayze - Agent Entry Point

Finalayze is an AI-powered multi-market stock trading system. It ingests news,
social sentiment, and market data; analyzes them with LLMs and ML ensembles;
and executes trades on US (Alpaca) and MOEX (Tinkoff Invest) markets.

## Development Workflow (mandatory)

GSD is the primary orchestration framework. Trading skills are domain hooks within it.

### Planned work (phases/milestones)

```
/gsd:new-project          # first time only — creates .planning/, ROADMAP, etc.
/gsd:discuss-phase N      # brainstorm design decisions for phase N
/gsd:plan-phase N         # research → plan → verify loop
/gsd:execute-phase N      # wave-based parallel execution with TDD
/gsd:verify-work N        # UAT / manual verification
/gsd:progress             # check status, route to next action
```

### Quick tasks (ad-hoc, no milestone ceremony)

```
/gsd:quick "description"           # plan + execute in one shot
/gsd:quick --discuss "description" # brainstorm gray areas first
/gsd:quick --full "description"    # adds plan-checking + verification
```

### Debugging

```
/gsd:debug "description"  # systematic debugging with checkpoints
```

### Session management

```
/gsd:pause-work           # save context for later
/gsd:resume-work          # restore context from previous session
```

### Trading-specific gates (mandatory)

After ANY change to strategies/risk/backtest/ML, run `backtest-iteration` skill
before completing the phase or quick task. Use other trading skills as needed
(see Trading Skills table below).

## Trading Skills (`.claude/skills/`)

| Skill | When to Use |
|---|---|
| `backtest-iteration` | After strategy/risk/backtest/ML changes -- run & compare metrics |
| `strategy-diagnose` | Debug why a strategy underperforms or fires rarely |
| `iteration-history` | Review metrics trajectory across iterations |
| `data-quality-check` | Validate market data integrity before backtests |
| `ml-experiment` | Train ML models, evaluate impact, gate enablement |
| `preset-tuner` | Structured parameter tuning with sensitivity checks |

External: `quantitative-research` (quant methodology), `risk-metrics-calculation` (risk math).
See [WORKFLOW.md §9](WORKFLOW.md) for full dispatch rules.

## Documentation Map

### Agent Context (read first)
| Document | Purpose |
|---|---|
| [docs/AGENTS.md](docs/AGENTS.md) | Agent dispatch rules, context loading order, coordination patterns |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Domain terminology reference |
| `src/finalayze/*/CLAUDE.md` | Module-level context (layer, public API, contracts) |

### Architecture & Design Specs
| Document | Purpose |
|---|---|
| [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md) | System architecture |
| [docs/architecture/DEPENDENCY_LAYERS.md](docs/architecture/DEPENDENCY_LAYERS.md) | Import layering rules |
| [docs/architecture/DATA_FLOW.md](docs/architecture/DATA_FLOW.md) | Event flow diagrams |
| [docs/design/](docs/design/) | STRATEGIES, RISK, ML_PIPELINE, MARKETS, SEGMENTS, NEWS_PIPELINE |
| [docs/design/BROKER_CONTRACTS.md](docs/design/BROKER_CONTRACTS.md) | Broker integration specs |
| [docs/database/SCHEMA.md](docs/database/SCHEMA.md) | Database tables, migrations, conventions |
| [docs/api/ENDPOINTS.md](docs/api/ENDPOINTS.md) | REST API contract reference |

### Quality & Operations
| Document | Purpose |
|---|---|
| [docs/quality/GRADES.md](docs/quality/GRADES.md) | Quality grades per domain |
| [docs/quality/GAPS.md](docs/quality/GAPS.md) | Tech debt tracker |
| [docs/operations/GO_LIVE_CHECKLIST.md](docs/operations/GO_LIVE_CHECKLIST.md) | Pre-production validation |

### Process
| Document | Purpose |
|---|---|
| [WORKFLOW.md](WORKFLOW.md) | Development process conventions |
| [.claude/agents/](/.claude/agents/) | 36 sub-agent definitions |
| [.planning/](/.planning/) | GSD state, milestones, retrospective |

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

## Agent System

18 Claude Code sub-agents in `.claude/agents/`. See §8 in [WORKFLOW.md](WORKFLOW.md) for dispatch rules.

**Domain experts (audit + design review):** `quant-analyst`, `risk-officer`, `ml-engineer`, `systems-architect`, `portfolio-strategist`

**Module agents (implementers):** `core-agent`, `config-agent`, `data-agent`, `markets-agent`, `analysis-agent`, `ml-agent`, `strategies-agent`, `risk-agent`, `execution-agent`, `backtest-agent`, `api-agent`, `infra-agent`, `news-pipeline-agent`

**Operations agents:** `live-monitor-agent`, `data-quality-agent`, `evaluation-agent`

**Specialized agents:** `evaluation-agent`, `data-quality-agent`

## Current Status (2026-03-08)

**Sprint state:** Week 5 — ML deep overhaul merged. 2325 tests.

### Domain Health

| Domain | Grade | Key Issue |
|---|---|---|
| **Strategies** | B- | 8 strategies (5 enabled). ADX regime routing separates trend/MR pools. Win rate 42-54%, PF 1.22. `event_driven`, `ml_ensemble`, `pead` disabled. |
| **Data** | B | US (yfinance) works. MOEX requires `FINALAYZE_TINKOFF_TOKEN`. Dividend pipeline wired (TinkoffFetcher + static YAML fallback). |
| **Risk** | B | Pipeline floor (15% of base). Strategy-specific ATR stops. Currency-aware sizing. Half-Kelly + 11-check pre-trade pipeline. |
| **ML** | C- | 16 new features (cross-asset, regime, calendar, z-scores). Training pipeline exists. Models accuracy suboptimal (~57% best fold). `ml_ensemble` disabled. |
| **Backtest** | B+ | Engine works. Grace bar. Walk-forward months (12mo train + 6mo test). Strategy-specific max hold bars. Optuna overfitting guardrails. |
| **Execution** | B+ | Alpaca + Tinkoff brokers wired. RetryPolicy with backoff. Simulated broker for backtests. |
| **Analysis** | D | LLM client + NewsAnalyzer exist but `event_driven` disabled (no real-time news feed). |
| **API/Dashboard** | B+ | 20+ REST endpoints, Prometheus metrics, Streamlit dashboard. All operational. |

### Recent Changes (Weeks 3-5)

**Week 3 — Structural fixes:**
- ADX(14) regime routing, DRY combiner hooks, strategy-specific ATR stops
- Pipeline floor (15%), grace bar, exit confidence 0.38, dual_momentum SELL
- Results: Win rate 15%→50%, PF 1.05→1.22, Max DD 0.46%→0.25%

**Week 4 — Optimization:**
- Optuna overfitting guardrails (DSR haircut, holdout validation, perturbation check)
- Market-neutral labels via benchmark alignment

**Week 5 — ML deep overhaul:**
- 16 new features: cross-asset correlations, regime indicators, calendar effects, z-scores
- Feature selection pipeline, calibrator gating, quality gates
- Phase 1+3 fixes: Brier validation, feature importance budget

### Isolated Strategy Performance (us_tech, 2022-2025)

| Strategy | Sharpe | PF | Trades | Status |
|---|---|---|---|---|
| dual_momentum | +0.137 | 1.29 | 414 | Enabled |
| mean_reversion | +0.034 | 1.98 | 27 | Enabled |
| rsi2_connors | +0.020 | 0.94 | 73 | Enabled |
| momentum | -0.014 | 1.46 | 27 | Enabled (reduced weight) |
| ou_mean_reversion | -0.038 | 0.91 | 67 | Enabled (us_tech) |

### MOEX Data Requirement

All MOEX data (candles, dividends, instruments) **must** use T-Bank (Tinkoff Invest) gRPC API.
yfinance cannot fetch MOEX tickers. Set `FINALAYZE_TINKOFF_TOKEN` env var.

## AST Index (ast-index)

`ast-index` is installed for fast codebase navigation. Use it FIRST for symbol/class lookups instead of grep:

```bash
ast-index class StrategyCombiner     # find class definition
ast-index symbol generate_signal     # find symbol across codebase
ast-index search "EnsembleModel"     # universal search (files + symbols)
ast-index outline src/finalayze/strategies/combiner.py  # show file structure
ast-index hierarchy BaseStrategy     # class hierarchy
ast-index usages StrategyCombiner    # find all usages
ast-index deps src/finalayze/strategies/  # module dependencies
ast-index map                        # compact project map
ast-index rebuild                    # rebuild index after code changes
```

## Quick Commands

```bash
uv sync                    # install dependencies
uv run pytest              # run tests
uv run ruff check .        # lint
uv run mypy src/           # type-check
uv run python scripts/run_iteration.py --name <name> --description <desc> --segments us_tech,us_broad
uv run python scripts/run_strategy_isolation.py --segment us_tech --all
```
