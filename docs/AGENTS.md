# Agent System Guide

How AI agents navigate and work with the Finalayze codebase.
This file is the canonical reference for agent dispatch, context loading, and coordination.

## Context Loading Order

When an agent starts a session, it loads context in this order:

```
1. CLAUDE.md                          ← project rules, conventions, current status
2. .claude/agents/<agent-name>.md     ← agent-specific role and domain
3. docs/GLOSSARY.md                   ← domain terminology (if unfamiliar terms)
4. src/finalayze/<module>/CLAUDE.md   ← module-specific context (if working in module)
5. docs/design/<SPEC>.md              ← detailed spec (if needed for design decisions)
6. .planning/STATE.md                 ← current project state (if doing planned work)
```

## Agent Dispatch Rules

### By Task Type

| Task | Primary Agent | Supporting Agents |
|------|--------------|-------------------|
| **Strategy implementation** | `strategies-agent` | `quant-analyst` (review), `backtest-agent` (validate) |
| **Risk rule changes** | `risk-agent` | `risk-officer` (audit), `evaluation-agent` (full eval) |
| **ML model work** | `ml-agent` | `ml-engineer` (review), `data-quality-agent` (validate data) |
| **News pipeline** | `news-pipeline-agent` | `analysis-agent` (LLM prompts), `data-agent` (fetchers) |
| **Broker integration** | `execution-agent` | `systems-architect` (async correctness) |
| **API endpoints** | `api-agent` | `systems-architect` (contract review) |
| **Portfolio allocation** | `portfolio-strategist` | `quant-analyst` (math), `risk-officer` (limits) |
| **Live debugging** | `live-monitor-agent` | domain-specific agent based on issue |
| **Data quality** | `data-quality-agent` | `data-agent` (fix implementation) |
| **Full evaluation** | `evaluation-agent` | all domain experts in parallel |

### By Dependency Layer

```
Layer 0: Types & Schemas       → core-agent
Layer 1: Configuration         → config-agent
Layer 2: Data / Repository     → data-agent, markets-agent, news-pipeline-agent
Layer 3: Analysis / ML         → analysis-agent, ml-agent
Layer 4: Strategy / Risk       → strategies-agent, risk-agent
Layer 5: Execution             → execution-agent
Layer 6: API / Dashboard       → api-agent
Cross-cutting                  → systems-architect, infra-agent
```

## Spec Files Index

Agents should read these specs before making design decisions in the relevant domain:

| Spec | Path | Covers |
|------|------|--------|
| Architecture Overview | `docs/architecture/OVERVIEW.md` | System diagram, tech stack |
| Dependency Layers | `docs/architecture/DEPENDENCY_LAYERS.md` | Import rules (MUST follow) |
| Data Flow | `docs/architecture/DATA_FLOW.md` | Event flow diagrams |
| Strategies | `docs/design/STRATEGIES.md` | All 8 strategies, signal contracts |
| Risk Management | `docs/design/RISK.md` | Sizing, stops, circuit breakers |
| ML Pipeline | `docs/design/ML_PIPELINE.md` | Features, models, training |
| Markets | `docs/design/MARKETS.md` | MOEX vs US, instruments, calendars |
| Segments | `docs/design/SEGMENTS.md` | 9 segments, universes, presets |
| News Pipeline | `docs/design/NEWS_PIPELINE.md` | RSS, Telegram, LLM analysis |
| API Endpoints | `docs/api/ENDPOINTS.md` | REST API contracts |
| Glossary | `docs/GLOSSARY.md` | Domain terminology |

## Module Context Files

Each module has a `CLAUDE.md` with layer rules, public API, contracts, and testing info:

```
src/finalayze/core/CLAUDE.md        ← schemas, exceptions, trading loop
src/finalayze/strategies/CLAUDE.md  ← strategy implementations, presets
src/finalayze/risk/CLAUDE.md        ← sizing, stops, circuit breakers
src/finalayze/ml/CLAUDE.md          ← features, models, training
src/finalayze/analysis/CLAUDE.md    ← LLM, sentiment, events
src/finalayze/data/CLAUDE.md        ← fetchers, normalizers
src/finalayze/execution/CLAUDE.md   ← brokers, order routing
src/finalayze/backtest/CLAUDE.md    ← engine, walk-forward
src/finalayze/markets/CLAUDE.md     ← instruments, calendars
src/finalayze/api/CLAUDE.md         ← REST API, metrics
```

## Coordination Patterns

### Sequential Pipeline (most common)
```
research-agent → planner → plan-checker → executor → verifier
```

### Parallel Domain Review
```
                ┌─→ quant-analyst ──────┐
code change ────┼─→ risk-officer ───────┼─→ merge findings
                └─→ systems-architect ──┘
```

### Live Issue Investigation
```
live-monitor-agent → identifies domain → spawns domain agent → fix → backtest-iteration
```

## Mandatory Gates

These MUST be followed regardless of which agent is working:

1. **Dependency layers** — imports flow downward only (Layer 0→6)
2. **TDD** — write failing test first, then implement
3. **Backtest-iteration** — after ANY strategy/risk/ML change
4. **Ruff + mypy** — code must pass `ruff check` and `mypy src/`
5. **MOEX data** — always use T-Invest API, never yfinance for MOEX tickers
