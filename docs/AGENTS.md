# docs/ — Documentation & Agent Dispatch (Area Node)

Parent: [root AGENTS.md](../AGENTS.md)

This node serves two jobs at once:
1. **Docs map** — pointers to all design specs, architecture, quality reports.
2. **Agent dispatch** — which sub-agent to invoke for which kind of task.

## Agent dispatch

### Domain experts (audit / design review)

Use these for high-level analysis and pre-merge gates, not for implementation.

| Invoke when… | Agent |
|---|---|
| Reviewing strategy math, signal quality, or backtest methodology | `quant-analyst` |
| Auditing risk thresholds, circuit breakers, pre-trade checks | `risk-officer` |
| Reviewing ML pipeline, feature engineering, model calibration | `ml-engineer` |
| Checking layer violations, async correctness, data-flow integrity | `systems-architect` |
| Portfolio allocation, cross-asset correlation, capital distribution | `portfolio-strategist` |

### Module agents (implementers)

Dispatch one per module touched. They own the files under that path.

| Module path | Agent |
|---|---|
| `src/finalayze/core/` | `core-agent` |
| `config/` | `config-agent` |
| `src/finalayze/data/` | `data-agent` |
| `src/finalayze/markets/` | `markets-agent` |
| `src/finalayze/analysis/` | `analysis-agent` |
| `src/finalayze/ml/` | `ml-agent` |
| `src/finalayze/strategies/` | `strategies-agent` |
| `src/finalayze/risk/` | `risk-agent` |
| `src/finalayze/execution/` | `execution-agent` |
| `src/finalayze/backtest/` | `backtest-agent` |
| `src/finalayze/api/`, `src/finalayze/dashboard/` | `api-agent` |
| News pipeline (cross-module: `data/` + `analysis/`) | `news-pipeline-agent` |
| `docker/`, `alembic/`, `pyproject.toml`, CI workflows | `infra-agent` |

### Operations / specialised agents

| Use case | Agent |
|---|---|
| Investigate live / sandbox execution issues, trade-log forensics | `live-monitor-agent` |
| Full strategy evaluation with backtest journaling + grade | `evaluation-agent` |
| Orchestrate a debate: collect outputs → ConflictDetector → arbiter | `agent-orchestrator` |
| Fact-check conflicting agent claims against codebase + history | `arbiter-agent` |

Multi-module task? Dispatch module agents **sequentially** (they edit overlapping files).
Independent audits? Dispatch domain experts **in parallel**.

## Spec index

Read these before making design decisions in the corresponding domain.

| Spec | Path | Covers |
|---|---|---|
| Architecture overview | [`architecture/OVERVIEW.md`](architecture/OVERVIEW.md) | System diagram, tech stack |
| Dependency layers | [`architecture/DEPENDENCY_LAYERS.md`](architecture/DEPENDENCY_LAYERS.md) | Import rules (MUST follow) |
| Data flow | [`architecture/DATA_FLOW.md`](architecture/DATA_FLOW.md) | Event flow diagrams |
| ADRs | [`architecture/DECISIONS.md`](architecture/DECISIONS.md) | Historical architecture decisions |
| Strategies | [`design/STRATEGIES.md`](design/STRATEGIES.md) | All strategies + ADX routing + combiner |
| Risk | [`design/RISK.md`](design/RISK.md) | Sizing, stops, circuit breakers |
| ML pipeline | [`design/ML_PIPELINE.md`](design/ML_PIPELINE.md) | Features, models, training |
| Markets | [`design/MARKETS.md`](design/MARKETS.md) | US vs MOEX, instruments, calendars |
| Segments | [`design/SEGMENTS.md`](design/SEGMENTS.md) | Segment system |
| News pipeline | [`design/NEWS_PIPELINE.md`](design/NEWS_PIPELINE.md) | RSS, Telegram, LLM analysis |
| Broker contracts | [`design/BROKER_CONTRACTS.md`](design/BROKER_CONTRACTS.md) | Broker integration specs |
| Database schema | [`database/SCHEMA.md`](database/SCHEMA.md) | Tables, migrations |
| REST API | [`api/ENDPOINTS.md`](api/ENDPOINTS.md) | Endpoint reference |
| Glossary | [`GLOSSARY.md`](GLOSSARY.md) | Domain terminology |

## Quality & operations

| Doc | Purpose |
|---|---|
| [`quality/GRADES.md`](quality/GRADES.md) | Per-domain quality grades |
| [`quality/GAPS.md`](quality/GAPS.md) | Tech-debt tracker |
| [`quality/TEST_STRATEGY.md`](quality/TEST_STRATEGY.md) | Testing approach per module |
| [`operations/DEPLOYMENT.md`](operations/DEPLOYMENT.md) | Production Docker deployment |
| [`operations/MONITORING.md`](operations/MONITORING.md) | Prometheus + Alertmanager |
| [`operations/RUNBOOK.md`](operations/RUNBOOK.md) | Incident runbook |
| [`operations/GO_LIVE_CHECKLIST.md`](operations/GO_LIVE_CHECKLIST.md) | Pre-production validation |

## Plans, evaluations, research

- `plans/ROADMAP.md` — active roadmap + phase status
- `plans/<YYYY-MM-DD>-*.md` — individual phase / design / improvement plans
- `evaluations/` — backtest evaluation reports
- `research/` — deep research notes (ML, quant, strategy)
- `reviews/` — expert audit reports from domain-expert agents

## Coordination patterns (when to parallelise vs serialise)

```
Sequential (default):
  research-agent → planner → plan-checker → executor → verifier

Parallel domain review:
                ┌─→ quant-analyst ──────┐
  code change ──┼─→ risk-officer ───────┼─→ arbiter → merge findings
                └─→ systems-architect ──┘

Live issue investigation:
  live-monitor-agent → identifies domain → spawns module agent → fix → backtest-iteration
```

## Mandatory gates (re-asserted from root)

1. Dependency layers — downward only
2. TDD — failing test first
3. `backtest-iteration` after any strategy / risk / backtest / ML change
4. `ruff check` + `mypy src/` green
5. MOEX data — Tinkoff gRPC only
