# Finalayze — Graph Root

> Canonical agent entry point. Claude Code auto-loads [CLAUDE.md](CLAUDE.md), which forwards here.
> Other tools (Codex, Cursor, Devin, GPT-Engineer) read this file directly.

## What this project is

AI-powered multi-market trading system. Ingests news + market data, analyses with LLMs and
ML ensembles, executes on US (Alpaca) and MOEX (Tinkoff Invest). Python 3.12, async, strict typing.

## How to navigate this graph

This repository uses a layered **AGENTS.md graph**: root → area → module. Every node answers
_"what lives here, what are the contracts, where do I go next"_ — you should rarely need to
read more than two nodes to locate the right file.

- **Machine-readable index:** [`.agents/manifest.jsonl`](.agents/manifest.jsonl) — one line per
  node with `path`, `kind`, `layer`, `parent`, `children`, `depends_on`, `keywords`. Parse
  this first when you need to pick a node without reading prose.
- **Schema:** [`.agents/MANIFEST.md`](.agents/MANIFEST.md)

## Area map

| Area | Purpose | Entry point |
|---|---|---|
| Source packages | All production Python code, layered 0–6 | [`src/AGENTS.md`](src/AGENTS.md) |
| Configuration | Settings, modes, segments, universes, gate thresholds | [`config/AGENTS.md`](config/AGENTS.md) |
| Tests | Unit / integration / e2e pyramid | [`tests/AGENTS.md`](tests/AGENTS.md) |
| Scripts | CLI entry points: iterations, training, evaluation | [`scripts/AGENTS.md`](scripts/AGENTS.md) |
| Documentation | Design specs, architecture, quality, ops, agent dispatch | [`docs/AGENTS.md`](docs/AGENTS.md) |

## Non-negotiable invariants (apply everywhere)

1. **Dependency layers flow downward only.** See [`docs/architecture/DEPENDENCY_LAYERS.md`](docs/architecture/DEPENDENCY_LAYERS.md).
   ```
   0 schemas → 1 config → 2 data/markets → 3 analysis/ml → 4 strategies/risk → 5 execution → 6 api/dashboard/monitoring
   ```
2. **TDD is mandatory.** Write the failing test first, then the implementation.
3. **MOEX data = Tinkoff Invest gRPC only.** `yfinance` cannot fetch MOEX tickers. Env var: `FINALAYZE_TINKOFF_TOKEN`.
4. **Backtest gate.** Any change to `strategies/`, `risk/`, `backtest/`, `ml/` triggers the
   `backtest-iteration` skill (`.claude/skills/backtest-iteration.md`) before the task is complete.
5. **Lint + type-check.** `uv run ruff check .` and `uv run mypy src/` must be green.

## Workflow

| Mode | Command | Use for |
|---|---|---|
| Planned work (phases) | `/gsd:discuss-phase N` → `/gsd:plan-phase N` → `/gsd:execute-phase N` | milestones, multi-file features |
| Quick tasks | `/gsd:quick "description"` (+ `--discuss`, `--full`) | ad-hoc fixes |
| Debugging | `/gsd:debug "description"` | systematic root-cause investigation |
| Session handoff | `/gsd:pause-work`, `/gsd:resume-work` | cross-session state |

Details: [`WORKFLOW.md`](WORKFLOW.md) (process conventions, PR flow, quality gates).

## Runtime state (refreshed separately from this graph)

| Artefact | Holds |
|---|---|
| `.planning/` | GSD state: ROADMAP, STATE, milestones, retrospective |
| `results/iterations/history.jsonl` | Backtest metrics trajectory |
| `models/<segment>/` | Trained ML models, keyed by segment |

## Search tooling

Use `ast-index` first for symbol/class lookups — it is 17-69× faster than grep and understands
Python AST. See `.claude/rules/ast-index.md` for the decision tree.
