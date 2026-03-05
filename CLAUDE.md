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
| [.claude/agents/](/.claude/agents/) | 16 sub-agent definitions (4 domain experts + 12 module agents) |

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

16 Claude Code sub-agents in `.claude/agents/`. See §8 in [WORKFLOW.md](WORKFLOW.md) for dispatch rules.

**Domain experts (audit + design review):** `quant-analyst`, `risk-officer`, `ml-engineer`, `systems-architect`

**Module agents (implementers):** `core-agent`, `config-agent`, `data-agent`, `markets-agent`, `analysis-agent`, `ml-agent`, `strategies-agent`, `risk-agent`, `execution-agent`, `backtest-agent`, `api-agent`, `infra-agent`

## Current Status (2026-03-05)

**Sprint state:** Week 2 structural fixes complete. 1995 tests passing.

### Domain Health

| Domain | Grade | Key Issue |
|---|---|---|
| **Strategies** | B- | ADX regime routing separates trend/MR pools. Win rate 42-54%, PF 1.22. `event_driven`, `ml_ensemble`, `pead` disabled. Trade count still low (626). |
| **Data** | B | US (yfinance) works. MOEX requires `FINALAYZE_TINKOFF_TOKEN`; yfinance .ME tickers return 0 data. Dividend pipeline wired (TinkoffFetcher + static YAML fallback). |
| **Risk** | B | Pipeline floor (15% of base) prevents cascade. Strategy-specific ATR stops. Currency-aware sizing (RUB 5000 / USD 500). Half-Kelly + 11-check pre-trade pipeline. |
| **ML** | D | Models untrained, `ml_ensemble` disabled in all presets. Feature engineering + training pipeline exist but unused. |
| **Backtest** | B+ | Engine works. Grace bar prevents same-candle stop-outs. Walk-forward uses months (12mo train + 6mo test). Strategy-specific max hold bars. |
| **Execution** | B+ | Alpaca + Tinkoff brokers wired. RetryPolicy with backoff. Simulated broker for backtests. |
| **Analysis** | D | LLM client + NewsAnalyzer exist but `event_driven` disabled (no real-time news feed). |
| **API/Dashboard** | B+ | 20+ REST endpoints, Prometheus metrics, Streamlit dashboard. All operational. |

### Week 3 Structural Fixes

**Completed:**
- **ADX(14) regime routing**: trend pool (ADX>30, momentum/dual_momentum), MR pool (ADX<20, mean_reversion/rsi2/OU/pairs), ambiguous zone (20-30, dominant-pool-wins)
- **DRY JournalingStrategyCombiner**: 4-hook architecture, no more code duplication
- **Strategy-specific ATR stops**: momentum=2.5, dual_momentum=3.0, mean_reversion=3.5, rsi2=2.5, MOEX 1.2x uplift
- **Pipeline floor**: 15% of base_position prevents cascading reduction
- **Walk-forward months**: 12mo train + 6mo test + 6mo step (was 3yr+1yr years which produced 0 OOS windows)
- **YAML params wired**: DualMomentum and OU now read YAML configs
- **Grace bar**: Skip stop-loss check on fill candle (prevents same-candle stop-outs)
- **Exit confidence 0.38**: Matches entry threshold, prevents weak SELL signal churn
- **dual_momentum SELL**: Score <= -0.05 triggers SELL signal with deduplication

**Results**: Win rate 15%→50%, PF 1.05→1.22, Max DD 0.46%→0.25%. Trade count dropped 1198→626 (fewer noise trades). WF Sharpe still negative (-0.004).

**Remaining bottleneck**: Trade count (626 vs 1300 target). Individual strategies fire rarely; ADX routing + 0.30 threshold filter most signals. Need more signal sources or lower-latency signals.

### Isolated Strategy Performance (us_tech, 2022-2025)

| Strategy | Sharpe | PF | Trades | Status |
|---|---|---|---|---|
| dual_momentum | +0.137 | 1.29 | 414 | Enabled (us_tech), SELL at -0.05 |
| mean_reversion | +0.034 | 1.98 | 27 | Enabled |
| rsi2_connors | +0.020 | 0.94 | 73 | Enabled |
| momentum | -0.014 | 1.46 | 27 | Enabled (reduced weight) |
| ou_mean_reversion | -0.038 | 0.91 | 67 | Disabled (us_tech) |

### MOEX Data Requirement

All MOEX data (candles, dividends, instruments) **must** use T-Bank (Tinkoff Invest) gRPC API.
yfinance cannot fetch MOEX tickers. Set `FINALAYZE_TINKOFF_TOKEN` env var.

## Quick Commands

```bash
uv sync                    # install dependencies
uv run pytest              # run tests
uv run ruff check .        # lint
uv run mypy src/           # type-check
uv run python scripts/run_iteration.py --name <name> --description <desc> --segments us_tech,us_broad
uv run python scripts/run_strategy_isolation.py --segment us_tech --all
```
