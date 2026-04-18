# tests/ — Test Suite (Area Node)

Parent: [root AGENTS.md](../AGENTS.md) · Coverage target: 80% on new code

## Layout

| Folder | Scope | When to write tests here |
|---|---|---|
| `unit/` | Single class / function, no I/O, <100ms per test | default: always start here |
| `unit/core/`, `unit/markets/`, `unit/config/` | Module-scoped unit groupings | when a module reaches ~5 test files, group them |
| `integration/` | Multiple modules with real DB / broker stub / network mock | event-bus chains, trading-loop cycles, news→signal pipeline |
| `integration/backtest/` | Full backtest runs with fixtures | walk-forward, portfolio aggregation |
| `e2e/` | Full-stack paper trading cycle, circuit-breaker escalation | rare — only for sandbox regression safety |
| `infra/` | Docker compose + nginx config integrity | CI guardrails for ops configs |

## Conventions

- **TDD is mandatory** — failing test first, then the implementation (see root AGENTS.md).
- Test files mirror the source: `src/finalayze/risk/circuit_breaker.py` ↔ `tests/unit/test_circuit_breaker.py`.
- No magic numbers in asserts — use named constants (ruff `PLR2004`).
- Exceptions caught in tests end with `Error` (ruff `N818`).
- `conftest.py` fixtures are shared — inspect before adding duplicates.
- Async tests: `pytest-asyncio` with `@pytest.mark.asyncio` (mode = `auto` in pyproject).

## Running

| Goal | Command |
|---|---|
| All tests + coverage | `uv run pytest --cov` |
| Single file | `uv run pytest tests/unit/test_adx_routing.py -v` |
| Module slice | `uv run pytest tests/unit/ -k risk -v` |
| Skip integration/e2e (fast loop) | `uv run pytest tests/unit/` |
| With coverage gate | `uv run pytest --cov --cov-fail-under=50` |

## Fixtures & data

- Market data fixtures: `tests/unit/fixtures/` (OHLCV CSVs, news dumps)
- Fixtures must be tiny and deterministic — no network calls in unit tests
- Integration tests may use the Tinkoff sandbox with `FINALAYZE_TINKOFF_TOKEN_SANDBOX`

## When tests touch a given module, also update

| Module edit | Test location |
|---|---|
| `strategies/` | `tests/unit/test_strategies.py`, `test_combiner.py`, `test_adx_routing.py` |
| `risk/` | `tests/unit/test_risk.py`, `test_circuit_breaker.py`, `test_pre_trade_check.py`, `test_position_sizing_pipeline.py` |
| `backtest/` | `tests/unit/test_backtest_engine.py`, `test_backtest_config.py`, `tests/integration/backtest/` |
| `ml/` | `tests/unit/` matched by `-k ml` |
| `execution/` | `tests/unit/test_broker*.py`, `tests/integration/test_alpaca_integration.py`, `test_tinkoff_integration.py` |
| `api/` | `tests/unit/test_api_*.py` |
