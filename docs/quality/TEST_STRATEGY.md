# Testing Strategy

## Test Pyramid

```
        /  E2E  \          Few, slow, full system
       /----------\
      / Integration \      Cross-module, requires DB/Redis
     /----------------\
    /    Unit Tests     \  Many, fast, isolated
   /____________________\
```

## Coverage Targets

| Module | Min Coverage | Test Types |
|--------|-------------|------------|
| `core/` | 90% | Unit |
| `config/` | 90% | Unit |
| `strategies/` | 90% | Unit + Integration |
| `risk/` | 95% | Unit + Integration |
| `execution/` | 85% | Unit + Integration (mock brokers) |
| `ml/` | 80% | Unit + Integration |
| `data/` | 80% | Unit + Integration |
| `analysis/` | 80% | Unit (mock LLM responses) |
| `markets/` | 90% | Unit |
| `backtest/` | 80% | Unit + Integration |
| `dashboard/` | 60% | Unit |
| `api/` | 80% | Unit + Integration |

## Test Conventions

- **File naming:** `tests/unit/test_{module}.py` mirrors `src/finalayze/{module}.py`
- **Integration tests:** `tests/integration/test_{flow}.py`
- **E2E tests:** `tests/e2e/test_{scenario}.py`
- **Markers:** `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.slow`
- **Fixtures:** Shared fixtures in `tests/conftest.py`
- **Async tests:** Use `pytest-asyncio` with `asyncio_mode = "auto"`

## Mocking Strategy

| External System | Mock Approach |
|----------------|---------------|
| Alpaca API | `pytest-mock` with response fixtures |
| Tinkoff API | `pytest-mock` with gRPC response fixtures |
| Claude LLM | Recorded response fixtures (JSON files) |
| PostgreSQL | Test database with rollback per test |
| Redis | Fakeredis or test Redis instance |
| External news APIs | Response fixtures (JSON files) |

## Running Tests

```bash
uv run pytest                           # All tests
uv run pytest -m unit                   # Unit tests only
uv run pytest -m integration            # Integration tests only
uv run pytest --cov --cov-report=html   # With HTML coverage report
```
