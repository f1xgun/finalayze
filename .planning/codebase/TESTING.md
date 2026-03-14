# Testing Patterns

**Analysis Date:** 2026-03-14

## Test Framework

**Runner:** pytest (v8.3.0+)
**Config File:** `pyproject.toml` under `[tool.pytest.ini_options]`
**Async Support:** pytest-asyncio (v0.25.0+) with `asyncio_mode = "auto"`

**Run Commands:**
```bash
uv run pytest                           # Run all tests with coverage
uv run pytest tests/unit/               # Run unit tests only
uv run pytest -k "test_normalizer"      # Run tests matching pattern
uv run pytest --cov=src/finalayze       # Generate coverage report
uv run pytest -v                        # Verbose output
```

**Key pytest settings in pyproject.toml:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = [
    "--strict-markers",
    "--strict-config",
    "-ra",
    "--cov=src/finalayze",
    "--cov-report=term-missing",
    "--cov-fail-under=50",
]
markers = [
    "unit: Unit tests (fast, no I/O)",
    "integration: Integration tests (requires DB/Redis)",
    "e2e: End-to-end tests (full system)",
    "slow: Slow tests (model training, backtests)",
]
```

**Coverage Threshold:** 50% minimum (enforced via `--cov-fail-under=50`)

## Test File Organization

**Location:** Co-located with source in mirror structure
```
src/finalayze/data/normalizer.py         → tests/unit/test_normalizer.py
src/finalayze/execution/tinkoff_broker.py → tests/unit/test_tinkoff_broker.py
src/finalayze/strategies/momentum.py     → tests/unit/test_momentum.py
```

**Naming Convention:**
- Test files: `test_<module_name>.py` (matches source module name)
- Test classes: `Test<Feature>` (e.g., `TestDataNormalizerSingle`, `TestTinkoffBrokerSubmitOrder`)
- Test methods: `test_<behavior_being_tested>` (e.g., `test_normalize_rejects_negative_price`)

**Test Classes:**
Organize related test methods into classes for clarity:
```python
class TestDataNormalizerSingle:
    """Tests for the normalize() method on individual candles."""

    def test_normalize_candle_sets_market_id(self) -> None: ...
    def test_normalize_candle_sets_source(self) -> None: ...
    def test_normalize_rejects_negative_price(self) -> None: ...

class TestDataNormalizerBatch:
    """Tests for the normalize_batch() method."""

    def test_normalize_batch_filters_invalid(self) -> None: ...
    def test_normalize_batch_all_valid(self) -> None: ...
```

**No magic numbers in tests:**
- Declare constants at file top to avoid ruff PLR2004 violations
- Example from `test_normalizer.py`:
```python
MARKET_ID = "us_equity"
SOURCE = "alpaca"
SYMBOL = "AAPL"
TIMEFRAME = "1m"
TIMESTAMP = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

OPEN = Decimal("150.00")
HIGH = Decimal("155.00")
LOW = Decimal("149.00")
CLOSE = Decimal("153.00")
VOLUME = 1000
```

## Test Structure

**Basic Unit Test Pattern:**
```python
"""Unit tests for feature X."""

from __future__ import annotations

import pytest
from finalayze.core.exceptions import CustomError
from finalayze.module.feature import SomeClass

# Constants at top (no magic numbers)
CONSTANT_NAME = "value"
TIMEOUT_SECONDS = 5

def test_basic_behavior() -> None:
    """Test description in imperative mood."""
    # Arrange
    obj = SomeClass()

    # Act
    result = obj.do_something()

    # Assert
    assert result is not None
    assert result == "expected"
```

**Test Method Docstrings:**
- Imperative mood: "Test X must do Y"
- Example from `test_normalizer.py`:
```python
def test_normalize_candle_sets_market_id(self) -> None:
    """normalize() must set the market_id on the returned candle."""
```

**Assertion Style:**
- Use simple `assert` statements (allowed in tests, ruff S101 disabled)
- One logical assertion per test (test one behavior per method)
- Fail early with descriptive messages
- Example:
```python
result = normalizer.normalize(candle)
assert result.market_id == MARKET_ID
assert result.source == SOURCE
```

## Setup & Teardown

**Fixtures (pytest):**
- Defined in `tests/conftest.py` for shared fixtures
- Module-specific fixtures in local test file

**Global conftest.py:**
Located at `tests/conftest.py`:
```python
"""Shared test fixtures for Finalayze."""

from __future__ import annotations

import os

import pytest
from config.modes import WorkMode
from config.settings import Settings

# torch must be imported before lightgbm to prevent OpenMP conflicts
import torch  # noqa: F401

os.environ.setdefault("FINALAYZE_API_KEY", "test-api-key")

@pytest.fixture
def settings() -> Settings:
    """Create test settings with debug mode."""
    return Settings(
        mode=WorkMode.DEBUG,
        database_url="postgresql+asyncpg://test:test@localhost:5432/test",
        redis_url="redis://localhost:6379/1",
    )
```

**Common Patterns:**

1. **Fixture for test object:**
```python
@pytest.fixture
def normalizer() -> DataNormalizer:
    return DataNormalizer(market_id="us_equity", source="alpaca")

def test_normalize_uses_normalizer(normalizer: DataNormalizer) -> None:
    result = normalizer.normalize(some_candle)
    assert result.source == "alpaca"
```

2. **Setup in test method (Arrange-Act-Assert):**
```python
def test_strategy_generates_buy_signal() -> None:
    strategy = MomentumStrategy()
    candles = [_make_candle(...) for _ in range(50)]

    signal = strategy.generate_signal("AAPL", candles, "us_tech")

    assert signal is not None
    assert signal.direction == SignalDirection.BUY
```

## Mocking

**Framework:** unittest.mock (standard library)
- `MagicMock` for creating mock objects
- `patch()` context manager for replacing functions/methods
- `AsyncMock` for async functions

**When to Mock:**
- External I/O (HTTP calls, database, file system)
- Broker integrations (Alpaca, Tinkoff)
- Slow operations (ML model training, backtests)

**When NOT to Mock:**
- Core business logic (strategies, risk checks)
- Pydantic models and validation
- Simple utility functions

**Mock Pattern Examples:**

1. **Simple mock replacement:**
```python
from unittest.mock import MagicMock, patch

def test_buy_order_success() -> None:
    mock_result = MagicMock()
    mock_result.order_id = "ord-123"
    mock_result.executed_order_price.units = 270
    mock_result.executed_order_price.nano = 0
    mock_result.lots_executed = 1

    with patch(
        "finalayze.execution.tinkoff_broker.asyncio.run",
        return_value=mock_result,
    ):
        broker = TinkoffBroker(token="fake", registry=registry)
        result = broker.submit_order(order)

    assert result.filled is True
```

2. **Mock side effect (function capture):**
```python
def test_lot_size_rounding() -> None:
    def capture_run(coro: object) -> MagicMock:
        mock_result = MagicMock()
        mock_result.order_id = "ord-789"
        mock_result.executed_order_price.units = 270
        mock_result.executed_order_price.nano = 0
        mock_result.lots_executed = 1
        return mock_result

    with patch("finalayze.execution.tinkoff_broker.asyncio.run", side_effect=capture_run):
        broker = TinkoffBroker(...)
        result = broker.submit_order(order)

    assert result.quantity == Decimal(10)
```

3. **Async mock:**
```python
from unittest.mock import AsyncMock, patch

@pytest.fixture(autouse=True)
def _mock_health_probes() -> object:
    """Mock out real health probes so tests don't need live DB/Redis."""
    with patch(
        "finalayze.api.v1.system._get_component_status",
        new_callable=AsyncMock,
        return_value=_mock_components,
    ):
        yield
```

4. **Dependency override (FastAPI):**
```python
def test_api_endpoint() -> None:
    app = create_app()
    fresh_manager = ModeManager()
    # Override the dependency with test instance
    app.dependency_overrides[get_mode_manager] = lambda: fresh_manager

    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    response = client.get("/health")

    assert response.status_code == 200
```

## Fixtures and Factories

**Factory Functions:**
Use `_make_*()` naming convention for test data factories:
```python
def _make_candle(
    open_: Decimal = OPEN,
    high: Decimal = HIGH,
    low: Decimal = LOW,
    close: Decimal = CLOSE,
    volume: int = VOLUME,
    market_id: str = "",
) -> Candle:
    """Factory for test candles."""
    return Candle(
        symbol=SYMBOL,
        market_id=market_id,
        timeframe=TIMEFRAME,
        timestamp=TIMESTAMP,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

def _make_stable_candles(price: Decimal, count: int, start_day: int = 0) -> list[Candle]:
    """Create N stable-price candles starting from a given day."""
    return [_candle(price, start_day + i) for i in range(count)]
```

**Test Data Location:**
- Keep factories in test file (not in fixtures)
- Shared fixtures in `tests/conftest.py`
- Domain-specific factories in test module

## Coverage

**Requirements:** Minimum 50% (enforced via pytest config)
- Will increase to 80% in Phase 2

**View Coverage:**
```bash
uv run pytest --cov=src/finalayze --cov-report=html
# Opens htmlcov/index.html
```

**Coverage Report Format:**
- Terminal: `--cov-report=term-missing` (shows missing lines)
- HTML: `--cov-report=html` (browse detailed report)

**What to Cover:**
- Happy paths (normal operation)
- Error conditions (exceptions, validation failures)
- Edge cases (boundary values, empty inputs)
- For risky code: aim for 90%+ coverage

**Not Counted (Acceptable Gaps):**
- Dashboard code (ignored in mypy, not enforced for coverage)
- Third-party integrations that are well-tested by upstream
- Development-only utilities

## Test Types

**Unit Tests:**
- **Scope:** Single function/method in isolation
- **Location:** `tests/unit/test_<module>.py`
- **Speed:** Sub-second execution
- **Dependencies:** Mocked (no I/O, no DB)
- **Examples:**
  - `test_normalizer.py` — validates candle normalization logic
  - `test_position_sizer.py` — tests vol-adjusted sizing calculations
  - `test_mean_reversion.py` — strategy signal generation

**Integration Tests:**
- **Scope:** Multiple components + real DB/Redis
- **Location:** `tests/integration/` (if exist)
- **Speed:** 1-10 seconds per test
- **Setup:** Use real services (or Docker containers)
- **Marker:** `@pytest.mark.integration`
- **Currently:** Minimal integration tests (mostly unit)

**E2E Tests:**
- **Scope:** Full system with real or simulated markets
- **Location:** `tests/e2e/` (if exist)
- **Speed:** 10+ seconds per test
- **Marker:** `@pytest.mark.e2e`
- **Currently:** Not implemented (backtest framework serves this role)

## Async Testing

**Pattern for async code:**
```python
@pytest.mark.asyncio
async def test_async_operation() -> None:
    """Test async function behavior."""
    result = await some_async_function()
    assert result == expected
```

**With pytest-asyncio auto mode:**
- No `@pytest.mark.asyncio` decorator needed
- Define test as `async def test_*` and pytest auto-detects
- Example from `test_api_health.py`:
```python
async def test_get_health_status() -> None:
    """GET /health returns component statuses."""
    app, manager = build_test_app()
    client = make_client(app)

    response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
```

## Error Testing

**Pattern for exception testing:**
```python
def test_error_condition() -> None:
    """Exception is raised when X condition occurs."""
    normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
    candle = _make_candle(open_=NEGATIVE_PRICE)

    with pytest.raises(DataFetchError):
        normalizer.normalize(candle)
```

**With error message matching:**
```python
def test_error_includes_context() -> None:
    """Error message includes all OHLC values."""
    normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
    candle = _make_candle(open_=ZERO_PRICE, ...)

    with pytest.raises(DataFetchError, match="non-positive"):
        normalizer.normalize(candle)

    # Can also inspect the exception:
    with pytest.raises(DataFetchError) as exc_info:
        normalizer.normalize(candle)
    msg = str(exc_info.value)
    assert "open=" in msg
    assert "high=" in msg
```

## Pre-commit Hooks

**Installed:** Yes, via `.pre-commit-config.yaml` (if present)

**Manual checks before commit:**
```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy src/

# Run tests
uv run pytest --cov=src/finalayze
```

## Common Test Patterns

**Testing signal generation (strategies):**
```python
def test_strategy_generates_buy_signal() -> None:
    strategy = MomentumStrategy()
    # Create candles with sufficient history
    candles = [_candle(BASE_PRICE, i) for i in range(STABLE_COUNT)]
    # Add final "now" candle
    candles.append(_candle(BASE_PRICE * Decimal(0.9), STABLE_COUNT))

    signal = strategy.generate_signal("AAPL", candles, "us_tech")

    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.confidence >= 0.0
```

**Testing with insufficient data:**
```python
def test_no_signal_with_insufficient_candles() -> None:
    strategy = MomentumStrategy()
    # Less than required candles
    candles = [_candle(BASE_PRICE, i) for i in range(MIN_CANDLES_INSUFFICIENT)]

    signal = strategy.generate_signal("AAPL", candles, "us_tech")

    assert signal is None
```

**Testing data validation:**
```python
def test_normalize_rejects_invalid_data() -> None:
    normalizer = DataNormalizer(market_id="us", source="test")
    candle = _make_candle(low=HIGH, high=LOW)  # inverted

    with pytest.raises(DataFetchError):
        normalizer.normalize(candle)
```

## Ruff Test Configuration

**Test-specific ignores (from pyproject.toml):**
```toml
"tests/**" = [
    "S101",      # assert is allowed in tests
    "T20",       # print statements OK in test output
    "ARG",       # unused arguments OK in test fixtures
    "PLC0415",   # import at module level not required
    "N806",      # lowercase function names OK (factory functions)
    "TC001",     # TYPE_CHECKING blocks not required
    "RUF100",    # unused noqa comments are OK
    "PT018",     # pytest assertion patterns
    "S108",      # hardcoded temp/log dirs OK in tests
    "F401",      # unused imports OK (test utilities)
    "SLF001",    # private member access OK in tests
    "PLR2004",   # magic numbers replaced with constants (not enforced)
    "TC003",     # TYPE_CHECKING guards not required
]
```

## Test Statistics

**Current State (as of 2026-03-14):**
- Total test files: ~230
- Coverage threshold: 50% (enforced)
- Test categories: Mostly unit tests (mock-heavy)
- Framework: pytest + pytest-asyncio + pytest-mock + pytest-cov

---

*Testing analysis: 2026-03-14*
