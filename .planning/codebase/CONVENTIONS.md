# Coding Conventions

**Analysis Date:** 2026-03-14

## Language & Core Requirements

**Python Version:** 3.12 (enforced via `pyproject.toml requires-python`)
**Type Checking:** mypy strict mode (all files)
**Async Framework:** asyncio-first (SQLAlchemy 2.0 async, httpx for HTTP)

All source files MUST include:
```python
from __future__ import annotations
```

## Naming Patterns

**Files:**
- Module files: `snake_case.py` (e.g., `momentum_strategy.py`, `position_sizer.py`)
- Test files: `test_<module>.py` or `test_<feature>.py` (e.g., `test_normalizer.py`, `test_tinkoff_broker.py`)
- Init files: Empty or re-export public APIs (no business logic in `__init__.py`)

**Exceptions:**
- Exception names MUST end in `Error` (enforced by ruff N818)
- Examples: `DataFetchError`, `BrokerError`, `InstrumentNotFoundError`, `ConfigurationError`
- All exceptions inherit from `FinalayzeError` base class (see `src/finalayze/core/exceptions.py`)

**Classes:**
- `PascalCase` for all class names
- Abstract base classes: `BaseStrategy`, `BrokerBase` (prefix with `Base`)
- Private/internal classes: prefix with `_` (e.g., `_SignalState`, `_Indicators`)

**Functions & Methods:**
- `snake_case` for all functions and methods
- Private methods: prefix with `_` (e.g., `_validate()`, `_compute_signal()`)
- Factory functions: `_make_*()` (e.g., `_make_candle()`, `_make_registry()`)

**Variables & Constants:**
- `snake_case` for all variables
- Module-level constants: `UPPER_CASE` (e.g., `_PRESETS_DIR`, `_MIN_CANDLES`, `DEFAULT_LOOKBACK_BARS`)
- Private constants: prefix with `_` (e.g., `_ZERO = Decimal(0)`, `_MIN_EXIT_CONFIDENCE = 0.38`)
- Enums: `StrEnum` not `str, Enum` (enforced by ruff UP042)

**Type Aliases:**
```python
type InstrumentType = Literal["stock", "etf", "bond"]
```
Use PEP 613 syntax in Python 3.12.

## Code Style

**Formatter/Linter:**
- Tool: **ruff** (version >= 0.9.0)
- Line length: **100 characters** (enforced via `tool.ruff.line-length`)
- Formatter command: `uv run ruff format .`
- Linter command: `uv run ruff check .`

**Ruff Rules (selected):**
```toml
select = ["E", "W", "F", "I", "N", "UP", "B", "A", "C4", "DTZ", "T20", "SIM",
          "TCH", "RUF", "S", "PT", "RET", "ARG", "PL", "PERF", "FURB", "LOG", "TID"]
ignore = ["S101", "PLR0913"]  # assert in tests OK; trading functions need many params
```

**Per-file exceptions in ruff:**
- Tests: `"tests/**" = ["S101", "T20", "ARG", "PLC0415", "N806", "TC001", "RUF100", "PT018", "S108", "F401", "SLF001", "PLR2004", "TC003"]`
  - S101: assert is allowed in tests
  - PLR2004: magic numbers allowed (use constants instead, but not enforced with RUF100 in test context)
  - SLF001: private member access OK in tests
- Scripts: `"scripts/**" = ["T20", "E402", "S112", "TC001"]`
- Dashboard: `"src/finalayze/dashboard/**" = ["TC001", "PLR0912", "PLR0915"]`

**No magic numbers in test assertions:**
- Declare constants at top of test file (per ruff PLR2004 enforcement)
- Example from `test_normalizer.py`:
```python
BATCH_SIZE_3 = 3
BATCH_SIZE_2 = 2
OPEN = Decimal("150.00")
HIGH = Decimal("155.00")
```

## Import Organization

**Import Order:**
1. `from __future__ import annotations` (first line, always)
2. Standard library (`datetime`, `os`, `typing`, etc.)
3. Third-party (`pydantic`, `sqlalchemy`, `pandas`, etc.)
4. Relative/local imports (`from finalayze.*`)

**Example:**
```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import Candle

if TYPE_CHECKING:
    from pathlib import Path
```

**Path Aliases:**
- `from finalayze.*` imports are allowed (no relative imports like `from ..core`)
- Imports are organized first-party as `finalayze` (ruff config: `known-first-party = ["finalayze"]`)

**TYPE_CHECKING Pattern:**
Use `TYPE_CHECKING` to avoid circular imports and unnecessary imports at runtime:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path  # Only imported for type checking
```

## Error Handling

**Exception Hierarchy:**
- All custom exceptions inherit from `FinalayzeError` (see `src/finalayze/core/exceptions.py`)
- Organize exceptions by domain (Configuration, Market, DataFetching, Strategy, Risk, Execution, etc.)
- Example structure:
```python
class DataFetchError(FinalayzeError):
    """Data fetching from an external provider failed."""

class RateLimitError(DataFetchError):
    """External API rate limit was hit."""
```

**Raising Exceptions:**
- Always include descriptive messages with context
- Use f-strings for message formatting
- Example from `data/normalizer.py`:
```python
msg = (
    f"Candle has non-positive price: open={candle.open}, high={candle.high},"
    f" low={candle.low}, close={candle.close}"
)
raise DataFetchError(msg)
```

**Catching & Handling:**
- Catch specific exception types, not bare `Exception`
- Log at warning level when handling gracefully (skip invalid data)
- Re-raise or wrap with context for higher-level failures
- Example from `data/normalizer.py`:
```python
try:
    result.append(self.normalize(candle))
except DataFetchError as exc:
    logger.warning(
        "Skipping invalid candle %s@%s: %s",
        candle.symbol,
        candle.timestamp,
        exc,
    )
```

## Logging

**Framework:** structlog (v24.4.0+)

**Logger Declaration:**
- At module level, AFTER imports:
```python
logger = structlog.get_logger()  # or _log = structlog.get_logger()
```
- Use `logger` or `_log` convention (be consistent per module)
- Requires `setup_logging(mode)` called at module level BEFORE any logger is used (see `config/logging.py`)

**Logging Levels:**
- `logger.debug()`: detailed diagnostic info (DEBUG mode only)
- `logger.info()`: significant events (state changes, batch completions)
- `logger.warning()`: recoverable errors (skipped candles, retries)
- `logger.error()`: unrecoverable errors that should be escalated

**Message Format:**
- Use structured key-value pairs (not f-strings in log calls)
- Example from `risk/drawdown_monitor.py`:
```python
logger.warning(
    "Circuit breaker triggered",
    current_dd=current_dd,
    dd_threshold=self._threshold,
    portfolio_equity=self._last_equity,
)
```

**Configuration:**
- JSON output (production-ready)
- ISO 8601 timestamps
- Set via `config/logging.py` → `structlog.configure()`
- Log level: DEBUG in `WorkMode.DEBUG`, INFO otherwise

## Comments

**When to Comment:**
- Explain *why*, not *what* (code should be clear what it does)
- Business logic rationale (e.g., "grace bar skips stop-loss on fill candle")
- Non-obvious mathematical formulas or domain-specific rules
- Warn about edge cases or common pitfalls

**DocStrings:**
- Use module docstrings (single-line) for file purpose and layer:
```python
"""Market data normalizer — validates and tags candles (Layer 2)."""
```
- Use class docstrings to explain intent:
```python
class DataNormalizer:
    """Validates OHLCV candles and tags them with market_id and source."""
```
- Use method docstrings for complex logic:
```python
def normalize(self, candle: Candle) -> Candle:
    """Validate and tag a single candle. Raises DataFetchError if invalid."""
```

**No JSDoc/TSDoc:** Python uses reST or Google-style docstrings (not enforced, but prefer clarity)

## Function & Method Design

**Size Guideline:** Keep functions under 50 lines; break complex logic into helpers
- Example: `DataNormalizer.normalize()` is 3 lines (delegates to `_validate()` helper)

**Parameters:**
- Positional parameters for required, simple data
- Keyword-only for optional or configurable behavior
- Example from `position_sizer.py`:
```python
def compute_vol_adjusted_position_size(
    base_position: Decimal,
    target_vol: Decimal,
    asset_vol: Decimal,
) -> Decimal:
```

**Return Values:**
- Use `-> Type | None` for optional returns (not `Optional[Type]`)
- Single responsibility: return one logical thing
- Example: `generate_signal()` returns `Signal | None` (one signal or none)

**Private Methods:**
- Use `_` prefix for internal helpers
- Keep private methods focused on single tasks
- Example: `_validate()`, `_compute_indicators()`, `_get_signal_state()`

## Module Design

**Exports:**
- Avoid `from module import *` (explicit imports only)
- Each module exports a small, focused set of public classes/functions
- Example from `strategies/base.py`: exports only `BaseStrategy` abstract class

**Barrel Files:**
- Use `__init__.py` to re-export public APIs:
```python
from finalayze.core.exceptions import DataFetchError

__all__ = ["DataFetchError"]
```
- Keep init files small; avoid business logic

**Layering:**
- Respect dependency layer rules (see `docs/architecture/DEPENDENCY_LAYERS.md`)
- Imports must flow downward only:
  - Layer 0: Types & Schemas (core/schemas.py, core/exceptions.py)
  - Layer 1: Configuration (config/)
  - Layer 2: Data / Repository (data/, markets/)
  - Layer 3: Analysis / ML (analysis/, ml/)
  - Layer 4: Strategy / Risk (strategies/, risk/)
  - Layer 5: Execution (execution/)
  - Layer 6: API / Dashboard (api/, dashboard/)
- Example: `data/normalizer.py` imports from `core/exceptions` (OK), never imports from `execution/` (FORBIDDEN)

## Pydantic Models

**Version:** Pydantic v2 (mandatory)

**Configuration:**
- Use `model_config` with `ConfigDict`:
```python
class Candle(BaseModel):
    model_config = ConfigDict(frozen=True)
```
- Common settings: `frozen=True` for immutable (value objects), `extra="forbid"` for strict validation

**Validators:**
- Use `@field_validator` with `mode="after"` for post-parse validation
- Example from `core/schemas.py`:
```python
@field_validator("confidence")
@classmethod
def confidence_must_be_probability(cls, v: float) -> float:
    if not (0.0 <= v <= 1.0):
        msg = f"confidence must be in [0.0, 1.0], got {v}"
        raise ValueError(msg)
    return v
```

**Type Hints:**
- Decimal for monetary values (not float)
- float for probabilities, percentages, ratios [0.0, 1.0]
- Example from `core/schemas.py`:
```python
class Signal(BaseModel):
    confidence: float  # probability, not Decimal
    entry_price: Decimal  # monetary value, must be Decimal
```

## Dataclasses

**When to Use:**
- For simple value holders (not Pydantic models)
- Add `frozen=True` and `slots=True` for efficiency
- Example from `strategies/momentum.py`:
```python
@dataclass(frozen=True, slots=True)
class _Indicators:
    current_rsi: float
    rsi_window: list[float]
```

## Database & ORM

**SQLAlchemy 2.0:** Async required (`from sqlalchemy.ext.asyncio import AsyncSession`)
- All DB operations async
- Use context managers for sessions

**No direct SQL:** Use ORM methods, not raw SQL strings (prevent injection)

## Async Patterns

**Async-First Rule:**
- Web: use `async def` on all FastAPI routes
- I/O: use `async with httpx.AsyncClient()` for HTTP calls
- DB: use `AsyncSession` from SQLAlchemy 2.0
- Brokers: use T-Bank `AsyncClient` (context manager required)

**Example from execution/tinkoff_broker.py:**
```python
async with AsyncClient(token=self._token, target=self._target) as client:
    services = await client.get_me()  # await all async calls
```

## Testing Patterns

- See TESTING.md for comprehensive testing conventions

---

*Convention analysis: 2026-03-14*
