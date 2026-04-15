# Phase 52: Portfolio Review Agent - Research

**Researched:** 2026-04-15
**Domain:** Advisory LLM agent, scheduled portfolio reporting, safety-by-schema design
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- `PortfolioReviewResult` Pydantic schema with position summaries, concentration warnings, catalyst list
- Schema MUST NOT have `direction`, `confidence`, or `symbol`+`market_id` combination that matches `Signal` or `OrderRequest`
- Type-checker assertion at handler entry prevents trade-like fields from being added
- Code-grep verification: `BrokerRouter`, `place_order`, `generate_signal` must return zero results inside handler
- Scheduled via APScheduler at 19:00 MSK daily (after MOEX close at 18:40 MSK)
- LLM receives: current positions, daily P&L, sector/ticker concentration, upcoming events/catalysts
- Output: structured Telegram message with clear sections (not free-form prose)
- Handler writes ONLY to `TelegramAlerter` — no other output path

### Claude's Discretion

- Exact Telegram message format and section layout
- LLM prompt design and context selection
- Whether to create a separate `PortfolioReviewAgent` class or use a scheduled function
- How to gather portfolio state (direct broker query vs cached positions)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PFRA-01 | Daily LLM portfolio review runs outside market hours with structured PortfolioReviewResult output | APScheduler cron job at 16:00 UTC (19:00 MSK); `LLMClient.parse_structured()` with `PortfolioReviewResult` response model |
| PFRA-02 | Review results delivered via Telegram with concentration risk and upcoming catalyst analysis | `TelegramAlerter.send_alert()` already fire-and-forget; format structured sections from `PortfolioReviewResult` fields |
| PFRA-03 | Advisory-only enforcement — schema has no trade-directive fields, no write access to order pipeline | `PortfolioReviewResult` fields must differ from `Signal` (no `direction`/`confidence`) and `OrderRequest` (no `side`); code-grep verification |
</phase_requirements>

---

## Summary

Phase 52 adds a daily advisory LLM portfolio review that runs at 19:00 MSK (16:00 UTC) via APScheduler, gathers open positions from broker(s), calls `LLMClient.parse_structured()` to produce a `PortfolioReviewResult`, and delivers a structured Telegram message. The safety contract — no write path to the order pipeline — is enforced at three levels: (1) schema design (no `direction`, `confidence`, or `symbol`+`market_id` pair), (2) a runtime type assertion at handler entry, and (3) a code-grep test that verifies zero references to `BrokerRouter`, `place_order`, or `generate_signal` inside the agent handler.

The implementation closely parallels Phase 51's anomaly interpreter: fire-and-forget async coroutine dispatched to `self._async_loop` via `run_coroutine_threadsafe`, graceful degradation on LLM failure, and `TelegramAlerter.send_alert()` as the only output path. The key difference is that Phase 52 is time-triggered (cron at 16:00 UTC daily) rather than event-triggered, and it calls `parse_structured()` to validate the LLM response against a Pydantic schema rather than accepting free-form text.

Portfolio data is gathered by calling `broker.get_portfolio()` and `broker.get_positions()` on each registered broker via `BrokerRouter.route(market_id)`. These broker calls are already used in `_daily_reset()` and `_get_cached_portfolio()`, so the access pattern is well-established.

**Primary recommendation:** Add a `_portfolio_review_cycle()` method to `TradingLoop` registered as an APScheduler cron job at `hour=16, minute=0` (UTC). Dispatch an async `_run_portfolio_review_async()` coroutine to `self._async_loop` via `run_coroutine_threadsafe` without `.result()` (fire-and-forget, same as Phase 51). Place `PortfolioReviewResult` in a new `analysis/portfolio_review_agent.py` file at Layer 3.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| apscheduler | >=3.10.4 | Cron job at 16:00 UTC | Already the scheduler for news/strategy/daily-reset cycles [VERIFIED: pyproject.toml] |
| asyncio (stdlib) | Python 3.12 | Fire-and-forget dispatch from sync APScheduler thread | `run_coroutine_threadsafe` + persistent `_async_loop` is the established pattern [VERIFIED: core/trading_loop.py] |
| pydantic v2 | >=2.10.0 | `PortfolioReviewResult` schema + `model_validate_json` | All schemas use Pydantic v2 frozen models [VERIFIED: core/schemas.py] |
| LLMClient (internal) | — | `parse_structured()` for type-safe LLM output | Already used by `NewsAnalyzer`; retries, caching, 3-provider support [VERIFIED: analysis/llm_client.py] |
| TelegramAlerter (internal) | — | `send_alert()` for delivery; `._send()` from inside async coroutine | Fire-and-forget, error-swallowing, the only approved output path [VERIFIED: core/alerts.py] |
| structlog | >=24.4.0 | Structured logging for failures and lifecycle events | Project-wide standard [VERIFIED: pyproject.toml] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| statistics (stdlib) | Python 3.12 | Concentration ratio calculations | Sufficient for position-count arithmetic; numpy available if array ops needed |
| decimal (stdlib) | Python 3.12 | Money arithmetic in concentration calculations | Required for all monetary values per project conventions |

**Installation:** No new packages needed. All dependencies already in `pyproject.toml`.

**Version verification:** APScheduler 3.10.4 current [VERIFIED: pyproject.toml]. Pydantic 2.10.0 current [VERIFIED: pyproject.toml].

---

## Architecture Patterns

### Recommended Project Structure Addition
```
src/finalayze/
├── analysis/
│   ├── portfolio_review_agent.py   # NEW — Layer 3: PortfolioReviewResult + PortfolioReviewAgent
│   └── ...existing...
├── core/
│   └── trading_loop.py             # MODIFIED — add _portfolio_review_cycle(), _run_portfolio_review_async()
```

### Pattern 1: APScheduler Cron at 16:00 UTC (= 19:00 MSK)

**What:** Add a cron job to `TradingLoop.start()` using the same `BackgroundScheduler` instance.

**When to use:** Time-triggered daily jobs outside market hours.

**19:00 MSK = 16:00 UTC** (MSK is UTC+3 year-round — Russia does not observe DST). [ASSUMED — standard UTC+3 offset for MSK; no DST transitions in Russia since 2014]

```python
# Source: [VERIFIED: core/trading_loop.py start() — _daily_reset cron pattern]

self._scheduler.add_job(
    self._portfolio_review_cycle,
    "cron",
    hour=16,
    minute=0,
    timezone="UTC",  # explicit for clarity; scheduler is already UTC
)
```

### Pattern 2: Fire-and-Forget Dispatch from Sync APScheduler Thread (Established)

**What:** Sync APScheduler thread dispatches an async coroutine to the persistent `_async_loop` without blocking.

**When to use:** Any async work triggered from an APScheduler job.

```python
# Source: [VERIFIED: core/trading_loop.py _process_instrument() anomaly enrichment]

def _portfolio_review_cycle(self) -> None:
    """APScheduler callback — dispatches async review without blocking."""
    if self._llm_client is None:
        _log.info("portfolio_review_skipped: no LLM client configured")
        return
    if self._async_loop is None or self._async_loop.is_closed():
        return
    asyncio.run_coroutine_threadsafe(
        self._run_portfolio_review_async(),
        self._async_loop,
    )
    # No .result() — fire-and-forget
```

### Pattern 3: `PortfolioReviewResult` Schema — Safety by Design

**What:** Pydantic schema whose field set is provably disjoint from `Signal` and `OrderRequest`.

**Safety proof:**
- `Signal` has `direction: SignalDirection`, `confidence: float`, `symbol: str`, `market_id: str`
- `OrderRequest` has `symbol: str`, `side: Literal["BUY", "SELL"]`, `quantity: Decimal`
- `PortfolioReviewResult` must have NONE of: `direction`, `confidence`, `side`, and must NOT have both `symbol` + `market_id` together

```python
# Source: [ASSUMED — consistent with existing Pydantic patterns in core/schemas.py and analysis/anomaly_detector.py]

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field


class PositionSummary(BaseModel):
    """Advisory summary of one open position."""
    model_config = ConfigDict(frozen=True)

    ticker: str                    # NOT "symbol" — avoids Signal field name
    market: str                    # NOT "market_id" — avoids Signal field name
    quantity: Decimal
    unrealized_pnl: Decimal
    pct_of_portfolio: float        # 0.0–1.0


class ConcentrationWarning(BaseModel):
    """A concentration risk flag."""
    model_config = ConfigDict(frozen=True)

    ticker: str                    # advisory field only
    market: str
    concentration_pct: float       # e.g. 0.25 = 25% of portfolio
    warning_level: str             # "HIGH" | "MEDIUM" — NOT a trade direction


class CatalystEvent(BaseModel):
    """Upcoming event that may affect a position."""
    model_config = ConfigDict(frozen=True)

    ticker: str
    event_type: str                # "earnings" | "cbr_meeting" | "dividend" | "other"
    expected_date: str             # ISO date string — plain str to avoid datetime complexity


class PortfolioReviewResult(BaseModel):
    """Advisory-only portfolio analysis from LLM.

    SAFETY INVARIANT: This schema has no field named 'direction', 'confidence',
    or 'side', and does not have a (symbol, market_id) pair — making it
    structurally incompatible with Signal and OrderRequest.
    """
    model_config = ConfigDict(frozen=True)

    reviewed_at: datetime
    positions: list[PositionSummary] = Field(default_factory=list)
    concentration_warnings: list[ConcentrationWarning] = Field(default_factory=list)
    catalyst_events: list[CatalystEvent] = Field(default_factory=list)
    overall_assessment: str        # brief narrative — NOT a trade recommendation
    risk_score: float              # 0.0–1.0 advisory risk level, NOT confidence
```

**Type-checker assertion at handler entry:**
```python
# Source: [ASSUMED — standard Python type assertion pattern]

from finalayze.analysis.portfolio_review_agent import PortfolioReviewResult
from finalayze.core.schemas import Signal
from finalayze.execution.broker_base import OrderRequest

# Static assertion: PortfolioReviewResult shares no trade-directive fields
assert not hasattr(PortfolioReviewResult, "direction")
assert not hasattr(PortfolioReviewResult, "confidence")
assert not hasattr(PortfolioReviewResult, "side")
# Can be placed as module-level assertions in portfolio_review_agent.py
```

### Pattern 4: Portfolio Data Gathering

**What:** Collect positions from all registered brokers to build the LLM prompt.

**Access path:** `BrokerRouter.route(market_id)` → `broker.get_portfolio()` and `broker.get_positions()` — same path as `_daily_reset()` and `_get_cached_portfolio()`.

```python
# Source: [VERIFIED: core/trading_loop.py _daily_reset() and _get_cached_portfolio()]

async def _run_portfolio_review_async(self) -> None:
    """Fire-and-forget: gather portfolio state, call LLM, send Telegram."""
    try:
        portfolio_data = self._gather_portfolio_data()  # sync — broker calls are sync
        assert self._llm_client is not None
        result: PortfolioReviewResult = await asyncio.wait_for(
            self._llm_client.parse_structured(
                prompt=_build_review_prompt(portfolio_data),
                system=_PORTFOLIO_REVIEW_SYSTEM_PROMPT,
                response_model=PortfolioReviewResult,
            ),
            timeout=_REVIEW_LLM_TIMEOUT,
        )
        message = _format_telegram_message(result)
        await self._alerter._send(message)  # direct async — already in coroutine
    except Exception:
        _log.warning("portfolio_review_llm_failure")
```

### Pattern 5: Structured Telegram Message Format (Discretion Area)

**What:** Format `PortfolioReviewResult` into a Telegram-friendly sectioned message.

**Recommendation:** Use emoji section headers for visual clarity in the Telegram UI, with newline-separated sections:

```
Portfolio Review — Mon 14 Apr 19:00 MSK

Positions (3 open)
• SBER [moex] x100 | PnL: +2,340 RUB | 22% of portfolio
• GAZP [moex] x50  | PnL: -850 RUB  | 18% of portfolio

Concentration Risk
⚠ SBER: 22% — HIGH

Upcoming Catalysts
• SBER — CBR meeting (2026-04-25)

Assessment: Portfolio is moderately concentrated in energy+finance.
Risk: MEDIUM (0.52)
```

### Anti-Patterns to Avoid

- **Calling `_run_async()` for the review coroutine:** `_run_async()` blocks with `.result(timeout=30)`. The portfolio review may take longer than 30s with a slow LLM. Use fire-and-forget (`run_coroutine_threadsafe` without `.result()`).
- **Using `send_alert()` inside the async coroutine:** Nested task creation. Call `await alerter._send(text)` directly.
- **Putting `PortfolioReviewResult` in `core/schemas.py`:** It is analysis-specific, not needed by Layer 0–2. Place in `analysis/portfolio_review_agent.py` (Layer 3) to keep `core/schemas.py` clean.
- **Accessing `_broker_router` without guarding for missing market:** Use `try/except` around each `BrokerRouter.route()` call — same pattern as `_daily_reset()`.
- **Hard-coding 19:00 MSK as the cron time without comment:** APScheduler uses UTC. Cron `hour=16` is 19:00 MSK. Document this explicitly in the code.
- **Adding `direction`, `confidence`, or both `symbol`+`market_id` to `PortfolioReviewResult`:** Violates PFRA-03 and the schema safety contract. Use `ticker`/`market` as field names instead.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Async timeout | Custom cancellation | `asyncio.wait_for(coro, timeout=60.0)` | stdlib; already used in Phase 51 anomaly enrichment |
| Telegram HTTP | Custom POST | `TelegramAlerter._send(text)` | Already error-swallowing, tested, handles token-empty no-op |
| LLM structured output | JSON parsing + error handling | `LLMClient.parse_structured(prompt, system, PortfolioReviewResult)` | Already retries, caches, handles provider differences |
| Task dispatch from sync thread | `threading.Thread` | `asyncio.run_coroutine_threadsafe(coro, self._async_loop)` | The persistent `_async_loop` is the right target; established pattern |
| Position aggregation math | Custom loop | `broker.get_portfolio()` + `broker.get_positions()` | BrokerBase abstract methods already on every broker |

**Key insight:** The safety guarantee "no write path to order pipeline" is an architectural property, not a runtime check. It is enforced by: (1) schema fields that cannot be mistaken for Signal/OrderRequest, (2) the handler only holding a reference to `TelegramAlerter` (not `BrokerRouter` or `StrategyCombiner`), and (3) a code-grep test.

---

## Common Pitfalls

### Pitfall 1: UTC vs MSK timezone confusion in the cron job
**What goes wrong:** Developer sets `hour=19` (MSK wall clock) but APScheduler runs in UTC, so the job fires at 19:00 UTC = 22:00 MSK — two hours after MOEX market close.
**Why it happens:** `BackgroundScheduler(timezone="UTC")` is explicit in `TradingLoop.start()`.
**How to avoid:** Set `hour=16, minute=0` (UTC). 19:00 MSK = 16:00 UTC (MSK is UTC+3, no DST). Document with a comment: `# 19:00 MSK = 16:00 UTC`.
**Warning signs:** If the Telegram review message arrives at 22:xx MSK, the cron hour is wrong.

### Pitfall 2: `parse_structured()` timeout is too short for portfolio analysis
**What goes wrong:** Using `_ANOMALY_LLM_TIMEOUT = 30.0` copied from Phase 51 is too tight for a portfolio review prompt, which may be longer and require more tokens.
**Why it happens:** Anomaly prompts are short (4 lines); portfolio prompts include position lists and may be 50+ lines.
**How to avoid:** Use a separate `_REVIEW_LLM_TIMEOUT = 60.0` constant. The fire-and-forget pattern means this does not block any trading cycle.
**Warning signs:** Repeated `portfolio_review_llm_failure` log events with `TimeoutError` in the underlying exception.

### Pitfall 3: Schema field naming accidentally mirrors Signal/OrderRequest
**What goes wrong:** Developer uses `symbol` and `market_id` in `PortfolioReviewResult.positions` sub-schema, creating a type that shares the identifying field combination of `Signal`.
**Why it happens:** `symbol` and `market_id` are natural names for position data.
**How to avoid:** Use `ticker` and `market` in all nested schemas. This is semantically identical but structurally different from `Signal.symbol`/`Signal.market_id`. The success criterion explicitly tests for field-set disjointness.
**Warning signs:** If `PositionSummary` has both a `symbol` and `market_id` field, the schema safety assertion fails.

### Pitfall 4: LLM returns invalid JSON for `PortfolioReviewResult`
**What goes wrong:** `LLMClient.parse_structured()` calls `model_validate_json(raw)` which raises `pydantic.ValidationError` if the LLM produces malformed JSON or omits required fields.
**Why it happens:** The portfolio review is a complex structured output request; free-form models may not reliably produce valid JSON with all required fields.
**How to avoid:** (1) Use an explicit JSON schema in the system prompt. (2) Catch `pydantic.ValidationError` alongside `Exception` in the outer `try/except` of `_run_portfolio_review_async()`. (3) Make most fields have `default_factory=list` defaults so partial JSON still validates.
**Warning signs:** `pydantic_core.ValidationError` in `portfolio_review_llm_failure` logs.

### Pitfall 5: `BrokerRouter.route()` raises `BrokerError` for unknown market
**What goes wrong:** If a market is registered in `circuit_breakers` but no broker is registered with `BrokerRouter`, `route()` raises `BrokerError`.
**Why it happens:** Test setups or partially-configured deployments may have circuit breaker keys without broker registrations.
**How to avoid:** Wrap each `broker_router.route(market_id)` call in `try/except Exception` and log a warning — same defensive pattern used in `_daily_reset()`.

### Pitfall 6: Code-grep test fails due to indirect reference
**What goes wrong:** Developer imports `BrokerRouter` for a type annotation (e.g., `TYPE_CHECKING` block or docstring), causing the code-grep test to report a false positive.
**Why it happens:** The test greps for the string `BrokerRouter` in the handler file, not the import block.
**How to avoid:** The grep test should target the handler's function/method body, not the entire file. Alternatively, ensure `TYPE_CHECKING` imports are in a separate block clearly outside the handler. In practice, the handler simply shouldn't need `BrokerRouter` at all — it should use `self._broker_router` only at data-gathering time, not in the agent module itself.

---

## Code Examples

### APScheduler Cron Registration (Verified Pattern)
```python
# Source: [VERIFIED: core/trading_loop.py start() — _daily_reset cron job]

# In TradingLoop.start():
self._scheduler.add_job(
    self._portfolio_review_cycle,
    "cron",
    hour=16,   # 19:00 MSK = 16:00 UTC (MSK is UTC+3, no DST)
    minute=0,
)
```

### Fire-and-Forget from Sync APScheduler Thread (Verified Pattern)
```python
# Source: [VERIFIED: core/trading_loop.py _process_instrument() — anomaly enrichment dispatch]

def _portfolio_review_cycle(self) -> None:
    if self._llm_client is None:
        return
    if self._async_loop is None or self._async_loop.is_closed():
        return
    asyncio.run_coroutine_threadsafe(
        self._run_portfolio_review_async(),
        self._async_loop,
    )  # no .result() — fire-and-forget
```

### Async Review Coroutine with Graceful Degradation (Based on Phase 51 Pattern)
```python
# Source: [VERIFIED pattern: core/trading_loop.py _enrich_anomaly_async()]

_REVIEW_LLM_TIMEOUT = 60.0
_PORTFOLIO_REVIEW_SYSTEM_PROMPT = (
    "You are a portfolio risk analyst. Analyze the given portfolio positions "
    "and return a structured JSON review. Do not give trade directives. "
    "Focus on concentration risk, upcoming catalysts, and overall risk assessment."
)

async def _run_portfolio_review_async(self) -> None:
    _log = structlog.get_logger()
    try:
        portfolio_data = self._gather_portfolio_data()
        assert self._llm_client is not None
        result: PortfolioReviewResult = await asyncio.wait_for(
            self._llm_client.parse_structured(
                prompt=_build_review_prompt(portfolio_data),
                system=_PORTFOLIO_REVIEW_SYSTEM_PROMPT,
                response_model=PortfolioReviewResult,
            ),
            timeout=_REVIEW_LLM_TIMEOUT,
        )
        message = _format_review_telegram(result)
        await self._alerter._send(message)
    except Exception:
        _log.warning("portfolio_review_llm_failure")
```

### Portfolio Data Gathering (Verified Data Access Path)
```python
# Source: [VERIFIED: core/trading_loop.py _daily_reset() — same broker access pattern]

def _gather_portfolio_data(self) -> dict[str, object]:
    """Synchronous — broker calls are sync. Called from within async coroutine."""
    markets_data = {}
    for market_id in self._circuit_breakers:
        try:
            broker = self._broker_router.route(market_id)
            portfolio = broker.get_portfolio()
            positions = broker.get_positions()
            markets_data[market_id] = {
                "equity": portfolio.equity,
                "cash": portfolio.cash,
                "positions": positions,
            }
        except Exception:
            _log.warning("portfolio_review_gather_failed", market_id=market_id)
    return markets_data
```

### Schema Safety Code-Grep Test Pattern
```python
# Source: [ASSUMED — based on success criterion in CONTEXT.md]

import subprocess

def test_no_order_pipeline_references_in_handler() -> None:
    """Code-grep: verify handler has zero write-path references."""
    result = subprocess.run(
        ["grep", "-rn", r"BrokerRouter\|place_order\|generate_signal",
         "src/finalayze/analysis/portfolio_review_agent.py"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0 or result.stdout == "", (
        f"Found forbidden references in portfolio review agent:\n{result.stdout}"
    )
```

### `PortfolioReviewResult` Schema Safety Assertion
```python
# Source: [ASSUMED — module-level assertion pattern]

# In analysis/portfolio_review_agent.py — module-level, verified at import time
_FORBIDDEN_FIELDS = {"direction", "confidence", "side"}
_review_fields = set(PortfolioReviewResult.model_fields)
assert not (_review_fields & _FORBIDDEN_FIELDS), (
    f"PortfolioReviewResult has forbidden trade-directive fields: "
    f"{_review_fields & _FORBIDDEN_FIELDS}"
)
```

---

## Dependency Layer Placement

| Component | Layer | Module | Allowed Imports |
|-----------|-------|--------|-----------------|
| `PortfolioReviewResult`, `PositionSummary`, `ConcentrationWarning`, `CatalystEvent` | Layer 3 | `analysis/portfolio_review_agent.py` | L0–L2 imports ok |
| `PortfolioReviewAgent` class or helper functions | Layer 3 | `analysis/portfolio_review_agent.py` | L0–L2 imports ok |
| `_portfolio_review_cycle()`, `_run_portfolio_review_async()`, `_gather_portfolio_data()` | Layer 6 (TradingLoop methods) | `core/trading_loop.py` | Already at L6; imports L3 ok |
| Scheduling registration | Layer 6 (TradingLoop.start()) | `core/trading_loop.py` | No new imports needed |

[VERIFIED: docs/architecture/DEPENDENCY_LAYERS.md]

**Key layering constraint:** `analysis/portfolio_review_agent.py` (Layer 3) must NOT import from `execution/` (Layer 5) or `core/trading_loop.py` (Layer 6). All broker data must be passed in as plain Python dicts or PortfolioState objects — not broker references.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Free-form LLM text → string formatting | `parse_structured()` with Pydantic response model | Phase 49 (NewsAnalyzer migration) | Reliable structured output, ValidationError catches malformed responses |
| Blocking `_run_async()` for LLM calls | `run_coroutine_threadsafe` without `.result()` | Phase 51 (anomaly enrichment) | Fire-and-forget, doesn't block APScheduler thread |

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.3+ with pytest-asyncio 0.25+ |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| asyncio_mode | `auto` (all async tests work without `@pytest.mark.asyncio`) |
| Quick run command | `uv run pytest tests/unit/test_portfolio_review_agent.py -x` |
| Full suite command | `uv run pytest --cov=src/finalayze --cov-fail-under=50` |

[VERIFIED: pyproject.toml `asyncio_mode = "auto"` and pytest-asyncio >=0.25.0]

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PFRA-01 | `PortfolioReviewResult` returned by LLM via `parse_structured()` | unit | `uv run pytest tests/unit/test_portfolio_review_agent.py::TestSchemaValidation -x` | ❌ Wave 0 |
| PFRA-01 | `_portfolio_review_cycle()` dispatches to async loop (cron wiring) | unit | `uv run pytest tests/unit/test_portfolio_review_agent.py::TestCronDispatch -x` | ❌ Wave 0 |
| PFRA-02 | Telegram message delivered with concentration + catalyst sections | unit | `uv run pytest tests/unit/test_portfolio_review_agent.py::TestTelegramFormat -x` | ❌ Wave 0 |
| PFRA-03 | `PortfolioReviewResult` has no `direction`, `confidence`, `side` fields | unit | `uv run pytest tests/unit/test_portfolio_review_agent.py::TestSchemaAdvisoryOnly -x` | ❌ Wave 0 |
| PFRA-03 | Code-grep: zero `BrokerRouter`/`place_order`/`generate_signal` in handler | unit | `uv run pytest tests/unit/test_portfolio_review_agent.py::TestNoOrderPipelineAccess -x` | ❌ Wave 0 |
| PFRA-03 | LLM failure: no Telegram message, `portfolio_review_llm_failure` logged | unit | `uv run pytest tests/unit/test_portfolio_review_agent.py::TestGracefulDegradation -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_portfolio_review_agent.py -x`
- **Per wave merge:** `uv run pytest --cov=src/finalayze --cov-fail-under=50`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_portfolio_review_agent.py` — covers PFRA-01, PFRA-02, PFRA-03
- [ ] `src/finalayze/analysis/portfolio_review_agent.py` — PortfolioReviewResult schema + agent helpers

*(No new conftest.py needed — existing `tests/conftest.py` with Settings fixture is sufficient)*

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a |
| V3 Session Management | no | n/a |
| V4 Access Control | yes | Advisory-only schema: no write path to order pipeline by design; code-grep test enforces |
| V5 Input Validation | yes | `PortfolioReviewResult` validated via Pydantic v2 `model_validate_json()`; LLM output never eval'd |
| V6 Cryptography | no | n/a — LLM API keys in Settings (pydantic-settings), loaded from env |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| LLM prompt injection via position data | Tampering | Position data (ticker names, quantities) comes from broker, not user input; tickers are pre-validated strings from InstrumentRegistry |
| LLM output triggers trade via side channel | Elevation | `PortfolioReviewResult` has no `direction`/`side` fields; handler has no reference to `BrokerRouter`; code-grep test is the verification gate |
| Advisory report mistaken for signal | Spoofing | Schema field names (`ticker`, `market`, `risk_score`) are structurally different from `Signal` (`symbol`, `market_id`, `direction`, `confidence`); module-level assertion enforces at import time |
| Alert flood from daily review | DoS | One cron job per day; no retry loop on success |

---

## Environment Availability

Step 2.6: SKIPPED — Phase 52 is code-only. All required libraries (apscheduler, asyncio, pydantic, anthropic/openai, httpx) are already installed dependencies. No new CLI tools, databases, or external services are introduced.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | 19:00 MSK = 16:00 UTC year-round (MSK is UTC+3, no DST since 2014) | Architecture Patterns | LOW — Russia abolished DST transitions in 2014; UTC offset is fixed at +3. If wrong, the cron fires 1 hour early/late at DST boundary — no trading impact since it is advisory-only |
| A2 | `PortfolioReviewResult` should be placed in `analysis/portfolio_review_agent.py` rather than `core/schemas.py` | Dependency Layer Placement | LOW — either location works architecturally; separate file is cleaner and consistent with Phase 51's `anomaly_detector.py` pattern |
| A3 | `_review_LLM_TIMEOUT = 60.0` is sufficient for portfolio review LLM calls | Common Pitfalls | LOW — adjustable constant; fire-and-forget means no trading cycle is blocked regardless of timeout |
| A4 | Module-level `assert not (fields & FORBIDDEN_FIELDS)` is the right enforcement point for the schema safety invariant | Architecture Patterns | LOW — assertions can be disabled with `-O`; a unit test (`TestSchemaAdvisoryOnly`) provides the non-bypassable verification |
| A5 | Using `ticker`/`market` field names (instead of `symbol`/`market_id`) is sufficient to satisfy PFRA-03 field-set disjointness | Architecture Patterns | MEDIUM — the success criterion checks for field-set disjointness from `Signal` and `OrderRequest`. If the criterion is interpreted as "ANY field named `symbol`", using `ticker` satisfies it. If interpreted as "structurally cannot be routed to a broker", the schema design + code-grep both reinforce it |

---

## Open Questions

1. **LLM prompt for structured `PortfolioReviewResult` output**
   - What we know: `parse_structured()` calls `model_validate_json()` on the raw LLM response; the system prompt must instruct the LLM to return valid JSON matching the schema
   - What's unclear: How explicit to make the JSON schema in the prompt (embed full schema vs describe fields narratively)
   - Recommendation: Include the full Pydantic model JSON schema in the system prompt as a reference block — reduces hallucination of extra fields or missing required fields

2. **Whether `_gather_portfolio_data()` should be sync or async**
   - What we know: `broker.get_portfolio()` and `broker.get_positions()` are synchronous methods on `BrokerBase` [VERIFIED: execution/broker_base.py]; calling sync code from inside an async coroutine is fine as long as it doesn't block the event loop for > ~100ms
   - What's unclear: Whether broker calls could be slow (e.g., live Tinkoff gRPC) and block the async loop
   - Recommendation: Keep as synchronous calls in the fire-and-forget coroutine for simplicity; if Tinkoff latency becomes a concern, wrap with `asyncio.get_event_loop().run_in_executor(None, broker.get_portfolio)` — but this is out of scope for Phase 52

---

## Sources

### Primary (HIGH confidence)
- `src/finalayze/core/trading_loop.py` — APScheduler cron pattern (`_daily_reset`), fire-and-forget dispatch (`_enrich_anomaly_async`), portfolio data gathering (`_get_cached_portfolio`, `_daily_reset`), `_async_loop` lifecycle
- `src/finalayze/core/alerts.py` — `TelegramAlerter.send_alert()` and `._send()` implementation
- `src/finalayze/analysis/llm_client.py` — `LLMClient.parse_structured()` signature, provider implementations
- `src/finalayze/analysis/anomaly_detector.py` — Phase 51 Layer 3 module placement precedent, `AnomalyResult` frozen Pydantic schema pattern
- `src/finalayze/core/schemas.py` — `Signal` and `PortfolioState` field definitions; basis for safety disjointness check
- `src/finalayze/execution/broker_base.py` — `OrderRequest` fields; `BrokerBase.get_portfolio()` / `get_positions()` contract
- `pyproject.toml` — APScheduler 3.10.4, Pydantic 2.10.0, pytest-asyncio 0.25+ (`asyncio_mode = "auto"`)
- `.planning/phases/51-anomaly-interpreter-agent/51-RESEARCH.md` — established fire-and-forget pattern, pitfall inventory
- `.planning/phases/52-portfolio-review-agent/52-CONTEXT.md` — locked decisions
- `.planning/REQUIREMENTS.md` — PFRA-01, PFRA-02, PFRA-03 definitions

### Secondary (MEDIUM confidence)
- Russia abolished DST in 2014 — MSK is permanently UTC+3 [ASSUMED: standard geopolitical/timezone knowledge]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages are already in the project; no new dependencies
- Architecture: HIGH — locked decisions in CONTEXT.md; Phase 51 pattern directly applicable
- Schema safety design: HIGH — field-level analysis of Signal/OrderRequest verified from source code
- Scheduling (UTC offset): MEDIUM — MSK=UTC+3 is standard knowledge but tagged ASSUMED

**Research date:** 2026-04-15
**Valid until:** 2026-05-15 (stable domain — no external API changes expected; APScheduler timezone behavior is stable)
