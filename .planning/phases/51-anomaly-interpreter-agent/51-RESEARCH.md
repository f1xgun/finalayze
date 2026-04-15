# Phase 51: Anomaly Interpreter Agent - Research

**Researched:** 2026-04-14
**Domain:** Async fire-and-forget LLM enrichment, statistical anomaly detection in trading signals
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- AnomalyDetector does not exist yet — this is greenfield
- Detect >3σ price moves and volume spikes vs 20-day rolling mean/std in `_process_instrument()` after candle fetch
- Raw alert = `TelegramAlerter.send_alert()` with ticker, magnitude, direction — fires IMMEDIATELY before any LLM call
- Keep detection as a helper function or lightweight class within the trading loop module — no complex architecture needed
- Fire-and-forget LLM enrichment via `loop.create_task()` — matches existing TelegramAlerter pattern
- LLM prompt includes: ticker, price move %, volume ratio, recent news headlines if available from sentiment cache
- Follow-up message format: `"AI interpretation (unverified): {explanation}"`
- 30s timeout on LLM call via `asyncio.wait_for()`
- On failure: log `anomaly_llm_failure` via structlog, do NOT send follow-up message
- Raw alert is NEVER delayed — `send_alert()` happens synchronously before `create_task(llm_enrichment)`
- "Suppressing the raw alert on LLM failure is impossible by design" — architectural guarantee, not just error handling

### Claude's Discretion

- Whether to create a new `AnomalyDetector` class in a separate file or keep as functions in trading_loop.py
- Exact σ threshold (3σ suggested, adjustable)
- Exact rolling window (20-day suggested)
- LLM prompt wording and context included
- Whether to add `on_anomaly()` method to TelegramAlerter or use generic `send_alert()`

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ANMI-01 | AnomalyDetector fires raw alert immediately, then async LLM enrichment follows | Existing `send_alert()` fire-and-forget via `loop.create_task()` is the canonical pattern — call it synchronously, then schedule LLM coroutine via `create_task()` or `run_coroutine_threadsafe()` |
| ANMI-02 | LLM explanation appended to Telegram alert labeled "AI interpretation (unverified)" | `LLMClient.complete()` wrapped in `asyncio.wait_for(30s)` → format result → call `send_alert()` from inside async coroutine |
| ANMI-03 | Graceful degradation — LLM timeout/failure does not suppress or delay raw statistical alert | Architectural separation: raw alert path has no `await`, LLM path is fully isolated in a background task |
</phase_requirements>

---

## Summary

Phase 51 adds statistical anomaly detection and async LLM interpretation to the trading loop. The design is a two-step fire-and-forget sequence: (1) detect a statistical anomaly in candle data and immediately call `TelegramAlerter.send_alert()` — which is already fire-and-forget via `loop.create_task()` — then (2) schedule an async LLM enrichment coroutine that sends a second follow-up Telegram message with "AI interpretation (unverified)".

The critical correctness guarantee is that the raw alert path contains zero `await` expressions. The LLM enrichment is isolated in a background coroutine that is created with `loop.create_task()` (from inside `send_alert()`'s running loop branch) or `asyncio.run_coroutine_threadsafe()` (from the sync `_process_instrument()` thread). Either approach works — but `_process_instrument()` runs on a sync APScheduler thread, so the enrichment must be dispatched to `self._async_loop` via `run_coroutine_threadsafe()` — the same pattern already used throughout `TradingLoop`.

Anomaly detection itself is pure statistics on the `candles` list already available inside `_process_instrument()`: compute rolling 20-bar mean and std for price change and volume, check if latest values exceed 3σ. No external data fetch is required.

**Primary recommendation:** Implement `_check_anomaly()` as a module-level function in `trading_loop.py`, called at the top of `_process_instrument()` after candle fetch. When an anomaly fires, call `self._alerter.send_alert(raw_text)` synchronously, then dispatch the LLM enrichment coroutine via `asyncio.run_coroutine_threadsafe(self._enrich_anomaly(...), self._async_loop)` without calling `.result()` (fire-and-forget).

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| asyncio (stdlib) | Python 3.12 | Task scheduling for fire-and-forget LLM enrichment | Already the async runtime; `create_task()` / `run_coroutine_threadsafe()` are the existing patterns |
| structlog | >=24.4.0 | Structured logging for `anomaly_llm_failure` event | Project-wide logging standard [VERIFIED: pyproject.toml] |
| httpx | >=0.28.0 | Used by TelegramAlerter internally | Already in project dependencies [VERIFIED: pyproject.toml] |
| anthropic / openai | >=0.42.0 / >=1.50.0 | LLMClient provider backends | Already wired via `create_llm_client(settings)` factory [VERIFIED: analysis/llm_client.py] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| statistics (stdlib) | Python 3.12 | Mean/std for rolling anomaly detection | Sufficient for 20-bar window; numpy already available if preferred |
| numpy / pandas | >=1.26.0 / >=2.2.0 | Rolling stats on candle series | Use if candles are already converted to arrays; overkill for 20-bar window in sync code |

**Installation:** No new packages needed. All dependencies already in `pyproject.toml`.

---

## Architecture Patterns

### Component Location (Discretion Area)

Two viable options per CONTEXT.md discretion:

**Option A (recommended): Module-level helper functions in `trading_loop.py`**
- `_check_anomaly(candles: list[Candle]) -> AnomalyResult | None`
- `_format_raw_alert(symbol: str, market_id: str, result: AnomalyResult) -> str`
- Called from `_process_instrument()` after candle fetch
- Keeps the blast radius small; no new file, no import additions

**Option B: New `analysis/anomaly_detector.py` (Layer 3)**
- `AnomalyDetector` class with `detect(candles, symbol, market_id) -> AnomalyResult | None`
- Cleaner if detector logic grows; testable in isolation
- Adds a new Layer 3 module; `TradingLoop` would import it (Layer 6 imports Layer 3 — allowed)
- Better separation for future features (configurable thresholds, multiple anomaly types)

**Recommendation:** Option B (separate file) because the success criteria explicitly names "AnomalyDetector" and a separate class is easier to unit-test in isolation without constructing a full TradingLoop.

### Recommended Project Structure Addition
```
src/finalayze/
├── analysis/
│   ├── anomaly_detector.py   # NEW — Layer 3: AnomalyDetector + AnomalyResult schema
│   └── ...existing...
├── core/
│   └── trading_loop.py       # MODIFIED — call detector, fire raw alert, dispatch LLM task
```

### Pattern 1: Raw Alert Before LLM Enrichment (Critical Ordering)

**What:** Synchronous `send_alert()` followed by non-blocking task dispatch. Zero `await` between detection and alert.

**When to use:** Any time an event must be reported immediately AND optionally enriched asynchronously.

```python
# Source: [VERIFIED: core/alerts.py send_alert() + core/trading_loop.py _run_async pattern]

# In _process_instrument() — sync method on APScheduler thread:

anomaly = _check_anomaly(candles, instrument.symbol, market_id)
if anomaly is not None:
    # STEP 1: raw alert — synchronous, fire-and-forget via existing send_alert()
    raw_text = (
        f"ANOMALY {instrument.symbol} [{market_id.upper()}]: "
        f"{anomaly.price_move_pct:+.1f}% price move "
        f"({anomaly.sigma:.1f}σ), vol {anomaly.volume_ratio:.1f}x avg"
    )
    self._alerter.send_alert(raw_text)

    # STEP 2: LLM enrichment — pure fire-and-forget, no .result() call
    if self._async_loop is not None and not self._async_loop.is_closed():
        asyncio.run_coroutine_threadsafe(
            self._enrich_anomaly_async(instrument.symbol, market_id, anomaly),
            self._async_loop,
        )
        # No .result() — fire-and-forget  # noqa: RUF006 pattern
```

### Pattern 2: Async LLM Enrichment with Timeout and Graceful Failure

**What:** Coroutine that calls LLM, applies 30s timeout, sends follow-up Telegram, logs failure on exception.

```python
# Source: [VERIFIED: analysis/news_analyzer.py asyncio.wait_for pattern]

async def _enrich_anomaly_async(
    self,
    symbol: str,
    market_id: str,
    anomaly: AnomalyResult,
) -> None:
    """Fire-and-forget LLM enrichment — never raises, never blocks raw alert."""
    try:
        prompt = _build_anomaly_prompt(symbol, market_id, anomaly)
        explanation = await asyncio.wait_for(
            self._llm_client.complete(prompt, _ANOMALY_SYSTEM_PROMPT),
            timeout=30.0,
        )
        follow_up = f"AI interpretation (unverified): {explanation}"
        await self._alerter._send(follow_up)  # direct async call
    except Exception:
        _log.warning(
            "anomaly_llm_failure",
            symbol=symbol,
            market_id=market_id,
        )
```

**Note on calling `_send()` directly:** Inside the async enrichment coroutine, calling `self._alerter._send()` (the private async method) is correct — the public `send_alert()` would check for a running loop and create a nested task, which is unnecessary overhead inside an already-running coroutine.

### Pattern 3: AnomalyResult Schema

```python
# Source: [ASSUMED — consistent with existing Pydantic schema patterns in core/schemas.py]

from pydantic import BaseModel, ConfigDict

class AnomalyResult(BaseModel):
    """Statistical anomaly detected in candle data."""
    model_config = ConfigDict(frozen=True)

    symbol: str
    market_id: str
    price_move_pct: float   # latest bar price change as percentage
    sigma: float            # how many σ above rolling mean
    volume_ratio: float     # volume / 20-bar rolling mean volume
    anomaly_type: str       # "price" | "volume" | "both"
```

### Pattern 4: Anomaly Detection Logic

```python
# Source: [ASSUMED — standard z-score detection, consistent with 3σ threshold in CONTEXT.md]

_ANOMALY_SIGMA_THRESHOLD = 3.0
_ROLLING_WINDOW = 20

def _check_anomaly(
    candles: list[Candle],
    symbol: str,
    market_id: str,
) -> AnomalyResult | None:
    """Return AnomalyResult if latest candle shows >3σ deviation, else None."""
    if len(candles) < _ROLLING_WINDOW + 1:
        return None

    window = candles[-(  _ROLLING_WINDOW + 1):]
    closes = [float(c.close) for c in window]
    volumes = [float(c.volume) for c in window]

    # Price changes for rolling window (excluding latest)
    price_changes = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes) - 1)
    ]
    latest_change = (closes[-1] - closes[-2]) / closes[-2]

    import statistics  # stdlib, no new dep
    mean_chg = statistics.mean(price_changes)
    std_chg = statistics.stdev(price_changes) if len(price_changes) > 1 else 0.0

    price_sigma = abs(latest_change - mean_chg) / std_chg if std_chg > 0 else 0.0

    # Volume ratio
    avg_vol = statistics.mean(volumes[:-1]) if volumes[:-1] else 1.0
    vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
    vol_sigma = (vol_ratio - 1.0) * (avg_vol / statistics.stdev(volumes[:-1]) if statistics.stdev(volumes[:-1]) > 0 else 1.0)

    is_price_anomaly = price_sigma >= _ANOMALY_SIGMA_THRESHOLD
    is_vol_anomaly = vol_ratio >= 2.0  # volume >2x average as additional gate

    if not (is_price_anomaly or is_vol_anomaly):
        return None

    anomaly_type = "both" if (is_price_anomaly and is_vol_anomaly) else (
        "price" if is_price_anomaly else "volume"
    )
    return AnomalyResult(
        symbol=symbol,
        market_id=market_id,
        price_move_pct=latest_change * 100,
        sigma=price_sigma,
        volume_ratio=vol_ratio,
        anomaly_type=anomaly_type,
    )
```

### Anti-Patterns to Avoid

- **Blocking on LLM before raw alert:** Any `await llm_client.complete()` before `send_alert()` violates ANMI-01/ANMI-03. The raw alert must use only `send_alert()` (sync fire-and-forget), with zero await before it.
- **Using `_run_async()` for fire-and-forget:** `_run_async()` blocks with `.result(timeout=30)` — it is a synchronous blocking bridge, not fire-and-forget. For fire-and-forget, use `asyncio.run_coroutine_threadsafe(..., self._async_loop)` WITHOUT calling `.result()`.
- **Calling `send_alert()` inside the async enrichment coroutine:** Nested task creation. Call `await alerter._send(text)` directly.
- **Raising exceptions in the enrichment coroutine:** Must catch all exceptions at the top level and log `anomaly_llm_failure`; re-raising would silently kill the background task.
- **Adding `LLMClient` as required parameter to `TradingLoop.__init__()`:** LLMClient is optional — if no key is configured, anomaly detection still works, just without LLM enrichment. Guard with `if self._llm_client is not None`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Async timeout | Custom cancellation logic | `asyncio.wait_for(coro, timeout=30.0)` | stdlib; raises `asyncio.TimeoutError` which is caught by the outer except |
| Telegram HTTP | Custom HTTP client | `TelegramAlerter.send_alert()` / `._send()` | Already error-swallowing, tested, fire-and-forget |
| LLM call | Custom provider code | `LLMClient.complete()` | Already has retry, caching, 3 provider implementations |
| Background task from sync thread | threading.Thread | `asyncio.run_coroutine_threadsafe(coro, loop)` | The persistent `_async_loop` is already the right target |
| Rolling statistics | Manual loop | `statistics.mean()` / `statistics.stdev()` | Stdlib; or numpy if candles are already numpy arrays |

**Key insight:** The entire "raw alert fires before LLM" guarantee is an ordering property enforced by code structure, not by locking or synchronization primitives. The raw `send_alert()` call returns before `run_coroutine_threadsafe()` is called. This is unconditional by construction.

---

## Common Pitfalls

### Pitfall 1: Using `_run_async()` instead of fire-and-forget dispatch
**What goes wrong:** Developer reaches for `self._run_async(enrich_coro)` because it is the established pattern, but `_run_async()` calls `.result(timeout=30)` which blocks the APScheduler thread for up to 30 seconds.
**Why it happens:** The existing code uses `_run_async()` for all async calls in the trading loop — it looks like the right tool.
**How to avoid:** Fire-and-forget requires `asyncio.run_coroutine_threadsafe(coro, self._async_loop)` with NO `.result()` call. The future is intentionally discarded.
**Warning signs:** Tests that assert raw alert fires "before" LLM will fail if the LLM task blocks the thread.

### Pitfall 2: `asyncio.TimeoutError` vs `TimeoutError` in Python 3.12
**What goes wrong:** In Python 3.11+, `asyncio.TimeoutError` is an alias for the built-in `TimeoutError`. Catching only `asyncio.TimeoutError` is fine in Python 3.12, but a broad `except Exception:` catches both.
**Why it happens:** Subtle stdlib change between Python 3.10 and 3.11.
**How to avoid:** Use broad `except Exception:` in the enrichment coroutine to catch all failures uniformly and log `anomaly_llm_failure`. [VERIFIED: Python 3.12 docs — asyncio.TimeoutError is `TimeoutError`]

### Pitfall 3: LLMClient not available in TradingLoop
**What goes wrong:** `TradingLoop.__init__()` currently has no `llm_client` parameter. Adding anomaly LLM enrichment requires threading it in.
**Why it happens:** LLMClient was added to `NewsAnalyzer` but not to `TradingLoop` directly.
**How to avoid:** Add `llm_client: LLMClient | None = None` parameter to `TradingLoop.__init__()` for clean dependency injection. Guard enrichment with `if self._llm_client is not None`.

### Pitfall 4: Insufficient candles for rolling window
**What goes wrong:** If `len(candles) < 21` (20 rolling + 1 latest), the statistics are undefined; calling `statistics.stdev([])` raises `StatisticsError`.
**Why it happens:** Early in trading session or for illiquid instruments, fewer than 20 candles are available.
**How to avoid:** Guard: `if len(candles) < _ROLLING_WINDOW + 1: return None`. [VERIFIED: analysis of `_CANDLE_LOOKBACK = 60` constant in trading_loop.py — normally 60 candles are fetched, so 20-bar window is safe]

### Pitfall 5: Division by zero in std calculation
**What goes wrong:** If all 20 candle closes are identical (zero std), dividing by std produces ZeroDivisionError.
**Why it happens:** Very illiquid MOEX instruments may have zero movement for many bars.
**How to avoid:** Guard: `price_sigma = abs(latest_change - mean_chg) / std_chg if std_chg > 0 else 0.0`.

### Pitfall 6: Alert ordering test is hard to write with mocks
**What goes wrong:** CONTEXT success criterion requires "a unit test asserting that TelegramAlerter.send() is called before any LLM await". Standard `AsyncMock` ordering is not self-evident.
**Why it happens:** The test must verify ordering across sync/async boundary.
**How to avoid:** Use a `call_order: list[str] = []` list with `side_effect` on both `alerter.send_alert` and `llm_client.complete` to record invocation sequence — same pattern as `test_rate_limiter_integration.py`. [VERIFIED: tests/unit/test_rate_limiter_integration.py uses this exact pattern]

---

## Code Examples

### Call Ordering Test Pattern (Verified Project Pattern)
```python
# Source: [VERIFIED: tests/unit/test_rate_limiter_integration.py]

call_order: list[str] = []

def fake_send_alert(msg: str) -> None:
    call_order.append("raw_alert")

async def fake_llm_complete(prompt: str, system: str) -> str:
    call_order.append("llm_call")
    return "test explanation"

alerter.send_alert = fake_send_alert
llm_client.complete = AsyncMock(side_effect=fake_llm_complete)

# trigger anomaly detection
detector.detect_and_alert(candles, symbol, market_id, alerter, llm_client)

assert call_order[0] == "raw_alert", "raw alert must fire before LLM"
```

### NewsAnalyzer `asyncio.wait_for` Pattern (Established)
```python
# Source: [VERIFIED: analysis/news_analyzer.py]

try:
    return await asyncio.wait_for(
        self._llm.parse_structured(user_prompt, system, SentimentResult),
        timeout=_LLM_TIMEOUT_SECONDS,  # 5.0 for news; 30.0 for anomaly enrichment
    )
except TimeoutError:
    _log.warning("llm_timeout", ...)
    return _FALLBACK
except Exception:
    _log.warning("llm_parse_error", ...)
    return _FALLBACK
```

### Fire-and-Forget Dispatch from Sync Thread (Established)
```python
# Source: [VERIFIED: core/alerts.py send_alert() and core/trading_loop.py _run_async()]
# Pattern: run_coroutine_threadsafe WITHOUT .result() = fire-and-forget

if self._async_loop is not None and not self._async_loop.is_closed():
    asyncio.run_coroutine_threadsafe(
        self._enrich_anomaly_async(symbol, market_id, anomaly),
        self._async_loop,
    )  # no .result() = fire-and-forget  # noqa: RUF006
```

---

## Dependency Layer Placement

| Component | Layer | Module | Allowed Imports |
|-----------|-------|--------|-----------------|
| `AnomalyResult` schema | Layer 0 | `core/schemas.py` OR `analysis/anomaly_detector.py` | If in schemas.py: L0 only. If in anomaly_detector.py: L0-L2 ok |
| `AnomalyDetector` class | Layer 3 | `analysis/anomaly_detector.py` | Imports from L0, L1, L2 |
| Detection call site | Layer 6 (in TradingLoop) | `core/trading_loop.py` | Already at L6; imports L3 ok |
| `_enrich_anomaly_async()` | Layer 6 (method on TradingLoop) | `core/trading_loop.py` | Uses `LLMClient` (L3) and `TelegramAlerter` (L0) — both ok |

[VERIFIED: docs/architecture/DEPENDENCY_LAYERS.md]

**Decision recommendation:** Put `AnomalyResult` as a Pydantic model in `analysis/anomaly_detector.py` (not `core/schemas.py`) since it is analysis-specific and not needed by lower layers. This keeps Layer 0 focused.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.3+ with pytest-asyncio 0.25+ |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/unit/test_anomaly_detector.py -x` |
| Full suite command | `uv run pytest --cov=src/finalayze --cov-fail-under=50` |
| asyncio_mode | `auto` (all async tests work without `@pytest.mark.asyncio`) |

[VERIFIED: pyproject.toml `asyncio_mode = "auto"`]

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ANMI-01 | `send_alert()` called BEFORE any LLM await | unit | `uv run pytest tests/unit/test_anomaly_detector.py::TestOrderingGuarantee -x` | ❌ Wave 0 |
| ANMI-02 | Follow-up message contains "AI interpretation (unverified)" | unit | `uv run pytest tests/unit/test_anomaly_detector.py::TestLLMEnrichment -x` | ❌ Wave 0 |
| ANMI-03 | LLM timeout/failure: raw alert delivered, `anomaly_llm_failure` logged | unit | `uv run pytest tests/unit/test_anomaly_detector.py::TestGracefulDegradation -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_anomaly_detector.py -x`
- **Per wave merge:** `uv run pytest --cov=src/finalayze --cov-fail-under=50`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_anomaly_detector.py` — covers ANMI-01, ANMI-02, ANMI-03
- [ ] `src/finalayze/analysis/anomaly_detector.py` — AnomalyDetector class + AnomalyResult

*(No new conftest.py needed — existing `tests/conftest.py` with Settings fixture is sufficient)*

---

## Security Domain

> `security_enforcement` not set to false in config.json — section required.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a |
| V3 Session Management | no | n/a |
| V4 Access Control | no | n/a |
| V5 Input Validation | yes | Candle data is typed via Pydantic `Candle` schema; LLM output is plain string — not parsed as code or used in queries |
| V6 Cryptography | no | n/a — LLM API keys are in Settings (pydantic-settings), loaded from env |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| LLM prompt injection via ticker symbol | Tampering | Ticker is a validated string from InstrumentRegistry — already whitelist-validated before this point |
| LLM output rendered as executable | Elevation | LLM output goes only to Telegram as plain text — no eval, no code execution |
| Alert flood from repeated anomalies | DoS | Out of scope for this phase (deduplication not required by ANMI-01–03); can be added later |

---

## Environment Availability

Step 2.6: SKIPPED — Phase 51 is code-only. All required libraries (asyncio, structlog, httpx, anthropic/openai) are already installed dependencies. No new CLI tools, databases, or external services are introduced.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `AnomalyResult` should be a Pydantic model in `analysis/anomaly_detector.py`, not in `core/schemas.py` | Architecture Patterns | Low — either location compiles; placing in schemas.py would also work |
| A2 | `statistics.stdev()` is preferred over numpy for 20-bar window (no new import) | Don't Hand-Roll | Low — numpy is available; both work correctly |
| A3 | Volume anomaly threshold of 2x average (not σ-based) is reasonable as secondary gate | Common Pitfalls | Low — threshold is adjustable; main detection is price σ |
| A4 | Calling `alerter._send(text)` directly inside the async enrichment coroutine is preferred over `alerter.send_alert()` | Architecture Patterns | Low — `send_alert()` would also work but creates a redundant nested task |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed.

---

## Open Questions

1. **LLMClient injection into TradingLoop**
   - What we know: `TradingLoop.__init__()` has no `llm_client` parameter today; `NewsAnalyzer` holds its own `LLMClient` instance
   - What's unclear: Should the phase share the same `LLMClient` instance as `NewsAnalyzer`, or inject a separate one?
   - Recommendation: Share the same instance (pass it as `llm_client: LLMClient | None = None` to `TradingLoop.__init__()`) to avoid creating redundant HTTP connections and LRU caches

2. **`on_anomaly()` method on TelegramAlerter vs generic `send_alert()`**
   - What we know: CONTEXT.md marks this as Claude's discretion
   - What's unclear: Adding `on_anomaly()` would make intent clear and allow emoji formatting parity with `on_trade_filled()`
   - Recommendation: Add `on_anomaly(symbol, market_id, price_move_pct, sigma, volume_ratio)` for consistency with the existing API surface — avoids raw string formatting in trading_loop.py

---

## Sources

### Primary (HIGH confidence)
- `src/finalayze/core/alerts.py` — TelegramAlerter.send_alert() implementation, fire-and-forget pattern
- `src/finalayze/core/trading_loop.py` — `_run_async()`, `_process_instrument()`, async thread lifecycle
- `src/finalayze/analysis/llm_client.py` — LLMClient ABC, `complete()` signature, provider implementations
- `src/finalayze/analysis/news_analyzer.py` — `asyncio.wait_for()` + timeout pattern, established LLM call style
- `tests/unit/test_rate_limiter_integration.py` — `call_order: list[str]` pattern for ordering assertions
- `tests/unit/test_telegram_alerter.py` — existing Telegram test structure and mock patterns
- `docs/architecture/DEPENDENCY_LAYERS.md` — layer placement rules
- `pyproject.toml` — test config (asyncio_mode=auto, pytest-asyncio version, coverage threshold)
- `.planning/phases/51-anomaly-interpreter-agent/51-CONTEXT.md` — locked decisions
- `.planning/REQUIREMENTS.md` — ANMI-01, ANMI-02, ANMI-03 definitions

### Secondary (MEDIUM confidence)
- Python 3.12 stdlib docs: `asyncio.TimeoutError` is `TimeoutError` alias (relevant for exception handling)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages are already in the project; no new dependencies
- Architecture: HIGH — locked decisions in CONTEXT.md, existing patterns directly applicable
- Pitfalls: HIGH — identified from direct code inspection of existing implementation
- Test ordering pattern: HIGH — `call_order` pattern verified in existing test suite

**Research date:** 2026-04-14
**Valid until:** 2026-05-14 (stable domain — no external API changes expected)
