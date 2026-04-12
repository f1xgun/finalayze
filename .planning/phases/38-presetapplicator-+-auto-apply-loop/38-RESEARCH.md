# Phase 38: PresetApplicator + Auto-Apply Loop - Research

**Researched:** 2026-04-12
**Domain:** Atomic YAML write-back, safety gates, cache invalidation, Telegram routing, REST endpoint
**Confidence:** HIGH

## Summary

Phase 38 delivers the final link in the autonomy chain: accepting an experiment verdict and propagating its `preset_overrides` back into the strategy YAML files that drive live trading. All six requirements are pure Python integration work — no new external dependencies are needed. Every pattern called for (atomic file write, DB query for sandbox metrics, circuit breaker gate, Telegram alert) already exists in the codebase and is used by adjacent modules.

The critical insight is that `StrategyCombiner._load_config()` reads YAML from disk on every call to `generate_signal()` — there is **no in-memory cache** to invalidate. The `invalidate_segment_cache()` method called for in APPLY-04 is therefore a no-op compatibility shim (or a future-proofing hook) rather than clearing a real cache. The planner must reflect this accurately to avoid building cache-clearing logic that does not match the actual combiner implementation.

`SandboxGate` can be implemented as a synchronous class that queries `SandboxMetricRow` from TimescaleDB. The query pattern is identical to `GoNoGoReporter._load_recent_metrics()`, with a simpler pass/fail filter: 3+ distinct calendar dates with at least one row having `fill_rate > 0` and no circuit-breaker trips (proxy: `drawdown_pct < 0.10` threshold to detect L2 HALTED days).

**Primary recommendation:** Implement `PresetApplicator` in `orchestration/preset_applicator.py` as a synchronous class (no async required — file I/O and circuit-breaker checks are sync). Wire `SandboxGate` using the `GoNoGoReporter` DB query pattern. Add `POST /experiments/{id}/apply` to the existing `api/v1/experiments.py` router.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**PresetApplicator Design**
- Lives in `orchestration/preset_applicator.py` (Layer 5) — reads ExperimentManager, writes YAML presets
- Backup naming: `{segment}.yaml.bak.{ISO-timestamp}` in same directory as original
- Key validation: check keys exist in current YAML + value types match — reject unknown keys
- Deep merge: only overrides specified keys, preserves all other settings
- Atomic write: write to `{segment}.yaml.pending`, then `os.replace()` to final path

**Safety Gates**
- Circuit breaker check is FIRST line of `apply_verdict()` — import CircuitBreaker from `risk/`, check level. Raise if `CircuitLevel != NORMAL`
- `_entry_strategy`: Dict `{symbol: strategy_name}` in TradingLoop, set on fill, cleared on close
- `_entry_strategy` lives in `orchestration/trading_loop.py` in the order execution path
- Block strategy-disable auto-apply if `_entry_strategy` has positions for that strategy
- INCONCLUSIVE → use existing `TelegramAlerter.send_alert()` with priority=HIGH, includes experiment_id and key metrics

**Sandbox Gate + Cache Invalidation**
- `SandboxGate` class with `check(experiment_id) -> bool` — reads sandbox monitor metrics
- Pass criteria: 3+ trading days with fill_rate > 0 AND no circuit breaker trips during the period
- `combiner.invalidate_segment_cache()`: clear the `_presets` dict for the affected segment
- Call invalidate immediately after `os.replace()` in PresetApplicator

### Claude's Discretion
- Internal error handling in PresetApplicator (rollback on failure vs leave pending file)
- SandboxGate metric collection mechanism (read from DB or sandbox monitor API)
- Telegram alert message formatting for INCONCLUSIVE verdicts
- REST endpoint response model structure for apply results

### Deferred Ideas (OUT OF SCOPE)
- Auto-revert on post-apply metric degradation (rollback gate) — deferred to v8.x
- Scheduled orchestrator + auto-apply loop (cron-based) — deferred to v8.x
- `ou_mean_reversion` and `pairs` strategies blocked from auto-apply by schema flag — deferred to v8.x
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| APPLY-01 | `PresetApplicator` writes experiment `preset_overrides` to strategy YAML with backup snapshot and atomic `os.replace()` rename | Verified: `iteration_tracker._atomic_write()` and `ml/loader._atomic_save()` provide the exact pattern; YAML presets live in `src/finalayze/strategies/presets/*.yaml` |
| APPLY-02 | Circuit-breaker gate blocks auto-apply when `CircuitLevel != NORMAL` | Verified: `CircuitBreaker.level` property (read-only) returns `CircuitLevel` enum; check `cb.level != CircuitLevel.NORMAL` |
| APPLY-03 | `_entry_strategy` dict in `TradingLoop` tracks which strategy opened each position; blocks strategy-disable if positions exist | Verified: `_entry_prices`, `_stop_states` dicts are the pattern; `_entry_strategy: dict[str, str]` is a new parallel dict |
| APPLY-04 | `combiner.invalidate_segment_cache()` method forces preset reload after YAML write | Verified: `_load_config()` reads YAML from disk on every call — no in-memory cache exists. Method is a no-op shim. |
| APPLY-05 | INCONCLUSIVE experiment verdicts route to Telegram alert (no auto-apply) | Verified: `TelegramAlerter.send_alert(message, priority=AlertPriority.IMPORTANT)` is the correct call |
| APPLY-06 | Sandbox validation gate (≥3 trading days) required between ACCEPT verdict and live apply | Verified: `SandboxMetricRow` in DB with `fill_rate`, `drawdown_pct`, `timestamp`; query pattern from `GoNoGoReporter._load_recent_metrics()` |
</phase_requirements>

---

## Standard Stack

### Core (all project-standard)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `yaml` (PyYAML) | Already in deps | YAML read/write for preset files | Used in `combiner._load_config()`, `experiment_manager._write_file()` |
| `os` (stdlib) | stdlib | `os.replace()` for atomic rename | Cross-platform atomic rename without external deps |
| `pathlib.Path` | stdlib | File path manipulation | Project-wide convention |
| `structlog` | Already in deps | Structured logging | Project-wide convention |
| `sqlalchemy` (async) | 2.0, already in deps | `SandboxMetricRow` query for SandboxGate | Same pattern as `GoNoGoReporter` |
| `fastapi` | Already in deps | `POST /experiments/{id}/apply` endpoint | Existing experiments router |
| `pydantic` v2 | Already in deps | Request/response models | Project convention — all schemas |

**No new dependencies required.** [VERIFIED: grep of pyproject.toml not needed — all packages confirmed present in adjacent modules]

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `os.replace()` (POSIX atomic) | `shutil.move()` | `os.replace()` is POSIX-guaranteed atomic; `shutil.move()` is not. Use `os.replace()`. |
| Sync DB query in SandboxGate | Async query | SandboxGate is called from sync context (REST endpoint or PresetApplicator). Use sync `asyncio.run()` wrapper or adopt GoNoGoReporter's `_run_async_safe` pattern from SandboxMonitorService. |
| Direct circuit breaker import | Read from API | PresetApplicator lives in Layer 5 and can import directly from `risk/circuit_breaker.py` (Layer 4). No indirection needed. |

---

## Architecture Patterns

### Recommended Project Structure

New files:
```
src/finalayze/orchestration/preset_applicator.py   # PresetApplicator + SandboxGate (Layer 5)
tests/unit/test_preset_applicator.py               # unit tests
```

Modified files:
```
src/finalayze/orchestration/trading_loop.py        # add _entry_strategy dict
src/finalayze/strategies/combiner.py               # add invalidate_segment_cache()
src/finalayze/api/v1/experiments.py                # add POST /{id}/apply endpoint
tests/unit/test_api_experiments.py                 # add apply endpoint tests
```

### Pattern 1: Atomic YAML Write (APPLY-01)

The canonical pattern in this codebase is used in `backtest/iteration_tracker.py`:

```python
# Source: src/finalayze/backtest/iteration_tracker.py:586-605 [VERIFIED]
@staticmethod
def _atomic_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = tempfile.NamedTemporaryFile(
        dir=target.parent,
        suffix=".tmp",
        delete=False,
        mode="w",
    )
    try:
        fd.write(content)
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        os.rename(fd.name, target)
    except Exception:
        fd.close()
        with contextlib.suppress(OSError):
            os.unlink(fd.name)
        raise
```

For PresetApplicator, the CONTEXT.md specifies using `{segment}.yaml.pending` as the staging filename, then `os.replace()`. Both achieve the same atomicity; the `.pending` → `os.replace()` variant is slightly simpler. The backup must be written **before** the pending file is created.

**PresetApplicator.apply_verdict() call order:**
1. Check `circuit_breaker.level != CircuitLevel.NORMAL` → raise `PresetApplyBlockedError`
2. Read experiment state from `ExperimentManager`
3. Check verdict is `ACCEPTED` (INCONCLUSIVE → call `TelegramAlerter.send_alert()` + return)
4. Check `SandboxGate.check()` passes
5. Check `_entry_strategy` for position ownership conflicts
6. Read current YAML, validate keys exist + types match `preset_overrides`
7. Deep merge `preset_overrides` into loaded dict
8. Write backup: `{segment}.yaml.bak.{ISO-timestamp}`
9. Write `{segment}.yaml.pending` (full merged YAML)
10. `os.replace("{segment}.yaml.pending", "{segment}.yaml")` — atomic
11. `combiner.invalidate_segment_cache(segment_id)` — clears nothing but is a defined contract

### Pattern 2: Circuit Breaker Check (APPLY-02)

```python
# Source: src/finalayze/risk/circuit_breaker.py:39-45 [VERIFIED]
class CircuitLevel(StrEnum):
    NORMAL = "normal"
    CAUTION = "caution"
    HALTED = "halted"
    LIQUIDATE = "liquidate"

# Usage pattern:
from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel

def apply_verdict(self, experiment_id: str) -> ApplyResult:
    if self._circuit_breaker.level != CircuitLevel.NORMAL:
        raise PresetApplyBlockedError(
            f"Circuit breaker level {self._circuit_breaker.level} blocks apply"
        )
```

The `level` property is read-only and thread-safe (no lock needed for read). [VERIFIED: circuit_breaker.py:78-80]

### Pattern 3: _entry_strategy in TradingLoop (APPLY-03)

Existing parallel dicts as model [VERIFIED: trading_loop.py:196-221]:
```python
self._stop_states: dict[str, StopLossState] = {}       # symbol -> stop state
self._entry_prices: dict[str, Decimal] = {}             # symbol -> entry price
```

New dict follows same pattern:
```python
self._entry_strategy: dict[str, str] = {}               # symbol -> strategy_name
```

Set on fill (where `_entry_prices` is set), cleared on position close (where `_stop_states` is cleared via `_stop_loss_lock`). Thread safety: same `_stop_loss_lock` or a new dedicated lock (simpler: reuse existing lock).

### Pattern 4: StrategyCombiner._load_config() — No Cache Exists (APPLY-04)

```python
# Source: src/finalayze/strategies/combiner.py:504-513 [VERIFIED]
def _load_config(self, segment_id: str) -> dict[str, object]:
    """Load segment YAML preset, returning an empty dict if not found or malformed."""
    try:
        path = self._presets_dir / f"{segment_id}.yaml"
        with path.open() as f:
            result = yaml.safe_load(f)
        return dict(result) if isinstance(result, dict) else {}
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return {}
```

`_load_config()` reads from disk on every invocation — no `_presets` dict, no LRU cache, no TTL. The method to add, `invalidate_segment_cache()`, has nothing to clear. It should be implemented as a no-op with a docstring explaining this, so callers can call it without conditional logic, and future caching additions have a defined hook.

### Pattern 5: SandboxGate DB Query (APPLY-06)

GoNoGoReporter's `_load_recent_metrics` is the template [VERIFIED: go_no_go.py:311-323]:
```python
async def _load_recent_metrics(self, session: AsyncSession) -> list[Any]:
    from sqlalchemy import select
    from finalayze.core.models import SandboxMetricRow
    stmt = (
        select(SandboxMetricRow)
        .where(SandboxMetricRow.market_id == self._market_id)
        .order_by(SandboxMetricRow.timestamp)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())
```

`SandboxMetricRow` columns available for SandboxGate: `timestamp`, `fill_rate`, `drawdown_pct`, `trade_count`, `market_id`. [VERIFIED: core/models.py:349-365]

**SandboxGate pass criteria (APPLY-06):**
- Count distinct calendar dates where `fill_rate > 0` → must be >= 3
- No rows with `drawdown_pct >= 0.10` in the same window (proxy for circuit breaker HALTED trips)

**Sync wrapper approach** (since `apply_verdict()` is sync): Use `asyncio.run()` in a helper or adopt the `SandboxMonitorService._run_async_safe()` background-loop pattern. Simpler: call `get_async_session_factory()` and `asyncio.run()` in a private sync method since `apply_verdict()` is not called from a running event loop.

### Pattern 6: Telegram Alert for INCONCLUSIVE (APPLY-05)

```python
# Source: src/finalayze/api/alerts.py:445-478 [VERIFIED]
# send_alert() is thread-safe, suppresses all exceptions, works from sync/async context
alerter.send_alert(
    message=(
        f"<b>INCONCLUSIVE Experiment: {experiment_id}</b>\n\n"
        f"Hypothesis: {state.hypothesis}\n"
        f"Metric: {state.success_criteria.metric} = {metric_value:.4f}\n"
        f"Threshold: {state.success_criteria.operator} {state.success_criteria.threshold}\n"
        f"Reasoning: {state.reasoning}"
    ),
    priority=AlertPriority.IMPORTANT,
)
```

`AlertPriority` must be imported from `finalayze.api.alerts`. PresetApplicator lives in Layer 5 (orchestration), which can import from Layer 6 (api) — wait: **Layer 5 cannot import from Layer 6 (api).** [VERIFIED: CLAUDE.md dependency layering: Layer 5 is Execution, Layer 6 is API/Dashboard. Layer 5 cannot import upward.]

**Resolution:** Pass `TelegramAlerter` to `PresetApplicator.__init__()` as an injected dependency (same pattern as `TradingLoop` which receives `alerter: TelegramAlerter` as a constructor parameter). The `TYPE_CHECKING` guard is used for the type hint:
```python
if TYPE_CHECKING:
    from finalayze.api.alerts import AlertPriority, TelegramAlerter
```
At runtime, call `alerter.send_alert(message, priority=None)` if not importing `AlertPriority` at runtime — or use a string sentinel and pass `priority=None` (send_alert accepts `priority: AlertPriority | None = None`).

**Cleaner approach:** Import `AlertPriority` inside the method body (deferred import, same pattern used throughout the codebase for upper-layer dependencies):
```python
def _alert_inconclusive(self, ...) -> None:
    from finalayze.api.alerts import AlertPriority  # noqa: PLC0415
    self._alerter.send_alert(message, priority=AlertPriority.IMPORTANT)
```

### Pattern 7: REST Endpoint (POST /experiments/{id}/apply)

The existing `api/v1/experiments.py` router already has the prefix, auth, and pattern [VERIFIED]. New endpoint:

```python
class ApplyResultResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    experiment_id: str
    applied: bool
    backup_path: str | None    # None when not applied (INCONCLUSIVE/blocked)
    verdict: str
    reason: str

@router.post("/{experiment_id}/apply", response_model=ApplyResultResponse)
async def apply_experiment(experiment_id: str) -> ApplyResultResponse:
    ...
```

The `PresetApplicator` instance must be accessible from the endpoint. Options:
1. Module-level singleton (fragile in tests)
2. FastAPI dependency injection (preferred — matches existing pattern)
3. Instantiate fresh per-request (acceptable if circuit_breaker reference is passed in)

**Recommendation (Claude's Discretion):** FastAPI dependency with `app.state` to hold the singleton `PresetApplicator` instance, injected via a `Depends()` factory. This is how `GoNoGoReporter` is injected into the Telegram bot handler.

### Anti-Patterns to Avoid

- **Reading YAML inside the lock:** Not needed — `_load_config()` is already lockless. The only atomicity concern is the write sequence.
- **Using `shutil.copy()` for backup then `shutil.move()` for the main file:** Use `os.replace()` for atomic semantics. `shutil.move()` is not guaranteed atomic cross-device.
- **Calling `asyncio.run()` from inside a running event loop:** The FastAPI endpoint is async context — use `await` and an async SandboxGate method; or make `SandboxGate.check()` fully async and `await` it from the endpoint.
- **Importing `TelegramAlerter` at module level in orchestration/:** Layer 5 cannot import Layer 6 at module level. Use `TYPE_CHECKING` guard + deferred runtime import, or constructor injection.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Atomic file write | Custom rename logic | `os.replace()` + pending file pattern (from iteration_tracker) | POSIX-guaranteed, handles interrupts |
| YAML merge | Custom dict-walker | Standard `dict.update()` with key validation pre-check | Preset YAML is shallow (2-level); no need for recursive deep merge |
| DB query for sandbox metrics | ORM reimplementation | `GoNoGoReporter._load_recent_metrics()` query pattern | Already uses correct async SQLAlchemy 2.0 pattern |
| Telegram alert | Custom HTTP post | `TelegramAlerter.send_alert()` | Thread-safe, rate-limited, suppresses exceptions |
| Circuit breaker level check | Custom state polling | `CircuitBreaker.level` property | Already thread-safe read |

**Key insight:** Every primitive needed for this phase already exists in the codebase. The implementation is integration work, not new algorithm design.

---

## Common Pitfalls

### Pitfall 1: Assuming combiner has an in-memory cache
**What goes wrong:** Implementing `invalidate_segment_cache()` to clear a `_presets` dict that doesn't exist, then discovering the combiner reads from disk every call anyway.
**Why it happens:** The CONTEXT.md says to add `invalidate_segment_cache()` — this is a forward-compatibility hook, not a cache eviction requirement.
**How to avoid:** Implement `invalidate_segment_cache(segment_id: str) -> None` as a documented no-op with a comment: "StrategyCombiner reads YAML from disk on every call — no in-memory cache to clear. This method is a hook for future caching additions."
**Warning signs:** If you see a `_presets` dict on `StrategyCombiner`, a cache was added since this research. Re-read `combiner.py` before implementing.

### Pitfall 2: Layer violation — orchestration importing from api/
**What goes wrong:** `preset_applicator.py` (Layer 5) imports `AlertPriority` or `TelegramAlerter` at module level from `finalayze.api.alerts` (Layer 6).
**Why it happens:** It's the natural import when you need to call `send_alert()`.
**How to avoid:** Use constructor injection for `TelegramAlerter`, type-hint it under `TYPE_CHECKING`, and use deferred `from finalayze.api.alerts import AlertPriority  # noqa: PLC0415` inside the method body at runtime.
**Warning signs:** mypy complains about circular imports, or `ruff` flags a forbidden upward import.

### Pitfall 3: Calling asyncio.run() from within FastAPI async endpoint
**What goes wrong:** `SandboxGate.check()` calls `asyncio.run()` internally, but FastAPI routes are already in a running event loop → `RuntimeError: This event loop is already running`.
**Why it happens:** SandboxGate needs to query DB (async), PresetApplicator is used from a sync context (TradingLoop) AND from an async FastAPI endpoint.
**How to avoid:** Make `SandboxGate` provide two interfaces: `async def check_async(session)` for use from FastAPI; and a sync `check(market_id)` that creates its own event loop (for use from non-async callers). Or: make `apply_verdict()` async throughout and let FastAPI `await` it. **Recommended:** Make the endpoint `async def apply_experiment()` and call `await applicator.apply_verdict_async(...)`. Internal sync helper for tests.
**Warning signs:** `RuntimeError: This event loop is already running` in test or runtime logs.

### Pitfall 4: Race between backup write and main file replace
**What goes wrong:** Backup is written AFTER the main file is replaced. If the process crashes between replace and backup, backup is missing.
**Why it happens:** Wanting to minimize I/O before the critical operation.
**How to avoid:** Always write backup FIRST, then write pending, then `os.replace()`. Order: backup → pending → replace.

### Pitfall 5: Key validation too strict for nested YAML
**What goes wrong:** `preset_overrides` may contain dotted-path keys like `strategies.dual_momentum.weight` that need to be validated against the nested YAML structure.
**Why it happens:** The CONTEXT.md says "check keys exist in current YAML" but doesn't specify the key format.
**How to avoid:** Clarify with the user. Likely format is flat dict matching top-level keys (`strategies`, `min_combined_confidence`, etc.) or nested dict matching the YAML structure. The YAML presets have a 2-level structure (top-level config + per-strategy config). Recommend supporting nested dicts mirroring the YAML tree structure, validated by traversal.
**Warning signs:** Test cases with `{"strategies": {"dual_momentum": {"weight": 0.30}}}` fail validation when only top-level key check is implemented.

---

## Code Examples

### Verified YAML Preset Structure

```yaml
# Source: src/finalayze/strategies/presets/us_tech.yaml [VERIFIED]
segment_id: us_tech
normalize_mode: "firing"
min_combined_confidence: 0.30
min_exit_confidence: 0.25
regime_routing:
  enabled: true
  adx_period: 14
  trend_threshold: 35
  mr_threshold: 15
strategies:
  dual_momentum:
    enabled: true
    weight: 0.25
    params:
      lookback_1m: 21
      ...
```

Top-level keys: `segment_id`, `normalize_mode`, `min_combined_confidence`, `min_exit_confidence`, `regime_routing`, `strategies`. The `preset_overrides` dict will most likely be a nested dict matching this structure (e.g., `{"strategies": {"dual_momentum": {"weight": 0.30}}}`).

### ExperimentState.preset_overrides Format

```python
# Source: src/finalayze/core/schemas.py:784 [VERIFIED]
preset_overrides: dict[str, Any] | None = None
```

Format is `dict[str, Any]` — no schema constraint beyond that. Convention from `ExperimentManager.create_experiment()` is to pass the same dict that will be written back. Example override matching YAML structure:
```python
{"strategies": {"dual_momentum": {"weight": 0.30, "enabled": True}}}
```

### _entry_strategy Population Site

Based on `_entry_prices` population pattern [VERIFIED: trading_loop.py:221]:
```python
# In order fill callback:
self._entry_prices[symbol] = fill_price
self._entry_strategy[symbol] = signal.strategy_name  # NEW: parallel dict
```

Position close clears both:
```python
# In position close / stop-loss path:
self._entry_prices.pop(symbol, None)
self._entry_strategy.pop(symbol, None)  # NEW: clear on close
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual YAML edits for preset updates | Atomic PresetApplicator write-back | Phase 38 (this phase) | Closes the autonomy loop — accepted experiments automatically update live config |
| Experiments router is read-only | POST /apply endpoint added | Phase 38 (this phase) | Enables one-command apply from REST/CLI |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `_entry_prices` population site is the correct place to also set `_entry_strategy` | Architecture Patterns, Pattern 3 | If fills are tracked elsewhere, `_entry_strategy` will be incomplete. Reading trading_loop.py fully before implementing is essential. |
| A2 | `preset_overrides` dict is structured as a nested dict mirroring YAML structure | Code Examples | If format is flat with dot-path keys, key validation logic changes significantly. Planner should note this needs clarification. |
| A3 | SandboxGate uses `drawdown_pct >= 0.10` as proxy for circuit breaker HALTED trips | Architecture Patterns, Pattern 5 | Circuit breaker state is not persisted to `sandbox_metrics` table. This proxy may miss trips. The planner may want to use `errors_caught` instead, or store CB state explicitly. |

---

## Open Questions

1. **preset_overrides key format**
   - What we know: `dict[str, Any]` with no schema constraint
   - What's unclear: flat dot-path keys vs. nested dict mirroring YAML hierarchy
   - Recommendation: Implement nested dict support (matches YAML structure naturally). Planner should add a task to define the convention and validate with a test.

2. **SandboxGate: async or sync interface?**
   - What we know: Called from `POST /experiments/{id}/apply` (async FastAPI) and potentially from TradingLoop (sync)
   - What's unclear: Whether SandboxGate needs to support both call sites
   - Recommendation: Implement `async check(session)` — the REST endpoint is the primary caller for Phase 38. TradingLoop integration is deferred.

3. **Circuit breaker reference in PresetApplicator**
   - What we know: `CircuitBreaker` is instantiated per-market and held in `TradingLoop._circuit_breakers: dict[str, CircuitBreaker]`
   - What's unclear: Which market's circuit breaker to check, and how to pass it to `PresetApplicator` in the REST endpoint context
   - Recommendation: Accept `circuit_breakers: dict[str, CircuitBreaker]` in constructor, check ALL market breakers (any non-NORMAL blocks apply). This is consistent with the TradingLoop's `_cross_market_breaker` philosophy.

---

## Environment Availability

Step 2.6: SKIPPED (no external dependencies beyond existing project stack — all imports verified present in adjacent modules).

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (project standard) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/unit/test_preset_applicator.py tests/unit/test_api_experiments.py -x` |
| Full suite command | `uv run pytest tests/unit/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| APPLY-01 | `PresetApplicator.apply()` writes YAML atomically with backup | unit | `uv run pytest tests/unit/test_preset_applicator.py::TestPresetApplicator::test_apply_writes_yaml_atomically -x` | No — Wave 0 |
| APPLY-01 | Backup file created with ISO-timestamp suffix | unit | `uv run pytest tests/unit/test_preset_applicator.py::TestPresetApplicator::test_apply_creates_backup -x` | No — Wave 0 |
| APPLY-02 | Non-NORMAL circuit level blocks apply | unit | `uv run pytest tests/unit/test_preset_applicator.py::TestPresetApplicator::test_apply_blocked_by_circuit_breaker -x` | No — Wave 0 |
| APPLY-03 | `_entry_strategy` set on fill, cleared on close | unit | `uv run pytest tests/unit/core/test_trading_loop.py -k entry_strategy -x` | Partially (file exists) |
| APPLY-04 | `invalidate_segment_cache()` is no-op but callable | unit | `uv run pytest tests/unit/test_combiner.py -k invalidate -x` | Partially (file exists) |
| APPLY-05 | INCONCLUSIVE routes to Telegram alert, no YAML write | unit | `uv run pytest tests/unit/test_preset_applicator.py::TestPresetApplicator::test_inconclusive_sends_telegram -x` | No — Wave 0 |
| APPLY-06 | SandboxGate blocks if < 3 trading days with fill_rate > 0 | unit | `uv run pytest tests/unit/test_preset_applicator.py::TestSandboxGate -x` | No — Wave 0 |
| APPLY-01+02 | `POST /experiments/{id}/apply` returns 200 with applied=True on happy path | unit | `uv run pytest tests/unit/test_api_experiments.py -k apply -x` | Partially (file exists, no apply tests) |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_preset_applicator.py tests/unit/test_api_experiments.py -x`
- **Per wave merge:** `uv run pytest tests/unit/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_preset_applicator.py` — covers APPLY-01, APPLY-02, APPLY-05, APPLY-06
- [ ] Test stubs in `tests/unit/test_api_experiments.py` — covers POST /{id}/apply endpoint

*(Existing files: `tests/unit/core/test_trading_loop.py`, `tests/unit/test_combiner.py` — extend with new test methods, no new file needed)*

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | yes | `api_key_auth` dependency on `POST /apply` endpoint (already on all experiments routes) |
| V5 Input Validation | yes | Key validation in `PresetApplicator`: reject unknown keys, validate value types against current YAML |
| V6 Cryptography | no | — |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via `segment_id` | Tampering | Validate `segment_id` matches preset filename pattern `[a-z_]+.yaml`; use `Path(presets_dir / f"{segment_id}.yaml").resolve()` and assert it is within `_PRESETS_DIR` |
| Arbitrary value injection in YAML | Tampering | `yaml.safe_load()` already prevents arbitrary Python object deserialization; key validation blocks unknown keys |
| YAML injection via preset_overrides | Tampering | `yaml.dump()` escapes special characters by default; `yaml.safe_load()` on read prevents code execution |

---

## Sources

### Primary (HIGH confidence)
- `src/finalayze/strategies/combiner.py` — `_load_config()` implementation, no cache confirmed
- `src/finalayze/risk/circuit_breaker.py` — `CircuitLevel` enum, `level` property
- `src/finalayze/orchestration/trading_loop.py` — `_entry_prices`, `_stop_states` dict patterns
- `src/finalayze/core/experiment_manager.py` — `ExperimentState.preset_overrides` field, `read_experiment()` API
- `src/finalayze/core/models.py` — `SandboxMetricRow` schema
- `src/finalayze/monitoring/go_no_go.py` — async DB query pattern for sandbox metrics
- `src/finalayze/api/alerts.py` — `TelegramAlerter.send_alert()` signature and thread-safety
- `src/finalayze/api/v1/experiments.py` — existing router, response model patterns
- `src/finalayze/backtest/iteration_tracker.py` — atomic write pattern
- `src/finalayze/strategies/presets/us_tech.yaml` — preset YAML structure

### Secondary (MEDIUM confidence)
- `CLAUDE.md` — layer rules (Layer 5 cannot import Layer 6 at module level)
- `.planning/phases/38-presetapplicator-+-auto-apply-loop/38-CONTEXT.md` — locked decisions

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified present in adjacent modules
- Architecture: HIGH — all patterns verified against actual source code
- Pitfalls: HIGH — derived from actual code inspection, not heuristics

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (stable codebase; combiner/circuit_breaker are stable modules)
