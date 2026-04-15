# Phase 50: EventDriven Activation - Research

**Researched:** 2026-04-15
**Domain:** Strategy activation, combiner dedup logic, Redis TTL management
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **CBR dedup location:** Duplicate suppression happens in `StrategyCombiner`, not in a separate strategy. If `event_driven` + any other strategy both fire for the same ticker with the same `cbr_rate` event type in one cycle, zero the lower-weight signal.
- **Detection mechanism:** Via `event_types` field in `Signal.features` dict.
- **Suppression scope:** Same ticker + same cycle only — cross-ticker and cross-cycle signals are independent.
- **"Freeze" implementation:** Extend Redis TTL when market closes so last sentiment survives until next open; resume normal 30-min TTL at open.
- **Market hours source:** Use `MOEX_MARKET_SCHEDULE.is_market_open()` from `markets/schedule.py`.
- **Weight change:** `event_driven.weight: 0.10` → `0.15` in all 4 ru_* preset YAMLs.
- **No `cbr_calendar` strategy exists** — this is aspirational; dedup is future-proofing + same-article double-processing guard.

### Claude's Discretion

- Exact TTL extension duration during closed hours (recommended: `seconds_to_next_moex_open + 1800`).
- Whether to add dividend-specific dedup or only CBR (recommended: both since mechanism is identical).
- How to structure combiner deduplication (hook vs. inline post-collection check).

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVNT-01 | EventDrivenStrategy enabled on all ru_* segments with weight 0.15 | Four preset YAMLs to flip + weight change; signal persistence to `signals` table required for verifiable success criterion |
| EVNT-02 | CBR/dividend duplicate signal guard prevents double-weight with cbr_calendar strategy | Combiner post-collection dedup; requires `event_type` propagation from article→sentiment→Signal.features; schema gap identified |
| EVNT-03 | Sentiment decay respects market hours (freeze during MOEX close, resume on open) | Dynamic TTL computed from `MOEX_MARKET_SCHEDULE.next_open()` at `set_sentiment()` call site |
</phase_requirements>

## Summary

Phase 50 activates `EventDrivenStrategy` live on all four MOEX segments (`ru_blue_chips`, `ru_energy`, `ru_finance`, `ru_tech`) and adds two protective mechanisms: a CBR/dividend duplicate-signal guard in the combiner and a market-hours-aware sentiment decay freeze.

The codebase already has the necessary primitives — `EventDrivenStrategy`, `StrategyCombiner`, `MOEX_MARKET_SCHEDULE`, and `RedisCache.set_sentiment()` with a configurable TTL param — but three code gaps must be closed before enabling the strategy is safe. First, `Signal.features` is typed `dict[str, float]` and cannot store event-type strings; a schema extension or numeric encoding is required for the dedup to work. Second, `StrategyCombiner.generate_signal()` does not thread `credibility` through to `EventDrivenStrategy.generate_signal()` (the base class signature omits it). Third, no code path currently writes generated signals to the `signals` PostgreSQL table, so the EVNT-01 success criterion — "at least one EventDrivenStrategy signal entry in the `signals` table" — cannot be satisfied without adding signal persistence to `_process_instrument()`.

**Primary recommendation:** Implement the three items above as Wave 0 prep tasks before enabling the presets. All three are small, focused changes that do not affect existing live strategies.

## Standard Stack

### Core (all already in-use)

| Component | Version/Location | Purpose | Notes |
|-----------|-----------------|---------|-------|
| `EventDrivenStrategy` | `src/finalayze/strategies/event_driven.py` | Generates signals from sentiment | Already functional; credibility param exists but is not threaded from combiner |
| `StrategyCombiner` | `src/finalayze/strategies/combiner.py` | Weighted combination of strategy signals | Synchronous loop; no dedup currently |
| `RedisCache` | `src/finalayze/data/cache.py` | Sentiment TTL cache | `set_sentiment(segment, score, ttl=int)` already accepts variable TTL |
| `MOEX_MARKET_SCHEDULE` | `src/finalayze/markets/schedule.py` | Trading hours + `next_open()` | `is_market_open()` and `next_open()` are available |
| `Signal` schema | `src/finalayze/core/schemas.py` | Pydantic model for signals | `features: dict[str, float]` — string values not allowed |
| `SignalModel` | `src/finalayze/core/models.py` | SQLAlchemy ORM for `signals` table | Table exists; no write path in trading loop yet |
| `get_async_session_factory` | `src/finalayze/core/db.py` | Async SQLAlchemy session factory | Used by FastAPI; needs wiring into trading loop for signal persistence |

[VERIFIED: codebase grep + direct file inspection]

### ru_* Preset YAMLs

| File | Current event_driven state | Required change |
|------|---------------------------|-----------------|
| `ru_blue_chips.yaml` | `enabled: false, weight: 0.10` | `enabled: true, weight: 0.15` |
| `ru_energy.yaml` | `enabled: false, weight: 0.10` | `enabled: true, weight: 0.15` |
| `ru_finance.yaml` | `enabled: false, weight: 0.10` | `enabled: true, weight: 0.15` |
| `ru_tech.yaml` | `enabled: false, weight: 0.10` | `enabled: true, weight: 0.15` |

[VERIFIED: direct file inspection]

**Weight budget impact:** All four segments use `normalize_mode: "total"`. The combiner normalizes net score by the sum of all enabled strategy weights, so weights do not need to sum to 1.0. Changing `event_driven` from `0.10` to `0.15` increases total enabled weight by 0.05 per segment — this has no correctness impact, only a slight reduction of event_driven's relative contribution in the normalized score.

[VERIFIED: combiner.py lines 142-145 — denominator is `total_enabled_weight` when `normalize_mode == "total"`]

## Architecture Patterns

### Recommended Project Structure (no new modules)

All three changes land in existing files:

```
src/finalayze/
├── core/
│   ├── schemas.py          # Add event_type encoding or extend features type
│   └── trading_loop.py     # Add credibility cache read + signal persistence
├── data/
│   └── cache.py            # Dynamic TTL logic in set_sentiment()
└── strategies/
    ├── base.py             # Add credibility kwarg to abstract generate_signal
    ├── combiner.py         # Thread credibility; add post-collection dedup
    ├── event_driven.py     # (no change needed — already accepts credibility)
    └── presets/
        ├── ru_blue_chips.yaml   # enabled + weight
        ├── ru_energy.yaml
        ├── ru_finance.yaml
        └── ru_tech.yaml
```

### Pattern 1: Dynamic Sentiment TTL (EVNT-03)

**What:** When `_process_news_article()` writes to the Redis sentiment cache, compute TTL based on whether MOEX is currently open.

**When to use:** Every call to `cache.set_sentiment()` that originates from the news cycle.

**Implementation:**

```python
# Source: CONTEXT.md + schedule.py analysis
from finalayze.markets.schedule import MOEX_MARKET_SCHEDULE

_SENTIMENT_TTL_SECONDS = 1800  # 30 min — market open
_SENTIMENT_TTL_BUFFER = 1800   # 30 min extra buffer beyond next open

def _compute_sentiment_ttl(now: datetime) -> int:
    """Return TTL in seconds. Extended when MOEX is closed."""
    if MOEX_MARKET_SCHEDULE.is_market_open(now):
        return _SENTIMENT_TTL_SECONDS
    next_open = MOEX_MARKET_SCHEDULE.next_open(now)
    seconds_to_open = int((next_open - now).total_seconds())
    return seconds_to_open + _SENTIMENT_TTL_BUFFER
```

The call site in `_process_news_article()` already calls `self._cache.set_sentiment(segment_id, score)` — change to pass the computed TTL: `self._cache.set_sentiment(segment_id, score, ttl=_compute_sentiment_ttl(now))`.

**Math verification for criterion 3 (±10% tolerance):**

If last session's final score was `S = 0.70` and TTL is extended so it survives overnight, the next-morning article applies the EMA update `new = 0.7 * S + 0.3 * new_sentiment`. For `new_sentiment ∈ [0.5, 0.9]`, `new ∈ [0.64, 0.76]` — all within ±10% of `0.70`. Without freeze, `S` decays to `0.0` and `new = 0.3 * new_sentiment ∈ [0.15, 0.27]`, which is far outside the ±10% band.

[VERIFIED: trading_loop.py lines 421-433; schedule.py lines 53-84]

### Pattern 2: CBR/Dividend Dedup (EVNT-02)

**What:** After the combiner collects all sub-signals, scan for pairs where both signals share a `cbr_rate` or `dividend` event type and target the same ticker. Zero out the lower-weight signal before computing the weighted sum.

**Schema constraint (CRITICAL):** `Signal.features` is `dict[str, float]`. String event types cannot be stored directly.

**Recommended solution — numeric encoding:** Add a convention that `event_driven` signals encode their primary event type as a float key:

```python
# In EventDrivenStrategy.generate_signal()
# event_type_code: cbr_rate=1.0, dividend=2.0, 0.0=other
EVENT_TYPE_CODES: dict[str, float] = {"cbr_rate": 1.0, "dividend": 2.0}

features = {
    "sentiment": sentiment_score,
    "credibility": credibility,
    "event_type_code": EVENT_TYPE_CODES.get(primary_event_type, 0.0),
}
```

The combiner dedup then reads `signal.features.get("event_type_code", 0.0)` and suppresses the lower-weight signal when two signals share the same non-zero code on the same ticker.

**Alternative if schema is relaxed:** Change `Signal.features` to `dict[str, float | str]` — but this is a larger schema change affecting validators, DB model (JSONB already supports mixed types), and all existing tests that check `features` types.

[VERIFIED: schemas.py line 68; event_driven.py lines 128-138]

### Pattern 3: Credibility Threading (prerequisite for EVNT-01)

**What:** `StrategyCombiner.generate_signal()` calls `strategy.generate_signal()` without `credibility`. `EventDrivenStrategy.generate_signal()` defaults to `credibility=1.0`, meaning source quality is ignored.

**Fix — two-step:**

1. Add optional `credibility: float = 1.0` to `BaseStrategy.generate_signal()` abstract signature.
2. `StrategyCombiner.generate_signal()` accepts `credibility: float = 1.0` and passes it through only to strategies that are named `event_driven` (or more cleanly: via `**kwargs` pattern for strategy-specific params).

**Simplest safe approach** (no base class change): In the combiner's inner loop, check `strategy.name == "event_driven"` and pass `credibility=credibility` via keyword arg. `BaseStrategy` contract is not violated since extra kwargs are allowed in Python.

The `credibility` value flows from `_news_cycle` → `article.credibility_score` → Redis credibility cache → `_process_instrument` → `StrategyCombiner.generate_signal(credibility=...)`.

**Credibility cache:** Add `set_credibility(segment, score)` / `get_credibility(segment)` to `RedisCache`, or store alongside sentiment as `credibility:{segment}` key with same TTL logic.

[VERIFIED: base.py lines 21-28; combiner.py lines 124-126; event_driven.py lines 86-87]

### Pattern 4: Signal Persistence (prerequisite for EVNT-01 success criterion)

**What:** The `signals` PostgreSQL table exists but is never written to during the live trading cycle. EVNT-01 success criterion requires "at least one EventDrivenStrategy signal entry in the `signals` table".

**Recommended approach:** Add a `_persist_signal()` async helper in `trading_loop.py` that writes the combined signal (with `event_driven_confidence > 0` in features) to the `signals` table. Call it from `_process_instrument()` after a non-None signal is generated.

```python
# In trading_loop._process_instrument() after signal check
async def _persist_signal(self, signal: Signal, mode: str) -> None:
    factory = get_async_session_factory()
    async with factory() as session:
        row = SignalModel(
            id=uuid.uuid4(),
            strategy_name=signal.strategy_name,  # "combined"
            symbol=signal.symbol,
            market_id=signal.market_id,
            segment_id=signal.segment_id,
            direction=signal.direction.value,
            confidence=Decimal(str(signal.confidence)),
            features=dict(signal.features),
            reasoning=signal.reasoning,
            created_at=datetime.now(UTC),
            mode=mode,
        )
        session.add(row)
        await session.commit()
```

This is called via `self._run_async(self._persist_signal(signal, self._settings.mode.value))`.

**Scope constraint:** Persist only when `event_driven_confidence` is present in the combined signal's features (i.e., EventDriven fired and contributed to the combination). This satisfies the EVNT-01 observable check without adding overhead on every single signal.

[VERIFIED: models.py lines 122-149; db.py lines 41-69; trading_loop.py lines 218-230 (_run_async pattern)]

### Anti-Patterns to Avoid

- **Enabling presets before fixing credibility threading:** EventDrivenStrategy will silently use `credibility=1.0` (maximum), inflating confidence scores for low-credibility sources. Wire credibility first.
- **Zeroing dedup at signal collection time (before weight calculation):** The dedup must happen post-collection but pre-aggregation so the weight of the zeroed signal is excluded from `total_weight`. Otherwise the normalized score drops incorrectly.
- **Using `redis.expireat()` for freeze:** Unnecessary complexity. Computing TTL at `set_sentiment()` call time is simpler and stateless — no separate "close event" hook needed.
- **Changing MOEX close time:** `schedule.py` shows `close_time=time(18, 40)` (18:40 MSK). The phase success criterion says "18:50 MSK" — but `is_market_open()` returns `False` at 18:40 and beyond. The 10-minute discrepancy is irrelevant to the implementation; the schedule's actual close time is the correct boundary.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Next-open time calculation | Custom datetime arithmetic | `MOEX_MARKET_SCHEDULE.next_open()` | Already handles weekends and DST |
| Redis TTL set | Custom `EXPIREAT` / periodic refresh job | `RedisCache.set_sentiment(ttl=N)` | Already accepts variable TTL; no new Redis commands needed |
| Event type storage in features | New dict type / separate class | Numeric encoding (`event_type_code: float`) | Keeps `dict[str, float]` contract; no schema migration needed |
| DB session management | Manual session open/close | `get_async_session_factory()` pattern from `core/db.py` | Already handles pool reuse, rollback on error |

## Common Pitfalls

### Pitfall 1: event_type_code Not Set — Dedup Never Triggers

**What goes wrong:** EventDrivenStrategy generates a signal without `event_type_code` in features. Combiner dedup checks for `event_type_code > 0` and finds nothing. Two CBR signals from different strategies accumulate without suppression.

**Why it happens:** The event type flows through `EventClassifier` → `ImpactEstimator` → sentiment cache update, but by the time `EventDrivenStrategy.generate_signal()` is called (during the separate strategy cycle), the event type is no longer available — only the cached `sentiment_score` is passed.

**How to avoid:** The event type must be cached alongside the sentiment score. Store the most recent event type code per segment in Redis (`event_type:{segment}` key with same TTL logic). Read it in the combiner's generate_signal and pass to EventDrivenStrategy, which embeds it in features.

**Warning signs:** Test for EVNT-02 passes even when two strategies fire CBR signals — dedup logic silently skips due to missing `event_type_code`.

### Pitfall 2: BaseStrategy Abstract Method Signature Mismatch

**What goes wrong:** If `credibility` kwarg is added to `StrategyCombiner.generate_signal()` and passed directly to `strategy.generate_signal(credibility=...)`, all non-EventDriven strategies raise `TypeError: unexpected keyword argument 'credibility'`.

**Why it happens:** `BaseStrategy.generate_signal()` does not include `credibility` in its abstract signature. Other concrete strategies (`MomentumStrategy`, `MeanReversionStrategy`, etc.) override the abstract method without `credibility`.

**How to avoid:** Either (a) add `credibility: float = 1.0` to all concrete `generate_signal` overrides, or (b) only pass `credibility` when `strategy.name == "event_driven"` (targeted injection pattern, no base class change).

**Warning signs:** `TypeError` in `_strategy_cycle_impl` after adding credibility threading.

### Pitfall 3: TTL Freeze Causes Stale Sentiment on Re-Enable

**What goes wrong:** Sentiment cached before a weekend (Friday market close) has an extended TTL of ~64 hours (Friday 18:40 MSK → Monday 10:00 MSK + 30 min buffer). If a geopolitical shock occurs over the weekend, the cached pre-shock sentiment is used for the first cycle Monday morning.

**Why it happens:** The freeze intent is to preserve Friday's sentiment so Monday's first article updates from the same baseline. But this also preserves any sentiment state that may be stale.

**How to avoid:** This is an acceptable tradeoff per the success criterion (±10% tolerance). The first article's EMA update will move the score towards the new reality within one cycle. No mitigation needed for Phase 50.

**Warning signs:** Unexpectedly confident signals on Monday morning when weekend news is strongly negative.

### Pitfall 4: Signal Persistence Breaks in Sandbox Mode

**What goes wrong:** `_persist_signal()` uses `self._settings.mode.value` to write `mode` column. In sandbox mode, the DB may not be running or the signals table may not exist (e.g., in a test environment without a DB migration applied).

**Why it happens:** The success criterion specifically says "sandbox strategy cycle" — the DB must be up.

**How to avoid:** Wrap `_persist_signal()` in a try/except that logs failure but does not raise. Signal persistence must be best-effort — a DB failure should not abort the trading cycle.

### Pitfall 5: Combiner Weight Budget Misunderstanding

**What goes wrong:** Developer thinks weights must sum to 1.0 and adjusts other strategy weights down to compensate for event_driven going 0.10 → 0.15.

**Why it happens:** Misunderstanding of `normalize_mode: "total"` semantics.

**How to avoid:** `normalize_mode: "total"` normalizes by the sum of ALL enabled strategy weights. No weight redistribution is needed when increasing event_driven from 0.10 to 0.15. Only the raw weights in the YAML need updating.

[VERIFIED: combiner.py lines 99-112, 142-145]

## Code Examples

### Dynamic TTL Calculation

```python
# Source: schedule.py next_open() + cache.py set_sentiment() analysis
_SENTIMENT_TTL_OPEN_SECONDS = 1800    # 30 min during market hours
_SENTIMENT_TTL_BUFFER_SECONDS = 1800  # 30 min buffer added to closed-hours TTL

def _compute_sentiment_ttl(now: datetime) -> int:
    """Return Redis TTL in seconds, extended when MOEX is closed."""
    if MOEX_MARKET_SCHEDULE.is_market_open(now):
        return _SENTIMENT_TTL_OPEN_SECONDS
    next_open: datetime = MOEX_MARKET_SCHEDULE.next_open(now)
    seconds_to_open = int((next_open - now).total_seconds())
    return max(seconds_to_open + _SENTIMENT_TTL_BUFFER_SECONDS, _SENTIMENT_TTL_OPEN_SECONDS)
```

### Post-Collection Dedup in StrategyCombiner

```python
# Source: combiner.py analysis + CONTEXT.md decision
_CBR_DIVIDEND_CODES = {1.0, 2.0}  # cbr_rate=1.0, dividend=2.0

def _dedup_cbr_dividend(
    collected: dict[str, tuple[Signal, Decimal]]  # strategy_name -> (signal, weight)
) -> dict[str, tuple[Signal, Decimal]]:
    """Zero lower-weight signal when two strategies share CBR/dividend event type."""
    # Group by (symbol, event_type_code) for non-zero codes
    groups: dict[tuple[str, float], list[str]] = {}
    for name, (sig, _) in collected.items():
        code = sig.features.get("event_type_code", 0.0)
        if code in _CBR_DIVIDEND_CODES:
            key = (sig.symbol, code)
            groups.setdefault(key, []).append(name)

    result = dict(collected)
    for names in groups.values():
        if len(names) < 2:  # noqa: PLR2004
            continue
        # Keep highest-weight signal, zero the rest
        sorted_by_weight = sorted(names, key=lambda n: collected[n][1], reverse=True)
        for name in sorted_by_weight[1:]:
            sig, weight = result[name]
            result[name] = (sig, Decimal("0"))
    return result
```

### Preset YAML Change (same for all four ru_* files)

```yaml
# Before:
  event_driven:
    enabled: false
    weight: 0.10

# After:
  event_driven:
    enabled: true
    weight: 0.15
```

### Signal Persistence in _process_instrument

```python
# Source: models.py + db.py analysis — call after signal confirmed non-None
# Only persist when event_driven contributed (check features key)
if signal and "event_driven_confidence" in signal.features:
    try:
        self._run_async(self._persist_signal(signal, self._settings.mode.value))
    except Exception:
        _log.debug("signal_persistence_failed", symbol=symbol)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `event_driven` disabled in ru_* presets | Enable + weight=0.15 | Phase 50 | Live signals on MOEX segments |
| Sentiment TTL = 30-min binary expiry regardless of market hours | Dynamic TTL extending through closed hours | Phase 50 | First-of-day signal preserves prior session baseline |
| No CBR/dividend dedup in combiner | Post-collection dedup by event_type_code | Phase 50 | Prevents double-weight when same event triggers multiple strategies |
| No signal persistence in live trading loop | Write combined signals to `signals` table | Phase 50 | EVNT-01 success criterion becomes verifiable |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The EVNT-01 success criterion's "signals table" refers to the PostgreSQL `signals` table, not an in-memory Prometheus counter | Architecture Patterns § Signal Persistence | If it means Prometheus, signal persistence is unnecessary and that task can be dropped |
| A2 | Event type must be cached in Redis (separate from sentiment score) to be available in the strategy cycle | Common Pitfalls § Pitfall 1 | If event type is somehow already available at strategy cycle time (e.g., via a different path), the Redis event-type cache is redundant |
| A3 | `normalize_mode: "total"` means weight budget does not need rebalancing | Standard Stack § Weight budget | Verified in combiner.py but if another preset has `normalize_mode: "firing"` and weights must sum to 1.0, redistribution needed |

**A3 is LOW risk:** all four ru_* presets explicitly set `normalize_mode: "total"`. [VERIFIED: direct file inspection]

## Open Questions (RESOLVED)

1. **Signal persistence scope**
   - RESOLVED: `_persist_signal_async()` already exists and is called at `trading_loop.py:1677` — persist ALL combined signals. EVNT-01 is satisfied as a subset.

2. **Event type code source**
   - RESOLVED: Inject `event_type_code` write in `_apply_impact_result()` alongside the sentiment cache write loop, per Plan 50-01 Task 1.

## Environment Availability

Step 2.6: SKIPPED (no external dependencies beyond already-running services — Redis, PostgreSQL. These are required by existing Phase 49 work and are assumed operational for Phase 50.)

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest with pytest-asyncio |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/unit/ -q --no-header` |
| Full suite command | `uv run pytest tests/unit/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVNT-01a | ru_* presets have `enabled: true, weight: 0.15` | unit | `uv run pytest tests/unit/test_event_driven_strategy.py -k "ru_" -x` | ❌ Wave 0 |
| EVNT-01b | `StrategyCombiner.generate_signal()` passes credibility to EventDrivenStrategy | unit | `uv run pytest tests/unit/test_strategy_combiner.py -k "credibility" -x` | ❌ Wave 0 |
| EVNT-01c | Combined signal with event_driven contribution is persisted to `signals` table | unit (mock DB) | `uv run pytest tests/unit/test_signal_persistence.py -x` | ❌ Wave 0 |
| EVNT-02 | Combiner dedup zeroes lower-weight signal when two strategies share cbr_rate/dividend event type | unit | `uv run pytest tests/unit/test_strategy_combiner.py -k "dedup" -x` | ❌ Wave 0 |
| EVNT-03 | `set_sentiment()` uses extended TTL when MOEX is closed | unit | `uv run pytest tests/unit/test_redis_cache.py -k "ttl_freeze" -x` | ❌ Wave 0 |
| EVNT-03b | `set_sentiment()` uses normal 1800s TTL when MOEX is open | unit | included in EVNT-03 test file | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/unit/test_strategy_combiner.py tests/unit/test_event_driven_strategy.py tests/unit/test_redis_cache.py -q`
- **Per wave merge:** `uv run pytest tests/unit/ -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/unit/test_event_driven_strategy.py` — add tests for ru_* segment support + credibility scaling at segment level
- [ ] `tests/unit/test_strategy_combiner.py` — add dedup tests (existing file; add test class `TestCombinerDedup`)
- [ ] `tests/unit/test_redis_cache.py` — add `TestSentimentCacheTTLFreeze` class (existing file)
- [ ] `tests/unit/test_signal_persistence.py` — new file covering `_persist_signal()` in trading_loop (mock DB session)

## Security Domain

Security enforcement applies. This phase introduces no new authentication, session management, or cryptography surfaces. Applicable ASVS categories:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Pydantic v2 validates `Signal.confidence` in [0,1]; `Signal.features` schema enforces `dict[str, float]` — prevents injection via event_type field |
| V6 Cryptography | no | — |

**Threat pattern:** The `event_type_code` float encoding (cbr_rate=1.0, dividend=2.0) prevents string injection through `Signal.features` since the type constraint rejects non-float values. No additional validation needed.

[VERIFIED: schemas.py field_validator on confidence; Pydantic v2 type coercion]

## Sources

### Primary (HIGH confidence)

- `src/finalayze/strategies/event_driven.py` — EventDrivenStrategy implementation, credibility param, features dict
- `src/finalayze/strategies/combiner.py` — StrategyCombiner synchronous collection loop, normalization modes
- `src/finalayze/data/cache.py` — RedisCache, `set_sentiment(ttl=)` signature
- `src/finalayze/markets/schedule.py` — MOEX_MARKET_SCHEDULE, `is_market_open()`, `next_open()`
- `src/finalayze/core/schemas.py` — Signal schema, `features: dict[str, float]` constraint
- `src/finalayze/core/models.py` — SignalModel ORM, `signals` table definition
- `src/finalayze/core/trading_loop.py` — `_process_news_article()`, `_process_instrument()`, `_run_async()` pattern
- `src/finalayze/core/db.py` — `get_async_session_factory()` pattern for DB access
- `src/finalayze/analysis/event_classifier.py` — EventType enum, `cbr_rate` and `dividend` codes
- `src/finalayze/analysis/impact_estimator.py` — event routing (does NOT carry event_type to Redis)
- `src/finalayze/strategies/presets/ru_*.yaml` — all four presets, current `event_driven` state
- `.planning/phases/50-eventdriven-activation/50-CONTEXT.md` — locked decisions and integration points

### Secondary (MEDIUM confidence)

- Test files (`test_event_driven_strategy.py`, `test_strategy_combiner.py`, `test_redis_cache.py`) — confirmed baseline: 1559 tests pass at 87% coverage

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all components directly inspected in codebase
- Architecture: HIGH — all integration points verified against source code
- Pitfalls: HIGH — pitfalls derived from concrete code gaps found during inspection, not from training assumptions

**Research date:** 2026-04-15
**Valid until:** 2026-05-15 (stable codebase, no fast-moving dependencies)
