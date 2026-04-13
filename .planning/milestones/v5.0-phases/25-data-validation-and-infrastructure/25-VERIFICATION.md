---
phase: 25-data-validation-and-infrastructure
verified: 2026-03-24T07:40:54Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 25: Data Validation and Infrastructure Verification Report

**Phase Goal:** Market data is validated before strategy consumption, stale data is detected and skipped, and data fetching is efficient with no redundant connections or downloads
**Verified:** 2026-03-24T07:40:54Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | DataNormalizer.validate() runs on every batch of fetched candles before they reach strategy processing — candles with negative prices, low > high, or zero volume are rejected | VERIFIED | `trading_loop.py:1388-1393` — `DataNormalizer(market_id, source="live").normalize_batch(candles)` called before any processing; empty result triggers early return |
| 2 | When candle data is older than 2x the expected timeframe interval, a warning is logged and the instrument is skipped | VERIFIED | `trading_loop.py:1395-1403` — `_is_candle_stale(candles[-1].timestamp, _STALENESS_THRESHOLD_HOURS)` called after normalization; `candle_data_stale` warning logged, instrument returns early |
| 3 | IMOEX index candles store share volume (column index 5) not turnover value (column index 4) | VERIFIED | `moex_iss.py:250-267` — `turnover_val = row[4]` (comment: not used), `share_volume = row[5]`, `volume=int(share_volume) if share_volume else 0` |
| 4 | TinkoffFetcher maintains a persistent gRPC channel that is reused across calls within the same session | VERIFIED | `tinkoff_data.py:84-162` — `_loop`, `_loop_thread`, `_loop_init_lock`, `_run_async()`, `_get_services_async()` all present; every fetch method uses `self._run_async(...)` instead of `asyncio.run(...)` |
| 5 | Brent crude candles are cached via _cached_fetch() in MarketDataLoader | VERIFIED | `loader.py:116-125` — `self._cached_fetch("yfinance.brent", self._brent_cache, Candle, "yfinance", "BZ_F", start, end, fn=...)` replaces previous `_safe_fetch` call |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/data/fetchers/moex_iss.py` | Corrected IMOEX volume column | VERIFIED | `share_volume = row[5]`, `volume=int(share_volume) if share_volume else 0` at line 267 |
| `src/finalayze/orchestration/trading_loop.py` | DataNormalizer wiring and staleness check | VERIFIED | `normalize_batch` at line 1390; `_is_candle_stale` at line 1396; `_STALENESS_THRESHOLD_HOURS = 48.0` at line 81 |
| `tests/unit/test_data_validation.py` | Tests for all three data validation requirements | VERIFIED | 272 lines (exceeds 60-line minimum); 7 tests — all pass |
| `src/finalayze/data/fetchers/tinkoff_data.py` | Persistent gRPC channel via background event loop | VERIFIED | `_run_async` present at line 98; all fetch methods use `_run_async`; no `asyncio.run()` calls |
| `src/finalayze/data/loader.py` | Brent caching via _cached_fetch | VERIFIED | `_cached_fetch` for Brent at lines 116-125; `brent_cache` parameter in `__init__` at line 54 |
| `tests/unit/test_tinkoff_persistent_client.py` | Tests for persistent channel reuse | VERIFIED | 236 lines (exceeds 30-line minimum); 15 tests — all pass |
| `tests/unit/test_market_data_loader.py` | Test for Brent caching | VERIFIED | 331 lines (exceeds 10-line minimum); `TestBrentCaching::test_brent_uses_cached_fetch` and `test_second_call_uses_cache` — both pass |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `trading_loop.py` | `data/normalizer.py` | `DataNormalizer.normalize_batch()` in `_process_instrument` | WIRED | `from finalayze.data.normalizer import DataNormalizer` at line 36; called at line 1390 before `generate_signal` at line 1428 |
| `trading_loop.py` | `_is_candle_stale()` | Staleness check before `generate_signal` | WIRED | `_is_candle_stale` called at line 1396, returns early at 1403 — `generate_signal` not reached on stale data |
| `tinkoff_data.py` | `t_tech.invest.AsyncClient` | Persistent background event loop with `_run_async` | WIRED | `_get_services_async` creates `AsyncClient` once and caches in `self._services`; `_run_async` dispatches via `run_coroutine_threadsafe` |
| `loader.py` | `_cached_fetch` | Brent candles use `_cached_fetch` instead of `_safe_fetch` | WIRED | `_cached_fetch("yfinance.brent", self._brent_cache, ...)` at line 116; pattern confirmed at lines 116-125 |

---

### Data-Flow Trace (Level 4)

Not applicable. This phase implements infrastructure fixes (data validation gates, volume column fix, caching). There are no new data-rendering components requiring Level 4 trace.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| test_data_validation.py — 7 tests pass | `uv run pytest tests/unit/test_data_validation.py -v` | 7 passed | PASS |
| test_tinkoff_persistent_client.py — 15 tests pass | `uv run pytest tests/unit/test_tinkoff_persistent_client.py -v` | 15 passed, 2 warnings | PASS |
| test_market_data_loader.py — 11 tests pass including Brent cache tests | `uv run pytest tests/unit/test_market_data_loader.py -v` | 11 passed | PASS |
| IMOEX volume column: `share_volume = row[5]` used, not `row[4]` | Code inspection `moex_iss.py:250-267` | share_volume at row[5], volume=int(share_volume) | PASS |
| No `asyncio.run()` calls remain in tinkoff_data.py | `grep "asyncio.run(" tinkoff_data.py` | Only `asyncio.run_coroutine_threadsafe` (not bare `asyncio.run`) | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DATA-01 | 25-01-PLAN.md | DataNormalizer.validate() runs on fetched candles before strategy processing — rejects negative prices, low > high, zero volume | SATISFIED | `trading_loop.py:1388-1393` — `normalize_batch()` wired before `generate_signal()` |
| DATA-02 | 25-01-PLAN.md | Candle staleness detection active — configurable threshold (default: 2x timeframe), warning logged and instrument skipped | SATISFIED | `trading_loop.py:81,1395-1403` — `_STALENESS_THRESHOLD_HOURS=48.0`, `_is_candle_stale` check with warning log |
| DATA-03 | 25-01-PLAN.md | IMOEX index candles use share volume (row[5]), not turnover value (row[4]) | SATISFIED | `moex_iss.py:250-267` — `share_volume = row[5]`, used in `Candle.volume` |
| INFRA-01 | 25-02-PLAN.md | TinkoffFetcher reuses a persistent gRPC channel across calls (like TinkoffBroker pattern) — no per-call channel churn | SATISFIED | `tinkoff_data.py:84-162,184,281,322,399,464,520,582,638` — `_run_async` + `_get_services_async` on all fetch methods |
| INFRA-02 | 25-02-PLAN.md | Brent crude candles cached via _cached_fetch() in MarketDataLoader — not re-downloaded on every backtest | SATISFIED | `loader.py:54,62,116-125` — `brent_cache` parameter + `_cached_fetch` for Brent |

All 5 requirements declared in REQUIREMENTS.md as Complete for Phase 25 — confirmed by code evidence.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

No TODOs, FIXMEs, placeholders, stub implementations, or empty handlers found in any of the 5 modified production files.

---

### Human Verification Required

None. All success criteria are verifiable programmatically. The phase delivers infrastructure plumbing with clear code-level evidence and passing unit tests for every requirement.

---

### Gaps Summary

No gaps. All 5 success criteria from ROADMAP.md are satisfied:

1. `DataNormalizer.normalize_batch()` is called at `trading_loop.py:1390` before `generate_signal` at line 1428. Early return on empty result confirmed at line 1392.
2. `_is_candle_stale()` is called at line 1396 immediately after normalization; stale instruments log `candle_data_stale` and return before reaching strategy processing.
3. `moex_iss.py` uses `share_volume = row[5]` (not `row[4]`) for `Candle.volume` — the variable name itself (`turnover_val`) documents that `row[4]` is the incorrect turnover value.
4. `TinkoffFetcher` has `_loop`, `_loop_thread`, `_loop_init_lock`, `_run_async()`, and `_get_services_async()` — every public async method routes through `_run_async`. No bare `asyncio.run()` calls remain.
5. `MarketDataLoader._load_moex()` calls `_cached_fetch()` for Brent with `self._brent_cache`; the constructor accepts `brent_cache: GenericFileCache | None = None` so existing callers are unaffected.

---

_Verified: 2026-03-24T07:40:54Z_
_Verifier: Claude (gsd-verifier)_
