---
phase: 40-moex-data-adapter-macro-features
plan: "02"
subsystem: scripts/ml-research
tags: [moex, macro-features, look-ahead-bias, market-context, cbr, brent, usdrub, turnover]
dependency_graph:
  requires: ["40-01"]
  provides: ["MOEX macro data plumbing", "MarketContext wiring", "look-ahead bias test"]
  affects: ["scripts/auto_ml_research.py", "tests/unit/test_auto_ml_research_moex.py"]
tech_stack:
  added: []
  patterns:
    - "CBRFetcher (sync) for key_rate and USDRUB fx_rates"
    - "MoexISSFetcher (sync) for IMOEX turnover"
    - "YFinanceFetcher for BZ=F Brent crude (commodity, not MOEX ticker)"
    - "MoexMarketData flows through MarketContext into build_triple_barrier_dataset"
    - "_EXTERNAL_DATA_LAG_BARS=2 prevents look-ahead bias on all macro series"
key_files:
  modified:
    - scripts/auto_ml_research.py
    - tests/unit/test_auto_ml_research_moex.py
decisions:
  - "CBRFetcher is synchronous (not async) — used with context manager, no asyncio.run needed"
  - "MoexISSFetcher is synchronous — same pattern"
  - "Brent (BZ=F) fetched via YFinanceFetcher (commodity, not MOEX)"
  - "Lag is 2 bars (_EXTERNAL_DATA_LAG_BARS=2), not shift(1) as suggested in plan docstring"
  - "Random-walk candles required for non-zero ATR so triple-barrier labels are generated"
metrics:
  duration: "~7 minutes"
  completed: "2026-04-13T06:07:00Z"
  tasks_completed: 2
  files_modified: 2
---

# Phase 40 Plan 02: MOEX Macro Data Fetching and MarketContext Wiring Summary

MOEX macro data (CBR key rate, USDRUB FX, Brent crude, IMOEX turnover) fetched once and wired through MoexMarketData → MarketContext into the feature pipeline, with a 2-bar lag verified by unit test.

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Add MOEX macro data fetching and MarketContext wiring | `60e3f5d` | scripts/auto_ml_research.py |
| 2 | Add shift(2) look-ahead bias test and macro features non-zero test | `218ef8e` | tests/unit/test_auto_ml_research_moex.py, scripts/auto_ml_research.py |

## What Was Built

### `_fetch_moex_macro_data()` (scripts/auto_ml_research.py)

Fetches all four MOEX macro data sources once at script start:

1. **CBR key rate** — `CBRFetcher.fetch_key_rate(start, end)` (synchronous)
2. **USDRUB FX rates** — `CBRFetcher.fetch_fx_rates("USD", start, end)` (synchronous)
3. **IMOEX turnover** — `MoexISSFetcher.fetch_market_turnover(start, end)` (synchronous)
4. **Brent crude (BZ=F)** — `YFinanceFetcher.fetch_candles("BZ=F", ...)` (BZ=F is a commodity, not MOEX)

Returns `MoexMarketData | None`. Returns `None` if `FINALAYZE_TINKOFF_TOKEN` is not set (token required for turnover endpoint).

### MarketContext wiring

- `build_full_dataset()` gains `moex_data: MoexMarketData | None = None` parameter
- `MarketContext` construction includes `moex_data=moex_data`
- `_prepare_data()` accepts and forwards `moex_data` for MOEX segments only
- `run_research_loop()` fetches macro data once before `_prepare_data()`

### Tests (23 total, all passing)

- `TestMacroShift2NoLookahead.test_macro_shift2_no_lookahead`: Creates 200 FX rates with stable value 80 for first 198, then spike value 200 for last 2. Verifies `usdrub_zscore_60d < 3.0` for the last sample — the 2-bar lag hides the spike.
- `TestMoexMacroFeaturesNonZero.test_moex_macro_features_nonzero`: Creates realistic MoexMarketData with FX drift, 10 key rate records, Brent candles, and turnover. Verifies ≥3 MOEX feature keys present with at least one non-zero value.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Guard ZeroDivisionError in build_full_dataset print**
- **Found during:** Task 2 test execution
- **Issue:** `sum(labels) / len(labels)` in the print statement raised `ZeroDivisionError` when label list was empty (e.g., constant-price candles with ATR=0)
- **Fix:** Replaced inline division with guarded `pos_pct = n_pos / n_total if n_total > 0 else 0.0`
- **Files modified:** scripts/auto_ml_research.py
- **Commit:** `218ef8e`

**2. [Rule 1 - Bug] Flat candles produce no labels (ATR=0)**
- **Found during:** Task 2 test execution
- **Issue:** Constant-price test candles (open=100, high=105, low=95, close=100 every day) produce ATR≈0, causing all triple-barrier labels to be skipped → empty features list
- **Fix:** Replaced `_make_candles` with a seeded random-walk implementation (`random.gauss(0, 0.02)` daily vol). Used `_RW_SEED=42` for reproducibility.
- **Files modified:** tests/unit/test_auto_ml_research_moex.py
- **Commit:** `218ef8e`

**3. [Plan Correction] API signatures differ from plan spec**
- **Issue:** Plan specified `CBRFetcher` as async with `fetch_key_rate()` taking no args and `fetch_fx_rates(from_date, to_date)`. Actual CBRFetcher is synchronous; `fetch_key_rate(start, end)` and `fetch_fx_rates(currency, start, end)` take `datetime` objects.
- **Fix:** Used actual synchronous API with context manager pattern. No `asyncio.run` needed.
- **Impact:** Simpler implementation — no nested async functions required.

**4. [Plan Correction] Lag is 2 bars, not shift(1)**
- **Issue:** Plan docstring says "shift(1) aligned" but actual `_EXTERNAL_DATA_LAG_BARS=2`. Test name updated to `test_macro_shift2_no_lookahead` to match reality.

## Known Stubs

None — all macro data sources are wired to real fetchers. Feature pipeline is fully connected. No placeholder data.

## Threat Flags

No new network endpoints or trust boundaries introduced beyond what was in the plan's threat model (T-40-04, T-40-05, T-40-06). The graceful failure pattern (each fetch wrapped in try/except returning empty data) implements T-40-06 mitigation.

## Self-Check: PASSED

Files exist:
- scripts/auto_ml_research.py — contains `_fetch_moex_macro_data`, `moex_data` parameter in `build_full_dataset`
- tests/unit/test_auto_ml_research_moex.py — contains `test_macro_shift2_no_lookahead` and `test_moex_macro_features_nonzero`

Commits exist:
- `60e3f5d` — feat(40-02): add MOEX macro data fetching and MarketContext wiring
- `218ef8e` — feat(40-02): add shift(2) look-ahead bias test and macro features non-zero test

23 tests pass.
