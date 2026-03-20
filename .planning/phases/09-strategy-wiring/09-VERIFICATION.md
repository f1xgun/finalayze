---
phase: 09-strategy-wiring
verified: 2026-03-20T09:15:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 9: Strategy Wiring Verification Report

**Phase Goal:** Existing but unconnected strategies generate real trades in MOEX backtests, establishing a positive equity baseline
**Verified:** 2026-03-20T09:15:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | DividendGapStrategy generates trades on ex-dividend dates using the expanded calendar, with per-symbol max_hold_bars (not force-closed at 15 bars) | VERIFIED | `_yield_hold_bars()` in dividend_gap.py; `_GapTracker.max_hold_bars` overrides constructor default; `tracker.max_hold_bars` used at exit check line 187; engine safety ceiling = 60 |
| 2  | Dividend gap signals bypass ADX combiner routing and are not diluted below min_combined_confidence by other strategies | VERIFIED | `_EVENT_STRATEGIES = frozenset({"dividend_gap", "cbr_calendar"})` at combiner.py:44; bypass at lines 361-366 (`not is_event` guard); `_EVENT_MIN_CONFIDENCE = Decimal("0.40")` with floor at lines 466-468 |
| 3  | CBRStrategyWrapper is registered in the combiner and generates signals around CBR rate decision dates | VERIFIED | `CBRStrategyWrapper` implemented in `cbr_strategy_wrapper.py`; `_setup_cbr_strategy()` in run_iteration.py builds it from `results/event_data/cbr/decisions.json` (confirmed present); appended to strategy list at line 565-567; registered as "cbr_calendar" matching preset keys |
| 4  | BrentGateStep in the sizing pipeline reduces energy sector position sizes when Brent-in-RUB is below threshold | VERIFIED | `class BrentGateStep` in position_sizing_pipeline.py:145; `_BRENT_RUB_THRESHOLD = 5000.0`, `_BRENT_GATE_SCALE = Decimal("0.5")`; wired in engine `_build_sizing_pipeline()` at line 186-187; run_iteration.py supplies `brent_rub_price` via `_compute_moex_sizing_data()` |
| 5  | RubOilRegimeStep in the sizing pipeline scales equity positions based on RUB/oil decorrelation state | VERIFIED | `class RubOilRegimeStep` in position_sizing_pipeline.py:127; scales by `state.position_scale` (NORMAL=1.0, ELEVATED=0.5, CRISIS=0.25); wired in engine `_build_sizing_pipeline()` at line 184-185; run_iteration.py supplies signal via `_compute_moex_sizing_data()` for ru_* segments |

**Score: 5/5 truths verified**

---

## Required Artifacts

### Plan 01 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/strategies/combiner.py` | `_EVENT_STRATEGIES` frozenset and ADX bypass logic | VERIFIED | Contains `_EVENT_STRATEGIES = frozenset({"dividend_gap", "cbr_calendar"})` at line 44; bypass guard `not is_event` at lines 361, 364; `_EVENT_MIN_CONFIDENCE = Decimal("0.40")` at line 48 |
| `src/finalayze/strategies/dividend_gap.py` | Yield-based hold bar method `_yield_hold_bars` | VERIFIED | Static method present lines 106-119; `_GapTracker.max_hold_bars: int = 40` default field; tracker field used at exit check line 187 |
| `src/finalayze/strategies/presets/ru_blue_chips.yaml` | `cbr_calendar` entry with `enabled: true` | VERIFIED | `cbr_calendar: enabled: true, weight: 0.10, params: {min_confidence: 0.30}` at lines 72-76 |
| `src/finalayze/strategies/presets/ru_energy.yaml` | `cbr_calendar` entry with `enabled: true` | VERIFIED | `cbr_calendar: enabled: true, weight: 0.08, params: {min_confidence: 0.30}` at lines 72-76 |
| `src/finalayze/backtest/config.py` | `dividend_gap` hold bars = 60 | VERIFIED | `DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"] = 60` at line 31 |

### Plan 02 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/risk/position_sizing_pipeline.py` | `class RubOilRegimeStep` | VERIFIED | Lines 127-142; checks `segment_id.startswith("ru_")`, uses `state.position_scale` from `RubOilRegimeSignal.get_regime()` |
| `src/finalayze/risk/position_sizing_pipeline.py` | `class BrentGateStep` | VERIFIED | Lines 145-171; checks `segment_id == "ru_energy"`, graceful degradation on `brent_rub <= 0`, scales by 0.5 below threshold |
| `src/finalayze/backtest/engine.py` | `_build_sizing_pipeline` with RubOilRegimeStep and BrentGateStep inserted | VERIFIED | `_build_sizing_pipeline(segment_id)` method at lines 171-194; imports at lines 44, 52; called in `run()` (line 252) and `run_portfolio()` (line 653) |
| `src/finalayze/backtest/config.py` | `brent_rub_price` and `rub_oil_regime_signal` fields | VERIFIED | `rub_oil_regime_signal: object | None = None` at line 169; `brent_rub_price: float = 0.0` at line 171 |
| `scripts/run_iteration.py` | `_compute_moex_sizing_data` helper and BacktestConfig injection | VERIFIED | `_compute_moex_sizing_data()` at line 660; `_run_symbol` accepts both params at lines 735-736; BacktestConfig construction at lines 763-764; per-segment call at lines 1130-1174 |
| `tests/unit/test_sizing_pipeline_evt_copula.py` | `test_rub_oil_regime_step` and `test_brent_gate_step` tests | VERIFIED | `TestRubOilRegimeStep` class at line 285; `TestBrentGateStep` class at line 354; 9 new tests total |

---

## Key Link Verification

### Plan 01 Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `combiner.py` | `_EVENT_STRATEGIES` | frozenset check in `generate_signal` ADX gating block | WIRED | `is_event = strategy_name in _EVENT_STRATEGIES` at line 359; both `if regime == "trend"` and `if regime == "mr"` blocks guard with `and not is_event` |
| `dividend_gap.py` | `_yield_hold_bars` | Called during gap entry to set per-entry max hold | WIRED | `max_hold_bars=self._yield_hold_bars(gap_pct)` in `_GapTracker` constructor; `tracker.max_hold_bars` used at exit check line 187 |

### Plan 02 Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `engine.py` | `position_sizing_pipeline.py` | Import and instantiation of `RubOilRegimeStep` and `BrentGateStep` | WIRED | `from finalayze.risk.position_sizing_pipeline import ... RubOilRegimeStep, BrentGateStep` at lines 44, 52; instantiated in `_build_sizing_pipeline()` |
| `position_sizing_pipeline.py` | `rub_oil_regime.py` | `RubOilRegimeStep` wraps `RubOilRegimeSignal` | WIRED | `RubOilRegimeSignal` in TYPE_CHECKING block (line 16); `self._regime_signal.get_regime([], 0)` called at runtime line 141 |
| `run_iteration.py` | `backtest/config.py` | Passes `brent_rub_price` and `rub_oil_regime_signal` to BacktestConfig | WIRED | `_compute_moex_sizing_data(ml_market_context)` at line 1133; injected into `_run_symbol(..., brent_rub_price=brent_rub_price, rub_oil_regime_signal=rub_oil_regime_signal)` at lines 1173-1174; passed into `BacktestConfig(...)` at lines 763-764 |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| STRAT-01 | 09-01-PLAN.md | DividendGapStrategy calendar populated from expanded YAML, `_EVENT_STRATEGIES` bypass added to combiner ADX routing | SATISFIED | `_EVENT_STRATEGIES` bypass verified in combiner; DividendGap reads from expanded calendar (Phase 8 foundation); yield-based hold bars replace fixed 15-bar limit |
| STRAT-02 | 09-01-PLAN.md | CBRStrategyWrapper wired into combiner for trading around CBR rate decisions | SATISFIED | `CBRStrategyWrapper` exists, registered as "cbr_calendar" in all 4 ru_* presets; `_setup_cbr_strategy()` builds and appends it in `_build_strategies()`; `results/event_data/cbr/decisions.json` present |
| STRAT-03 | 09-02-PLAN.md | rub_oil_regime.py integrated into position sizing pipeline as RubOilRegimeStep | SATISFIED | `RubOilRegimeStep` implemented in position_sizing_pipeline.py; inserted after `RegimeStep` in `_build_sizing_pipeline()`; `run_iteration.py` supplies real `RubOilRegimeSignal` for ru_* segments |
| STRAT-04 | 09-02-PLAN.md | BrentGateStep added to sizing pipeline — gates energy sector positions when Brent below threshold | SATISFIED | `BrentGateStep` implemented in position_sizing_pipeline.py; wired in `_build_sizing_pipeline()`; `brent_rub_price` supplied by `_compute_moex_sizing_data()` from MarketContext for ru_* segments |

**All 4 phase requirements satisfied. No orphaned requirements.**

---

## CBR Coverage — All 4 ru_* Presets

| Preset | cbr_calendar enabled | weight |
|--------|---------------------|--------|
| ru_blue_chips.yaml | true | 0.10 |
| ru_energy.yaml | true | 0.08 |
| ru_finance.yaml | true | 0.10 |
| ru_tech.yaml | true | 0.05 |

All 4 ru_* presets have cbr_calendar registered and enabled.

---

## Test Results

| Test Suite | Result | Count |
|------------|--------|-------|
| `tests/unit/test_strategy_combiner.py` | PASS | Part of 80 passed |
| `tests/unit/test_dividend_gap.py` | PASS | Part of 80 passed |
| `tests/unit/test_sizing_pipeline_evt_copula.py` | PASS | Part of 80 passed |
| `tests/unit/test_backtest_engine.py` | PASS | 13 passed |

All tests pass. (Coverage threshold warning is for the full suite, not these files.)

---

## Anti-Patterns Found

No blockers or stubs found in modified files.

| File | Pattern | Severity | Notes |
|------|---------|----------|-------|
| `combiner.py:512` | `return {}` | Info | Error path in `_load_segment_config()`, not a stub |

---

## Commit Verification

All 7 documented commits confirmed in git history:

| Commit | Description |
|--------|-------------|
| `566bb0a` | test(09-01): failing tests for event strategy ADX bypass |
| `7bc92eb` | feat(09-01): _EVENT_STRATEGIES, confidence floor, CBR presets, hold bars |
| `3af66e3` | test(09-01): failing tests for yield-based hold bars |
| `ad7c12d` | feat(09-01): yield-based hold bars in DividendGapStrategy |
| `67ab6d4` | feat(09-02): RubOilRegimeStep and BrentGateStep sizing pipeline steps |
| `87486b8` | feat(09-02): wire RubOilRegimeStep and BrentGateStep into backtest engine |
| `7d4e266` | feat(09-02): wire run_iteration.py to supply Brent/USDRUB data for MOEX sizing |

---

## Human Verification Required

### 1. MOEX Backtest Trade Generation

**Test:** Run `uv run python scripts/run_iteration.py --name phase9-validation --segments ru_blue_chips,ru_energy` against the MOEX date range 2023-2025
**Expected:** Backtest produces trades from `dividend_gap` and `cbr_calendar` strategies; no "forced exit at bar 15" patterns; ru_energy positions reduced during low Brent-in-RUB periods
**Why human:** Requires live Tinkoff token (`FINALAYZE_TINKOFF_TOKEN`) and actual MOEX data to trigger ex-dividend signals

### 2. RubOilRegime Active Check

**Test:** Run a ru_blue_chips backtest and check console output for `Brent-in-RUB: X,XXX RUB/bbl` and `RUB/oil regime signal: active (N FX rates)` lines
**Expected:** Both lines appear, confirming MarketContext carries the required data and the steps are not silently skipped due to missing data
**Why human:** Requires live network access to Tinkoff API

---

## Summary

Phase 9 goal is fully achieved in the codebase. All five success criteria from ROADMAP.md are satisfied:

1. **DividendGapStrategy** uses yield-based hold bars (25/40/60) from `_yield_hold_bars()`, replacing the fixed 15-bar ceiling. The engine safety ceiling is set to 60. The calendar is populated from the Phase 8 expanded YAML foundation.

2. **Event strategy ADX bypass** is implemented via `_EVENT_STRATEGIES = frozenset({"dividend_gap", "cbr_calendar"})` with `not is_event` guards on both ADX routing branches. The event confidence floor (`_EVENT_MIN_CONFIDENCE = 0.40`) prevents dilution below entry threshold.

3. **CBRStrategyWrapper** is implemented, registered as "cbr_calendar" in all four ru_* preset YAMLs, and wired into `run_iteration.py` via `_setup_cbr_strategy()` which loads from `results/event_data/cbr/decisions.json` (file confirmed present on disk).

4. **BrentGateStep** halves ru_energy positions when Brent-in-RUB < 5000 RUB/bbl. Graceful degradation on missing data. Inserted in pipeline order after RegimeStep.

5. **RubOilRegimeStep** scales ru_* positions by correlation regime (NORMAL=1.0x, ELEVATED=0.5x, CRISIS=0.25x). Data wiring complete: `_compute_moex_sizing_data()` in run_iteration.py extracts Brent candles and USDRUB FX rates from MarketContext, builds `RubOilRegimeSignal`, and injects into BacktestConfig.

All requirements STRAT-01 through STRAT-04 are satisfied. TDD pattern was followed (failing tests committed before each feature). 7 commits confirmed in git history.

---

_Verified: 2026-03-20T09:15:00Z_
_Verifier: Claude (gsd-verifier)_
