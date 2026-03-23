---
phase: 23-order-sizing-bug-fixes
verified: 2026-03-23T19:40:00Z
status: gaps_found
score: 2/3 must-haves verified
gaps:
  - truth: "SELL orders use actual held position quantity, not Kelly-computed amount"
    status: partial
    reason: >
      The fix is correctly implemented in trading_loop.py (_build_order SELL branch uses
      portfolio.positions.get(symbol)). However, the phase 23 fix introduced a regression:
      TestPDTTrackerWiring::test_day_trade_recorded_on_fill now raises
      TypeError('<=' not supported between instances of 'MagicMock' and 'decimal.Decimal')
      because the test's mock portfolio returns a MagicMock from .positions.get(), which
      the new SELL branch then compares to _ZERO. The test was passing before phase 23.
    artifacts:
      - path: "src/finalayze/orchestration/trading_loop.py"
        issue: >
          _build_order SELL branch calls portfolio.positions.get(symbol, _ZERO) then
          compares held <= _ZERO. When portfolio is a MagicMock with untyped .positions,
          the comparison raises TypeError. The fix itself is correct but exposes a test
          mock that returns MagicMock instead of Decimal.
      - path: "tests/unit/test_trading_loop.py"
        issue: >
          _make_trading_loop() creates mock_broker.get_portfolio returning
          MagicMock(equity=..., cash=...) — the .positions attribute is an untyped
          MagicMock, not a dict[str, Decimal]. After phase 23, any SELL signal through
          _strategy_cycle() hits the new portfolio.positions.get() call and crashes
          before the PDT tracker is reached.
    missing:
      - "Fix _make_trading_loop() in tests/unit/test_trading_loop.py to include a real
         PortfolioState (or at minimum positions={SYMBOL_AAPL: Decimal(10)} in the mock)
         so the SELL branch does not crash on MagicMock comparison."
      - "Alternatively, guard the SELL branch: if portfolio is None or not isinstance(
         portfolio.positions, dict), fall back gracefully."
---

# Phase 23: Order Sizing Bug Fixes — Verification Report

**Phase Goal:** SELL orders, sector exposure, and CAUTION thresholds produce correct values — no over-sells, no cross-contaminated prices, no hardcoded thresholds
**Verified:** 2026-03-23T19:40:00Z
**Status:** gaps_found — 2/3 must-haves verified, 1 partial (regression in existing test)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SELL orders use actual held position quantity, not Kelly-computed amount | PARTIAL | Fix implemented in `_build_order` (line 1522-1528), all 3 SIZE-01 regression tests pass. But `TestPDTTrackerWiring::test_day_trade_recorded_on_fill` now raises TypeError from the new SELL branch — regression introduced by this phase. |
| 2 | Sector exposure computes each position notional using its own last price | VERIFIED | `_process_instrument` lines 1442-1448 iterate `portfolio.positions`, call `_get_last_price(pos_symbol)` per symbol. `_last_prices` cache is populated at line 1331 during candle fetch. SIZE-02 regression test passes. |
| 3 | CAUTION threshold uses segment preset min_combined_confidence * 1.2, not hardcoded 0.6 | VERIFIED | `_get_segment_min_confidence()` (line 1554) reads preset YAML, caches result. No literal `0.5 * _MIN_CONFIDENCE_BOOST` remains. SIZE-03 tests for us_tech (threshold 0.36) and ru_blue_chips (threshold 0.456) both pass. |

**Score:** 2/3 truths fully verified, 1 partial (implementation correct, regression in pre-existing test)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/orchestration/trading_loop.py` | Fixed `_build_order` and `_process_instrument` | VERIFIED | 2178 lines. SELL branch at line 1522, sector exposure at 1442-1448, `_get_last_price` at 1550, `_get_segment_min_confidence` at 1554. No hardcoded `0.5 * _MIN_CONFIDENCE_BOOST`. |
| `tests/unit/test_trading_loop_sizing_bugs.py` | Regression tests for all 3 sizing bugs | VERIFIED | 386 lines (min_lines: 100 exceeded). 7 tests, 3 classes matching SIZE-01/02/03. All 7 pass. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `trading_loop.py` | `strategies/presets/*.yaml` | `_get_segment_min_confidence` reads `min_combined_confidence` | WIRED | Line 1572: `float(config.get("min_combined_confidence", default_conf))`. Path constructed at line 1565: `presets_dir / f"{seg_id}.yaml"`. Pattern present. |
| `trading_loop.py` | `core/schemas.py PortfolioState` | `portfolio.positions` in `_build_order` | WIRED | Lines 1509-1524: `portfolio: PortfolioState | None`, `portfolio.positions.get(symbol, _ZERO)`. Import at top of file (noqa suppression removed per SUMMARY). |

---

### Data-Flow Trace (Level 4)

Not applicable — `trading_loop.py` is an orchestrator with no dynamic rendering. The data flows are:
- `_last_prices` cache: populated at line 1331 from `candles[-1].close`, consumed at line 1447 via `_get_last_price()`. Flow is internal.
- `_segment_min_confidence` cache: populated from YAML reads, consumed in CAUTION check at line 1532-1534.

Both flows use real data sources (live candles, real YAML files) — not hardcoded.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| SIZE-01: SELL with 50 shares -> order.quantity == 50 | `uv run pytest tests/unit/test_trading_loop_sizing_bugs.py::TestSellOrderUsesHeldQuantity -v --no-cov` | 3 passed | PASS |
| SIZE-02: Sector exposure uses per-position prices | `uv run pytest tests/unit/test_trading_loop_sizing_bugs.py::TestSectorExposurePerPositionPrice -v --no-cov` | 1 passed | PASS |
| SIZE-03: CAUTION threshold = preset * 1.2 | `uv run pytest tests/unit/test_trading_loop_sizing_bugs.py::TestCautionThresholdFromPreset -v --no-cov` | 3 passed | PASS |
| No hardcoded 0.5/0.6 CAUTION threshold | `grep -n "0\.5 \* _MIN_CONFIDENCE_BOOST" trading_loop.py` | 0 matches | PASS |
| Regression: PDT tracker test (pre-existing) | `uv run pytest tests/unit/test_trading_loop.py::TestPDTTrackerWiring::test_day_trade_recorded_on_fill --no-cov` | FAIL — TypeError MagicMock <= Decimal | FAIL |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SIZE-01 | 23-01-PLAN.md | SELL orders use actual held position quantity, not Kelly-computed amount | PARTIAL | Implementation correct; regression in `test_day_trade_recorded_on_fill` (SELL signal through `_strategy_cycle` now crashes before PDT check). The fix broke a test that exercises SIZE-01's code path end-to-end. |
| SIZE-02 | 23-01-PLAN.md | Sector exposure calculation uses each position's own last price | SATISFIED | Per-position price loop at lines 1442-1448; `_last_prices` cache built during candle fetch. |
| SIZE-03 | 23-01-PLAN.md | CAUTION confidence threshold computed as `segment.min_combined_confidence * 1.2` | SATISFIED | `_get_segment_min_confidence()` reads preset YAML; no literal 0.5 or 0.6 in code path. |

No orphaned requirements: REQUIREMENTS.md maps only SIZE-01/02/03 to Phase 23 (lines 12-14, 68-70). All three are claimed by 23-01-PLAN.md.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/finalayze/orchestration/trading_loop.py` | 1567 | `default_conf = 0.5` | Info | Fallback when preset YAML not found. Safe default — documented in SUMMARY. Not a stub; live code falls back to 0.5 only on FileNotFoundError. |
| `tests/unit/test_trading_loop.py` | 168-170 | `MagicMock(equity=..., cash=...)` without `positions={}` | Blocker | `mock_broker.get_portfolio` returns a MagicMock with untyped `.positions`, causing TypeError in the new SELL branch. Phase 23 exposes this pre-existing mock weakness. |

---

### Human Verification Required

None — all three SIZE fixes are mechanically verifiable. The regression is programmatically confirmed.

---

## Gaps Summary

**Root cause of gap:** Phase 23 correctly fixes SIZE-01 by adding `portfolio.positions.get(symbol, _ZERO)` to the SELL branch of `_build_order`. However, the pre-existing test `test_day_trade_recorded_on_fill` in `tests/unit/test_trading_loop.py` constructs its mock portfolio as `MagicMock(equity=..., cash=...)` — the `.positions` attribute is an untyped `MagicMock`, not a `dict[str, Decimal]`. When a SELL signal is processed through `_strategy_cycle()`, the new branch calls `.positions.get(symbol, _ZERO)` which returns another `MagicMock`, and then `held <= _ZERO` raises `TypeError`.

This test was passing before phase 23 (confirmed by checking out `ceb0af2^` — the pre-fix trading_loop). The fix is correct and the regression is a test mock that was under-specified. The fix requires either:

1. Update `_make_trading_loop()` in `tests/unit/test_trading_loop.py` to return a proper `PortfolioState` with `positions={}` (or `{SYMBOL_AAPL: Decimal(10)}` for SELL tests), OR
2. Add a guard in `_build_order` that handles `portfolio.positions` not being a dict-like object.

Option 1 is preferred (fix the test mock, which is clearly wrong — `positions` should always be `dict[str, Decimal]`).

The `TestDailyPnLComputation::test_daily_reset_reports_pnl_to_metrics` failure is pre-existing (confirmed same failure at `ceb0af2^`) and unrelated to phase 23.

---

_Verified: 2026-03-23T19:40:00Z_
_Verifier: Claude (gsd-verifier)_
