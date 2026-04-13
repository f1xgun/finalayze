---
phase: 24-live-backtest-parity
verified: 2026-03-23T19:59:38Z
status: gaps_found
score: 3/4 must-haves verified
re_verification: false
gaps:
  - truth: "All 14 pre-trade checks receive their required parameters: stop_loss_price, has_pending_order, regime_state, strategy_name, correlations"
    status: partial
    reason: "PARITY-03 correctly wires all 14 parameters, but unconditional require_stop_loss=True breaks test_strategy_cycle_proceeds_when_not_halted in test_critical_safety.py — the pre-trade check rejects a BUY order when no stop state exists for the symbol, a regression introduced because the test was not updated to account for the new require_stop_loss=True default"
    artifacts:
      - path: "src/finalayze/orchestration/trading_loop.py"
        issue: "require_stop_loss=True is passed unconditionally at line 1489; new BUY orders that do not yet have a _stop_states entry will always fail pre-trade check 9 unless stop_loss_price is set first — but the order hasn't been built yet at pre-trade check time"
      - path: "tests/unit/test_critical_safety.py"
        issue: "test_strategy_cycle_proceeds_when_not_halted (line 507) was not updated after PARITY-03 landed; uses real PreTradeChecker which now rejects due to missing stop_loss_price"
    missing:
      - "Either: (a) change require_stop_loss to only be True when a stop state already exists for the symbol (i.e., for existing positions not new orders), or (b) update test_critical_safety.py to pre-populate _stop_states for AAPL before calling _strategy_cycle(), or (c) clarify the intended semantics — new BUY orders cannot have a stop_loss_price set before submission"
human_verification:
  - test: "Verify trailing stop behavior in live sandbox run"
    expected: "Stop price for an open position should ratchet upward as price climbs, and never move downward"
    why_human: "Cannot verify real-time APScheduler cycle behavior or live broker state in automated checks"
  - test: "Verify same-cycle re-entry guard in live sandbox run"
    expected: "After a stop-loss fires for symbol X in a cycle, X should not generate new signals in that same cycle's remaining _process_instrument calls"
    why_human: "Integration behavior across APScheduler cycles requires live sandbox observation"
---

# Phase 24: Live-Backtest Parity Verification Report

**Phase Goal:** Live trading loop risk pipeline matches the backtest engine — same PositionSizingPipeline steps, trailing stops, all pre-trade checks, no same-cycle re-entry
**Verified:** 2026-03-23T19:59:38Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Live trailing stop ratchets upward after activation threshold and never ratchets downward | VERIFIED | `_check_stop_losses` implements 5-step logic at lines 1847-1892: `max(state.highest_price, current_price)` updates high-water mark; `max(state.current_stop, trail_stop)` enforces ratchet-only-up |
| 2 | When a symbol is stopped out during an equity cycle, that symbol is excluded from signal generation for the remainder of the same cycle | VERIFIED | `_cycle_exited_symbols.add(symbol)` at line 1892 (stop trigger); early return at lines 1341-1343 (`if instrument.symbol in self._cycle_exited_symbols: return`); set cleared in `_reset_cycle_counters` at line 249 |
| 3 | Live _build_order() calls PositionSizingPipeline.compute() with a SizingContext containing equity, volatility, regime scale -- matching the backtest engine pipeline | VERIFIED | `_build_sizing_pipeline` at lines 1533-1574 constructs Kelly→VolTarget→Regime→Copula→EVT→MetaLabel→HardCaps; `_build_order` at lines 1667-1689 calls `pipeline.compute(context)` with full SizingContext |
| 4 | All 14 pre-trade checks receive their required parameters: stop_loss_price, has_pending_order, regime_state, strategy_name, correlations | PARTIAL | Parameters are wired at lines 1459-1500 and helpers exist (`_has_pending_order`, `_get_regime_state`, `_get_correlations`). However, `require_stop_loss=True` is passed unconditionally, breaking `test_strategy_cycle_proceeds_when_not_halted` in `test_critical_safety.py` — the pre-trade check rejects new BUY orders that don't yet have a stop state |

**Score:** 3/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/orchestration/trading_loop.py` | Trailing stop state machine + per-cycle exited symbols set | VERIFIED | Contains `StopLossState` (line 125 import), `_stop_states` dict (line 171), `_cycle_exited_symbols` set (line 175), `_check_stop_losses` method (line 1829), `_build_sizing_pipeline` (line 1533), all pre-trade helpers |
| `tests/unit/test_trading_loop_parity.py` | Tests for trailing stop ratcheting, re-entry guard, pipeline sizing, pre-trade params | VERIFIED | 582 lines (exceeds 150 min), 19 tests passing covering all 4 PARITY requirements |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `trading_loop.py` | `StopLossState` | `_stop_states` dict replaces `_stop_loss_prices` | WIRED | Pattern `_stop_states.*StopLossState` confirmed at lines 134, 171 |
| `trading_loop.py` | `_cycle_exited_symbols` | Set populated by `_check_stop_losses`, checked before signal generation | WIRED | 5 occurrences: init (175), reset (249), check (1341), add (1892), comment (1842) |
| `trading_loop.py` | `finalayze.risk.position_sizing_pipeline` | `_build_sizing_pipeline()` constructs pipeline, `_build_order()` calls `compute()` | WIRED | Inline import at line 1539; `PositionSizingPipeline` returned at line 1574; `pipeline.compute(context)` at line 1689 |
| `trading_loop.py` | `PreTradeChecker.check()` | Passes stop_loss_price, has_pending_order, regime_state, strategy_name, correlations | PARTIAL | Parameters are passed at lines 1488-1500, but `require_stop_loss=True` is unconditional causing regression |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `trading_loop.py _check_stop_losses` | `state.current_stop` | `StopLossState` stored at BUY fill (line 1796-1807) | Yes — derived from `entry_price`, `stop`, `atr_value` | FLOWING |
| `trading_loop.py _build_order` | `order_value` | `PositionSizingPipeline.compute(context)` with real equity/vol | Yes — real portfolio equity and computed vol from candles | FLOWING |
| `trading_loop.py _get_correlations` | `correlations` | Returns `{}` always | Intentional graceful degradation (documented) | STATIC — documented intentional stub, check 14 passes with empty dict |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All parity methods exist at runtime | `uv run python -c "from finalayze.orchestration.trading_loop import TradingLoop; ..."` | All 7 methods FOUND | PASS |
| 19 parity tests pass | `uv run pytest tests/unit/test_trading_loop_parity.py` | 19 passed in 2.26s | PASS |
| 10 sizing tests pass (no regressions) | `uv run pytest tests/unit/test_trading_loop_sizing_bugs.py tests/unit/test_trading_loop_kelly.py` | 10 passed | PASS |
| Stop-loss atomicity tests pass | `uv run pytest tests/unit/test_stop_loss_atomicity.py tests/unit/test_phase5_stop_loss.py` | 10 passed | PASS |
| Critical safety tests | `uv run pytest tests/unit/test_critical_safety.py` | 1 FAILED (test_strategy_cycle_proceeds_when_not_halted), 36 passed | FAIL |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|------------|------------|-------------|--------|---------|
| PARITY-01 | 24-02 | Live trading loop uses PositionSizingPipeline with all steps matching backtest engine | SATISFIED | `_build_sizing_pipeline` mirrors backtest `engine._build_sizing_pipeline` step order: Kelly→VolTarget→Regime→Copula→EVT→MetaLabel→HardCaps |
| PARITY-02 | 24-01 | Live trailing stops ratchet upward after activation threshold, matching SimulatedBroker | SATISFIED | `_check_stop_losses` implements identical 5-step logic to `SimulatedBroker.check_stop_losses`; `max()` enforces ratchet-only-up at line 1864 |
| PARITY-03 | 24-02 | All 14 pre-trade checks receive required parameters in live path | BLOCKED | Parameters are wired but `require_stop_loss=True` unconditionally causes regression in `test_critical_safety.py::TestLossLimitTrackerWired::test_strategy_cycle_proceeds_when_not_halted` |
| PARITY-04 | 24-01 | Stop-loss exit in a cycle prevents same-cycle re-entry for the same symbol | SATISFIED | `_cycle_exited_symbols.add(symbol)` at stop trigger; early return guard in `_process_instrument`; set cleared in `_reset_cycle_counters` |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/finalayze/orchestration/trading_loop.py` | 1628 | `return {}` in `_get_correlations` | Info | Intentional graceful degradation per plan decision — check 14 runs and passes with empty correlations. Future phase will wire real correlations. Not a blocker. |
| `src/finalayze/orchestration/trading_loop.py` | 1489 | `require_stop_loss=True` unconditional | Warning | New BUY orders cannot have a pre-existing stop_loss_price (the stop price is set after the order fills). Passing `require_stop_loss=True` before the order exists appears logically inverted — check 9 will always fail for new entries with no open position. |

### Human Verification Required

#### 1. Trailing Stop Ratchet in Live Sandbox

**Test:** Run TradingLoop in sandbox mode against a rising price instrument; inspect `_stop_states[symbol].current_stop` across multiple cycles
**Expected:** Stop price should ratchet up as price climbs past the activation threshold (entry + 1.0 * ATR), then follow at (highest_price - 1.5 * ATR)
**Why human:** Real-time APScheduler cycle behavior cannot be reliably tested with unit mocks

#### 2. Same-Cycle Re-Entry Guard in Live Sandbox

**Test:** Trigger a stop-loss exit in sandbox mode; confirm the stopped symbol does not generate a new BUY signal in the remainder of the same cycle
**Expected:** After stop fires, `_cycle_exited_symbols` contains the symbol; any subsequent `_process_instrument` call for that symbol returns early
**Why human:** Integration behavior across multiple `_process_instrument` calls in a live cycle requires sandbox observation

### Gaps Summary

**One gap blocks goal achievement:**

PARITY-03 wires all 14 pre-trade check parameters, which is the correct direction. However, `require_stop_loss=True` is passed unconditionally at line 1489. This causes pre-trade check 9 to reject new BUY orders that have no prior stop state — because `stop_loss_price=None` comes from `self._stop_states.get(symbol)` returning `None` for a new trade.

The root cause is a semantic mismatch: `require_stop_loss` was intended to enforce that every new BUY has a stop plan, but pre-trade checks run before the order is built and before a stop can be registered. The stop price is only set after the BUY fills (at line 1796). Therefore, any new BUY for a symbol not already in `_stop_states` will always fail check 9.

**Specific regression:** `tests/unit/test_critical_safety.py::TestLossLimitTrackerWired::test_strategy_cycle_proceeds_when_not_halted` — 1 failure, 36 passes in this file.

**Fix options:**
1. Change `require_stop_loss=False` for new entries (symbols not in `_stop_states`), `True` only when adding to an existing position
2. Or remove `require_stop_loss=True` from the live path entirely — stop-loss registration happens after fill, not before
3. Or update the test to accept that new BUY orders are now blocked unless a stop state pre-exists (but this changes the intended behavior)

Options 1 or 2 are preferable. The test regression accurately exposes the logic inversion.

---

_Verified: 2026-03-23T19:59:38Z_
_Verifier: Claude (gsd-verifier)_
