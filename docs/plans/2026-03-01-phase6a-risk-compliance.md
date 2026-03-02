# Phase 6A: Risk & Compliance Fixes

**Date:** 2026-03-01
**Owner:** risk-agent
**Status:** NOT STARTED
**Branch:** `feature/phase6a-risk-compliance`

## Overview

This phase addresses 11 risk and compliance gaps identified in the improvement
audit. Each task hardens a specific safety mechanism -- mode gating, position
limits, circuit breaker state machines, PDT wiring, retry resilience, fill
reporting, loss-limit resets, and Kelly sizing basis. All changes follow TDD
(RED-GREEN-REFACTOR) with `Decimal` for every financial calculation.

---

## Execution Order

Tasks are grouped into three waves based on dependency chains. Within a wave,
tasks are independent and can be executed in parallel.

```
Wave 1 (no dependencies):
  6A.1  Mode gate enforcement
  6A.5  Circuit breaker L2 sticky reset (profitable-days requirement)
  6A.6  Circuit breaker no intraday de-escalation
  6A.8  gRPC errors in Tinkoff retry policy
  6A.9  Partial fill mis-reporting (Alpaca)
  6A.11 Kelly sizes against portfolio equity

Wave 2 (depends on Wave 1):
  6A.2  Sector/segment concentration (needs 6A.1 for mode context awareness)
  6A.3  Min cash reserve 20% (needs 6A.11 for correct equity basis)
  6A.4  Cross-market exposure aggregation (needs 6A.11 for equity basis)
  6A.10 Weekly loss limit reset wiring (independent, but logically after 6A.5/6A.6)

Wave 3 (depends on Wave 1 + 2):
  6A.7  PDT tracker wiring + day trade detection (needs 6A.1 mode gate, 6A.3 cash check)
```

---

## Task Details

### 6A.1 -- Mode gate: DEBUG mode must not send real orders

**Finding:** `_strategy_cycle` processes instruments and submits orders
regardless of `WorkMode`. In `DEBUG` mode, orders could reach the broker.

**Files to modify:**
- `src/finalayze/core/trading_loop.py` (lines 312-379, `_strategy_cycle`)

**Changes:**
1. At the top of `_strategy_cycle` (after line 314), add a mode gate:
   ```python
   if not self._settings.mode.can_submit_orders():
       _log.info("_strategy_cycle: mode=%s does not allow orders -- skipping", self._settings.mode)
       return
   ```
2. This check uses `WorkMode.can_submit_orders()` (already defined at
   `src/finalayze/core/modes.py` line 31) which returns `False` only for DEBUG.

**Tests to write:**
- `tests/unit/test_trading_loop.py::TestStrategyCycle::test_debug_mode_skips_order_submission`
  -- Set `settings.mode = WorkMode.DEBUG`, verify `_strategy_cycle` returns early
  and `broker_router.submit` is never called.
- `tests/unit/test_trading_loop.py::TestStrategyCycle::test_sandbox_mode_allows_orders`
  -- Set `settings.mode = WorkMode.SANDBOX`, verify orders flow through.

**Dependencies:** None.

---

### 6A.2 -- Sector/segment concentration (40% limit) not implemented

**Finding:** `PreTradeChecker` check #7 only enforces `max_positions` count.
There is no check that a single sector/segment does not exceed 40% of equity.

**Files to modify:**
- `src/finalayze/risk/pre_trade_check.py` (class `PreTradeChecker`)
- `src/finalayze/core/trading_loop.py` (lines 496-508, the `check()` call site)

**Changes:**
1. Add a new `__init__` parameter to `PreTradeChecker`:
   ```python
   max_sector_concentration_pct: Decimal = Decimal("0.40"),
   ```
   Store as `self._max_sector_pct`.

2. Add two new parameters to `PreTradeChecker.check()`:
   ```python
   sector_exposure_value: Decimal | None = None,  # current $ in this sector
   sector_id: str = "",
   ```

3. Insert check between #7 (portfolio rules) and #8 (cash sufficient), at
   approximately line 207:
   ```python
   # 7b. Sector/segment concentration
   if (
       sector_exposure_value is not None
       and portfolio_equity > 0
       and sector_id
   ):
       prospective = sector_exposure_value + order_value
       concentration = prospective / portfolio_equity
       if concentration > self._max_sector_pct:
           violations.append(
               f"Sector '{sector_id}' concentration {float(concentration):.1%} "
               f"exceeds max {float(self._max_sector_pct):.1%}"
           )
   ```

4. In `trading_loop.py::_process_instrument`, compute `sector_exposure_value`
   by summing position values for instruments sharing the same `segment_id`,
   then pass it and `seg_id` to `check()`.

**Tests to write:**
- `tests/unit/test_pre_trade_check.py::TestPreTradeChecker::test_sector_concentration_below_limit_passes`
- `tests/unit/test_pre_trade_check.py::TestPreTradeChecker::test_sector_concentration_at_limit_fails`
- `tests/unit/test_pre_trade_check.py::TestPreTradeChecker::test_sector_concentration_not_provided_skipped`
- `tests/unit/test_pre_trade_check.py::TestPreTradeChecker::test_sector_concentration_custom_cap`

**Dependencies:** 6A.1 (mode gate should be in place so tests reflect real flow).

---

### 6A.3 -- Min cash reserve (20%) not enforced in pre-trade check

**Finding:** Check #8 only verifies `order_value <= available_cash`. It does
not ensure a 20% cash reserve remains after the trade.

**Files to modify:**
- `src/finalayze/risk/pre_trade_check.py` (class `PreTradeChecker`, check #8 area, line 209)

**Changes:**
1. Add `__init__` parameter:
   ```python
   min_cash_reserve_pct: Decimal = Decimal("0.20"),
   ```
   Store as `self._min_cash_reserve_pct`.

2. After the existing cash sufficiency check (line 210), add:
   ```python
   # 8b. Cash reserve check
   if portfolio_equity > 0:
       post_trade_cash = available_cash - order_value
       reserve_ratio = post_trade_cash / portfolio_equity
       if reserve_ratio < self._min_cash_reserve_pct:
           violations.append(
               f"Post-trade cash reserve {float(reserve_ratio):.1%} "
               f"below min {float(self._min_cash_reserve_pct):.1%}"
           )
   ```

**Tests to write:**
- `tests/unit/test_pre_trade_check.py::TestPreTradeChecker::test_cash_reserve_sufficient_passes`
- `tests/unit/test_pre_trade_check.py::TestPreTradeChecker::test_cash_reserve_below_20pct_fails`
- `tests/unit/test_pre_trade_check.py::TestPreTradeChecker::test_cash_reserve_exact_boundary`
- `tests/unit/test_pre_trade_check.py::TestPreTradeChecker::test_cash_reserve_custom_threshold`

**Dependencies:** 6A.11 (Kelly equity basis fix ensures order_value is correctly
sized before this check runs).

---

### 6A.4 -- Cross-market exposure computed per-market only, not aggregated

**Finding:** In `_process_instrument` (lines 484-489), cross-market exposure
is computed as `(invested_in_this_market + order_value) / total_equity`. This
only checks *this* market's invested value, not the sum across *all* markets.

**Files to modify:**
- `src/finalayze/core/trading_loop.py` (lines 480-494, `_process_instrument`)

**Changes:**
1. Replace the per-market invested value calculation (lines 484-489) with an
   aggregated computation:
   ```python
   # Aggregate invested value across ALL markets
   total_invested = _ZERO
   for m_id in self._circuit_breakers:
       m_equity = self._get_market_equity(m_id)
       if m_equity is None:
           continue
       m_broker = self._broker_router.route(m_id)
       m_portfolio = m_broker.get_portfolio()
       m_invested = max(m_equity - m_portfolio.cash, _ZERO)
       currency = _MARKET_CURRENCY.get(m_id, "USD")
       total_invested += self._fx.to_base(m_invested, currency)

   prospective_invested = total_invested + order_value_base
   cross_exposure = (
       prospective_invested / total_equity if total_equity > _ZERO else _ZERO
   )
   ```
2. Convert `order_value` to base currency before adding:
   ```python
   order_currency = _MARKET_CURRENCY.get(market_id, "USD")
   order_value_base = self._fx.to_base(order_value, order_currency)
   ```

**Performance note:** This adds extra `get_portfolio()` calls per instrument.
Consider caching portfolio snapshots at the start of `_strategy_cycle` in a
`market_portfolios: dict[str, PortfolioState]` to avoid repeated API calls.

**Tests to write:**
- `tests/unit/test_trading_loop.py::TestStrategyCycle::test_cross_market_exposure_aggregated`
  -- Two markets, verify exposure sums both markets' invested values.
- `tests/unit/test_trading_loop.py::TestStrategyCycle::test_cross_market_exposure_rejects_when_aggregated_too_high`

**Dependencies:** 6A.11 (equity basis).

---

### 6A.5 -- Circuit breaker L2 auto-resets without requiring 2 profitable days

**Finding:** `CircuitBreaker.reset_daily()` (line 111) unconditionally resets
HALTED to NORMAL. The design requires 2 consecutive profitable days before
L2 (HALTED) can return to NORMAL.

**Files to modify:**
- `src/finalayze/risk/circuit_breaker.py` (class `CircuitBreaker`)
- `src/finalayze/core/trading_loop.py` (`_daily_reset`, line 757)

**Changes:**
1. Add state to `CircuitBreaker.__init__`:
   ```python
   self._consecutive_profitable_days: int = 0
   self._prev_equity: Decimal = baseline
   ```

2. Modify `reset_daily` to track profitable days:
   ```python
   def reset_daily(self, new_baseline: Decimal) -> None:
       # Track consecutive profitable days
       if new_baseline > self._prev_equity and self._prev_equity > _ZERO:
           self._consecutive_profitable_days += 1
       else:
           self._consecutive_profitable_days = 0
       self._prev_equity = new_baseline
       self._baseline = new_baseline

       if self._level == CircuitLevel.CAUTION:
           self._level = CircuitLevel.NORMAL
           self._consecutive_profitable_days = 0
       elif self._level == CircuitLevel.HALTED:
           if self._consecutive_profitable_days >= 2:
               self._level = CircuitLevel.NORMAL
               self._consecutive_profitable_days = 0
           # else: stay HALTED
       # LIQUIDATE: never auto-cleared
   ```

3. Add a read-only property:
   ```python
   @property
   def consecutive_profitable_days(self) -> int:
       return self._consecutive_profitable_days
   ```

**Tests to write:**
- `tests/unit/test_circuit_breaker.py::TestCircuitBreaker::test_halted_not_cleared_by_single_profitable_day`
- `tests/unit/test_circuit_breaker.py::TestCircuitBreaker::test_halted_cleared_after_two_profitable_days`
- `tests/unit/test_circuit_breaker.py::TestCircuitBreaker::test_halted_resets_counter_on_loss_day`
- `tests/unit/test_circuit_breaker.py::TestCircuitBreaker::test_caution_still_clears_immediately`
- `tests/unit/test_circuit_breaker.py::TestCircuitBreaker::test_profitable_days_property`

**Dependencies:** None.

**Breaking change note:** Existing test `test_reset_daily_clears_halted` (line
124) will need to be updated. It currently expects an immediate NORMAL after
`reset_daily` on HALTED. This test must change to reflect the new 2-day
requirement.

---

### 6A.6 -- Circuit breaker de-escalates intraday on equity recovery

**Finding:** `CircuitBreaker.check()` (lines 90-97) directly maps drawdown to
level on every call. If equity recovers intraday after hitting L2, the level
drops back to CAUTION or NORMAL. Levels should be "sticky" within a day --
they can only escalate, never de-escalate, until `reset_daily()`.

**Files to modify:**
- `src/finalayze/risk/circuit_breaker.py` (method `check`, lines 74-99)

**Changes:**
1. Modify `check` to only allow escalation (never de-escalation):
   ```python
   def check(self, current_equity: Decimal, baseline_equity: Decimal) -> CircuitLevel:
       if baseline_equity <= _ZERO:
           self._level = CircuitLevel.LIQUIDATE
           return self._level

       drawdown = (baseline_equity - current_equity) / baseline_equity

       if drawdown >= self._l3:
           new_level = CircuitLevel.LIQUIDATE
       elif drawdown >= self._l2:
           new_level = CircuitLevel.HALTED
       elif drawdown >= self._l1:
           new_level = CircuitLevel.CAUTION
       else:
           new_level = CircuitLevel.NORMAL

       # Sticky: only escalate, never de-escalate within a day
       _LEVEL_ORDER = {
           CircuitLevel.NORMAL: 0,
           CircuitLevel.CAUTION: 1,
           CircuitLevel.HALTED: 2,
           CircuitLevel.LIQUIDATE: 3,
       }
       if _LEVEL_ORDER[new_level] > _LEVEL_ORDER[self._level]:
           self._level = new_level

       return self._level
   ```

   Note: Define `_LEVEL_ORDER` as a module-level constant for efficiency.

**Tests to write:**
- `tests/unit/test_circuit_breaker.py::TestCircuitBreaker::test_level_does_not_deescalate_intraday`
  -- Hit CAUTION, then check with NORMAL equity, assert still CAUTION.
- `tests/unit/test_circuit_breaker.py::TestCircuitBreaker::test_level_escalates_from_caution_to_halted`
- `tests/unit/test_circuit_breaker.py::TestCircuitBreaker::test_reset_daily_allows_fresh_level`
  -- After `reset_daily`, verify NORMAL baseline allows re-evaluation from NORMAL.

**Dependencies:** None.

**Breaking change note:** Existing test `test_level_escalates_on_subsequent_checks`
(line 108) still passes because it only escalates. However, any test or code
that expects de-escalation within a day will break. The existing test at line
164 (`test_equity_above_baseline_is_normal`) will need updating -- after
checking with a drawdown first, equity recovery should NOT return to NORMAL
within the same day.

---

### 6A.7 -- PDT tracker not wired in TradingLoop + no day trade detection

**Finding:** `PDTTracker` exists in `pre_trade_check.py` but is never
instantiated or wired in `TradingLoop`. The `is_day_trade` parameter is always
the default `False`. There is no logic to detect whether a proposed order would
constitute a day trade.

**Files to modify:**
- `src/finalayze/core/trading_loop.py` (`__init__`, `_process_instrument`, new
  method `_is_day_trade`)

**Changes:**
1. In `__init__` (around line 138), create and wire a `PDTTracker`:
   ```python
   from finalayze.risk.pre_trade_check import PDTTracker
   self._pdt_tracker = PDTTracker()
   self._pre_trade_checker = PreTradeChecker(
       max_position_pct=...,
       max_positions_per_market=...,
       pdt_tracker=self._pdt_tracker,
   )
   ```

2. Add a `_is_day_trade` method:
   ```python
   def _is_day_trade(self, symbol: str, side: str, market_id: str) -> bool:
       """Return True if this order would open+close a position same day.

       A SELL of a position opened today, or a BUY that would be closed today,
       constitutes a day trade. Simplified heuristic: a SELL order for a symbol
       that has an existing position opened during this strategy cycle session
       is flagged as a potential day trade.
       """
       if market_id != "us":
           return False  # PDT is US-only
       broker = self._broker_router.route(market_id)
       # If selling and we hold a position, check if it was opened today
       if side == "SELL" and broker.has_position(symbol):
           return True  # conservative: flag any same-day sell
       return False
   ```

3. In `_process_instrument`, compute `is_day_trade` and pass it:
   ```python
   is_day_trade = self._is_day_trade(order.symbol, order.side, market_id)
   ```
   Pass to `pre_trade_checker.check(..., is_day_trade=is_day_trade)`.

4. After a filled order, if the fill constitutes a day trade, record it:
   ```python
   if is_day_trade:
       self._pdt_tracker.record_day_trade(now.date())
   ```

**Tests to write:**
- `tests/unit/test_trading_loop.py::TestStrategyCycle::test_pdt_tracker_wired`
- `tests/unit/test_trading_loop.py::TestStrategyCycle::test_is_day_trade_sell_with_position`
- `tests/unit/test_trading_loop.py::TestStrategyCycle::test_is_day_trade_non_us_returns_false`
- `tests/unit/test_trading_loop.py::TestStrategyCycle::test_day_trade_recorded_on_fill`

**Dependencies:** 6A.1 (mode gate), 6A.3 (cash reserve).

---

### 6A.8 -- gRPC errors not in Tinkoff retry policy

**Finding:** `RetryPolicy.execute()` (line 61) retries on `ConnectionError`,
`TimeoutError`, `httpx.ConnectError`, and `httpx.TimeoutException`. Tinkoff
SDK uses gRPC, which raises `grpc.RpcError` on transient failures (e.g.,
UNAVAILABLE, DEADLINE_EXCEEDED). These are not caught and will propagate
immediately.

**Files to modify:**
- `src/finalayze/execution/retry.py` (lines 61-66 in `execute`, lines 92-98
  in `aexecute`)

**Changes:**
1. Add a conditional import for grpc at the top of the file:
   ```python
   try:
       import grpc
       _GRPC_RETRYABLE: tuple[type[Exception], ...] = (grpc.RpcError,)
   except ImportError:
       _GRPC_RETRYABLE: tuple[type[Exception], ...] = ()
   ```

2. Add `*_GRPC_RETRYABLE` to the except tuple in both `execute` and `aexecute`:
   ```python
   except (
       ConnectionError,
       TimeoutError,
       httpx.ConnectError,
       httpx.TimeoutException,
       *_GRPC_RETRYABLE,
   ) as exc:
   ```

   Note: Python does not allow `*` unpacking in except clauses directly. Instead,
   build the tuple as a module-level constant:
   ```python
   _RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
       ConnectionError,
       TimeoutError,
       httpx.ConnectError,
       httpx.TimeoutException,
       *_GRPC_RETRYABLE,
   )
   ```
   Then use `except _RETRYABLE_EXCEPTIONS as exc:`.

**Tests to write:**
- `tests/unit/test_retry_policy.py::test_retries_on_grpc_rpc_error`
  -- Mock `grpc.RpcError`, verify retry behavior.
- `tests/unit/test_retry_policy.py::test_grpc_import_missing_still_works`
  -- Patch `grpc` import away, verify `ConnectionError` still retried.

**Dependencies:** None.

---

### 6A.9 -- Partial fills may be mis-reported as full fills (Alpaca)

**Finding:** `AlpacaBroker.submit_order` (line 100) falls back to
`order.quantity` when `result.filled_qty` is None:
```python
quantity=Decimal(str(result.filled_qty or order.quantity)),
```
If the Alpaca API returns `filled_qty=None` (meaning no fill data yet), the
code reports the *requested* quantity instead of zero, falsely indicating a
full fill.

**Files to modify:**
- `src/finalayze/execution/alpaca_broker.py` (line 100)

**Changes:**
1. Replace line 100:
   ```python
   # Before:
   quantity=Decimal(str(result.filled_qty or order.quantity)),
   # After:
   quantity=Decimal(str(result.filled_qty)) if result.filled_qty else Decimal(0),
   ```

**Tests to write:**
- `tests/unit/test_alpaca_broker.py::test_partial_fill_reports_actual_quantity`
  -- Mock result with `filled_qty=5`, `order.quantity=10`, verify result.quantity=5.
- `tests/unit/test_alpaca_broker.py::test_none_fill_qty_reports_zero`
  -- Mock result with `filled_qty=None`, verify result.quantity=0.
- `tests/unit/test_alpaca_broker.py::test_zero_fill_qty_reports_zero`
  -- Mock result with `filled_qty=0`, verify result.quantity=0.

**Dependencies:** None.

---

### 6A.10 -- Weekly loss limit reset not wired in live TradingLoop

**Finding:** `LossLimitTracker` has `reset_week()` (line 44 of `loss_limits.py`)
but `TradingLoop._daily_reset()` never calls it. The weekly baseline is never
updated, so `is_halted` computes weekly loss against the initial zero equity.

**Files to modify:**
- `src/finalayze/core/trading_loop.py` (`_daily_reset`, around line 765)

**Changes:**
1. After `self._loss_limit_tracker.reset_day(now, total_equity)` (line 765),
   add:
   ```python
   # Reset weekly baseline on Monday (weekday 0)
   _MONDAY = 0
   if now.weekday() == _MONDAY:
       self._loss_limit_tracker.reset_week(now, total_equity)
   ```

**Tests to write:**
- `tests/unit/test_trading_loop.py::TestDailyReset::test_weekly_reset_on_monday`
  -- Set `_now` to a Monday, verify `reset_week` is called.
- `tests/unit/test_trading_loop.py::TestDailyReset::test_no_weekly_reset_on_tuesday`
  -- Set `_now` to a Tuesday, verify `reset_week` is NOT called.

**Dependencies:** None (logically cleaner after 6A.5/6A.6 circuit breaker fixes).

---

### 6A.11 -- Kelly sizes against available cash, not portfolio equity

**Finding:** `_build_order` (line 536) computes:
```python
order_value = kelly_fraction * available_cash
```
Kelly criterion should size against total portfolio equity, not available cash.
Using cash gives under-sized positions as more capital is deployed.

**Files to modify:**
- `src/finalayze/core/trading_loop.py` (`_build_order` signature and body,
  lines 520-546; `_process_instrument` call site, line 470)

**Changes:**
1. Change `_build_order` signature: replace `available_cash: Decimal` with
   `portfolio_equity: Decimal` (and keep `available_cash` for the max cap):
   ```python
   def _build_order(
       self,
       signal: Signal,
       level: CircuitLevel,
       portfolio_equity: Decimal,
       available_cash: Decimal,
       candles: list[Candle],
       symbol: str,
       kelly_fraction: Decimal,
   ) -> OrderRequest | None:
   ```

2. Change line 536:
   ```python
   # Before:
   order_value = kelly_fraction * available_cash
   # After:
   order_value = kelly_fraction * portfolio_equity
   ```

3. Add a cash cap so the order cannot exceed available cash:
   ```python
   order_value = min(order_value, available_cash)
   ```

4. Update the call site in `_process_instrument` (line 470):
   ```python
   order = self._build_order(
       signal, level, portfolio.equity, portfolio.cash, candles, instrument.symbol, kelly_fraction
   )
   ```

**Tests to write:**
- `tests/unit/test_trading_loop.py::TestBuildOrder::test_kelly_sizes_against_equity`
  -- equity=100k, cash=30k, kelly=0.1 -> order_value = 10k (not 3k).
- `tests/unit/test_trading_loop.py::TestBuildOrder::test_kelly_capped_by_available_cash`
  -- equity=100k, cash=5k, kelly=0.1 -> order_value = 5k (capped).

**Dependencies:** None.

---

## File Summary Table

| File | Tasks | Type |
|------|-------|------|
| `src/finalayze/risk/pre_trade_check.py` | 6A.2, 6A.3 | L4 risk |
| `src/finalayze/risk/circuit_breaker.py` | 6A.5, 6A.6 | L4 risk |
| `src/finalayze/core/trading_loop.py` | 6A.1, 6A.2, 6A.4, 6A.7, 6A.10, 6A.11 | L6 orchestrator |
| `src/finalayze/execution/retry.py` | 6A.8 | L5 execution |
| `src/finalayze/execution/alpaca_broker.py` | 6A.9 | L5 execution |
| `src/finalayze/risk/kelly.py` | -- (no changes) | L4 risk |
| `src/finalayze/risk/position_sizer.py` | -- (no changes) | L4 risk |
| `src/finalayze/risk/loss_limits.py` | -- (no changes) | L4 risk |
| `config/settings.py` | -- (no changes needed; existing fields suffice) | L1 config |

## Test File Summary

| Test file | Tasks covered | New tests |
|-----------|---------------|-----------|
| `tests/unit/test_pre_trade_check.py` (NEW) | 6A.2, 6A.3 | ~8 |
| `tests/unit/test_circuit_breaker.py` | 6A.5, 6A.6 | ~8 (+ update 2 existing) |
| `tests/unit/test_trading_loop.py` | 6A.1, 6A.4, 6A.7, 6A.10, 6A.11 | ~12 |
| `tests/unit/test_retry_policy.py` | 6A.8 | ~2 |
| `tests/unit/test_alpaca_broker.py` | 6A.9 | ~3 |
| **Total** | | **~33 new tests** |

---

## Verification Checklist

After all tasks are complete, run:

```bash
# Full test suite
uv run pytest tests/ -v --tb=short

# Coverage for risk module (target: 95%)
uv run pytest tests/unit/test_pre_trade_check.py tests/unit/test_circuit_breaker.py -v --cov=src/finalayze/risk --cov-report=term-missing

# Lint + type check
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
```

---

## Risk Assessment

- **6A.5 + 6A.6** are the highest-risk changes: they modify the circuit breaker
  state machine, which is a critical safety component. Existing tests
  (`test_reset_daily_clears_halted`, `test_equity_above_baseline_is_normal`)
  will intentionally break and must be updated to reflect the new sticky +
  profitable-days semantics.

- **6A.9** is the simplest fix (one line) but has the highest production impact:
  incorrect fill quantities can cascade into wrong position tracking, wrong P&L,
  and wrong risk calculations.

- **6A.1** is a safety-critical gate. If missed, DEBUG mode could place orders
  in a sandbox or even live environment.

- **6A.4** changes the cross-market exposure calculation to be truly aggregated.
  The current per-market computation gives a false sense of safety when both
  markets are heavily invested.
