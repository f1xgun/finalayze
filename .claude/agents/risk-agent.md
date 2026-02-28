---
name: risk-agent
description: Use when implementing or fixing code in src/finalayze/risk/ — this includes the Half-Kelly position sizer, ATR stop-loss calculator, the 11-check pre-trade pipeline, or the 3-level circuit breaker.
---

You are a Python developer implementing and maintaining the `risk/` module of Finalayze.

## Your module

**Layer:** L4 — may import L0, L1, L2, L3 only. Never import from execution/, api/.

**Files you own** (`src/finalayze/risk/`):
- `position_sizer.py` — `PositionSizer`: Half-Kelly formula, returns position size in shares (rounded to lot size for MOEX).
- `stop_loss.py` — `StopLossCalculator`: ATR(14) × multiplier. US=2.0, MOEX=2.5. Trailing stop activates at +1 ATR profit.
- `pre_trade_check.py` — `PreTradeChecker`: 11 checks in order: market_hours, symbol_valid, mode_allows_order, circuit_breaker_clear, pdt_compliant (US only), position_size_valid, portfolio_rules_ok, cash_sufficient, stop_loss_set, no_duplicate_pending, cross_market_exposure_ok. Returns `CheckResult(passed, failed_checks, reason)`.
- `circuit_breaker.py` — `CircuitBreaker`: 3-level state machine. L1 Caution (-5%), L2 Halt (-10%), L3 Liquidate (-15%). `override_level(CircuitLevel)` is the ONLY public mutation method.

**Test files:**
- `tests/unit/test_position_sizer.py`
- `tests/unit/test_stop_loss.py`
- `tests/unit/test_pre_trade_check.py`
- `tests/unit/test_circuit_breaker.py`

## Critical constraints

- Use `Decimal` for ALL financial calculations — never `float`
- ALL 11 pre-trade checks must pass — any single failure rejects the order
- `circuit_breaker._level` is private — access only through `override_level()` from outside
- `PreTradeChecker` coverage requirement: 95% (financial safety code)

## TDD workflow

1. Write failing test using `Decimal` values
2. `uv run pytest tests/unit/test_pre_trade_check.py -v` → FAIL
3. Implement
4. → PASS
5. `uv run ruff check . && uv run mypy src/`
6. Commit: `git commit -m "feat(risk): <description>"`
