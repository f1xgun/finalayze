# Phase 78 (v11.2 W3) — Budget Input + Risk-Profile Parameterization

**Status:** DESIGN (consilium-approved). **Branch:** `gsd/phase-78-budget-input` (off origin/main 5a8451f = #275).
**Route item 3** of the Asset-Allocation MVP: turn the measurement harness into a usable product path —
the operator inputs a real RUB budget + a risk profile, it's persisted as the active portfolio, and the
allocator runs on THAT budget.

## Goal / Why

Phase 77 shipped the `saa_portfolios`/`deposit_tranches` schema + a read-only `deposit_loader`, but there is
**no production code that CREATES a portfolio** (ast-index: 0 non-test `SaaPortfolioModel` constructors), no
"fetch the active portfolio" helper, and no input validation. Phase 78 adds the missing CREATE path + a
**budget-driver** that scales the allocator's three pre-computed TR-curve legs to opening notionals
= `budget × profile.weights[leg]`, so the persisted (budget, profile) actually drives the allocation —
**without touching the FROZEN orchestrator or the gate cert**.

## Decisions taken (consilium open questions → MVP defaults)

1. **Input surface = CLI-only** (`scripts/create_saa_portfolio.py`); no API/dashboard (single-operator MVP).
2. **Include the budget-driver** (`budget_driver.py`) so the persisted budget is usable, not just stored.
3. **Single-active = app-level** deactivate-then-insert in one transaction; DB partial-unique-index deferred.
4. **Both args mandatory** (no silent default on a money-bearing choice).
5. **Budget = integer RUB** (`type=int`) — sidesteps `Numeric(20,2)` truncation entirely for the MVP.

## Scope

**In:** the CLI + an async writer (`create_active_portfolio` / `get_active_portfolio` in a new L5
`execution/saa_portfolio_writer.py`) + fail-closed validation + the budget-driver
(`orchestration/budget_driver.py`) that rescales each leg multiplicatively to
`budget × weight` using the leg's OWN `curve[0]`. `from __future__ import annotations`, `Decimal` money,
async session pattern from `deposit_loader.py`.

**Out:** modifying `run_allocation_gate.py` or its `Decimal(100_000)` cert bases (the HARD_FAIL×3 cert stays
byte-identical); adding a `budget` arg to `AllocationOrchestrator.run()` (D-12: it merges pre-computed
curves only — the driver scales curves DOWNSTREAM); any solver/weight-fitting (weights are FIXED config);
REST/Pydantic/dashboard; User/multi-tenant; creating tranches at portfolio-creation; a DB unique index.

## TDD subtasks (P3-01 … P3-06)

- **P3-01 RED:** failing validation unit tests (budget≤0 raises; `Decimal(str(v))` exact; unknown profile
  raises) + an integration test (extend `tests/integration/test_saa_persistence_db.py`): create → exactly
  one `is_active=True`; second create flips the first to `False` (one active remains); invalid input writes
  zero rows.
- **P3-02:** `execution/saa_portfolio_writer.py` — `get_active_portfolio(session_factory)`
  (`where(is_active.is_(True)).order_by(created_at.desc()).limit(1)`) + `create_active_portfolio(session_factory,
  *, budget_rub: Decimal, risk_profile: RiskProfile) -> UUID` (ONE txn: deactivate all active, then insert
  the new active row; id/created_at default; `deposit_accumulators=None`; no tranches).
- **P3-03:** fail-closed validation — `resolve_risk_profile(str) -> RiskProfile` (re-raise with valid
  choices), `coerce_budget(v) -> Decimal` (`Decimal(str(v))`, reject ≤0, quantize `0.01` ROUND_HALF_EVEN);
  writer calls `load_allocation_profiles()` and asserts the chosen profile resolves BEFORE insert.
- **P3-04:** `scripts/create_saa_portfolio.py` — argparse `--budget-rub` (type=int, required) + `--risk-profile`
  (required, `choices=[p.value for p in RiskProfile]`); dotenv; reuse `get_async_session_factory()`
  (core/db.py); `asyncio.run`; structlog INFO; print the new UUID; clear non-zero exit if the DB env var
  is unset.
- **P3-05:** `orchestration/budget_driver.py` — read active (budget, profile), load weights, for each of the
  three legs rescale `leg_curve[i] *= (budget × weight) / leg_curve[0]` (each leg's OWN `curve[0]`; the
  MCFTRR equity leg is a real index level rescaled multiplicatively, NOT a base swap), then call
  `AllocationOrchestrator.run(...)`. **ANTI-HOLLOW (Phase 72/73/77 lesson):** costs/NDFL must arise from the
  REAL per-leg rescale delta through the real `run()` — NO forced-delta hook. Preserve netting (deposit+OFZ
  through the one shared accumulator; equity NEVER through NDFL). Test: `total_return_pct` is
  **scale-invariant** across budgets while `rebalance_cost`/`realized_ndfl` scale **linearly** — proving the
  budget genuinely drives the economics, not a literal.
- **P3-06:** `ruff` + `mypy src/` green; full suite; assert `run_allocation_gate.py` cert bases are
  byte-unchanged; run `backtest-iteration` IF the driver seam touches measured economics.

## Success criteria

`create_saa_portfolio.py --budget-rub 100000 --risk-profile balanced` writes exactly one active portfolio
(budget_rub Decimal-exact, risk_profile `balanced`, no tranches) and prints its UUID; a second create leaves
exactly one active; invalid input fails BEFORE any write; the budget-driver rescales all three legs to
`budget × weight` with `total_return_pct` scale-invariant and cost/NDFL linear in the budget (real delta, no
hook); `run_allocation_gate.py` + its `Decimal(100_000)` bases are byte-unchanged (HARD_FAIL×3 reproduces);
`ruff` + `mypy` green; the P3-01 RED tests pass.
