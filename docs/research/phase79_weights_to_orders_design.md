# Phase 79 — Weights-to-Orders Engine (SAA deposit-anchored MVP, MOEX)

**Status:** DESIGN (consilium-synthesized + facts independently verified).
**Worktree:** `/Users/f1xgun/finalayze/.claude/worktrees/phase79-exec`
(branch `gsd/phase-79-weights-to-orders`, == `origin/main` after Phase 78 #276 = `20e7026`).
**Builds on:** Phase 72 (frozen `AllocationOrchestrator`), Phase 77 (SAA persistence), Phase 78
(`budget_driver` + `saa_portfolio_writer`).

## Goal / Why

The frozen `AllocationOrchestrator` (`src/finalayze/orchestration/allocation.py` `AllocationResult`) is
analytics-only: it merges three pre-computed total-return curves and emits curves, `weight_series`,
`rebalance_dates`, `rebalance_cost`, `realized_ndfl`. It owns no broker and produces no orders. Phase 78
added `run_with_active_budget` (rescales each leg to `budget * weight`) but still emits only analytics.

There is **no path today from a per-leg target (`budget * weight`) to a concrete broker order** (verified:
`ast-index symbol RebalancePlan` / `plan_rebalance` both empty). Phase 79 closes that gap for the
deposit-anchored SAA MVP: turn the active portfolio's target weights into a **REBALANCE PLAN** — real
Tinkoff/T-Bank gRPC orders for the equity and OFZ-PK legs (AUTO), and a structured operator ACTION ITEM for
the deposit leg (MANUAL, no API). It must default to SANDBOX/dry-run and must never place a live real-money
order without an explicit operator confirmation gate.

## Verified facts (independent check, not consilium-trusted)

The requirements consilium asserted "no equity ETF ticker appears anywhere in src/, config/, or the snapshot"
and recommended OFZ-PK = "the single longest-maturity floater present." **Both are wrong** — corrected here
against the real committed snapshot `src/finalayze/markets/data/moex_universe.json` (2306 instruments: 268
stock / 64 etf / 1539 bond / 421 future / 14 currency):

- **EQMX** (ВИМ – Индекс МосБиржи, figi `TCS00A101EJ5`, lot 1) IS in the snapshot — a MOEX-Index ETF, the
  tradeable real-instrument proxy for the MCFTR analytics leg. `SBMX` (figi `BBG00M0C8YM7`, lot 1) is a
  fallback. `TMOS` is NOT present.
- A naive "longest floater" pick would select a **corporate** floater (Сегежа/Яндекс/ГТЛК…). The federal
  OFZ-PK universe is the 21 `SU29*` issues (`floating_coupon=true`, lot 1, face 1000), maturities 2026-12 →
  2041-10. Default chosen: `SU29024RMFS5` (ОФЗ 29024, figi `BBG01GJ1FRZ6`, mat 2035-04-18) — liquid, long.

`Instrument` (`instruments.py:33`) carries `figi`, `lot_size`, `face_value`, `floating_coupon`;
`InstrumentRegistry.get(symbol, market_id)` (`instruments.py:85`) raises `InstrumentNotFoundError` when
missing; `build_default_registry()` (`instruments.py:268`) registers the whole snapshot.

## Scope

**In:**
- A pure, broker-free, deterministic **planner** (`orchestration/rebalance_planner.py`) turning
  `(active_portfolio, target_weights, current_positions, last_prices, leg_instruments, mode)` into a
  `RebalancePlan` of typed legs. Zero I/O, fully TDD-able.
- A thin **executor** (`orchestration/rebalance_executor.py`) dispatching `RebalancePlan.auto_legs` through
  the EXISTING `BrokerRouter` and classifying each per-leg outcome (FILLED/PARTIAL/FAILED/SKIPPED_BELOW_LOT).
  Defaults to dry-run; live is triple-gated.
- The deposit leg as a structured `ManualAction` (never an `OrderRequest`), with a READ-ONLY funding advisory.
- Lot-size-aware signed delta sizing (BUY/SELL vs current positions), dust suppression / no-churn band.
- Reading the FROZEN regime-tilted weights via an additive read-only accessor (no mutation of `allocation.py`).
- A CLI/script entry that prints the plan and (sandbox) submits.

**Out:** DB audit-table persistence/crash-resume (`RebalancePlanModel`); auto top-up of partial fills;
automatic deposit-tranche breaking; market-hours queueing; circuit-breaker/kill-switch wiring;
SELL-before-BUY cash sequencing; post-rebalance reconciliation loop; NKD/accrued-coupon-precise bond sizing
or a bond-specific `OrderRequest` subclass; multi-instrument equity basket / active selection (SAA-04 forbids
it); any modification to `allocation.py` or `budget_driver.py`; US/Alpaca legs.

## Locked decisions

- **L-01 Plan output shape.** `RebalancePlan` has `auto_legs: list[PlannedLeg]` (EQUITY + OFZ_PK, each
  carrying an `OrderRequest` + `market_id`) and `manual_actions: list[ManualAction]` (DEPOSIT only). DEPOSIT
  NEVER produces an `OrderRequest` (`DepositSimulatedBroker` is mark-only; no T-Bank deposit endpoint).
- **L-02 Equity + OFZ-PK = AUTO (real broker orders); Deposit = MANUAL.** (Operator-locked this session.)
  Both AUTO legs route through `TinkoffBroker` on `market_id="moex"` only — MOEX = Tinkoff/T-Bank gRPC,
  never yfinance (CLAUDE.md invariant #3).
- **L-03 Sandbox hard stop.** The executor DEFAULTS to dry-run. `mode != LIVE` never touches a live channel.
  `mode == LIVE` is permitted ONLY when `mode_manager.current_mode == WorkMode.REAL` (itself requiring
  `FINALAYZE_REAL_CONFIRMED=true`) AND an explicit per-call `confirm=True`. No path defaults to LIVE.
- **L-04 AllocationOrchestrator + budget_driver are FROZEN.** Phase 79 consumes `AllocationResult` / reads
  weights only; it adds no params/methods to `run()`/`_apply_allocation_and_rebalancing`. Phase 78 cert
  bases stay byte-identical.
- **L-05 Equity sleeve is passive MCFTR-tracking only (SAA-04 closed-alpha invariant).** The new modules
  have ZERO import edges to `finalayze.strategies.*` or `orchestration.signal_executor`.
- **L-06 Equity leg never routes through NDFL.** The planner emits gross order notionals; it never
  instantiates `YtdTaxAccumulator`. (deposit+OFZ NDFL netting stays inside the frozen analytics path.)
- **L-07 Deterministic idempotency.** Every `PlannedLeg.order.client_order_id` derives deterministically
  from `(plan_id, asset_class, side)`, NOT the default `uuid4` factory — re-running for the same `plan_id`
  yields byte-identical ids so Tinkoff `post_order(order_id=...)` collapses accidental duplicates.
- **L-08 Instruments are operator-overridable, fail-closed config constants.** `SAA_EQUITY_SYMBOL="EQMX"`,
  `SAA_OFZ_PK_SYMBOL="SU29024RMFS5"` in `config/rebalance_config.py`. No ticker hardcoded in the engine; a
  missing value raises `ConfigurationError`. (Resolves the consilium's "blocking" ticker question with
  verified defaults; the operator changes one line to pick a different ETF/bond — no code change.)

## Requirements (numbered, testable — RED-first)

- **P79-R1 planner module.** `orchestration/rebalance_planner.py` exposing pure `plan_rebalance(...) ->
  RebalancePlan`; ZERO broker handles, ZERO I/O, no live channel. Inputs as above; `mode` default `DRY_RUN`.
- **P79-R2 plan dataclasses.** Frozen `RebalancePlan{plan_id, created_at, portfolio_id: UUID, risk_profile,
  budget_rub: Decimal, mode, auto_legs, manual_actions}`; `PlannedLeg{asset_class, market_id, order:
  OrderRequest, side, target_notional, est_price}`; `ManualAction{asset_class, description, target_notional,
  current_notional, funding_advisory: FundingBreakdown | None}`; `LegOutcome{asset_class, requested_qty,
  result: OrderResult, status}`. DEPOSIT only ever in `manual_actions`.
- **P79-R3 delta sizing (signed, not absolute).** Per AUTO leg: `target_notional = budget_rub * weight`;
  qty via `last_prices[symbol]`; subtract `current_positions.get(symbol, 0)`; `BUY` if positive, `SELL` if
  negative.
- **P79-R4 dust / no-churn band.** `SAA_REBALANCE_BAND_PCT = Decimal("0.02")` (in `config/rebalance_config.py`).
  Suppress a leg when `abs(delta_notional) / budget_rub < band`.
- **P79-R5 lot-size pre-rounding.** Floor each AUTO qty to `Instrument.lot_size` (same
  `floor(qty/lot)*lot` the broker applies) so plan qty == broker qty; a sub-one-lot target emits no order
  (SKIPPED_BELOW_LOT).
- **P79-R6 deterministic client_order_id** from `(plan_id, asset_class, side)`; `plan_rebalance` byte-stable
  for a fixed `plan_id`.
- **P79-R7 deposit = manual only.** No `OrderRequest` for `AssetClass.DEPOSIT`; the delta becomes a
  `ManualAction`.
- **P79-R8 deposit funding advisory, READ-ONLY.** When the deposit delta is NEGATIVE, attach an advisory
  `FundingBreakdown` computed from a NON-mutating shadow of the strict funding order (matured → income →
  cash → last-resort break). Phase 79 MUST NOT call the mutating `fund_underweight` break path — assert
  `broker._tranches` / `_cash` unchanged. (Reconciles RM "wire it" vs RSK "keep unwired": surface advisory,
  never execute the break.)
- **P79-R9 executor seam, dry-run default.** `submit_rebalance_plan(plan, broker_router, mode_manager, *,
  confirm=False) -> list[LegOutcome]` (in `rebalance_executor.py`); receives an already-wired `BrokerRouter`
  (constructs no brokers), iterates `auto_legs`, dispatches via `broker_router.submit(order,
  market_id=leg.market_id)`. `mode == DRY_RUN` (default) never touches a live channel; `manual_actions` are
  returned/logged, never submitted.
- **P79-R10 live triple gate.** LIVE submission only when `mode == LIVE` AND `mode_manager.current_mode ==
  WorkMode.REAL` AND `confirm is True`; otherwise raise/downgrade to dry-run.
- **P79-R11 per-leg outcome classification.** From `OrderResult.filled`/`.quantity`: partial → PARTIAL,
  non-fill → FAILED (with reason), lot-too-small → SKIPPED_BELOW_LOT. A failed equity leg MUST NOT abort the
  OFZ-PK leg; return a per-leg outcome list. No auto top-up.
- **P79-R12 notional sanity guard.** `abs(sum(leg target_notional) - budget_rub) < Decimal("0.01")`, every
  leg target ≥ 0, no leg > budget.
- **P79-R13 FIGI fail-loud.** Each AUTO leg symbol resolves to a FIGI via the registry before order
  construction; a missing FIGI aborts the WHOLE plan (`InstrumentNotFoundError`) — no half-rebalance.
- **P79-R14 configurable leg symbols, fail-closed** (see L-08).
- **P79-R15 regime-tilted weights via additive accessor.** Add a read-only
  `AllocationOrchestrator.get_rebalance_weights(as_of: date) -> dict[AssetClass, Decimal]` reusing the
  existing tilt logic WITHOUT modifying any existing method/`run()`.
- **P79-R16 TDD anti-hollow.** Every requirement lands RED-first; tests verify the SHIPPED path (real
  `OrderRequest`, real registry resolution), never a test-only hook (Phase 72/73/77 lesson).

## Design sketch

New Layer-5 modules (`orchestration/` + `config/`; may import L0 schemas/modes + L5 broker primitives, never
L6 api/dashboard):

```
config/rebalance_config.py            # SAA_REBALANCE_BAND_PCT, SAA_EQUITY_SYMBOL, SAA_OFZ_PK_SYMBOL (fail-closed)
orchestration/rebalance_planner.py    # pure plan_rebalance(...) -> RebalancePlan + frozen dataclasses
orchestration/rebalance_executor.py   # submit_rebalance_plan(...) -> list[LegOutcome] + outcome classifier
```

`OrderRequest` (`broker_base.py`) is reused as-is `{symbol, side, quantity, client_order_id}` — it carries
NO price and NO `market_id`, so the planner pairs each order with its `market_id` inside `PlannedLeg` and
submits via `BrokerRouter.submit(order, market_id=...)`. `TinkoffBroker.submit_order` + `RetryPolicy` are
synchronous (blocking gRPC bridge), so `submit_rebalance_plan` is SYNC; an async caller offloads via
`asyncio.to_thread` (documented in the executor docstring).

## Instrument resolution + sizing

- Symbols from `config/rebalance_config.py` (L-08), fail-closed. Planner calls `registry.get(symbol, "moex")`
  → `Instrument` (figi, lot_size, face_value, floating_coupon). `est_price` = `last_prices[symbol]` passed in
  by the caller (keeps the planner pure / gRPC-free).
- Sizing: shares → `target_notional / est_price`, floored to `lot_size`. OFZ-PK floaters have `lot_size=1`,
  `face_value=1000`; W1 sizes off the clean-price proxy in `last_prices` (NKD precision OUT of scope). The
  caller normalizes held positions to the resolver's SYMBOL key before computing the delta (live
  `get_positions()` is FIGI-keyed; SimulatedBroker/`PortfolioState` are symbol-keyed).

## Rebalance math + funding advisory

`target_notional = budget_rub * weight[leg]` for all three classes; `delta = target − current`. AUTO legs →
signed BUY/SELL; deposit → `ManualAction`. Weights come from the additive `get_rebalance_weights(as_of)` so
the live plan and analytics curve agree on the high-rate/easing tilt. `fund_underweight` is wired ONLY as a
READ-ONLY advisory on a negative deposit delta — the engine never mutates the deposit broker.

## Risk guards (W1 minimum)

Sandbox hard-stop / live triple gate (P79-R10, highest consequence); deterministic client_order_id
replay-safety (P79-R6); notional sanity (P79-R12); lot pre-rounding (P79-R5); FIGI fail-loud abort
(P79-R13); per-leg outcome isolation (P79-R11); deposit non-tradeability (P79-R7) + non-mutating advisory
(P79-R8). Deferred (out of scope): circuit-breaker/kill-switch, market-hours queueing, SELL-before-BUY
sequencing, reconciliation loop, DB crash-resume.

## TDD subtasks (ordered, RED-first)

- **P79-01** frozen dataclasses (P79-R2).
- **P79-02** `config/rebalance_config.py` band + fail-closed symbols (P79-R4, R14).
- **P79-03** pure signed delta sizing (P79-R3).
- **P79-04** dust / no-churn band suppression (P79-R4).
- **P79-05** lot-size pre-rounding + below-lot skip (P79-R5).
- **P79-06** deterministic client_order_id / byte-stable plan (P79-R6, R1).
- **P79-07** deposit ManualAction + read-only funding advisory, zero broker mutation (P79-R7, R8).
- **P79-08** FIGI resolution fail-loud whole-plan abort (P79-R13).
- **P79-09** notional sanity guard (P79-R12).
- **P79-10** additive `get_rebalance_weights(as_of)` regime accessor (P79-R15).
- **P79-11** dry-run executor + per-leg outcome classification + leg isolation (P79-R9, R11).
- **P79-12** live triple gate (P79-R10, L-03).

## Open items (non-blocking; operator-overridable)

- Equity default `EQMX` / OFZ-PK default `SU29024RMFS5` are config constants — the operator can swap to
  `SBMX` or any `SU29*` issue in one line. Dry-run-only this phase; real-money is the hard stop.
- NKD-precise bond sizing, DB audit table, partial-fill auto top-up, circuit-breaker wiring: deferred to a
  follow-up wave.
