# Phase 80 (v11.2) — Execution Wiring (end-to-end SAA rebalance run)

**Status:** DESIGN (grounded read-only scan; consilium-substitute for a low-fork glue phase — the
adversarial code review provides the multi-perspective scrutiny).
**Worktree:** `.claude/worktrees/phase80-exec` (branch `gsd/phase-80-execution-wiring` off
`origin/main` `48ffae5` = post Phase 79 #277).
**Builds on:** Phase 77 (deposit persistence), Phase 78 (budget/portfolio writer), Phase 79
(weights→orders planner + executor).

## Goal / Why

Route item 5 of the Asset-Allocation MVP. Phase 79 gave us `plan_rebalance` + `submit_rebalance_plan`,
but nothing **assembles their real inputs** from the live system. Phase 80 adds the orchestration that
wires the active SAA portfolio to an end-to-end (sandbox) rebalance run: read the active portfolio →
resolve the leg instruments → fetch current positions + last prices → load the deposit mark → compute
regime-tilted weights → build the plan → (dry-run) submit it. **SANDBOX/dry-run by default; real-money
go-live remains a hard stop.**

## Scope

**In:**
- A new orchestration module `orchestration/rebalance_execution.py` — the MOCK-TESTABLE core:
  - `normalize_positions_to_symbols(positions, registry)` — broker positions → symbol-keyed (handles
    BOTH FIGI-keyed TinkoffBroker and symbol-keyed SimulatedBroker; unknown keys skipped with a log).
  - `resolve_leg_instruments(registry)` — the configured equity/OFZ-PK tickers → resolved `Instrument`s.
  - `to_rub_price(instrument, raw_price)` — convert a broker quote to a RUB-per-unit price; **bonds are
    quoted as % of face value, so `rub = raw/100 * face_value`** (equity/ETF passes through).
  - `run_rebalance(...)` — the end-to-end async orchestration returning `(RebalancePlan, list[LegOutcome])`.
- A token-gated CLI `scripts/run_rebalance.py` mirroring `scripts/run_sandbox.py` — wires the sandbox
  `TinkoffBroker` + `BrokerRouter` + session factory + `RealClock`, parses `--mode`/`--confirm`, calls
  `run_rebalance`, prints the plan + outcomes. This is the **operator checkpoint** (needs
  `FINALAYZE_TINKOFF_TOKEN` + a sandbox account).

**Out:** any real-money/LIVE run (hard stop); modifying the Phase 79 planner/executor or the frozen
`AllocationOrchestrator`; a scheduling daemon / recurring loop (one-shot run only); auto top-up of
partial fills; DB persistence of the executed plan (a later wave); US/Alpaca legs; fetching a full
position book beyond the leg symbols.

## Locked decisions

- **L-01 Split: mock-testable core vs token-gated CLI.** `run_rebalance` takes its token-dependent
  inputs as INJECTED collaborators — a `broker_router` (already wired), a `fetch_last_prices:
  Callable[[list[str]], Mapping[str, Decimal]]`, a `session_factory`, and a `Clock`. So the whole
  orchestration is unit-testable with `SimulatedBroker` + a fake price source + a mock session, with NO
  Tinkoff token. The CLI injects the real `TinkoffBroker.get_last_prices` + sandbox broker.
- **L-02 Bond price convention.** Tinkoff quotes bonds as **% of face value**; the OFZ-PK leg's
  RUB-per-bond price is `raw_pct/100 * face_value` (`to_rub_price`). Skipping this would mis-size the
  OFZ leg by ~100×. Equity/ETF quotes are already RUB-per-share (pass-through).
- **L-03 Position key normalization.** `plan_rebalance` needs symbol-keyed `current_positions`.
  TinkoffBroker `get_positions()` is FIGI-keyed; SimulatedBroker is symbol-keyed.
  `normalize_positions_to_symbols` tries `registry.get_by_figi(key)` first (→ its symbol), else treats
  the key as an already-resolved symbol, else skips with a debug log. Only the leg symbols matter to the
  3-leg SAA plan; unrelated holdings are harmless.
- **L-04 Deterministic plan_id.** `plan_id = f"{portfolio_id}:{as_of.isoformat()}"` so a same-day re-run
  is idempotent (composes with Phase 79's deterministic `client_order_id`s → no duplicate orders).
- **L-05 Dry-run default + hard stop preserved.** `run_rebalance(mode="DRY_RUN", confirm=False)` by
  default; LIVE still triple-gated inside `submit_rebalance_plan` (unchanged from Phase 79). The CLI
  defaults to dry-run and never sets LIVE without explicit flags.

## Requirements (numbered, testable — RED-first)

- **P80-R1 normalize_positions_to_symbols** — FIGI key → instrument symbol; already-symbol key →
  passthrough; unknown key → skipped (logged), not an error.
- **P80-R2 resolve_leg_instruments** — `get_equity_symbol()`/`get_ofz_pk_symbol()` → `registry.get(sym,
  "moex")` for EQUITY + OFZ_PK; a missing instrument raises `InstrumentNotFoundError` (fail-loud).
- **P80-R3 to_rub_price** — a bond (`instrument_type=="bond"` / has `face_value`) converts `raw/100 *
  face_value`; an ETF/share passes through unchanged. A bond with no `face_value` fails loud.
- **P80-R4 run_rebalance happy path (mock)** — with a `SimulatedBroker`, a fake `fetch_last_prices`, a
  mock `session_factory`/deposit, and a fixed `Clock`, it assembles inputs and returns a `RebalancePlan`
  whose opening notional == budget, plus dry-run `LegOutcome`s — through the REAL `plan_rebalance` +
  `submit_rebalance_plan` (no test-only hook).
- **P80-R5 no active portfolio** — `run_rebalance` raises a clear error when `get_active_portfolio`
  returns `None` (no silent no-op).
- **P80-R6 deterministic plan_id + as_of from clock** — `plan_id` is `{portfolio_id}:{as_of}`; `as_of`
  comes from the injected `Clock`; a re-run with the same clock yields an identical `plan_id` and
  identical leg `client_order_id`s.
- **P80-R7 deposit mark wired** — `deposit_current_notional` comes from
  `load_deposit_broker_from_db(...).deposit_value()` (0 when no persisted deposit); the deposit
  `ManualAction` reflects it.
- **P80-R8 CLI** — `scripts/run_rebalance.py` parses `--mode {DRY_RUN,SANDBOX,LIVE}` (default DRY_RUN) +
  `--confirm`; fails loud if `FINALAYZE_TINKOFF_TOKEN` (for SANDBOX/LIVE) or the DB URL is unset; the
  real sandbox/live run is the operator checkpoint.
- **P80-R9 anti-hollow / quality** — tests verify the SHIPPED path (real planner/executor, real
  SimulatedBroker), not a hook; `ruff` + `mypy src/` green; full suite (no regressions).

## Design sketch

```
orchestration/rebalance_execution.py        # mock-testable core (no Tinkoff token)
  normalize_positions_to_symbols(positions, registry) -> dict[str, Decimal]
  resolve_leg_instruments(registry) -> dict[AssetClass, Instrument]
  to_rub_price(instrument, raw_price: Decimal) -> Decimal
  async run_rebalance(*, broker_router, mode_manager, registry, session_factory, clock,
                      fetch_last_prices, mode="DRY_RUN", confirm=False)
        -> tuple[RebalancePlan, list[LegOutcome]]

scripts/run_rebalance.py                     # token-gated CLI (operator checkpoint)
```

`run_rebalance` flow: `as_of = clock.now().date()` → `active = await get_active_portfolio(sf)` (raise
if None) → `weights = AllocationOrchestrator(RiskProfile(profile)).get_rebalance_weights(as_of)` →
`leg_instruments = resolve_leg_instruments(registry)` → `positions =
normalize_positions_to_symbols(broker_router.route("moex").get_positions(), registry)` → `raw =
fetch_last_prices([sym...])`; `last_prices = {sym: to_rub_price(inst, raw[sym])}` → `deposit_broker =
await load_deposit_broker_from_db(portfolio_id, as_of, sf)`; `deposit_current = deposit_broker.deposit_value()
if deposit_broker else 0` → `plan = plan_rebalance(...)` → `outcomes = submit_rebalance_plan(plan,
broker_router, mode_manager, confirm=confirm)`.

## Token-gating (operator checkpoint)

Mock-testable NOW (no token): normalization, instrument resolution, price conversion, weights, plan
build, dry-run submit via `SimulatedBroker`, deposit load (test DB / fixture). Token-gated (the real
sandbox cert, run by the operator): `TinkoffBroker.get_last_prices` (live quote),
`TinkoffBroker.get_positions` (live book), and a SANDBOX/LIVE `submit`. Real-money LIVE stays a hard
stop (`_enforce_live_gate`, Phase 79).

## TDD subtasks

P80-01 normalize_positions_to_symbols · P80-02 resolve_leg_instruments · P80-03 to_rub_price (bond %
→ RUB) · P80-04 run_rebalance happy path (mock, real planner/executor) · P80-05 no-active-portfolio
raises · P80-06 deterministic plan_id + clock as_of · P80-07 deposit mark wired · P80-08
scripts/run_rebalance.py CLI (+ env fail-loud) · P80-09 ruff/mypy/full-suite + document the operator
sandbox checkpoint.

## Out of scope / operator checkpoint

The real Tinkoff **sandbox** run (live quotes/positions/orders) needs `FINALAYZE_TINKOFF_TOKEN` + a
sandbox account and is an explicit operator checkpoint, not part of the autonomous build. Real-money
go-live is a hard stop requiring explicit operator confirmation.
