# Phase 82 (v11.2) — Harden the rebalance: audit persistence + reconciliation + NKD sizing

**Status:** DESIGN (grounded read-only scan; ultracode adversarial review provides the scrutiny).
**Worktree:** `.claude/worktrees/phase82-exec` (branch `gsd/phase-82-rebalance-audit` off
`origin/main` `7f4fc1d` = post Phase 81 #279). **Builds on:** Phases 77–81.

## Goal / Why

The Asset-Allocation MVP route is complete, but a rebalance run currently leaves **no record** and
is **not verified against fills**, and the OFZ-PK leg is sized off the clean price (ignoring accrued
coupon). This phase closes the gap from "demo works" to "trustworthy, auditable execution":
1. **Persist** every rebalance run + per-leg order outcome to the DB (auditable, resumable record).
2. **Reconcile** actual fills vs the plan → a per-leg report + overall status/fill-rate.
3. **NKD-precise OFZ-PK sizing** — size the bond leg off the DIRTY price (clean + accrued coupon).

All token-free + mock-testable in the core; the real NKD fetch (Tinkoff) is injected (CLI) / stubbed
(tests). No change to the read-only API/dashboard or the real-money hard stop.

## Scope

**In:**
- `core/models.py`: `SaaRebalanceRunModel` (`saa_rebalance_runs`) + `SaaRebalanceOrderModel`
  (`saa_rebalance_orders`) mirroring the Phase 77 model style.
- `alembic/versions/013_rebalance_persistence.py` (revises 012) + static-AST + DB-introspection tests.
- `execution/rebalance_writer.py`: pure row-builders + async `persist_rebalance_run(session_factory,
  plan, outcomes, reconciliation) -> UUID`.
- `orchestration/rebalance_reconcile.py`: pure `reconcile_rebalance_run(plan, outcomes) ->
  RebalanceReconciliation`.
- `run_rebalance`: NKD dirty-price sizing (injected `nkd_by_symbol`), + persist & reconcile after a
  submit; `scripts/run_rebalance.py` fetches real NKD (`fetch_accrued_interest`).

**Out:** changing the read-only API/dashboard; the real-money hard stop; auto top-up of partial
fills; crash-resume replay (the deterministic plan_id + client_order_ids already give idempotency);
modifying the frozen `AllocationOrchestrator`.

## Locked decisions

- **L-01 Persist on a real submit only.** `run_rebalance(submit=False)` (preview) records nothing;
  `submit=True` persists the run + per-leg orders + the reconciliation rollup. Persist is **best
  effort** (orders are already placed): a persist failure logs an error but does not fail the return.
- **L-02 Reconciliation is pure + token-free.** `reconcile_rebalance_run(plan, outcomes)` computes
  per-leg planned-vs-filled, an overall status (COMPLETE / PARTIAL / FAILED / MIXED) and a fill-rate
  from in-memory data (`LegOutcome.requested_qty`, `.result.quantity`, `.status`). No DB, no broker.
- **L-03 NKD via injection.** `run_rebalance` gains `nkd_by_symbol: Mapping[str, Decimal] | None`
  (RUB accrued-coupon per bond). Bond leg price = `to_rub_price(inst, clean) + nkd[symbol]` (==
  `bond_math.dirty_price`); equity unaffected (not in the map). Default `None` → clean-only
  (backward-compatible). The CLI fetches NKD via `fetch_accrued_interest` (token-gated); tests stub it.
- **L-04 Schema parity.** The migration matches the ORM byte-for-byte (Numeric/UUID/DateTime tz);
  FK `saa_rebalance_orders.run_id -> saa_rebalance_runs.id` ON DELETE CASCADE;
  `saa_rebalance_runs.portfolio_id -> saa_portfolios.id` ON DELETE RESTRICT.

## Requirements (numbered, testable — RED-first)

- **P82-R1** `SaaRebalanceRunModel` + `SaaRebalanceOrderModel` ORM rows (run: id, portfolio_id FK,
  plan_id, as_of, mode, budget_rub, status, fill_rate, created_at; order: id, run_id FK CASCADE,
  asset_class, symbol, side, requested_qty, filled_qty, status, client_order_id, reason, created_at).
- **P82-R2** migration 013 (revises 012) creates both tables + indexes + FKs; static-AST test
  (revision/down_revision/tablenames/FK ondelete) + a DB-introspection integration test (gated).
- **P82-R3** pure `_run_row(plan, reconciliation)` / `_order_rows(run_id, outcomes)` build the model
  instances correctly from a plan + outcomes (unit-testable, no DB).
- **P82-R4** `persist_rebalance_run(session_factory, plan, outcomes, reconciliation) -> UUID` inserts
  one run + N order rows in ONE transaction (integration test, gated on FINALAYZE_DATABASE_URL).
- **P82-R5** `reconcile_rebalance_run(plan, outcomes)` → per-leg planned/filled/status, overall
  status rollup, fill-rate, and alerts for any non-FILLED leg. Pure.
- **P82-R6** NKD dirty-price sizing: with `nkd_by_symbol={ofz: X}`, the OFZ leg sizes off
  `clean_rub + X` (fewer bonds than clean-only); equity is unchanged; default None == Phase 80 behavior.
- **P82-R7** `run_rebalance` on a submit calls `reconcile_rebalance_run` + `persist_rebalance_run`
  (best-effort); a persist failure logs an error and still returns `(plan, outcomes)`. Existing Phase
  80 run_rebalance tests patch the new persist collaborator (no real DB).
- **P82-R8** `scripts/run_rebalance.py` fetches NKD for the bond leg (token-gated) and passes
  `nkd_by_symbol`; preview/sandbox/live gating unchanged.
- **P82-R9** `ruff` + `mypy src/` green; full suite (no regressions).

## Design sketch

```
core/models.py                       # + SaaRebalanceRunModel, SaaRebalanceOrderModel
alembic/versions/013_rebalance_persistence.py
execution/rebalance_writer.py        # _run_row/_order_rows (pure) + persist_rebalance_run (async)
orchestration/rebalance_reconcile.py # RebalanceReconciliation (frozen) + reconcile_rebalance_run (pure)
orchestration/rebalance_execution.py # + nkd_by_symbol sizing; + persist/reconcile after submit
scripts/run_rebalance.py             # + fetch_accrued_interest -> nkd_by_symbol
```

Reuse: `bond_math.nkd()` + `dirty_price()` (exist); `fetch_accrued_interest` (exists, token-gated);
the Phase 77 session/writer pattern; the `equity_reconcile.py` report-dataclass pattern.

## TDD subtasks

P82-01 models + migration 013 (+ static AST test) · P82-02 reconcile (pure) · P82-03 NKD dirty-price
sizing · P82-04 pure row-builders · P82-05 persist_rebalance_run (+ integration test gated) · P82-06
hook persist+reconcile into run_rebalance (patch in Phase-80 tests) · P82-07 CLI NKD fetch ·
P82-08 ruff/mypy/full-suite + ultracode adversarial review + PR.

## Out of scope / hard stop

Real-money go-live remains a hard stop (operator confirmation). Auto top-up of partial fills, intent
pre-write/crash-replay, and a reconciliation dashboard view are deferred follow-ups.
