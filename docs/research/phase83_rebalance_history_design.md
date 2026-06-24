# Phase 83 (v11.2) — Rebalance history view (read the audit trail)

**Status:** DESIGN (grounded; ultracode adversarial review provides scrutiny).
**Worktree:** `.claude/worktrees/phase83-exec` (branch `gsd/phase-83-rebalance-history` off
`origin/main` `2ab74a9` = post Phase 82 #280). **Builds on:** Phase 81 (SAA API/dashboard), Phase 82
(audit tables).

## Goal / Why

Phase 82 persists every rebalance run + per-leg orders (`saa_rebalance_runs` / `saa_rebalance_orders`),
but nothing surfaces them. Phase 83 closes the loop: a **read-only, token-free** API endpoint +
dashboard page that show the operator their rebalance history — when each run happened, its mode +
reconciliation status + fill rate, and the per-leg requested/filled outcomes. Entirely DB reads; no
broker, no Tinkoff token, no order placement.

## Scope

**In:**
- `execution/rebalance_reader.py`: pure `_to_record` mapper + async `list_rebalance_runs(session_factory,
  portfolio_id, *, limit)` (newest first, eager-loads orders).
- `GET /api/v1/saa/rebalance-runs` (auth, token-free) in `api/v1/saa.py` (the existing SAA router).
- `dashboard/pages/rebalance_history.py` (`render` + pure row helper + the module-level `render(_api)`
  entry block — Phase 81 CR-01 lesson) + nav entry + `api_client.saa_rebalance_runs()` helper.

**Out:** any live broker data / live rebalance preview (token-gated → CLI); order placement; mutating
endpoints; modifying Phase 77–82 code or the real-money hard stop.

## Locked decisions

- **L-01 Read-only, token-free.** Pure DB reads of the Phase 82 audit tables; constructs no broker.
- **L-02 404 on no active portfolio** (consistent with `/saa/target-allocation`); **200 with an empty
  list** when the active portfolio simply has no runs yet.
- **L-03 Money/qty/weights as strings** (exact Decimal); frozen Pydantic + frozen record dataclasses.
- **L-04 Dashboard page MUST invoke `render` at module level** (Phase 81 CR-01) + a test that executes
  the module (mocked streamlit) so a blank page can't ship.

## Requirements (numbered, testable — RED-first)

- **P83-R1** `_to_record(run_model)` maps a `SaaRebalanceRunModel` (+ its orders) to a frozen
  `RebalanceRunRecord` (run_id, plan_id, as_of, mode, status, fill_rate, created_at, orders); pure,
  no DB (instantiate the ORM objects in memory).
- **P83-R2** `list_rebalance_runs(sf, portfolio_id, *, limit=20)` returns the portfolio's runs newest
  first, each with its orders eager-loaded (integration test, gated on FINALAYZE_DATABASE_URL).
- **P83-R3** `GET /api/v1/saa/rebalance-runs` returns 401 without `X-API-Key`.
- **P83-R4** returns 404 when there is no active portfolio.
- **P83-R5** returns 200 with `{portfolio_id, runs: [...]}`; each run carries its per-leg orders;
  token-free (the test patches only the DB reads — no broker constructed).
- **P83-R6** `?limit=N` is honored + capped (sane default 20, max e.g. 100).
- **P83-R7** `api_client.saa_rebalance_runs()` helper (respx test); `pages/rebalance_history.py`
  `render(api)` + module-level entry block + a module-execution guard test (Phase 81 CR-01).
- **P83-R8** `ruff` + `mypy src/` green; full suite (no regressions).

## Design sketch

```
execution/rebalance_reader.py        # RebalanceRunRecord/OrderRecord (frozen) + _to_record + list_rebalance_runs
api/v1/saa.py                        # + GET /rebalance-runs -> RebalanceRunsResponse
dashboard/pages/rebalance_history.py # render + _build_run_rows + module-level render(_api)
dashboard/app.py                     # + nav Page
dashboard/api_client.py              # + saa_rebalance_runs()
```

Endpoint flow: `sf = get_async_session_factory()` → `active = await get_active_portfolio(sf)` (404 if
None) → `runs = await list_rebalance_runs(sf, portfolio_id, limit=limit)` → map to Pydantic. Reader uses
`select(SaaRebalanceRunModel).where(portfolio_id==...).order_by(created_at.desc()).limit(limit)
.options(selectinload(SaaRebalanceRunModel.orders))`.

## TDD subtasks

P83-01 reader (`_to_record` pure + `list_rebalance_runs` gated) · P83-02 endpoint (auth/404/200/limit) ·
P83-03 dashboard (api_client helper + page render + module-exec guard + nav) · P83-04 ruff/mypy/full-suite
+ ultracode adversarial review + PR.

## Out of scope / hard stop

Live rebalance preview (current positions vs target) + the sandbox cert stay token-gated (CLI / operator
checkpoint). Real-money go-live remains a hard stop.
