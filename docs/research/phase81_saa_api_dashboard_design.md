# Phase 81 (v11.2) — SAA API + Dashboard (FINAL route item)

**Status:** DESIGN (grounded read-only scan; consilium-substitute for a thin-wrapper UI phase —
adversarial review provides the multi-perspective scrutiny).
**Worktree:** `.claude/worktrees/phase81-exec` (branch `gsd/phase-81-saa-api-dashboard` off
`origin/main` `ca4d685` = post Phase 80 #278).
**Builds on:** Phase 77 (deposit), Phase 78 (portfolio writer), Phase 79 (weights), Phase 80 (wiring).

## Goal / Why

Final route item of the Asset-Allocation MVP: make the SAA **visible**. A read-only API endpoint +
dashboard page that surface the operator's Strategic Asset Allocation — the active portfolio (budget,
risk profile), the regime-tilted target weights for today, the per-leg target notionals
(`budget × weight`), and the deposit mark. Entirely **token-free** (DB + pure compute + committed
snapshot); the live rebalance preview (current positions vs target) needs a Tinkoff token and stays
the CLI (operator checkpoint). No order placement from the web UI; real-money go-live is a hard stop.

## Scope

**In:**
- `GET /api/v1/saa/target-allocation` (auth-protected, token-free) in a new `api/v1/saa.py` router,
  registered in `api/v1/router.py`.
- A Streamlit page `dashboard/pages/saa_allocation.py` (`render(api)`) + nav entry in
  `dashboard/app.py` + a `get_saa_target_allocation` helper in `dashboard/api_client.py`.
- Tests: `tests/unit/test_api_saa.py` (auth/404/200/structure) + a dashboard api-client test.

**Out:** any order placement or live-broker data from the web UI (CLI-only, hard stop); mutating
endpoints; modifying Phase 77–80 code or the frozen `AllocationOrchestrator`; a live rebalance preview
(token-gated → CLI).

## Locked decisions

- **L-01 Read/preview-only, token-free.** The endpoint constructs no `TinkoffBroker`, fetches no live
  positions/prices, places no orders. All data is DB read (`get_active_portfolio`,
  `load_deposit_broker_from_db`) + pure compute (`get_rebalance_weights`) + the committed registry
  snapshot (`resolve_leg_instruments`).
- **L-02 Money + weights as strings.** Decimal serialized to `str` (exact), mirroring the existing API
  money convention; frozen Pydantic response models (`ConfigDict(frozen=True)`).
- **L-03 404 on no active portfolio** (clear, not a silent empty body).
- **L-04 as_of = today via RealClock** (`RealClock().now().date()`), feeding the look-ahead-safe regime
  tilt (`get_rebalance_weights`).

## Requirements (numbered, testable — RED-first)

- **P81-R1** `GET /api/v1/saa/target-allocation` returns **401** without a valid `X-API-Key`.
- **P81-R2** returns **404** when there is no active SAA portfolio.
- **P81-R3** returns **200** with `{portfolio_id, risk_profile, budget_rub, as_of,
  deposit_current_notional_rub, legs}`; `legs` is keyed by asset class (`deposit`/`ofz_pk`/`equity`),
  each `{weight, target_notional_rub, symbol|null}`.
- **P81-R4** per-leg `target_notional_rub == budget_rub × weight` exactly; the weights sum to 1; the
  equity + OFZ-PK legs carry a tradeable `symbol`, the deposit leg's `symbol` is null.
- **P81-R5** the endpoint is token-free — it never constructs a `TinkoffBroker` (verified by the test
  patching only the two DB reads; weights/instruments resolve for real).
- **P81-R6** `dashboard/api_client.py` gains `get_saa_target_allocation(base_url, api_key)`; a test
  asserts it injects `X-API-Key` and hits `/api/v1/saa/target-allocation` (respx).
- **P81-R7** `dashboard/pages/saa_allocation.py` exposes `render(api)`; registered in `app.py` nav.
- **P81-R8** `ruff` + `mypy src/` green; full suite (no regressions).

## Design sketch

```
api/v1/saa.py                         # GET /saa/target-allocation -> SaaTargetAllocation (frozen)
api/v1/router.py                      # + include_router(saa_router)
dashboard/pages/saa_allocation.py     # render(api): metrics + per-leg weights/targets table
dashboard/app.py                      # + nav Page("pages/saa_allocation.py", "SAA Target")
dashboard/api_client.py               # + get_saa_target_allocation(...)
```

Endpoint flow: `session_factory = get_async_session_factory()` (in-endpoint, repo pattern) →
`active = await get_active_portfolio(sf)` (404 if None) → `weights =
AllocationOrchestrator(RiskProfile(profile)).get_rebalance_weights(RealClock().now().date())` →
`leg_instruments = resolve_leg_instruments(build_default_registry())` → `deposit =
await load_deposit_broker_from_db(pid, as_of, sf)`; `deposit_value = deposit.deposit_value() if deposit
else 0` → assemble `legs[ac] = {weight, budget*weight, symbol (None for deposit)}`.

Response models (frozen): `LegTarget{weight: str, target_notional_rub: str, symbol: str | None}`;
`SaaTargetAllocation{portfolio_id, risk_profile, budget_rub, as_of, deposit_current_notional_rub,
legs: dict[str, LegTarget]}`.

Auth: `router = APIRouter(prefix="/saa", tags=["saa"], dependencies=[Depends(api_key_auth)])`.

## TDD subtasks

P81-01 endpoint auth/404/200 + structure · P81-02 per-leg math (budget×weight) + deposit mark +
token-free assertion · P81-03 api_client helper (respx) · P81-04 dashboard page render(api) ·
P81-05 register router + nav + ruff/mypy/full-suite + adversarial review + PR.

## Out of scope / hard stop

The live rebalance preview (current positions/prices), order placement, and any real/sandbox execution
stay in `scripts/run_rebalance.py` (token-gated operator checkpoint). Real-money go-live requires
explicit operator confirmation. This phase completes the Asset-Allocation MVP route.
