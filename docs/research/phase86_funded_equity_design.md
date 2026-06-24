# Phase 86 — Fully-Funded Synthetic Equity (margin + reserve + deposit-as-plug)

Status: design (corrected after a 3-lens consilium + 5-axis adversarial refutation).
Scope: EXECUTION-LAYER refinement of the SAA rebalance **planner** only. The FROZEN
`AllocationOrchestrator` (Phase 72/73) and the binding cert/gate (Phase 73/74/75) are **not**
touched — confirmed disjoint by import-channel analysis (refuter 2, HELD).

## 1. Problem & decided policy

The equity sleeve is a **leveraged** index future (IMOEXF: perpetual cash-settled, lot 1,
point_value 10 RUB/pt, contract notional ≈ points·10 ≈ 22,750, initial margin ≈ 2,342 ≈ 10%).
Phase 85 sizes the equity leg to `target_notional = budget·equity_weight` of **exposure**, but a
future only debits ~10% **margin** — so ~31.5% of the budget sat idle and the portfolio was
implicitly ~1.3× levered.

**Operator decision — "fully-funded synthetic equity":** keep the target equity EXPOSURE via the
future, but fund it fully — charge only margin, hold a cash drawdown **reserve** so a deep IMOEX
move never forces a margin-call liquidation, and **sweep** the rest into the deposit anchor. Net:
idle cash = 0, portfolio leverage = 1.0×, margin-call-safe.

## 2. Cash-flow identity — deposit is the PLUG (idle == 0 BY CONSTRUCTION)

The earlier draft asserted `margin + reserve + deposit + ofz == budget within 0.01`. That is
**structurally un-satisfiable** the moment a leg lot-floors (refuter 0 + 4): the future floors to
whole contracts, leaving an unavoidable residual. **Fix:** make the deposit (mark-only,
infinitely-divisible RUB) the residual **plug**, so the identity holds exactly:

```
# Per AUTO leg, the CASH it actually consumes, on the lot-floored TARGET position (NOT the trade delta):
#   future  (instrument_type == "future"):  leg_cash = margin_cash + reserve_cash      (see §3)
#   cash    (bond/etf/share):                leg_cash = floor(target_notional/price · lot)·lot · price
auto_leg_cash      = Σ leg_cash  over {EQUITY, OFZ_PK}
deposit_realized   = budget − auto_leg_cash            # the plug — absorbs all lot-flooring + leverage slack
assert auto_leg_cash + deposit_realized == budget      # trivially true; idle == 0 by construction
HARD STOP: deposit_realized < 0  →  raise (equity+ofz funded cash exceeds budget; cannot fund at 1.0×)
```

The deposit `ManualAction` reports the **target stock** `deposit_realized` and the **flow**
`deposit_delta = deposit_realized − deposit_current_notional` (the cash the operator actually
moves) — the existing delta/advisory machinery is reused unchanged. This is an **advisory target
allocation** (where the budget should go), not a post-fill cash-movement guarantee — the deposit is
mark-only and real-money execution stays a hard stop.

**Critical discipline (refuter 4):** the cash split is computed on the **lot-floored TARGET
position** (`floor(target_notional / contract_notional) · lot`), *independent of whether THIS
rebalance trades*. The ORDER quantity still comes from `size_auto_leg` (delta-based, **unchanged**
from Phase 85). Computing the split off `sizing.delta_qty` would wrongly reserve against the trade
increment (1 contract on a 14→15 top-up) and abort every non-greenfield / within-band cycle.

## 3. Reserve sizing — survives `target_dd` EVEN under an IM hike

`reserve = exposure · 0.30` survives only a −30% move, and only if initial margin (IM) is constant.
MOEX raised IM ~2.5× overnight in Feb-2022 (refuter 1); with a static 0.30 reserve a 2.5× IM hike
forces liquidation at −14.6%. **Fix — add IM-hike headroom:**

```
target_contracts = floor(target_notional / contract_notional) over lot
exposure         = target_contracts · contract_notional
margin_cash      = target_contracts · margin_per_contract
reserve_cash     = exposure · drawdown_survival_pct  +  margin_cash · (im_hike_mult − 1)
equity_cash      = margin_cash + reserve_cash
```

Force-liquidation fires when `posted_cash − exposure·P < IM_after_hike`. With the formula above,
`posted_cash − im_hike_mult·margin_cash = exposure·drawdown_survival_pct`, so the survivable
drawdown `P ≤ drawdown_survival_pct` **even after the IM is hiked by `im_hike_mult`**. Defaults:
`drawdown_survival_pct = 0.45` (operator's stated "-45%" intent; brackets the 2022 −33% cluster
with headroom), `im_hike_mult = 2.5` (the observed Feb-2022 ratio). Both are env-overridable and
fail-closed. The reserve is held as **CASH** (never invested): MOEX FORTS margin calls settle
same-day, while OFZ/deposit sales are T+1 / locked. This is a static one-shot buffer with **no
intra-drawdown de-risking** (no live VM/maintenance-margin monitoring exists in the broker) — hence
the conservative default. The reserve earns ~0: the honest, documented cost of margin-call safety.

## 4. Margin source — fail LOUD, fail CLOSED

`TinkoffFetcher.fetch_futures_margin(symbol) -> Decimal` resolves the FIGI and calls
`services.instruments.get_futures_margin(figi=…)`, returning `initial_margin_on_buy` via
`_money_to_decimal`. Availability is confirmed: it rides the **same** instruments channel as
`get_accrued_interests`, which already runs against the **sandbox** in this very CLI
(`run_rebalance.py` NKD fetch). Unlike `fetch_all_futures` (which swallows errors → `[]`), this
**RAISES `DataFetchError`** on any gRPC error and **rejects a zero / non-finite IM at the boundary**
(a real IMOEXF IM is never 0) — so a sandbox `UNIMPLEMENTED`/outage can never masquerade as
margin = 0. Mirrors `fetch_accrued_interest`'s raise pattern, not the swallow pattern.

The pure planner stays I/O-free: `run_rebalance` builds `margin_by_symbol` and **injects** it
(exactly like `point_value_by_symbol`). The CLI builds it **non-best-effort**: a fetch failure
aborts the run *before* any plan/preview with an explicit operator-facing message
(`run_rebalance_margin_fetch_failed`), distinct from a generic whole-plan `ValueError`. The static
`FINALAYZE_SAA_EQUITY_MARGIN_RATE` is reachable **only when explicitly set** (offline/weekend
planning) — never an automatic fallback on a live fetch failure (a too-low guess would silently
under-reserve); WARN-logged and stamped on the plan.

The planner-side guard mirrors the Phase-85 point_value guard: a `future` leg with no
`margin_by_symbol[symbol]` raises and aborts the WHOLE plan (no half-rebalance).

## 5. Code changes (smallest blast radius)

- **`config/rebalance_config.py`**: `SAA_EQUITY_DRAWDOWN_SURVIVAL_PCT_DEFAULT = Decimal("0.45")`,
  `SAA_EQUITY_IM_HIKE_MULT_DEFAULT = Decimal("2.5")`; getters
  `get_equity_drawdown_survival_pct()` (finite, `0 < x ≤ 1`), `get_equity_im_hike_mult()`
  (finite, `x ≥ 1`), `get_equity_margin_rate() -> Decimal | None` (None if unset; else finite
  `0 < x ≤ 1`). All fail-closed, mirroring `get_equity_point_value`.
- **`orchestration/rebalance_planner.py`**: new frozen `FundedEquityCash`
  (target_contracts, exposure, margin_cash, reserve_cash, equity_cash) + pure
  `compute_funded_equity_cash(...)` (§3, fail-closed on non-finite/≤0 margin or ≤0 contract
  notional). `PlannedLeg` gains additive optional `margin_cash`/`reserve_cash` (None for cash
  legs); `target_notional` stays the EXPOSURE (audit honesty). `plan_rebalance` gains
  `margin_by_symbol` + `equity_drawdown_survival_pct` + `equity_im_hike_mult` (defaults = the
  config constants); computes per-leg target cash, sets `deposit_realized = budget − auto_leg_cash`,
  HARD-STOPs on `deposit_realized < 0`, and feeds `deposit_realized` to the existing deposit
  delta/advisory/ManualAction path. `size_auto_leg` is **UNCHANGED** (order qty byte-identical).
  `_assert_notional_sane` is **UNCHANGED** (validates the input weight vector, not the plug).
- **`data/fetchers/tinkoff_data.py`**: `fetch_futures_margin` (+ `_fetch_futures_margin_async`),
  raise-on-error, reject-zero-IM.
- **`orchestration/rebalance_execution.py`**: `run_rebalance` gains `margin_by_symbol` +
  the two policy scalars (optional), threaded to `plan_rebalance`.
- **`scripts/run_rebalance.py`**: build `margin_by_symbol` via a NON-best-effort
  `fetch_equity_margin_by_symbol` helper (fail-loud abort, distinct error), source the two scalars
  from the config getters; allow the static rate only when explicitly set.

## 6. Frozen-allocator / cert confirmation (refuter 2, HELD)

Touch set = {rebalance_config, rebalance_planner, rebalance_execution, tinkoff_data, run_rebalance
CLI, new test}. The binding gate (`backtest/allocation_gate.py`, driven by
`scripts/run_allocation_gate.py`) imports **none** of these; its only fetcher is `cbr.py`, which
does not import `tinkoff_data`. The planner reaches the allocator only via the read-only
`AllocationOrchestrator(...).get_rebalance_weights(as_of)` (pure table lookup, zero writes, never
calls `run()`). Target WEIGHTS, order qty, and `run()`'s TR-curves are all unchanged — only the
post-sizing CASH split is new. Any residual bug fails CLOSED on the PLAN (raises inside
`plan_rebalance`), never mutates the allocator or the cert.

## 7. TDD test list

Config (`test_rebalance_config.py`):
1. `get_equity_drawdown_survival_pct` default == 0.45; 2. env override; 3. fail-closed on
   non-numeric / ≤0 / >1 / inf. 4. `get_equity_im_hike_mult` default == 2.5; 5. fail-closed on
   <1 / inf. 6. `get_equity_margin_rate` None when unset; set+validated; fail-closed.

Planner (`test_rebalance_planner.py`):
7. `compute_funded_equity_cash`: 15-contract target → exposure/margin/reserve exact Decimals;
   8. reserve survives `target_dd` even at `im_hike_mult·margin` (force-liq algebra);
   9. fail-closed on ≤0 / non-finite margin; 10. lot-flooring → exposure < budget·weight.
11. `plan_rebalance` with a FUTURE equity + `margin_by_symbol`: deposit_realized = budget −
    equity_cash − ofz_cash, and equity_cash + ofz_cash + deposit_realized == budget EXACTLY (idle 0).
12. **Top-up regression** (current_qty 14, target 15): plan does NOT abort; reserve computed on the
    15-contract held target, deposit reconciles; order qty == 1 (delta, unchanged).
13. **Within-band regression** (delta 0): plan does NOT abort; deposit reflects the standing
    margin+reserve on the target position.
14. **Non-greenfield deposit** (deposit_current ≠ 0): deposit_delta = deposit_realized − current,
    not the stock.
15. Future leg with NO `margin_by_symbol[symbol]` → whole-plan abort (mirrors point_value guard).
16. HARD STOP: oversized reserve → deposit_realized < 0 → raise.
17. **ETF backward-compat**: existing non-future equity path → deposit_realized == budget·weight
    (byte-identical to pre-86); `size_auto_leg` order qty unchanged.

Fetcher (`test_tinkoff_futures_margin.py`):
18. parses `initial_margin_on_buy` → Decimal; 19. RAISES `DataFetchError` on gRPC error (does NOT
    return empty); 20. rejects zero / non-finite IM at the boundary.

Execution + CLI (`test_rebalance_execution.py`, `test_run_rebalance_cli.py`):
21. `run_rebalance` threads `margin_by_symbol` to the plan (future leg funded);
22. CLI aborts LOUDLY + distinctly when the margin fetch fails (not a silent `{}`); static rate
    used only when explicitly set.
