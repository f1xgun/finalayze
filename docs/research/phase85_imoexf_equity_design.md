# Phase 85 (v11.2) — IMOEXF futures equity leg (exposure-matched)

**Status:** DESIGN. **Worktree:** `.claude/worktrees/phase85-exec` (branch `gsd/phase-85-imoexf-equity`
off `origin/main` `9efed3a` = post Phase 84 #282). **Builds on:** Phases 79/80/82/84.

## Goal / Why

The sandbox cert proved MOEX index ETFs (EQMX/SBMX) are **forbidden for API trading** — so the equity
leg can't auto-execute as an ETF. The operator chose to switch the equity sleeve to **IMOEXF** — the
perpetual (evergreen, exp 2099, no rollover) cash-settled MOEX-index future, which IS API-tradeable
(`api_trade=True`, lot 1, in the snapshot as `FUTIMOEXF000`, `market_id=moex`). The equity sleeve thus
becomes a **leveraged margin position** sized to match the target index **exposure** (not cash).

## Verified economics (live sandbox)

- 1 IMOEXF contract = `last_price_points × point_value`. `point_value = min_price_increment_amount /
  min_price_increment = 5 / 0.5 = 10 ₽/point`. At ~2275 pts → **~22,750 ₽ exposure/contract**.
- Initial margin ≈ **2,342 ₽/contract** (~10% — leveraged). Cash-settled, no delivery.
- For a 350k equity target: `floor(350000 / 22750) = 15` contracts (~341k exposure, ~35k margin).

## Locked decisions

- **L-01 Equity symbol → IMOEXF.** `SAA_EQUITY_SYMBOL` default becomes `"IMOEXF"` (config,
  operator-overridable, fail-closed).
- **L-02 Exposure-matched sizing.** The future leg's per-unit price for sizing = the **contract
  notional** = `last_price_points × point_value`. Then `size_auto_leg` gives `qty = target_notional /
  contract_notional` = the number of contracts (lot 1), matching the target index EXPOSURE. (The freed
  cash from leverage is a side effect; the SAA targets exposure, per the operator's choice.)
- **L-03 point_value is injected** (like NKD): `run_rebalance` gains `point_value_by_symbol`; for a
  `future` leg, `est_price = raw_points × point_value[symbol]`. Non-future legs unchanged
  (bond %/face + NKD; share/etf passthrough). A future with no injected point_value fails loud.
- **L-04 Accounting note (no model change).** The equity `target_notional` is now index EXPOSURE, not
  cash deployed; documented in the plan/endpoint. No variation-margin/total-return change in this
  phase (the SAA gate/curves are unchanged; this phase is the live ORDER path only).
- **L-05 Real-money hard stop unchanged.** Sandbox-validated; live stays triple-gated.

## Requirements (numbered, testable — RED-first)

- **P85-R1** `get_equity_symbol()` defaults to `"IMOEXF"` (env-overridable, fail-closed).
- **P85-R2** futures price path: with `point_value_by_symbol={IMOEXF: 10}` and a raw points price, the
  equity (future) leg's est_price = `points × 10`; a bond/share leg is unaffected.
- **P85-R3** exposure-matched sizing: a 350k equity target at 2275 pts × 10 → 15 contracts (lot 1).
- **P85-R4** a `future` leg with no injected point_value raises a clear error (fail-loud).
- **P85-R5** the existing run_rebalance flow still works with IMOEXF (future) as the equity leg +
  SU29024 (bond) as OFZ (the happy-path test updated; OFZ unchanged).
- **P85-R6** `scripts/run_rebalance.py` fetches the equity future's point_value best-effort
  (`future_by` → `min_price_increment_amount / min_price_increment`) and passes `point_value_by_symbol`.
- **P85-R7** sandbox cert: an IMOEXF order is placed + FILLED via the live sandbox (operator-run).
- **P85-R8** `ruff` + `mypy src/` green; full suite (no regressions).

## Design sketch

```
config/rebalance_config.py            # SAA_EQUITY_SYMBOL default "IMOEXF"
orchestration/rebalance_execution.py  # run_rebalance: + point_value_by_symbol; future-leg est_price = pts*pv
scripts/run_rebalance.py              # + fetch_point_value (future_by) -> point_value_by_symbol
docs/research/phase85_imoexf_equity_design.md
```

Price loop (run_rebalance): for each leg, if `instrument.instrument_type == "future"`: `est_price =
raw_points * point_value[symbol]` (fail loud if missing); else the existing `to_rub_price + nkd` path.

## TDD subtasks

P85-01 config default IMOEXF · P85-02 futures price path + fail-loud + exposure sizing (run_rebalance
tests updated to IMOEXF/point_value) · P85-03 CLI point_value fetch · P85-04 sandbox cert (operator-run)
+ ruff/mypy/full-suite + adversarial review + PR.

## Out of scope / hard stop

Variation-margin / daily MtM modelling, a pre-trade margin guard (rely on the broker reject + audit),
the SAA gate/total-return-curve treatment of leverage, and quarterly-future rollover (IMOEXF is
perpetual). Real-money go-live remains a hard stop.
