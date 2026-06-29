# Gold Sleeve Experiment — beyond-MOEX-edge R&D, Phase A

**Status:** complete · **Verdict:** `NO` — gold does NOT clear the pre-registered ≥3pp
crash-window de-risking bar; it is at most a *marginal* drawdown-reducer (~2–3.7pp in
calm/acute regimes) bought at a risk-adjusted cost, and it *increases* drawdown in the
prolonged crash year. · **Deposit anchor holds.** · diagnostic / backtest-only — no real
money touched.

## Why this experiment

Active equity **selection** on MOEX was proven to have no net-of-retail edge (0/113 prior
runs + three HARD_FAIL low-turnover tilt experiments: equal-weight, dividend-yield, low-vol
vs the real IMOEX cap-weight). A feasibility scan of directions *beyond* equity selection
(gold, замещающие облигации, corporate-credit carry, regime-timing, CNY/FX) ranked **gold**
first: it is the only candidate whose behaviour is *opposite* to the all-ruble
deposit/OFZ/equity stack in the exact 2022 ruble/geopolitical tail the geo-risk overlay
(PR #300) flags as structurally unhedged — and it is testable today, token-free, on real
data covering the crash in full.

The honest, pre-registered question is **diversification, never alpha**:

> Does adding a small gold sleeve cut portfolio **MaxDD** AND not worsen the risk-adjusted
> return, *in the 2022 crash where equity selection failed* — after the ETF-TER + retail
> cost haircut?

## Method

- **Data (token-free MOEX ISS-REST):** gold = `GLDRUB_TOM` daily CLOSE on the currency/selt
  **CETS** board (`MoexISSFetcher.fetch_currency_close_history`) — spot gold in RUB, which
  kept trading through the 27-day 2022 equity halt; equity = `MCFTRR` net total-return index
  (`load_mcftr_series`). Committed snapshot: `results/research/gold/panel_snapshot.json`
  (gold 1126 bars, equity 1107 bars, 2022-01..2026-06-10).
- **Sleeves (all NET):** deposit via `accrue_real_risk_free_leg` (CBR key − 1pp, net-NDFL,
  from the committed CBR archive); equity = MCFTRR (already net); gold = `GLDRUB` netted via
  `net_index_returns` (NDFL on the daily positive mark — **conservative**: it taxes
  unrealized gains, over-stating gold's tax drag) **plus** a continuous **0.8%/yr ETF-TER**
  haircut (`apply_ter_drag`, the wrapper holding cost).
- **Allocation:** baseline = deposit 40% / equity 60%; gold carved from equity
  (sweep 5/10/15%). Fixed-weight, **quarterly rebalanced**; per-side retail cost 0.55%
  (`MOEX_RETAIL_COSTS` equivalent) charged on traded turnover of the non-free legs; the
  deposit leg is cost-free. (The OFZ floater is intentionally excluded — no pre-2023 data,
  and its role was already settled in Phase 76.)
- **Windows:** full, **acute_crash_2022** (Feb21–Apr30), **crash_year_2022** (Feb21–Dec30),
  **high_rate_2024_25**, easing_2025_26. (Explicit windows, **not** `regime_split` — that
  assumes a high-rate-era *start*, which a 2022-start axis does not have.)
- **Date alignment (review fix):** the MCFTRR equity leg comes through the index-candle path
  (`load_mcftr_series` → `_parse_history_row`), which parses ISS `TRADEDATE` as MSK-midnight →
  UTC and so stores a trade on T under T−1; the gold leg keeps the true ISS date. The cert
  shifts the equity leg +1 day to recover its true trade date, so both legs sit on one true
  calendar. (This also corrected a deposit over-compounding artefact: the misaligned union
  axis had 1365 spurious bars; aligned it is 1126 ≈ the real ~252×4.4 trading days.)

### Metric honesty (a real trap this cert documents)

The RUONIA-excess Sharpe/Sortino use a **fixed 15% basis** that is only apt for the
high-rate era. Over the 2022–2023 **low-rate** era (CBR key 7.5%) the deposit *underperforms*
that basis, so its excess-Sharpe goes hugely negative — which would make a naive
"vs deposit Sharpe" test on a 2022-start window report a **false PASS** (anything beats a
deposit measured below its own bar). So:

- the **deposit-anchor** point is made on **basis-free TOTAL RETURN** (and confirmed on the
  *proper* high-rate window where the deposit's excess-Sharpe is legitimately positive);
- the **gold verdict** is the **baseline-vs-+gold** comparison (the common 15% basis cancels;
  MaxDD is basis-free).

## Results (real data, date-aligned)

| window | arm | MaxDD% | TR% | Sortino\* |
| --- | --- | ---: | ---: | ---: |
| full_window | deposit | 0.0 | **60.9** | — |
| full_window | baseline | 30.3 | 17.0 | −0.67 |
| full_window | +gold 15% | 28.3 | 19.3 | −0.92 |
| high_rate_2024_25 | deposit | 0.0 | **22.4** | +1.39 |
| high_rate_2024_25 | baseline | 14.5 | 9.2 | −0.75 |
| high_rate_2024_25 | +gold 15% | **10.7** | 11.3 | −0.84 |
| acute_crash_2022 | baseline | 18.2 | −10.4 | −1.33 |
| acute_crash_2022 | +gold 10% | **15.2** | −10.6 | −1.89 |
| crash_year_2022 | baseline | 20.8 | −10.5 | −0.94 |
| crash_year_2022 | +gold 15% | 23.2 | −11.8 | −1.49 |

\*RUONIA-excess on a fixed 15% basis — see the metric caveat. Full table:
`results/research/gold/gold_cert_report.md`. Best acute-crash MaxDD cut = **2.95pp**
(+gold 10%), just under the 3pp bar; gold **increases** MaxDD in the crash *year*.

## Honest findings

1. **The deposit anchor holds, robustly.** On raw total return the 100%-deposit leg crushes
   the allocation (full 61% vs 17%; high-rate 22% vs 9%), and on the *proper* high-rate window
   its risk-adjusted return is legitimately positive (Sharpe +0.94 vs −0.52). Nothing here
   changes the deposit-anchored conclusion.
2. **Gold shaves drawdown modestly, but does not clear the bar and is never a free lunch.**
   In four of five windows a gold sleeve *lowers* portfolio MaxDD (full −2.0pp, acute crash
   −3.0pp, high-rate −3.7pp, easing −1.9pp) — but the **best acute-crash cut is 2.95pp, just
   under the pre-registered 3pp bar**, and in the **crash year gold *increases* MaxDD**
   (+0.5…+2.4pp) as the give-back drags. And it **worsens the risk-adjusted return (Sortino)**
   in *every* regime, because zero-yield gold is a drag against the deposit/equity carry. So
   the strict crash de-risking verdict is **`NO`** (the bar is not moved to fit — anti-overfit).
3. **The 2022 ruble hedge was a ~2-week flash, not holdable.** Gold spiked ~+60%
   (2022-02-23→03-10) as the ruble flash-crashed — but capital controls force-recovered the
   ruble, so the gold leg's **raw price** round-tripped to **−17.1%** over the crash year
   (acute −12.8%); the modelled **net** leg ended **−35.1%** (the extra drag is the
   conservative daily-mark NDFL + TER, which over-states gold's tax cost — it makes gold look
   *worse*, never better). A quarterly-rebalanced holder would NOT have captured the spike;
   "buy gold to hedge the sanctions tail" is a mirage at holding horizons. Gold's daily-return
   correlation to equity in the crash is genuinely **negative (−0.13)** — it is uncorrelated,
   it just isn't a *durable* RUB hedge once controls intervene.

## Recommendation

**Do not add gold as a crash hedge.** It does not clear the pre-registered ≥3pp crash
de-risking bar, it *increases* drawdown in a prolonged crash year, and it worsens
risk-adjusted return in every regime. The one thing it does — shave ~2–3.7pp off MaxDD in
calm/acute regimes — is real but marginal and is bought with a Sortino cost; a deeply
drawdown-averse investor *could* hold a small (≤10%) gold sleeve knowing it trades
risk-adjusted return for slightly lower drawdown, but it must be labelled exactly that:
**a marginal de-risker, NOT alpha, NOT a deposit-beater, NOT a reliable sanctions-tail
hedge** (that hedge lasts ~2 weeks and is reversed by capital controls). This is consistent
with the broader honest pattern: the MOEX toolkit has no easy edge; the system's real
defence remains **allocation** (deposit anchor + bounded passive equity), which it already
does.

**N=1 caveat:** the crash evidence rests on a single 2022 episode (and a single, atypical,
capital-controlled one); it is suggestive, not statistically robust.

## Reproduce (token-free, offline)

```
uv run python scripts/research/fetch_gold_panel.py     # refresh the committed snapshot (network)
uv run python scripts/research/run_gold_sleeve.py      # deterministic cert from the snapshot
```

Artifacts: `results/research/gold/{panel_snapshot.json, gold_cert_summary.json,
gold_cert_report.md}`. Pure simulator + verdict: `src/finalayze/backtest/gold_sleeve_lab.py`
(unit-tested in `tests/unit/test_gold_sleeve_lab.py`); currency fetch in
`MoexISSFetcher.fetch_currency_close_history` (`tests/unit/test_moex_iss_currency.py`).
