# ЗО (Replacement-Bond) Sleeve Experiment — beyond-MOEX-edge R&D, Phase B

**Status:** complete · **Verdict:** `FX_LINKED_DIVERSIFIER_TAIL_UNTESTED` — ЗО (замещающие
облигации, index `RURPLRUBTR`) is a *genuine, non-redundant, FX-linked* diversifier (confirmed
in-window), but its acute-2022-crash hedge value is **structurally un-backtestable** and
in-window it is **insurance with a cost**. · diagnostic / backtest-only — no real money.

## Why this experiment

Phase B of the beyond-MOEX-edge track (operator chose **золото → затем ЗО**). Replacement
bonds (ЗО) are the standout "remove the regulatory wall" candidate: they pay an FX-linked
(USD-eurobond-successor) coupon + principal but **settle in RUB on MOEX, bypassing Euroclear**
— exactly the hard-currency exposure the all-ruble deposit/OFZ/equity stack lacks, and the
concrete instrument the geo-risk overlay (PR #300) should rotate *into* instead of more RUB.

## The decisive structural limit (pre-registered, honest)

The ЗО index (`RURPLRUBTR`) and the CNY-bond index (`RUCNYTR`) both start **2023-01-03** —
they **postdate the 2022 crash** they would hedge (replacement bonds were *created by* the
2022 freeze that trapped eurobonds). The only 2022-spanning eurobond index (`RUCEU`) shows the
Euroclear freeze slamming shut (a survivorship trap), not a hedge payoff. **So the
acute-2022-tail benefit cannot be backtested.** Phase B measures what *is* observable in-window
(2023–2026, which had real ruble moves) and reports the tail as a forward-structural argument,
never as measured.

## Method (token-free, reuses the Phase-A gold harness)

- **Data (public ISS-REST):** `RURPLRUBTR` (ЗО RUB-TR) + `RUCNYTR` (CNY-bond TR) + `MCFTRR`
  (equity) via the index path; `CNYRUB_TOM` (durable daily FX proxy) + `USD000UTSTOM` (exchange
  USDRUB — daily until the Jun-2024 NCC sanction halts it, then a ~20-month gap) via the
  currency/selt CETS path. Committed snapshot `results/research/zo/panel_snapshot.json`.
- **Date alignment:** index legs carry the MCFTRR MSK-midnight→UTC **T−1** convention; the cert
  shifts them +1 day to the true ISS date to align with the true-dated currency legs (Phase-A
  lesson).
- **FX-linkage:** daily-return **beta** of ЗО on CNYRUB (full window) and on USDRUB
  (pre-sanction clean window only). **Correlation** of ЗО vs equity (diversification) and vs the
  deposit leg (redundancy).
- **In-window diversification blend:** deposit 40% / equity 60% vs + ЗО (carved from equity,
  5/10/15%) through the shared `gold_sleeve_lab.blend_portfolio` (net-NDFL, quarterly rebalance,
  retail cost; deposit free). Windows: full 2023–26, high_rate_2024_25, easing_2025_26 (no
  crash window — data starts 2023).

## Results (real data)

| measure | value | reading |
| --- | ---: | --- |
| ЗО beta vs **CNYRUB** (873 bars) | **0.330** | genuine FX pass-through (durable proxy) |
| ЗО beta vs **USDRUB** (365 pre-sanction bars) | **0.401** | genuine USD pass-through |
| ЗО corr vs **equity** | **0.052** | uncorrelated → real diversifier |
| ЗО corr vs **deposit leg** | **−0.059** | not a redundant RUB carry leg |
| `RUCNYTR` (CNY-bond) beta vs CNYRUB | **0.063** | only *weakly* FX-linked — RURPL is the one |

In-window blend (2023–2026, **no crash**): the ЗО sleeve **cuts MaxDD ~1–2pp** (full
14.5%→12.3% at 15%) but **lowers total return** (full 48.6%→37.9% at 15%) and **worsens
Sortino** — there was no crash in-sample to reward the hedge. Full table:
`results/research/zo/zo_cert_report.md`.

## Honest findings

1. **ЗО is genuinely FX-linked** — ~0.33 CNY / ~0.40 USD daily-return beta. The structural
   ruble-devaluation-hedge mechanism is real and confirmed (the ЗО price revalues with the FX).
   This is materially *better* than gold (Phase A), which was a volatile zero-yield drag.
2. **ЗО is a non-redundant diversifier** — ~0.05 corr to equity and ~−0.06 to the deposit leg.
   It is independent of both the equity and the RUB-rate legs, so it genuinely widens the
   opportunity set (and it pays a coupon, unlike gold).
3. **But the 2022 tail benefit is un-backtestable** — the instrument postdates the crash. Any
   "it would have hedged 2022" claim is a forward-structural argument, never measured here.
4. **In-window it is insurance with a cost** — in the calm/bull 2023–2026 window ЗО shaves
   ~1–2pp off MaxDD but costs ~10pp of total return at a 15% weight and worsens risk-adjusted
   return. With no crash in-sample, the insurance simply did not pay off — exactly what you
   expect from an unexercised hedge.
5. **Use RURPL, not RUCNY** — the CNY-bond index `RUCNYTR` is only weakly FX-linked (0.06 CNY
   beta); the replacement-bond index `RURPLRUBTR` is the FX-linked instrument.

## Recommendation

ЗО (`RURPLRUBTR`) is the **most structurally sound domestic ruble-devaluation hedge found** —
FX-linked, coupon-paying, uncorrelated, RUB-settled (no Euroclear wall). It is the right
instrument for the geo-risk overlay to **rotate into** (instead of trimming equity toward more
RUB). BUT deploy it only as a **small (≤10%), forward-looking FX-tail insurance leg**, labelled
exactly that: its crash payoff is structurally argued but **unproven** (the instrument postdates
the only crash), and in calm regimes it **costs return**. Not alpha, not a deposit-beater, not a
*measured* hedge. As with every beyond-edge candidate, the system's core defence remains
**allocation** (deposit anchor + bounded passive equity).

**N=1 caveat:** one easing cycle, no in-sample crash, ~3.5 years of ЗО history.

## Reproduce (token-free, offline)

```
uv run python scripts/research/fetch_zo_panel.py   # refresh the committed snapshot (network)
uv run python scripts/research/run_zo_sleeve.py    # deterministic cert from the snapshot
```

Artifacts: `results/research/zo/{panel_snapshot.json, zo_cert_summary.json, zo_cert_report.md}`.
Reuses `src/finalayze/backtest/gold_sleeve_lab.py` (the shared sleeve blender) +
`MoexISSFetcher.fetch_currency_close_history` — no new production code.
