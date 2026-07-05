# Real-Estate Sleeve Experiment — beyond-MOEX-edge R&D, Phase C

**Status:** complete · **Verdict:** `SMOOTHED_ILLIQUID_DIVERSIFIER_DEPOSIT_DOMINATED` —
real estate (index `MREDC`) is the **strongest of the three "new asset class" candidates**
(the only income-payer, and a genuine equity diversifier that *beat* equity 2022–2026) but it
is **NOT a robust deposit-beater**: once the investable rental-ЗПИФ wrapper fee is charged its
price-only return is crushed by the deposit, and even a generous rental assumption fails to
close the gap. Its low measured risk is partly a **weekly-smoothing artifact** and its
appreciation was largely a **subsidised-mortgage (льготная ипотека) policy** now wound down. ·
**Deposit anchor holds.** · diagnostic / backtest-only — no real money touched.

## Why this experiment

The operator, asked to think through remaining income ideas, chose **"новые классы активов"**.
Two of the three candidates were already settled on `main`: **gold** (Phase A → `NO`, a marginal
de-risker that worsens Sortino and is not a durable RUB hedge) and **ЗО / замещающие облигации**
(Phase B → an FX-linked diversifier that is *insurance with a cost*, already wired into the
geo-risk overlay as the stress-rotation leg). The **one untested** member was **real estate** —
and it is the only candidate that pays *income* (rent), which is exactly what the "получения
дохода" goal asks for. This phase completes the trilogy, honestly.

The pre-registered question is **diversification + income, never alpha**:

> Once you charge the real (illiquid, fee-laden) investable form and model rent honestly, does
> a real-estate sleeve **beat the deposit** — or at least **diversify** the deposit+equity core
> without worsening risk-adjusted return — over 2022–2026?

## Method (token-free, reuses the Phase-A/B sleeve harness)

- **Data (public ISS-REST, no token):** real estate = `MREDC` (МосБиржа/ДомКлик Moscow
  residential price index, RUB/sq.m) via the index path (`load_mcftr_series` secid swap); equity
  = `MCFTRR` net total-return index (the leg real estate is carved from). Committed snapshot
  `results/research/realestate/panel_snapshot.json` (MREDC 232 bars, equity 1107 bars,
  2022-01..2026-06).
- **Sleeves (all NET):** deposit via `accrue_real_risk_free_leg` (CBR key − 1pp, net-NDFL);
  equity = MCFTRR (already net); real estate = `net_index_returns(MREDC)` **then** a continuous
  **2%/yr wrapper TER** (`apply_ter_drag`) — the fee a real rental ЗПИФ charges that the bare
  index hides (Phase-A gold used 0.8%). Rent is a **labelled net overlay** (`accrue_rental_yield`,
  new tested primitive), swept 3/4/6% net (post-cost, post-NDFL) — never a measured number.
- **Allocation:** baseline deposit 40% / equity 60%; real estate carved from equity (5/10/15%),
  fixed-weight, quarterly-rebalanced, per-side retail cost 0.55% on the non-free legs (deposit
  cost-free). Two arm families: `price_*` (MREDC price-only) and `total_*` (+ 4% net rent).
- **Windows:** full, crash_year_2022 (Feb21–Dec30), high_rate_2024_25, easing_2025_26.
- **Date alignment:** both legs come through the index path (MCFTRR MSK-midnight→UTC **T−1**
  convention); the cert shifts both +1 day to the true ISS trade date (the Phase-A/B lesson) so
  window boundaries are honest.

### Two structural limits (pre-registered, honest — the analogues of Phase B's un-backtestable tail)

1. **Smoothing artifact.** MREDC updates ~**weekly** (52 bars/yr vs equity's 250) and is a
   transaction/appraisal index, so its measured volatility and drawdown are **structurally
   understated** vs a traded asset — and its near-zero correlation to equity is partly the same
   artifact (a sticky asset that barely reprices looks "uncorrelated"). The investable rental-ЗПИФ
   wrapper carries the real market volatility + illiquidity + wide bid/ask + 1–3%/yr fees the
   index hides.
2. **Policy-driven appreciation.** The ~+8.5%/yr residential price rise was largely driven by
   subsidised mortgages (льготная ипотека), a programme now wound down — so the historical
   appreciation is **not a forward-looking expectation**.

## Results (real data, date-aligned)

| measure | value | reading |
| --- | ---: | --- |
| corr(real estate, equity) | **0.004** (1108 pairs) | uncorrelated — but partly a smoothing artifact |
| corr(real estate, deposit leg) | **0.042** | not a redundant RUB carry leg |
| MREDC sampling frequency | **52 bars/yr** (vs equity 250) | ~weekly → SMOOTHED |
| 100%-deposit total return | **59.7%** | the anchor |
| real-estate price-only TR (after 2% ЗПИФ fee) | **19.9%** | crushed by the deposit |
| + 3% net rent | 36.9% | does not close the gap |
| + 4% net rent (base) | 43.1% | does not close the gap |
| + 6% net rent (generous) | 56.3% | **still** below the 59.7% deposit |
| **price beats deposit? / base rent? / generous rent?** | **False / False / False** | deposit-dominated |

Full per-window blend table: `results/research/realestate/realestate_cert_report.md`.

**Blend nuance (honest, both directions):** within the deposit40/equity60 frame, adding real
estate *did* help the equity sleeve — it **cut portfolio MaxDD 3–7.5pp** (full 30.2%→22.7% at
15% price) **and raised total return** (full 17.1%→25.8% at 15% total), because residential real
estate **beat the equity it was carved from** (MCFTRR was net-negative 2022–2026). Yet the
pre-registered `diversifies` flag is **`no`** in almost every window, because the RUONIA-excess
**Sortino worsens** (zero-real-yield-relative drag against the deposit basis) — the same
conjunctive MaxDD∧Sortino bar gold/ЗО faced. The single `yes` is easing_2025_26 +RE total 15%
(MaxDD 9.7%→6.5% **and** Sortino −1.30→−1.22) — one window, N=1.

### The honest case *against* this verdict (anti-cherry-pick)

The daily-mark NDFL in `net_index_returns` **over-taxes** real estate — a >3-year ЛДВ or
primary-residence hold is often price-gain-**exempt**, the assumption that cuts hardest against
it. Removing the price NDFL (still charging the 2% wrapper fee) lifts price-only to **32.6%**,
and with the generous 6% rent to **73.0%** — which **would** beat the 59.7% deposit. So under the
*most favourable defensible* stack (tax-exempt price + generous commercial-grade rent) real estate
nominally wins. But that best case still rests on all four fragile props at once: the smoothing
artifact (real risk hidden), non-repeatable policy-driven appreciation, a lumpy illiquid ЗПИФ
wrapper, and a generous yield. It is **not robust** — hence the deposit-dominated verdict stands,
with the favourable bound disclosed rather than buried.

## Honest findings

1. **The deposit anchor holds.** 100% deposit returns 59.7% net; the best realistic
   real-estate arm (price + 4% net rent, wrapper fee charged) returns 43.1%, and even a generous
   6% arm returns 56.3% — both below the deposit. In the 16–21% rate regime nothing here beats
   just holding the deposit.
2. **Real estate is the strongest of the three candidates.** It is the only income-payer, it is
   genuinely uncorrelated to equity, and it materially **outperformed equity** over 2022–2026
   (positive vs MCFTRR-negative). Inside the equity sleeve it lowered drawdown and raised return —
   it is a better equity-diversifier than gold (a zero-yield drag) and comparable to ЗО (which at
   least is liquid and RUB-settled).
3. **But its low measured risk is partly an illusion.** MREDC's weekly smoothing understates
   volatility/drawdown and inflates its "uncorrelated" reading; the investable rental-ЗПИФ wrapper
   (illiquid, 1–3%/yr fees, wide spread) would expose the risk the index hides.
4. **And its return was policy-driven.** The ~+8.5%/yr appreciation leaned on subsidised
   mortgages now being wound down — forward-looking, residential price growth is more likely to
   track inflation, which the deposit already out-yields.

## Recommendation

**Do not add a real-estate sleeve chasing return** — in the current high-rate regime the deposit
dominates it, its investable form is illiquid and fee-laden, and its historical appreciation was
a policy artifact. A deeply drawdown-averse, long-horizon investor *could* hold a small (≤10%)
real-estate position as an **inflation-linked, equity-diversifying income leg** — but it must be
labelled exactly that: **a diversifier, NOT alpha, NOT a deposit-beater**, whose measured low risk
is partly a smoothing artifact. This closes the "new asset classes" trilogy on the same honest
note as gold and ЗО: the system's real defence remains **allocation** (deposit anchor + bounded
passive equity), which it already does — and the regime-gated allocator (Phases 74–76) is the
right place to let any of these three rotate in **when the rate regime turns** (a normalised
~7–8% key rate is where an inflation-linked income asset like real estate stops losing to the
deposit).

**N=1 caveat:** one easing cycle, one atypical (sticky-price, capital-controlled) crash, ~4.4
years of weekly MREDC history. Suggestive, not statistically robust.

## Reproduce (token-free, offline)

```
uv run python scripts/research/fetch_realestate_panel.py   # refresh the committed snapshot (network)
uv run python scripts/research/run_realestate_sleeve.py    # deterministic cert from the snapshot
```

Artifacts: `results/research/realestate/{panel_snapshot.json, realestate_cert_summary.json,
realestate_cert_report.md}`. New tested primitives: `src/finalayze/backtest/realestate_sleeve_lab.py`
(`accrue_rental_yield`, `bars_per_year`; unit-tested in `tests/unit/test_realestate_sleeve_lab.py`).
The blend/verdict machinery is reused unchanged from `gold_sleeve_lab.py`.
