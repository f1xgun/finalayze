# PEAD deposit-gate cert -- does post-earnings drift beat the deposit?

**Verdict: `PEAD_DRIFT_DEPOSIT_DOMINATED`**  ·  integration-gate tier: **`REJECT`** (N=1 caveat)

Window 2024-01-01..2026-07-05 (the 16-21% high-rate regime -- a ~14%/yr NET risk-free bar, so deposit-dominance for ANY long-equity strategy is close to foregone here; the more regime-robust read is the market-ADJUSTED abnormal drift). 40 MOEX earnings events across 10 issuers. Price-reaction PEAD (no consensus EPS, D-01): surprise = announcement abnormal return vs IMOEX; drift net of 0.0055/side + 0.13 NDFL from the post-cluster entry.

## The PEAD signal -- market-ADJUSTED abnormal drift (the regime-robust read)

If earnings surprises DRIFT, positive-surprise names should out-return the index over the following weeks. They do the opposite:

| horizon | pos-surprise abnormal drift | strong-surprise (|abn|>=2%) drift |
| --- | ---: | ---: |
| W20 | -1.50% | +1.66% (n=7) |
| W40 | -3.92% | -4.63% (n=7) |
| W60 | -3.79% | -11.42% (n=7) |

The abnormal drift is **negative and worsens with horizon** -- MOEX surprises REVERSE, not drift -- and a genuine-surprise (|abn|>=2%) filter makes it worse, not better. This is beta-neutral, so it is not just the falling market.

## The deposit gate -- absolute return, net of everything

| horizon | pos-surprise RAW long (median) | % beat deposit | buy-the-dip (neg) raw |
| --- | ---: | ---: | ---: |
| W20 | -0.40% | 33% | -3.82% |
| W40 | -3.31% | 28% | -4.24% |
| W60 | -4.06% | 22% | -4.86% |

Median deposit carry over the 60-day window: **+3.49%** -- above the raw PEAD long. Formal gate: PEAD sleeve tier **REJECT** (no measurable net benefit over the deposit+equity core).

The raw PEAD long sleeve returned **-18.45%** vs the deposit's **+40.24%** -- but that raw number is **beta-dominated**: IMOEX itself fell **-28.35%** over the window, so most of the sleeve loss is holding equity in a bear market, NOT the PEAD signal (which is the ~-5% abnormal drift above). The gate's tier is basis-robust (it nets the market out); the raw TR is shown only to size the beta the strategy must pay for.

## Honest limits

- **Regime.** One deep high-rate regime (2024-2026; Tinkoff `get_asset_reports` only reaches back 730 days). At a ~14%/yr net deposit bar, deposit-dominance is near-foregone for any equity sleeve -- so the load-bearing finding is the NEGATIVE abnormal drift (reversal), which would need a normal-rate regime to retest for drift.
- **N and noise.** 40 events / 10 issuers; ~28% of surprises are sub-1% (measurement-noise band), so the sign is partly noisy -- hence the |abn|>=2% strong-surprise column, which confirms (not softens) the reversal. Diagnostic case study, not a powered test.
- **No consensus EPS on MOEX** (D-01): the surprise is the announcement price reaction, not a fundamental SUE (no eps_ttm history to build one). A fundamental-SUE PEAD could differ and is data-blocked.
- **Dating.** Tinkoff `report_date` is a scheduled calendar anchor; the surprise uses a 3-session cluster to absorb after-close/next-day reaction, and entry is strictly after it (no leak), but the exact reaction instant is unknown.
- **Issuer dedup + short exclusion.** SBERP (same issuer as SBER) dropped; same-symbol reports within 30 days collapsed. Negative-surprise SHORTs are not retail-accessible on MOEX; only the long legs enter the verdict.

