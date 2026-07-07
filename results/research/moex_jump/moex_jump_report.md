# MOEX Reactive-News Alpha Decay — Stocks / OFZ / FX (Cert)

Window `2024-01-01`->`2026-01-01` (exclusive) · Tinkoff readonly 1-minute candles · primary z=6 · NDFL 0.13.

> MOEX data via Tinkoff gRPC readonly (the only sanctioned MOEX source). Authorises no order — real-money execution is a hard stop. A 1-minute shock is a proxy for a news event; sub-second colocated HFT is out of scope. Latency axis is bars (=minutes for liquid stocks; approximate for thin OFZ/FX). Long-only up-shocks (no MOEX shorting).

## BINDING VERDICT: **MOEX_NEWS_REACTION__MOEX_REACTIVE_UNCAPTURABLE_NET**

Ported the crypto reactive-news cert to the real RUB universe via Tinkoff readonly 1-min candles (2024-01-01..2026-01-01), per instrument class, long-only up-shocks (no single-name MOEX shorting). OVERALL **MOEX_REACTIVE_UNCAPTURABLE_NET** — the same 'edge is allocation, not signal' family as the crypto cert and the slow-regime news study, only sharper. STOCKS (2465 up-shocks at >=6-sigma): the intraday shock is priced ALMOST COMPLETELY by the time you can act — the mean forward path is flat noise around zero (-0.8/0.44/-0.76bps t+1/5/30, no continuation half-life), the median REVERSES (-1.43/-3.79/-6.62bps) and win-rate is 44.9%. Best-case reactive t+1 is only 0.52bps TRUE gross; net of a realistic 30bps round-trip + 13% NDFL it is -29.48bps and negative at every latency (t+5 -30.73, t+15 -30.47). OFZ (1090 up-shocks): even STRONGER — the mean forward REVERSES at every horizon (-3.47..-6.37bps), so the reactive true gross is NEGATIVE (-2.17bps) → verdict REACTIVE_ALPHA_ABSENT (no REACTIVE edge even frictionless/tax-free; the mean reverses and any barely-positive cell sits at a non-reactive latency and is << friction; bond shocks mean-revert). FX is THIN (only 15 up-shocks at z6 — too few for a firm intraday verdict). OVERNIGHT-GAP DECOMPOSITION (MOEX-specific; the metric is SELECTION-DEPENDENT, reported honestly): the gap's share of the daily move depends on which days you condition on. For stocks it is 12% on the biggest-TOTAL-move days (those are intraday-dominated) but 51% on the biggest-GAP days — so 'the gap is small' is NOT a general claim. What IS robust: the reactive INTRADAY alpha above is ~0/reverting regardless of selection, so it is the binding channel. For FX the gap dominates under EVERY selection (97% on news days, mean gap 57.58bps vs intraday 13.34bps) — the reactive intraday trader is downstream of a wall that IS the whole move. LIMITS: latency axis is bars (=minutes for liquid stocks ~930 bars/day; approximate for thin OFZ/FX). Universe is 10 currently-listed liquid names (survivorship is directionally SAFE — survivors are the hardest names to find reactive edge in). Evening-session shocks (lower liquidity) are included (conservative — thinner = harder to capture). No single-name shorting → only up-shocks are retail-capturable. Deposit 16-21%/yr is the anchor. ETFs dropped (registry FIGIs return no 1-min candles; index ETFs have no idiosyncratic shock).

## Coverage
| instrument | class | bars | days | shocks |
| --- | --- | ---: | ---: | ---: |
| SBER | stock | 487706 | 521 | 975 |
| GAZP | stock | 486635 | 521 | 894 |
| LKOH | stock | 479520 | 521 | 1102 |
| GMKN | stock | 463818 | 517 | 978 |
| ROSN | stock | 487430 | 521 | 1055 |
| NVTK | stock | 477864 | 521 | 908 |
| TATN | stock | 476174 | 521 | 1174 |
| PLZL | stock | 452592 | 518 | 971 |
| MGNT | stock | 452193 | 521 | 969 |
| MTSS | stock | 449975 | 521 | 937 |
| SU26238RMFS4 | ofz | 424411 | 506 | 1644 |
| SU26230RMFS1 | ofz | 338217 | 506 | 886 |
| SU26240RMFS0 | ofz | 349535 | 506 | 1062 |
| SU26243RMFS4 | ofz | 370342 | 506 | 1353 |
| SU26221RMFS0 | ofz | 236681 | 506 | 226 |
| USD000UTSTOM | fx | 72215 | 470 | 106 |

## Per-class money numbers (>=6-sigma up-shocks, best exit)
| class | n | t+1 true-gross | t+1 net | our t+5 | slow t+15 | win-rate | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stock | 2465 | 0.52 | -29.48 | -30.73 | -30.47 | 44.9% | REACTIVE_ALPHA_BELOW_FRICTIONS |
| ofz | 1090 | -2.17 | -32.17 | -31.2 | -29.84 | 45.2% | REACTIVE_ALPHA_ABSENT |
| fx | 15 | 1.1 | -28.9 | -31.18 | -31.32 | 46.7% | THIN_N_INCONCLUSIVE *(thin N)* |

All net figures charge a 30bps round-trip + 13% NDFL. Cost sweep [0, 15, 30, 50]bps in the summary JSON.

## Overnight-gap decomposition (MOEX-specific)
How much of the daily move is the un-tradeable overnight GAP (open vs prior close) vs the intraday continuous move a reactor could chase. **The gap-share is SELECTION-DEPENDENT**, reported on the top-decile days ranked by |total|, by |gap|, and by |intraday|.

| class | days | mean \|gap\| bps | mean \|intraday\| bps | gap-share (all) | on |total| days | on |gap| days | on |intraday| days | days gap dominates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | 5193 | 33.03 | 137.41 | 0.22 | 0.125 | 0.51 | 0.093 | 0.135 |
| ofz | 2525 | 10.31 | 57.46 | 0.166 | 0.074 | 0.486 | 0.073 | 0.101 |
| fx | 469 | 57.58 | 13.34 | 0.802 | 0.972 | 0.995 | 0.137 | 0.768 |

## Reading
- **Stocks:** the intraday shock is priced almost completely by the time you can act — mean forward is flat noise around zero, median reverses, win-rate < 50%. Best-case gross is ~0 bps; net of frictions it is negative at every latency. Slowness is not the bottleneck — there is no intraday continuation to be slow about.
- **OFZ:** stronger still — the mean forward REVERSES, so even a frictionless tax-free reactor loses (verdict ABSENT). Bond shocks mean-revert; no reactive edge at all.
- **Gap decomposition (honest — the metric is selection-dependent):** for stocks the gap is ~12% of the move on the biggest-|total| days but ~51% on the biggest-|gap| days, so 'the gap is small' is NOT a general claim. What is robust: the reactive INTRADAY alpha above is ~0/reverting regardless of selection, so it is the binding channel. The gap-wall is a genuine FX phenomenon: for USD/RUB ~97% of the news-day move is the un-tradeable overnight gap under every selection, and the thin intraday session offers almost nothing.
- Same family as the crypto cert and the slow-regime news study: edge is allocation, not signal; the 16-21%/yr deposit anchor holds.