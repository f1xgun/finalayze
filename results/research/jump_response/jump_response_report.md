# Reactive-News Alpha Decay — Are We Slow, or Is There No Edge? (Cert)

Window `2024-01-01`->`2026-01-01` (exclusive) · 4.0 coin-years of real 1-minute BTC/ETH klines · vol window 60min · NDFL 0.13.

> Public read-only data. Authorises no order — real-money execution is a hard stop. A large 1-minute move is a proxy for a news event; sub-second colocated HFT is out of scope and unreachable for an LLM/RSS pipeline.

## BINDING VERDICT: **NEWS_REACTION__REACTIVE_ALPHA_BELOW_FRICTIONS__UNCAPTURABLE_NET**

On 4.0 coin-years of real 1-minute BTC/ETH data, 1218 up-shocks and 1206 down-shocks at >=6-sigma (~605.6/yr). This tests the operator's question directly: after a large 1-minute move fires, is our news pipeline's failure a LATENCY problem or is there simply no edge? A reactive bot enters the ALREADY-STARTED move, so the honest metric is the forward path from the shock bar's close. The MEAN up-shock path is mildly positive (-1.09bps at t+1min, 1.39bps at t+5, 6.56bps at t+30) — but the outlier-robust MEDIAN is NEGATIVE at every horizon (-1.53/-2.02/-4.49bps) and the reactive long win-rate is 49.4%. So the positive mean is a FAT-TAIL ARTIFACT: the typical shock reverses, and a few big-continuation shocks drag the average up (same lottery character as crypto TSMOM) — continuation half-life 12 min. That alone makes it un-harvestable, and cost buries it regardless: the maximally reactive entry (t+1min, best exit) captures only 7.64bps TRUE gross (no cost/no tax; even zero trading-cost but post-NDFL is 6.65bps); net of a realistic 20bps round-trip + 13% NDFL it is -12.36bps, our-pipeline latency (t+5min) -14.84bps, slow (t+15min) -16.96bps — negative at EVERY latency and every non-zero cost tier. So the answer to 'are we slow, or is there no edge?' is BOTH-but-latency-is-second-order: the ladder confirms faster captures more gross (a real decay), yet even zero-latency zero-cost yields single-digit bps that frictions erase, so we cannot win this race. VERDICT **REACTIVE_ALPHA_BELOW_FRICTIONS__UNCAPTURABLE_NET**. HONEST LIMITS: a 1-minute move is a PROXY for a news event (most 6-sigma moves in liquid BTC/ETH are info-driven, but some are liquidations/microstructure); this measures continuation available to a reactor regardless of cause, which is exactly the reactive-capturability question. Down-shocks are reported sign-aligned but are SHORT-only (not spot-retail-capturable). Minute-close sampling understates intra-minute slippage during a shock (conservative against any edge); true sub-second colocated HFT is a different regime this 1-minute panel cannot resolve and an LLM/RSS pipeline can never reach. Deposit context (honest): annualising the zero-cost net over ~304.3 up-shocks/coin-yr gives a frictionless UPPER BOUND of ~20.2%/yr — SAME ORDER as the 16-21%/yr deposit, NOT a rounding error. But that ceiling is unreachable on two counts: you cannot harvest the mean (median reverses, win-rate < 50%), and any realistic cost turns every ladder cell deeply negative. So the deposit anchor holds via COST + fat-tail-unharvestability, not a negligible magnitude.

## 1. Money numbers — the reactive vs slow answer
| entry latency | best net (bps) | 
| --- | ---: |
| t+1 min (maximally reactive) | -12.36 |
| t+5 min (our LLM/RSS pipeline) | -14.84 |
| t+15 min (slow batch) | -16.96 |

Reactive t+1 TRUE gross (no cost, no tax): **7.64bps** (best exit 30min); zero trading-cost but post-NDFL: **6.65bps**. Up-continuation half-life: **12** min. Annualising the zero-cost net over ~304.3 up-shocks/coin-yr gives a frictionless UPPER BOUND of **~20.2%/yr** — same order as the deposit, so the anchor holds by way of COST + fat-tail-unharvestability (median reverses, win-rate < 50%), not negligible magnitude. All net figures charge a 20bps round-trip + 13% NDFL on gains.

## 2. Latency ladder (>= 6-sigma up-shocks, mean NET bps)
Rows = reaction latency; columns = hold horizon. Positive = a reactor captures net edge.

**cost0bps round-trip:**

| entry \ exit | exit30 | exit60 | exit120 |
| --- | ---: | ---: | ---: |
| t+1 | 6.65 | 5.66 | 6.45 |
| t+2 | 6.26 | 5.27 | 6.05 |
| t+5 | 4.49 | 3.48 | 4.24 |
| t+15 | 2.65 | 1.63 | 2.37 |

**cost10bps round-trip:**

| entry \ exit | exit30 | exit60 | exit120 |
| --- | ---: | ---: | ---: |
| t+1 | -2.36 | -3.5 | -2.59 |
| t+2 | -2.81 | -3.95 | -3.05 |
| t+5 | -4.84 | -6.0 | -5.12 |
| t+15 | -6.96 | -8.12 | -7.28 |

**cost20bps round-trip:**

| entry \ exit | exit30 | exit60 | exit120 |
| --- | ---: | ---: | ---: |
| t+1 | -12.36 | -13.5 | -12.59 |
| t+2 | -12.81 | -13.95 | -13.05 |
| t+5 | -14.84 | -16.0 | -15.12 |
| t+15 | -16.96 | -18.12 | -17.28 |

**cost30bps round-trip:**

| entry \ exit | exit30 | exit60 | exit120 |
| --- | ---: | ---: | ---: |
| t+1 | -22.36 | -23.5 | -22.59 |
| t+2 | -22.81 | -23.95 | -23.05 |
| t+5 | -24.84 | -26.0 | -25.12 |
| t+15 | -26.96 | -28.12 | -27.28 |

## 3. Decay curves by shock size (mean cumulative forward, bps)
The mean move already realised by each horizon after the shock's close. If the curve is flat/negative past t+1, the move is fully priced by the time a reactor can act.

| z | n_up | n_down | /yr | up t+1 | up t+5 | up t+30 | up t+120 | up half-life |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 2387 | 2288 | 1167.9 | -0.33 | 1.31 | 3.27 | 4.76 | 25 |
| 6 | 1218 | 1206 | 605.6 | -1.09 | 1.39 | 6.56 | 6.33 | 12 |
| 8 | 443 | 402 | 211.1 | -1.25 | 3.34 | 8.7 | 8.38 | 10 |

Outlier-robust MEDIAN up-shock path at z=6 (bps): t+1 -1.53, t+5 -1.6, t+15 -2.41, t+30 -2.02, t+60 -3.42, t+120 -4.49.

Reactive long win-rate (enter t+1, exit 60) at z=6: **49.4%**.

## 4. Reading
- **t+1 net <= 0** → slowness is NOT the bottleneck: even a maximally reactive entry loses net of frictions. The move is priced faster than any retail actor (LLM/RSS) acts.
- **t+1 net > 0 but t+5/15 <= 0** → latency IS the bottleneck, but the edge lives in a sub-minute window that a colocated HFT owns and our pipeline can never reach.
- The frictionless upper bound (~20.2%/yr) is deposit-competitive, but unreachable on BOTH counts: you cannot harvest the mean (median reverses, win-rate < 50%) and any realistic cost turns every cell negative. The deposit anchor holds by way of cost + fat-tail, not magnitude. Same family as the slow-regime news event study — edge is allocation, not signal.