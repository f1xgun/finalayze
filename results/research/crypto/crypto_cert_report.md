# Crypto — Cross-Exchange Arbitrage & Trend Sleeve vs the RUB Deposit (Cert)

Trend window `2021-01-05`->`2026-06-10` · 1983 bars · RUONIA-excess 15.0% · symbols BTCUSDT, ETHUSDT.

> Public read-only data. Authorises no order — real-money execution is a hard stop. Crypto carries custody, exchange-counterparty and RU regulatory/access risk no backtest captures.

## BINDING VERDICT: **CRYPTO_ARB_INFEASIBLE__CRYPTO_TREND_GATE_REJECT**

Cross-exchange SPOT top-of-book arbitrage is INFEASIBLE for a RUB retail investor: across 45 real polls of 5 venues the best realisable top-of-book spread is a median 2.605bps (max 4.236bps) — below round-trip taker fees at every tier, so the net per-trip edge is negative (best case -4.764bps incl. a generous 5.0bps withdrawal). Even at ZERO fees the capital-lockup carry would demand ~354.1 profitable cross-venue round trips a year (each a multi-minute on-chain transfer) merely to match the deposit. On the TREND side the gate returns REJECT (risk-adjusted, crash-inclusive). Over this window NO simple crypto holding beat the deposit net of NDFL: buy-and-hold basket 51.53% (BTC-only 65.74%, ETH-only 37.32%) vs the deposit 98.38%, and the basket carried an 82.44% drawdown the deposit never takes. The 90-day TSMOM sleeve returned 66.34% — also below the deposit. Crucially the trend 'edge' is a LOOKBACK-LOTTERY: the basket sleeve TR ranges 66.34%..607.68% across 30/90/180-day lookbacks (x9.16) though all hold long ~55% of days — a handful of fat-tail days dominate, so which lookback 'wins' is NOT knowable ex-ante. The gate REJECTs (tail tested & FAILED): crypto's 2022 crash is in-window and the sleeve RAISED the blended crash-year drawdown (delta +1.122pp) — and the REJECT is over-determined: even setting that veto aside it fails the INTEGRATE bar on marginal Sharpe (+0.0059 vs +0.10) and on full-window MaxDD (delta -1.999pp — RAISED, not the +3pp cut). Same family conclusion as gold/ZO/real estate: risk-adjusted, the deposit anchor holds. HONEST LIMITS: raw crypto TR is highly START-DATE-SENSITIVE (BTC began this window 2021 mid-cycle; a 2023-bottom start flips the raw-return read AND roughly halves the drawdown). Crypto carries 33-82% drawdowns under every start tested (82% in this 2021 window, ~33-66% from a 2023 bottom) — an order of magnitude beyond the deposit's 0%; the magnitude is start-dependent but the deposit-dominant risk gap is not. Arb infeasibility and the 9x lookback fragility are structural. N=1 easing cycle; only SPOT top-of-book arb was measured (funding/basis + triangular are out of scope, but the capital-lockup carry applies to any capital-locking cross-venue play); arb poll is a within-session snapshot; the deposit leg is floored at 0% pre-2022-02-28 so its 98% is a conservative lower bound; and custody/exchange/RU regulatory-access + USDT/P2P acquisition premium are uncosted and one-directional against crypto.

## 1. Cross-exchange arbitrage feasibility
Best realisable top-of-book spread across 5 venues (binance, bybit, coinbase, kraken, okx), 45 polls. Amortised withdrawal 5.0bps/trip.

| gross spread | bps |
| --- | ---: |
| median | 2.605 |
| p90 | 3.242 |
| max | 4.236 |

**Net per-trip edge (bps) after 2 taker legs + withdrawal:**

| fee tier | at median | at p90 | at max |
| --- | ---: | ---: | ---: |
| vip_0.02pct | -6.39bps | -5.76bps | -4.76bps |
| mid_0.075pct | -17.39bps | -16.76bps | -15.76bps |
| retail_0.20pct | -42.39bps | -41.76bps | -40.76bps |

Best-case net edge (max spread, cheapest fee): **-4.764bps**. Even at ZERO fees, the capital-lockup carry demands ~354.1 profitable cross-venue round trips/yr to match the deposit. Verdict: **ARB_INFEASIBLE_FEES_AND_CARRY_DOMINATE**.

## 2. Trend sleeve — total return vs the deposit anchor
| measure | value |
| --- | ---: |
| 100%-deposit total return | 98.38% |
| buy-and-hold BTC/ETH basket (net NDFL) | 51.53% |
| buy-and-hold basket MaxDD | 82.44% |
| buy-and-hold BTC-only (net NDFL) | 65.74% |
| buy-and-hold ETH-only (net NDFL) | 37.32% |
| 90-day TSMOM sleeve (net cost+NDFL) | 66.34% |
| sleeve beats deposit? | False |
| buy-and-hold basket beats deposit (RAW TR)? | False |
| **any simple holding beats deposit?** | **False** |
| **risk-adjusted gate reject?** | **True** |

**Sleeve total return by family x lookback (net):**

| family | LB30 | LB90 | LB180 |
| --- | ---: | ---: | ---: |
| basket | 607.68% | 66.34% | 70.45% |
| BTCUSDT | 310.65% | 70.5% | 205.56% |
| ETHUSDT | 901.71% | 40.49% | 52.89% |

> **Lookback lottery (fragility).** The basket sleeve TR ranges 66.34%..607.68% across the three lookbacks (x9.16, unstable=True) even though all hold long ~55% of days — a few fat-tail days dominate, so no lookback is knowable ex-ante. This dispersion IS the evidence that crypto TSMOM is not a dependable edge.

## 3. Canonical Instrument Integration Gate
Same pre-registered gate as gold (REJECT) / ZO (PROBATION) / real estate (REJECT). Crypto's 2022 crash is in-window, so it is held to the strict INTEGRATE bar.

**GATE TIER: `REJECT`** (proposed weight 0, carved from equity) — tail tested & FAILED: raised crash-year MaxDD by 1.12pp

| scorecard | value |
| --- | ---: |
| window bars / regimes | 1983 / 2 |
| tail backtestable | True |
| delta Sharpe (10% eval) | +0.0059 |
| delta Sortino (10% eval) | +0.0037 |
| delta MaxDD pp (+ = cut) | -1.999 |
| crash-year delta MaxDD pp (+ = raised) | +1.122 |
| max \|corr\| to existing legs | 0.0024 |

## 4. Per-regime sleeve vs deposit
| window | sleeve TR% | deposit TR% | sleeve MaxDD% | beats deposit |
| --- | ---: | ---: | ---: | :---: |
| full_window *(N=1)* | 66.3 | 98.4 | 79.0 | no |
| crash_year_2022 *(N=1)* | -51.4 | 10.5 | 62.6 | no |
| high_rate_2024_25 | 64.5 | 33.8 | 30.0 | yes |
| easing_2025_26 *(N=1)* | 7.4 | 20.3 | 31.9 | no |

_Sleeve idle bars earn the deposit; NDFL on realised gains only (no loss offset). The per-regime crash row uses the 2022-02-21..12-30 MOEX-invasion sub-window (cross-cert comparability); the BINDING crash-year delta uses the gate's calendar-2022 window._