# Crypto — Cross-Exchange Arbitrage & Trend Sleeve vs the RUB Deposit

**Verdict: `CRYPTO_ARB_INFEASIBLE__CRYPTO_TREND_GATE_REJECT`.** Cross-exchange **spot top-of-book**
arbitrage is structurally infeasible for a RUB retail investor (fees ≫ the realisable spread), and a
crypto
trend sleeve is REJECTed by the same deposit-anchored Instrument Integration Gate that rejected
gold, ЗО, real estate, fast-news and PEAD. Crypto is a high-risk **directional** bet whose only
historical edge was un-risk-managed buy-and-hold in a lucky, start-date-sensitive window — which the
risk-aware, crash-inclusive gate correctly declines.

## The question

The operator asked whether to experimentally trade crypto: different strategies, **cross-exchange
spread arbitrage**, and **news-based** trading. We tested the two that are measurable on real public
data; the news lane is settled by transfer, not re-run (see below). All work is measurement only on
**public read-only** data — no orders, no API keys; real-money execution is a hard stop.

## Method

Two independent measurements, both token-free and offline-reproducible from the committed snapshot
`results/research/crypto/crypto_panel.json`:

1. **Arbitrage feasibility.** 45 rounds polling the top-of-book (best bid/ask) for BTC/USDT across
   five public venues (Kraken, Coinbase, Bybit, OKX, Binance). The best *realisable* cross-venue
   spread each round = buy at the cheapest ask, sell at the richest bid. Compared against round-trip
   taker fees at three tiers + an amortised withdrawal/rebalance cost, and against the capital-lockup
   carry (the deposit forgone on funds pre-positioned across venues).
2. **Trend sleeve.** A RUB investor's BTC/ETH price path = Binance daily USD close × CBR official
   USD/RUB (the deposit's numeraire). A time-series-momentum (TSMOM) long/flat sleeve parks idle
   bars in the risk-free deposit (the strategy's *fairest* cash state), nets 10 bps/side trading cost
   and 13% NDFL on realised gains, and is run through the **canonical Instrument Integration Gate**
   against the same MCFTRR net equity leg the gold/ЗО/real-estate battery used. Crypto's 2022 crash
   is in-window, so it is held to the strict INTEGRATE bar (no tail-untestable PROBATION toe-hold).

The RUB deposit leg is the real CBR-archive key-rate−1pp accrual, daily-compounded, net of the
progressive 13/15% NDFL band — identical to every other battery cert.

## Results

### 1. Cross-exchange arbitrage — INFEASIBLE

| best cross-venue gross spread | bps |
| --- | ---: |
| median | 2.605 |
| p90 | 3.242 |
| max | 4.236 |

Round-trip taker fees at every tier exceed even the **max** observed gross spread, so the net
per-trip edge is negative everywhere — best case **−4.76 bps** (max spread, cheapest VIP fee, incl.
a generous 5 bps amortised withdrawal). And the spread is a *top-of-book* number: real depth,
maker/taker asymmetry and inventory rebalancing make it worse, never better.

Even ignoring fees entirely, the **capital-lockup carry** is decisive: to run arb without transfer
latency you pre-position capital on both venues earning 0% while the deposit pays 16–21%. Matching
the deposit at zero fees would need **~354 profitable cross-venue round trips a year**, each a
multi-minute on-chain transfer — before fees turn every trip negative. The persistent *large*
cross-venue spreads that do exist (regional/"premium" markets) are precisely the ones a RU resident
**cannot** access, and any residual spread is compensation for real counterparty risk (withdrawal
suspensions, exchange insolvency — cf. FTX), not free money.

**Scope & conservatism.** Only **spot cross-venue top-of-book** was measured — perp funding/basis
carry and triangular arb are out of scope. But the gross spread is a *frictionless upper bound*
(top-of-book, zero slippage, simultaneous fills that themselves require pre-staged inventory), so
the negative net edge only widens under real execution; and the capital-lockup carry applies to
*any* capital-locking cross-venue play. (There **is** a stable directional signal in the data — OKX
was the richest bid in 45/45 rounds, Coinbase the cheapest ask in 42/45, a ~3 bps regional premium —
exactly what a basis/regional variant would chase, still well below round-trip fees.)

### 2. Trend sleeve — deposit anchor & the lookback lottery

Over the window (2021-01-05 → 2026-06-10, 1983 bars), **no simple crypto holding beat the deposit**
net of NDFL:

| measure | value |
| --- | ---: |
| 100%-deposit total return | 98.38% |
| buy-and-hold BTC/ETH basket (net NDFL) | 51.53% |
| buy-and-hold basket MaxDD | 82.44% |
| buy-and-hold BTC-only | 65.74% |
| buy-and-hold ETH-only | 37.32% |
| 90-day TSMOM sleeve (net cost+NDFL) | 66.34% |
| **any simple holding beats deposit?** | **False** |

The trend "edge" is a **lookback lottery**: the basket sleeve total return ranges **66% … 608%**
across 30/90/180-day lookbacks (×9.16) even though all three hold long ~54.7% of days — a handful of
fat-tail days dominate, so *which* lookback wins is not knowable ex-ante. That dispersion **is** the
evidence that crypto TSMOM is not a dependable edge; the pre-registered 90-day standard trails the
deposit.

**Canonical Integration Gate: `REJECT`.** Crypto's 2022 crash is in-window (tail-backtestable), and
the sleeve **raised** the blended crash-year drawdown (+1.12 pp) while ΔMaxDD −2.0 pp fails the
≥3 pp cut bar — the same risk-adjusted, crash-inclusive test that rejected gold and real estate.

### News-based crypto

Not re-run: our own MOEX event study (PR #315) showed 81–92% of a news shock is priced before a
realistic retail entry, and crypto is 24/7 and bot-dominated — it prices news *faster* than MOEX.
The same disproof applies a fortiori.

## Honest limits

- **Start-date sensitivity (the biggest caveat).** Raw crypto TR is highly window-dependent: BTC
  began this window 2021 mid-cycle (~$29k); a 2023-bottom start would flip the *raw-return* read in
  crypto's favour **and roughly halve the drawdown**. The window is dictated by the equity leg
  (MCFTRR starts 2021) for battery-comparability. The specific **82% drawdown is window-specific**
  (this window spans the full 2022 collapse; a 2023-bottom start gives ~33–66%) — but the
  **deposit-dominant risk gap is regime-robust**: crypto carries 33–82% drawdowns under *every*
  start tested, an order of magnitude beyond the deposit's 0%, and the arb infeasibility + ×9
  lookback fragility are structural.
- **Deposit TR is a conservative lower bound.** The deposit leg is floored at 0% before 2022-02-28
  (~21% of the window; the real 2021 CBR key rate was 4.25→8.5%), so the reported 98.4% deposit
  return *understates* the true anchor — which cuts *for* crypto, and it still loses.
- **N=1 easing cycle**; the arb poll is a **within-session** snapshot (calm market, one
  time-of-day) — the fees-vs-spread conclusion is structural. Daily-bar sampling under-samples 24/7
  intraday moves but is conservative for a once-daily-rebalanced sleeve.
- **Uncosted, one-directional risks against crypto:** custody, exchange-counterparty insolvency, RU
  regulatory/access constraints, and the RU **USDT/P2P acquisition premium** (typically several %) —
  none in any backtest, all cutting against crypto.
- NDFL is charged on realised gains with no loss-offset (slightly conservative against crypto).

## Conclusion

Crypto joins the same family as every other lane we measured: **the edge is allocation, not signal.**
Arbitrage is a fee-dominated mirage for retail; a trend sleeve is a fragile, drawdown-heavy
directional bet the deposit-anchored gate rejects; news is priced before retail can act. In the
16–21% RUB rate regime the deposit anchor holds. If a candidate ever clears this gate at a
normalised ~7–8% regime, the existing regime-gated allocator rotates it in automatically — crypto is
not that candidate.

## Reproduce

```bash
# deterministic, token-free (uses the committed snapshot)
uv run python scripts/research/run_crypto_gate.py

# refresh the public snapshot (token-free public GETs; overwrites the committed panel)
uv run python scripts/research/fetch_crypto_panel.py
```

Artifacts: `src/finalayze/backtest/crypto_lab.py` (tested primitives),
`scripts/research/{fetch_crypto_panel,run_crypto_gate}.py`,
`results/research/crypto/{crypto_panel.json, crypto_cert_summary.json, crypto_cert_report.md}`.
