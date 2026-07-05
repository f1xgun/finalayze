# Reactive-News Alpha Decay — Are We Slow, or Is There No Edge?

**Verdict: `NEWS_REACTION__REACTIVE_ALPHA_BELOW_FRICTIONS__UNCAPTURABLE_NET`**

Measurement only, on real public read-only Binance 1-minute klines (BTC + ETH, 2024-01-01 →
2026-01-01, ~4.0 coin-years, ~2.1M bars). No API keys, no orders — real-money execution is a hard
stop. Reproduces offline from the committed snapshot.

## Why this study exists

The slow-regime news event study (`event_study_lab`, PR #315) showed that in a single-name shock
81–92% of the abnormal move is priced before a realistic **next-daily-open** entry, and concluded
fast-news alpha is not retail-capturable. The operator asked the right follow-up:

> *"Isn't the problem that we react **slowly** (batch / daily), not **reactively** (event-driven,
> within minutes)?"*

That number (81–92% priced before next open) is a measurement of **our pipeline's latency cost**,
not a proof that no intraday edge exists. The slow regime was disproven; the **reactive** regime was
asserted by analogy, never measured. This cert measures it at **minute resolution**.

## Design — conditional forward path after a shock

A reactive news bot does not predict the jump; it enters an **already-started** move. So the honest
question is not "can you forecast the shock" but *"once a large move fires, how much move is left at
t+1 / t+5 / t+15 minutes, net of cost?"*

1. **Detect shocks** (`detect_jumps`, look-ahead-free): bars whose 1-minute return exceeds
   `z_threshold` × the **trailing** 60-minute realised vol. The vol window strictly precedes the
   bar, so a shock never inflates its own denominator (pinned by a unit test).
2. **Forward path** (`forward_path`): cumulative return from the shock bar's **close** — the
   earliest a reactor can act (you only learn of the shock once the bar closes), so `path[0] = 0`.
3. **Latency ladder**: the mean **net** trade return (round-trip cost + 13% NDFL) for a reactor
   entering at t+{1,2,5,15} min and exiting at t+{30,60,120} min.

Up-shocks are the only spot-retail-capturable direction (long); down-shocks are reported
sign-aligned but are short-only. At ≥6-sigma (the primary, unambiguously news-scale cut): **1218
up-shocks, 1206 down-shocks, ~606/coin-year.**

## The answer: both — but latency is second-order

**Money numbers (≥6-sigma up-shocks, best exit horizon, 20 bps round-trip + 13% NDFL):**

| entry latency | best net (bps) |
| --- | ---: |
| t+1 min (maximally reactive) | **−12.36** |
| t+5 min (our LLM/RSS pipeline) | −14.84 |
| t+15 min (slow batch) | −16.96 |

Reactive t+1 **true gross (no cost, no tax): +7.64 bps**; zero trading-cost but post-NDFL **+6.65
bps** (best exit 30 min). Continuation half-life: **12 min.**

Two things are true at once:

- **Latency matters at the margin.** The ladder decays monotonically with reaction time: at *zero
  cost* the t+1 entry captures +6.65 bps, t+5 +4.49, t+15 +2.65. So being reactive genuinely
  captures more of the gross continuation than being slow. The operator's instinct is not wrong.
- **But it's a second-order effect.** Even the *maximally reactive, zero-cost* entry captures only
  ~6.65 bps — single digits. A realistic round-trip friction (≥10 bps taker + shock slippage) turns
  **every cell of the ladder negative**, at every latency. You cannot win this race by reacting
  faster; the signal is simply too small relative to the cost of trading it.

**Latency ladder (≥6-sigma up-shocks, mean net bps):**

| cost (round-trip) | t+1 | t+5 | t+15 |
| --- | ---: | ---: | ---: |
| 0 bps | +6.65 | +4.49 | +2.65 |
| 10 bps | −2.36 | −4.84 | −6.96 |
| 20 bps | −12.36 | −14.84 | −16.96 |
| 30 bps | −22.36 | −24.84 | −26.96 |

(best-exit column shown; full 3-horizon grid in the cert report.)

## The decisive honesty point: the mean is a fat-tail artifact

The mean up-shock continuation looks mildly positive, but the **median reverses**:

| ≥6-sigma up-shock path | t+1 | t+5 | t+30 | t+120 |
| --- | ---: | ---: | ---: | ---: |
| **mean** (bps) | −1.09 | +1.39 | +6.56 | +6.33 |
| **median** (bps) | −1.53 | −1.60 | −2.02 | −4.49 |

The reactive long **win-rate is 49.4%** — worse than a coin flip. So the positive *mean* is dragged
up by a minority of big-continuation shocks while the **typical shock reverses**. This is the same
lottery character as the crypto TSMOM lookback finding: a positive average that is not a harvestable
edge because you cannot ex-ante select the fat-tail winners. Even before frictions, this alone kills
it; frictions then bury it regardless.

**Shock-size sensitivity** (larger shocks show slightly more mean continuation, but the same
median-reversal / friction story holds):

| z | n_up | n_down | /coin-yr | mean t+1 | mean t+5 | mean t+30 | half-life |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 2387 | 2288 | 1168 | −0.33 | +1.31 | +3.27 | 25 min |
| 6 | 1218 | 1206 | 606 | −1.09 | +1.39 | +6.56 | 12 min |
| 8 | 443 | 402 | 211 | −1.25 | +3.34 | +8.70 | 10 min |

## Why this closes the reactive-news question

- **For us (LLM + RSS pipeline).** Our architecture is definitionally slow: RSS poll interval
  (minutes) + one LLM entity-extraction call (seconds–minutes). We sit at t+5 min or worse, where
  the net is −14.84 bps. We cannot be reactive by construction.
- **For anyone retail.** Even a hypothetical maximally reactive retail bot (t+1 min, event-triggered
  trigger) captures only single-digit-bps gross that any realistic cost erases, on a signal whose
  typical outcome is a reversal. The residual edge, such as it is, lives in a **sub-minute**,
  **sub-basis-point-cost**, **colocated** window — HFT territory, structurally the same
  "retail-hostile" verdict as cross-exchange arbitrage.
- **Against the deposit (honest framing).** The frictionless upper bound is *not* a rounding error:
  annualising the zero-cost net over ~304 up-shocks/coin-year gives **~20%/yr**, the *same order* as
  the near-vol-free **16–21%/yr RUB deposit**. But that ceiling is unreachable on **both** counts —
  you cannot harvest the mean (the median reverses, win-rate < 50%), and any realistic cost turns
  every ladder cell deeply negative. So the deposit anchor holds by way of **cost +
  fat-tail-unharvestability**, not by way of negligible magnitude. Edge is allocation, not signal.

## Honest limits

- A large 1-minute move is a **proxy** for a news event. Most 6-sigma moves in liquid BTC/ETH are
  information-driven, but some are liquidations / microstructure. This measures continuation
  available to a reactor **regardless of cause**, which is exactly the reactive-capturability
  question — a headline-timestamped study would be *narrower*, not broader, and is gated on
  news-timestamp precision (the core difficulty a real reactive trader also faces).
- **Minute-close sampling** understates intra-minute slippage during a shock, so the net figures are
  **conservative against** any edge (real slippage is worse).
- True **sub-second colocated HFT** is a different regime this 1-minute panel cannot resolve — and
  one an LLM/RSS pipeline can never reach, so it is out of scope by construction, not by omission.
- Down-shocks are short-only (not spot-retail-capturable) and reported sign-aligned for completeness.
- Crypto is used as the **fastest, cleanest, token-free** testbed (24/7, second-precise, no MOEX
  rule since crypto is not MOEX). MOEX single names digest news *more slowly*, which could leave a
  **wider** post-shock continuation window — but MOEX is not more capturable, for the *opposite*
  reasons: higher frictions (wider spreads, thinner books), no retail single-name shorting (most
  shocks are bad-news, short-only), and the same deposit anchor. The slow-regime MOEX event study
  (`news_event_study.md`) already measured that its drift was **not systematically capturable net**.
  So crypto answers the *speed* question cleanly; MOEX's answer is governed by frictions + shorting
  + allocation, not by a narrower window.

## Reproduce

```bash
# 1. (optional) re-fetch the raw shocks — ~2M 1-minute bars, token-free public GET, ~20 min
uv run python scripts/research/fetch_jump_response_panel.py

# 2. deterministic cert from the committed snapshot (offline)
uv run python scripts/research/run_jump_response_study.py
```

Artifacts: `src/finalayze/backtest/jump_response_lab.py` (pure, tested primitives),
`scripts/research/{fetch_jump_response_panel,run_jump_response_study}.py`, committed snapshot
`results/research/jump_response/{jump_panel.json, jump_response_summary.json, jump_response_report.md}`.

## Relation to prior work

Same conclusion family as the slow-regime news event study (`docs/research/news_event_study.md`),
the PEAD earnings-drift cert, and the crypto arb/trend cert: measured honestly on real data, the
edge is **allocation, not signal**. This cert specifically forecloses the *"we were just too slow"*
escape hatch — we measured the reactive regime directly, and it is uncapturable net for any retail
actor.
