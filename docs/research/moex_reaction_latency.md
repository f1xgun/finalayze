# MOEX Reactive-News Alpha Decay — Stocks / OFZ / FX

**Verdict: `MOEX_NEWS_REACTION__MOEX_REACTIVE_UNCAPTURABLE_NET`**

Measurement only, on real MOEX 1-minute candles via **Tinkoff gRPC readonly** (the only sanctioned
MOEX data source — never yfinance). BTC/ETH's reactive-news cert (PR #318) showed a reactive trade
into an already-started 1-minute shock is uncapturable net on the fastest 24/7 market. This ports the
**same** measurement to the real RUB trading universe. No API orders, no real-money execution — a
hard stop. Reproduces offline from the committed snapshot.

- **Window:** 2024-01-01 → 2026-01-01 (24 months), ~520 weekdays.
- **Universe:** 10 liquid stocks (SBER, GAZP, LKOH, GMKN, ROSN, NVTK, TATN, PLZL, MGNT, MTSS) +
  5 benchmark OFZ (26238/26230/26240/26243/26221) + USD/RUB. **15,240 shocks** at ≥5-sigma.
- **ETFs dropped:** their registry FIGIs return no 1-min candles, and an index ETF has no
  idiosyncratic news shock anyway (it tracks the basket).

## Design (identical to the crypto cert, plus two MOEX realities)

A reactive news bot enters an **already-started** move, so the honest metric is the forward path
from the shock bar's **close** (the earliest you can act). A large 1-minute move (≥z·trailing-60min
vol, look-ahead-free) is a proxy for a news event. A **latency ladder** reports mean **net** return
(round-trip cost + 13% NDFL) for entry at t+{1,2,5,15} min and exit at t+{30,60,120} min.

Two MOEX-specific realities:

1. **Session continuity.** MOEX is not 24/7 — the tape has main→evening breaks and overnight/weekend
   gaps. A tested `split_runs_on_gaps` primitive splits each instrument into contiguous runs so
   detection and the forward path never straddle a non-tradeable gap.
2. **Wider frictions, long-only.** MOEX retail round-trip (broker + spread) is wider than crypto — we
   sweep 0/15/30/50 bps, primary 30. Single-name MOEX shorting is unavailable to retail, so only
   **up-shocks** (long) are capturable.

## Result — the same answer as crypto, only sharper

**Per-class money numbers (≥6-sigma up-shocks, best exit, 30 bps round-trip + 13% NDFL):**

| class | n | t+1 true-gross | t+1 net | t+5 net | t+15 net | win-rate | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **stock** | 2465 | **+0.52** | −29.48 | −30.73 | −30.47 | 44.9% | REACTIVE_ALPHA_BELOW_FRICTIONS |
| **ofz** | 1090 | **−2.17** | −32.17 | −31.20 | −29.84 | 45.2% | REACTIVE_ALPHA_ABSENT |
| fx | 15 | +1.10 | −28.90 | −31.18 | −31.32 | 46.7% | thin N — inconclusive |

- **Stocks:** the intraday shock is priced **almost completely** by the time you can act. The mean
  forward path is flat noise around zero (−0.8 / +0.44 / −0.76 bps at t+1/5/30, **no continuation
  half-life**); the **median reverses** (−1.43 / −3.79 / −6.62 bps at t+1/30/120) with a **44.9%**
  win-rate. Best-case reactive t+1 is **+0.52 bps** true gross — essentially zero. Net of frictions
  it is negative at every latency. **Slowness is not the bottleneck — there is no intraday
  continuation to be slow about.** (Crypto had +7.64 bps of gross to erode; MOEX stocks have ~none.)
- **OFZ:** stronger still — the mean forward **reverses at every horizon** (−3.47 → −6.37 bps), so
  the *reactive* (t+1) true gross is **negative** (−2.17 bps). Verdict **ABSENT**: even a
  frictionless, tax-free, maximally reactive entry loses. (The best gross over *all* entry/exit combos
  is a negligible **+0.16 bps** — and it sits at t+15, which is not "reactive", and is ≪ any friction.)
  Bond shocks mean-revert; there is no reactive edge at all.
- **FX:** only 15 up-shocks at z6 — too thin for a firm intraday verdict (see the gap section, which
  is where FX's real story lives).

The verdict is robust across the cost sweep: at the *most generous* 15 bps round-trip, every ladder
cell is still deeply negative for stocks and OFZ.

## Overnight-gap decomposition (MOEX-specific — and the metric is selection-dependent)

Since MOEX is closed overnight, part of a news move lands in the un-tradeable open **gap** (open vs
prior close), upstream of any intraday reactor. How much? **The gap's share of the move depends on
which days you condition on** — an adversarial reviewer caught that "news days = biggest total move"
preferentially selects big-*intraday* days, so it is reported here under three selections:

| class | days | mean \|gap\| bps | mean \|intraday\| bps | on \|total\| days | on \|gap\| days | on \|intraday\| days | days gap dominates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | 5193 | 33.0 | 137.4 | 0.125 | **0.510** | 0.093 | 0.135 |
| ofz | 2525 | 10.3 | 57.5 | 0.074 | **0.486** | 0.073 | 0.101 |
| fx | 469 | 57.6 | 13.3 | **0.972** | **0.995** | 0.137 | 0.768 |

The honest reading:

- **Stocks & OFZ:** the gap-share is genuinely selection-dependent — for stocks it is ~12% on the
  biggest-|total| days (which are intraday-dominated) but **~51% on the biggest-|gap| days**. So "the
  gap is small" is *not* a general claim. What **is** robust: the reactive **intraday** alpha above is
  ~0 / reverting *regardless* of selection, so it is the binding channel for the reactive question —
  and it is empty. (This corrected an earlier draft that overstated "the move is mostly intraday".)
- **FX:** here the gap **dominates under every selection** — ~97% of the news-day USD/RUB move is the
  un-tradeable overnight gap (mean gap 57.6 bps vs intraday 13.3 bps). USD/RUB is driven by
  global-session and policy news that hits while MOEX FX is shut, and its thin intraday session offers
  almost nothing. For FX the reactive intraday trader is downstream of a wall that **is** the move.

## Honest limits

- **Latency axis is bars, = minutes only for liquid stocks.** Liquid names are near-fully
  minute-filled (SBER ~936 bars/day for the ~16h extended session), so bar ≈ minute. Thin OFZ / FX
  have sparse 1-min bars (FX only ~154/day), so their latency axis is approximate — a further reason
  their (already negative / thin) numbers aren't a green light.
- **Survivorship:** the universe is 10 currently-listed most-liquid MOEX names. This is directionally
  **safe** for a *negative* verdict — survivors are the most-priced, hardest names to find a reactive
  edge in, so a broader / noisier universe would only be worse to trade reactively.
- **Evening-session shocks** (lower liquidity) are included. That is conservative *against* an edge —
  thinner conditions are harder, not easier, to capture.
- A large 1-minute move is a **proxy** for a news event (most ≥6-sigma moves are information-driven,
  some are microstructure/liquidity). This measures continuation available to a reactor regardless of
  cause — exactly the reactive-capturability question.
- **No single-name shorting** on MOEX for retail → only up-shocks are capturable; down-shocks are
  reported (as counts) but not tradeable long.
- Sub-second colocated HFT is out of scope and unreachable for an LLM/RSS pipeline by construction.

## Reproduce

```bash
# 1. (optional) re-fetch shocks — Tinkoff readonly, needs FINALAYZE_TINKOFF_TOKEN + certs/, ~25 min
GRPC_DNS_RESOLVER=native uv run python scripts/research/fetch_moex_jump_panel.py

# 2. deterministic cert from the committed snapshot (offline)
uv run python scripts/research/run_moex_jump_study.py
```

Artifacts: `src/finalayze/backtest/jump_response_lab.py` (pure, tested — `split_runs_on_gaps` added),
`scripts/research/{fetch_moex_jump_panel,run_moex_jump_study}.py`, committed snapshot
`results/research/moex_jump/{moex_jump_panel.json, moex_jump_summary.json, moex_jump_report.md}`.

## Relation to prior work

Same conclusion family as the crypto reactive-news cert (`docs/research/news_reaction_latency.md`),
the slow-regime news event study, and the PEAD cert: measured honestly on real data, the edge is
**allocation, not signal**. The MOEX result is even sharper than crypto — the intraday shock leaves
essentially nothing (stocks) or reverses (OFZ), and where a gap-wall exists (FX) it is un-tradeable
by construction. The `16–21%/yr` RUB deposit anchor holds.
