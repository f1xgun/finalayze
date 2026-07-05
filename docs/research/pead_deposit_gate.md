# PEAD Deposit-Gate — does post-earnings drift beat the deposit on MOEX?

**Status:** complete · **Verdict:** `PEAD_DRIFT_DEPOSIT_DOMINATED` · **integration-gate tier:
`REJECT`** (N=1 caveat) · diagnostic / backtest-only — no real money, no orders.

The operator asked to test the **one event edge** the news-event study flagged as the only
lane worth measuring: post-earnings-announcement drift (PEAD) — the documented tendency of
stocks to keep drifting in the direction of an earnings surprise for weeks. On real MOEX
earnings dates it **fails twice**: the market-adjusted drift is **negative** (surprises
*reverse*, not drift), and the strategy is crushed by the ~16–21 % deposit. This closes the
last measurable alpha lane on the same honest note as everything before it.

## Why this experiment (and the data reality)

The news-event study showed fast-news reaction is un-capturable, but named PEAD as the one
event edge that is *slow enough* for retail and possibly real. Testing it honestly runs
straight into MOEX's data wall:

- **No consensus-EPS feed on MOEX** (project decision D-01) and **no `eps_ttm` history**
  (T-Bank fundamentals are point-in-time) — so a *fundamental*-SUE PEAD is **data-blocked**.
- What IS reachable: real **earnings report DATES** via Tinkoff `get_asset_reports`
  (readonly, calendar-only — no EPS) and token-free ISS prices.

So this is the **price-reaction PEAD** (Chan-Jegadeesh-Lakonishok variant): the surprise is
proxied by the announcement-window abnormal return, and the drift is what an investor
captures *after* it. The pre-registered question is the deposit gate:

> Once you condition on a real earnings surprise and measure the post-announcement drift over
> 20/40/60 trading days, net of 0.55 %/side + 13 % NDFL — does a long PEAD strategy beat just
> holding the ~18 % deposit?

## Method (real dates, token-free cert, adversarially reviewed)

- **Data:** earnings report dates for 14 MOEX blue chips via Tinkoff `get_asset_reports`
  (token-gated once; a `t_tech` SDK deserialization bug — an int where a Timestamp is
  expected — is worked around by tolerating ints, `report_date` parses cleanly). Daily OHLC
  for the shares + IMOEX + MCFTRR via the **same** token-free ISS `/history` endpoint (one
  true `TRADEDATE` convention, no cross-source off-by-one). Committed snapshot
  `results/research/pead/pead_panel.json`; the cert reproduces offline.
- **Surprise** = abnormal return (asset − IMOEX) over a **3-session cluster** `[D-1 → the
  session +2 after the scheduled report date]` — the release is often after-close, so the
  reaction bleeds into the next day. **Entry** is the session *after* the cluster (a strict
  gap → no look-ahead).
- **Drift** measured two ways, net of cost + NDFL: the **RAW absolute** return (the deposit
  gate's unit — money in the pocket) and the market-**ADJUSTED abnormal** return (the PEAD
  alpha — does the surprise drift beyond the index?).
- **Deposit gate (formal):** a PEAD long sleeve (deposit idle + long each positive-surprise
  name over its drift window, **riding the name's real daily path**) fed to the pre-registered
  `instrument_integration_gate` (deposit40/equity60 core) → INTEGRATE/PROBATION/REJECT.
- **Window ~2024–2026** (the deep 16–21 % regime; Tinkoff only reaches back 730 days). Issuer
  dedup: SBERP (Sberbank prefs, same earnings as SBER) dropped; same-symbol report dates
  within 30 days (RAS+IFRS pairs) collapsed. Final sample: **40 events across 10 issuers**
  (18 positive / 22 negative surprises).

## Results

### The PEAD signal — market-adjusted abnormal drift (the regime-robust read)

If surprises drift, positive-surprise names should out-return the index over the following
weeks. They do the **opposite**:

| horizon | positive-surprise abnormal drift (median) | strong surprise (\|abn\|≥2 %, n=7) |
| --- | ---: | ---: |
| 20 days | **−1.50 %** | +1.66 % |
| 40 days | **−3.92 %** | −4.63 % |
| 60 days | **−3.79 %** | **−11.42 %** |

The drift is **negative** and a *genuine-surprise* filter (|abn|≥2 %) makes the 60-day
reversal **worse** (−11.4 %), not better. There's a faint short-horizon drift for strong
surprises (+1.7 % at 20d) but it fully reverses by 40–60 days — no durable, tradeable PEAD.
This is beta-neutral (market removed), so it is not merely the falling market.

### The deposit gate — absolute return, net of everything

| horizon | positive-surprise RAW long (median) | % that beat the deposit | buy-the-dip (neg) raw |
| --- | ---: | ---: | ---: |
| 20 days | −0.40 % | 33 % | ~flat |
| 40 days | −3.31 % | 28 % | — |
| 60 days | **−4.06 %** | **22 %** | — |

Median deposit carry over 60 days ≈ **+3.49 %** — above the raw PEAD long, which loses. Only
22 % of positive-surprise events beat the deposit at 60 days. Buy-the-dip after a *negative*
surprise doesn't rescue it either. **Formal gate: `REJECT`** — "no measurable net benefit
over the deposit+equity core".

The raw PEAD long **sleeve** returned **−18.45 %** vs the deposit's **+40.24 %**, but that raw
number is **beta-dominated**: IMOEX itself fell **−28.35 %** over the window, so most of the
sleeve's loss is holding equity through a bear market, **not** the PEAD signal (the ~−4 %
abnormal drift above). The gate's tier is basis-robust (it nets the market out); the raw TR
is shown only to size the beta the strategy must pay for.

## What the adversarial review changed (four skeptics, before ship)

All four confirmed the REJECT is **correct and conservative** (no look-ahead; sound NAV math;
correct deposit convention; one date axis; the weaknesses cut *toward* the verdict). Their
findings drove real honesty fixes, none of which flipped it:

- **Issuer dedup** — SBER + SBERP had byte-identical earnings dates (same issuer); dropping
  SBERP and collapsing IRAO's RAS+IFRS pairs cut the sample from 52→40 events (22→18 pos).
- **Unit fix** — the "beats deposit" column had compared *abnormal alpha* to the *absolute*
  deposit; it now uses the raw absolute return (the deposit gate is an absolute-return test).
- **Real-path sleeve** — the sleeve had used a flat per-bar factor that smoothed daily vol to
  ~4 % vs the real ~21 %, handing the gate a fictitiously smooth curve (the same smoothing
  artifact real estate's weekly index had). It now rides each name's real daily returns, so
  the gate scorecard reflects true risk (and the raw sleeve TR corrected from −44.6 % to
  −18.5 %).
- **Magnitude filter** — 27.5 % of surprises are sub-1 % (measurement noise); the strong
  (|abn|≥2 %) column confirms the reversal rather than softening it.
- **Robust dating + framing** — a 3-session announcement cluster absorbs after-close/next-day
  reaction; the write-up now leads with the beta-neutral abnormal drift and the gate tier, not
  the beta-dominated raw TR.

## Honest limits

- **Regime.** One deep high-rate window. At a ~14 %/yr net deposit bar, deposit-dominance is
  near-**foregone** for any long-equity sleeve — so the load-bearing, more regime-robust
  finding is the **negative abnormal drift (reversal)**, which would need a normal-rate regime
  to retest for genuine drift.
- **N and noise.** 40 events / 10 issuers, ~27 % sub-1 % noise-band surprises. A diagnostic
  case study, not a statistically-powered test.
- **No consensus EPS** (D-01): this is a price-reaction proxy, not a fundamental SUE; the
  fundamental version is data-blocked. **Short exclusion**: negative-surprise shorts aren't
  retail-accessible on MOEX, so only long legs enter the verdict.

## Recommendation

**Do not build a PEAD strategy for this book.** On real MOEX earnings 2024–2026 the surprise
**reverses** (negative abnormal drift, worsening with horizon and with surprise strength), and
even the raw long is deposit-dominated (gate `REJECT`). This closes the last measurable alpha
lane the news-event study left open, on the same honest note as gold / ЗО / real estate and
fast-news: **the deposit anchor holds; the system's edge is allocation, not signal.** If the
rate regime ever normalises to ~7–8 %, PEAD (and the other candidates) is worth a retest — the
regime-gated allocator is where any of them would rotate in.

## Reproduce

```
# once, token-gated (readonly Tinkoff for report dates + token-free ISS prices):
GRPC_DNS_RESOLVER=native uv run python scripts/research/fetch_pead_panel.py
# deterministic, token-free, from the committed snapshot:
uv run python scripts/research/run_pead_gate.py
```

Artifacts: `results/research/pead/{pead_panel.json, pead_gate_summary.json,
pead_gate_report.md}`. Primitives: `src/finalayze/backtest/pead_lab.py` (`net_window_factor`,
`realpath_window`, `blend_pead_nav`, `daily_factors`; unit-tested). Reuses
`event_study_lab` (abnormal/net math), `allocation_gate` (deposit), and
`instrument_integration_gate` (the formal gate).
