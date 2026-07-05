# News-Event Study — can a retail investor trade a MOEX news shock?

**Status:** complete · **Verdict:** `JUMP_UNCAPTURABLE__POST_SHOCK_LONG_DRIFT_NOT_SYSTEMATICALLY_CAPTURABLE`
· diagnostic / backtest-only — no real money, no orders, token-free public ISS-REST.

The operator asked whether we should improve the news pipeline to *predict the price
impact of headlines and trade it* — "gasoline crisis → fuel stocks drop", "VK pulled from
the App Store → VK drops", "sanctions lifted → stocks up". This study measures, on those
exact examples plus a direction-blind base rate, whether that is a real retail edge. It is
**not**: by the time an ordinary reader can act the move is already priced (81–92 % of the
abnormal move in the clean single-name shocks), and going long systematically *after* a
shock — chasing the pop or buying the dip — **loses 0.8–3.7 % net** across ~115 real
events. The value of news is **not fast reaction**.

## Why this experiment

The system already ingests and analyses news end-to-end on MOEX (RSS РБК/Интерфакс/ТАСС +
Telegram → `NewsImpactAnalyzer` LLM → `SentimentManager` 4h-decay cache → the
`event_driven` strategy → orders, enabled on all 4 MOEX presets). So "react to the
headline" is already *wired*. The open question the operator raised is whether **pushing
harder on it** — better impact prediction, faster reaction — would pay. Before building,
we measure. This is the honest complement to two existing labs:

- `dividend_event_lab.py` studies the run-up before a **known, scheduled** dividend record
  date (an *anticipated* event).
- This study covers **unanticipated news shocks** — the whole point is reaction *speed*.

The pre-registered question is **can retail capture it, net of everything** — never alpha
for its own sake:

> When an unexpected headline hits a MOEX name, how much of the move is gone before a
> retail investor can act (the un-capturable JUMP), and is there any tradeable DRIFT left
> afterwards, market-adjusted and net of 0.55 %/side cost + 13 % NDFL?

## Method (token-free, deterministic, adversarially reviewed)

- **Data (public ISS-REST, no token):** daily OHLC for 8 shares + the IMOEX benchmark over
  2022-06…2023-11, committed to `results/research/event_study/panel_snapshot.json` (373
  bars each). Both the shares and the index come through the **same** raw `/history`
  endpoint keyed on `TRADEDATE`, so there is one date convention and no cross-source
  off-by-one when subtracting the market (the prior MCFTRR T−1 trap does not apply here).
- **Decomposition (`event_study_lab.py`, unit-tested):** per event, split the **abnormal**
  (asset − IMOEX, beta = 1) move into the **JUMP** (`pre-close → first realistic retail
  entry`) and the **DRIFT** (`entry → close at H = 1/3/5/10 trading days`). Realistic entry
  = next-session open for intraday news, the gap open for overnight/weekend news. Net every
  tradeable drift for a round-trip retail cost + NDFL-on-gain.
- **Two findings, kept apart on purpose** (an adversarial review flagged that conflating
  them overclaims):
  1. **JUMP** — measured on the 5 named events. Lead metric is the **missed favourable
     jump** (an absolute abnormal %); the `jump_share` ratio is shown only when it is a
     meaningful in-`[0,1]` fraction (a near-monotone move) and `n/a` when the move overshot
     and reversed (the near-zero-denominator artifact is never printed as a "share").
  2. **DRIFT** — 4 of the 5 named events are **bad-news short-only** (retail on MOEX cannot
     short single names), so they carry no capturability information; the only long-
     accessible named event is SBER (N=1). The drift verdict therefore rests on a
     **direction-blind base rate**: across all 8 names, take **every** large abnormal daily
     move (a news-shock proxy, not hand-picked) and measure the median **net long** drift of
     chasing the pop / buying the dip, judged against the **deposit carry** over the same
     window.

## The five named events (the JUMP finding)

| event | naive dir | access | day-of abnormal | favourable jump MISSED | net drift @H5 |
| --- | :-: | --- | ---: | ---: | ---: |
| GAZP 2021 dividend cancelled (2022-06-30) | −1 | short-only | **−23.15 %** | **+25.05 %** | +1.27 %¹ |
| Gasoline/diesel export ban (2023-09-21) | −1 | short-only | −0.3…−2.4 % (5/6 down) | +0.53 % | **−2.28 %** |
| VK pulled from App Store (2022-09-26) | −1 | short-only | **−13.85 %** | **+12.58 %** | **−12.28 %** |
| SBER record dividend (2023-03-17) | +1 | **long** | **+7.48 %** | **+8.20 %** | +0.38 % |
| Wagner mutiny weekend (2023-06-26) | −1 | short-only | −1.36 % (gap open) | +0.84 % | −1.90 % |

¹ a *short's* profit (retail cannot short a single MOEX name); 92 % of the −23 % move was
already gone by the next open regardless.

Reading:

- **The jump is un-capturable.** In the clean single-name shocks, 81–92 % of the abnormal
  move is priced before a realistic next-open entry (GAZP `jump_share` 0.92, SBER 0.81);
  the favourable jump *missed* is +25 % (GAZP), +13 % (VK), +8 % (SBER). A scheduled LLM
  reading RSS/Telegram is downstream of that.
- **The "obvious" direction is a trap.** The fuel export ban (the operator's own example)
  produced a *small, mixed* day-of reaction — LKOH actually **rose** (+0.9 % abnormal, the
  bearish read was wrong) — and over the next 5 days the naive "short fuel on the crisis"
  bet **reversed and lost 1.6–4 %** in 5 of 6 names. An export ban curbs export revenue but
  supports domestic supply/margins; the sign requires reasoning about the policy *response*,
  not the headline.
- **Confounds dominate.** VK's −13.9 % abnormal drop was real, but it landed inside the
  mobilisation crash and then **bounced +12 %** — a naive short is crushed on the reversal.
  Wagner broke on a Saturday; the market first traded Monday at a **−1.4 % gap open** you
  cannot act on.

## The direction-blind base rate (the DRIFT finding — the real tradeable test)

Across all 8 names, every abnormal daily move ≥ the threshold, net LONG drift after a
realistic next-open entry:

| shock ≥ | horizon | chase the pop (n, median net) | buy the dip (n, median net) |
| --- | :-: | ---: | ---: |
| 3 % | H5 | n=115, **−0.83 %** | n=64, **−1.68 %** |
| 3 % | H10 | n=114, **−1.29 %** | n=64, **−1.13 %** |
| 5 % | H5 | n=32, **−0.84 %** | n=20, **−3.68 %** |
| 5 % | H10 | n=32, **−1.82 %** | n=20, **−2.24 %** |

**Every cell is negative.** Systematically going long after a MOEX news shock — whether
chasing an up-move or buying a down-move — loses money net of cost + NDFL, and of course
loses to the ~0.2–0.3 % deposit carry over the same window. This is the decisive evidence:
the one long-accessible *named* event (SBER: net-positive at H1/H5/H10, +3.93 % at H10,
beating the deposit) is **idiosyncratic** — the systematic base rate over dozens of shocks
is negative, so SBER was that name's 2023 uptrend, not a repeatable news edge.

## Honest limits (pre-registered)

- **N and selection.** The 5 named events are hand-picked ex-post-large shocks, which
  biases the JUMP finding toward "uncapturable". The DRIFT finding leans on the
  direction-blind base rate (~115 + ~64 events) precisely to avoid that bias.
- **Daily data.** With no intraday bars the whole reaction-day move is charged to the
  un-capturable jump, so measured `jump_share` is an **upper bound** — conservative for the
  "retail is slow" thesis (a same-day-close aggressive entry is reported alongside).
- **Short-only exclusion.** 4/5 named events are bad news; retail on MOEX largely cannot
  short single names, so those are excluded from the capturability verdict and shown only
  for the jump.
- **Benchmark.** IMOEX is price-return; a total-return benchmark would shave ~0.15 %/5d off
  measured long drift — conservative for the long test (it would push the marginal cases
  further negative).
- **N=1 long event.** SBER alone is not evidence; the base rate is what generalises it (and
  it says: not systematically capturable).

## Recommendation

**Do not build a system to predict-and-trade the price impact of breaking news.** It fails
on all three axes we can measure: the move is priced before a realistic retail entry, the
"obvious" direction frequently reverses and loses, and — decisively — the systematic base
rate of trading *after* a shock is negative net of costs. This closes the "react to the
headline" idea on the same honest note as the new-asset-class trilogy (gold / ЗО / real
estate): the system's real defence remains **allocation** (the deposit anchor + bounded
passive equity), which it already does.

Where news genuinely *does* have defensible value — and where any further effort should go —
is **not** fast reaction:

1. **De-risk overlay (allocation, not alpha).** The LLM already detects
   sanctions/geopolitical events; wiring that into an actual book de-risk (today the
   de-risk fires off a market-correlation proxy, `rub_oil_regime.py`, not the news itself)
   is the one legitimate product improvement. Defensive, fits the deposit-anchor thesis.
2. **Anticipated, slow drift (PEAD).** The only event edge that is both measurable on the
   history we have and slow enough for retail is post-earnings-announcement drift — a
   *scheduled* event, studied separately in `dividend_event_lab.py`, not a fast news
   reaction. That is the honest place to keep looking, through the same deposit-anchor gate.

## Reproduce (token-free, offline for the cert)

```
uv run python scripts/research/fetch_event_study_panel.py   # refresh the committed snapshot (network, public ISS)
uv run python scripts/research/run_event_study.py           # deterministic cert from the snapshot
```

Artifacts: `results/research/event_study/{panel_snapshot.json, event_study_summary.json,
event_study_report.md}`. Primitives: `src/finalayze/backtest/event_study_lab.py`
(`decompose_event`, `jump_share_reliable`, `net_abnormal_long_return`,
`net_after_costs`; unit-tested in `tests/unit/test_event_study_lab.py`). Adversarially
reviewed (look-ahead, abnormal-return math, data faithfulness, methodology) before ship —
the math and data verified byte-reproducible; the review's overclaim finding (a confident
disproof from 4/5 short-only events) is what drove the JUMP/DRIFT split and the
direction-blind base rate above.
