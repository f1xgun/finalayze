# Budget-Diversification Autonomous Program — instrument integration

**Mandate (operator, 2026-06-30):** continuously, autonomously try instruments of different
risk to diversify the budget; for each, design how its contribution is measured + the threshold
for full integration; implement, verify, merge on green CI; loop until stopped. Operator does NOT
pre-approve hypotheses — Claude decides them.

**Hard constraints (always):** real-money/LIVE execution = HARD STOP (operator confirmation only);
MOEX data = Tinkoff/ISS only (never yfinance); sandbox permitted; all R&D diagnostic/backtest-only;
subscription billing only.

This is the durable program ledger (survives sessions). Each instrument is one iteration:
`hypothesis → analyze/fetch (token-free) → run through the Instrument Integration Gate →
verdict → implement (integrate passers / document rejects) → adversarial review → PR → merge`.

## The Instrument Integration Gate (the measurement + threshold standard)

`src/finalayze/backtest/instrument_integration_gate.py` — a reusable L5 measurement layer.
Given a candidate net-TR sleeve + metadata `(risk_tier, intended_role)`, it measures the
**marginal contribution** to the crash-inclusive deposit-anchored core (deposit 40% + equity 60%,
2022–2026 so a real crash is in-window) and emits a pre-registered 3-tier verdict. Marginal
deltas (aug − base) are basis-robust (the fixed-15% RUONIA basis cancels). Reuses the reviewed
`gold_sleeve_lab` blender + `_metrics`/`regime_split`/`net_index_returns`/`accrue_real_risk_free_leg`.

**Scorecard (per candidate):** window_bars, regimes_covered, tail_backtestable (covers the 2022
crash), marginal Δexcess-Sharpe / Δexcess-Sortino / ΔMaxDD-pp (at a 10% eval weight),
crash-year ΔMaxDD-pp (positive = raised the crash drawdown), toe-hold Δexcess-Sortino (at 3%),
correlation to each existing leg, anti-hollow reproduction.

**Pre-registered tiers (NEVER moved to fit a candidate):**
- **INSUFFICIENT_DATA** — window_bars < 300 or anti-hollow fails. "Cannot judge", not "failed".
- **INTEGRATE** — tail_backtestable AND 2 regimes AND ΔSharpe ≥ +0.10 AND ΔSortino ≥ 0 AND
  ΔMaxDD ≥ +3.0pp AND crash-year ΔMaxDD ≤ 0 AND max-corr ≤ 0.60. (A genuine free improvement.)
- **REJECT** — redundant (max-corr > 0.60) OR (tail_backtestable AND crash-year ΔMaxDD > 0 —
  tested and raised the crash drawdown) OR (tail_backtestable AND ΔSortino(10%) < −0.10) OR no
  benefit.
- **PROBATION** — uncorrelated (corr-equity ≤ 0.35, corr-deposit ≤ 0.20) AND real if small
  drawdown relief (ΔMaxDD ≥ +1.0pp) AND toe-hold ΔSortino ≥ −0.10 AND tail UN-backtestable (or
  single-regime). A structurally-sound but unproven hedge → small forward-looking toe-hold.

**Weight rule:** carve from the role leg (cash→deposit; hedge/diversifier/carry/growth→equity,
never carve a hedge from the near-vol-free deposit anchor). Per-tier nominal caps low 10% /
medium 8% / high 4%; PROBATION fixed 3% toe-hold; REJECT & INSUFFICIENT_DATA 0%.

**Full integration (only on a passer + operator commit):** extend `AssetClass` + the fail-closed
profile loader; generalize `AllocationOrchestrator.run()` (currently hardwired to 3 legs) to
iterate `AssetClass`. The gate authorizes a CONFIG weight, NEVER an order — real money stays a
HARD STOP.

## Validation (the gate must reproduce the hand-done results)

- **Gold** (Phase A, PR #302) → **REJECT** (tail in-window, raised the crash-year drawdown +
  worsened Sortino).
- **ЗО replacement bonds** (Phase B, PR #303) → **PROBATION** (FX-linked + uncorrelated, but the
  2022 tail is un-backtestable → 3% toe-hold).

## Iteration ledger

| # | Instrument | risk / role | source (token-free) | verdict | PR |
|---|---|---|---|---|---|
| A | Gold (GLDRUB) | high / hedge | ISS CETS GLDRUB_TOM | REJECT (de-risk at a cost) | #302 ✅ |
| B | ЗО (RURPLRUBTR) | medium / diversifier | ISS index RURPLRUBTR | PROBATION (FX-linked, tail untested) | #303 ✅ |
| 0 | **Integration Gate framework** | — | — | built (gold→REJECT, ЗО→PROBATION validated) | #304 ✅ |
| 1a | RGBITR fixed-coupon OFZ | medium / carry | ISS index RGBITR | **REJECT** — redundant (corr 0.61 to rate/bond factor); confirms Phase 76 | #305 ✅ |
| 1b | RUCBITR corporate IG | medium / carry | ISS index RUCBITR | **REJECT** — corr 0.59 too high to diversify + regime-limited (data ends 2023-05; ΔSortino −0.07 after the iter-2 clamp fix) | #305 ✅ |
| 1c | RUCBHYTR corporate HY (ВДО) | high / carry | ISS index RUCBHYTR | **REJECT** — HY credit too equity-correlated (0.48) to diversify | #305 ✅ |
| 1d | LQDT money-market | low / cash | ISS shares TQTF | **REJECT** — no material benefit (cash-like, ~flat marginal effect) | #305 ✅ |
| 1e | CNYRUB FX | high / diversifier | ISS CETS CNYRUB_TOM | **REJECT** — uncorrelated but big zero-carry drag (ΔSortino −0.33), tail tested & failed | #305 ✅ |

**Iteration 1 finding (instrument battery):** all 5 candidates REJECT, each for a distinct honest
reason — the gate is discriminating, not a rubber stamp. Nothing clears the deposit+equity core;
the "no easy edge" pattern holds across every risk tier (cash, OFZ duration, IG/HY credit, FX).
ЗО (PROBATION) remains the only non-REJECT — a structurally-sound but unproven FX-tail hedge.

| # | Instrument / test | verdict | PR |
|---|---|---|---|
| GATE-FIX | window-end clamp (a discontinued candidate was forward-filled flat to 2026 → phantom-drag artifact) | fixed + regression-tested; corrected RUCBITR ΔSortino −0.13→−0.07 (tier unchanged) | #306 ✅ |
| 2a | INFLTR inflation-linker OFZ (medium / inflation-hedge) | **INSUFFICIENT_DATA** — only 279 real bars (index discontinued ~2023); "cannot judge", not a fake REJECT | #306 ✅ |
| 2b | gold + CNY multi-hedge basket (3%+3%) | **does NOT help** — cuts crash MaxDD <1pp but drags full-window ΔSortino −0.14; combining two REJECTED hedges doesn't rescue diversification | #306 ✅ |

**Iteration 2 finding:** the inflation linker is un-judgeable (discontinued index → INSUFFICIENT_DATA,
honestly distinguished from "failed"); and a *combination* of the two crash-covering zero-carry
hedges (gold+CNY) does not produce the diversification neither gave alone. The gate-fix (clamp the
eval window to the candidate's real data) is an honesty correction — a short/discontinued series
can no longer fake a REJECT via flat-fill drag.

| # | Instrument / test | verdict | PR |
|---|---|---|---|
| 3 | regime-CONDITIONAL gold hedging (hold gold only on a trailing-stress quarter) | **NO** — DOMINATED by the no-hedge core (worse Sortino −0.79 vs −0.67, MaxDD 31.6 vs 30.3, TR 11.9 vs 17.0) | this PR |

**Iteration 3 finding:** the idea that should have rescued the hedges — hold them ONLY under stress —
*fails*. A look-ahead-safe trailing-drawdown flag fired 13/19 quarters (MOEX was in drawdown most
of 2022-2025) and **mis-timed the rotation**: it buys gold AFTER drawdowns and sells AFTER
recoveries (buy-high-sell-low) plus pays switching turnover, while the 2022 acute gap is too fast
for a quarterly flag. So conditional gold is *dominated* by the plain core — even SMART use of the
hedge does not beat the deposit anchor. (Reusable: `blend_portfolio` gained a `weight_schedule=`
seam for any regime/conditional overlay.)

### META-FINDING (after 3 iterations / 10 instrument-tests + a timing overlay + an event-timing study)
The diversification space for a sanctioned-RU deposit-anchored book is **effectively exhausted at
the static, the conditional, AND the event-timing level**: every RUB fixed-income leg (OFZ duration,
IG/HY credit, money-market, linkers) is **redundant with the rate factor** (corr ≥ 0.4 → REJECT);
every genuinely-uncorrelated asset (gold, CNY, ЗО) is a **zero-carry drag** that can't beat the
~16-21% near-vol-free deposit on risk-adjusted return (REJECT), or an **unprovable-tail hedge** (ЗО →
PROBATION); combining hedges (gold+CNY) doesn't rescue it; **timing** the hedge (conditional)
mis-fires and is dominated by the plain core; and **event-timing** distribution dates (iter-6) is a
non-edge in both directions — the equity dividend run-up has no tradeable drift (net-negative at short
horizons, only equity beta by k=20) and its k5 overlay REJECTs as a redundant equity factor
(corr 0.997), while the bond coupon "run-up" is confirmed to be the НКД
(ACCINT) accounting artifact, not alpha. The deposit anchor genuinely dominates — now proven across
the static, combination, conditional, AND event-timing axes, enforced by a reusable pre-registered
gate.

**Implication for the loop:** further single-instrument / hedge-timing iterations have low expected
value — the honest answer is converging. The remaining constructive moves are forward-looking
INTEGRATIONS of what little is sound, not new REJECT candidates.

| # | Constructive integration | status | PR |
|---|---|---|---|
| 4 | **ЗО PROBATION → geo-risk overlay rotation** — on elevated/high geo-risk the overlay now recommends rotating a small ЗО FX-linked toe-hold (≤3% PROBATION cap; 1.5% elevated / 3% high) INSTEAD of trimming only into more ruble deposit/OFZ. Forward-only advisory; config weight, never an order. | DONE | #308 ✅ |

**Iteration 4 (constructive):** the one hedge that survived the gate (ЗО, PROBATION — FX-linked +
uncorrelated) is now operationalized exactly where it belongs. The geo-risk overlay (#300) already
trims equity on sanctions stress but only toward ruble assets (which don't hedge devaluation); it
now also surfaces `recommended_fx_hedge_pct` — a small ЗО rotation, the structurally-sound (but
tail-unproven, hence toe-hold-capped) destination. `geopolitical_risk.py` brain + the alert
surface it; real money stays a hard stop.

### Iteration 5 — operator question: "bonds at 15-16% vs a 14% deposit — is it captured?" (PER-REGIME)

Operator (2026-07-04) challenged the "deposit dominates" conclusion: bonds currently yield 15-16%
while the best 3-month deposit is ~14%. This surfaced a **regime-blind spot**: the iter-1 gate
REJECTED fixed-coupon OFZ on a **2022-2026 full-window average dominated by the rate-HIKING era**
(where fixed bonds lose price) — it never isolated the current EASING regime. `run_duration_regimes.py`
compares deposit vs floater vs fixed-coupon OFZ per regime (raw total return):

| arm | full TR% | hiking 2023-24 | **easing 2025-26 (LIVE)** |
|---|---:|---:|---:|
| deposit (key−1pp) | 50.2 | 19.6 | 13.7 |
| **ОФЗ floater (SAA holds this)** | 44.2 | 11.4 | **16.7** |
| ОФЗ fixed-coupon (duration) | 9.1 | −9.3 | 12.3 (MaxDD 6.2) |

**Finding — the intuition is RIGHT and mostly ALREADY CAPTURED, via the FLOATER not duration.** In
the live easing regime the OFZ **floater** (the SAA's `OFZ_PK` leg) returned **16.7% vs the deposit's
13.7%** — a bond at ~key out-carries a deposit at key−1pp by ~3pp. So "bonds beat the deposit" DOES
show up in the system. But **fixed-coupon DURATION did NOT win** (12.3%): the curve was inverted
(short > long yields), so locking a lower long yield gave up carry + took a duration drawdown — the
iter-1 fixed-OFZ REJECT holds even per-regime. **Genuine gap:** the floater FALLS with the key, so it
does NOT *lock* today's 15-16% for the future; if the curve un-inverts and a fixed 15-16% bond sits
above the falling deposit, locking that carry is a real forward call the floater can't make (with
duration risk). Components exist (`ru_ofz_pd.yaml`, `bond_duration_rotation.py`) but the fixed-coupon
yield-lock is not wired into the SAA. Diagnostic only; not advice.

### Iteration 6 — event-timing: dividend run-up (equities) + coupon "run-up" (bonds)

Hypothesis (self-chosen): is there a monetizable **event-timing** edge around cash-distribution
dates — buy shares into a *known, announced* dividend record date and capture the pre-payout
"run-up"; and, as the honesty control, does the pre-**coupon** rise in a bond's dirty price carry
any alpha? Both are the classic efficient-market dividend/coupon question. This is the one
event-study where **look-ahead is NOT cheating** — `registryclosedate`/`value` are public ex-ante
(ISS `securities/{SECID}/dividends.json`), so a buy-N-days-before / sell-before-ex trade uses only
information available at trade time. That keeps the door open a crack; the honest prior is still
REJECT.

Ran `scripts/research/run_dividend_event_study.py` on token-free MOEX ISS-REST: 21-name blue-chip
universe, **139 dividend events**, 2021-01-04..2026-06-10 (1363-bar axis), MCFTRR-net as the equity
benchmark. **Ex-date is DETECTED per event from the price series** (the session near `registryclosedate`
whose drop best matches −dividend/price), with a settlement-convention fallback (T+2 → ex = record−1
trading day through 2023; T+1 → ex = record from 2024, MOEX's T+1 migration); the run-up window ends at
LDD = the trading day *before* the detected gap, so the ex-gap is EXCLUDED (see the methodology note —
the first pass hard-coded T+1 for all years and contaminated the run-up with the gap; this is the
corrected re-run). 2022 halt window excluded (0 halt-void events fell in it); cancelled declarations
record-date-void; faded payers (GAZP-zero-post-2022 etc.) KEPT in-panel (anti-survivorship). Per-year:
2021=40, 2022=22, 2023=27, 2024=31, 2025=19; no 2026 events in-window yet.

**Two arms, both NET of 2×0.55% = 1.10% round trip:**

| arm | k=1 | k=3 | k=5 | k=10 | k=20 |
|---|---:|---:|---:|---:|---:|
| **A — run-up** (buy LDD−k close, sell LDD close, gap EXCLUDED) mean % | −1.08 | −0.52 | −0.40 | +0.40 | +1.78 |
| A t-stat | −7.45 | −2.28 | −1.23 | +0.89 | +2.40 |
| A hit-rate | 21% | 41% | 44% | 50% | 58% |
| **B — collect-and-hold** (hold through ex, +div net-13% NDFL) mean % | −0.62 | — | +0.07 | — | — |

**Run-up (arm A), with the ex-gap now correctly EXCLUDED, has no tradeable edge.** At the short
horizons where a "run-up" would actually be traded it is **net-negative** (k=1 −1.08%, t=−7.4; k=3
−0.52%; k=5 −0.40%) — the 1.10% round-trip alone dominates — and the hit-rate is ≤50% until k=20. It
only turns positive at k=10–20 (k=20 +1.78%, t=+2.4, hit 58%), but that is **~4–6 weeks of ordinary
equity beta** over a rising 2021-2025 tape (the window mostly holds non-event days), not a
dividend-specific drift. Ex-gap check (now isolated on the true ex-session): the price **drops ~by the
dividend** — ex_gap mean **−4.68%**, median −3.77%, **97.1% negative** ≈ the gross dividend yield. So
**collect-and-hold (arm B)** leaves nothing after 13% NDFL + 1.10% cost: k=1 −0.62%, k=5 +0.07% (flat).
**No mispricing to harvest** in either arm.

**Two gate lenses (both on MCFTRR-net, both `anti_hollow_ok=true`, 1363 bars, 2 regimes):**

| sleeve | tier | Δexcess-Sharpe | Δexcess-Sortino | ΔMaxDD pp | max-corr | reason |
|---|---|---:|---:|---:|---:|---|
| runup_only (k5 nav, growth) | **REJECT** | −0.147 | −0.180 | **+4.60** | 0.129 | uncorrelated (0.13) but loses money and worsens drawdown +4.6pp / Sortino −0.18 |
| equity_overlay (k5, 20% tilt on MCFTRR) | **REJECT** | −0.011 | −0.014 | +0.18 | **0.997** | redundant factor (corr ≈1.00 > 0.60) — it *is* the equity leg |

The **overlay** is a realistic unlevered **20%-tilt** on MCFTRR (`DEPLOY_FRACTION=0.20`): full-window TR
**+0.60% vs MCFTRR +7.75%** (alpha **−7.15pp**) at corr **0.997** — the gate REJECTs it as a **redundant
equity factor**, not a diversifier (the timing tilt just shaves ~7pp off buy-and-hold). The
**runup_only** timing stream is genuinely uncorrelated (corr 0.13) but is a standalone money-loser
(sleeve TR **−5.69%**) that **worsens the book's drawdown by +4.6pp** (Δexcess-Sortino −0.18), so the
gate REJECTs it on downside risk. **Both sleeves REJECT, weight 0.** (An earlier draft reported the
overlay at −88% TR / −96pp alpha — a *levered* toy construction that multiplied the whole book by each
event's single-name factor; the corrected convex-blend 20%-tilt above replaces it.)

**Bond coupon "run-up" = НКД (ACCINT) accounting artifact — CONFIRMED, not alpha.** Empirically
verified on 3 real bonds via ISS bond history (ACCINT column) + `bondization` coupon dates: ACCINT
accretes ~linearly each trading day and **resets to ~0 on the coupon date**, with the dirty price
dropping by ~the coupon — the buyer *pre-pays the seller* that accrual, so it nets to zero.
- SU26238RMFS4 (OFZ, TQOB): coupon 2023-12-06 = 35.40 rub; ACCINT reset 35.21 → 0.00 (≈ the coupon).
- SU26246RMFS7 (OFZ, TQOB): coupon 2025-09-24 = 59.84 rub; ACCINT reset 59.51 → 0.00.
- RU000A106HB4 (corp, TQCB): coupon 2025-01-02 = 29.42 rub; ACCINT reset 28.45 → ~0 (the +91-rub
  pre-coupon dirty move there is a genuine clean-price re-rating, which *reinforces* the point: any
  dirty "gain" beyond ACCINT is market repricing, and the accrual part telescopes to zero).

Net of 13% coupon tax + 1.10% round trip, buy-before-coupon is ≤0 by construction — the a-priori
accounting null holds, exactly as the equity ex-gap eats the dividend.

**Adversarial review (3/3 skeptics refute any positive edge):** *survivorship/look-ahead* — the biases
all cut the SAME way (toward REJECT); the faded-payer drop that would flip k=20 run-up to ≈flat is a
survivorship *inflation*, and CLOSE-mark + naive-T+1 mis-timing make the arm *more* negative, never
positive; the ex-gap already fully prices the dividend so correcting the anchor moves toward flat, not
alpha. *cost/tax realism* — higher (more realistic) per-side cost strengthens REJECT; the lone k=20
capture positive dies at ~150 bps/side (t=−0.14); run-up is negative even GROSS of cost for k≤10.
*regime/crash robustness* — every "positive" is one-season / one-name noise (2023-dependent, RTKM
outlier, easing n=12), while the core NEGATIVE survives excluding 2022 (k1 −2.90%/t=−11.3).

**Methodology note — a dating bug was caught and FIXED (why the run-up numbers moved).** The first
pass hard-coded LDD = record − 1 trading day / ex = record (T+1 settlement) for ALL years. But MOEX
equities settled **T+2 through 2023** (ex = record − 1) and only moved to **T+1 in 2024** (ex = record).
So for 2021-2023 events the naive "LDD" was actually the true ex-gap session, and the run-up window
ended *on* the gap — it was measuring the −4%-style ex-drop, not the pre-ex drift (this inflated the
old arm-A means to −3.46%..−0.92% and shrank the reported ex-gap to −2.09%). Caught via a raw-price
spot-check (SBER 2021: the −4.07% drop sits on 2021-05-11, which the code had labelled "LDD"). Fixed by
DETECTING the ex-date per event from the price gap (with a T+2/T+1 settlement fallback by date) so the
run-up window ends the session *before* the gap; re-ran end-to-end. The verdict is **unchanged
(REJECT)** but the numbers are now honest: the gap is isolated (ex-gap −4.68%, 97.1% negative ≈ the
dividend), and the run-up is ~flat-to-mildly-negative net of cost rather than a spurious −3.46%.

| # | Instrument / test | verdict | PR |
|---|---|---|---|
| 6a | Dividend **run-up** (buy N days pre-LDD, sell LDD, gap excluded) — 139 events, net 1.10% | **REJECT** — no tradeable drift; net-negative at short k (k1 −1.08%, t=−7.4; k5 −0.40%), hit-rate ≤50% until k20; k20 +1.78% is equity beta | this PR (diagnostic) |
| 6b | Dividend **collect-and-hold** (through ex, +div net-13% NDFL) | **REJECT** — ex-gap ≈ gross dividend (mean −4.68%, 97.1% neg); nothing left after NDFL+cost (k5 +0.07%, flat) | this PR (diagnostic) |
| 6c | `runup_only` sleeve → gate (MCFTRR-net) | **REJECT** — Δexcess-Sortino −0.18, worsens drawdown +4.6pp, weight 0 | this PR |
| 6d | `equity_overlay` (k5, 20% tilt) sleeve → gate (MCFTRR-net) | **REJECT** — redundant factor (corr 0.997 > 0.60); alpha −7.15pp vs buy-and-hold | this PR |
| 6e | Bond coupon "run-up" (НКД / ACCINT) — 3 real bonds | **REJECT (accounting null CONFIRMED)** — ACCINT resets to 0 on coupon date; dirty "run-up" is pre-paid accrued interest, a wash | this PR (control) |

**Iteration 6 finding — REJECT the event-timing edge (both equities and bonds); prior held on real
ISS data.** There is **no net-of-retail MOEX dividend edge** in either the run-up-and-exit arm (the
pre-ex drift is ~flat-to-negative and never beats the 1.10% round-trip at tradeable horizons) or the
collect-and-hold arm (the ex-gap ≈ the gross dividend, so 13% NDFL + 1.10% cost leaves nothing; the one
positive is 4–6-week equity beta, not a dividend mechanic). Both sleeves fail the pre-registered gate —
`runup_only` on downside risk (Sortino −0.18, drawdown +4.6pp), the k5 20%-tilt overlay as a
**redundant equity factor** (corr 0.997, alpha −7.15pp). The bond
coupon "run-up" is empirically the **НКД (ACCINT) accounting artifact** — accrued interest the buyer
pre-pays, resetting to zero on the coupon date — not alpha. This is diagnostic-only: it authorizes
**nothing** — no order, no config weight, real money remains a HARD STOP. **The deposit-anchor pivot
stands**, now confirmed on the event-timing axis too. Caveat: the crash-2022 and easing sub-windows
are each effectively N=1; the full-window gate verdict (2 regimes, 1363 bars) carries no
`n1_caveat`, but the per-regime positives are not robust. The one legitimate pro (no look-ahead)
does not rescue the hypothesis — a public, announced distribution is already priced by the ex-drop.

### Still open (low priority — the edge question is answered)
- **TGLD / SBGD gold ETF** vs spot GLDRUB (does the ETF wrapper change the gold verdict — likely not).
- A **leading** (not trailing) stress signal — the rate-regime/CBR or geo-risk sentiment flag — if a
  backtestable leading proxy can be found (the trailing-DD flag in iter-3 lagged fatally).
- Re-test ЗО / linkers when their indices accumulate more history + a future easing/crash cycle
  (the N=1 / short-window caveats would lift).

Next instruments are pulled from this ledger top-down; new hypotheses appended as discovered.
