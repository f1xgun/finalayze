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

### META-FINDING (after 3 iterations / 10 instrument-tests + a timing overlay)
The diversification space for a sanctioned-RU deposit-anchored book is **effectively exhausted at
the static AND the conditional level**: every RUB fixed-income leg (OFZ duration, IG/HY credit,
money-market, linkers) is **redundant with the rate factor** (corr ≥ 0.4 → REJECT); every
genuinely-uncorrelated asset (gold, CNY, ЗО) is a **zero-carry drag** that can't beat the ~16-21%
near-vol-free deposit on risk-adjusted return (REJECT), or an **unprovable-tail hedge** (ЗО →
PROBATION); combining hedges (gold+CNY) doesn't rescue it; and **timing** the hedge (conditional)
mis-fires and is dominated by the plain core. The deposit anchor genuinely dominates — now proven
across the static, combination, AND conditional axes, enforced by a reusable pre-registered gate.

**Implication for the loop:** further single-instrument / hedge-timing iterations have low expected
value — the honest answer is converging. The remaining constructive moves are forward-looking
INTEGRATIONS of what little is sound, not new REJECT candidates.

| # | Constructive integration | status | PR |
|---|---|---|---|
| 4 | **ЗО PROBATION → geo-risk overlay rotation** — on elevated/high geo-risk the overlay now recommends rotating a small ЗО FX-linked toe-hold (≤3% PROBATION cap; 1.5% elevated / 3% high) INSTEAD of trimming only into more ruble deposit/OFZ. Forward-only advisory; config weight, never an order. | DONE | this PR |

**Iteration 4 (constructive):** the one hedge that survived the gate (ЗО, PROBATION — FX-linked +
uncorrelated) is now operationalized exactly where it belongs. The geo-risk overlay (#300) already
trims equity on sanctions stress but only toward ruble assets (which don't hedge devaluation); it
now also surfaces `recommended_fx_hedge_pct` — a small ЗО rotation, the structurally-sound (but
tail-unproven, hence toe-hold-capped) destination. `geopolitical_risk.py` brain + the alert
surface it; real money stays a hard stop.

### Still open (low priority — the edge question is answered)
- **TGLD / SBGD gold ETF** vs spot GLDRUB (does the ETF wrapper change the gold verdict — likely not).
- A **leading** (not trailing) stress signal — the rate-regime/CBR or geo-risk sentiment flag — if a
  backtestable leading proxy can be found (the trailing-DD flag in iter-3 lagged fatally).
- Re-test ЗО / linkers when their indices accumulate more history + a future easing/crash cycle
  (the N=1 / short-window caveats would lift).

Next instruments are pulled from this ledger top-down; new hypotheses appended as discovered.
