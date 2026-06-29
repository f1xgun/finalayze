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
| 1a | RGBITR fixed-coupon OFZ | medium / carry | ISS index RGBITR | **REJECT** — redundant (corr 0.61 to rate/bond factor); confirms Phase 76 | this PR |
| 1b | RUCBITR corporate IG | medium / carry | ISS index RUCBITR | **REJECT** — tested & worsens risk-adj (ΔSortino −0.13); data ends 2023-05 | this PR |
| 1c | RUCBHYTR corporate HY (ВДО) | high / carry | ISS index RUCBHYTR | **REJECT** — HY credit too equity-correlated (0.48) to diversify | this PR |
| 1d | LQDT money-market | low / cash | ISS shares TQTF | **REJECT** — no material benefit (cash-like, ~flat marginal effect) | this PR |
| 1e | CNYRUB FX | high / diversifier | ISS CETS CNYRUB_TOM | **REJECT** — uncorrelated but big zero-carry drag (ΔSortino −0.33), tail tested & failed | this PR |

**Iteration 1 finding (instrument battery):** all 5 candidates REJECT, each for a distinct honest
reason — the gate is discriminating, not a rubber stamp. Nothing clears the deposit+equity core;
the "no easy edge" pattern holds across every risk tier (cash, OFZ duration, IG/HY credit, FX).
ЗО (PROBATION) remains the only non-REJECT — a structurally-sound but unproven FX-tail hedge.

### Candidate hypotheses still open (appended as discovered)
- **OFZ-IN inflation linkers** (low-medium / inflation-hedge) — hedge the inflation that erodes the
  deposit; check linker TR index availability on ISS.
- **TGLD / SBGD gold ETF** vs spot GLDRUB (does the ETF wrapper change the gold verdict — likely not).
- **Equal-weight / risk-parity blend of the deposit + ЗО PROBATION leg** — does combining the one
  sound hedge at its toe-hold with the anchor improve the frozen SAA at all?
- **A multi-hedge PROBATION basket** (ЗО + gold small) under the aggregate 5% probation cap.

Next instruments are pulled from this ledger top-down; new hypotheses appended as discovered.
