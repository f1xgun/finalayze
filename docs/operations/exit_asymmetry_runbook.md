# Exit-Path Loss-Asymmetry Runbook (all enabled MOEX segments)

**Phase:** 69 — exit-path-loss-asymmetry-diagnostic-across-all-segments (v10.4 follow-up to Phase 67)
**Run name:** `phase69-diagnostic-baseline`
**Date:** 2026-06-06
**Owner:** trading-quant operator (live T-Invest token holder for the sidecar generation step)
**Single source of numbers:** `results/iterations/phase69-diagnostic-baseline/exit_asymmetry_report.txt`
(the saved, token-free `scripts/diagnose_exit_asymmetry.py` consolidated output) — keep this
runbook in lock-step with that file. Every number below is transcribed from it; none is invented.

This runbook is the consolidated, honest, severity-ranked record of the Phase-69 exit-path
loss-asymmetry diagnostic generalized from the single Phase-67 `ru_finance` analysis to **every
enabled post-68 MOEX segment** (7 equity + 2 OFZ bond). It names the implicated exit lever per
segment, assigns a per-segment verdict (CONTROL / ACCEPT-tune / DEFER / DIAGNOSE-ONLY), and gates
the one conditional inline tune that Plan 05 may run. A diagnose-only outcome — for a segment or for
the whole phase — is recorded here as a legitimate, honest result (D-02), never engineered away.

---

## 1. Method, boundaries, and frozen constants

The diagnostic is a **pure file reader** (Layer-7, stdlib + the frozen `TradeResult` schema only;
no network, no token). It reads each segment's `trades.jsonl` sidecar from the
`phase69-diagnostic-baseline` run and attributes the loss-asymmetry signature: win rate, avg-win,
avg-loss, payoff ratio (win / |loss|), exit-reason share (count + summed PnL), per-strategy PnL,
per-symbol PnL, and the named lever (emitted per segment as a `LEVER VERDICT:` line in the report,
transcribed into the "Named lever" column of the §2 table).

- **Frozen break-even boundary:** `_LOSS_DOMINANCE_FACTOR = 1.0` — preserved byte-identical from
  Phase 67 (the bare `>` break-even the 67 D-04 ACCEPT was decided on; not silently re-decided).
  Payoff `< 1.0` = small winners / big losers (the adverse asymmetry); payoff `> 1.0` = winners
  outsize losers (no adverse asymmetry).
- **Thin-sample floor:** **25 closed trades** (D-05). Any segment below the floor is tagged
  **low-confidence — informational only** and is **never tuned in this phase** (anti-curve-fit on
  noise; mirrors Phase 67's caution that a ~9-trade payoff ratio is statistically meaningless).
- **Bond branch (D-04):** OFZ segments run `bond_carry` / `bond_duration_rotation` and have **no ATR
  chandelier / exit-confidence lever**. The bond branch reports the same generic asymmetry metrics
  but names ONLY honestly-wired bond levers (`yield_stop_bps`, `max_hold` bars,
  `rebalance_interval_bars`, `max_positions`) — it never emits a chandelier verdict.

**Data caveat (sidecar generation).** The 9 input sidecars were produced by live Tinkoff backtests
(equity via `run_iteration`, OFZ via `run_bond_iteration` with the operator token + `certs/`). Two
segments (`ru_energy`, `ru_ofz_pd`) needed a transient-gRPC retry before completing — a network
flake, not a defect; their sidecars are complete and consistent. Both OFZ segments emitted
**non-null `exit_reason`** values on real data (`ru_ofz_pk` = {time: 12, force_close: 4}; `ru_ofz_pd`
= {stop: 24, time: 4, force_close: 5}), proving the Plan-03 bond exit-reason wiring works on live OFZ
data.

---

## 2. Consolidated severity-ranked table (most-asymmetric first, by payoff ascending)

Ranked by payoff ascending = most adverse loss-asymmetry first. All numbers from
`exit_asymmetry_report.txt`. (PnL figures are USD-normalized per the run harness.)

| # | Segment | Type | Trades | Win rate | Avg win | Avg loss | Payoff | Named lever | Thin (<25)? | VERDICT |
|---|---------|------|-------:|---------:|--------:|---------:|-------:|-------------|:-----------:|---------|
| 1 | ru_telecom | equity | 3 | 100.0% | +61.81 | +0.00 | 0.000 | min_exit_confidence (winners cut early) | thin | DIAGNOSE-ONLY |
| 2 | ru_transport | equity | 1 | 100.0% | +73.61 | +0.00 | 0.000 | min_exit_confidence (winners cut early) | thin | DIAGNOSE-ONLY |
| 3 | ru_construction | equity | 3 | 33.3% | +6.36 | -21.42 | 0.297 | chandelier stop multiplier | thin | DIAGNOSE-ONLY |
| 4 | ru_metals | equity | 10 | 30.0% | +17.67 | -37.23 | 0.475 | chandelier stop multiplier | thin | DIAGNOSE-ONLY |
| 5 | ru_finance | equity | 71 | 59.2% | +524.79 | -805.80 | 0.651 | chandelier stop multiplier (3.5→3.0) | no | **ACCEPT-tune (candidate, Wave-4 gated)** |
| 6 | ru_tech | equity | 8 | 37.5% | +10.07 | -11.85 | 0.849 | chandelier stop multiplier | thin | DIAGNOSE-ONLY |
| 7 | ru_energy | equity | 166 | 60.8% | +525.10 | -608.03 | 0.864 | chandelier stop multiplier | no | **CONTROL** |
| 8 | ru_ofz_pd | bond | 33 | 36.4% | +3896.39 | -2191.22 | 1.778 | yield_stop_bps (bond) | no | DEFER |
| 9 | ru_ofz_pk | bond | 16 | 93.8% | +9352.82 | -19.69 | 475.025 | max_hold bars (bond) | thin | DIAGNOSE-ONLY |

Note the diagnostic discriminates correctly: the two control-class workhorses (`ru_energy`,
`ru_finance`) and the one adequate-sample bond (`ru_ofz_pd`) are the only three segments above the
25-trade floor; everything else is informational only.

---

## 3. Per-segment verdicts and named levers

### ru_energy — CONTROL (166 trades, payoff 0.864, WR 60.8%)

ru_energy is **diagnosed like every other segment but held byte-identical as the no-regression
control** (D-07). It DOES show a mild adverse asymmetry (avg-loss -608.03 > avg-win +525.10, payoff
0.864 < 1.0 — losers run slightly farther than winners), and its named lever is the chandelier stop
multiplier. **It is NEVER tuned in this phase.** It is the workhorse anchor (PF ~1.34 in the Phase-68
A/B) whose profile must survive any change made elsewhere; tuning it would forfeit the control. Its
loss is dominated by stop exits (15 | -14390.38) but profit_target carries it (139 | +37462.36) and
`dual_momentum` (+16395.17) is the PnL engine — exactly the profile a control should have.

### ru_finance — ACCEPT-tune (candidate, pending Wave-4 A/B gate)

ru_finance is the **only non-control segment that clears the 25-trade floor AND shows the textbook
adverse asymmetry**: 71 trades, WR 59.2% (high win rate) with payoff 0.651 (< 1.0), avg-win +524.79
vs avg-loss **-805.80**, and a dominant stop-exit loss tail — **30 stop exits summing -19,717.80**,
which more than wipes out the +16,517.60 from 31 profit_target exits. Per-symbol bleed concentrates
in SBER (-5392.22), SVCB (-1679.09), RENI (-1460.07).

- **Named lever:** the **chandelier `stop_atr_multiplier` for ru_finance** (the proven Phase-67
  lever), currently **3.5**.
- **Bounded proposed move:** **3.5 → 3.0** (a further tighten).
- **Economic rationale:** the asymmetry is a fat stop-exit loss tail (-19,717.80 across 30 stops). A
  tighter chandelier cuts the depth of the losing tail (lowers |avg-loss|), pulling payoff back
  toward break-even. The success test is whether it **raises payoff toward break-even on OOS data**,
  NOT whether in-sample PF ticks over 1.0 (the asymmetry-narrowing standard, not the manufactured-PF
  standard).
- **IMPORTANT double-tune caveat.** ru_finance was **already tuned in Phase 67** on this exact lever
  (chandelier 4.0 → 3.5; PF 0.895 → 0.943, which did **NOT** cross 1.0). The candidate 3.5 → 3.0 move
  re-pulls the same lever one phase later, so it carries **double-tune / curve-fit risk**. That is
  precisely why it is a *candidate*, gated by the Wave-4 D-11 A/B no-regression gate, and **may
  legitimately be REJECTED** (PF ≥ −5%, MaxDD ≤ +15%, WF-Sharpe ≥ −10% vs the frozen pre-69 baseline,
  with no material ru_energy regression). It is not accepted until that gate passes.

### ru_ofz_pd — DEFER (33 trades, payoff 1.778, WR 36.4%)

ru_ofz_pd **clears the 25-trade floor but shows NO adverse loss-asymmetry**: payoff **1.778 > 1.0**
(favorable). A few large time-exit winners (4 | +35363.14) plus force_close (5 | +6563.65) more than
offset many small yield-stop losses (24 stops | -41185.68 gross, but small per-trade) — a
carry/duration profile, not a "small winners / big losers" exit pathology (D-04). The bond lever is
named for visibility — **`yield_stop_bps`** (yield-stop exits dominate the loss count) — but it is
**NOT actioned**: there is no adverse asymmetry to fix, and a bond is **never given a chandelier
verdict** (D-04). Deferred, not tuned.

### ru_ofz_pk — DIAGNOSE-ONLY (16 trades, payoff 475.0, WR 93.8%)

Thin (< 25) and a bond. Profile is pure carry (`bond_carry` +140,272.67), all exits are time (12 |
+99187.73) or force_close (4 | +41084.94) — no loss tail at all (1 loss of -19.69). Bond lever named
for visibility (**`max_hold` bars** — time exits dominate) but informational only; never tuned (D-04,
D-05).

### ru_tech, ru_metals, ru_construction, ru_telecom, ru_transport — DIAGNOSE-ONLY (all thin)

All five equity sectors are **below the 25-trade floor** (8, 10, 3, 3, 1 trades respectively) and are
recorded **low-confidence — informational only**, never tuned in this phase (D-05; anti-curve-fit on
noise). For visibility their named levers are transcribed: `ru_tech` / `ru_metals` /
`ru_construction` → chandelier stop multiplier (avg-loss dominates avg-win); `ru_telecom` /
`ru_transport` → min_exit_confidence (100% WR with tiny avg-win = winners cut early — but on 3 and 1
trade, statistically meaningless). These are diagnosed for the record, not actioned.

---

## 4. Honest outcome

**One segment — `ru_finance` — is the sole ACCEPT-tune candidate, and it is gated by the Wave-4 D-11
A/B no-regression check.** Every other segment is either the byte-identical CONTROL (`ru_energy`), a
favorable/no-asymmetry DEFER (`ru_ofz_pd`), or a thin DIAGNOSE-ONLY (the remaining six). No tune is
manufactured to produce a number (D-02).

If the Wave-4 gate **rejects** the ru_finance 3.5 → 3.0 move (a real possibility given the double-tune
/ curve-fit risk noted in §3), then **the phase is effectively diagnose-only** — which is a
legitimate, honest recorded result, exactly the Phase-67 / Phase-70 discipline (honest verdict over
forced pass). A diagnose-only phase outcome is not a failure; it is the diagnostic doing its job.

---

## 5. Next step

- **If the candidate is pursued (default):** **Plan 05** runs the gated `ru_finance` chandelier
  **3.5 → 3.0** A/B through the `backtest-iteration` skill against the frozen pre-69 baseline. It
  holds the D-11 tolerances (PF ≥ −5%, MaxDD ≤ +15%, WF-Sharpe ≥ −10%) and must not materially
  regress `ru_energy` (the byte-identical control). If it **passes** → ACCEPT (record the tightened
  multiplier as live). If it **fails D-11** → record **REJECT**; the segment stays at 3.5 and the
  phase result is diagnose-only.
- **No other segment is a tune target.** `ru_ofz_pd` (DEFER), `ru_ofz_pk` and the five thin equity
  sectors (DIAGNOSE-ONLY), and `ru_energy` (CONTROL) are all explicitly out-of-scope for tuning in
  this phase. Borderline / multi-lever segments are diagnosed here and deferred to dedicated
  follow-up phases (the one-segment-at-a-time Phase-67 model).
- No segment removed in Phase-68 is referenced or treated as a tune target anywhere in this runbook
  — only the 9 enabled post-68 segments are diagnosed, and only `ru_finance` is a (gated) candidate.
