# MOEX Liquidity Universe Runbook

**Phase:** 66 — MOEX liquidity filter + universe/ML expansion (v10.4 sub-area 2)
**Owner:** trading-infra operator (live T-Invest token holder)
**Single source of numbers:** `src/finalayze/markets/liquidity.py` constants +
`src/finalayze/markets/data/moex_liquidity_universe.json` (`params` block) — keep this
runbook in lock-step with those.

This runbook is the operational record for the committed MOEX liquid universe: how it is
regenerated quarterly (D-06), the chosen Top-N / RUB floor / D-11 tolerances, and the
recorded `backtest-iteration` acceptance verdict (filled in after the Task-4 operator gate).

---

## 1. What the universe is

The committed snapshot `src/finalayze/markets/data/moex_liquidity_universe.json` is a
`sector -> ranked-symbols` map (ranked by trailing-60d median RUB turnover, descending),
generated one-shot from a live read-only T-Invest enumeration of the 268 MOEX shares. The
runtime selector (`finalayze.markets.liquidity.select_segment_symbols` /
`eligible_universe_as_of`) reads it **offline, fail-closed** — no live gRPC at selection
time (D-03/D-04). A missing/corrupt/unknown-sector file raises `ConfigurationError`; only
true file ABSENCE bootstraps to the prior hardcoded lists (pre-66 compat shim).

The committed file is the **trust boundary** (T-66-14): every sector key is validated
against `config.segments.SECTOR_TO_SEGMENT`, every symbol against the registry, and the
universal safety filter (toxic/sanctioned names + preferred-share duplicates) is applied at
both generation and selection so the file and the live selection agree.

---

## 2. Chosen parameters (Task-2 distribution → operator decision)

Derived from the Task-2 live `--dry-run` turnover distribution (96 of 268 shares scored
≥ 60 clean bars) and chosen by the operator. **Single-sourced** as named constants in
`src/finalayze/markets/liquidity.py` (consumed by both the selector default `top_n` and the
generator argparse defaults) — no magic numbers, no divergent copies.

| Parameter | Value | Constant | Rationale |
|-----------|-------|----------|-----------|
| Top-N per sector | **10** | `liquidity._TOP_N_PER_SECTOR` | Top-10 by turnover bounds the universe and balances across sectors; most sectors have < 10 liquid names so N=10 keeps the full liquid tail per sector while capping the deepest (oil_gas / metals / tech / banks). |
| RUB turnover floor | **1,000,000 RUB/day** (median) | `liquidity._MIN_TURNOVER_FLOOR_RUB` | Sanity floor below which a name is dropped regardless of rank — excludes structurally-illiquid names a sparse sector would otherwise admit purely on rank. |
| D-11 PF tolerance | **PF ≥ −5%** vs baseline | `liquidity._D11_PF_REGRESSION_TOLERANCE_PCT` | Portfolio profit-factor may not regress more than 5% relative to the current curated baseline. |
| D-11 MaxDD tolerance | **MaxDD ≤ +15%** (relative) vs baseline | `liquidity._D11_MAXDD_REGRESSION_TOLERANCE_PCT` | Max drawdown may not worsen (grow) by more than 15% relative to baseline. |
| D-11 WF-Sharpe tolerance | **WF-Sharpe ≥ −10%** vs baseline | `liquidity._D11_WF_SHARPE_REGRESSION_TOLERANCE_PCT` | Walk-forward Sharpe may not regress more than 10% relative to baseline. |

**Committed snapshot result** (`params`: `top_n=10`, `min_turnover_rub="1000000"`,
`window=60`, `share_count=268`, `scored_count=96`): **50 names across 10 sectors**.
`utilities` was dropped — its only liquid name `IRAO` is sanctioned/toxic, so the sector
emptied after the universal safety filter (a legitimate "no tradeable non-toxic name", warned
and dropped, NOT a fail-closed refusal).

D-11 acceptance is judged **relative** (each metric vs the curated baseline within the
tolerance above) **AND** the per-segment walk-forward gate must still pass (D-10). Tolerances
are **not** to be relaxed post-hoc — a regression is recorded as a finding (T-66-15 / LIQ-10).

---

## 3. Quarterly regeneration recipe (D-06)

Regenerate the committed snapshot each quarter (index-reconstitution cadence). This is a
**live read-only T-Invest one-shot** — run it from the worktree per
`project_worktree_moex_retrain_recipe` (memory):

### Prerequisites (worktree)

1. Symlink `.env` and `certs/` from the main repo into this worktree if absent
   (`certs/grpc_roots.pem` = Russian Trusted Root CA for the tbank gRPC endpoint).
2. Export the token into the shell (do **NOT** `source .env` — it breaks pydantic settings;
   the generator reads `os.environ` directly):
   ```bash
   export FINALAYZE_TINKOFF_TOKEN=...        # T-Bank Invest API token (read-only)
   export GRPC_DNS_RESOLVER=native           # gRPC C-ares DNS resolver workaround
   ```
   The token is read from the environment ONLY and is NEVER logged or serialized (T-66-12).

### Commands

```bash
# 1. Review the per-sector turnover distribution (no write):
uv run python scripts/generate_liquidity_universe.py --dry-run

# 2. Confirm the chosen N / RUB floor still fit the distribution. If they change, edit the
#    single-source constants in src/finalayze/markets/liquidity.py FIRST (the generator
#    argparse defaults pick them up automatically), then:

# 3. Generate + commit the refreshed snapshot:
uv run python scripts/generate_liquidity_universe.py
```

The generator **refuses to write** (`SystemExit`) if any curated sector enumerates to 0
ranked names — a wholesale gRPC/auth/cert/DNS failure must surface, never commit a
partial/stale universe (T-66-13 / D-04). A sector emptied ONLY by the safety filter (every
liquid name sanctioned, e.g. `utilities`/`IRAO`) is warned + dropped, not refused.

After regeneration: commit the refreshed JSON, re-run the validation (`uv run pytest
tests/unit/test_liquidity.py`), and re-run the ML re-gate + `backtest-iteration` acceptance
(section 4) — update the verdict below.

---

## 4. ML re-gate + backtest-iteration acceptance (Task-4 operator gate)

> **Status: PENDING** — to be filled in by the operator after the Task-4 checkpoint.

### 4a. ML re-gate (D-10, BONUS — not a gate on trading)

For each affected `ru_*` segment, retrain on the EXPANDED (old + new) symbol set via the
UNCHANGED pipeline (NO `--force-save`):

```bash
uv run python scripts/train_models.py --segment ru_energy \
    --walk-forward --excess-returns --sequential-bootstrap
```

Record per-segment KEEP/DISC from `models/<segment>/wf_gate_results.json` (trust
`overall_passed` / `bh_passed`, NOT headline accuracy — Pitfall 5).

| Segment | KEEP / DISC | Source |
|---------|-------------|--------|
| ru_energy | _pending_ | |
| ru_finance | _pending_ | |
| ru_metals | _pending_ | |
| ru_tech | _pending_ | |
| ... | _pending_ | |

### 4b. backtest-iteration (LIQ-12, MANDATORY phase gate — CLAUDE.md #4)

Run the expanded universe through the migrated `run_portfolio` path and log an iteration
under `results/iterations/`; invoke the `backtest-iteration` skill to record PF / MaxDD /
WF-Sharpe / trade_count.

| Metric | Baseline (curated) | Expanded universe | Δ | Within tolerance? |
|--------|--------------------|--------------------|---|-------------------|
| PF | _pending_ | _pending_ | | ≥ −5% |
| MaxDD | _pending_ | _pending_ | | ≤ +15% |
| WF-Sharpe | _pending_ | _pending_ | | ≥ −10% |
| trade_count | _pending_ | _pending_ | | (informational) |

- **Logged iteration id:** _pending_
- **Verdict (PASS / WARN / REJECT vs the section-2 tolerances):** _pending_
- If thinner names look optimistic, sanity-check with a liquidity-scaled slippage run
  (`OFF_THE_RUN_SPREAD_UPLIFT_BPS` precedent) and note if it materially moves the verdict
  (Pitfall 6).
- A REJECT/regression is recorded here as a **finding** — tolerances are NOT silently
  relaxed (T-66-15 / LIQ-10). The follow-up is a liquidity-scaled slippage re-run or a
  smaller N.

---

## 5. D-12: the expanded universe trades RULE-BASED regardless of ML KEEP/DISC

**IMPORTANT (LIQ-11 / D-12):** the liquid expanded universe ships on the **rule-based** path
(momentum / mean-reversion / etc.) **regardless** of the ML re-gate KEEP/DISC outcome,
subject only to the D-11 portfolio no-regression bar. ML is an optional enhancement layer,
NOT a gate on trading the expanded universe — all four `ru_*` ML segments are legitimately
DISC today (no force-save debt), and rule-based is the live path. A passing ML re-gate is a
bonus; the headline deliverable is a liquid, point-in-time, rule-based-tradeable expanded
universe that does not regress the portfolio.
