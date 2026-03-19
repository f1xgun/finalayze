# Phase 2: MOEX Equity Validation - Research

**Researched:** 2026-03-14
**Domain:** MOEX equity strategy calibration & walk-forward validation
**Confidence:** HIGH

## Summary

Phase 2 is a parameter-tuning and validation phase -- no engine code changes, only YAML preset edits and script runs. The existing infrastructure (isolation testing, Optuna tuning, walk-forward validation, backtest-iteration skill) is fully wired and ready. The main challenge is that current MOEX presets are essentially US copies with minor tweaks, producing negative Sharpe ratios on most MOEX symbols (ru_blue_chips aggregate Sharpe is approximately -0.4, ru_energy approximately -0.5 based on the `moex-new-datasources` iteration).

Three critical issues must be addressed: (1) momentum and dual_momentum are currently disabled on all MOEX segments -- they need to be enabled and calibrated; (2) the cointegration test script (`test_pairs_cointegration.py`) uses yfinance with `.ME` suffix which is unreliable for MOEX data -- it needs to use TinkoffFetcher instead; (3) the `run_iteration.py` UNIVERSE does not include `ru_finance`, so that segment cannot be backtested without adding it. The 2022 MOEX trading suspension (Feb 28 - Mar 28) is not explicitly handled in the calendar -- the TinkoffFetcher will simply return no candles for that period, which the engine handles gracefully, but walk-forward window generation should be aware of this data gap.

**Primary recommendation:** Follow the user-decided sequence: isolation tests first per strategy per segment, then enable momentum/dual_momentum, run cointegration tests (with TinkoffFetcher), Optuna-tune combined presets, validate with walk-forward backtests. Maximum 5 tuning iterations. Target: OOS Sharpe > 0.1 and PF > 1.05 on 2+ MOEX segments.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Enable ALL strategies including momentum and dual_momentum on MOEX (currently disabled)
- Equal emphasis weighting (~0.15-0.20 per strategy) as starting point
- Differentiate strategy mix by sector:
  - ru_energy: heavier momentum weight (oil-driven trends)
  - ru_finance: heavier MR weight (banking stocks range-bound between CBR rate decisions)
  - ru_blue_chips: balanced across strategies
- Keep dividend_gap enabled (0.15-0.20 weight) -- Russian dividends are 6-12%, natural MOEX edge
- Auto-disable any strategy with negative Sharpe on ALL 3 MOEX segments in isolation testing; redistribute weight
- Use Optuna optimization with existing overfitting guardrails (DSR haircut, holdout validation, perturbation check)
- Sequence: isolation tests first, then combined Optuna tuning
- Maximum 5 tuning iterations before accepting results or pivoting approach
- Minimum bar: out-of-sample Sharpe > 0.1 AND Profit Factor > 1.05 on at least 2 MOEX segments
- Maximum drawdown hard cap: 20% -- reject any parameter set exceeding this
- Walk-forward out-of-sample metrics are the gate, not in-sample
- Run cointegration tests (Engle-Granger or Johansen, p < 0.05) on all included pairs
- Test 3-5 additional candidate pairs beyond current 3
- Include only pairs that pass cointegration testing

### Claude's Discretion
- ATR stop multiplier calibration for MOEX (currently 1.2x uplift vs US)
- ADX regime routing thresholds (currently 30/20 in YAML, US code default is 35/15)
- Bollinger Band std_dev (include in Optuna search space, no floor constraint)
- max_hold_bars adjustment for MOEX session length
- 2022 trading suspension period handling
- Walk-forward window size
- Per-sector exact weight splits for ru_energy (momentum tilt) and ru_finance (MR tilt)
- Whether to keep or drop pairs that fail cointegration with borderline p-values
- z_entry/z_exit thresholds for pairs strategy

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EQF-02 | MOEX backtest produces positive PnL with walk-forward validation | Isolation testing -> Optuna tuning -> walk-forward validation pipeline. Scripts exist: `run_strategy_isolation.py`, `tune_strategy_params.py`, `run_iteration.py`. Current baseline is negative; calibration is the core work. |
| EQF-03 | MOEX-specific strategy parameters tuned (ru_* YAML presets calibrated) | Three ru_* equity presets need genuine MOEX calibration. Currently near-copies of US presets with momentum/dual_momentum disabled. Must enable all strategies, adjust ADX routing, BB std_dev, confidence thresholds, vol targets. |
</phase_requirements>

## Standard Stack

### Core
| Tool | Location | Purpose | Why Standard |
|------|----------|---------|--------------|
| `run_strategy_isolation.py` | `scripts/` | Test each strategy solo per segment | Measures standalone viability before combining |
| `tune_strategy_params.py` | `scripts/` | Optuna Bayesian optimization with guardrails | DSR haircut, holdout validation, perturbation check built in |
| `run_iteration.py` | `scripts/` | Combined backtest with walk-forward and full metrics | Primary validation tool |
| `test_pairs_cointegration.py` | `scripts/` | Engle-Granger cointegration + half-life + Hurst | Pair selection gate |
| `backtest-iteration` skill | `.claude/skills/` | Run & compare metrics after changes | Mandatory gate after strategy changes |
| `preset-tuner` skill | `.claude/skills/` | Structured parameter tuning workflow | Anti-overfitting discipline |
| `strategy-diagnose` skill | `.claude/skills/` | Debug underperforming strategies | Root cause analysis |

### YAML Presets (edit targets)
| File | Segment | Current State |
|------|---------|---------------|
| `ru_blue_chips.yaml` | ru_blue_chips | momentum/dual_momentum DISABLED, MR weight 0.30 |
| `ru_energy.yaml` | ru_energy | momentum/dual_momentum DISABLED, MR weight 0.30 |
| `ru_finance.yaml` | ru_finance | momentum/dual_momentum DISABLED, has cbr_calendar (0.05) |

## Architecture Patterns

### Workflow Sequence (from user decisions)

```
Phase 2 Execution Order:
├── Wave 0: Fix infrastructure gaps
│   ├── Add ru_finance to run_iteration.py UNIVERSE
│   ├── Fix test_pairs_cointegration.py to use TinkoffFetcher
│   └── Handle 2022 suspension in backtest date range
├── Wave 1: Isolation testing (all strategies x 3 segments)
│   ├── Enable momentum + dual_momentum in all ru_* presets
│   ├── Run isolation per strategy per segment
│   └── Auto-disable strategies with negative Sharpe on ALL 3 segments
├── Wave 2: Cointegration testing
│   ├── Test existing pairs + 3-5 candidates
│   └── Keep only cointegrated pairs (p < 0.05, HL < 30d, H < 0.5)
├── Wave 3: Combined preset calibration
│   ├── Set initial weights per user decisions (sector-specific)
│   ├── Run Optuna tuning per segment
│   └── Apply overfitting guardrails
├── Wave 4: Walk-forward validation
│   ├── Full walk-forward backtest per segment
│   └── Gate: OOS Sharpe > 0.1, PF > 1.05, DD < 20%
└── Wave 5: Final iteration comparison
    └── backtest-iteration skill comparison vs baseline
```

### Preset YAML Edit Pattern
**What:** Edit strategy weights, enable/disable flags, and strategy-specific parameters in YAML files.
**When to use:** All preset changes. Never modify combiner.py or engine code.
**Example:**
```yaml
# src/finalayze/strategies/presets/ru_blue_chips.yaml
strategies:
  momentum:
    enabled: true          # Was: false
    weight: 0.15           # Equal emphasis starting point
    params:
      # MOEX-calibrated params go here
  dual_momentum:
    enabled: true          # Was: false
    weight: 0.15
```

### Isolation Test Pattern
**What:** Run each strategy solo with weight=1.0 to measure standalone viability.
**When to use:** Before combining strategies. Identifies which strategies work on MOEX.
```bash
uv run python scripts/run_strategy_isolation.py \
  --segment ru_blue_chips --all --cash 1000000
uv run python scripts/run_strategy_isolation.py \
  --segment ru_energy --all --cash 1000000
uv run python scripts/run_strategy_isolation.py \
  --segment ru_finance --all --cash 1000000
```
Note: `--cash 1000000` for RUB denomination (1M RUB, matching Phase 1 fix).

### Anti-Patterns to Avoid
- **Tuning in-sample only:** Walk-forward OOS metrics are the gate, not in-sample PF.
- **Too many Optuna trials without guardrails:** Always use DSR haircut, holdout validation, perturbation check.
- **Copying US params verbatim:** MOEX has different volatility, liquidity, session length, dividend patterns.
- **Ignoring trade count:** PF 2.0 with 10 trades is noise. Need statistical significance.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Strategy isolation testing | Manual per-strategy YAML edits | `run_strategy_isolation.py --all` | Script creates temp presets, runs all strategies, produces ranking table |
| Bayesian parameter optimization | Manual grid search | `tune_strategy_params.py --segment X` | Optuna TPE sampler with built-in overfitting guardrails |
| Walk-forward validation | Single-period backtest | Walk-forward engine (12mo train + 6mo test) | Already wired in `run_iteration.py` |
| Cointegration testing | Custom statistics code | `test_pairs_cointegration.py` | Engle-Granger + half-life + Hurst + stability analysis |
| ATR stop calibration | Manual multiplier guessing | `resolve_stop_atr_multiplier()` | Already handles MOEX 1.2x uplift per segment |
| MOEX costs | Custom fee calculations | `MOEX_COSTS` preset | Already wired: 0.04% commission + 10bps spread + 7bps slippage |

## Common Pitfalls

### Pitfall 1: run_iteration.py Missing ru_finance Universe
**What goes wrong:** Running `--segments ru_finance` produces zero trades because the UNIVERSE dict in `run_iteration.py` does not contain a `ru_finance` key.
**Why it happens:** Only `ru_blue_chips` and `ru_energy` were added to the iteration script.
**How to avoid:** Add `ru_finance` universe to `run_iteration.py` UNIVERSE dict before any ru_finance testing.
**Warning signs:** Zero trades, "no symbols" in output.

### Pitfall 2: Cointegration Script Uses yfinance for MOEX
**What goes wrong:** `test_pairs_cointegration.py` fetches MOEX prices via `yf.download()` with `.ME` suffix. yfinance is unreliable for MOEX tickers -- may return empty data or wrong prices.
**Why it happens:** Script was written before TinkoffFetcher was wired.
**How to avoid:** Refactor the script to use TinkoffFetcher for MOEX pair data. Requires `FINALAYZE_TINKOFF_TOKEN`.
**Warning signs:** "No price data returned from Yahoo Finance" error, or empty DataFrames.

### Pitfall 3: 2022 MOEX Trading Suspension
**What goes wrong:** MOEX was closed from Feb 28 to Mar 28, 2022. Walk-forward windows spanning this period will have a 1-month data gap.
**Why it happens:** The moex_calendar.py only lists individual holidays, not the prolonged suspension.
**How to avoid:** Use backtest start date of 2022-04-01 or later for MOEX segments. Alternatively, handle the gap in data fetching (TinkoffFetcher returns no candles for closed days, which the engine handles). The walk-forward window generator should be robust to this.
**Warning signs:** Unusually few candles in early 2022 windows, distorted Sharpe from short test periods.

### Pitfall 4: Isolation Script Default Cash is 100K (USD)
**What goes wrong:** `run_strategy_isolation.py` defaults to `--cash 100000`, which is 100K in the backtest currency. For MOEX this means 100K RUB (~$1,100), which is unrealistically small.
**Why it happens:** Default was set for US markets.
**How to avoid:** Use `--cash 1000000` (1M RUB) for MOEX segments, matching the Phase 1 fix.
**Warning signs:** Very small position sizes, many pre-trade check failures, low trade counts.

### Pitfall 5: Tune Script Uses YFinanceFetcher for MOEX
**What goes wrong:** `tune_strategy_params.py` line 302 hardcodes `CachingFetcher(YFinanceFetcher(...))` -- for MOEX segments this will fail to fetch data.
**Why it happens:** Tuning script was built for US segments.
**How to avoid:** Must patch `tune_strategy_params.py` to use TinkoffFetcher for `ru_*` segments (same pattern as `run_strategy_isolation.py`).
**Warning signs:** All Optuna trials return `{wf_sharpe: -1.0, trades: 0, max_dd: 1.0}`.

### Pitfall 6: ADX Routing Threshold Mismatch
**What goes wrong:** The YAML presets use trend_threshold=30 and mr_threshold=20 for MOEX, but the combiner code defaults are 35/15. If YAML loading fails or routing config is missing, the code defaults take over.
**Why it happens:** Divergent defaults between code and YAML.
**How to avoid:** Always verify the YAML `regime_routing` section is present and loaded correctly. The combiner reads these from the YAML config dict.
**Warning signs:** Strategies firing at unexpected ADX levels.

### Pitfall 7: Overfitting to Small MOEX Universes
**What goes wrong:** ru_blue_chips has only 4 symbols in the segment config (SBER, GAZP, LKOH, GMKN), though the iteration script universe has 10. Small universes make Optuna optimization prone to overfitting.
**Why it happens:** MOEX has fewer liquid stocks than US markets.
**How to avoid:** Use the broader UNIVERSE from `run_iteration.py` (10 symbols for ru_blue_chips, 8 for ru_energy). Apply Optuna guardrails strictly.
**Warning signs:** Holdout degradation ratio below 0.50, fragile params in perturbation check.

## Code Examples

### Adding ru_finance to run_iteration.py UNIVERSE
```python
# scripts/run_iteration.py - add after ru_energy entry
"ru_finance": [
    "SBER",
    "SBERP",
    "VTBR",
    "TCSG",
    "CBOM",
    "BSPB",
    "MOEX",
],
```

### Fixing tune_strategy_params.py for MOEX
```python
# In run_backtest_for_trial(), replace:
#   fetcher = CachingFetcher(YFinanceFetcher(market_id=market_id))
# With:
if segment_id.startswith("ru_"):
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    if token:
        from finalayze.markets.instruments import build_default_registry
        registry = build_default_registry()
        fetcher = CachingFetcher(TinkoffFetcher(token=token, registry=registry, sandbox=False))
    else:
        return {"wf_sharpe": -1.0, "trades": 0, "max_dd": 1.0}
else:
    fetcher = CachingFetcher(YFinanceFetcher(market_id=market_id))
```

### MOEX-Specific Preset Calibration Starting Points
```yaml
# Key differences from US presets:
regime_routing:
  trend_threshold: 25    # Lower than US (30) -- MOEX trends at lower ADX
  mr_threshold: 15       # Narrower ambiguous zone

# BB std_dev wider for MOEX volatility
mean_reversion:
  params:
    bb_std_dev: 2.5      # US uses 2.0; MOEX needs wider bands

# OU window shorter for MOEX
ou_mean_reversion:
  params:
    ou_window: 45        # US uses 126; MOEX mean-reverts faster

# Dual momentum shorter lookbacks for MOEX
dual_momentum:
  params:
    lookback_6m: 100     # MOEX regime changes faster than US
    min_confidence: 0.45  # Lower bar to allow more signals
```

### Running Walk-Forward Validation
```bash
uv run python scripts/run_iteration.py \
  --name "moex-phase2-calibrated" \
  --description "MOEX presets after isolation + Optuna tuning" \
  --segments ru_blue_chips,ru_energy,ru_finance
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| US params copied to MOEX | Must calibrate per-segment | Phase 2 (now) | Current MOEX Sharpe is negative; needs genuine tuning |
| yfinance for MOEX data | TinkoffFetcher via gRPC | Phase 1 / earlier | yfinance unreliable for .ME tickers |
| Fixed ADX thresholds (35/15) | YAML-configurable per segment | Week 3 | MOEX may need different thresholds |
| Single ATR stop for all markets | MOEX 1.2x uplift | Week 3 | Already wired, may need further calibration |
| No MOEX cost model | MOEX_COSTS preset (0.04% + 10bps + 7bps) | Phase 1 | Already wired in engine |

## Key Numbers for Planning

### Current MOEX Baseline (moex-new-datasources iteration, 2023-2025)
| Segment | Symbols | Avg Sharpe | Avg PF | Total Trades | Issue |
|---------|---------|------------|--------|--------------|-------|
| ru_blue_chips | 10 | -0.43 | 0.60 | 170 | Most symbols negative; LKOH (+0.23) and POLY (+0.43) positive |
| ru_energy | 8 | -0.66 | 0.52 | 166 | Only LKOH positive; high negative Sharpe across board |
| ru_finance | -- | N/A | N/A | N/A | Not in UNIVERSE; needs adding |

### Target Performance (from user decisions)
| Metric | Minimum | Notes |
|--------|---------|-------|
| OOS Sharpe | > 0.1 | Walk-forward out-of-sample |
| Profit Factor | > 1.05 | On at least 2 MOEX segments |
| Max Drawdown | < 20% | Hard reject above this |
| Segments passing | >= 2 | ru_blue_chips + ru_energy minimum |

### Currently Disabled Strategies on MOEX
| Strategy | ru_blue_chips | ru_energy | ru_finance |
|----------|---------------|-----------|------------|
| momentum | DISABLED | DISABLED | DISABLED |
| dual_momentum | DISABLED | DISABLED | DISABLED |
| event_driven | DISABLED | DISABLED | DISABLED |
| ml_ensemble | DISABLED | DISABLED | DISABLED |
| pead | -- | -- | DISABLED |

Only mean_reversion, rsi2_connors, ou_mean_reversion, pairs, and dividend_gap are currently enabled.

## Open Questions

1. **MOEX ADX Distribution**
   - What we know: YAML uses trend_threshold=30/mr_threshold=20; US code defaults are 35/15
   - What's unclear: What is the actual ADX distribution on MOEX stocks? If ADX rarely exceeds 30, momentum strategies will never fire.
   - Recommendation: Run isolation tests with ADX logging first; adjust thresholds based on empirical ADX distribution. Discretion area per CONTEXT.md.

2. **Optimal Walk-Forward Window for MOEX**
   - What we know: Default is 12mo train + 6mo test with 3mo step. MOEX has regime breaks (2022 suspension, sanctions).
   - What's unclear: Whether shorter windows (9mo train + 3mo test) would handle MOEX regime changes better.
   - Recommendation: Start with default 12+6, evaluate after first walk-forward run. Discretion area per CONTEXT.md.

3. **Pair Candidates Beyond Current Three**
   - What we know: Current pairs are SBER/VTBR, GAZP/LKOH, SBER/GMKN. Need to test 3-5 more.
   - Candidates: MGNT/FIVE (retail), NLMK/CHMF (steel), ALRS/PLZL (mining), ROSN/TATN (energy, already in ru_energy), SBER/TCSG (banks).
   - Recommendation: Test all candidates with fixed cointegration script; include only those passing p < 0.05 + half-life < 30d + Hurst < 0.5.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via uv run pytest) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/test_tune_strategy_params.py tests/unit/test_strategy_combiner.py tests/unit/test_backtest_config.py -x` |
| Full suite command | `uv run pytest` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EQF-02 | MOEX backtest produces positive PnL with walk-forward | integration (manual run) | `uv run python scripts/run_iteration.py --segments ru_blue_chips,ru_energy` | N/A (script-based validation, not unit test) |
| EQF-03 | ru_* YAML presets calibrated with MOEX-specific params | unit | `uv run pytest tests/unit/test_backtest_config.py -x` | Exists (tests MOEX uplift) |
| EQF-03 | Preset YAML loads correctly with new params | unit | `uv run pytest tests/unit/test_strategy_combiner.py -x` | Exists |
| EQF-03 | Optuna tuning produces valid params | unit | `uv run pytest tests/unit/test_tune_strategy_params.py -x` | Exists |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_tune_strategy_params.py tests/unit/test_strategy_combiner.py tests/unit/test_backtest_config.py -x`
- **Per wave merge:** `uv run pytest`
- **Phase gate:** Full suite green + walk-forward OOS Sharpe > 0.1 on 2+ segments

### Wave 0 Gaps
- [ ] `tests/unit/test_moex_preset_validation.py` -- validate ru_* YAML presets load correctly with all strategies enabled
- [ ] Fix `run_iteration.py` UNIVERSE to include `ru_finance`
- [ ] Fix `tune_strategy_params.py` to use TinkoffFetcher for `ru_*` segments
- [ ] Fix `test_pairs_cointegration.py` to use TinkoffFetcher instead of yfinance

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/finalayze/strategies/presets/ru_*.yaml` -- current preset state
- Codebase analysis: `scripts/run_strategy_isolation.py` -- isolation testing infrastructure
- Codebase analysis: `scripts/tune_strategy_params.py` -- Optuna tuning with guardrails
- Codebase analysis: `scripts/run_iteration.py` -- UNIVERSE definitions, MOEX fetcher wiring
- Codebase analysis: `src/finalayze/backtest/config.py` -- MOEX ATR uplift (1.2x), hold bars uplift (1.3x)
- Codebase analysis: `src/finalayze/backtest/costs.py` -- MOEX_COSTS preset
- Iteration results: `results/iterations/moex-new-datasources/summary.json` -- current MOEX baseline metrics

### Secondary (MEDIUM confidence)
- MOEX 2022 trading suspension (Feb 28 - Mar 28) -- widely documented fact
- MOEX ADX distribution expectations -- based on MOEX market microstructure knowledge

### Tertiary (LOW confidence)
- Optimal MOEX-specific parameter ranges (bb_std_dev, ou_window, lookbacks) -- needs empirical validation through isolation testing

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all tools exist in codebase, verified
- Architecture: HIGH -- workflow sequence derives from user decisions, scripts are wired
- Pitfalls: HIGH -- identified from code inspection (yfinance in tune script, missing UNIVERSE entry)
- Parameter ranges: LOW -- initial estimates, need empirical validation

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable -- no fast-moving external dependencies)
