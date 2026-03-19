# Phase 2: MOEX Equity Validation - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Tune ru_* strategy presets and achieve positive walk-forward backtest PnL on MOEX segments (ru_blue_chips, ru_energy, ru_finance). Phase 2 calibrates parameters for MOEX market dynamics — it does NOT add new strategies or change engine code. Uses infrastructure from Phase 1 (RUB sizing, MOEX costs, holiday calendar).

</domain>

<decisions>
## Implementation Decisions

### Strategy Mix
- Enable ALL strategies including momentum and dual_momentum on MOEX (currently disabled)
- Equal emphasis weighting (~0.15-0.20 per strategy) as starting point
- Differentiate strategy mix by sector:
  - ru_energy: heavier momentum weight (oil-driven trends) — exact split at Claude's discretion based on isolation results
  - ru_finance: heavier MR weight (banking stocks range-bound between CBR rate decisions) — exact split at Claude's discretion
  - ru_blue_chips: balanced across strategies
- Keep dividend_gap enabled (0.15-0.20 weight) — Russian dividends are 6-12%, natural MOEX edge
- Auto-disable any strategy with negative Sharpe on ALL 3 MOEX segments in isolation testing; redistribute weight

### Tuning Approach
- Use Optuna optimization with existing overfitting guardrails (DSR haircut, holdout validation, perturbation check)
- Sequence: isolation tests first (each strategy solo per segment), then combined Optuna tuning
- Maximum 5 tuning iterations before accepting results or pivoting approach
- Walk-forward window configuration: Claude's discretion (current default 12mo train + 6mo test)
- Backtest period: Claude's discretion (roadmap says 2022-2025, extending to 2020-2025 acceptable if data supports it)

### Performance Targets
- Minimum bar: out-of-sample Sharpe > 0.1 AND Profit Factor > 1.05 on at least 2 MOEX segments
- Maximum drawdown hard cap: 20% — reject any parameter set exceeding this
- Walk-forward out-of-sample metrics are the gate, not in-sample

### Pairs Selection
- Run cointegration tests (Engle-Granger or Johansen, p < 0.05) on all included pairs
- Test 3-5 additional candidate pairs beyond current 3 (MGNT/X5 retail, NLMK/CHMF steel, ALRS/PLZL mining, etc.)
- Include only pairs that pass cointegration testing
- Failed pair handling and z-score threshold tuning: Claude's discretion based on test results

### Claude's Discretion
- ATR stop multiplier calibration for MOEX (currently 1.2x uplift vs US)
- ADX regime routing thresholds (currently 30/20, US is 35/15)
- Bollinger Band std_dev (include in Optuna search space, no floor constraint)
- max_hold_bars adjustment for MOEX session length
- 2022 trading suspension period handling
- Walk-forward window size
- Per-sector exact weight splits for ru_energy (momentum tilt) and ru_finance (MR tilt)
- Whether to keep or drop pairs that fail cointegration with borderline p-values
- z_entry/z_exit thresholds for pairs strategy

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/run_strategy_isolation.py`: runs each strategy solo per segment — use for isolation phase
- `scripts/tune_strategy_params.py`: Optuna tuning with guardrails — main tuning tool
- `scripts/run_iteration.py`: combined backtest run with metrics — for combined validation
- `scripts/test_pairs_cointegration.py`: cointegration testing for pairs
- Strategy presets: `src/finalayze/strategies/presets/ru_*.yaml` — 3 equity segments + 2 OFZ (out of scope)

### Established Patterns
- ADX regime routing in `strategies/adx.py`: gates strategy firing by ADX level
- Strategy-specific ATR stops in `backtest/config.py`: `resolve_stop_atr_multiplier()` with MOEX 1.2x uplift
- Optuna guardrails: DSR haircut, holdout validation, perturbation check — already wired
- Walk-forward: 12mo train + 6mo test windows in `backtest/walk_forward.py`

### Integration Points
- YAML presets loaded by `StrategyCombiner` — edit YAML files, no code changes needed
- `run_iteration.py` reads segment config from `config/segments.py` — MOEX segments already defined
- `TinkoffFetcher` provides MOEX candle data via T-Invest gRPC API
- Results saved to `results/iterations/` for comparison via `iteration-history` skill

</code_context>

<specifics>
## Specific Ideas

- Run isolation first to identify which strategies actually work on MOEX before combining
- Use backtest-iteration skill after each tuning round to track metrics trajectory
- ru_energy presets should reflect commodity-driven price behavior (oil correlation)
- Current ru_* presets look like US copies with minor tweaks — need genuine MOEX calibration

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-moex-equity-validation*
*Context gathered: 2026-03-14*
