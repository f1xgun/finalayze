# Phase 6B: Strategy Quality Fixes

**Date:** 2026-03-01
**Scope:** 8 tasks addressing strategy correctness, backtest fidelity, and position sizing
**Estimated effort:** ~2 weeks
**Prerequisite:** Phase 5 (critical safety fixes) should be complete first

---

## Task Summary

| # | Finding | Complexity | Files Modified | Tests Added | Depends On |
|---|---------|------------|----------------|-------------|------------|
| 6B.1 | Momentum `_SignalState` shared across segments | M | `strategies/momentum.py` | ~4 | None |
| 6B.2 | Combiner normalizes by firing weight, not total configured weight | M | `strategies/combiner.py` | ~4 | None |
| 6B.3 | Mean reversion has no exit-at-mean signal | M | `strategies/mean_reversion.py` | ~4 | None |
| 6B.4 | MOEX commission model is 30x too low | M | `backtest/costs.py` | ~3 | None |
| 6B.5 | Walk-forward discards training data, no optimization | M | `backtest/walk_forward.py` | ~5 | None |
| 6B.6 | No portfolio-level backtest simulation | L | `backtest/engine.py` | ~6 | 6B.4 |
| 6B.7 | No volatility-adjusted position sizing | M | `risk/position_sizer.py`, `backtest/engine.py` | ~4 | None |
| 6B.8 | Direct `broker._cash` mutation in backtest engine | S | `execution/simulated_broker.py`, `backtest/engine.py` | ~3 | None |

---

## Execution Order

```
Wave 1 (independent, parallelizable):
  6B.1  Momentum per-segment state
  6B.2  Combiner normalization mode
  6B.3  Mean reversion exit signal
  6B.4  MOEX commission model
  6B.8  SimulatedBroker deduct_fees

Wave 2 (depends on Wave 1):
  6B.5  Walk-forward optimization     (independent)
  6B.7  Volatility-adjusted sizing    (independent)
  6B.6  Portfolio-level backtest       (depends on 6B.4 for correct multi-market costs)
```

---

## Task 6B.1: Per-Segment `_SignalState` in Momentum Strategy

### Problem

`MomentumStrategy.__init__()` creates a single `_SignalState` instance (line 77 of `momentum.py`). When the same `MomentumStrategy` instance is used across multiple segments (e.g., `us_tech` and `us_broad`), the signal deduplication state leaks between segments. A BUY emitted for symbol `AAPL` in segment `us_tech` will suppress the same BUY for `AAPL` in segment `us_broad`, even though these are independent strategy contexts.

Additionally, `generate_signal()` mutates `self._signal_state._neutral_reset_bars` (line 126) on every call using the current segment's params, which means the reset interval for segment A can be overwritten by a subsequent call for segment B.

### Files to Modify

- `src/finalayze/strategies/momentum.py` (lines 40-63, 76-77, 122-128)

### Changes

1. Change `_SignalState` internal dicts to be keyed by `(segment_id, symbol)` tuples instead of just `symbol`, OR change `MomentumStrategy` to hold a `dict[str, _SignalState]` keyed by `segment_id`.

   The cleaner approach is option B -- a dict of `_SignalState` per segment:

   ```python
   # In MomentumStrategy.__init__ (line 77):
   # BEFORE:
   self._signal_state = _SignalState()

   # AFTER:
   self._signal_states: dict[str, _SignalState] = {}
   ```

2. Add a helper to get or create a per-segment state:

   ```python
   def _get_signal_state(self, segment_id: str, neutral_reset_bars: int) -> _SignalState:
       if segment_id not in self._signal_states:
           self._signal_states[segment_id] = _SignalState(neutral_reset_bars)
       return self._signal_states[segment_id]
   ```

3. Update `generate_signal()` (lines 122-128) to use the per-segment state:

   ```python
   # BEFORE (lines 122-128):
   neutral_reset_bars = int(params.get("neutral_reset_bars", _DEFAULT_NEUTRAL_RESET_BARS))
   self._signal_state._neutral_reset_bars = neutral_reset_bars
   self._signal_state.tick(symbol)

   # AFTER:
   neutral_reset_bars = int(params.get("neutral_reset_bars", _DEFAULT_NEUTRAL_RESET_BARS))
   signal_state = self._get_signal_state(segment_id, neutral_reset_bars)
   signal_state.tick(symbol)
   ```

4. Update all references from `self._signal_state` to `signal_state` in `generate_signal()` (line 143):

   ```python
   # BEFORE (line 143):
   if not self._signal_state.should_emit(symbol, direction):

   # AFTER:
   if not signal_state.should_emit(symbol, direction):
   ```

### Tests to Write

In `tests/unit/test_strategies.py`:

1. **`test_signal_state_isolated_per_segment`** -- Create one `MomentumStrategy` instance. Generate a BUY signal for `AAPL` in `us_tech`. Verify that the same candle data still produces a BUY signal for `AAPL` in `us_broad` (state is not shared).

2. **`test_signal_state_duplicate_suppressed_within_segment`** -- Confirm that calling `generate_signal` twice for the same `(symbol, segment_id)` still suppresses duplicates within a single segment.

3. **`test_neutral_reset_bars_per_segment`** -- Use two segments with different `neutral_reset_bars` values. Confirm each segment respects its own reset interval independently.

4. **`test_signal_state_no_mutation_of_neutral_reset`** -- Confirm that calling `generate_signal` for segment A does not change the `neutral_reset_bars` value for an already-created segment B state.

### TDD Sequence

1. Write test `test_signal_state_isolated_per_segment` -- expect it to FAIL (current code shares state).
2. Implement per-segment `_signal_states` dict.
3. Run tests -- PASS.
4. Write remaining tests, verify all pass.
5. `uv run ruff check . && uv run mypy src/`

---

## Task 6B.2: Combiner Normalization Mode (Firing vs Total Weight)

### Problem

`StrategyCombiner.generate_signal()` (line 80 of `combiner.py`) normalizes the weighted score by `total_weight`, which is the sum of weights of strategies that actually fired (returned a non-None signal). This means if only 1 out of 4 strategies fires with confidence 0.9, the net score is `0.9 * weight / weight = 0.9`, which can overstate consensus.

The alternative is to normalize by the sum of all enabled strategy weights (regardless of whether they fired), which would produce a lower score reflecting that only 1 out of 4 strategies had an opinion.

Both modes have valid use cases, so this should be configurable.

### Files to Modify

- `src/finalayze/strategies/combiner.py` (lines 31-103)

### Changes

1. Add a `normalize_mode` parameter to `StrategyCombiner.__init__()`:

   ```python
   def __init__(
       self,
       strategies: list[BaseStrategy],
       normalize_mode: str = "firing",  # "firing" or "total"
   ) -> None:
       self._strategies: dict[str, BaseStrategy] = {s.name: s for s in strategies}
       self._presets_dir = _PRESETS_DIR
       self._normalize_mode = normalize_mode
   ```

2. In `generate_signal()`, compute `total_enabled_weight` by iterating enabled strategies and summing their weights (lines 49-55), regardless of whether they fire:

   ```python
   total_enabled_weight = _ZERO
   # ... in the loop over strategies_cfg:
   if not strategy_cfg.get("enabled", True):
       continue
   try:
       weight = Decimal(str(strategy_cfg.get("weight", "1.0")))
   except InvalidOperation:
       weight = Decimal("1.0")
   total_enabled_weight += weight

   # Only proceed with signal generation if strategy exists
   strategy = self._strategies.get(strategy_name)
   if strategy is None:
       continue
   # ... rest of signal generation
   ```

3. Choose normalization denominator based on mode (replace line 80):

   ```python
   # BEFORE (line 80):
   net = weighted_score / total_weight

   # AFTER:
   denominator = total_enabled_weight if self._normalize_mode == "total" else total_weight
   if denominator == _ZERO:
       return None
   net = weighted_score / denominator
   ```

4. Keep the existing `total_weight == _ZERO` check (line 77) for the case where no strategies fired at all.

### Tests to Write

In `tests/unit/test_strategy_combiner.py`:

1. **`test_normalize_firing_mode_default`** -- Confirm the default behavior (normalizing by firing weight) remains unchanged. One strategy fires with BUY 0.9, weight 0.5 out of two enabled. Net = 0.9 (only firing weight in denominator).

2. **`test_normalize_total_mode_reduces_score`** -- Same setup but with `normalize_mode="total"`. Net = 0.9 * 0.5 / 1.0 = 0.45, which is below `MIN_COMBINED_CONFIDENCE` -> returns None.

3. **`test_normalize_total_mode_strong_consensus`** -- Two strategies both fire BUY in "total" mode. Confirm the combined signal passes.

4. **`test_normalize_total_accounts_for_enabled_only`** -- Three strategies configured, one disabled. Total weight uses only the two enabled strategies' weights.

### TDD Sequence

1. Write test `test_normalize_total_mode_reduces_score` -- expect FAIL (no `normalize_mode` param).
2. Implement `normalize_mode` parameter and dual-mode normalization.
3. Run tests -- PASS.
4. Write remaining tests.
5. `uv run ruff check . && uv run mypy src/`

---

## Task 6B.3: Mean Reversion Exit-at-Mean Signal

### Problem

`MeanReversionStrategy` generates BUY below the lower Bollinger Band and SELL above the upper band, but has no mechanism to signal an exit when price reverts to the mean (middle band). The current code (line 142-143 of `mean_reversion.py`) only resets internal state when price returns inside the bands; it never emits a counter-signal to close the position.

A mean reversion strategy should ideally generate a SELL signal (to exit a long) when price crosses back above the middle band after a BUY, and a BUY signal (to exit a short) when price crosses back below the middle band after a SELL.

### Files to Modify

- `src/finalayze/strategies/mean_reversion.py` (lines 84-183)
- `src/finalayze/core/schemas.py` -- verify `SignalDirection` includes `SELL` (it does)

### Changes

1. Add an `exit_at_mean` parameter (default `False` to maintain backward compatibility):

   ```python
   exit_at_mean = bool(params.get("exit_at_mean", False))
   ```

2. In the `else` branch (lines 141-143), where price is inside the bands and we currently just reset state, add exit signal logic:

   ```python
   # BEFORE (lines 141-143):
   else:
       # Price has returned inside the bands -- reset active signal state
       self._active_signal.pop(symbol, None)

   # AFTER:
   else:
       # Price has returned inside the bands
       active = self._active_signal.pop(symbol, None)
       if exit_at_mean and active is not None:
           # Price crossed back to mean -- emit exit signal
           # If we had a BUY (long entry), emit SELL to close
           # If we had a SELL (short entry), emit BUY to close
           exit_direction = (
               SignalDirection.SELL if active == SignalDirection.BUY
               else SignalDirection.BUY
           )
           # Confidence based on how close to the middle band
           mid_distance = abs(last_close - mid) / band_width if band_width > 0 else 0.0
           exit_confidence = min(1.0, 0.6 + (1.0 - mid_distance) * 0.3)
           if Decimal(str(exit_confidence)) >= min_confidence:
               return Signal(
                   strategy_name=self.name,
                   symbol=symbol,
                   market_id=candles[0].market_id,
                   segment_id=segment_id,
                   direction=exit_direction,
                   confidence=exit_confidence,
                   features={
                       "bb_lower": lower,
                       "bb_upper": upper,
                       "bb_mid": mid,
                       "close": last_close,
                       "exit_type": "mean_reversion_exit",
                   },
                   reasoning=(
                       f"Price {last_close:.2f} returned to mean region "
                       f"BB [{lower:.2f}, {upper:.2f}] (exit at mean)"
                   ),
               )
   ```

3. The existing `direction is None` check (line 145) and RSI confirmation (lines 149-154) only apply to entry signals and should remain unchanged.

### Tests to Write

In `tests/unit/test_strategies.py` (or a new `tests/unit/test_mean_reversion.py`):

1. **`test_exit_at_mean_generates_sell_after_buy`** -- Price drops below lower BB (BUY emitted), then price rises back to the middle band. With `exit_at_mean=True`, a SELL signal is emitted.

2. **`test_exit_at_mean_generates_buy_after_sell`** -- Price rises above upper BB (SELL emitted), then price drops back to the middle band. With `exit_at_mean=True`, a BUY signal is emitted.

3. **`test_exit_at_mean_disabled_by_default`** -- Same scenario as test 1, but without setting `exit_at_mean`. No exit signal emitted, only state reset.

4. **`test_exit_at_mean_no_signal_without_active_position`** -- Price is inside bands with no prior active signal. No exit signal emitted regardless of `exit_at_mean` setting.

### TDD Sequence

1. Write test `test_exit_at_mean_generates_sell_after_buy` -- expect FAIL.
2. Implement exit-at-mean logic.
3. Run tests -- PASS.
4. Write remaining tests.
5. `uv run ruff check . && uv run mypy src/`

---

## Task 6B.4: Fix MOEX Commission Model

### Problem

`MOEX_COSTS` in `backtest/costs.py` (line 64-69) uses `commission_per_share=Decimal("0.003")`. The `total_cost()` method (line 42) computes commission as `commission_per_share * quantity`. For a 100-ruble stock with 100 shares, this gives `0.003 * 100 = 0.30 RUB`, whereas the real MOEX/Tinkoff commission is `price * quantity * 0.0003 = 100 * 100 * 0.0003 = 3.0 RUB` -- a 10x understatement. For higher-priced stocks, the error is even larger.

The root cause is that MOEX commissions are percentage-of-value (0.03% = 3 bps), not per-share. The `TransactionCosts` dataclass has no field for this.

### Files to Modify

- `src/finalayze/backtest/costs.py` (lines 16-69)

### Changes

1. Add a `commission_rate` field to `TransactionCosts` for percentage-of-value commission:

   ```python
   @dataclass(frozen=True)
   class TransactionCosts:
       commission_per_share: Decimal = Decimal("0.005")
       min_commission: Decimal = Decimal("1.00")
       spread_bps: Decimal = Decimal(5)
       slippage_bps: Decimal = Decimal(3)
       commission_rate: Decimal = Decimal(0)  # NEW: percentage of trade value (e.g., 0.0003 for 3 bps)
   ```

2. Update `total_cost()` to use `commission_rate` when it is non-zero:

   ```python
   def total_cost(self, price: Decimal, quantity: Decimal) -> Decimal:
       if self.commission_rate > 0:
           commission = max(self.min_commission, price * quantity * self.commission_rate)
       else:
           commission = max(self.min_commission, self.commission_per_share * quantity)
       spread = price * self.spread_bps / _BPS_DIVISOR
       slippage = price * self.slippage_bps / _BPS_DIVISOR
       return commission + (spread + slippage) * quantity
   ```

3. Update `MOEX_COSTS` to use the rate-based model:

   ```python
   MOEX_COSTS = TransactionCosts(
       commission_per_share=Decimal(0),          # Not used for MOEX
       commission_rate=Decimal("0.0003"),         # 0.03% of trade value (Tinkoff Invest standard)
       min_commission=Decimal("0.10"),
       spread_bps=Decimal(10),
       slippage_bps=Decimal(7),
   )
   ```

### Tests to Write

In `tests/unit/test_costs.py` (new file) or `tests/unit/test_backtest_engine.py`:

1. **`test_us_costs_per_share_unchanged`** -- Verify `US_COSTS.total_cost(Decimal(150), Decimal(100))` matches expected per-share commission model.

2. **`test_moex_costs_rate_based`** -- Verify `MOEX_COSTS.total_cost(Decimal(100), Decimal(100))` computes commission as `100 * 100 * 0.0003 = 3.0 RUB` (not the old 0.30 RUB).

3. **`test_commission_rate_respects_min_commission`** -- Verify that for very small trades, `min_commission` is applied.

### TDD Sequence

1. Write test `test_moex_costs_rate_based` -- expect FAIL (no `commission_rate` field).
2. Add `commission_rate` field and update `total_cost()`.
3. Run tests -- PASS.
4. Update `MOEX_COSTS` preset.
5. `uv run ruff check . && uv run mypy src/`

---

## Task 6B.5: Walk-Forward Optimization with Parameter Grid Search

### Problem

`WalkForwardOptimizer.run()` (lines 110-161 of `walk_forward.py`) splits data into train/test windows but completely discards the training data (line 142: `_train, test = self.split_candles(...)`). The train set is never used. There is no parameter optimization -- the same engine with the same strategy parameters is run on every test window.

A proper walk-forward optimizer should:
1. Run parameter grid search on the training window.
2. Select the best parameters.
3. Evaluate those parameters on the test window (out-of-sample).

### Files to Modify

- `src/finalayze/backtest/walk_forward.py` (lines 58-161)

### Changes

1. Add a `ParameterGrid` type alias and an optional `param_grid` argument to `WalkForwardOptimizer`:

   ```python
   from typing import Callable

   ParameterGrid = dict[str, list[object]]

   class WalkForwardOptimizer:
       def __init__(
           self,
           config: WalkForwardConfig | None = None,
           param_grid: ParameterGrid | None = None,
           engine_factory: Callable[[dict[str, object]], BacktestEngine] | None = None,
       ) -> None:
           self._config = config or WalkForwardConfig()
           self._param_grid = param_grid
           self._engine_factory = engine_factory
   ```

2. Add a `_optimize_on_train()` method that runs grid search over the training candles:

   ```python
   def _optimize_on_train(
       self,
       symbol: str,
       segment_id: str,
       train_candles: list[Candle],
       default_engine: BacktestEngine,
   ) -> BacktestEngine:
       """Find best parameters on training data via grid search.

       Returns the engine with the best-performing parameter set,
       or the default engine if no param_grid/engine_factory is configured.
       """
       if not self._param_grid or not self._engine_factory:
           return default_engine

       best_sharpe = float("-inf")
       best_engine = default_engine

       for combo in _iter_param_combinations(self._param_grid):
           engine = self._engine_factory(combo)
           trades, snapshots = engine.run(symbol, segment_id, train_candles)
           equities = [float(s.equity) for s in snapshots]
           sharpe = _compute_sharpe_from_snapshots(equities)
           if sharpe > best_sharpe:
               best_sharpe = sharpe
               best_engine = engine

       return best_engine
   ```

3. Add `_iter_param_combinations()` helper:

   ```python
   def _iter_param_combinations(grid: ParameterGrid) -> list[dict[str, object]]:
       """Generate all combinations from a parameter grid."""
       import itertools
       keys = list(grid.keys())
       values = list(grid.values())
       combos = []
       for combo in itertools.product(*values):
           combos.append(dict(zip(keys, combo)))
       return combos
   ```

4. Update `run()` to use training data (line 142-145):

   ```python
   # BEFORE:
   _train, test = self.split_candles(candles, window)
   if not test:
       continue
   trades, snapshots = engine.run(symbol, segment_id, test)

   # AFTER:
   train, test = self.split_candles(candles, window)
   if not test:
       continue
   optimized_engine = self._optimize_on_train(
       symbol, segment_id, train, engine
   )
   trades, snapshots = optimized_engine.run(symbol, segment_id, test)
   ```

5. Add `best_params` tracking to `WalkForwardResult`:

   ```python
   @dataclass
   class WalkForwardResult:
       windows: list[WalkForwardWindow] = field(default_factory=list)
       oos_trades: list[TradeResult] = field(default_factory=list)
       total_oos_trades: int = 0
       oos_sharpe: float = 0.0
       oos_total_return_pct: float = 0.0
       oos_win_rate: float = 0.0
       oos_max_drawdown_pct: float = 0.0
       per_window_params: list[dict[str, object]] = field(default_factory=list)  # NEW
   ```

### Tests to Write

In `tests/unit/test_walk_forward.py`:

1. **`test_walk_forward_without_grid_uses_default_engine`** -- Existing behavior: no `param_grid` -> runs default engine on test windows (backward compatible).

2. **`test_walk_forward_with_grid_optimizes_on_train`** -- Provide a `param_grid` with 2 sets of `kelly_fraction` values and an `engine_factory`. Verify the optimizer calls the factory and selects the best.

3. **`test_optimize_on_train_selects_best_sharpe`** -- Unit test `_optimize_on_train` directly. Mock two engines returning different Sharpe ratios. Verify the higher-Sharpe engine is selected.

4. **`test_iter_param_combinations`** -- Verify grid `{"a": [1, 2], "b": [3, 4]}` produces 4 combinations.

5. **`test_walk_forward_train_data_not_discarded`** -- Verify that `train` candles are passed to `_optimize_on_train` (not discarded like before).

### TDD Sequence

1. Write test `test_walk_forward_with_grid_optimizes_on_train` -- expect FAIL.
2. Implement `_optimize_on_train`, `_iter_param_combinations`, and `engine_factory`.
3. Run tests -- PASS.
4. `uv run ruff check . && uv run mypy src/`

---

## Task 6B.6: Portfolio-Level Backtest Simulation

### Problem

`BacktestEngine.run()` (line 73-251 of `engine.py`) accepts a single `symbol: str` parameter. There is no way to backtest a portfolio of multiple symbols simultaneously, which means position-level interactions (capital allocation, diversification, rebalancing) are not captured.

### Files to Modify

- `src/finalayze/backtest/engine.py` (lines 73-251)

### Changes

1. Add a new `run_portfolio()` method alongside the existing `run()` (do NOT modify `run()` to preserve backward compatibility):

   ```python
   def run_portfolio(
       self,
       symbols: list[str],
       segment_id: str,
       candles_by_symbol: dict[str, list[Candle]],
   ) -> tuple[list[TradeResult], list[PortfolioState]]:
       """Run a portfolio-level backtest over multiple symbols.

       Iterates through time, generating signals for each symbol on each bar,
       managing shared capital across all positions.

       Args:
           symbols: List of ticker symbols to trade.
           segment_id: Market segment identifier.
           candles_by_symbol: Candle data keyed by symbol.

       Returns:
           A tuple of (trades, portfolio_snapshots).
       """
   ```

2. The implementation aligns timestamps across symbols, then on each bar:
   - Updates prices for all symbols.
   - Checks stop-losses for all symbols.
   - Generates signals for each symbol using its candle history.
   - Processes BUY/SELL orders using shared `SimulatedBroker` capital.
   - Records a single portfolio snapshot per bar (reflecting all positions).

3. Build a unified timeline from all candle timestamps:

   ```python
   all_timestamps = sorted(
       {c.timestamp for candles in candles_by_symbol.values() for c in candles}
   )
   ```

4. For each timestamp, iterate over symbols that have data at that timestamp.

### Tests to Write

In `tests/unit/test_backtest_engine.py`:

1. **`test_portfolio_backtest_two_symbols`** -- Run with 2 symbols where one gets a BUY, verify capital is shared correctly.

2. **`test_portfolio_backtest_respects_max_positions`** -- Set `max_positions=1`. Verify only one position is opened even when two symbols generate BUY signals.

3. **`test_portfolio_backtest_single_symbol_matches_run`** -- Verify `run_portfolio(["SYM"], ...)` produces equivalent results to `run("SYM", ...)`.

4. **`test_portfolio_backtest_empty_symbols`** -- Empty symbol list returns empty trades and snapshots.

5. **`test_portfolio_backtest_unaligned_timestamps`** -- Symbols with different candle date ranges. Verify correct handling (no crash, proper time alignment).

6. **`test_portfolio_backtest_transaction_costs`** -- Verify transaction costs are applied per-trade across multiple symbols.

### TDD Sequence

1. Write test `test_portfolio_backtest_two_symbols` -- expect FAIL (no `run_portfolio` method).
2. Implement `run_portfolio()`.
3. Run tests -- PASS.
4. Write remaining tests.
5. `uv run ruff check . && uv run mypy src/`

### Dependencies

- **6B.4** should be complete first so that MOEX-market portfolio backtests use correct commission models.

---

## Task 6B.7: Volatility-Adjusted Position Sizing

### Problem

`compute_position_size()` in `risk/position_sizer.py` uses only Kelly criterion to size positions. It does not account for the asset's realized volatility, leading to oversized positions in high-volatility assets and undersized positions in low-volatility assets.

### Files to Modify

- `src/finalayze/risk/position_sizer.py` (lines 14-50)
- `src/finalayze/backtest/engine.py` (lines 258-288, the `_handle_buy` method)

### Changes

1. Add a new `compute_vol_adjusted_position_size()` function in `position_sizer.py`:

   ```python
   def compute_vol_adjusted_position_size(
       base_position: Decimal,
       target_vol: Decimal,
       asset_vol: Decimal,
       min_scale: Decimal = Decimal("0.25"),
       max_scale: Decimal = Decimal("2.0"),
   ) -> Decimal:
       """Scale position size by target_vol / asset_vol.

       Args:
           base_position: Position size from Kelly or other sizer (currency units).
           target_vol: Target annualized portfolio volatility (e.g., 0.15 for 15%).
           asset_vol: Realized annualized volatility of the asset.
           min_scale: Minimum scaling factor (floor).
           max_scale: Maximum scaling factor (cap).

       Returns:
           Adjusted position size.
       """
       if asset_vol <= 0:
           return base_position
       scale = target_vol / asset_vol
       scale = max(min_scale, min(max_scale, scale))
       return base_position * scale
   ```

2. Add a helper to compute realized volatility from candles:

   ```python
   def compute_realized_vol(candles: list[Candle], lookback: int = 20) -> Decimal | None:
       """Compute annualized realized volatility from daily log returns.

       Uses the last `lookback` candles. Returns None if insufficient data.
       """
       if len(candles) < lookback + 1:
           return None
       import math
       closes = [float(c.close) for c in candles[-(lookback + 1):]]
       log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
       if not log_returns:
           return None
       import statistics
       daily_vol = statistics.stdev(log_returns)
       annualized = daily_vol * math.sqrt(252)
       return Decimal(str(annualized))
   ```

3. Update `BacktestEngine._handle_buy()` to optionally apply vol-scaling:

   Add `target_vol: Decimal | None = None` to `BacktestEngine.__init__()`. In `_handle_buy()`, after computing `position_value` (line 288):

   ```python
   if self._target_vol is not None:
       asset_vol = compute_realized_vol(history)
       if asset_vol is not None and asset_vol > 0:
           position_value = compute_vol_adjusted_position_size(
               base_position=position_value,
               target_vol=self._target_vol,
               asset_vol=asset_vol,
           )
   ```

### Tests to Write

In `tests/unit/test_strategies.py` or a new `tests/unit/test_position_sizer.py`:

1. **`test_vol_scaling_reduces_size_for_high_vol`** -- Asset vol = 0.40, target vol = 0.15. Verify position is scaled down by ~0.375x.

2. **`test_vol_scaling_increases_size_for_low_vol`** -- Asset vol = 0.08, target vol = 0.15. Verify position is scaled up by ~1.875x.

3. **`test_vol_scaling_clamped_to_bounds`** -- Extreme vol ratios. Verify `min_scale` and `max_scale` are respected.

4. **`test_vol_scaling_zero_vol_returns_base`** -- Asset vol = 0. Verify base position is returned unchanged.

### TDD Sequence

1. Write test `test_vol_scaling_reduces_size_for_high_vol` -- expect FAIL.
2. Implement `compute_vol_adjusted_position_size()` and `compute_realized_vol()`.
3. Run tests -- PASS.
4. Wire into `BacktestEngine._handle_buy()`.
5. `uv run ruff check . && uv run mypy src/`

---

## Task 6B.8: Replace Direct `broker._cash` Mutation

### Problem

`BacktestEngine._handle_buy()` (line 327 of `engine.py`) directly mutates the broker's private `_cash` attribute:

```python
broker._cash -= cost
```

This violates encapsulation, makes the `SimulatedBroker` harder to reason about (cash can change without any corresponding order), and would break if `SimulatedBroker` were refactored.

### Files to Modify

- `src/finalayze/execution/simulated_broker.py` (add public method)
- `src/finalayze/backtest/engine.py` (line 327, replace private access)

### Changes

1. Add a `deduct_fees(amount: Decimal) -> None` method to `SimulatedBroker`:

   ```python
   def deduct_fees(self, amount: Decimal) -> None:
       """Deduct transaction fees from available cash.

       This is used by the backtest engine to account for commission,
       spread, and slippage costs that are not part of the order fill.

       Args:
           amount: Fee amount to deduct (must be non-negative).

       Raises:
           ValueError: If amount is negative.
       """
       if amount < 0:
           msg = f"Fee amount must be non-negative, got {amount}"
           raise ValueError(msg)
       self._cash -= amount
   ```

2. Update `BacktestEngine._handle_buy()` (line 327):

   ```python
   # BEFORE (line 327):
   broker._cash -= cost

   # AFTER:
   broker.deduct_fees(cost)
   ```

### Tests to Write

In `tests/unit/test_backtest_engine.py` or a new `tests/unit/test_simulated_broker.py`:

1. **`test_deduct_fees_reduces_cash`** -- Call `broker.deduct_fees(Decimal(10))`. Verify `broker.get_portfolio().cash` decreased by 10.

2. **`test_deduct_fees_negative_raises`** -- Call `broker.deduct_fees(Decimal(-1))`. Verify `ValueError` is raised.

3. **`test_backtest_engine_uses_deduct_fees`** -- Run a backtest with transaction costs. Verify the engine no longer accesses `broker._cash` directly (could use `monkeypatch` to make `_cash` raise on access, or just verify correct fee deduction via portfolio state).

### TDD Sequence

1. Write test `test_deduct_fees_reduces_cash` -- expect FAIL (no `deduct_fees` method).
2. Add `deduct_fees()` to `SimulatedBroker`.
3. Run tests -- PASS.
4. Update `BacktestEngine._handle_buy()` to use `deduct_fees()`.
5. Write `test_deduct_fees_negative_raises`.
6. `uv run ruff check . && uv run mypy src/`

---

## File Summary Table

| File | Tasks | Type of Change |
|------|-------|----------------|
| `src/finalayze/strategies/momentum.py` | 6B.1 | Per-segment `_SignalState` dict |
| `src/finalayze/strategies/combiner.py` | 6B.2 | Add `normalize_mode` param and total-weight denominator |
| `src/finalayze/strategies/mean_reversion.py` | 6B.3 | Add exit-at-mean signal on band crossback |
| `src/finalayze/backtest/costs.py` | 6B.4 | Add `commission_rate` field, fix `MOEX_COSTS` |
| `src/finalayze/backtest/walk_forward.py` | 6B.5 | Add `param_grid`, `engine_factory`, train-window optimization |
| `src/finalayze/backtest/engine.py` | 6B.6, 6B.7, 6B.8 | Add `run_portfolio()`, wire vol-scaling, replace `_cash` mutation |
| `src/finalayze/execution/simulated_broker.py` | 6B.8 | Add `deduct_fees()` public method |
| `src/finalayze/risk/position_sizer.py` | 6B.7 | Add `compute_vol_adjusted_position_size()`, `compute_realized_vol()` |

### Test Files

| Test File | Tasks Covered |
|-----------|---------------|
| `tests/unit/test_strategies.py` | 6B.1, 6B.3 |
| `tests/unit/test_strategy_combiner.py` | 6B.2 |
| `tests/unit/test_costs.py` (new) | 6B.4 |
| `tests/unit/test_walk_forward.py` | 6B.5 |
| `tests/unit/test_backtest_engine.py` | 6B.6, 6B.8 |
| `tests/unit/test_position_sizer.py` (new or extend existing) | 6B.7 |

---

## Verification Checklist

After all 8 tasks are complete:

- [ ] `uv run pytest tests/unit/test_strategies.py -v` -- all pass
- [ ] `uv run pytest tests/unit/test_strategy_combiner.py -v` -- all pass
- [ ] `uv run pytest tests/unit/test_costs.py -v` -- all pass
- [ ] `uv run pytest tests/unit/test_walk_forward.py -v` -- all pass
- [ ] `uv run pytest tests/unit/test_backtest_engine.py -v` -- all pass
- [ ] `uv run pytest tests/unit/test_position_sizer.py -v` -- all pass
- [ ] `uv run ruff check .` -- no lint errors
- [ ] `uv run ruff format --check .` -- no formatting issues
- [ ] `uv run mypy src/` -- no type errors
- [ ] No YAML files are loaded in any unit test (params passed directly)
- [ ] `ddof=1` preserved in `pairs.py` spread std
- [ ] Momentum regime lookback still uses 200-period SMA where configured
- [ ] All signal confidence values remain in `[0.0, 1.0]`
- [ ] Backward compatibility: existing tests pass without modification (new features are opt-in)
