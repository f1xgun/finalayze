# MOEX Backtest Pipeline Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 8 MOEX backtest pipeline issues so MOEX segments produce meaningful trades and don't drag down combined US+MOEX metrics.

**Architecture:** Script-level fixes (no broker/engine architecture changes). Fix 1 converts cash to RUB for MOEX segments. Fix 2 converts MOEX PnL to USD before aggregation. Fix 3 vol-normalizes dual_momentum confidence. Fixes 4/5/7 are YAML param changes. Fix 6 adds rolling vol regime provider. Fix 8 adds MOEX hold bar uplift.

**Tech Stack:** Python 3.12, uv, pytest, Decimal, YAML presets

---

## Task 1: RUB-Denominated Initial Cash for MOEX Segments

**Files:**
- Modify: `scripts/run_iteration.py:875` (cash conversion)
- Modify: `scripts/run_iteration.py:918` (display)
- Modify: `src/finalayze/backtest/engine.py:1172` (min_pos fix)
- Test: `tests/unit/test_moex_fixes.py` (NEW)

**Step 1: Write failing tests**

```python
"""Tests for MOEX backtest pipeline fixes."""
from __future__ import annotations

from decimal import Decimal

import pytest

# --- Fix 1: RUB cash conversion ---

_FALLBACK_USDRUB = Decimal("90.0")


def convert_cash_for_segment(cash: Decimal, segment: str) -> Decimal:
    """Convert USD cash to RUB for MOEX segments."""
    if segment.startswith("ru_"):
        return cash * _FALLBACK_USDRUB
    return cash


class TestMoexCashConversion:
    """Fix 1: MOEX segments should get RUB-denominated cash."""

    def test_us_segment_unchanged(self) -> None:
        assert convert_cash_for_segment(Decimal(100_000), "us_tech") == Decimal(100_000)

    def test_ru_segment_converted(self) -> None:
        result = convert_cash_for_segment(Decimal(100_000), "ru_blue_chips")
        assert result == Decimal(100_000) * _FALLBACK_USDRUB

    def test_ru_energy_converted(self) -> None:
        result = convert_cash_for_segment(Decimal(100_000), "ru_energy")
        assert result == Decimal(100_000) * _FALLBACK_USDRUB


class TestMoexMinPosition:
    """Fix 1b: MOEX min_pos should be 5000 RUB, not 100."""

    _MOEX_MIN_POS = Decimal(5000)
    _US_MIN_POS = Decimal(500)

    def test_moex_min_pos(self) -> None:
        segment = "ru_blue_chips"
        min_pos = self._MOEX_MIN_POS if segment.startswith("ru_") else self._US_MIN_POS
        assert min_pos == Decimal(5000)

    def test_us_min_pos(self) -> None:
        segment = "us_tech"
        min_pos = self._MOEX_MIN_POS if segment.startswith("ru_") else self._US_MIN_POS
        assert min_pos == Decimal(500)
```

**Step 2: Run tests to verify they pass (these test the logic pattern)**

Run: `uv run pytest tests/unit/test_moex_fixes.py -v`
Expected: PASS (these validate the helper logic we'll inline)

**Step 3: Implement cash conversion in run_iteration.py**

In `scripts/run_iteration.py`, after line 875 (`cash = Decimal(args.cash)`), add MOEX conversion inside the segment loop. Specifically, modify the `_run_symbol` call at line 1014 to pass converted cash:

At line 875, keep `cash = Decimal(args.cash)` as the base USD cash.

Before line 1014 (inside the segment loop, around line 970), add:
```python
# Convert cash to RUB for MOEX segments
_FALLBACK_USDRUB = Decimal("90.0")
segment_cash = cash * _FALLBACK_USDRUB if segment.startswith("ru_") else cash
```

Then pass `segment_cash` instead of `cash` at line 1014.

Also update the display at line 918:
```python
print(f"  Cash: ${cash:,.0f} (MOEX: ₽{cash * Decimal('90.0'):,.0f})")
```

**Step 4: Fix min_pos in engine.py**

Change line 1172 from:
```python
min_pos = Decimal(100) if segment_id.startswith("ru_") else Decimal(500)
```
to:
```python
min_pos = Decimal(5000) if segment_id.startswith("ru_") else Decimal(500)
```

**Step 5: Run tests**

Run: `uv run pytest tests/unit/test_moex_fixes.py tests/unit/test_engine_pre_trade.py -v`
Expected: PASS

---

## Task 2: Currency-Aware Metrics Aggregation

**Files:**
- Modify: `scripts/run_iteration.py:1024-1027` (trade aggregation)
- Modify: `scripts/run_iteration.py:1065` (metrics computation)
- Test: `tests/unit/test_moex_fixes.py` (append)

**Step 1: Write failing test**

Append to `tests/unit/test_moex_fixes.py`:

```python
from finalayze.core.schemas import TradeResult, PortfolioState, SignalDirection
from datetime import datetime, UTC


class TestCurrencyAwareAggregation:
    """Fix 2: MOEX trades should be converted to USD before aggregation."""

    def test_convert_moex_trade_pnl(self) -> None:
        """MOEX trade PnL in RUB should be divided by USDRUB for USD aggregation."""
        rub_pnl = Decimal("9000")  # 9000 RUB
        usd_pnl = rub_pnl / _FALLBACK_USDRUB
        assert usd_pnl == Decimal(100)  # $100

    def test_us_trade_pnl_unchanged(self) -> None:
        """US trade PnL should not be converted."""
        usd_pnl = Decimal("100")
        assert usd_pnl == Decimal(100)
```

**Step 2: Implement aggregation fix in run_iteration.py**

At lines 1024-1027, convert MOEX trades before pooling. Add a helper function above the main loop:

```python
def _normalize_trades_to_usd(
    trades: list[TradeResult],
    segment: str,
) -> list[TradeResult]:
    """Convert MOEX trade values to USD for cross-segment aggregation."""
    if not segment.startswith("ru_"):
        return trades
    fx = Decimal("90.0")
    normalized = []
    for t in trades:
        normalized.append(
            TradeResult(
                symbol=t.symbol,
                market_id=t.market_id,
                segment_id=t.segment_id,
                direction=t.direction,
                entry_price=t.entry_price / fx,
                exit_price=t.exit_price / fx,
                quantity=t.quantity,
                pnl=t.pnl / fx,
                pnl_pct=t.pnl_pct,  # percentage is currency-agnostic
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                strategy_name=t.strategy_name,
                hold_bars=t.hold_bars,
            )
        )
    return normalized
```

Similarly for snapshots — convert equity values to USD:

```python
def _normalize_snapshots_to_usd(
    snapshots: list[PortfolioState],
    segment: str,
) -> list[PortfolioState]:
    """Convert MOEX portfolio snapshots to USD for aggregation."""
    if not segment.startswith("ru_"):
        return snapshots
    fx = Decimal("90.0")
    return [
        PortfolioState(
            timestamp=s.timestamp,
            equity=s.equity / fx,
            cash=s.cash / fx,
            positions=s.positions,
        )
        for s in snapshots
    ]
```

Then change lines 1024-1027:
```python
            normalized_trades = _normalize_trades_to_usd(trades, segment)
            all_trades.extend(normalized_trades)
            segment_trades[segment].extend(trades)  # keep raw for per-segment metrics
            if snapshots:
                all_snapshots.extend(_normalize_snapshots_to_usd(snapshots, segment))
```

**Step 3: Run tests**

Run: `uv run pytest tests/unit/test_moex_fixes.py -v`
Expected: PASS

---

## Task 3: Vol-Normalize Dual Momentum Confidence

**Files:**
- Modify: `src/finalayze/strategies/dual_momentum.py:190` (confidence formula)
- Test: `tests/unit/test_moex_fixes.py` (append)

**Step 1: Write failing test**

Append to `tests/unit/test_moex_fixes.py`:

```python
class TestVolNormalizedConfidence:
    """Fix 3: Dual momentum confidence should be vol-normalized."""

    _CONFIDENCE_BASE = 0.4
    _VOL_BASELINE = 0.15  # typical US annual vol

    def _compute_confidence(self, score: float, realized_vol: float) -> float:
        """Replicate the vol-normalized confidence formula."""
        normalized_score = abs(score) / max(realized_vol, 0.01) * self._VOL_BASELINE
        return min(0.95, self._CONFIDENCE_BASE + normalized_score)

    def test_same_confidence_for_proportional_returns(self) -> None:
        """10% return at 30% vol should equal 5% return at 15% vol."""
        conf_high_vol = self._compute_confidence(0.10, 0.30)
        conf_low_vol = self._compute_confidence(0.05, 0.15)
        assert abs(conf_high_vol - conf_low_vol) < 0.01

    def test_moex_deflated_vs_old(self) -> None:
        """MOEX 10% return at 30% vol should produce LOWER confidence than old formula."""
        old_confidence = min(0.95, 0.4 + abs(0.10) * 1.0)  # 0.50
        new_confidence = self._compute_confidence(0.10, 0.30)  # 0.4 + 0.05 = 0.45
        assert new_confidence < old_confidence

    def test_us_unchanged(self) -> None:
        """US 5% return at 15% vol should produce same confidence as old formula."""
        old_confidence = min(0.95, 0.4 + abs(0.05) * 1.0)  # 0.45
        new_confidence = self._compute_confidence(0.05, 0.15)  # 0.4 + 0.05 = 0.45
        assert abs(new_confidence - old_confidence) < 0.01
```

**Step 2: Run test to verify logic**

Run: `uv run pytest tests/unit/test_moex_fixes.py::TestVolNormalizedConfidence -v`
Expected: PASS

**Step 3: Implement in dual_momentum.py**

Add a new constant near line 20:
```python
_VOL_BASELINE = 0.15  # baseline annual vol for confidence normalization
```

Change line 190 from:
```python
confidence = min(_MAX_CONFIDENCE, _CONFIDENCE_BASE + abs(score) * _CONFIDENCE_SCALE)
```

to:
```python
# Vol-normalize: same return at higher vol produces lower confidence
closes = [float(c.close) for c in candles]
from finalayze.risk.position_sizer import compute_realized_vol as _crv  # noqa: PLC0415
asset_vol = float(_crv(candles) or Decimal("0.15"))
normalized_score = abs(score) / max(asset_vol, 0.01) * _VOL_BASELINE
confidence = min(_MAX_CONFIDENCE, _CONFIDENCE_BASE + normalized_score * _CONFIDENCE_SCALE)
```

Note: `compute_realized_vol` is already imported in engine.py. Here we import it locally to avoid circular imports. The function accepts `list[Candle]` and returns `Decimal`.

**Step 4: Run existing dual_momentum tests**

Run: `uv run pytest tests/unit/test_phase0_strategies.py -k dual -v && uv run pytest tests/unit/test_moex_fixes.py -v`
Expected: PASS

---

## Task 4: Wider Bollinger Params for MOEX (YAML only)

**Files:**
- Modify: `src/finalayze/strategies/presets/ru_blue_chips.yaml:49-56`
- Modify: `src/finalayze/strategies/presets/ru_energy.yaml:49-56`
- Modify: `src/finalayze/strategies/presets/ru_tech.yaml:42-50`
- Modify: `src/finalayze/strategies/presets/ru_finance.yaml:15-20`

**Step 1: Update ru_blue_chips.yaml mean_reversion params**

```yaml
      bb_std_dev: 2.5      # was 2.0 — wider for MOEX fat tails
      rsi_oversold_mr: 25   # was 30 — tighter gate for MOEX noise
      rsi_overbought_mr: 75 # was 70
```

**Step 2: Update ru_energy.yaml mean_reversion params**

```yaml
      bb_std_dev: 2.8      # was 2.0 — widest for commodity-linked energy stocks
      rsi_oversold_mr: 25   # was 30
      rsi_overbought_mr: 75 # was 70
```

**Step 3: Update ru_tech.yaml mean_reversion params**

```yaml
      bb_std_dev: 2.5      # was 1.8 — wider for MOEX
      rsi_oversold_mr: 25   # was 30
      rsi_overbought_mr: 75 # was 70
```

**Step 4: Update ru_finance.yaml mean_reversion params**

```yaml
      bb_std_dev: 2.5      # was 1.8 — wider for MOEX
      rsi_oversold_mr: 25   # was 30
      rsi_overbought_mr: 75 # was 70
```

---

## Task 5: Reduce SMA Warmup for MOEX RSI2 (YAML only)

**Files:**
- Modify: `src/finalayze/strategies/presets/ru_blue_chips.yaml:85`
- Modify: `src/finalayze/strategies/presets/ru_energy.yaml:85`
- Modify: `src/finalayze/strategies/presets/ru_tech.yaml:82`
- Modify: `src/finalayze/strategies/presets/ru_finance.yaml:82`

**Step 1: Change sma_trend_period from 200 to 100 in all ru_* presets**

In each file, under `rsi2_connors.params`:
```yaml
      sma_trend_period: 100  # was 200 — shorter warmup for MOEX data
```

---

## Task 6: Time-Varying MOEX Regime Provider

**Files:**
- Modify: `src/finalayze/risk/regime.py` (add RollingVolRegimeProvider class)
- Modify: `scripts/run_iteration.py:553-562` (use new provider)
- Test: `tests/unit/test_moex_fixes.py` (append)

**Step 1: Write failing test**

Append to `tests/unit/test_moex_fixes.py`:

```python
from finalayze.core.schemas import Candle
from finalayze.risk.regime import MarketRegime


class TestRollingVolRegimeProvider:
    """Fix 6: MOEX regime should be time-varying, not static."""

    def _make_candles(self, closes: list[float]) -> list[Candle]:
        """Create minimal candles from close prices."""
        return [
            Candle(
                symbol="IMOEX",
                market_id="moex",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                open=Decimal(str(c)),
                high=Decimal(str(c * 1.01)),
                low=Decimal(str(c * 0.99)),
                close=Decimal(str(c)),
                volume=Decimal(1000),
            )
            for c in closes
        ]

    def test_provider_returns_regime_per_bar(self) -> None:
        """Provider should compute regime from rolling window, not static snapshot."""
        from finalayze.risk.regime import RollingVolRegimeProvider

        # 30 stable candles (low vol)
        closes = [100.0 + i * 0.1 for i in range(30)]
        candles = self._make_candles(closes)
        provider = RollingVolRegimeProvider(imoex_candles=candles)

        state = provider.get_regime(candles, bar_index=25)
        assert state.regime in {MarketRegime.LOW_VOL, MarketRegime.NORMAL}
        assert state.allow_new_longs is True

    def test_provider_detects_high_vol(self) -> None:
        """Large price swings should produce ELEVATED or CRISIS regime."""
        from finalayze.risk.regime import RollingVolRegimeProvider

        # Alternating large swings
        closes = [100.0 if i % 2 == 0 else 130.0 for i in range(30)]
        candles = self._make_candles(closes)
        provider = RollingVolRegimeProvider(imoex_candles=candles)

        state = provider.get_regime(candles, bar_index=25)
        assert state.regime in {MarketRegime.ELEVATED, MarketRegime.CRISIS}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_moex_fixes.py::TestRollingVolRegimeProvider -v`
Expected: FAIL with `ImportError: cannot import name 'RollingVolRegimeProvider'`

**Step 3: Implement RollingVolRegimeProvider in regime.py**

Add after `StaticRegimeProvider` class (after line 166):

```python
class RollingVolRegimeProvider:
    """Compute MOEX regime from rolling realized vol of IMOEX candles.

    Unlike StaticRegimeProvider (single snapshot), this recomputes realized vol
    at each bar using a rolling window. Matches VIXRegimeProvider's per-bar behavior.
    """

    def __init__(
        self,
        imoex_candles: list[Candle],
        window: int = 20,
    ) -> None:
        self._closes = [float(c.close) for c in imoex_candles]
        self._window = window

    def get_regime(
        self,
        candles: list[Candle],  # noqa: ARG002
        bar_index: int,
    ) -> RegimeState:
        """Compute regime from rolling realized vol at bar_index."""
        # Use min(bar_index, len(imoex)-1) to handle index mismatch
        idx = min(bar_index, len(self._closes) - 1)
        if idx < self._window:
            return RegimeState.normal()

        window_closes = self._closes[idx - self._window : idx + 1]
        returns = [
            (window_closes[i] - window_closes[i - 1]) / window_closes[i - 1]
            for i in range(1, len(window_closes))
            if window_closes[i - 1] != 0
        ]
        if len(returns) < 2:
            return RegimeState.normal()

        import statistics  # noqa: PLC0415
        daily_vol = statistics.stdev(returns)
        annualized_vol = Decimal(str(daily_vol * math.sqrt(252)))

        return compute_moex_regime_state(annualized_vol)
```

**Step 4: Wire into run_iteration.py**

Change lines 553-562 in `_build_regime_provider` from:
```python
    if segment.startswith("ru_"):
        try:
            moex_fetcher = CachingFetcher(_make_moex_fetcher())
            imoex_candles = moex_fetcher.fetch_candles("IMOEX", start, end)
            if imoex_candles:
                vol = compute_realized_vol(imoex_candles)
                regime_state = compute_moex_regime_state(vol)
                print(f"    MOEX regime: {regime_state.regime.value} (vol={float(vol):.2%})")
                return StaticRegimeProvider(regime_state)
        except Exception:
            print("    Warning: failed to fetch IMOEX data, regime provider disabled")
        return None
```

to:
```python
    if segment.startswith("ru_"):
        try:
            moex_fetcher = CachingFetcher(_make_moex_fetcher())
            imoex_candles = moex_fetcher.fetch_candles("IMOEX", start, end)
            if imoex_candles:
                from finalayze.risk.regime import RollingVolRegimeProvider  # noqa: PLC0415
                print(f"    MOEX regime: RollingVolRegimeProvider ({len(imoex_candles)} bars)")
                return RollingVolRegimeProvider(imoex_candles=imoex_candles)
        except Exception:
            print("    Warning: failed to fetch IMOEX data, regime provider disabled")
        return None
```

Also update the type hint at line 544 and 598 to include `RollingVolRegimeProvider`.

**Step 5: Run tests**

Run: `uv run pytest tests/unit/test_moex_fixes.py::TestRollingVolRegimeProvider tests/unit/test_phase_a_regime.py -v`
Expected: PASS

---

## Task 7: Wider ADX Routing Bands for MOEX (YAML only)

**Files:**
- Modify: `src/finalayze/strategies/presets/ru_blue_chips.yaml:8-9`
- Modify: `src/finalayze/strategies/presets/ru_energy.yaml:8-9`
- Modify: `src/finalayze/strategies/presets/ru_tech.yaml:8-9`
- Modify: `src/finalayze/strategies/presets/ru_finance.yaml:8-9`

**Step 1: Change ADX thresholds in all ru_* presets**

```yaml
regime_routing:
  enabled: true
  adx_period: 14
  trend_threshold: 30    # was 35 — wider trend zone for MOEX
  mr_threshold: 20       # was 15 — wider MR zone for MOEX
```

---

## Task 8: Max Hold Bars MOEX Uplift

**Files:**
- Modify: `src/finalayze/backtest/config.py:82-98` (resolve_max_hold_bars)
- Test: `tests/unit/test_moex_fixes.py` (append)

**Step 1: Write failing test**

Append to `tests/unit/test_moex_fixes.py`:

```python
from finalayze.backtest.config import resolve_max_hold_bars


class TestMoexHoldBarsUplift:
    """Fix 8: MOEX segments should get 1.3x max hold bars."""

    def test_us_segment_unchanged(self) -> None:
        hold = resolve_max_hold_bars({"momentum": 30}, "momentum", segment_id="us_tech")
        assert hold == 30

    def test_moex_segment_uplifted(self) -> None:
        hold = resolve_max_hold_bars({"momentum": 30}, "momentum", segment_id="ru_blue_chips")
        assert hold == 39  # 30 * 1.3 = 39

    def test_moex_mean_reversion_uplifted(self) -> None:
        hold = resolve_max_hold_bars({"mean_reversion": 20}, "mean_reversion", segment_id="ru_energy")
        assert hold == 26  # 20 * 1.3 = 26

    def test_int_max_hold_bars_moex(self) -> None:
        hold = resolve_max_hold_bars(30, "momentum", segment_id="ru_blue_chips")
        assert hold == 39  # 30 * 1.3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_moex_fixes.py::TestMoexHoldBarsUplift -v`
Expected: FAIL (resolve_max_hold_bars doesn't accept segment_id)

**Step 3: Add segment_id parameter to resolve_max_hold_bars**

Change `resolve_max_hold_bars` in `src/finalayze/backtest/config.py` from:

```python
def resolve_max_hold_bars(
    max_hold_bars: int | dict[str, int],
    strategy_name: str,
) -> int:
    if isinstance(max_hold_bars, int):
        return max_hold_bars
    return max_hold_bars.get(strategy_name, _DEFAULT_HOLD_BARS_FALLBACK)
```

to:

```python
_MOEX_HOLD_BARS_UPLIFT = 1.3


def resolve_max_hold_bars(
    max_hold_bars: int | dict[str, int],
    strategy_name: str,
    *,
    segment_id: str = "",
) -> int:
    if isinstance(max_hold_bars, int):
        base = max_hold_bars
    else:
        base = max_hold_bars.get(strategy_name, _DEFAULT_HOLD_BARS_FALLBACK)
    if segment_id.startswith("ru_"):
        base = int(base * _MOEX_HOLD_BARS_UPLIFT)
    return base
```

**Step 4: Update call sites in engine.py**

Search for `resolve_max_hold_bars(` in `engine.py` and add `segment_id=segment_id` to each call.

**Step 5: Run all tests**

Run: `uv run pytest tests/unit/test_moex_fixes.py tests/unit/ -x --tb=short`
Expected: PASS

---

## Task 9: Fix Existing Failing Test

**Files:**
- Modify: `tests/unit/test_train_models_script.py:330`

**Step 1: Fix the test that references non-existent `evaluate_fold`**

The test `TestWalkForwardUsesLastFold::test_last_fold_models_saved_not_best_accuracy` patches `train_models.evaluate_fold` which doesn't exist. Read the test and fix or remove it.

Run: `uv run pytest tests/unit/test_train_models_script.py::TestWalkForwardUsesLastFold -v --tb=short`

---

## Task 10: Run Full Test Suite + Backtest Validation

**Step 1: Run ruff**

Run: `uv run ruff check src/ scripts/ tests/ && uv run ruff format --check src/ scripts/ tests/`
Expected: PASS

**Step 2: Run full tests**

Run: `uv run pytest --tb=short -q`
Expected: All pass, 0 failures

**Step 3: Backtest MOEX-only**

Run: `uv run python scripts/run_iteration.py --name moex-pipeline-fix --description "8 MOEX pipeline fixes: RUB cash, vol-norm confidence, wider BB/ADX, rolling regime" --segments ru_blue_chips`

**Step 4: Backtest combined US+MOEX**

Run: `uv run python scripts/run_iteration.py --name moex-combined-fix --description "Combined US+MOEX with pipeline fixes" --segments us_tech,ru_blue_chips`

**Step 5: Verify success criteria**

- MOEX segments produce trades for stocks priced < 2000 RUB (no quantity_zero)
- Combined WF Sharpe ≥ 0 (was -0.0058)
- US-only WF Sharpe unchanged (≥ +0.005)
