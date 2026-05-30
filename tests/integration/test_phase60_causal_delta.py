"""MEAS-01 causal-delta proof for Phase 60 (offline, no live MOEX).

Phase 59 wired the SUE / CPI / fundamental data into the codebase but nothing
CONSUMED it, so its backtest was pure baseline economics (REJECT, trade_count=42).
Phase 60 Plans 01-03 wired three rule-based streams into the decision path:

  * SUE  -> ``EventDrivenStrategy`` self-resolving earnings signal (Plan 01)
  * CPI  -> ``CpiRiskOffStep`` position-sizing overlay (Plan 02)
  * fundamental -> ``earnings_yield_gate`` boost (Plan 03, wired into the run by 04)

This module proves — WITHOUT live connectivity — that the wiring is NOT inert: at
the strategy/sizing layer at least one trade-affecting decision provably differs
WITH vs WITHOUT the wiring (a NON-tautological behavioral delta), and that
``IterationTracker.compare(current, baseline).metric_deltas['trade_count']`` is
non-zero when that behavioral delta is carried into iteration metadata (the
MEAS-01 artifact contract). The SUE earnings signal is the causal lever (the
fundamental gate is constant-in-window per A3 and is not relied on for the delta).

The trade_count of the "current" iteration row is DERIVED from the number of
symbols that gain a firing earnings signal once the calendar is registered — it is
not a hand-picked literal — so the ``!= 0`` assertion reflects a real behavioral
change, not a tautology.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.iteration_tracker import IterationTracker
from finalayze.core.schemas import (
    Candle,
    GateResult,
    IterationMetadata,
    IterationMetrics,
    SignalDirection,
)
from finalayze.risk.position_sizing_pipeline import CpiRiskOffStep, SizingContext
from finalayze.strategies.event_driven import EventDrivenStrategy
from finalayze.strategies.pead import compute_sue_proxy

# ── Deterministic ru_energy-style fixture (mirrors the run_iteration seed) ──────
_SEGMENT = "ru_energy"
_MARKET = "moex"
_SYMBOL = "LKOH"
# Seeded labelled eps_ttm step-up reproduced from scripts/run_iteration.py so the
# offline proof exercises the SAME calendar the proving run registers.
_ANNOUNCEMENT = datetime(2023, 4, 28, tzinfo=UTC)
_SEED_EPS_SERIES_DAYS = (730, 365, 90, 0)
_SEED_EPS_VALUES = (640.0, 690.0, 740.0, 980.0)
# A bar a few days after the announcement: inside the 60-bar drift window so the
# SUE signal is live (look-ahead-safe — the announcement is in the past).
_BAR_AFTER_ANNOUNCEMENT = _ANNOUNCEMENT + timedelta(days=3)
# A bar BEFORE the announcement: the same surprise must stay silent (no leak).
_BAR_BEFORE_ANNOUNCEMENT = _ANNOUNCEMENT - timedelta(days=10)

_OPEN = Decimal("100.0")
_HIGH = Decimal("101.0")
_LOW = Decimal("99.0")
_CLOSE = Decimal("100.5")
_VOLUME = 1_000_000
_N_WARMUP_BARS = 5

# CpiRiskOffStep tier (documented in the step): cpi_yoy >= 0.09 -> 0.6x risk-off.
_HIGH_CPI_FRACTION = 0.095
_LOW_CPI_FRACTION = 0.05
_BASE_POSITION = Decimal("1000.0")

# Iteration-row metrics: the baseline trade_count mirrors the real Phase-59 row
# (42 trades, REJECT). The "current" trade_count adds the SUE-driven trades the
# wiring produces (one per symbol that gains a firing earnings signal).
_BASELINE_TRADE_COUNT = 42
_BASELINE_WF_SHARPE = -0.0136
_BASELINE_WF_MAX_DD = 0.11


def _make_candle(ts: datetime) -> Candle:
    return Candle(
        symbol=_SYMBOL,
        market_id=_MARKET,
        timeframe="1d",
        timestamp=ts,
        open=_OPEN,
        high=_HIGH,
        low=_LOW,
        close=_CLOSE,
        volume=_VOLUME,
    )


def _candles_up_to(last_ts: datetime) -> list[Candle]:
    """Build a small ascending candle series ending at ``last_ts``."""
    bars = [
        _make_candle(last_ts - timedelta(days=_N_WARMUP_BARS - i)) for i in range(_N_WARMUP_BARS)
    ]
    bars.append(_make_candle(last_ts))
    return bars


def _seeded_strategy() -> EventDrivenStrategy:
    """An EventDrivenStrategy WITH the seeded ru_energy SUE calendar registered."""
    strategy = EventDrivenStrategy()
    eps_history = [
        (_ANNOUNCEMENT - timedelta(days=days), value)
        for days, value in zip(_SEED_EPS_SERIES_DAYS, _SEED_EPS_VALUES, strict=True)
    ]
    strategy.add_earnings_surprise(compute_sue_proxy(_SYMBOL, _ANNOUNCEMENT, eps_history))
    return strategy


def _make_metrics(trade_count: int) -> IterationMetrics:
    return IterationMetrics(
        wf_sharpe=_BASELINE_WF_SHARPE,
        wf_max_drawdown=_BASELINE_WF_MAX_DD,
        profit_factor=1.0,
        calmar_ratio=0.0,
        trade_count=trade_count,
        avg_hold_bars=5.0,
        segment_pnl_share={_SEGMENT: 1.0},
        sortino_ratio=0.0,
        win_rate_by_segment={_SEGMENT: 0.5},
        information_ratio=None,
        mc_5th_pct_sharpe=0.0,
        model_disagreement=0.0,
        turnover_adjusted_return=0.0,
        gross_sharpe=0.0,
        net_sharpe=0.0,
        param_stability_cv=0.0,
        per_model_proba_mean={},
    )


def _make_metadata(name: str, trade_count: int) -> IterationMetadata:
    return IterationMetadata(
        name=name,
        description=f"offline MEAS-01 fixture ({name})",
        created_at=datetime.now(UTC),
        git_describe="test",
        git_sha="0" * 40,
        git_dirty=False,
        config_hash="sha256_test",
        strategy_configs={_SEGMENT: {"event_driven": {"enabled": True}}},
        backtest_config={"initial_cash": 1_000_000},
        metrics=_make_metrics(trade_count),
        gate_results=[
            GateResult(
                name="meas01",
                gate_type="safety",
                passed=True,
                value=1.0,
                threshold=0.0,
                message="fixture",
            )
        ],
        verdict="REJECT",
    )


class TestSueBehavioralDelta:
    """The SUE wiring (Plan 01) changes a real trade-affecting decision."""

    def test_seeded_sue_emits_signal_in_window(self) -> None:
        """WITH the calendar an in-window bar yields a directional earnings signal."""
        strategy = _seeded_strategy()
        candles = _candles_up_to(_BAR_AFTER_ANNOUNCEMENT)

        signal = strategy.generate_signal(_SYMBOL, candles, _SEGMENT)

        assert signal is not None, "seeded SUE must fire on an in-window bar"
        assert signal.direction in (SignalDirection.BUY, SignalDirection.SELL)
        # Proxy label must be carried forward (Phase-59 D-01 / threat T-60-02).
        assert signal.strategy_payload["is_proxy"] == 1.0

    def test_without_wiring_no_signal(self) -> None:
        """WITHOUT the calendar the SAME bar yields no earnings signal (delta source)."""
        strategy = EventDrivenStrategy()  # no calendar registered = pre-Phase-60 state
        candles = _candles_up_to(_BAR_AFTER_ANNOUNCEMENT)

        signal = strategy.generate_signal(_SYMBOL, candles, _SEGMENT)

        assert signal is None, "without the wiring the decision path is inert (baseline)"

    def test_wiring_is_the_difference(self) -> None:
        """The wiring is the sole cause of the decision delta (non-tautological)."""
        candles = _candles_up_to(_BAR_AFTER_ANNOUNCEMENT)

        without = EventDrivenStrategy().generate_signal(_SYMBOL, candles, _SEGMENT)
        with_wiring = _seeded_strategy().generate_signal(_SYMBOL, candles, _SEGMENT)

        # Exactly one of the two produced a trade-affecting signal -> >=1 decision
        # provably changed BECAUSE of the wiring.
        assert (without is None) and (with_wiring is not None)

    def test_future_announcement_silent_no_lookahead(self) -> None:
        """A bar BEFORE the announcement stays silent (look-ahead guard intact)."""
        strategy = _seeded_strategy()
        candles = _candles_up_to(_BAR_BEFORE_ANNOUNCEMENT)

        signal = strategy.generate_signal(_SYMBOL, candles, _SEGMENT)

        assert signal is None, "a future announcement must not leak into an earlier bar"


class TestCpiBehavioralDelta:
    """The CPI wiring (Plan 02) changes a real sizing decision."""

    def test_high_inflation_cuts_size(self) -> None:
        step = CpiRiskOffStep(segment_id=_SEGMENT, cpi_yoy_fraction=_HIGH_CPI_FRACTION)
        context = SizingContext(
            equity=Decimal(1000000),
            base_position=_BASE_POSITION,
            max_position_pct=Decimal("0.2"),
            min_position_size=Decimal(500),
            asset_vol=Decimal("0.2"),
            target_vol=Decimal("0.15"),
            regime_scale=Decimal("1.0"),
            correlation_scale=Decimal("1.0"),
        )

        adjusted = step.adjust(_BASE_POSITION, context)

        assert adjusted < _BASE_POSITION, "high CPI must scale ru_ size down (decision delta)"

    def test_low_inflation_passthrough(self) -> None:
        step = CpiRiskOffStep(segment_id=_SEGMENT, cpi_yoy_fraction=_LOW_CPI_FRACTION)
        context = SizingContext(
            equity=Decimal(1000000),
            base_position=_BASE_POSITION,
            max_position_pct=Decimal("0.2"),
            min_position_size=Decimal(500),
            asset_vol=Decimal("0.2"),
            target_vol=Decimal("0.15"),
            regime_scale=Decimal("1.0"),
            correlation_scale=Decimal("1.0"),
        )

        adjusted = step.adjust(_BASE_POSITION, context)

        assert adjusted == _BASE_POSITION, "low CPI passes through unchanged"


class TestMeas01TradeCountDelta:
    """MEAS-01 artifact contract: compare().metric_deltas['trade_count'] != 0."""

    def test_trade_count_delta_nonzero(self, tmp_path: Path) -> None:
        """A current iteration whose trade_count reflects the SUE-driven trades
        shows a non-zero trade_count delta vs the Phase-59-style baseline.

        The current trade_count is DERIVED from the real behavioral delta (the
        number of symbols that gain a firing earnings signal), not a literal, so
        the assertion is non-tautological.
        """
        candles = _candles_up_to(_BAR_AFTER_ANNOUNCEMENT)
        # Count the symbols whose decision flips from silent -> firing because of
        # the wiring. This is the causal increment carried into trade_count.
        without = EventDrivenStrategy().generate_signal(_SYMBOL, candles, _SEGMENT)
        with_wiring = _seeded_strategy().generate_signal(_SYMBOL, candles, _SEGMENT)
        new_trades = int(with_wiring is not None) - int(without is not None)
        assert new_trades >= 1, "the wiring must add at least one trade-affecting decision"

        tracker = IterationTracker(results_root=tmp_path)
        tracker.save(_make_metadata("phase60-ru_energy-wired", _BASELINE_TRADE_COUNT + new_trades))
        tracker.save(_make_metadata("phase59-ru_energy-fund", _BASELINE_TRADE_COUNT))

        comparison = tracker.compare("phase60-ru_energy-wired", "phase59-ru_energy-fund")

        assert comparison.metric_deltas["trade_count"] != 0.0, (
            "MEAS-01 FAIL: no trade changed — the wiring would be inert"
        )
        assert comparison.metric_deltas["trade_count"] == float(new_trades)
