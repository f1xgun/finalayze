"""Integration test for IterationTracker end-to-end flow.

Tests: synthetic candles -> tracker -> save -> load -> compare.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from finalayze.backtest.iteration_tracker import IterationTracker
from finalayze.backtest.monte_carlo import BootstrapCI, BootstrapResult
from finalayze.backtest.walk_forward import WalkForwardResult
from finalayze.core.schemas import (
    GateResult,
    IterationMetadata,
    IterationMetrics,
    PortfolioState,
    TradeResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshots(equities: list[float]) -> list[PortfolioState]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return [
        PortfolioState(
            cash=Decimal(str(e)),
            positions={},
            equity=Decimal(str(e)),
            timestamp=start + timedelta(days=i),
        )
        for i, e in enumerate(equities)
    ]


def _make_trade(pnl: float, symbol: str = "AAPL") -> TradeResult:
    return TradeResult(
        signal_id=uuid4(),
        symbol=symbol,
        side="BUY",
        quantity=Decimal(10),
        entry_price=Decimal(100),
        exit_price=Decimal(str(100 + pnl / 10)),
        pnl=Decimal(str(pnl)),
        pnl_pct=Decimal(str(pnl / 1000)),
    )


def _make_bootstrap(sharpe_5th: float = 0.3) -> BootstrapResult:
    ci = BootstrapCI(point_estimate=0.5, lower=sharpe_5th, upper=1.0)
    zero_ci = BootstrapCI(point_estimate=0.0, lower=0.0, upper=0.0)
    return BootstrapResult(
        total_return=zero_ci,
        sharpe_ratio=ci,
        max_drawdown=zero_ci,
        win_rate=zero_ci,
        profit_factor=zero_ci,
        n_simulations=100,
        n_trades=50,
    )


def _make_wf_result(
    sharpe: float = 0.5,
    max_dd: float = 5.0,
    snapshots: list[PortfolioState] | None = None,
) -> WalkForwardResult:
    return WalkForwardResult(
        oos_sharpe=sharpe,
        oos_max_drawdown_pct=max_dd,
        per_fold_sharpes=[sharpe],
        per_fold_trade_counts=[100],
        oos_snapshots=snapshots or [],
    )


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


class TestIterationTrackerIntegration:
    """End-to-end: compute metrics -> save -> load -> compare."""

    def test_full_iteration_flow(self, tmp_path: Path) -> None:
        """Run two sequential iterations, save both, compare."""
        tracker = IterationTracker(results_root=tmp_path)

        # --- Iteration 1: baseline ---
        trades_1 = [_make_trade(100), _make_trade(-30), _make_trade(80)]
        snapshots_1 = _make_snapshots([100000, 100100, 100070, 100150])
        wf_1 = _make_wf_result(sharpe=0.8, max_dd=5.0, snapshots=snapshots_1)
        mc_1 = _make_bootstrap(sharpe_5th=0.3)

        metrics_1 = tracker.compute_metrics(
            wf_result=wf_1,
            trades=trades_1,
            snapshots=snapshots_1,
            segment_trades={"seg_a": trades_1},
            mc_result=mc_1,
        )

        gates_1, verdict_1 = tracker.evaluate_gates(metrics_1, baseline=None)

        meta_1 = IterationMetadata(
            name="baseline-v1",
            description="Initial baseline iteration",
            created_at=datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
            git_describe="v0.1-10-g1234567",
            git_sha="1234567890abcdef",
            git_dirty=False,
            config_hash=tracker.compute_config_hash(
                {"initial_cash": 100000},
                {"seg_a": {"momentum": {"weight": 1.0}}},
            ),
            strategy_configs={"seg_a": {"momentum": {"weight": 1.0}}},
            backtest_config={"initial_cash": 100000},
            metrics=metrics_1,
            gate_results=gates_1,
            verdict=verdict_1,
        )

        path_1 = tracker.save(meta_1)
        assert (path_1 / "metadata.json").exists()

        # --- Iteration 2: improved ---
        trades_2 = [_make_trade(150), _make_trade(-20), _make_trade(120)]
        snapshots_2 = _make_snapshots([100000, 100150, 100130, 100250])
        wf_2 = _make_wf_result(sharpe=1.1, max_dd=4.0, snapshots=snapshots_2)
        mc_2 = _make_bootstrap(sharpe_5th=0.5)

        metrics_2 = tracker.compute_metrics(
            wf_result=wf_2,
            trades=trades_2,
            snapshots=snapshots_2,
            segment_trades={"seg_a": trades_2},
            mc_result=mc_2,
        )

        gates_2, verdict_2 = tracker.evaluate_gates(metrics_2, baseline=metrics_1)

        meta_2 = IterationMetadata(
            name="improved-v2",
            description="Added sentiment signal",
            created_at=datetime(2026, 3, 2, 10, 0, 0, tzinfo=UTC),
            git_describe="v0.1-12-gabcdef0",
            git_sha="abcdef0123456789",
            git_dirty=False,
            config_hash=tracker.compute_config_hash(
                {"initial_cash": 100000},
                {"seg_a": {"momentum": {"weight": 1.0}, "sentiment": {"weight": 0.5}}},
            ),
            strategy_configs={"seg_a": {"momentum": {"weight": 1.0}}},
            backtest_config={"initial_cash": 100000},
            metrics=metrics_2,
            gate_results=gates_2,
            verdict=verdict_2,
        )

        path_2 = tracker.save(meta_2)
        assert (path_2 / "metadata.json").exists()

        # --- Verify history.jsonl ---
        history_path = tmp_path / "history.jsonl"
        assert history_path.exists()
        lines = history_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["name"] == "baseline-v1"
        assert json.loads(lines[1])["name"] == "improved-v2"

        # --- Load and verify ---
        loaded_1 = tracker.load("baseline-v1")
        assert loaded_1.name == "baseline-v1"
        assert loaded_1.metrics.wf_sharpe == metrics_1.wf_sharpe

        loaded_2 = tracker.load("improved-v2")
        assert loaded_2.name == "improved-v2"

        # --- Load latest ---
        latest = tracker.load_latest()
        assert latest is not None
        assert latest.name == "improved-v2"

        # --- Compare ---
        comparison = tracker.compare("improved-v2", "baseline-v1")
        assert comparison.current == "improved-v2"
        assert comparison.baseline == "baseline-v1"
        sharpe_delta = comparison.metric_deltas["wf_sharpe"]
        assert sharpe_delta > 0  # v2 has higher Sharpe

        # --- List iterations ---
        iterations = tracker.list_iterations()
        assert len(iterations) == 2

    def test_history_jsonl_append_is_sequential(self, tmp_path: Path) -> None:
        """Two sequential saves produce two lines in history.jsonl."""
        tracker = IterationTracker(results_root=tmp_path)

        for i in range(3):
            metrics = IterationMetrics(
                wf_sharpe=0.5 + i * 0.1,
                wf_max_drawdown=5.0,
                profit_factor=1.5,
                calmar_ratio=1.0,
                trade_count=100,
                avg_hold_bars=5.0,
                segment_pnl_share={"seg": 1.0},
                sortino_ratio=1.0,
                win_rate_by_segment={"seg": 0.5},
                information_ratio=None,
                mc_5th_pct_sharpe=0.2,
                model_disagreement=0.05,
                turnover_adjusted_return=10.0,
                gross_sharpe=0.6,
                net_sharpe=0.5,
                param_stability_cv=0.1,
                per_model_proba_mean={},
            )
            gate = GateResult(
                name="S1",
                gate_type="safety",
                passed=True,
                value=0.5,
                threshold=0.0,
                message="ok",
            )
            meta = IterationMetadata(
                name=f"iter-{i}",
                description=f"Iteration {i}",
                created_at=datetime(2026, 3, 1 + i, tzinfo=UTC),
                git_describe="v0.1",
                git_sha="abc",
                git_dirty=False,
                config_hash="h",
                strategy_configs={},
                backtest_config={},
                metrics=metrics,
                gate_results=[gate],
                verdict="PASS",
            )
            tracker.save(meta)

        history_path = tmp_path / "history.jsonl"
        lines = history_path.read_text().strip().split("\n")
        assert len(lines) == 3
        names = [json.loads(line)["name"] for line in lines]
        assert names == ["iter-0", "iter-1", "iter-2"]

    def test_atomic_write_no_partial_file(self, tmp_path: Path) -> None:
        """Verify atomic write pattern: no partial metadata.json on failure."""
        from unittest.mock import patch

        tracker = IterationTracker(results_root=tmp_path)
        metrics = IterationMetrics(
            wf_sharpe=0.5,
            wf_max_drawdown=5.0,
            profit_factor=1.5,
            calmar_ratio=1.0,
            trade_count=100,
            avg_hold_bars=5.0,
            segment_pnl_share={},
            sortino_ratio=1.0,
            win_rate_by_segment={},
            information_ratio=None,
            mc_5th_pct_sharpe=0.2,
            model_disagreement=0.0,
            turnover_adjusted_return=10.0,
            gross_sharpe=0.5,
            net_sharpe=0.5,
            param_stability_cv=0.1,
            per_model_proba_mean={},
        )
        gate = GateResult(
            name="S1",
            gate_type="safety",
            passed=True,
            value=0.5,
            threshold=0.0,
            message="ok",
        )
        meta = IterationMetadata(
            name="atomic-fail",
            description="Test atomic write failure",
            created_at=datetime(2026, 3, 2, tzinfo=UTC),
            git_describe="v0.1",
            git_sha="abc",
            git_dirty=False,
            config_hash="h",
            strategy_configs={},
            backtest_config={},
            metrics=metrics,
            gate_results=[gate],
            verdict="PASS",
        )

        with (
            patch(
                "finalayze.backtest.iteration_tracker.os.rename",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(OSError, match="disk full"),
        ):
            tracker.save(meta)

        # No metadata.json should exist
        assert not (tmp_path / "atomic-fail" / "metadata.json").exists()
