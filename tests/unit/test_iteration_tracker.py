"""Tests for IterationTracker class.

Task 3.2: snapshot_context, compute_metrics, evaluate_gates, save/load/compare.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
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


def _make_snapshots(
    equities: list[float],
    start: datetime | None = None,
) -> list[PortfolioState]:
    """Create PortfolioState list from equity values."""
    if start is None:
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


def _make_trade(pnl: float, pnl_pct: float, symbol: str = "AAPL") -> TradeResult:
    return TradeResult(
        signal_id=uuid4(),
        symbol=symbol,
        side="BUY",
        quantity=Decimal(10),
        entry_price=Decimal(100),
        exit_price=Decimal(str(100 + pnl / 10)),
        pnl=Decimal(str(pnl)),
        pnl_pct=Decimal(str(pnl_pct)),
    )


def _make_wf_result(
    sharpe: float = 0.5,
    max_dd: float = 5.0,
    per_fold_sharpes: list[float] | None = None,
    per_fold_trade_counts: list[int] | None = None,
    snapshots: list[PortfolioState] | None = None,
) -> WalkForwardResult:
    return WalkForwardResult(
        oos_sharpe=sharpe,
        oos_max_drawdown_pct=max_dd,
        per_fold_sharpes=per_fold_sharpes or [sharpe],
        per_fold_trade_counts=per_fold_trade_counts or [100],
        oos_snapshots=snapshots or [],
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


_SAMPLE_METRICS_KWARGS: dict = {
    "wf_sharpe": 1.02,
    "wf_max_drawdown": 10.8,
    "profit_factor": 1.58,
    "calmar_ratio": 1.7,
    "trade_count": 812,
    "avg_hold_bars": 5.3,
    "segment_pnl_share": {"us_large_cap": 0.35, "us_tech": 0.35, "us_mid": 0.30},
    "sortino_ratio": 1.71,
    "win_rate_by_segment": {"us_large_cap": 0.55, "us_tech": 0.62},
    "information_ratio": 0.95,
    "mc_5th_pct_sharpe": 0.52,
    "model_disagreement": 0.09,
    "turnover_adjusted_return": 12.5,
    "gross_sharpe": 1.15,
    "net_sharpe": 1.02,
    "param_stability_cv": 0.15,
    "per_model_proba_mean": {"xgboost": 0.62, "lightgbm": 0.58},
}


# ---------------------------------------------------------------------------
# TestSnapshotContext
# ---------------------------------------------------------------------------


class TestSnapshotContext:
    """Test git context capture."""

    def test_snapshot_captures_sha(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                type("R", (), {"stdout": "abcdef1234567890\n", "returncode": 0})(),
                type("R", (), {"stdout": "v0.1-5-gabcdef1\n", "returncode": 0})(),
                type("R", (), {"stdout": "", "returncode": 1})(),  # git diff --quiet
            ]
            ctx = tracker.snapshot_context()

        assert ctx["git_sha"] == "abcdef1234567890"
        assert ctx["git_describe"] == "v0.1-5-gabcdef1"
        assert ctx["git_dirty"] is True

    def test_snapshot_clean_repo(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                type("R", (), {"stdout": "abcdef1234567890\n", "returncode": 0})(),
                type("R", (), {"stdout": "v0.1\n", "returncode": 0})(),
                type("R", (), {"stdout": "", "returncode": 0})(),  # git diff --quiet = clean
            ]
            ctx = tracker.snapshot_context()

        assert ctx["git_dirty"] is False


class TestSnapshotConfig:
    """Test config hash computation."""

    def test_config_hash_deterministic(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        config = {"initial_cash": 100000, "max_positions": 10}
        strategy_configs = {"seg1": {"momentum": {"weight": 1.0}}}

        hash1 = tracker.compute_config_hash(config, strategy_configs)
        hash2 = tracker.compute_config_hash(config, strategy_configs)
        assert hash1 == hash2

    def test_config_hash_changes_with_config(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        config1 = {"initial_cash": 100000}
        config2 = {"initial_cash": 200000}
        strategy_configs = {"seg1": {"momentum": {"weight": 1.0}}}

        hash1 = tracker.compute_config_hash(config1, strategy_configs)
        hash2 = tracker.compute_config_hash(config2, strategy_configs)
        assert hash1 != hash2


# ---------------------------------------------------------------------------
# TestComputeMetrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Test metric computation from backtest outputs."""

    def test_compute_metrics_basic(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)

        trades = [
            _make_trade(100, 0.10, "AAPL"),
            _make_trade(-50, -0.05, "AAPL"),
            _make_trade(200, 0.20, "MSFT"),
        ]
        snapshots = _make_snapshots([100000, 100100, 100050, 100250])
        wf_result = _make_wf_result(
            sharpe=0.8,
            max_dd=5.0,
            per_fold_sharpes=[0.8],
            per_fold_trade_counts=[3],
            snapshots=snapshots,
        )
        mc_result = _make_bootstrap(sharpe_5th=0.3)
        segment_trades = {"us_large_cap": trades}

        metrics = tracker.compute_metrics(
            wf_result=wf_result,
            trades=trades,
            snapshots=snapshots,
            segment_trades=segment_trades,
            mc_result=mc_result,
        )

        assert isinstance(metrics, IterationMetrics)
        assert metrics.wf_sharpe == 0.8
        assert metrics.wf_max_drawdown == 5.0
        assert metrics.trade_count == 3
        assert metrics.mc_5th_pct_sharpe == 0.3

    def test_compute_metrics_segment_pnl_share(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)

        trades_a = [_make_trade(100, 0.10, "AAPL")]
        trades_b = [_make_trade(300, 0.30, "MSFT")]
        all_trades = trades_a + trades_b
        snapshots = _make_snapshots([100000, 100100, 100400])
        wf_result = _make_wf_result(snapshots=snapshots)
        mc_result = _make_bootstrap()
        segment_trades = {"seg_a": trades_a, "seg_b": trades_b}

        metrics = tracker.compute_metrics(
            wf_result=wf_result,
            trades=all_trades,
            snapshots=snapshots,
            segment_trades=segment_trades,
            mc_result=mc_result,
        )

        assert abs(metrics.segment_pnl_share["seg_a"] - 0.25) < 0.01
        assert abs(metrics.segment_pnl_share["seg_b"] - 0.75) < 0.01


# ---------------------------------------------------------------------------
# TestEvaluateGates
# ---------------------------------------------------------------------------

SAFETY_GATE_COUNT = 3
CALIBRATION_GATE_COUNT = 6


class TestEvaluateGates:
    """Test acceptance gate evaluation."""

    def test_all_pass_no_baseline(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        metrics = IterationMetrics(**_SAMPLE_METRICS_KWARGS)

        gate_results, verdict = tracker.evaluate_gates(metrics, baseline=None)

        assert verdict == "PASS"
        assert all(g.passed for g in gate_results)

    def test_safety_s1_fail_negative_sharpe(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        kwargs = {**_SAMPLE_METRICS_KWARGS, "wf_sharpe": -0.5}
        metrics = IterationMetrics(**kwargs)

        gate_results, verdict = tracker.evaluate_gates(metrics, baseline=None)

        assert verdict == "REJECT"
        s1 = next(g for g in gate_results if g.name == "S1")
        assert s1.passed is False

    def test_safety_s2_fail_high_drawdown(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        kwargs = {**_SAMPLE_METRICS_KWARGS, "wf_max_drawdown": 18.0}
        metrics = IterationMetrics(**kwargs)

        gate_results, verdict = tracker.evaluate_gates(metrics, baseline=None)

        assert verdict == "REJECT"
        s2 = next(g for g in gate_results if g.name == "S2")
        assert s2.passed is False

    def test_safety_s3_fail_mc_sharpe(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        kwargs = {**_SAMPLE_METRICS_KWARGS, "mc_5th_pct_sharpe": -0.1}
        metrics = IterationMetrics(**kwargs)

        gate_results, verdict = tracker.evaluate_gates(metrics, baseline=None)

        assert verdict == "REJECT"
        s3 = next(g for g in gate_results if g.name == "S3")
        assert s3.passed is False

    def test_single_calibration_fail_is_warn(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        metrics = IterationMetrics(**_SAMPLE_METRICS_KWARGS)
        # Baseline has sharpe 1.1, current 1.02 => regression 0.08 > 0.05
        baseline_kwargs = {**_SAMPLE_METRICS_KWARGS, "wf_sharpe": 1.1}
        baseline = IterationMetrics(**baseline_kwargs)

        gate_results, verdict = tracker.evaluate_gates(metrics, baseline=baseline)

        assert verdict == "WARN"
        c1 = next(g for g in gate_results if g.name == "C1")
        assert c1.passed is False

    def test_two_calibration_fails_is_reject(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        # C1: sharpe regresses > 0.05
        # C2: drawdown increases > 2%
        kwargs = {**_SAMPLE_METRICS_KWARGS, "wf_sharpe": 0.8, "wf_max_drawdown": 13.0}
        metrics = IterationMetrics(**kwargs)
        baseline = IterationMetrics(**_SAMPLE_METRICS_KWARGS)

        _gate_results, verdict = tracker.evaluate_gates(metrics, baseline=baseline)

        assert verdict == "REJECT"

    def test_calibration_c4_segment_concentration(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        kwargs = {
            **_SAMPLE_METRICS_KWARGS,
            "segment_pnl_share": {"us_large_cap": 0.40, "us_tech": 0.60},
        }
        metrics = IterationMetrics(**kwargs)

        gate_results, _verdict = tracker.evaluate_gates(metrics, baseline=None)

        c4 = next(g for g in gate_results if g.name == "C4")
        assert c4.passed is False

    def test_calibration_c6_param_stability(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        kwargs = {**_SAMPLE_METRICS_KWARGS, "param_stability_cv": 0.35}
        metrics = IterationMetrics(**kwargs)

        gate_results, _verdict = tracker.evaluate_gates(metrics, baseline=None)

        c6 = next(g for g in gate_results if g.name == "C6")
        assert c6.passed is False


# ---------------------------------------------------------------------------
# TestSaveAndLoad
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    """Test save/load iteration persistence."""

    def _make_metadata(self, name: str = "test-v1") -> IterationMetadata:
        gate = GateResult(
            name="S1",
            gate_type="safety",
            passed=True,
            value=1.02,
            threshold=0.0,
            message="ok",
        )
        return IterationMetadata(
            name=name,
            description="Test run",
            created_at=datetime(2026, 3, 2, 12, 0, 0, tzinfo=UTC),
            git_describe="v0.1-5-gabcdef0",
            git_sha="abcdef0123456789",
            git_dirty=False,
            config_hash="sha256_placeholder",
            strategy_configs={"us_large_cap": {"momentum": {"weight": 1.0}}},
            backtest_config={"initial_cash": 100000},
            metrics=IterationMetrics(**_SAMPLE_METRICS_KWARGS),
            gate_results=[gate],
            verdict="PASS",
        )

    def test_save_creates_files(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        meta = self._make_metadata()
        result_path = tracker.save(meta)

        assert (result_path / "metadata.json").exists()
        assert (tmp_path / "history.jsonl").exists()

    def test_load_roundtrip(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        meta = self._make_metadata()
        tracker.save(meta)

        loaded = tracker.load("test-v1")
        assert loaded.name == "test-v1"
        assert loaded.metrics.wf_sharpe == meta.metrics.wf_sharpe
        assert loaded.verdict == "PASS"

    def test_load_latest(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        meta1 = self._make_metadata("iter-1")
        meta2 = self._make_metadata("iter-2")
        tracker.save(meta1)
        tracker.save(meta2)

        latest = tracker.load_latest()
        assert latest is not None
        assert latest.name == "iter-2"

    def test_load_latest_empty(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        assert tracker.load_latest() is None

    def test_history_jsonl_append(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        meta1 = self._make_metadata("iter-1")
        meta2 = self._make_metadata("iter-2")
        tracker.save(meta1)
        tracker.save(meta2)

        history_path = tmp_path / "history.jsonl"
        lines = history_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["name"] == "iter-1"
        assert json.loads(lines[1])["name"] == "iter-2"

    def test_list_iterations(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        meta1 = self._make_metadata("iter-1")
        meta2 = self._make_metadata("iter-2")
        tracker.save(meta1)
        tracker.save(meta2)

        iterations = tracker.list_iterations()
        assert len(iterations) == 2
        assert iterations[0]["name"] == "iter-1"
        assert iterations[1]["name"] == "iter-2"


# ---------------------------------------------------------------------------
# TestCompare
# ---------------------------------------------------------------------------


class TestCompare:
    """Test iteration comparison."""

    def _make_metadata(self, name: str, sharpe: float = 1.02) -> IterationMetadata:
        gate = GateResult(
            name="S1",
            gate_type="safety",
            passed=True,
            value=sharpe,
            threshold=0.0,
            message="ok",
        )
        kwargs = {**_SAMPLE_METRICS_KWARGS, "wf_sharpe": sharpe}
        return IterationMetadata(
            name=name,
            description="Test run",
            created_at=datetime(2026, 3, 2, 12, 0, 0, tzinfo=UTC),
            git_describe="v0.1",
            git_sha="abc123",
            git_dirty=False,
            config_hash="hash",
            strategy_configs={},
            backtest_config={},
            metrics=IterationMetrics(**kwargs),
            gate_results=[gate],
            verdict="PASS",
        )

    def test_compare_metric_deltas(self, tmp_path: Path) -> None:
        tracker = IterationTracker(results_root=tmp_path)
        meta1 = self._make_metadata("base", sharpe=0.87)
        meta2 = self._make_metadata("current", sharpe=1.02)
        tracker.save(meta1)
        tracker.save(meta2)

        comp = tracker.compare("current", "base")
        assert abs(comp.metric_deltas["wf_sharpe"] - 0.15) < 0.001


# ---------------------------------------------------------------------------
# TestAtomicWrite
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """Test that writes are atomic (temp + rename pattern)."""

    def test_save_no_partial_on_error(self, tmp_path: Path) -> None:
        """If something goes wrong during save, no partial metadata.json exists."""
        tracker = IterationTracker(results_root=tmp_path)
        gate = GateResult(
            name="S1",
            gate_type="safety",
            passed=True,
            value=1.0,
            threshold=0.0,
            message="ok",
        )
        meta = IterationMetadata(
            name="atomic-test",
            description="Test",
            created_at=datetime(2026, 3, 2, 12, 0, 0, tzinfo=UTC),
            git_describe="v0.1",
            git_sha="abc",
            git_dirty=False,
            config_hash="h",
            strategy_configs={},
            backtest_config={},
            metrics=IterationMetrics(**_SAMPLE_METRICS_KWARGS),
            gate_results=[gate],
            verdict="PASS",
        )

        # Patch os.rename to simulate failure
        rename_target = "finalayze.backtest.iteration_tracker.os.rename"
        with (
            patch(rename_target, side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            tracker.save(meta)

        # No metadata.json should exist
        assert not (tmp_path / "atomic-test" / "metadata.json").exists()
