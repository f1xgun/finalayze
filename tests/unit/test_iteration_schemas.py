"""Tests for iteration tracking schemas in core/schemas.py.

Task 3.1: GateResult, IterationMetrics, IterationMetadata, IterationComparison.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from finalayze.core.schemas import (
    GateResult,
    IterationComparison,
    IterationMetadata,
    IterationMetrics,
)

# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------


class TestGateResult:
    """Tests for GateResult schema."""

    def test_create_passing_gate(self) -> None:
        gate = GateResult(
            name="S1",
            gate_type="safety",
            passed=True,
            value=0.5,
            threshold=0.0,
            message="WF Sharpe >= 0.0",
        )
        assert gate.name == "S1"
        assert gate.gate_type == "safety"
        assert gate.passed is True
        assert gate.value == 0.5
        assert gate.threshold == 0.0

    def test_create_failing_gate(self) -> None:
        gate = GateResult(
            name="S2",
            gate_type="safety",
            passed=False,
            value=0.18,
            threshold=0.15,
            message="Max DD 18% exceeds 15% ceiling",
        )
        assert gate.passed is False

    def test_frozen(self) -> None:
        gate = GateResult(
            name="S1",
            gate_type="safety",
            passed=True,
            value=0.5,
            threshold=0.0,
            message="ok",
        )
        with pytest.raises(ValidationError):
            gate.name = "S2"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        gate = GateResult(
            name="C1",
            gate_type="calibration",
            passed=True,
            value=0.02,
            threshold=0.05,
            message="Sharpe regression within tolerance",
        )
        data = gate.model_dump()
        restored = GateResult(**data)
        assert restored == gate

    def test_json_roundtrip(self) -> None:
        gate = GateResult(
            name="C3",
            gate_type="calibration",
            passed=False,
            value=45.0,
            threshold=60.0,
            message="Trade count below 60/fold",
        )
        json_str = gate.model_dump_json()
        restored = GateResult.model_validate_json(json_str)
        assert restored == gate


# ---------------------------------------------------------------------------
# IterationMetrics
# ---------------------------------------------------------------------------

_SAMPLE_METRICS_KWARGS: dict = {
    "wf_sharpe": 1.02,
    "wf_max_drawdown": 10.8,
    "profit_factor": 1.58,
    "calmar_ratio": 1.7,
    "trade_count": 812,
    "avg_hold_bars": 5.3,
    "segment_pnl_share": {"us_large_cap": 0.29, "us_tech": 0.71},
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


class TestIterationMetrics:
    """Tests for IterationMetrics schema."""

    def test_create_metrics(self) -> None:
        metrics = IterationMetrics(**_SAMPLE_METRICS_KWARGS)
        assert metrics.wf_sharpe == 1.02
        assert metrics.trade_count == 812
        assert metrics.segment_pnl_share["us_tech"] == 0.71

    def test_frozen(self) -> None:
        metrics = IterationMetrics(**_SAMPLE_METRICS_KWARGS)
        with pytest.raises(ValidationError):
            metrics.wf_sharpe = 2.0  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        metrics = IterationMetrics(**_SAMPLE_METRICS_KWARGS)
        data = metrics.model_dump()
        restored = IterationMetrics(**data)
        assert restored == metrics

    def test_information_ratio_nullable(self) -> None:
        kwargs = {**_SAMPLE_METRICS_KWARGS, "information_ratio": None}
        metrics = IterationMetrics(**kwargs)
        assert metrics.information_ratio is None


# ---------------------------------------------------------------------------
# IterationMetadata
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 2, 12, 0, 0, tzinfo=UTC)


class TestIterationMetadata:
    """Tests for IterationMetadata schema."""

    def _make_metadata(self, **overrides: object) -> IterationMetadata:
        gate = GateResult(
            name="S1",
            gate_type="safety",
            passed=True,
            value=1.02,
            threshold=0.0,
            message="ok",
        )
        defaults: dict = {
            "name": "test-iteration",
            "description": "Test run",
            "created_at": _NOW,
            "git_describe": "v0.1-5-gabcdef0",
            "git_sha": "abcdef0123456789",
            "git_dirty": False,
            "config_hash": "sha256_placeholder",
            "strategy_configs": {"us_large_cap": {"momentum": {"weight": 1.0}}},
            "backtest_config": {"initial_cash": 100000},
            "metrics": IterationMetrics(**_SAMPLE_METRICS_KWARGS),
            "gate_results": [gate],
            "verdict": "PASS",
        }
        defaults.update(overrides)
        return IterationMetadata(**defaults)

    def test_create_metadata(self) -> None:
        meta = self._make_metadata()
        assert meta.name == "test-iteration"
        assert meta.schema_version == 1
        assert meta.verdict == "PASS"
        assert len(meta.gate_results) == 1

    def test_schema_version_default(self) -> None:
        meta = self._make_metadata()
        assert meta.schema_version == 1

    def test_tags_default_empty(self) -> None:
        meta = self._make_metadata()
        assert meta.tags == []

    def test_tags_provided(self) -> None:
        meta = self._make_metadata(tags=["experiment", "momentum-v2"])
        assert meta.tags == ["experiment", "momentum-v2"]

    def test_frozen(self) -> None:
        meta = self._make_metadata()
        with pytest.raises(ValidationError):
            meta.name = "changed"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        meta = self._make_metadata()
        json_str = meta.model_dump_json()
        restored = IterationMetadata.model_validate_json(json_str)
        assert restored.name == meta.name
        assert restored.metrics.wf_sharpe == meta.metrics.wf_sharpe


# ---------------------------------------------------------------------------
# IterationComparison
# ---------------------------------------------------------------------------


class TestIterationComparison:
    """Tests for IterationComparison schema."""

    def test_create_comparison(self) -> None:
        gate = GateResult(
            name="C1",
            gate_type="calibration",
            passed=True,
            value=0.15,
            threshold=0.05,
            message="Sharpe improved",
        )
        comp = IterationComparison(
            current="v2",
            baseline="v1",
            metric_deltas={"wf_sharpe": 0.15, "wf_max_drawdown": -0.4},
            gate_results=[gate],
            verdict="PASS",
        )
        assert comp.current == "v2"
        assert comp.baseline == "v1"
        assert comp.metric_deltas["wf_sharpe"] == 0.15
        assert comp.verdict == "PASS"

    def test_frozen(self) -> None:
        comp = IterationComparison(
            current="v2",
            baseline="v1",
            metric_deltas={},
            gate_results=[],
            verdict="PASS",
        )
        with pytest.raises(ValidationError):
            comp.verdict = "REJECT"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        gate = GateResult(
            name="S1",
            gate_type="safety",
            passed=True,
            value=1.0,
            threshold=0.0,
            message="ok",
        )
        comp = IterationComparison(
            current="iter-2",
            baseline="iter-1",
            metric_deltas={"wf_sharpe": 0.1},
            gate_results=[gate],
            verdict="WARN",
        )
        json_str = comp.model_dump_json()
        restored = IterationComparison.model_validate_json(json_str)
        assert restored == comp
