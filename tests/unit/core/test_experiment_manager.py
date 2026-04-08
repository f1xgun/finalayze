"""Tests for ExperimentManager CRUD operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from finalayze.core.debate_manager import DebateManager
from finalayze.core.experiment_manager import ExperimentManager, _compute_verdict
from finalayze.core.schemas import (
    DebateStatus,
    ExperimentResult,
    ExperimentStatus,
    SuccessCriteria,
)

_CRITERIA = SuccessCriteria(metric="pf", threshold=1.3, operator=">=")


class TestCreateAndRead:
    def test_create_and_read(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        path = mgr.create_experiment(
            experiment_id="exp-001",
            hypothesis="Dual momentum improves PF",
            success_criteria=_CRITERIA,
        )

        assert path.exists()
        assert path.name == "exp-001.md"

        state = mgr.read_experiment("exp-001")
        assert state.experiment_id == "exp-001"
        assert state.hypothesis == "Dual momentum improves PF"
        assert state.status == ExperimentStatus.PENDING
        assert state.success_criteria.metric == "pf"
        assert state.success_criteria.threshold == pytest.approx(1.3)
        assert state.success_criteria.operator == ">="
        assert state.debate_id is None
        assert state.results == []

    def test_create_with_preset_overrides(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        overrides = {"dual_momentum.weight": 0.15}
        mgr.create_experiment(
            experiment_id="exp-002",
            hypothesis="test",
            success_criteria=_CRITERIA,
            preset_overrides=overrides,
        )

        state = mgr.read_experiment("exp-002")
        assert state.preset_overrides == overrides


class TestCreateWithDebateLink:
    def test_create_with_debate_link(self, tmp_path: Path) -> None:
        debates_dir = tmp_path / "debates"
        experiments_dir = tmp_path / "experiments"

        # Create a real debate first so escalate_debate can find it
        dm = DebateManager(debates_dir=debates_dir)
        dm.create_debate("debate-01", "Test topic", ["quant-analyst"])

        mgr = ExperimentManager(
            experiments_dir=experiments_dir,
            debates_dir=debates_dir,
        )
        mgr.create_experiment(
            experiment_id="exp-003",
            hypothesis="test",
            success_criteria=_CRITERIA,
            debate_id="debate-01",
        )

        # Verify experiment has debate_id
        state = mgr.read_experiment("exp-003")
        assert state.debate_id == "debate-01"

        # Verify debate was escalated with experiment_id (bidirectional link)
        debate = dm.read_debate("debate-01")
        assert debate.status == DebateStatus.ESCALATED
        assert debate.experiment_id == "exp-003"


class TestListExperiments:
    def test_list_experiments(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        mgr.create_experiment("alpha", "test", _CRITERIA)
        mgr.create_experiment("beta", "test", _CRITERIA)

        result = mgr.list_experiments()
        assert result == ["alpha", "beta"]

    def test_list_experiments_empty(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        assert mgr.list_experiments() == []


class TestUpdateStatus:
    def test_update_status(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        mgr.create_experiment("exp-001", "test", _CRITERIA)

        mgr.update_status("exp-001", ExperimentStatus.RUNNING)

        state = mgr.read_experiment("exp-001")
        assert state.status == ExperimentStatus.RUNNING


class TestLinkResult:
    def test_link_result(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        mgr.create_experiment("exp-001", "test", _CRITERIA)

        result = ExperimentResult(
            run_name="A-only",
            iteration_name="iter-01",
            metrics={"wf_sharpe": 0.15, "pf": 1.4},
        )
        mgr.link_result("exp-001", result)

        state = mgr.read_experiment("exp-001")
        assert len(state.results) == 1
        assert state.results[0].run_name == "A-only"
        assert state.results[0].metrics["pf"] == pytest.approx(1.4)

    def test_link_multiple_results(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        mgr.create_experiment("exp-001", "test", _CRITERIA)

        r1 = ExperimentResult(run_name="A", iteration_name="i1", metrics={"pf": 1.2})
        r2 = ExperimentResult(run_name="B", iteration_name="i2", metrics={"pf": 1.5})
        mgr.link_result("exp-001", r1)
        mgr.link_result("exp-001", r2)

        state = mgr.read_experiment("exp-001")
        assert len(state.results) == 2


class TestComputeVerdict:
    def test_accept(self) -> None:
        verdict, reasoning = _compute_verdict(_CRITERIA, 1.5)
        assert verdict == "ACCEPTED"
        assert "meets threshold" in reasoning

    def test_reject(self) -> None:
        # 1.0 is ~23% below 1.3 -> REJECTED (>10% miss)
        verdict, reasoning = _compute_verdict(_CRITERIA, 1.0)
        assert verdict == "REJECTED"
        assert "misses threshold" in reasoning

    def test_inconclusive(self) -> None:
        # 1.25 is ~3.8% below 1.3 -> INCONCLUSIVE (within 10%)
        verdict, reasoning = _compute_verdict(_CRITERIA, 1.25)
        assert verdict == "INCONCLUSIVE"
        assert "within 10%" in reasoning

    def test_exact_threshold(self) -> None:
        verdict, _ = _compute_verdict(_CRITERIA, 1.3)
        assert verdict == "ACCEPTED"

    def test_less_equal_operator(self) -> None:
        criteria = SuccessCriteria(metric="max_dd", threshold=0.05, operator="<=")
        verdict, _ = _compute_verdict(criteria, 0.03)
        assert verdict == "ACCEPTED"

        verdict, _ = _compute_verdict(criteria, 0.10)
        assert verdict == "REJECTED"


class TestRecordVerdict:
    def test_record_verdict_accept(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        mgr.create_experiment("exp-001", "test", _CRITERIA)

        mgr.record_verdict("exp-001", metric_value=1.5)

        state = mgr.read_experiment("exp-001")
        assert state.status == ExperimentStatus.ACCEPTED
        assert state.verdict == "ACCEPTED"
        assert state.reasoning is not None

    def test_record_verdict_reject(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        mgr.create_experiment("exp-001", "test", _CRITERIA)

        mgr.record_verdict("exp-001", metric_value=1.0)

        state = mgr.read_experiment("exp-001")
        assert state.status == ExperimentStatus.REJECTED
        assert state.verdict == "REJECTED"

    def test_record_verdict_inconclusive(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        mgr.create_experiment("exp-001", "test", _CRITERIA)

        mgr.record_verdict("exp-001", metric_value=1.25)

        state = mgr.read_experiment("exp-001")
        assert state.status == ExperimentStatus.INCONCLUSIVE
        assert state.verdict == "INCONCLUSIVE"


class TestGetByDebate:
    def test_get_by_debate(self, tmp_path: Path) -> None:
        debates_dir = tmp_path / "debates"
        experiments_dir = tmp_path / "experiments"

        # Create a real debate so escalate_debate works
        dm = DebateManager(debates_dir=debates_dir)
        dm.create_debate("debate-01", "Test topic", ["agent-a"])

        mgr = ExperimentManager(
            experiments_dir=experiments_dir,
            debates_dir=debates_dir,
        )
        mgr.create_experiment("exp-001", "test", _CRITERIA, debate_id="debate-01")

        state = mgr.get_by_debate("debate-01")
        assert state is not None
        assert state.experiment_id == "exp-001"

    def test_get_by_debate_none(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        assert mgr.get_by_debate("nonexistent") is None


class TestReadNonexistent:
    def test_read_nonexistent(self, tmp_path: Path) -> None:
        mgr = ExperimentManager(experiments_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            mgr.read_experiment("no-such-experiment")
