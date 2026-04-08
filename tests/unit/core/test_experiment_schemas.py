"""Tests for experiment registry schemas (Layer 0)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from finalayze.core.schemas import (
    ExperimentResult,
    ExperimentState,
    ExperimentStatus,
    SuccessCriteria,
)


# ── ExperimentStatus ────────────────────────────────────────────────────────


class TestExperimentStatus:
    def test_experiment_status_values(self) -> None:
        assert ExperimentStatus.PENDING == "pending"
        assert ExperimentStatus.RUNNING == "running"
        assert ExperimentStatus.COMPLETED == "completed"
        assert ExperimentStatus.ACCEPTED == "accepted"
        assert ExperimentStatus.REJECTED == "rejected"
        assert ExperimentStatus.INCONCLUSIVE == "inconclusive"


# ── SuccessCriteria ─────────────────────────────────────────────────────────


class TestSuccessCriteria:
    def test_success_criteria_valid(self) -> None:
        sc = SuccessCriteria(metric="profit_factor", threshold=1.3, operator=">=")
        assert sc.metric == "profit_factor"
        assert sc.threshold == pytest.approx(1.3)
        assert sc.operator == ">="

    def test_success_criteria_default_operator(self) -> None:
        sc = SuccessCriteria(metric="pf", threshold=1.0)
        assert sc.operator == ">="

    def test_success_criteria_bad_operator(self) -> None:
        with pytest.raises(ValidationError):
            SuccessCriteria(metric="pf", threshold=1.0, operator="==")

    def test_success_criteria_arbitrary_string_operator(self) -> None:
        with pytest.raises(ValidationError):
            SuccessCriteria(metric="pf", threshold=1.0, operator="foo")

    def test_success_criteria_frozen(self) -> None:
        sc = SuccessCriteria(metric="pf", threshold=1.0)
        with pytest.raises(ValidationError):
            sc.metric = "other"  # type: ignore[misc]


# ── ExperimentResult ────────────────────────────────────────────────────────


class TestExperimentResult:
    def test_experiment_result_valid(self) -> None:
        er = ExperimentResult(
            run_name="A-only",
            iteration_name="test",
            metrics={"wf_sharpe": 0.1},
        )
        assert er.run_name == "A-only"
        assert er.iteration_name == "test"
        assert er.metrics == {"wf_sharpe": 0.1}


# ── ExperimentState ─────────────────────────────────────────────────────────


_CRITERIA = SuccessCriteria(metric="pf", threshold=1.3)


class TestExperimentState:
    def test_experiment_state_valid(self) -> None:
        state = ExperimentState(
            experiment_id="exp-001",
            hypothesis="Dual momentum improves PF",
            success_criteria=_CRITERIA,
            status=ExperimentStatus.PENDING,
            created="2026-04-07",
        )
        assert state.experiment_id == "exp-001"
        assert state.status == ExperimentStatus.PENDING
        assert state.verdict is None
        assert state.debate_id is None

    def test_experiment_state_terminal_requires_verdict(self) -> None:
        with pytest.raises(ValidationError, match="verdict is required"):
            ExperimentState(
                experiment_id="exp-001",
                hypothesis="test",
                success_criteria=_CRITERIA,
                status=ExperimentStatus.ACCEPTED,
                created="2026-04-07",
            )

    def test_experiment_state_terminal_with_verdict(self) -> None:
        state = ExperimentState(
            experiment_id="exp-001",
            hypothesis="test",
            success_criteria=_CRITERIA,
            status=ExperimentStatus.ACCEPTED,
            created="2026-04-07",
            verdict="ACCEPTED",
            reasoning="PF meets threshold",
        )
        assert state.verdict == "ACCEPTED"

    def test_experiment_state_bad_id(self) -> None:
        with pytest.raises(ValidationError, match="experiment_id must match"):
            ExperimentState(
                experiment_id="../evil",
                hypothesis="test",
                success_criteria=_CRITERIA,
                status=ExperimentStatus.PENDING,
                created="2026-04-07",
            )

    def test_experiment_state_bad_id_special_chars(self) -> None:
        with pytest.raises(ValidationError, match="experiment_id must match"):
            ExperimentState(
                experiment_id="exp/bad",
                hypothesis="test",
                success_criteria=_CRITERIA,
                status=ExperimentStatus.PENDING,
                created="2026-04-07",
            )

    def test_experiment_state_frozen(self) -> None:
        state = ExperimentState(
            experiment_id="exp-001",
            hypothesis="test",
            success_criteria=_CRITERIA,
            status=ExperimentStatus.PENDING,
            created="2026-04-07",
        )
        with pytest.raises(ValidationError):
            state.status = ExperimentStatus.RUNNING  # type: ignore[misc]
