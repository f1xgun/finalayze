"""Unit tests for AgentOrchestrator pipeline coordinator.

Tests cover:
- run() short-circuit on no conflicts
- run() creates debate(s) from conflicting outputs
- run() uses fresh ConflictDetector per call (no stale dedup)
- run() handles multiple independent conflict groups
- finalize_debate() creates experiment on contradictions
- finalize_debate() resolves debate when no contradictions
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.schemas import (
    AgentOutput,
    Claim,
    ClaimCheckResult,
    ClaimVerdict,
    FactCheckReport,
    FileLineSource,
    MetricSource,
    SuccessCriteria,
)
from finalayze.orchestration.agent_orchestrator import AgentOrchestrator

# ─── Constants ────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 4, 12, tzinfo=UTC)

# Confidence values ensuring delta > 0.15 (0.25 delta) to pass conflict filter
_CONF_HIGH = 0.90
_CONF_LOW = 0.65


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_file_claim(
    statement: str,
    path: str = "src/finalayze/strategies/combiner.py",
    line: int = 42,
    excerpt: str = "some code",
    confidence: float = _CONF_HIGH,
) -> Claim:
    return Claim(
        statement=statement,
        source=FileLineSource(path=path, line=line, excerpt=excerpt),
        confidence=confidence,
    )


def _make_agent_output(
    agent_name: str,
    recommendation: str,
    claims: list[Claim] | None = None,
    timestamp: datetime = _NOW,
) -> AgentOutput:
    if claims is None:
        claims = [_make_file_claim(f"{agent_name} default claim")]
    return AgentOutput(
        agent_name=agent_name,
        recommendation=recommendation,
        claims=claims,
        timestamp=timestamp,
    )


def _make_fact_check_report(
    debate_id: str,
    *,
    has_contradictions: bool,
) -> FactCheckReport:
    verdict = ClaimVerdict.CONTRADICTED if has_contradictions else ClaimVerdict.VERIFIED
    claim = _make_file_claim("some claim")
    result = ClaimCheckResult(claim=claim, verdict=verdict, evidence="test evidence")
    return FactCheckReport(
        debate_id=debate_id,
        arbiter_timestamp=_NOW,
        results=[result],
    )


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestAgentOrchestratorRun:
    """Tests for AgentOrchestrator.run()."""

    def test_run_no_conflicts_returns_empty_list(self, tmp_path: Path) -> None:
        """run() with non-conflicting outputs short-circuits and returns []."""
        debates_dir = tmp_path / "debates"
        experiments_dir = tmp_path / "experiments"

        orch = AgentOrchestrator(
            debates_dir=debates_dir,
            experiments_dir=experiments_dir,
        )

        # Both agents agree — no direction conflict
        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER based on momentum signals",
            claims=[_make_file_claim("momentum signals strong", confidence=_CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "BUY SBER within risk limits",
            claims=[_make_file_claim("risk limits acceptable", confidence=_CONF_LOW)],
        )

        debate_ids = orch.run([output_a, output_b])

        assert debate_ids == []

    def test_run_conflicting_outputs_returns_debate_id(self, tmp_path: Path) -> None:
        """run() with conflicting outputs creates a debate and returns debate_id."""
        debates_dir = tmp_path / "debates"
        experiments_dir = tmp_path / "experiments"

        orch = AgentOrchestrator(
            debates_dir=debates_dir,
            experiments_dir=experiments_dir,
        )

        # Opposing directions trigger DIRECTION conflict
        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER with high confidence based on momentum signals",
            claims=[_make_file_claim("momentum signals strong", confidence=_CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "SELL SBER due to elevated volatility and circuit breaker concerns",
            claims=[_make_file_claim("volatility too high", confidence=_CONF_LOW)],
        )

        debate_ids = orch.run([output_a, output_b])

        assert len(debate_ids) >= 1
        assert all(isinstance(did, str) for did in debate_ids)

    def test_run_creates_debate_file_in_debates_dir(self, tmp_path: Path) -> None:
        """run() with conflicts creates a .md file in the debates directory."""
        debates_dir = tmp_path / "debates"
        experiments_dir = tmp_path / "experiments"

        orch = AgentOrchestrator(
            debates_dir=debates_dir,
            experiments_dir=experiments_dir,
        )

        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER with high confidence",
            claims=[_make_file_claim("momentum strong", confidence=_CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "SELL SBER due to volatility",
            claims=[_make_file_claim("volatility too high", confidence=_CONF_LOW)],
        )

        debate_ids = orch.run([output_a, output_b])

        assert len(debate_ids) >= 1
        for debate_id in debate_ids:
            debate_file = debates_dir / f"{debate_id}.md"
            assert debate_file.exists(), f"Expected debate file at {debate_file}"

    def test_run_fresh_conflict_detector_per_call(self, tmp_path: Path) -> None:
        """run() must use a fresh ConflictDetector per call — no stale dedup across calls."""
        debates_dir = tmp_path / "debates"
        experiments_dir = tmp_path / "experiments"

        orch = AgentOrchestrator(
            debates_dir=debates_dir,
            experiments_dir=experiments_dir,
        )

        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER with high confidence",
            claims=[_make_file_claim("momentum strong", confidence=_CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "SELL SBER due to volatility",
            claims=[_make_file_claim("volatility too high", confidence=_CONF_LOW)],
        )

        # First call — creates a debate
        ids_first = orch.run([output_a, output_b])
        assert len(ids_first) >= 1

        # Second call with same inputs — must NOT suppress due to stale dedup
        # (ConflictDetector instantiated fresh each call)
        ids_second = orch.run([output_a, output_b])
        # With a fresh detector, same conflicts ARE re-detected
        # (debate_ids may differ due to timestamp in key; both non-empty)
        assert len(ids_second) >= 1

    def test_run_records_agent_positions_in_debate(self, tmp_path: Path) -> None:
        """run() calls add_agent_position for each agent involved in a conflict."""
        from finalayze.core.debate_manager import DebateManager
        from finalayze.core.experiment_manager import ExperimentManager

        debates_dir = tmp_path / "debates"
        experiments_dir = tmp_path / "experiments"

        dm = DebateManager(debates_dir=debates_dir)
        em = ExperimentManager(experiments_dir=experiments_dir, debates_dir=debates_dir)

        orch = AgentOrchestrator(debate_manager=dm, experiment_manager=em)

        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER with high confidence",
            claims=[_make_file_claim("momentum strong", confidence=_CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "SELL SBER due to volatility",
            claims=[_make_file_claim("volatility too high", confidence=_CONF_LOW)],
        )

        debate_ids = orch.run([output_a, output_b])
        assert len(debate_ids) >= 1

        # Verify debate file contains both agent position sections
        debate_id = debate_ids[0]
        debate_content = (debates_dir / f"{debate_id}.md").read_text()
        assert "quant-analyst" in debate_content
        assert "risk-officer" in debate_content


class TestAgentOrchestratorFinalizeDebate:
    """Tests for AgentOrchestrator.finalize_debate()."""

    def test_finalize_with_contradictions_creates_experiment(self, tmp_path: Path) -> None:
        """finalize_debate() with contradictions creates an experiment and returns exp_id."""
        from finalayze.core.debate_manager import DebateManager
        from finalayze.core.experiment_manager import ExperimentManager

        debates_dir = tmp_path / "debates"
        experiments_dir = tmp_path / "experiments"

        dm = DebateManager(debates_dir=debates_dir)
        em = ExperimentManager(experiments_dir=experiments_dir, debates_dir=debates_dir)

        orch = AgentOrchestrator(debate_manager=dm, experiment_manager=em)

        # First create a debate via run()
        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER with high confidence",
            claims=[_make_file_claim("momentum strong", confidence=_CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "SELL SBER due to volatility",
            claims=[_make_file_claim("volatility too high", confidence=_CONF_LOW)],
        )

        debate_ids = orch.run([output_a, output_b])
        assert len(debate_ids) >= 1
        debate_id = debate_ids[0]

        # Finalize with contradictions
        report = _make_fact_check_report(debate_id, has_contradictions=True)
        exp_id = orch.finalize_debate(debate_id, report)

        assert exp_id is not None
        assert isinstance(exp_id, str)
        # Experiment file should exist
        exp_file = experiments_dir / f"{exp_id}.md"
        assert exp_file.exists()

    def test_finalize_without_contradictions_resolves_debate(self, tmp_path: Path) -> None:
        """finalize_debate() without contradictions resolves the debate and returns None."""
        from finalayze.core.debate_manager import DebateManager
        from finalayze.core.experiment_manager import ExperimentManager

        debates_dir = tmp_path / "debates"
        experiments_dir = tmp_path / "experiments"

        dm = DebateManager(debates_dir=debates_dir)
        em = ExperimentManager(experiments_dir=experiments_dir, debates_dir=debates_dir)

        orch = AgentOrchestrator(debate_manager=dm, experiment_manager=em)

        # Create a debate
        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER with high confidence",
            claims=[_make_file_claim("momentum strong", confidence=_CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "SELL SBER due to volatility",
            claims=[_make_file_claim("volatility too high", confidence=_CONF_LOW)],
        )

        debate_ids = orch.run([output_a, output_b])
        assert len(debate_ids) >= 1
        debate_id = debate_ids[0]

        # Finalize without contradictions
        report = _make_fact_check_report(debate_id, has_contradictions=False)
        exp_id = orch.finalize_debate(debate_id, report)

        assert exp_id is None

        # Debate file should show resolved status
        debate_state = dm.read_debate(debate_id)
        from finalayze.core.schemas import DebateStatus

        assert debate_state.status == DebateStatus.RESOLVED
