"""Tests for DebateManager CRUD operations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from finalayze.core.debate_manager import DebateManager
from finalayze.core.schemas import (
    AgentOutput,
    Claim,
    ClaimCheckResult,
    ClaimVerdict,
    DebateStatus,
    FactCheckReport,
    FileLineSource,
)

_DT = datetime(2026, 4, 7, tzinfo=timezone.utc)

_FILE_SOURCE = FileLineSource(
    kind="file",
    path="src/finalayze/strategies/combiner.py",
    line=142,
    excerpt="class StrategyCombiner",
)

_CLAIM = Claim(
    statement="Combiner uses ADX routing",
    source=_FILE_SOURCE,
    confidence=0.9,
)

_AGENT_OUTPUT = AgentOutput(
    agent_name="quant-analyst",
    recommendation="Enable dual_momentum on ru_blue_chips",
    claims=[_CLAIM],
    timestamp=_DT,
)


def _make_fact_check_report(debate_id: str) -> FactCheckReport:
    return FactCheckReport(
        debate_id=debate_id,
        arbiter_timestamp=_DT,
        results=[
            ClaimCheckResult(
                claim=_CLAIM,
                verdict=ClaimVerdict.VERIFIED,
                evidence="Found class StrategyCombiner at line 142.",
            )
        ],
    )


class TestCreateDebate:
    def test_create_debate(self, tmp_path: Path) -> None:
        manager = DebateManager(debates_dir=tmp_path)
        path = manager.create_debate(
            debate_id="2026-04-07-test",
            topic="Should we enable dual_momentum on ru_blue_chips?",
            agents=["quant-analyst", "risk-officer"],
        )

        assert path.exists()
        assert path.name == "2026-04-07-test.md"
        content = path.read_text()
        assert "debate_id: 2026-04-07-test" in content
        assert "status: open" in content
        assert "quant-analyst" in content
        assert "risk-officer" in content
        assert "dual_momentum" in content


class TestReadDebate:
    def test_read_debate(self, tmp_path: Path) -> None:
        manager = DebateManager(debates_dir=tmp_path)
        manager.create_debate(
            debate_id="2026-04-07-test",
            topic="Should we enable dual_momentum?",
            agents=["quant-analyst"],
        )

        state = manager.read_debate("2026-04-07-test")
        assert state.debate_id == "2026-04-07-test"
        assert state.topic == "Should we enable dual_momentum?"
        assert state.status == DebateStatus.OPEN
        assert "quant-analyst" in state.agents

    def test_read_debate_not_found(self, tmp_path: Path) -> None:
        manager = DebateManager(debates_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            manager.read_debate("nonexistent")


class TestUpdateDebateResolution:
    def test_update_debate_resolution(self, tmp_path: Path) -> None:
        manager = DebateManager(debates_dir=tmp_path)
        manager.create_debate(
            debate_id="2026-04-07-test",
            topic="Test topic",
            agents=["quant-analyst"],
        )

        manager.resolve_debate("2026-04-07-test", resolution="Agreed on enabling dual_momentum")

        state = manager.read_debate("2026-04-07-test")
        assert state.status == DebateStatus.RESOLVED
        assert state.resolution == "Agreed on enabling dual_momentum"


class TestEscalateDebate:
    def test_escalate_debate(self, tmp_path: Path) -> None:
        manager = DebateManager(debates_dir=tmp_path)
        manager.create_debate(
            debate_id="2026-04-07-test",
            topic="Test topic",
            agents=["quant-analyst"],
        )

        manager.escalate_debate("2026-04-07-test", experiment_id="EXP-001")

        state = manager.read_debate("2026-04-07-test")
        assert state.status == DebateStatus.ESCALATED
        assert state.experiment_id == "EXP-001"


class TestListDebates:
    def test_list_debates(self, tmp_path: Path) -> None:
        manager = DebateManager(debates_dir=tmp_path)
        manager.create_debate("2026-04-07-test1", "Topic 1", ["a"])
        manager.create_debate("2026-04-07-test2", "Topic 2", ["b"])
        manager.create_debate("2026-04-07-test3", "Topic 3", ["c"])

        debates = manager.list_debates()
        assert len(debates) == 3
        assert "2026-04-07-test1" in debates
        assert "2026-04-07-test2" in debates
        assert "2026-04-07-test3" in debates

    def test_list_debates_empty(self, tmp_path: Path) -> None:
        manager = DebateManager(debates_dir=tmp_path)
        assert manager.list_debates() == []


class TestAddAgentPosition:
    def test_add_agent_position(self, tmp_path: Path) -> None:
        manager = DebateManager(debates_dir=tmp_path)
        manager.create_debate(
            debate_id="2026-04-07-test",
            topic="Test topic",
            agents=["quant-analyst"],
        )

        manager.add_agent_position("2026-04-07-test", "quant-analyst", _AGENT_OUTPUT)

        content = (tmp_path / "2026-04-07-test.md").read_text()
        assert "quant-analyst Position" in content
        assert "Enable dual_momentum on ru_blue_chips" in content
        assert "Combiner uses ADX routing" in content


class TestAddArbiterReport:
    def test_add_arbiter_report(self, tmp_path: Path) -> None:
        manager = DebateManager(debates_dir=tmp_path)
        manager.create_debate(
            debate_id="2026-04-07-test",
            topic="Test topic",
            agents=["quant-analyst"],
        )

        report = _make_fact_check_report("2026-04-07-test")
        manager.add_arbiter_report("2026-04-07-test", report)

        state = manager.read_debate("2026-04-07-test")
        assert state.arbiter_report is not None
        assert state.arbiter_report.debate_id == "2026-04-07-test"

        content = (tmp_path / "2026-04-07-test.md").read_text()
        assert "Arbiter Fact-Check" in content


class TestRoundtrip:
    def test_roundtrip(self, tmp_path: Path) -> None:
        manager = DebateManager(debates_dir=tmp_path)

        # Create
        manager.create_debate(
            debate_id="2026-04-07-roundtrip",
            topic="Roundtrip test topic",
            agents=["quant-analyst", "risk-officer"],
        )

        # Add agent position
        manager.add_agent_position("2026-04-07-roundtrip", "quant-analyst", _AGENT_OUTPUT)

        # Add arbiter report
        report = _make_fact_check_report("2026-04-07-roundtrip")
        manager.add_arbiter_report("2026-04-07-roundtrip", report)

        # Read back
        state = manager.read_debate("2026-04-07-roundtrip")
        assert state.debate_id == "2026-04-07-roundtrip"
        assert state.topic == "Roundtrip test topic"
        assert state.status == DebateStatus.OPEN
        assert state.agents == ["quant-analyst", "risk-officer"]
        assert state.arbiter_report is not None
        assert state.arbiter_report.debate_id == "2026-04-07-roundtrip"
        assert len(state.arbiter_report.results) == 1
        assert state.arbiter_report.results[0].verdict == ClaimVerdict.VERIFIED
