"""Tests for debate protocol schemas (Layer 0)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timezone

import pytest
from pydantic import ValidationError

from finalayze.core.schemas import (
    AgentOutput,
    Claim,
    ClaimCheckResult,
    ClaimVerdict,
    ConflictReport,
    ConflictSeverity,
    ConflictType,
    DebateState,
    DebateStatus,
    FactCheckReport,
    FileLineSource,
    MetricSource,
)

_DT = datetime(2026, 4, 7, tzinfo=UTC)

_FILE_SOURCE = FileLineSource(
    kind="file",
    path="src/finalayze/strategies/combiner.py",
    line=142,
    excerpt="class StrategyCombiner",
)

_METRIC_SOURCE = MetricSource(
    kind="metric",
    metric_name="profit_factor",
    value=1.29,
    iteration="2026-04-05-adx-routing",
)

_CLAIM_FILE = Claim(
    statement="Combiner uses ADX routing",
    source=_FILE_SOURCE,
    confidence=0.9,
)

_CLAIM_METRIC = Claim(
    statement="PF is 1.29",
    source=_METRIC_SOURCE,
    confidence=0.85,
)


# ── FileLineSource ───────────────────────────────────────────────────────────


class TestFileLineSource:
    def test_file_line_source_valid(self) -> None:
        src = FileLineSource(
            kind="file",
            path="src/finalayze/strategies/combiner.py",
            line=142,
            excerpt="class StrategyCombiner",
        )
        assert src.kind == "file"
        assert src.path == "src/finalayze/strategies/combiner.py"
        assert src.line == 142
        assert src.excerpt == "class StrategyCombiner"


# ── MetricSource ─────────────────────────────────────────────────────────────


class TestMetricSource:
    def test_metric_source_valid(self) -> None:
        src = MetricSource(
            kind="metric",
            metric_name="profit_factor",
            value=1.29,
            iteration="2026-04-05-adx-routing",
        )
        assert src.kind == "metric"
        assert src.metric_name == "profit_factor"
        assert src.value == pytest.approx(1.29)
        assert src.iteration == "2026-04-05-adx-routing"


# ── Claim ────────────────────────────────────────────────────────────────────


class TestClaim:
    def test_claim_with_file_source(self) -> None:
        claim = Claim(
            statement="Combiner uses ADX routing",
            source=_FILE_SOURCE,
            confidence=0.9,
        )
        assert claim.statement == "Combiner uses ADX routing"
        assert isinstance(claim.source, FileLineSource)

    def test_claim_with_metric_source(self) -> None:
        claim = Claim(
            statement="PF is 1.29",
            source=_METRIC_SOURCE,
            confidence=0.85,
        )
        assert claim.statement == "PF is 1.29"
        assert isinstance(claim.source, MetricSource)

    def test_claim_confidence_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            Claim(statement="x", source=_FILE_SOURCE, confidence=-0.1)

    def test_claim_confidence_above_one(self) -> None:
        with pytest.raises(ValidationError):
            Claim(statement="x", source=_FILE_SOURCE, confidence=1.1)

    def test_claim_confidence_boundary_zero(self) -> None:
        claim = Claim(statement="x", source=_FILE_SOURCE, confidence=0.0)
        assert claim.confidence == pytest.approx(0.0)

    def test_claim_confidence_boundary_one(self) -> None:
        claim = Claim(statement="x", source=_FILE_SOURCE, confidence=1.0)
        assert claim.confidence == pytest.approx(1.0)

    def test_claim_source_discriminator(self) -> None:
        file_claim = Claim(
            statement="x",
            source={"kind": "file", "path": "x.py", "line": 1, "excerpt": "y"},
            confidence=0.5,
        )
        assert isinstance(file_claim.source, FileLineSource)

        metric_claim = Claim(
            statement="x",
            source={
                "kind": "metric",
                "metric_name": "pf",
                "value": 1.5,
                "iteration": "2026-04-01",
            },
            confidence=0.5,
        )
        assert isinstance(metric_claim.source, MetricSource)


# ── AgentOutput ──────────────────────────────────────────────────────────────


class TestAgentOutput:
    def test_agent_output_valid(self) -> None:
        out = AgentOutput(
            agent_name="quant-analyst",
            recommendation="Enable dual_momentum on ru_blue_chips",
            claims=[_CLAIM_FILE],
            timestamp=_DT,
        )
        assert out.agent_name == "quant-analyst"
        assert len(out.claims) == 1

    def test_agent_output_empty_claims(self) -> None:
        with pytest.raises(ValidationError):
            AgentOutput(
                agent_name="quant-analyst",
                recommendation="Enable dual_momentum",
                claims=[],
                timestamp=_DT,
            )


# ── ClaimVerdict ─────────────────────────────────────────────────────────────


class TestClaimVerdict:
    def test_claim_verdict_enum(self) -> None:
        assert ClaimVerdict.VERIFIED == "verified"
        assert ClaimVerdict.CONTRADICTED == "contradicted"
        assert ClaimVerdict.UNTESTABLE == "untestable"


# ── ClaimCheckResult ─────────────────────────────────────────────────────────


class TestClaimCheckResult:
    def test_claim_check_result(self) -> None:
        result = ClaimCheckResult(
            claim=_CLAIM_FILE,
            verdict=ClaimVerdict.VERIFIED,
            evidence="Found class StrategyCombiner at line 142 with ADX routing logic.",
        )
        assert result.verdict == ClaimVerdict.VERIFIED
        assert "StrategyCombiner" in result.evidence


# ── FactCheckReport ──────────────────────────────────────────────────────────


class TestFactCheckReport:
    def _make_report(self, verdicts: list[ClaimVerdict]) -> FactCheckReport:
        results = [
            ClaimCheckResult(
                claim=_CLAIM_FILE,
                verdict=v,
                evidence="test evidence",
            )
            for v in verdicts
        ]
        return FactCheckReport(
            debate_id="2026-04-07-test",
            arbiter_timestamp=_DT,
            results=results,
        )

    def test_fact_check_report_has_contradictions_true(self) -> None:
        report = self._make_report([ClaimVerdict.VERIFIED, ClaimVerdict.CONTRADICTED])
        assert report.has_contradictions is True

    def test_fact_check_report_has_contradictions_false(self) -> None:
        report = self._make_report([ClaimVerdict.VERIFIED, ClaimVerdict.VERIFIED])
        assert report.has_contradictions is False

    def test_fact_check_report_to_markdown(self) -> None:
        report = self._make_report(
            [
                ClaimVerdict.VERIFIED,
                ClaimVerdict.CONTRADICTED,
                ClaimVerdict.UNTESTABLE,
            ]
        )
        md = report.to_markdown()
        assert "## Verified" in md
        assert "## Contradicted" in md
        assert "## Untestable" in md


# ── DebateStatus ─────────────────────────────────────────────────────────────


class TestDebateStatus:
    def test_debate_state_status_enum(self) -> None:
        assert DebateStatus.OPEN == "open"
        assert DebateStatus.RESOLVED == "resolved"
        assert DebateStatus.ESCALATED == "escalated"


# ── DebateState ──────────────────────────────────────────────────────────────


class TestDebateState:
    def test_debate_state_valid(self) -> None:
        state = DebateState(
            debate_id="2026-04-07-test",
            topic="Should we enable dual_momentum on ru_blue_chips?",
            status="open",
            created="2026-04-07",
            agents=["quant-analyst", "risk-officer"],
        )
        assert state.debate_id == "2026-04-07-test"
        assert state.status == DebateStatus.OPEN

    def test_debate_state_escalation(self) -> None:
        with pytest.raises(ValidationError):
            DebateState(
                debate_id="2026-04-07-test",
                topic="test topic",
                status=DebateStatus.ESCALATED,
                created="2026-04-07",
                agents=["a", "b"],
                experiment_id=None,
            )

        state = DebateState(
            debate_id="2026-04-07-test",
            topic="test topic",
            status=DebateStatus.ESCALATED,
            created="2026-04-07",
            agents=["a", "b"],
            experiment_id="EXP-001",
        )
        assert state.experiment_id == "EXP-001"


# ── ConflictType ─────────────────────────────────────────────────────────────


class TestConflictType:
    def test_conflict_type_enum_values(self) -> None:
        assert ConflictType.DIRECTION == "direction"
        assert ConflictType.METRIC == "metric"
        assert ConflictType.STATEMENT == "statement"


# ── ConflictSeverity ─────────────────────────────────────────────────────────


class TestConflictSeverity:
    def test_conflict_severity_enum_values(self) -> None:
        assert ConflictSeverity.CRITICAL == "critical"
        assert ConflictSeverity.HIGH == "high"
        assert ConflictSeverity.LOW == "low"


# ── ConflictReport ───────────────────────────────────────────────────────────


class TestConflictReport:
    def _make_report(
        self,
        *,
        involved_claims: list[Claim] | None = None,
        agent_names: list[str] | None = None,
        confidence_delta: float | None = None,
    ) -> ConflictReport:
        if involved_claims is None:
            involved_claims = [_CLAIM_FILE, _CLAIM_METRIC]
        if agent_names is None:
            agent_names = ["quant-analyst", "risk-officer"]
        return ConflictReport(
            conflict_id="abc123def456",
            conflict_type=ConflictType.DIRECTION,
            severity=ConflictSeverity.HIGH,
            involved_claims=involved_claims,
            agent_names=agent_names,
            detected_at=_DT,
            confidence_delta=confidence_delta,
        )

    def test_conflict_report_valid(self) -> None:
        report = self._make_report()
        assert report.conflict_id == "abc123def456"
        assert report.conflict_type == ConflictType.DIRECTION
        assert report.severity == ConflictSeverity.HIGH
        assert len(report.involved_claims) == 2
        assert len(report.agent_names) == 2
        assert report.detected_at == _DT
        assert report.confidence_delta is None

    def test_conflict_report_rejects_fewer_than_two_claims(self) -> None:
        with pytest.raises(ValidationError):
            self._make_report(involved_claims=[_CLAIM_FILE])

    def test_conflict_report_rejects_fewer_than_two_agent_names(self) -> None:
        with pytest.raises(ValidationError):
            self._make_report(agent_names=["quant-analyst"])

    def test_conflict_report_with_confidence_delta_none(self) -> None:
        report = self._make_report(confidence_delta=None)
        assert report.confidence_delta is None

    def test_conflict_report_with_confidence_delta_value(self) -> None:
        report = self._make_report(confidence_delta=0.25)
        assert report.confidence_delta == pytest.approx(0.25)

    def test_conflict_report_is_frozen(self) -> None:
        report = self._make_report()
        with pytest.raises(Exception, match=r".*"):  # noqa: B017, PT011
            report.conflict_id = "mutated"  # type: ignore[misc]


# ── FileLineSource snapshot_sha ──────────────────────────────────────────────


class TestFileLineSourceSnapshotSha:
    _SHA = hashlib.sha256(b"test content").hexdigest()

    def test_file_line_source_without_snapshot_sha_is_backward_compatible(self) -> None:
        """Constructing without snapshot_sha uses None default — backward compatible."""
        src = FileLineSource(
            path="src/finalayze/strategies/combiner.py",
            line=142,
            excerpt="class StrategyCombiner",
        )
        assert src.snapshot_sha is None

    def test_file_line_source_with_snapshot_sha(self) -> None:
        """Constructing with a SHA stores it correctly."""
        src = FileLineSource(
            path="src/finalayze/strategies/combiner.py",
            line=142,
            excerpt="class StrategyCombiner",
            snapshot_sha=self._SHA,
        )
        assert src.snapshot_sha == self._SHA

    def test_file_line_source_none_sha_serializes_without_error(self) -> None:
        """FileLineSource with snapshot_sha=None serializes via model_dump_json()."""
        src = FileLineSource(
            path="src/x.py",
            line=10,
            excerpt="foo",
            snapshot_sha=None,
        )
        json_str = src.model_dump_json()
        assert "snapshot_sha" in json_str

    def test_claim_with_file_line_source_snapshot_sha_validates(self) -> None:
        """Claim wrapping a FileLineSource with snapshot_sha still validates (regression)."""
        src = FileLineSource(
            path="src/finalayze/core/schemas.py",
            line=541,
            excerpt="class FileLineSource",
            snapshot_sha=self._SHA,
        )
        claim = Claim(
            statement="FileLineSource is a frozen Pydantic model",
            source=src,
            confidence=0.95,
        )
        assert isinstance(claim.source, FileLineSource)
        assert claim.source.snapshot_sha == self._SHA
