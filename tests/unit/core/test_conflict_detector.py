"""Tests for ConflictDetector -- rule-based conflict detection between agent outputs.

All tests are deterministic (no LLM calls). Uses synthetic AgentOutput fixtures.

Confidence delta rule: delta = abs(max_conf_a - max_conf_b). If <= 0.15, conflict
is NOT produced. Tests that expect conflicts use delta > 0.15 (e.g. 0.90 vs 0.65 = 0.25).
Test 5 specifically tests the filter using delta = 0.10.
"""

from __future__ import annotations

from datetime import UTC, datetime

from finalayze.core.schemas import (
    AgentOutput,
    Claim,
    ConflictSeverity,
    ConflictType,
    FileLineSource,
    MetricSource,
)
from finalayze.orchestration.conflict_detector import ConflictDetector

# ─── Fixtures / helpers ────────────────────────────────────────────────────────

_NOW = datetime(2026, 4, 12, tzinfo=UTC)

# Standard confidence values used to ensure delta > 0.15 (0.25 delta)
_CONF_HIGH = 0.90
_CONF_LOW = 0.65


def _make_metric_claim(
    statement: str,
    metric_name: str,
    value: float,
    iteration: str,
    confidence: float,
) -> Claim:
    return Claim(
        statement=statement,
        source=MetricSource(metric_name=metric_name, value=value, iteration=iteration),
        confidence=confidence,
    )


def _make_file_claim(
    statement: str,
    path: str,
    line: int,
    excerpt: str,
    confidence: float,
) -> Claim:
    return Claim(
        statement=statement,
        source=FileLineSource(path=path, line=line, excerpt=excerpt),
        confidence=confidence,
    )


def _make_agent_output(
    agent_name: str,
    recommendation: str,
    claims: list[Claim],
    timestamp: datetime = _NOW,
) -> AgentOutput:
    return AgentOutput(
        agent_name=agent_name,
        recommendation=recommendation,
        claims=claims,
        timestamp=timestamp,
    )


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestDirectionConflict:
    """Test 1: BUY vs SELL direction conflict returns DIRECTION/CRITICAL."""

    def test_detect_direction_conflict(self) -> None:
        """BUY vs SELL on same topic triggers CRITICAL direction conflict.

        Confidence delta = abs(0.90 - 0.65) = 0.25 > 0.15 => not filtered.
        """
        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER with high confidence based on momentum signals",
            [
                _make_metric_claim(
                    "dual_momentum PF=1.29",
                    "profit_factor",
                    1.29,
                    "2026-04-05-adx-routing",
                    _CONF_HIGH,
                )
            ],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "SELL SBER due to elevated volatility and circuit breaker concerns",
            [
                _make_metric_claim(
                    "SBER volatility ATR=2.5x",
                    "atr_multiplier",
                    2.5,
                    "2026-04-05-adx-routing",
                    _CONF_LOW,
                )
            ],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b])

        assert len(reports) >= 1
        direction_reports = [r for r in reports if r.conflict_type == ConflictType.DIRECTION]
        assert len(direction_reports) == 1
        report = direction_reports[0]
        assert report.severity == ConflictSeverity.CRITICAL
        assert "quant-analyst" in report.agent_names
        assert "risk-officer" in report.agent_names
        assert len(report.involved_claims) >= 2

    def test_detect_no_conflict_same_direction(self) -> None:
        """No conflict when both agents have same direction (both BUY)."""
        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER strong momentum signal",
            [_make_metric_claim("PF=1.29", "profit_factor", 1.29, "iter-1", _CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "BUY SBER risk is acceptable",
            [_make_metric_claim("max_dd=5%", "max_drawdown", 0.05, "iter-1", _CONF_LOW)],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b])

        direction_reports = [r for r in reports if r.conflict_type == ConflictType.DIRECTION]
        assert len(direction_reports) == 0


class TestMetricConflict:
    """Tests 2-3: Metric divergence detection and severity scoring."""

    def test_detect_metric_conflict_above_15_pct(self) -> None:
        """Test 2: Metric with >15% relative divergence triggers METRIC conflict.

        va=1.20, vb=1.00 => divergence = 0.20/1.20 = 16.7% > 15%.
        Confidence delta = abs(0.90 - 0.65) = 0.25 > 0.15 => not filtered.
        """
        output_a = _make_agent_output(
            "quant-analyst",
            "ENABLE dual_momentum strategy",
            [
                _make_metric_claim(
                    "profit_factor=1.20",
                    "profit_factor",
                    1.20,
                    "2026-04-05-adx-routing",
                    _CONF_HIGH,
                )
            ],
        )
        output_b = _make_agent_output(
            "ml-engineer",
            "DISABLE dual_momentum strategy",
            [
                _make_metric_claim(
                    "profit_factor=1.00",
                    "profit_factor",
                    1.00,
                    "2026-04-05-adx-routing",
                    _CONF_LOW,
                )
            ],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b])

        metric_reports = [r for r in reports if r.conflict_type == ConflictType.METRIC]
        assert len(metric_reports) >= 1

    def test_metric_conflict_high_severity_above_30_pct(self) -> None:
        """Test 3a: >30% divergence => HIGH severity.

        va=1.50, vb=1.00 => 0.50/1.50 = 33.3% > 30%.
        Confidence delta = abs(0.90 - 0.65) = 0.25 > 0.15 => not filtered.
        """
        output_a = _make_agent_output(
            "quant-analyst",
            "ENABLE strategy",
            [_make_metric_claim("pf=1.50", "profit_factor", 1.50, "iter-1", _CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "ml-engineer",
            "DISABLE strategy",
            [_make_metric_claim("pf=1.00", "profit_factor", 1.00, "iter-1", _CONF_LOW)],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b])

        metric_reports = [r for r in reports if r.conflict_type == ConflictType.METRIC]
        assert len(metric_reports) >= 1
        assert metric_reports[0].severity == ConflictSeverity.HIGH

    def test_metric_conflict_low_severity_between_15_and_30_pct(self) -> None:
        """Test 3b: 15-30% divergence => LOW severity.

        va=1.20, vb=1.00 => 0.20/1.20 = 16.7% in (15%, 30%].
        Confidence delta = abs(0.90 - 0.65) = 0.25 > 0.15 => not filtered.
        """
        output_a = _make_agent_output(
            "quant-analyst",
            "ENABLE strategy",
            [_make_metric_claim("pf=1.20", "profit_factor", 1.20, "iter-1", _CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "ml-engineer",
            "DISABLE strategy",
            [_make_metric_claim("pf=1.00", "profit_factor", 1.00, "iter-1", _CONF_LOW)],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b])

        metric_reports = [r for r in reports if r.conflict_type == ConflictType.METRIC]
        assert len(metric_reports) >= 1
        assert metric_reports[0].severity == ConflictSeverity.LOW

    def test_no_metric_conflict_same_iteration_no_divergence(self) -> None:
        """No metric conflict when divergence <= 15%.

        va=1.00, vb=1.10 => 0.10/1.10 = 9% <= 15% => no conflict.
        """
        output_a = _make_agent_output(
            "quant-analyst",
            "ENABLE strategy",
            [_make_metric_claim("pf=1.00", "profit_factor", 1.00, "iter-1", _CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "ml-engineer",
            "ENABLE strategy",
            [_make_metric_claim("pf=1.10", "profit_factor", 1.10, "iter-1", _CONF_LOW)],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b])

        metric_reports = [r for r in reports if r.conflict_type == ConflictType.METRIC]
        assert len(metric_reports) == 0

    def test_no_metric_conflict_different_iterations(self) -> None:
        """No metric conflict when metric_name matches but iterations differ."""
        output_a = _make_agent_output(
            "quant-analyst",
            "ENABLE strategy",
            [_make_metric_claim("pf=1.50", "profit_factor", 1.50, "iter-1", _CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "ml-engineer",
            "DISABLE strategy",
            [_make_metric_claim("pf=1.00", "profit_factor", 1.00, "iter-2", _CONF_LOW)],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b])

        metric_reports = [r for r in reports if r.conflict_type == ConflictType.METRIC]
        assert len(metric_reports) == 0


class TestStatementConflict:
    """Test 4: Statement similarity + divergent recommendations => STATEMENT/LOW."""

    def test_detect_statement_conflict(self) -> None:
        """Test 4: Similar statements + opposite recommendations => STATEMENT conflict.

        Confidence delta = abs(0.90 - 0.65) = 0.25 > 0.15 => not filtered.
        """
        common_text = (
            "dual_momentum strategy has profit factor above 1.20 in the us_tech segment backtest"
        )
        output_a = _make_agent_output(
            "quant-analyst",
            "ENABLE dual_momentum on us_tech -- metrics confirm outperformance",
            [
                _make_file_claim(
                    common_text,
                    "src/strategies/momentum.py",
                    42,
                    "pf=1.29",
                    _CONF_HIGH,
                )
            ],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "DISABLE dual_momentum on us_tech -- risk limits exceeded",
            [_make_file_claim(common_text, "src/strategies/momentum.py", 42, "pf=1.29", _CONF_LOW)],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b])

        statement_reports = [r for r in reports if r.conflict_type == ConflictType.STATEMENT]
        assert len(statement_reports) >= 1
        assert statement_reports[0].severity == ConflictSeverity.LOW


class TestConfidenceDeltaFilter:
    """Test 5: Confidence delta <= 0.15 should NOT produce conflicts."""

    def test_confidence_delta_filter_suppresses_conflict(self) -> None:
        """Test 5: BUY vs SELL with delta <= 0.15 => no conflict produced.

        Confidence delta = abs(0.80 - 0.70) = 0.10 <= 0.15 => filtered out.
        """
        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER",
            [_make_metric_claim("pf=1.29", "profit_factor", 1.29, "iter-1", 0.80)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "SELL SBER",
            [_make_metric_claim("max_dd=10%", "max_drawdown", 0.10, "iter-1", 0.70)],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b])

        # delta = abs(0.80 - 0.70) = 0.10 <= 0.15 => suppressed
        direction_reports = [r for r in reports if r.conflict_type == ConflictType.DIRECTION]
        assert len(direction_reports) == 0


class TestDeduplication:
    """Test 6: Dedup suppresses duplicate conflicts within same session."""

    def test_dedup_same_conflict_detected_only_once(self) -> None:
        """Test 6: Calling detect() twice with same inputs returns conflict only on first call.

        Confidence delta = abs(0.90 - 0.65) = 0.25 > 0.15 => conflict fires on first call.
        """
        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER based on momentum",
            [_make_metric_claim("pf=1.29", "profit_factor", 1.29, "iter-1", _CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "SELL SBER due to volatility",
            [_make_metric_claim("atm=2.5", "atr_multiplier", 2.5, "iter-1", _CONF_LOW)],
        )

        detector = ConflictDetector()
        first_result = detector.detect([output_a, output_b])
        second_result = detector.detect([output_a, output_b])

        # First call: conflict detected
        first_direction = [r for r in first_result if r.conflict_type == ConflictType.DIRECTION]
        assert len(first_direction) >= 1

        # Second call: same key already in dedup store => suppressed
        second_direction = [r for r in second_result if r.conflict_type == ConflictType.DIRECTION]
        assert len(second_direction) == 0


class TestPairwiseComparisons:
    """Test 7: Three outputs produce 3 pairwise comparisons."""

    def test_three_outputs_pairwise_comparisons(self) -> None:
        """Test 7: Three agents produce conflicts from AB and AC pairs (BC is same direction).

        a=BUY, b=SELL, c=SELL.
        Pair (a,b): BUY vs SELL, delta=abs(0.90-0.65)=0.25 > 0.15 => DIRECTION conflict.
        Pair (a,c): BUY vs SELL, delta=abs(0.90-0.65)=0.25 > 0.15 => DIRECTION conflict.
        Pair (b,c): SELL vs SELL => no direction conflict.
        """
        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER",
            [_make_metric_claim("pf=1.29", "profit_factor", 1.29, "iter-1", _CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "SELL SBER",
            [_make_metric_claim("max_dd=15%", "max_drawdown", 0.15, "iter-1", _CONF_LOW)],
        )
        output_c = _make_agent_output(
            "ml-engineer",
            "SELL SBER model predicts decline",
            [_make_metric_claim("acc=0.60", "accuracy", 0.60, "iter-1", _CONF_LOW)],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b, output_c])

        # Both (a,b) and (a,c) should produce direction conflicts
        direction_reports = [r for r in reports if r.conflict_type == ConflictType.DIRECTION]
        assert len(direction_reports) >= 2

        # Verify agents involved in conflicts
        agent_pairs = [frozenset(r.agent_names) for r in direction_reports]
        assert frozenset({"quant-analyst", "risk-officer"}) in agent_pairs
        assert frozenset({"quant-analyst", "ml-engineer"}) in agent_pairs


class TestNoConflicts:
    """Test 8: Completely compatible outputs return empty list."""

    def test_no_conflicts_compatible_outputs(self) -> None:
        """Test 8: Outputs without direction/metric/statement contradictions return []."""
        output_a = _make_agent_output(
            "quant-analyst",
            "ENABLE dual_momentum strategy",
            [_make_metric_claim("pf=1.29", "profit_factor", 1.29, "iter-1", _CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "ENABLE dual_momentum strategy -- risk acceptable",
            [_make_metric_claim("max_dd=5%", "max_drawdown", 0.05, "iter-1", _CONF_LOW)],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b])

        assert reports == []


class TestMetricDivergenceDenominator:
    """Test 9: Metric divergence uses max(va, vb) as denominator."""

    def test_divergence_uses_max_denominator(self) -> None:
        """Test 9: Relative divergence = abs(va-vb) / max(|va|, |vb|).

        va=2.0, vb=1.0 => 1.0/2.0 = 50% > 30% => HIGH severity.
        Confidence delta = abs(0.90 - 0.65) = 0.25 > 0.15 => not filtered.
        """
        output_a = _make_agent_output(
            "quant-analyst",
            "ENABLE strategy",
            [_make_metric_claim("sharpe=2.0", "sharpe_ratio", 2.0, "iter-1", _CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "ml-engineer",
            "DISABLE strategy",
            [_make_metric_claim("sharpe=1.0", "sharpe_ratio", 1.0, "iter-1", _CONF_LOW)],
        )

        detector = ConflictDetector()
        reports = detector.detect([output_a, output_b])

        metric_reports = [r for r in reports if r.conflict_type == ConflictType.METRIC]
        assert len(metric_reports) >= 1
        assert metric_reports[0].severity == ConflictSeverity.HIGH

    def test_divergence_symmetric_regardless_of_order(self) -> None:
        """Divergence is symmetric: swapping (va, vb) gives same severity.

        Both arrangements: abs(2.0-1.0)/max(2.0,1.0) = 50% => HIGH.
        """
        output_a1 = _make_agent_output(
            "agent-a",
            "ENABLE strategy",
            [_make_metric_claim("sharpe=1.0", "sharpe_ratio", 1.0, "iter-1", _CONF_HIGH)],
        )
        output_b1 = _make_agent_output(
            "agent-b",
            "DISABLE strategy",
            [_make_metric_claim("sharpe=2.0", "sharpe_ratio", 2.0, "iter-1", _CONF_LOW)],
        )

        detector1 = ConflictDetector()
        reports1 = detector1.detect([output_a1, output_b1])

        output_a2 = _make_agent_output(
            "agent-a",
            "ENABLE strategy",
            [_make_metric_claim("sharpe=2.0", "sharpe_ratio", 2.0, "iter-1", _CONF_HIGH)],
        )
        output_b2 = _make_agent_output(
            "agent-b",
            "DISABLE strategy",
            [_make_metric_claim("sharpe=1.0", "sharpe_ratio", 1.0, "iter-1", _CONF_LOW)],
        )

        detector2 = ConflictDetector()
        reports2 = detector2.detect([output_a2, output_b2])

        metric1 = [r for r in reports1 if r.conflict_type == ConflictType.METRIC]
        metric2 = [r for r in reports2 if r.conflict_type == ConflictType.METRIC]
        assert len(metric1) == len(metric2)
        if metric1 and metric2:
            assert metric1[0].severity == metric2[0].severity


class TestReset:
    """Test 10: reset() clears dedup store."""

    def test_reset_clears_dedup_store(self) -> None:
        """Test 10: After reset(), previously seen conflicts are detected again.

        Confidence delta = abs(0.90 - 0.65) = 0.25 > 0.15 => conflict fires.
        """
        output_a = _make_agent_output(
            "quant-analyst",
            "BUY SBER",
            [_make_metric_claim("pf=1.29", "profit_factor", 1.29, "iter-1", _CONF_HIGH)],
        )
        output_b = _make_agent_output(
            "risk-officer",
            "SELL SBER",
            [_make_metric_claim("atm=2.5", "atr_multiplier", 2.5, "iter-1", _CONF_LOW)],
        )

        detector = ConflictDetector()

        # First call: conflict detected and recorded in dedup store
        first_result = detector.detect([output_a, output_b])
        first_direction = [r for r in first_result if r.conflict_type == ConflictType.DIRECTION]
        assert len(first_direction) >= 1

        # Second call without reset: dedup suppresses
        second_result = detector.detect([output_a, output_b])
        second_direction = [r for r in second_result if r.conflict_type == ConflictType.DIRECTION]
        assert len(second_direction) == 0

        # After reset: dedup store cleared, conflict fires again
        detector.reset()
        third_result = detector.detect([output_a, output_b])
        third_direction = [r for r in third_result if r.conflict_type == ConflictType.DIRECTION]
        assert len(third_direction) >= 1
