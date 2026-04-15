"""Deterministic, rule-based ConflictDetector (no LLM in hot path).

Detects direction, metric, and statement conflicts between agent outputs.
Deduplicates within session via SHA-256 keying.

Layer 5 -- Orchestration. Imports from Layer 0 (core/schemas.py) only.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from datetime import UTC, datetime
from itertools import combinations

import structlog

from finalayze.core.schemas import (
    AgentOutput,
    Claim,
    ConflictReport,
    ConflictSeverity,
    ConflictType,
    MetricSource,
)

_log = structlog.get_logger(__name__)

# ─── Thresholds (from CONTEXT.md locked decisions) ─────────────────────────────

_DIRECTION_BUY: frozenset[str] = frozenset({"BUY", "LONG", "ENABLE", "INCREASE"})
_DIRECTION_SELL: frozenset[str] = frozenset({"SELL", "SHORT", "DISABLE", "DECREASE"})

_MIN_CONFIDENCE_DELTA = 0.15
_METRIC_CONTRADICTION_THRESHOLD = 0.15
_METRIC_HIGH_SEVERITY_THRESHOLD = 0.30
_STATEMENT_SIMILARITY_THRESHOLD = 0.85

# Regex to extract an uppercase ticker or snake_case identifier from a statement
_TOPIC_RE = re.compile(r"\b([A-Z]{2,}|[a-z_]+_[a-z_]+)\b")


class ConflictDetector:
    """Deterministic rule-based conflict detector for agent output pairs.

    Not thread-safe -- intended for single-threaded orchestrator use.
    Session-scoped dedup: call reset() between orchestrator cycles.
    """

    def __init__(self) -> None:
        self._seen_conflicts: set[str] = set()

    # ─── Public API ────────────────────────────────────────────────────────────

    def detect(self, outputs: list[AgentOutput]) -> list[ConflictReport]:
        """Compare all pairs of outputs and return new (not-yet-seen) conflicts.

        Args:
            outputs: List of agent outputs to compare pairwise.

        Returns:
            List of ConflictReport objects for newly detected conflicts.
            Conflicts seen in previous detect() calls within the same session
            are suppressed.
        """
        results: list[ConflictReport] = []
        for a, b in combinations(outputs, 2):
            results.extend(self._compare_pair(a, b))
        return results

    def reset(self) -> None:
        """Clear the session-scoped dedup store.

        Call this between orchestrator cycles (e.g. weekly/daily agent runs)
        to allow previously seen conflicts to be re-detected.
        """
        self._seen_conflicts.clear()

    # ─── Pair comparison ───────────────────────────────────────────────────────

    def _compare_pair(self, a: AgentOutput, b: AgentOutput) -> list[ConflictReport]:
        """Run all conflict checks on a single pair of agent outputs."""
        reports: list[ConflictReport] = []

        direction = self._check_direction(a, b)
        if direction is not None:
            reports.append(direction)

        reports.extend(self._check_metrics(a, b))
        reports.extend(self._check_statements(a, b))

        return reports

    # ─── Direction conflict ────────────────────────────────────────────────────

    def _check_direction(self, a: AgentOutput, b: AgentOutput) -> ConflictReport | None:
        """Return DIRECTION/CRITICAL if one recommends BUY and other SELL.

        Confidence delta filter: if |max_conf_a - max_conf_b| <= 0.15, skip.
        Deduplicates by SHA-256 key within session.
        """
        words_a = set(re.findall(r"\b[A-Z]{2,}\b", a.recommendation.upper()))
        words_b = set(re.findall(r"\b[A-Z]{2,}\b", b.recommendation.upper()))

        a_is_buy = bool(words_a & _DIRECTION_BUY)
        a_is_sell = bool(words_a & _DIRECTION_SELL)
        b_is_buy = bool(words_b & _DIRECTION_BUY)
        b_is_sell = bool(words_b & _DIRECTION_SELL)

        conflict = (a_is_buy and b_is_sell) or (a_is_sell and b_is_buy)
        if not conflict:
            return None

        # Confidence delta filter
        if self._should_filter_by_confidence(a, b):
            return None

        # Extract topics for dedup key
        topics_a = [self._extract_topic(c.statement) for c in a.claims]
        topics_b = [self._extract_topic(c.statement) for c in b.claims]
        topics = topics_a + topics_b

        key = self._dedup_key([a.agent_name, b.agent_name], topics, ConflictType.DIRECTION)
        if key in self._seen_conflicts:
            return None
        self._seen_conflicts.add(key)

        # Pick one representative claim from each agent
        claims = [a.claims[0], b.claims[0]]
        delta = self._confidence_delta(a, b)

        _log.info(
            "conflict.direction",
            agents=[a.agent_name, b.agent_name],
            key=key[:16],
        )

        return ConflictReport(
            conflict_id=key,
            conflict_type=ConflictType.DIRECTION,
            severity=ConflictSeverity.CRITICAL,
            involved_claims=claims,
            agent_names=[a.agent_name, b.agent_name],
            detected_at=datetime.now(UTC),
            confidence_delta=delta,
        )

    # ─── Metric conflict ───────────────────────────────────────────────────────

    def _check_metrics(self, a: AgentOutput, b: AgentOutput) -> list[ConflictReport]:
        """Return METRIC conflicts for claims with same metric_name+iteration but divergent values.

        Severity:
            HIGH  -- divergence > 30%
            LOW   -- divergence in (15%, 30%]
        """
        reports: list[ConflictReport] = []

        # Build MetricSource index for agent b: (metric_name, iteration) -> (claim, value)
        b_metrics: dict[tuple[str, str], tuple[Claim, float]] = {}
        for claim in b.claims:
            if isinstance(claim.source, MetricSource):
                key = (claim.source.metric_name, claim.source.iteration)
                b_metrics[key] = (claim, claim.source.value)

        for claim_a in a.claims:
            if not isinstance(claim_a.source, MetricSource):
                continue
            key = (claim_a.source.metric_name, claim_a.source.iteration)
            if key not in b_metrics:
                continue

            claim_b, vb = b_metrics[key]
            va = claim_a.source.value

            denom = max(abs(va), abs(vb))
            if denom == 0.0:
                continue

            divergence = abs(va - vb) / denom
            if divergence <= _METRIC_CONTRADICTION_THRESHOLD:
                continue

            # Confidence delta filter
            if self._should_filter_by_confidence(a, b):
                continue

            # Severity
            severity = (
                ConflictSeverity.HIGH
                if divergence > _METRIC_HIGH_SEVERITY_THRESHOLD
                else ConflictSeverity.LOW
            )

            # Dedup
            topics = [claim_a.source.metric_name, claim_a.source.iteration]
            dedup_key = self._dedup_key([a.agent_name, b.agent_name], topics, ConflictType.METRIC)
            if dedup_key in self._seen_conflicts:
                continue
            self._seen_conflicts.add(dedup_key)

            delta = self._confidence_delta(a, b)

            _log.info(
                "conflict.metric",
                agents=[a.agent_name, b.agent_name],
                metric=claim_a.source.metric_name,
                divergence=round(divergence, 3),
                severity=severity,
            )

            reports.append(
                ConflictReport(
                    conflict_id=dedup_key,
                    conflict_type=ConflictType.METRIC,
                    severity=severity,
                    involved_claims=[claim_a, claim_b],
                    agent_names=[a.agent_name, b.agent_name],
                    detected_at=datetime.now(UTC),
                    confidence_delta=delta,
                )
            )

        return reports

    # ─── Statement conflict ────────────────────────────────────────────────────

    def _check_statements(self, a: AgentOutput, b: AgentOutput) -> list[ConflictReport]:
        """Return STATEMENT/LOW conflicts where claims are similar but recommendations diverge.

        Similarity threshold: SequenceMatcher.ratio() > 0.85
        Recommendation divergence: one has BUY-keyword, other has SELL-keyword.
        """
        reports: list[ConflictReport] = []

        words_a = set(re.findall(r"\b[A-Z]{2,}\b", a.recommendation.upper()))
        words_b = set(re.findall(r"\b[A-Z]{2,}\b", b.recommendation.upper()))

        a_is_buy = bool(words_a & _DIRECTION_BUY)
        a_is_sell = bool(words_a & _DIRECTION_SELL)
        b_is_buy = bool(words_b & _DIRECTION_BUY)
        b_is_sell = bool(words_b & _DIRECTION_SELL)

        recommendations_diverge = (a_is_buy and b_is_sell) or (a_is_sell and b_is_buy)
        if not recommendations_diverge:
            return reports

        for claim_a in a.claims:
            for claim_b in b.claims:
                ratio = difflib.SequenceMatcher(None, claim_a.statement, claim_b.statement).ratio()
                if ratio <= _STATEMENT_SIMILARITY_THRESHOLD:
                    continue

                # Confidence delta filter
                if self._should_filter_by_confidence(a, b):
                    continue

                topics = [
                    self._extract_topic(claim_a.statement),
                    self._extract_topic(claim_b.statement),
                ]
                dedup_key = self._dedup_key(
                    [a.agent_name, b.agent_name], topics, ConflictType.STATEMENT
                )
                if dedup_key in self._seen_conflicts:
                    continue
                self._seen_conflicts.add(dedup_key)

                delta = self._confidence_delta(a, b)

                _log.info(
                    "conflict.statement",
                    agents=[a.agent_name, b.agent_name],
                    similarity=round(ratio, 3),
                )

                reports.append(
                    ConflictReport(
                        conflict_id=dedup_key,
                        conflict_type=ConflictType.STATEMENT,
                        severity=ConflictSeverity.LOW,
                        involved_claims=[claim_a, claim_b],
                        agent_names=[a.agent_name, b.agent_name],
                        detected_at=datetime.now(UTC),
                        confidence_delta=delta,
                    )
                )

        return reports

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _extract_topic(self, statement: str) -> str:
        """Extract primary topic token from a statement string.

        Returns first uppercase ticker (e.g. 'SBER') or first snake_case
        identifier found, lowercased. Falls back to first 20 chars.
        """
        match = _TOPIC_RE.search(statement)
        if match:
            return match.group(1).lower()
        return statement[:20].lower()

    def _dedup_key(
        self,
        agents: list[str],
        topics: list[str],
        conflict_type: ConflictType,
    ) -> str:
        """Compute SHA-256 dedup key from sorted agents, sorted topics, and conflict type."""
        raw = str(sorted(agents)) + str(sorted(topics)) + str(conflict_type)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _confidence_delta(self, a: AgentOutput, b: AgentOutput) -> float:
        """Compute absolute confidence delta between the maximum-confidence claims."""
        max_a = max(c.confidence for c in a.claims)
        max_b = max(c.confidence for c in b.claims)
        return abs(max_a - max_b)

    def _should_filter_by_confidence(self, a: AgentOutput, b: AgentOutput) -> bool:
        """Return True if confidence delta is too small to escalate (delta <= 0.15)."""
        return self._confidence_delta(a, b) <= _MIN_CONFIDENCE_DELTA
