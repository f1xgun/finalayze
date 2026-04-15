"""AgentOrchestrator — pipeline coordinator for multi-agent conflict resolution.

Coordinates the full conflict-to-debate pipeline:
  1. Collect AgentOutput objects from multiple analysis agents
  2. Detect conflicts via ConflictDetector (fresh instance per run)
  3. Group conflicts by agent pair and create debates via DebateManager
  4. On finalize: add arbiter report, create experiment if contradictions found

Layer 5 — Orchestration. Imports from Layers 0–5.
See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.

ORCH-01: Full conflict-to-debate pipeline without manual intervention.
ORCH-03: snapshot_sha on FileLineSource prevents false CONTRADICTED verdicts.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from finalayze.core.debate_manager import DebateManager
from finalayze.core.experiment_manager import ExperimentManager
from finalayze.core.schemas import FactCheckReport, SuccessCriteria
from finalayze.orchestration.conflict_detector import ConflictDetector

if TYPE_CHECKING:
    from finalayze.core.schemas import AgentOutput, ConflictReport

_log = structlog.get_logger(__name__)

# Default success criteria for experiments created from debates
_DEFAULT_SUCCESS_CRITERIA = SuccessCriteria(
    metric="profit_factor",
    threshold=1.1,
    operator=">=",
)

# Maximum topic length when deriving debate topic from claim statements
_MAX_TOPIC_LENGTH = 100


class AgentOrchestrator:
    """Pipeline coordinator: conflict detection → debate creation → experiment escalation.

    Usage:
        orch = AgentOrchestrator()
        debate_ids = orch.run([output_a, output_b, output_c])
        # ... arbiter runs ...
        exp_id = orch.finalize_debate(debate_id, arbiter_report)

    A fresh ConflictDetector is instantiated per run() call to avoid stale
    deduplication state across orchestrator cycles.
    """

    def __init__(
        self,
        debate_manager: DebateManager | None = None,
        experiment_manager: ExperimentManager | None = None,
        debates_dir: Path | None = None,
        experiments_dir: Path | None = None,
    ) -> None:
        """Initialize with optional dependency injection.

        Args:
            debate_manager: DebateManager instance (or None to create default).
            experiment_manager: ExperimentManager instance (or None to create default).
            debates_dir: Override debates directory (used when creating default managers).
            experiments_dir: Override experiments directory (used when creating default managers).
        """
        self._dm = debate_manager or DebateManager(debates_dir=debates_dir)
        self._em = experiment_manager or ExperimentManager(
            experiments_dir=experiments_dir,
            debates_dir=debates_dir,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, outputs: list[AgentOutput]) -> list[str]:
        """Run the conflict-to-debate pipeline on a list of agent outputs.

        Steps:
        1. Instantiate a fresh ConflictDetector (no stale dedup state)
        2. Detect conflicts across all output pairs
        3. Short-circuit if no conflicts
        4. Group conflicts by unique agent-pair sets
        5. Create a debate for each conflict group
        6. Record each involved agent's position in the debate
        7. Return list of created debate IDs

        Args:
            outputs: List of AgentOutput objects from multiple agents.

        Returns:
            List of debate_id strings for created debates (empty if no conflicts).
        """
        # Step 1 & 2: Fresh detector, no stale dedup (ORCH-01 pitfall 4)
        detector = ConflictDetector()
        conflicts = detector.detect(outputs)

        if not conflicts:
            _log.info("orchestrator.run.no_conflicts", agent_count=len(outputs))
            return []

        _log.info(
            "orchestrator.run.conflicts_detected",
            conflict_count=len(conflicts),
            agent_count=len(outputs),
        )

        # Step 4: Group conflicts by frozenset of agent names
        groups = self._group_conflicts_by_agents(conflicts)

        # Build a lookup from agent_name -> AgentOutput for position recording
        output_by_agent = {o.agent_name: o for o in outputs}

        debate_ids: list[str] = []

        for agent_set, group_conflicts in groups.items():
            agent_names = sorted(agent_set)
            debate_id = self._generate_debate_id(agent_names)
            topic = self._derive_topic(group_conflicts)

            # Step 5: Create debate
            self._dm.create_debate(debate_id, topic, agent_names)
            _log.info(
                "orchestrator.debate_created",
                debate_id=debate_id[:16],
                agents=agent_names,
            )

            # Step 6: Record each involved agent's position
            for agent_name in agent_names:
                if agent_name in output_by_agent:
                    self._dm.add_agent_position(
                        debate_id, agent_name, output_by_agent[agent_name]
                    )

            debate_ids.append(debate_id)

        return debate_ids

    def finalize_debate(self, debate_id: str, report: FactCheckReport) -> str | None:
        """Finalize a debate with an arbiter fact-check report.

        If the report contains contradictions, creates a linked experiment
        via ExperimentManager and returns the experiment_id.

        If no contradictions, resolves the debate and returns None.

        Args:
            debate_id: Unique identifier for the debate to finalize.
            report: Completed arbiter FactCheckReport.

        Returns:
            experiment_id string if contradictions found, None otherwise.
        """
        # Store arbiter report in debate file
        self._dm.add_arbiter_report(debate_id, report)

        if report.has_contradictions:
            # Build experiment from contradicted claims
            experiment_id = f"exp-{debate_id[:12]}"
            hypothesis = self._build_hypothesis(report)

            self._em.create_experiment(
                experiment_id,
                hypothesis,
                _DEFAULT_SUCCESS_CRITERIA,
                debate_id=debate_id,
            )

            _log.info(
                "orchestrator.experiment_created",
                debate_id=debate_id[:16],
                experiment_id=experiment_id,
            )
            return experiment_id
        # No contradictions — resolve debate
        self._dm.resolve_debate(
            debate_id,
            "No contradictions — claims verified by arbiter",
        )
        _log.info(
            "orchestrator.debate_resolved",
            debate_id=debate_id[:16],
        )
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _group_conflicts_by_agents(
        self, conflicts: list[ConflictReport]
    ) -> dict[frozenset[str], list[ConflictReport]]:
        """Group conflicts by the frozenset of involved agent names.

        Conflicts between the same pair of agents are grouped into one debate.
        Independent conflicts between different agent pairs produce separate debates.

        Args:
            conflicts: List of ConflictReport objects from ConflictDetector.

        Returns:
            Dict mapping frozenset(agent_names) -> list of conflicts for that group.
        """
        groups: dict[frozenset[str], list[ConflictReport]] = {}
        for conflict in conflicts:
            key = frozenset(conflict.agent_names)
            if key not in groups:
                groups[key] = []
            groups[key].append(conflict)
        return groups

    def _generate_debate_id(self, agent_names: list[str]) -> str:
        """Generate a deterministic-ish debate_id from agent names + ISO-minute timestamp.

        Uses SHA-256 of sorted(agent_names) + current UTC minute so repeated
        calls in the same minute with the same agents get the same ID (idempotent
        within a minute), but different minutes produce different IDs.

        Args:
            agent_names: Sorted list of agent names in the debate.

        Returns:
            64-char hex SHA-256 digest string.
        """
        minute_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M")
        raw = str(sorted(agent_names)) + minute_ts
        return hashlib.sha256(raw.encode()).hexdigest()

    def _derive_topic(self, conflicts: list[ConflictReport]) -> str:
        """Derive a human-readable debate topic from the first conflict's claims.

        Concatenates the statements of the first two involved claims, truncated
        to _MAX_TOPIC_LENGTH characters.

        Args:
            conflicts: List of ConflictReport objects (non-empty).

        Returns:
            Debate topic string (at most _MAX_TOPIC_LENGTH characters).
        """
        first = conflicts[0]
        parts = [c.statement for c in first.involved_claims[:2]]
        topic = " vs. ".join(parts)
        return topic[:_MAX_TOPIC_LENGTH]

    def _build_hypothesis(self, report: FactCheckReport) -> str:
        """Build a hypothesis string from contradicted claims in a FactCheckReport.

        Args:
            report: Arbiter FactCheckReport with at least one CONTRADICTED verdict.

        Returns:
            Human-readable hypothesis string for the experiment.
        """
        from finalayze.core.schemas import ClaimVerdict  # noqa: PLC0415

        contradicted = [
            r.claim.statement
            for r in report.results
            if r.verdict == ClaimVerdict.CONTRADICTED
        ]
        if contradicted:
            return f"Verify contradicted claims: {'; '.join(contradicted[:3])}"
        return f"Verify claims from debate {report.debate_id}"
