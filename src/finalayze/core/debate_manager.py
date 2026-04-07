"""DebateManager — CRUD operations for structured debate files (Layer 0).

Each debate is a markdown file with YAML frontmatter matching the
DebateState schema. The manager provides operations for creating,
reading, updating, and listing debates.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from finalayze.core.schemas import AgentOutput, DebateState, FactCheckReport

_FRONTMATTER_DELIM = "---"

# Expected number of parts after splitting content on frontmatter delimiters
# Structure: ["", yaml_text, body_text]
_EXPECTED_FRONTMATTER_PARTS = 3


class DebateManager:
    """Manages structured debate files in a directory.

    Each debate is a markdown file with YAML frontmatter matching the
    DebateState schema. The manager provides CRUD operations for
    creating, reading, updating, and listing debates.
    """

    def __init__(self, debates_dir: Path | None = None) -> None:
        self._dir = debates_dir or Path(".planning/debates")
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _debate_path(self, debate_id: str) -> Path:
        """Return path to the debate file."""
        return self._dir / f"{debate_id}.md"

    def _read_file(self, debate_id: str) -> tuple[dict, str]:  # type: ignore[type-arg]
        """Read debate file and return (frontmatter_dict, body_text).

        Raises:
            FileNotFoundError: if the debate file does not exist.
        """
        path = self._debate_path(debate_id)
        if not path.exists():
            msg = f"Debate file not found: {path}"
            raise FileNotFoundError(msg)

        content = path.read_text(encoding="utf-8")
        # Split on frontmatter delimiters: ---\n<yaml>\n---\n<body>
        parts = content.split(f"{_FRONTMATTER_DELIM}\n", maxsplit=2)
        if len(parts) >= _EXPECTED_FRONTMATTER_PARTS:
            # parts[0] = "" (before first ---), parts[1] = yaml, parts[2] = body
            frontmatter = yaml.safe_load(parts[1]) or {}
            body = parts[2]
        else:
            frontmatter = {}
            body = content

        return frontmatter, body

    def _write_file(self, debate_id: str, frontmatter: dict, body: str) -> None:  # type: ignore[type-arg]
        """Write debate file with YAML frontmatter + body."""
        path = self._debate_path(debate_id)
        yaml_text = yaml.dump(
            frontmatter,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        content = f"{_FRONTMATTER_DELIM}\n{yaml_text}{_FRONTMATTER_DELIM}\n{body}"
        path.write_text(content, encoding="utf-8")

    # ── Public API ────────────────────────────────────────────────────────────

    def create_debate(self, debate_id: str, topic: str, agents: list[str]) -> Path:
        """Create a new debate file with YAML frontmatter.

        Args:
            debate_id: Unique identifier for the debate (used as filename stem).
            topic: Human-readable debate topic.
            agents: List of agent names participating in the debate.

        Returns:
            Path to the created debate file.
        """
        today = datetime.now(tz=UTC).date().isoformat()
        frontmatter: dict = {  # type: ignore[type-arg]
            "debate_id": debate_id,
            "topic": topic,
            "status": "open",
            "created": today,
            "agents": agents,
            "arbiter_report": None,
            "resolution": None,
            "experiment_id": None,
        }
        body = f"# Debate: {topic}\n"
        self._write_file(debate_id, frontmatter, body)
        return self._debate_path(debate_id)

    def read_debate(self, debate_id: str) -> DebateState:
        """Read a debate file and return a DebateState.

        Args:
            debate_id: Unique identifier for the debate.

        Returns:
            DebateState parsed from the YAML frontmatter.

        Raises:
            FileNotFoundError: if the debate file does not exist.
        """
        from finalayze.core.schemas import DebateState, FactCheckReport  # noqa: PLC0415

        frontmatter, _ = self._read_file(debate_id)
        # If arbiter_report is stored as a dict, convert to FactCheckReport
        if isinstance(frontmatter.get("arbiter_report"), dict):
            frontmatter["arbiter_report"] = FactCheckReport(**frontmatter["arbiter_report"])
        return DebateState(**frontmatter)

    def resolve_debate(self, debate_id: str, resolution: str) -> None:
        """Set debate status to resolved and record the resolution.

        Args:
            debate_id: Unique identifier for the debate.
            resolution: Text describing the agreed resolution.
        """
        frontmatter, body = self._read_file(debate_id)
        frontmatter["status"] = "resolved"
        frontmatter["resolution"] = resolution
        self._write_file(debate_id, frontmatter, body)

    def escalate_debate(self, debate_id: str, experiment_id: str) -> None:
        """Set debate status to escalated and record the experiment ID.

        Args:
            debate_id: Unique identifier for the debate.
            experiment_id: Identifier for the experiment that will resolve the debate.
        """
        frontmatter, body = self._read_file(debate_id)
        frontmatter["status"] = "escalated"
        frontmatter["experiment_id"] = experiment_id
        self._write_file(debate_id, frontmatter, body)

    def list_debates(self) -> list[str]:
        """Return sorted list of debate IDs (filename stems) from the directory.

        Returns:
            Sorted list of debate IDs found in the debates directory.
        """
        return sorted(p.stem for p in self._dir.glob("*.md"))

    def add_agent_position(
        self, debate_id: str, agent_name: str, agent_output: AgentOutput
    ) -> None:
        """Append an agent's position section to the debate markdown body.

        Args:
            debate_id: Unique identifier for the debate.
            agent_name: Name of the agent providing the position.
            agent_output: Structured agent recommendation with claims.
        """
        frontmatter, body = self._read_file(debate_id)

        claims_lines = [
            f"- **{claim.statement}** (confidence: {claim.confidence:.2f})"
            for claim in agent_output.claims
        ]
        claims_text = "\n".join(claims_lines) if claims_lines else "_No claims provided._"
        position_section = (
            f"\n## {agent_name} Position\n\n"
            f"{agent_output.recommendation}\n\n"
            f"### Claims\n\n"
            f"{claims_text}\n"
        )
        self._write_file(debate_id, frontmatter, body + position_section)

    def add_arbiter_report(self, debate_id: str, report: FactCheckReport) -> None:
        """Update frontmatter with arbiter report and append a markdown section.

        Serializes the FactCheckReport to the YAML frontmatter (as a nested dict)
        and appends a human-readable fact-check section to the body.

        Args:
            debate_id: Unique identifier for the debate.
            report: Completed arbiter fact-check report.
        """
        frontmatter, body = self._read_file(debate_id)
        frontmatter["arbiter_report"] = report.model_dump(mode="json")

        arbiter_section = f"\n## Arbiter Fact-Check\n\n{report.to_markdown()}\n"
        self._write_file(debate_id, frontmatter, body + arbiter_section)
