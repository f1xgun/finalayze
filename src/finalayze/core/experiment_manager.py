"""ExperimentManager -- CRUD operations for experiment registry files (Layer 0).

Each experiment is a markdown file with YAML frontmatter matching the
ExperimentState schema. The manager provides operations for creating,
reading, updating, listing, and recording verdicts for experiments.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import operator as op
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from finalayze.core.schemas import ExperimentResult, ExperimentState, SuccessCriteria

_FRONTMATTER_DELIM = "---"

# Expected number of parts after splitting content on frontmatter delimiters
# Structure: ["", yaml_text, body_text]
_EXPECTED_FRONTMATTER_PARTS = 3

# Relative miss within this band -> INCONCLUSIVE (not REJECTED)
_INCONCLUSIVE_BAND = 0.10


class ExperimentManager:
    """Manages experiment registry files in a directory.

    Each experiment is a markdown file with YAML frontmatter matching the
    ExperimentState schema. The manager provides CRUD operations, automated
    verdict computation, and debate linkage.
    """

    def __init__(
        self,
        experiments_dir: Path | None = None,
        debates_dir: Path | None = None,
    ) -> None:
        self._dir = experiments_dir or Path(".planning/experiments")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._debates_dir = debates_dir

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _experiment_path(self, experiment_id: str) -> Path:
        """Return path to the experiment file."""
        return self._dir / f"{experiment_id}.md"

    def _read_file(self, experiment_id: str) -> tuple[dict[str, Any], str]:
        """Read experiment file and return (frontmatter_dict, body_text).

        Raises:
            FileNotFoundError: if the experiment file does not exist.
        """
        path = self._experiment_path(experiment_id)
        if not path.exists():
            msg = f"Experiment file not found: {path}"
            raise FileNotFoundError(msg)

        content = path.read_text(encoding="utf-8")
        parts = content.split(f"{_FRONTMATTER_DELIM}\n", maxsplit=2)
        if len(parts) >= _EXPECTED_FRONTMATTER_PARTS:
            frontmatter: dict[str, Any] = yaml.safe_load(parts[1]) or {}
            body = parts[2]
        else:
            frontmatter = {}
            body = content

        return frontmatter, body

    def _write_file(
        self, experiment_id: str, frontmatter: dict[str, Any], body: str
    ) -> None:
        """Write experiment file with YAML frontmatter + body."""
        path = self._experiment_path(experiment_id)
        yaml_text = yaml.dump(
            frontmatter,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        content = f"{_FRONTMATTER_DELIM}\n{yaml_text}{_FRONTMATTER_DELIM}\n{body}"
        path.write_text(content, encoding="utf-8")

    # ── Public API ────────────────────────────────────────────────────────────

    def create_experiment(
        self,
        experiment_id: str,
        hypothesis: str,
        success_criteria: SuccessCriteria,
        debate_id: str | None = None,
        preset_overrides: dict[str, Any] | None = None,
    ) -> Path:
        """Create a new experiment file with YAML frontmatter.

        Args:
            experiment_id: Unique identifier (used as filename stem).
            hypothesis: Human-readable hypothesis being tested.
            success_criteria: Metric threshold that determines success.
            debate_id: Optional debate to link (calls DebateManager.escalate_debate).
            preset_overrides: Optional strategy preset overrides for the experiment.

        Returns:
            Path to the created experiment file.
        """
        today = datetime.now(tz=UTC).date().isoformat()
        frontmatter: dict[str, Any] = {
            "experiment_id": experiment_id,
            "hypothesis": hypothesis,
            "success_criteria": {
                "metric": success_criteria.metric,
                "threshold": success_criteria.threshold,
                "operator": success_criteria.operator,
            },
            "status": "pending",
            "created": today,
            "debate_id": debate_id,
            "results": [],
            "verdict": None,
            "reasoning": None,
            "preset_overrides": preset_overrides,
        }
        body = f"# Experiment: {hypothesis}\n"
        self._write_file(experiment_id, frontmatter, body)

        # Bidirectional link: escalate the debate to point to this experiment
        if debate_id is not None:
            from finalayze.core.debate_manager import DebateManager  # noqa: PLC0415

            dm = DebateManager(debates_dir=self._debates_dir)
            dm.escalate_debate(debate_id, experiment_id)

        return self._experiment_path(experiment_id)

    def read_experiment(self, experiment_id: str) -> ExperimentState:
        """Read an experiment file and return an ExperimentState.

        Args:
            experiment_id: Unique identifier for the experiment.

        Returns:
            ExperimentState parsed from the YAML frontmatter.

        Raises:
            FileNotFoundError: if the experiment file does not exist.
        """
        from finalayze.core.schemas import ExperimentResult as ExperimentResultModel  # noqa: PLC0415, I001
        from finalayze.core.schemas import ExperimentState as ExperimentStateModel  # noqa: PLC0415
        from finalayze.core.schemas import SuccessCriteria as SuccessCriteriaModel  # noqa: PLC0415

        frontmatter, _ = self._read_file(experiment_id)

        # Reconstruct nested models from dicts
        if isinstance(frontmatter.get("success_criteria"), dict):
            frontmatter["success_criteria"] = SuccessCriteriaModel(
                **frontmatter["success_criteria"]
            )

        if isinstance(frontmatter.get("results"), list):
            frontmatter["results"] = [
                ExperimentResultModel(**r) if isinstance(r, dict) else r
                for r in frontmatter["results"]
            ]

        return ExperimentStateModel(**frontmatter)

    def update_status(self, experiment_id: str, status: str) -> None:
        """Update the status field of an experiment.

        Args:
            experiment_id: Unique identifier for the experiment.
            status: New status value (must be a valid ExperimentStatus).
        """
        frontmatter, body = self._read_file(experiment_id)
        frontmatter["status"] = str(status)
        self._write_file(experiment_id, frontmatter, body)

    def link_result(self, experiment_id: str, result: ExperimentResult) -> None:
        """Append an ExperimentResult to the experiment's results list.

        Args:
            experiment_id: Unique identifier for the experiment.
            result: Backtest result to append.
        """
        frontmatter, body = self._read_file(experiment_id)
        results = frontmatter.get("results") or []
        results.append({
            "run_name": result.run_name,
            "iteration_name": result.iteration_name,
            "metrics": dict(result.metrics),
        })
        frontmatter["results"] = results
        self._write_file(experiment_id, frontmatter, body)

    def record_verdict(self, experiment_id: str, metric_value: float) -> None:
        """Compute and record the verdict for an experiment.

        Reads the experiment's success_criteria, computes the verdict
        using _compute_verdict(), and updates status + verdict + reasoning.

        Args:
            experiment_id: Unique identifier for the experiment.
            metric_value: The observed metric value to compare against threshold.
        """
        from finalayze.core.schemas import ExperimentStatus  # noqa: PLC0415
        from finalayze.core.schemas import SuccessCriteria as SuccessCriteriaModel  # noqa: PLC0415

        frontmatter, body = self._read_file(experiment_id)

        # Reconstruct criteria
        sc_data = frontmatter["success_criteria"]
        criteria = SuccessCriteriaModel(**sc_data) if isinstance(sc_data, dict) else sc_data

        verdict, reasoning = _compute_verdict(criteria, metric_value)

        # Map verdict string to ExperimentStatus
        status_map = {
            "ACCEPTED": ExperimentStatus.ACCEPTED,
            "REJECTED": ExperimentStatus.REJECTED,
            "INCONCLUSIVE": ExperimentStatus.INCONCLUSIVE,
        }
        frontmatter["status"] = str(status_map[verdict])
        frontmatter["verdict"] = verdict
        frontmatter["reasoning"] = reasoning
        self._write_file(experiment_id, frontmatter, body)

    def list_experiments(self) -> list[str]:
        """Return sorted list of experiment IDs (filename stems).

        Returns:
            Sorted list of experiment IDs found in the experiments directory.
        """
        return sorted(p.stem for p in self._dir.glob("*.md"))

    def get_by_debate(self, debate_id: str) -> ExperimentState | None:
        """Find an experiment linked to a specific debate.

        Args:
            debate_id: The debate ID to search for.

        Returns:
            ExperimentState if found, None otherwise.
        """
        for exp_id in self.list_experiments():
            state = self.read_experiment(exp_id)
            if state.debate_id == debate_id:
                return state
        return None


def _compute_verdict(
    criteria: SuccessCriteria, metric_value: float
) -> tuple[str, str]:
    """Compute experiment verdict from criteria and observed metric value.

    Args:
        criteria: The success criteria with metric, threshold, and operator.
        metric_value: The observed metric value.

    Returns:
        Tuple of (verdict_string, reasoning_string).
        verdict is one of: "ACCEPTED", "REJECTED", "INCONCLUSIVE".
    """
    ops: dict[str, Any] = {">=": op.ge, "<=": op.le, ">": op.gt, "<": op.lt}
    op_fn = ops[criteria.operator]

    if op_fn(metric_value, criteria.threshold):
        return (
            "ACCEPTED",
            f"{criteria.metric}={metric_value:.4f} meets threshold "
            f"{criteria.operator} {criteria.threshold}",
        )

    relative_miss = abs(metric_value - criteria.threshold) / max(
        abs(criteria.threshold), 1e-9
    )
    if relative_miss <= _INCONCLUSIVE_BAND:
        return (
            "INCONCLUSIVE",
            f"{criteria.metric}={metric_value:.4f} within 10% of "
            f"threshold {criteria.threshold}",
        )

    return (
        "REJECTED",
        f"{criteria.metric}={metric_value:.4f} misses threshold "
        f"{criteria.operator} {criteria.threshold} by {relative_miss:.1%}",
    )
