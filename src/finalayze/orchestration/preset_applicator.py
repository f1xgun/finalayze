"""PresetApplicator -- atomic YAML preset writer with safety gates (Layer 5).

Closes the autonomy loop: accepted experiment verdicts atomically update
strategy YAML presets with backup, safety validation, cache invalidation,
and REST invocability.

Safety gate order (enforced in apply_verdict):
  1. Circuit breaker check (FIRST) -- non-NORMAL level raises immediately
  2. Experiment read -- FileNotFoundError if missing
  3. INCONCLUSIVE routing -- Telegram alert, no YAML write
  4. ACCEPTED gate -- non-ACCEPTED (other than INCONCLUSIVE) returns early
  5. Sandbox gate -- requires 3+ trading days with fill_rate > 0 and no high drawdown
  6. Preset validation -- unknown keys or type mismatches rejected
  7. Position ownership check -- cannot disable strategy with open positions
  8. Atomic write -- backup + pending + os.replace()
  9. Cache invalidation -- combiner.invalidate_segment_cache() if combiner provided

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import yaml

from finalayze.risk.circuit_breaker import CircuitLevel

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from finalayze.core.experiment_manager import ExperimentManager
    from finalayze.core.schemas import ExperimentState
    from finalayze.strategies.combiner import StrategyCombiner

_log = structlog.get_logger(__name__)

# Drawdown threshold for SandboxGate blocking
_SANDBOX_MAX_DRAWDOWN = 0.10
# Minimum distinct calendar days with fill_rate > 0 required to apply
_SANDBOX_MIN_DAYS = 3


# ── Exception classes ─────────────────────────────────────────────────────────


class PresetApplyBlockedError(Exception):
    """Raised when a safety gate blocks a preset apply operation."""


class PresetValidationError(ValueError):
    """Raised when key or type validation of preset_overrides fails."""


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ApplyResult:
    """Result of a PresetApplicator.apply_verdict() call."""

    experiment_id: str
    applied: bool
    backup_path: str | None
    verdict: str
    reason: str


# ── SandboxGate ───────────────────────────────────────────────────────────────


class SandboxGate:
    """Validates sandbox metrics before allowing a preset apply.

    Requires at least 3 distinct calendar dates where fill_rate > 0 AND
    no rows with drawdown_pct >= 0.10.
    """

    async def check(
        self,
        experiment_id: str,
        market_id: str,
        session: AsyncSession,
    ) -> bool:
        """Return True if sandbox metrics pass the gate criteria.

        Args:
            experiment_id: Experiment being applied (used for logging).
            market_id: Market to query metrics for.
            session: Async DB session for SandboxMetricRow query.

        Returns:
            True if >= 3 trading days with fill_rate > 0 and no high drawdown row.
            False otherwise.
        """
        from sqlalchemy import select  # noqa: PLC0415

        from finalayze.core.models import SandboxMetricRow  # noqa: PLC0415

        stmt = (
            select(SandboxMetricRow)
            .where(SandboxMetricRow.market_id == market_id)
            .order_by(SandboxMetricRow.timestamp)
        )
        result = await session.execute(stmt)
        rows = list(result.scalars().all())

        distinct_active_dates: set[Any] = set()
        for row in rows:
            fill_rate = float(row.fill_rate) if row.fill_rate is not None else 0.0
            drawdown = float(row.drawdown_pct) if row.drawdown_pct is not None else 0.0

            # Any row with high drawdown fails the gate immediately
            if drawdown >= _SANDBOX_MAX_DRAWDOWN:
                _log.info(
                    "sandbox_gate_blocked_high_drawdown",
                    experiment_id=experiment_id,
                    market_id=market_id,
                    drawdown_pct=drawdown,
                )
                return False

            if fill_rate > 0:
                distinct_active_dates.add(row.timestamp.date())

        num_days = len(distinct_active_dates)
        if num_days < _SANDBOX_MIN_DAYS:
            _log.info(
                "sandbox_gate_blocked_insufficient_days",
                experiment_id=experiment_id,
                market_id=market_id,
                active_days=num_days,
                required=_SANDBOX_MIN_DAYS,
            )
            return False

        _log.info(
            "sandbox_gate_passed",
            experiment_id=experiment_id,
            market_id=market_id,
            active_days=num_days,
        )
        return True


# ── PresetApplicator ──────────────────────────────────────────────────────────


class PresetApplicator:
    """Applies accepted experiment verdicts to strategy YAML presets.

    Implements the full safety gate pipeline described in the module docstring.
    Injected with all dependencies; no singletons accessed directly.
    """

    def __init__(
        self,
        circuit_breakers: dict[str, Any],
        alerter: Any,
        experiment_manager: ExperimentManager,
        presets_dir: Path,
        sandbox_gate: SandboxGate,
        entry_strategy_getter: Callable[[], dict[str, str]],
        combiner: StrategyCombiner | None = None,
    ) -> None:
        self._circuit_breakers = circuit_breakers
        self._alerter = alerter
        self._experiment_manager = experiment_manager
        self._presets_dir = Path(presets_dir)
        self._sandbox_gate = sandbox_gate
        self._entry_strategy_getter = entry_strategy_getter
        self._combiner = combiner

    async def apply_verdict(
        self,
        experiment_id: str,
        market_id: str,
        session: AsyncSession,
    ) -> ApplyResult:
        """Apply an accepted experiment verdict to the strategy YAML preset.

        Safety gates are enforced in order. The circuit breaker check is always
        the FIRST gate -- it runs before any file I/O or DB queries.

        Args:
            experiment_id: The experiment to apply.
            market_id: The market context for sandbox gate queries.
            session: Async DB session.

        Returns:
            ApplyResult describing the outcome.

        Raises:
            PresetApplyBlockedError: if any safety gate blocks the apply.
            PresetValidationError: if preset_overrides fail key/type validation.
            FileNotFoundError: if the experiment does not exist.
        """
        # ── Gate 1: Circuit breaker (MUST be first, before any I/O) ──────────
        for cb in self._circuit_breakers.values():
            if cb.level != CircuitLevel.NORMAL:
                msg = f"Circuit breaker {cb.market_id} at level {cb.level}"
                _log.warning(
                    "preset_apply_blocked_circuit_breaker",
                    experiment_id=experiment_id,
                    market_id=cb.market_id,
                    level=str(cb.level),
                )
                raise PresetApplyBlockedError(msg)

        # ── Gate 2: Read experiment (raises FileNotFoundError if missing) ─────
        state: ExperimentState = self._experiment_manager.read_experiment(experiment_id)

        # ── Gate 3: INCONCLUSIVE routing ──────────────────────────────────────
        if state.verdict == "INCONCLUSIVE":
            self._alert_inconclusive(state)
            return ApplyResult(
                experiment_id=experiment_id,
                applied=False,
                backup_path=None,
                verdict="INCONCLUSIVE",
                reason="Routed to operator via Telegram",
            )

        # ── Gate 4: Only ACCEPTED proceeds ────────────────────────────────────
        if state.verdict != "ACCEPTED":
            return ApplyResult(
                experiment_id=experiment_id,
                applied=False,
                backup_path=None,
                verdict=state.verdict or "UNKNOWN",
                reason=f"Verdict is {state.verdict}, not ACCEPTED",
            )

        # ── Gate 5: Sandbox gate ──────────────────────────────────────────────
        if not await self._sandbox_gate.check(experiment_id, market_id, session):
            raise PresetApplyBlockedError(
                "Sandbox validation failed: < 3 trading days or high drawdown"
            )

        # ── Validate overrides present ────────────────────────────────────────
        overrides = state.preset_overrides
        if not overrides:
            raise PresetValidationError("preset_overrides is None or empty")

        # Determine segment_id from overrides
        segment_id = overrides.get("segment_id")
        if not segment_id or not isinstance(segment_id, str):
            raise PresetValidationError(
                "preset_overrides must contain a top-level 'segment_id' string key"
            )

        # Resolve preset file path
        preset_path = (self._presets_dir / f"{segment_id}.yaml").resolve()

        # Security: ensure path is within presets_dir (T-38-03)
        presets_dir_resolved = self._presets_dir.resolve()
        if not str(preset_path).startswith(str(presets_dir_resolved)):
            raise PresetValidationError(
                f"segment_id '{segment_id}' resolves outside presets directory"
            )

        if not preset_path.exists():
            raise FileNotFoundError(f"Preset file not found: {preset_path}")

        current_yaml: dict[str, Any] = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}

        # ── Gate 6: Key and type validation ───────────────────────────────────
        # Validate all keys in overrides (except segment_id itself, which is the routing key)
        overrides_to_validate = {k: v for k, v in overrides.items() if k != "segment_id"}
        self._validate_keys(overrides_to_validate, current_yaml)

        # ── Gate 7: Position ownership check ─────────────────────────────────
        self._check_position_ownership(overrides)

        # ── Steps 8-10: Backup + deep merge + atomic write ────────────────────
        merged = self._deep_merge(current_yaml, overrides_to_validate)
        backup_path = self._atomic_write_yaml(preset_path, merged, segment_id)

        # ── Gate 9: Cache invalidation ────────────────────────────────────────
        _log.debug(
            "preset_applied_cache_invalidation",
            segment_id=segment_id,
            combiner_available=self._combiner is not None,
        )
        if self._combiner is not None:
            self._combiner.invalidate_segment_cache(segment_id)

        _log.info(
            "preset_applied",
            experiment_id=experiment_id,
            segment_id=segment_id,
            backup_path=str(backup_path),
        )
        return ApplyResult(
            experiment_id=experiment_id,
            applied=True,
            backup_path=str(backup_path),
            verdict="ACCEPTED",
            reason="Applied successfully",
        )

    def _check_position_ownership(self, overrides: dict[str, Any]) -> None:
        """Raise PresetApplyBlockedError if a disabled strategy has open positions.

        Args:
            overrides: The full preset_overrides dict (may contain 'strategies' key).

        Raises:
            PresetApplyBlockedError: if any disabled strategy has open positions.
        """
        strategy_overrides: dict[str, Any] = overrides.get("strategies", {})
        disabled_strategies = [
            name
            for name, cfg in strategy_overrides.items()
            if isinstance(cfg, dict) and cfg.get("enabled") is False
        ]
        if not disabled_strategies:
            return
        current_positions = self._entry_strategy_getter()
        for strategy_name in disabled_strategies:
            open_symbols = [
                sym for sym, strat in current_positions.items() if strat == strategy_name
            ]
            if open_symbols:
                raise PresetApplyBlockedError(
                    f"Strategy {strategy_name} has open positions: {open_symbols}"
                )

    def _alert_inconclusive(self, state: ExperimentState) -> None:
        """Send a Telegram alert for an INCONCLUSIVE verdict."""
        from finalayze.api.alerts import AlertPriority  # noqa: PLC0415

        message = (
            f"<b>INCONCLUSIVE Experiment: {state.experiment_id}</b>\n\n"
            f"Hypothesis: {state.hypothesis}\n"
            f"Verdict: INCONCLUSIVE\n"
            f"Reasoning: {state.reasoning or 'N/A'}"
        )
        self._alerter.send_alert(message, priority=AlertPriority.IMPORTANT)

    def _validate_keys(
        self,
        overrides: dict[str, Any],
        current: dict[str, Any],
        path: str = "",
    ) -> None:
        """Recursively validate that override keys exist in current YAML and types match.

        Args:
            overrides: The override dict to validate.
            current: The current YAML dict to validate against.
            path: Dot-separated path for error messages.

        Raises:
            PresetValidationError: if an unknown key or type mismatch is found.
        """
        for key, value in overrides.items():
            full_key = f"{path}.{key}" if path else key
            if key not in current:
                raise PresetValidationError(
                    f"Unknown key in preset_overrides: '{full_key}'. "
                    f"Available: {list(current.keys())}"
                )
            current_value = current[key]
            if isinstance(value, dict) and isinstance(current_value, dict):
                # Recurse into nested dicts
                self._validate_keys(value, current_value, full_key)
            elif value is not None and current_value is not None:
                # Type check: allow int<->float coercion
                cv_type = type(current_value)
                ov_type = type(value)
                numeric_types = (int, float)
                if cv_type in numeric_types and ov_type in numeric_types:
                    pass  # int<->float coercion is fine
                elif not isinstance(value, cv_type):
                    raise PresetValidationError(
                        f"Type mismatch for key '{full_key}': "
                        f"expected {cv_type.__name__}, got {ov_type.__name__}"
                    )

    def _deep_merge(
        self,
        base: dict[str, Any],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Recursively merge overrides into base dict.

        Only specified keys are overridden; unspecified keys are preserved.

        Args:
            base: The original dict.
            overrides: Keys to override.

        Returns:
            New merged dict (base is not mutated).
        """
        result = dict(base)
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _atomic_write_yaml(
        self,
        preset_path: Path,
        merged: dict[str, Any],
        segment_id: str,
    ) -> Path:
        """Write merged YAML atomically with backup.

        1. Creates timestamped backup of current file (via shutil.copy2).
        2. Writes merged YAML to .pending staging file.
        3. Uses os.replace() for atomic rename.

        Args:
            preset_path: The YAML preset file to update.
            merged: The merged dict to write.
            segment_id: Used for backup filename.

        Returns:
            Path to the backup file.
        """
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_path = preset_path.parent / f"{segment_id}.yaml.bak.{ts}"

        # Backup: write original content
        shutil.copy2(preset_path, backup_path)

        # Pending file: write merged content
        pending_path = preset_path.parent / f"{segment_id}.yaml.pending"
        pending_path.write_text(
            yaml.dump(merged, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        # Atomic rename
        os.replace(pending_path, preset_path)

        return backup_path
