"""Tests for PresetApplicator and SandboxGate (Phase 38, Plan 01).

TDD RED phase: all tests written before implementation exists.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# ── Constants (no magic numbers) ─────────────────────────────────────────────

_HIGH_DRAWDOWN = 0.10  # SandboxGate blocks if drawdown_pct >= 0.10
_SAFE_FILL_RATE = 0.80  # A fill_rate > 0 that counts as an active day
_ZERO_FILL_RATE = 0.0   # fill_rate = 0 → day does NOT count


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_metric_row(
    ts: datetime,
    fill_rate: float,
    drawdown_pct: float,
    market_id: str = "moex",
) -> MagicMock:
    """Return a mock SandboxMetricRow with explicit attribute values."""
    row = MagicMock()
    row.timestamp = ts
    row.fill_rate = fill_rate
    row.drawdown_pct = drawdown_pct
    row.market_id = market_id
    return row


def _make_experiment_state(
    experiment_id: str = "exp-001",
    verdict: str | None = "ACCEPTED",
    preset_overrides: dict[str, Any] | None = None,
    segment_id: str = "us_tech",
) -> MagicMock:
    """Return a minimal mock ExperimentState."""
    state = MagicMock()
    state.experiment_id = experiment_id
    state.verdict = verdict
    state.preset_overrides = preset_overrides or {
        "segment_id": segment_id,
        "strategies": {"dual_momentum": {"weight": 0.30}},
    }
    state.hypothesis = "Test hypothesis"
    state.reasoning = "Profit factor exceeded threshold"
    return state


def _write_preset_yaml(tmp_path: Path, segment_id: str) -> Path:
    """Write a minimal us_tech-style YAML preset and return its path."""
    content = {
        "segment_id": segment_id,
        "normalize_mode": "firing",
        "min_combined_confidence": 0.30,
        "strategies": {
            "dual_momentum": {
                "enabled": True,
                "weight": 0.25,
                "params": {"lookback_1m": 21},
            },
            "mean_reversion": {
                "enabled": True,
                "weight": 0.25,
            },
        },
    }
    preset_file = tmp_path / f"{segment_id}.yaml"
    preset_file.write_text(yaml.dump(content, default_flow_style=False))
    return preset_file


def _make_applicator(
    tmp_path: Path,
    *,
    circuit_breaker_level: str = "normal",
    verdict: str | None = "ACCEPTED",
    preset_overrides: dict[str, Any] | None = None,
    entry_strategies: dict[str, str] | None = None,
    combiner: Any = None,
    sandbox_gate_result: bool = True,
    alerter: Any = None,
    segment_id: str = "us_tech",
) -> tuple[Any, Any]:
    """Build a PresetApplicator with mocked deps. Returns (applicator, mock_session)."""
    from finalayze.orchestration.preset_applicator import PresetApplicator, SandboxGate

    # Write YAML preset
    _write_preset_yaml(tmp_path, segment_id)

    # Mock circuit breaker
    mock_cb = MagicMock()
    mock_cb.level = circuit_breaker_level
    mock_cb.market_id = "moex"

    # Mock experiment manager
    mock_em = MagicMock()
    state = _make_experiment_state(
        verdict=verdict,
        preset_overrides=preset_overrides,
        segment_id=segment_id,
    )
    mock_em.read_experiment.return_value = state

    # Mock alerter
    if alerter is None:
        alerter = MagicMock()

    # Mock sandbox gate
    mock_gate = MagicMock(spec=SandboxGate)
    mock_gate.check = AsyncMock(return_value=sandbox_gate_result)

    # Mock DB session
    mock_session = AsyncMock()

    applicator = PresetApplicator(
        circuit_breakers={"moex": mock_cb},
        alerter=alerter,
        experiment_manager=mock_em,
        presets_dir=tmp_path,
        sandbox_gate=mock_gate,
        entry_strategy_getter=lambda: entry_strategies or {},
        combiner=combiner,
    )
    return applicator, mock_session


# ── SandboxGate unit tests ────────────────────────────────────────────────────


class TestSandboxGate:
    """Tests for SandboxGate.check()."""

    @pytest.mark.asyncio
    async def test_sandbox_gate_passes_three_days(self) -> None:
        """SandboxGate with 3+ distinct dates with fill_rate > 0 returns True."""
        from finalayze.orchestration.preset_applicator import SandboxGate

        rows = [
            _make_metric_row(datetime(2026, 4, 1, 10, 0, tzinfo=UTC), _SAFE_FILL_RATE, 0.02),
            _make_metric_row(datetime(2026, 4, 2, 10, 0, tzinfo=UTC), _SAFE_FILL_RATE, 0.01),
            _make_metric_row(datetime(2026, 4, 3, 10, 0, tzinfo=UTC), _SAFE_FILL_RATE, 0.03),
        ]
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = rows
        mock_session.execute = AsyncMock(return_value=mock_result)

        gate = SandboxGate()
        result = await gate.check("exp-001", "moex", mock_session)
        assert result is True

    @pytest.mark.asyncio
    async def test_sandbox_gate_blocks_insufficient_days(self) -> None:
        """SandboxGate with only 2 days of data returns False."""
        from finalayze.orchestration.preset_applicator import SandboxGate

        rows = [
            _make_metric_row(datetime(2026, 4, 1, 10, 0, tzinfo=UTC), _SAFE_FILL_RATE, 0.01),
            _make_metric_row(datetime(2026, 4, 2, 10, 0, tzinfo=UTC), _SAFE_FILL_RATE, 0.02),
        ]
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = rows
        mock_session.execute = AsyncMock(return_value=mock_result)

        gate = SandboxGate()
        result = await gate.check("exp-001", "moex", mock_session)
        assert result is False

    @pytest.mark.asyncio
    async def test_sandbox_gate_blocks_zero_fill_rate_days(self) -> None:
        """SandboxGate ignores rows with fill_rate=0, needs 3 with fill_rate > 0."""
        from finalayze.orchestration.preset_applicator import SandboxGate

        rows = [
            _make_metric_row(datetime(2026, 4, 1, 10, 0, tzinfo=UTC), _SAFE_FILL_RATE, 0.01),
            _make_metric_row(datetime(2026, 4, 2, 10, 0, tzinfo=UTC), _ZERO_FILL_RATE, 0.00),
            _make_metric_row(datetime(2026, 4, 3, 10, 0, tzinfo=UTC), _ZERO_FILL_RATE, 0.00),
        ]
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = rows
        mock_session.execute = AsyncMock(return_value=mock_result)

        gate = SandboxGate()
        result = await gate.check("exp-001", "moex", mock_session)
        assert result is False

    @pytest.mark.asyncio
    async def test_sandbox_gate_blocks_high_drawdown(self) -> None:
        """SandboxGate with 3 days but one has drawdown_pct >= 0.10 returns False."""
        from finalayze.orchestration.preset_applicator import SandboxGate

        rows = [
            _make_metric_row(datetime(2026, 4, 1, 10, 0, tzinfo=UTC), _SAFE_FILL_RATE, 0.02),
            _make_metric_row(datetime(2026, 4, 2, 10, 0, tzinfo=UTC), _SAFE_FILL_RATE, _HIGH_DRAWDOWN),
            _make_metric_row(datetime(2026, 4, 3, 10, 0, tzinfo=UTC), _SAFE_FILL_RATE, 0.03),
        ]
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = rows
        mock_session.execute = AsyncMock(return_value=mock_result)

        gate = SandboxGate()
        result = await gate.check("exp-001", "moex", mock_session)
        assert result is False


# ── PresetApplicator unit tests ───────────────────────────────────────────────


class TestPresetApplicatorCircuitBreaker:
    """Tests for circuit breaker gate (must be first check)."""

    @pytest.mark.asyncio
    async def test_apply_blocked_by_circuit_breaker(self, tmp_path: Path) -> None:
        """CircuitLevel.CAUTION -> raises PresetApplyBlockedError, no file I/O."""
        from finalayze.orchestration.preset_applicator import PresetApplicator, PresetApplyBlockedError, SandboxGate

        mock_cb = MagicMock()
        mock_cb.level = "caution"
        mock_cb.market_id = "moex"
        mock_em = MagicMock()
        mock_gate = MagicMock(spec=SandboxGate)
        mock_gate.check = AsyncMock(return_value=True)
        mock_session = AsyncMock()

        applicator = PresetApplicator(
            circuit_breakers={"moex": mock_cb},
            alerter=MagicMock(),
            experiment_manager=mock_em,
            presets_dir=tmp_path,
            sandbox_gate=mock_gate,
            entry_strategy_getter=lambda: {},
        )

        with pytest.raises(PresetApplyBlockedError, match="caution"):
            await applicator.apply_verdict("exp-001", "moex", mock_session)

        # No file I/O: experiment was not read
        mock_em.read_experiment.assert_not_called()
        # No sandbox check
        mock_gate.check.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_blocked_by_circuit_breaker_halted(self, tmp_path: Path) -> None:
        """CircuitLevel.HALTED -> raises PresetApplyBlockedError."""
        from finalayze.orchestration.preset_applicator import PresetApplicator, PresetApplyBlockedError, SandboxGate

        mock_cb = MagicMock()
        mock_cb.level = "halted"
        mock_cb.market_id = "moex"
        mock_em = MagicMock()
        mock_gate = MagicMock(spec=SandboxGate)
        mock_gate.check = AsyncMock(return_value=True)
        mock_session = AsyncMock()

        applicator = PresetApplicator(
            circuit_breakers={"moex": mock_cb},
            alerter=MagicMock(),
            experiment_manager=mock_em,
            presets_dir=tmp_path,
            sandbox_gate=mock_gate,
            entry_strategy_getter=lambda: {},
        )

        with pytest.raises(PresetApplyBlockedError, match="halted"):
            await applicator.apply_verdict("exp-001", "moex", mock_session)


class TestPresetApplicatorInconclusive:
    """Tests for INCONCLUSIVE verdict Telegram routing."""

    @pytest.mark.asyncio
    async def test_inconclusive_sends_telegram_no_write(self, tmp_path: Path) -> None:
        """INCONCLUSIVE verdict -> alerter.send_alert() called once, no YAML file modified."""
        _write_preset_yaml(tmp_path, "us_tech")
        mock_alerter = MagicMock()

        applicator, mock_session = _make_applicator(
            tmp_path,
            verdict="INCONCLUSIVE",
            alerter=mock_alerter,
        )

        result = await applicator.apply_verdict("exp-001", "moex", mock_session)

        assert result.applied is False
        assert result.verdict == "INCONCLUSIVE"
        # Telegram alert must have been sent
        mock_alerter.send_alert.assert_called_once()
        # No YAML file modified (no backup files)
        backup_files = list(tmp_path.glob("*.bak.*"))
        assert backup_files == []


class TestPresetApplicatorMissingExperiment:
    """Tests for missing experiment handling."""

    @pytest.mark.asyncio
    async def test_apply_rejects_missing_experiment(self, tmp_path: Path) -> None:
        """Non-existent experiment_id -> raises FileNotFoundError."""
        from finalayze.orchestration.preset_applicator import PresetApplicator, SandboxGate

        mock_cb = MagicMock()
        mock_cb.level = "normal"
        mock_cb.market_id = "moex"
        mock_em = MagicMock()
        mock_em.read_experiment.side_effect = FileNotFoundError("not found")
        mock_gate = MagicMock(spec=SandboxGate)
        mock_gate.check = AsyncMock(return_value=True)
        mock_session = AsyncMock()

        applicator = PresetApplicator(
            circuit_breakers={"moex": mock_cb},
            alerter=MagicMock(),
            experiment_manager=mock_em,
            presets_dir=tmp_path,
            sandbox_gate=mock_gate,
            entry_strategy_getter=lambda: {},
        )

        with pytest.raises(FileNotFoundError):
            await applicator.apply_verdict("nonexistent", "moex", mock_session)


class TestPresetApplicatorSandboxGate:
    """Tests for sandbox gate integration in apply_verdict."""

    @pytest.mark.asyncio
    async def test_sandbox_gate_blocks_apply(self, tmp_path: Path) -> None:
        """When sandbox gate returns False, apply raises PresetApplyBlockedError."""
        from finalayze.orchestration.preset_applicator import PresetApplyBlockedError

        applicator, mock_session = _make_applicator(
            tmp_path,
            sandbox_gate_result=False,
        )

        with pytest.raises(PresetApplyBlockedError, match="Sandbox"):
            await applicator.apply_verdict("exp-001", "moex", mock_session)


class TestPresetApplicatorAtomicWrite:
    """Tests for atomic YAML write and backup creation."""

    @pytest.mark.asyncio
    async def test_apply_creates_backup(self, tmp_path: Path) -> None:
        """After successful apply, backup file exists with original content."""
        preset_file = _write_preset_yaml(tmp_path, "us_tech")
        original_content = preset_file.read_text()

        applicator, mock_session = _make_applicator(tmp_path)
        result = await applicator.apply_verdict("exp-001", "moex", mock_session)

        assert result.applied is True
        assert result.backup_path is not None
        backup = Path(result.backup_path)
        assert backup.exists()
        assert backup.read_text() == original_content

    @pytest.mark.asyncio
    async def test_apply_writes_yaml_atomically(self, tmp_path: Path) -> None:
        """After successful apply, target YAML contains merged overrides, no .pending file remains."""
        _write_preset_yaml(tmp_path, "us_tech")

        applicator, mock_session = _make_applicator(
            tmp_path,
            preset_overrides={
                "segment_id": "us_tech",
                "strategies": {"dual_momentum": {"weight": 0.30}},
            },
        )
        result = await applicator.apply_verdict("exp-001", "moex", mock_session)

        assert result.applied is True

        # No pending file left over
        pending_files = list(tmp_path.glob("*.pending"))
        assert pending_files == []

        # Target YAML has merged weight
        preset_data = yaml.safe_load((tmp_path / "us_tech.yaml").read_text())
        assert preset_data["strategies"]["dual_momentum"]["weight"] == 0.30

    @pytest.mark.asyncio
    async def test_apply_deep_merge(self, tmp_path: Path) -> None:
        """Deep merge: overrides partial nested dict, preserving other keys."""
        _write_preset_yaml(tmp_path, "us_tech")

        applicator, mock_session = _make_applicator(
            tmp_path,
            preset_overrides={
                "segment_id": "us_tech",
                "strategies": {"dual_momentum": {"weight": 0.30}},
            },
        )
        await applicator.apply_verdict("exp-001", "moex", mock_session)

        preset_data = yaml.safe_load((tmp_path / "us_tech.yaml").read_text())
        # dual_momentum.weight overridden
        assert preset_data["strategies"]["dual_momentum"]["weight"] == 0.30
        # dual_momentum.enabled still present (preserved from original)
        assert preset_data["strategies"]["dual_momentum"]["enabled"] is True
        # mean_reversion still present
        assert "mean_reversion" in preset_data["strategies"]


class TestPresetApplicatorValidation:
    """Tests for key and type validation."""

    @pytest.mark.asyncio
    async def test_apply_validates_keys(self, tmp_path: Path) -> None:
        """Unknown top-level key in preset_overrides raises ValueError."""
        from finalayze.orchestration.preset_applicator import PresetValidationError

        applicator, mock_session = _make_applicator(
            tmp_path,
            preset_overrides={
                "segment_id": "us_tech",
                "totally_unknown_key": "bad",
            },
        )

        with pytest.raises(PresetValidationError):
            await applicator.apply_verdict("exp-001", "moex", mock_session)

    @pytest.mark.asyncio
    async def test_apply_validates_types(self, tmp_path: Path) -> None:
        """Wrong type for existing key raises PresetValidationError."""
        from finalayze.orchestration.preset_applicator import PresetValidationError

        # min_combined_confidence is a float; passing a string should fail
        applicator, mock_session = _make_applicator(
            tmp_path,
            preset_overrides={
                "segment_id": "us_tech",
                "min_combined_confidence": "not-a-float",
            },
        )

        with pytest.raises(PresetValidationError):
            await applicator.apply_verdict("exp-001", "moex", mock_session)


class TestPresetApplicatorPositionOwnership:
    """Tests for open position ownership check."""

    @pytest.mark.asyncio
    async def test_position_ownership_blocks_disable(self, tmp_path: Path) -> None:
        """Disabling strategy with open positions raises PresetApplyBlockedError."""
        from finalayze.orchestration.preset_applicator import PresetApplyBlockedError

        applicator, mock_session = _make_applicator(
            tmp_path,
            entry_strategies={"SBER": "dual_momentum"},
            preset_overrides={
                "segment_id": "us_tech",
                "strategies": {"dual_momentum": {"enabled": False, "weight": 0.0}},
            },
        )

        with pytest.raises(PresetApplyBlockedError, match="dual_momentum"):
            await applicator.apply_verdict("exp-001", "moex", mock_session)


class TestPresetApplicatorCacheInvalidation:
    """Tests for combiner cache invalidation after successful apply."""

    @pytest.mark.asyncio
    async def test_apply_calls_invalidate_cache(self, tmp_path: Path) -> None:
        """After successful apply with combiner provided, invalidate_segment_cache called once."""
        _write_preset_yaml(tmp_path, "us_tech")
        mock_combiner = MagicMock()
        mock_combiner.invalidate_segment_cache = MagicMock()

        applicator, mock_session = _make_applicator(
            tmp_path,
            combiner=mock_combiner,
        )
        result = await applicator.apply_verdict("exp-001", "moex", mock_session)

        assert result.applied is True
        mock_combiner.invalidate_segment_cache.assert_called_once_with("us_tech")

    @pytest.mark.asyncio
    async def test_apply_succeeds_without_combiner(self, tmp_path: Path) -> None:
        """After successful apply with combiner=None, no error (cache invalidation skipped)."""
        _write_preset_yaml(tmp_path, "us_tech")

        applicator, mock_session = _make_applicator(
            tmp_path,
            combiner=None,
        )
        result = await applicator.apply_verdict("exp-001", "moex", mock_session)

        assert result.applied is True
