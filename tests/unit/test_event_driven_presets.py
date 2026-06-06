"""Tests for event_driven strategy configuration on MOEX presets.

Validates that event_driven is enabled on all ru_* segments (v10.0 EVNT-01),
has weight 0.15, and enabled weights are consistent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_PRESETS_DIR = Path(__file__).resolve().parents[2] / "src" / "finalayze" / "strategies" / "presets"

_RU_PRESETS = ["ru_energy", "ru_finance", "ru_tech"]

_EVENT_DRIVEN_RESERVED_WEIGHT = 0.15


def _load_preset(name: str) -> dict:
    path = _PRESETS_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestEventDrivenPresets:
    """Validate event_driven configuration on all ru_* YAML presets."""

    @pytest.mark.parametrize("preset_name", _RU_PRESETS)
    def test_event_driven_enabled(self, preset_name: str) -> None:
        """event_driven strategy is enabled on all ru_* presets (EVNT-01)."""
        preset = _load_preset(preset_name)
        ed = preset["strategies"]["event_driven"]
        assert ed["enabled"] is True, f"{preset_name}: event_driven should be enabled"

    @pytest.mark.parametrize("preset_name", _RU_PRESETS)
    def test_event_driven_weight(self, preset_name: str) -> None:
        """event_driven has weight 0.15 reserved on all ru_* presets."""
        preset = _load_preset(preset_name)
        ed = preset["strategies"]["event_driven"]
        assert ed["weight"] == pytest.approx(_EVENT_DRIVEN_RESERVED_WEIGHT, abs=0.001), (
            f"{preset_name}: event_driven weight is {ed['weight']}, expected 0.15"
        )

    @pytest.mark.parametrize("preset_name", _RU_PRESETS)
    def test_weights_sum_reasonable(self, preset_name: str) -> None:
        """All enabled strategy weights sum to a reasonable value (0.80-1.15)."""
        preset = _load_preset(preset_name)
        total = sum(s["weight"] for s in preset["strategies"].values() if s.get("enabled", False))
        _min_weight_sum = 0.80
        _max_weight_sum = 1.16  # ru_blue_chips sums to ~1.1 with event_driven enabled
        assert _min_weight_sum <= total <= _max_weight_sum, (
            f"{preset_name}: enabled weights sum to {total}, expected 0.80-1.15"
        )

    @pytest.mark.parametrize("preset_name", _RU_PRESETS)
    def test_event_driven_min_sentiment(self, preset_name: str) -> None:
        """event_driven has min_sentiment >= 0.6."""
        preset = _load_preset(preset_name)
        ed = preset["strategies"]["event_driven"]
        assert ed["params"]["min_sentiment"] >= 0.6, (
            f"{preset_name}: min_sentiment is {ed['params']['min_sentiment']}"
        )

    @pytest.mark.parametrize("preset_name", _RU_PRESETS)
    def test_event_driven_has_event_types(self, preset_name: str) -> None:
        """event_driven has a non-empty event_types list."""
        preset = _load_preset(preset_name)
        ed = preset["strategies"]["event_driven"]
        event_types = ed["params"]["event_types"]
        assert len(event_types) >= 3, (
            f"{preset_name}: event_types has only {len(event_types)} entries"
        )
