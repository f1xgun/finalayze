"""Tests for event_driven strategy enablement on MOEX presets.

Validates that event_driven is enabled at weight 0.15 on all ru_* segments
and that total enabled weights sum to approximately 1.00.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_PRESETS_DIR = Path(__file__).resolve().parents[2] / "src" / "finalayze" / "strategies" / "presets"

_RU_PRESETS = ["ru_blue_chips", "ru_energy", "ru_finance", "ru_tech"]


def _load_preset(name: str) -> dict:
    path = _PRESETS_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestEventDrivenPresets:
    """Validate event_driven enablement on all ru_* YAML presets."""

    @pytest.mark.parametrize("preset_name", _RU_PRESETS)
    def test_event_driven_enabled(self, preset_name: str) -> None:
        """event_driven strategy is enabled on all ru_* presets."""
        preset = _load_preset(preset_name)
        ed = preset["strategies"]["event_driven"]
        assert ed["enabled"] is True, f"{preset_name}: event_driven not enabled"

    @pytest.mark.parametrize("preset_name", _RU_PRESETS)
    def test_event_driven_weight(self, preset_name: str) -> None:
        """event_driven has weight 0.15 on all ru_* presets."""
        preset = _load_preset(preset_name)
        ed = preset["strategies"]["event_driven"]
        assert ed["weight"] == pytest.approx(0.15, abs=0.001), (
            f"{preset_name}: event_driven weight is {ed['weight']}, expected 0.15"
        )

    @pytest.mark.parametrize("preset_name", _RU_PRESETS)
    def test_weights_sum_to_one(self, preset_name: str) -> None:
        """All enabled strategy weights sum to approximately 1.00."""
        preset = _load_preset(preset_name)
        total = sum(
            s["weight"]
            for s in preset["strategies"].values()
            if s.get("enabled", False)
        )
        assert total == pytest.approx(1.0, abs=0.01), (
            f"{preset_name}: enabled weights sum to {total}, expected ~1.00"
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
