"""Validate MOEX strategy presets: enabled strategies, weights, sector tilts.

These tests ensure all ru_* YAML presets have the correct strategy configuration
after enabling momentum and dual_momentum with sector-specific weight tilts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_PRESETS_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "finalayze" / "strategies" / "presets"

# All equity strategies that should be enabled on MOEX presets
_EQUITY_STRATEGIES = [
    "momentum",
    "dual_momentum",
    "mean_reversion",
    "rsi2_connors",
    "ou_mean_reversion",
    "dividend_gap",
]

_RU_PRESETS = ["ru_blue_chips", "ru_energy", "ru_finance"]

_WEIGHT_TOLERANCE = 0.02


def _load_preset(name: str) -> dict:
    """Load a YAML preset by name."""
    path = _PRESETS_DIR / f"{name}.yaml"
    assert path.exists(), f"Preset file {path} does not exist"
    with path.open() as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("preset_name", _RU_PRESETS)
def test_preset_loads_without_error(preset_name: str) -> None:
    """Test 1: All 3 ru_* YAML presets load without error."""
    data = _load_preset(preset_name)
    assert isinstance(data, dict)
    assert "strategies" in data


@pytest.mark.parametrize("preset_name", _RU_PRESETS)
def test_momentum_strategies_enabled(preset_name: str) -> None:
    """Test 2: Each preset has momentum and dual_momentum enabled."""
    data = _load_preset(preset_name)
    strategies = data["strategies"]

    mom = strategies.get("momentum", {})
    assert mom.get("enabled") is True, f"{preset_name}: momentum should be enabled"

    dual = strategies.get("dual_momentum", {})
    assert dual.get("enabled") is True, f"{preset_name}: dual_momentum should be enabled"


@pytest.mark.parametrize("preset_name", _RU_PRESETS)
def test_weights_sum_to_one(preset_name: str) -> None:
    """Test 3: Strategy weights in each preset sum to approximately 1.0."""
    data = _load_preset(preset_name)
    strategies = data["strategies"]

    enabled_weights = [
        cfg.get("weight", 0.0)
        for cfg in strategies.values()
        if cfg.get("enabled", False)
    ]
    total = sum(enabled_weights)
    assert abs(total - 1.0) < _WEIGHT_TOLERANCE, (
        f"{preset_name}: enabled weights sum to {total}, expected ~1.0"
    )


def test_ru_energy_momentum_tilt() -> None:
    """Test 4: ru_energy has momentum weight >= 0.18 (momentum tilt)."""
    data = _load_preset("ru_energy")
    strategies = data["strategies"]
    mom_weight = strategies.get("momentum", {}).get("weight", 0.0)
    assert mom_weight >= 0.18, (
        f"ru_energy momentum weight {mom_weight} should be >= 0.18"
    )


def test_ru_finance_mr_tilt() -> None:
    """Test 5: ru_finance has mean_reversion weight >= 0.18 (MR tilt)."""
    data = _load_preset("ru_finance")
    strategies = data["strategies"]
    mr_weight = strategies.get("mean_reversion", {}).get("weight", 0.0)
    assert mr_weight >= 0.18, (
        f"ru_finance mean_reversion weight {mr_weight} should be >= 0.18"
    )


def test_ru_blue_chips_balanced() -> None:
    """Test 6: ru_blue_chips has balanced weights (no single strategy > 0.20)."""
    data = _load_preset("ru_blue_chips")
    strategies = data["strategies"]

    for name, cfg in strategies.items():
        if not cfg.get("enabled", False):
            continue
        weight = cfg.get("weight", 0.0)
        assert weight <= 0.20, (
            f"ru_blue_chips {name} weight {weight} exceeds balanced threshold 0.20"
        )


@pytest.mark.parametrize("preset_name", _RU_PRESETS)
def test_all_equity_strategies_present_and_enabled(preset_name: str) -> None:
    """All 6 equity strategies are present and enabled in each ru_* preset."""
    data = _load_preset(preset_name)
    strategies = data["strategies"]

    # ru_finance has no pairs configured, so skip pairs check for it
    required = list(_EQUITY_STRATEGIES)
    if preset_name == "ru_finance":
        required = [s for s in required if s != "pairs"]

    for strat_name in required:
        assert strat_name in strategies, (
            f"{preset_name}: missing strategy '{strat_name}'"
        )
        assert strategies[strat_name].get("enabled") is True, (
            f"{preset_name}: strategy '{strat_name}' should be enabled"
        )
