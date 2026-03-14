"""Unit tests for LayerConfig and DEFAULT_LAYER_CONFIGS."""

from __future__ import annotations

from finalayze.core.schemas import DEFAULT_LAYER_CONFIGS, PortfolioLayer


def test_short_layer_allows_bonds() -> None:
    """SHORT layer must allow bond instruments for OFZ-PK parking."""
    short_config = DEFAULT_LAYER_CONFIGS[PortfolioLayer.SHORT]
    assert "bond" in short_config.allowed_instrument_types


def test_core_layer_allows_only_bonds() -> None:
    core_config = DEFAULT_LAYER_CONFIGS[PortfolioLayer.CORE]
    assert core_config.allowed_instrument_types == ("bond",)


def test_tactical_layer_allows_both() -> None:
    tactical_config = DEFAULT_LAYER_CONFIGS[PortfolioLayer.TACTICAL]
    assert "bond" in tactical_config.allowed_instrument_types
    assert "stock" in tactical_config.allowed_instrument_types


def test_all_layers_have_configs() -> None:
    for layer in PortfolioLayer:
        assert layer in DEFAULT_LAYER_CONFIGS, f"Missing config for {layer}"
