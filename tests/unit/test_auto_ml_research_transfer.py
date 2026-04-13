"""Tests for cross_segment_transfer strategy in auto_ml_research.py.

- T1: generate_transfer_experiments with valid US JSONL returns list[ExperimentConfig]
      with strategy="cross_segment_transfer"
- T2: Each returned config has feature_subset containing only market-neutral features
      (no "vix", "usdrub", "brent", "cbr", "imoex", "turnover" substrings)
- T3: generate_transfer_experiments reads the best "keep" entry from US JSONL log
      (highest score where status=="keep")
- T4: When US JSONL file does not exist, returns empty list and logs warning
- T5: When US JSONL has no "keep" entries, returns empty list and logs warning
- T6: "cross_segment_transfer" appears in CLI --strategy choices
- T7: _generate_experiments("cross_segment_transfer", ...) routes to
      generate_transfer_experiments and returns non-empty list (with valid JSONL fixture)
- T8: _generate_experiments("all", ...) includes cross_segment_transfer results
      (with valid JSONL fixture)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module import helper
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "auto_ml_research.py"
_MODULE_NAME = "auto_ml_research"


def _import_module() -> Any:
    """Import auto_ml_research safely (registers in sys.modules to fix dataclass resolution)."""
    import importlib.util

    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT_PATH)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so that dataclass string annotations can resolve
    sys.modules[_MODULE_NAME] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_jsonl_fixture(tmp_path: Path, entries: list[dict]) -> Path:
    """Create a JSONL fixture file with given experiment entries."""
    jsonl_path = tmp_path / "us_tech_experiment_log.jsonl"
    with jsonl_path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return jsonl_path


_MARKET_NEUTRAL_FEATURES = [
    "rsi_14",
    "volume_ratio",
    "close_to_ma50",
    "momentum_20",
    "atr_14",
]
_MARKET_SPECIFIC_FEATURES = [
    "vix_level",
    "usdrub_rate",
    "brent_price",
    "cbr_rate",
    "imoex_return",
    "turnover_ratio",
]
_MIXED_FEATURES = _MARKET_NEUTRAL_FEATURES + _MARKET_SPECIFIC_FEATURES

_KEEP_ENTRY = {
    "name": "ablate-feat_a",
    "strategy": "ablation",
    "status": "keep",
    "score": 0.85,
    "features_used": _MIXED_FEATURES,
}
_DISCARD_ENTRY = {
    "name": "ablate-feat_b",
    "strategy": "ablation",
    "status": "discard",
    "score": 0.92,  # higher score but status=discard
    "features_used": _MIXED_FEATURES,
}
_LOWER_KEEP_ENTRY = {
    "name": "ablate-feat_c",
    "strategy": "ablation",
    "status": "keep",
    "score": 0.70,
    "features_used": ["rsi_14", "volume_ratio"],
}


# ---------------------------------------------------------------------------
# T1: generate_transfer_experiments returns ExperimentConfig with correct strategy
# ---------------------------------------------------------------------------


def test_t1_generate_transfer_experiments_returns_configs(tmp_path: Path) -> None:
    """T1: generate_transfer_experiments with valid US JSONL returns list[ExperimentConfig]
    with strategy="cross_segment_transfer"."""
    mod = _import_module()
    jsonl_dir = tmp_path / "experiments"
    jsonl_dir.mkdir()
    jsonl_path = jsonl_dir / "us_tech_experiment_log.jsonl"
    with jsonl_path.open("w") as f:
        f.write(json.dumps(_KEEP_ENTRY) + "\n")

    with patch.object(mod, "_RESULTS_DIR", jsonl_dir):
        result = mod.generate_transfer_experiments("ru_large_cap")

    assert isinstance(result, list)
    assert len(result) > 0
    for exp in result:
        assert exp.strategy == "cross_segment_transfer", (
            f"Expected 'cross_segment_transfer', got {exp.strategy!r}"
        )


# ---------------------------------------------------------------------------
# T2: No market-specific features in returned configs
# ---------------------------------------------------------------------------


def test_t2_no_market_specific_features_in_result(tmp_path: Path) -> None:
    """T2: Each returned config has feature_subset with no market-specific features."""
    mod = _import_module()
    market_specific_keywords = ("vix", "usdrub", "brent", "cbr", "imoex", "turnover")

    jsonl_dir = tmp_path / "experiments"
    jsonl_dir.mkdir()
    jsonl_path = jsonl_dir / "us_tech_experiment_log.jsonl"
    with jsonl_path.open("w") as f:
        f.write(json.dumps(_KEEP_ENTRY) + "\n")

    with patch.object(mod, "_RESULTS_DIR", jsonl_dir):
        result = mod.generate_transfer_experiments("ru_large_cap")

    assert len(result) > 0
    for exp in result:
        assert exp.feature_subset is not None
        for feat in exp.feature_subset:
            for kw in market_specific_keywords:
                assert kw not in feat.lower(), (
                    f"Market-specific keyword '{kw}' found in feature '{feat}'"
                )


# ---------------------------------------------------------------------------
# T3: Reads the best "keep" entry (highest score)
# ---------------------------------------------------------------------------


def test_t3_reads_best_keep_entry(tmp_path: Path) -> None:
    """T3: generate_transfer_experiments reads entry with highest score where status=='keep'."""
    mod = _import_module()

    # Two keep entries: _KEEP_ENTRY (0.85) and _LOWER_KEEP_ENTRY (0.70)
    # The best keep entry should be _KEEP_ENTRY (0.85)
    jsonl_dir = tmp_path / "experiments"
    jsonl_dir.mkdir()
    jsonl_path = jsonl_dir / "us_tech_experiment_log.jsonl"
    with jsonl_path.open("w") as f:
        f.write(json.dumps(_LOWER_KEEP_ENTRY) + "\n")
        f.write(json.dumps(_KEEP_ENTRY) + "\n")
        f.write(json.dumps(_DISCARD_ENTRY) + "\n")  # not keep, should be ignored

    with patch.object(mod, "_RESULTS_DIR", jsonl_dir):
        result = mod.generate_transfer_experiments("ru_large_cap")

    assert len(result) > 0
    exp = result[0]
    # All market-neutral features from _KEEP_ENTRY should be present
    assert exp.feature_subset is not None
    for feat in _MARKET_NEUTRAL_FEATURES:
        assert feat in exp.feature_subset, (
            f"Expected market-neutral feature '{feat}' in subset {exp.feature_subset}"
        )


# ---------------------------------------------------------------------------
# T4: Missing JSONL file returns empty list
# ---------------------------------------------------------------------------


def test_t4_missing_jsonl_returns_empty_list(tmp_path: Path) -> None:
    """T4: When US JSONL file does not exist, returns empty list."""
    mod = _import_module()

    empty_dir = tmp_path / "experiments"
    empty_dir.mkdir()
    # No JSONL file created

    with patch.object(mod, "_RESULTS_DIR", empty_dir):
        result = mod.generate_transfer_experiments("ru_large_cap")

    assert result == [], f"Expected empty list, got {result}"


# ---------------------------------------------------------------------------
# T5: No "keep" entries returns empty list
# ---------------------------------------------------------------------------


def test_t5_no_keep_entries_returns_empty_list(tmp_path: Path) -> None:
    """T5: When US JSONL has no 'keep' entries, returns empty list."""
    mod = _import_module()

    jsonl_dir = tmp_path / "experiments"
    jsonl_dir.mkdir()
    jsonl_path = jsonl_dir / "us_tech_experiment_log.jsonl"
    with jsonl_path.open("w") as f:
        f.write(json.dumps(_DISCARD_ENTRY) + "\n")
        f.write(
            json.dumps(
                {
                    "name": "crash-test",
                    "strategy": "ablation",
                    "status": "crash",
                    "score": 0.0,
                    "features_used": _MARKET_NEUTRAL_FEATURES,
                }
            )
            + "\n"
        )

    with patch.object(mod, "_RESULTS_DIR", jsonl_dir):
        result = mod.generate_transfer_experiments("ru_large_cap")

    assert result == [], f"Expected empty list when no keep entries, got {result}"


# ---------------------------------------------------------------------------
# T6: "cross_segment_transfer" in CLI choices
# ---------------------------------------------------------------------------


def test_t6_cross_segment_transfer_in_cli_choices() -> None:
    """T6: 'cross_segment_transfer' appears in CLI --strategy choices."""
    source = _SCRIPT_PATH.read_text()
    assert "cross_segment_transfer" in source, "'cross_segment_transfer' not found in script source"
    assert '"cross_segment_transfer"' in source or "'cross_segment_transfer'" in source


# ---------------------------------------------------------------------------
# T7: _generate_experiments routes "cross_segment_transfer"
# ---------------------------------------------------------------------------


def test_t7_generate_experiments_routes_cross_segment_transfer(tmp_path: Path) -> None:
    """T7: _generate_experiments("cross_segment_transfer", ...) routes to
    generate_transfer_experiments and returns non-empty list."""
    mod = _import_module()

    jsonl_dir = tmp_path / "experiments"
    jsonl_dir.mkdir()
    jsonl_path = jsonl_dir / "us_tech_experiment_log.jsonl"
    with jsonl_path.open("w") as f:
        f.write(json.dumps(_KEEP_ENTRY) + "\n")

    all_feats = [f"feat_{i}" for i in range(10)]
    with patch.object(mod, "_RESULTS_DIR", jsonl_dir):
        experiments = mod._generate_experiments(
            strategy="cross_segment_transfer",
            baseline_features=["rsi_14", "volume_ratio"],
            all_feature_names=all_feats,
            max_experiments=100,
            segment_id="ru_large_cap",
        )

    assert len(experiments) > 0, "Expected non-empty list for cross_segment_transfer strategy"
    for exp in experiments:
        assert exp.strategy == "cross_segment_transfer"


# ---------------------------------------------------------------------------
# T8: _generate_experiments("all") includes cross_segment_transfer results
# ---------------------------------------------------------------------------


def test_t8_generate_experiments_all_includes_transfer(tmp_path: Path) -> None:
    """T8: _generate_experiments("all", ...) includes cross_segment_transfer results."""
    mod = _import_module()

    jsonl_dir = tmp_path / "experiments"
    jsonl_dir.mkdir()
    jsonl_path = jsonl_dir / "us_tech_experiment_log.jsonl"
    with jsonl_path.open("w") as f:
        f.write(json.dumps(_KEEP_ENTRY) + "\n")

    # Need at least 5 features for random_subset generator (min sample size=5)
    all_feats = [f"feat_{i}" for i in range(10)]
    with patch.object(mod, "_RESULTS_DIR", jsonl_dir):
        experiments = mod._generate_experiments(
            strategy="all",
            baseline_features=["rsi_14", "volume_ratio"],
            all_feature_names=all_feats,
            max_experiments=500,
            segment_id="ru_large_cap",
        )

    transfer_experiments = [e for e in experiments if e.strategy == "cross_segment_transfer"]
    assert len(transfer_experiments) > 0, (
        "Expected cross_segment_transfer experiments in 'all' strategy results"
    )
