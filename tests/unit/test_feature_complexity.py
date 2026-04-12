"""Tests for feature complexity scoring and efficiency-weighted selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finalayze.ml.training.feature_complexity import (
    FEATURE_COMPLEXITY,
    ComputeCost,
    DataDependency,
    FeatureComplexity,
    compute_efficiency,
    get_complexity,
    rank_by_efficiency,
    select_features_pareto,
    summarize_complexity,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_KNOWN_FEATURE = "rsi_14"
_UNKNOWN_FEATURE = "totally_unknown_feature_xyz"
_N_SAMPLES = 200


# ---------------------------------------------------------------------------
# FeatureComplexity dataclass
# ---------------------------------------------------------------------------


class TestFeatureComplexity:
    """Tests for the FeatureComplexity dataclass."""

    def test_trivial_self_is_low_complexity(self) -> None:
        fc = FeatureComplexity(1, ComputeCost.TRIVIAL, DataDependency.SELF)
        assert fc.complexity_score < 0.05

    def test_high_external_is_high_complexity(self) -> None:
        fc = FeatureComplexity(252, ComputeCost.HIGH, DataDependency.EXTERNAL)
        assert fc.complexity_score > 0.9

    def test_score_in_zero_one(self) -> None:
        for fc in FEATURE_COMPLEXITY.values():
            score = fc.complexity_score
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"

    def test_lookback_capped_at_252(self) -> None:
        fc = FeatureComplexity(500, ComputeCost.TRIVIAL, DataDependency.SELF)
        # lookback contribution should be capped at 1.0 (252/252)
        fc_252 = FeatureComplexity(252, ComputeCost.TRIVIAL, DataDependency.SELF)
        assert fc.complexity_score == fc_252.complexity_score

    def test_frozen_dataclass(self) -> None:
        fc = FeatureComplexity(14, ComputeCost.LOW, DataDependency.SELF)
        with pytest.raises(AttributeError):
            fc.lookback_bars = 20  # type: ignore[misc]

    def test_redundancy_group_optional(self) -> None:
        fc = FeatureComplexity(10, ComputeCost.LOW, DataDependency.SELF)
        assert fc.redundancy_group is None
        fc2 = FeatureComplexity(10, ComputeCost.LOW, DataDependency.SELF, "oscillator")
        assert fc2.redundancy_group == "oscillator"


# ---------------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the FEATURE_COMPLEXITY registry."""

    def test_known_features_are_registered(self) -> None:
        known = ["rsi_14", "ret_1d", "macd_hist_pct", "vix_level", "wavelet_approx_energy"]
        for name in known:
            assert name in FEATURE_COMPLEXITY, f"{name} not in registry"

    def test_get_complexity_known(self) -> None:
        fc = get_complexity(_KNOWN_FEATURE)
        assert fc.lookback_bars == 14
        assert fc.compute_cost == ComputeCost.LOW
        assert fc.data_dependency == DataDependency.SELF

    def test_get_complexity_unknown_returns_worst_case(self) -> None:
        fc = get_complexity(_UNKNOWN_FEATURE)
        assert fc.lookback_bars == 252
        assert fc.compute_cost == ComputeCost.HIGH
        assert fc.data_dependency == DataDependency.EXTERNAL

    def test_registry_covers_main_features(self) -> None:
        # At least 40 features should be registered
        min_expected = 40
        assert len(FEATURE_COMPLEXITY) >= min_expected


# ---------------------------------------------------------------------------
# Efficiency scoring
# ---------------------------------------------------------------------------


class TestEfficiency:
    """Tests for compute_efficiency and rank_by_efficiency."""

    def test_higher_importance_gives_higher_efficiency(self) -> None:
        eff_low = compute_efficiency(_KNOWN_FEATURE, 0.01)
        eff_high = compute_efficiency(_KNOWN_FEATURE, 0.10)
        assert eff_high > eff_low

    def test_simpler_feature_has_higher_efficiency_at_same_importance(self) -> None:
        # ret_1d (trivial, 2 bars) vs wavelet_approx_energy (high compute, 16 bars)
        eff_simple = compute_efficiency("ret_1d", 0.05)
        eff_complex = compute_efficiency("wavelet_approx_energy", 0.05)
        assert eff_simple > eff_complex

    def test_zero_importance_gives_zero_efficiency(self) -> None:
        eff = compute_efficiency(_KNOWN_FEATURE, 0.0)
        assert eff == 0.0

    def test_rank_by_efficiency_sorted_descending(self) -> None:
        importances = {"ret_1d": 0.10, "rsi_14": 0.08, "wavelet_approx_energy": 0.05}
        ranked = rank_by_efficiency(importances)
        efficiencies = [r[3] for r in ranked]
        assert efficiencies == sorted(efficiencies, reverse=True)

    def test_rank_by_efficiency_contains_all_features(self) -> None:
        importances = {"ret_1d": 0.10, "rsi_14": 0.08}
        ranked = rank_by_efficiency(importances)
        names = {r[0] for r in ranked}
        assert names == {"ret_1d", "rsi_14"}


# ---------------------------------------------------------------------------
# Pareto-optimal selection
# ---------------------------------------------------------------------------


class TestParetoSelection:
    """Tests for select_features_pareto."""

    def test_respects_max_features(self) -> None:
        importances = {f"feat_{i}": 0.1 - i * 0.01 for i in range(10)}
        max_feats = 5
        selected = select_features_pareto(importances, max_features=max_feats)
        assert len(selected) <= max_feats

    def test_empty_importances(self) -> None:
        selected = select_features_pareto({})
        assert selected == []

    def test_min_efficiency_filters(self) -> None:
        importances = {"ret_1d": 0.10, "vix_level": 0.001}
        # vix_level has external dependency → higher complexity → lower efficiency
        selected = select_features_pareto(importances, min_efficiency=0.5)
        # ret_1d should survive, vix_level likely filtered
        assert "ret_1d" in selected

    def test_complexity_budget(self) -> None:
        importances = dict.fromkeys(list(FEATURE_COMPLEXITY.keys())[:20], 0.05)
        # Very tight budget should select fewer features
        tight = select_features_pareto(importances, max_total_complexity=1.0)
        loose = select_features_pareto(importances, max_total_complexity=10.0)
        assert len(tight) <= len(loose)

    def test_prefers_efficient_features(self) -> None:
        importances = {
            "ret_1d": 0.05,  # trivial, self → high efficiency
            "real_rate_zscore": 0.05,  # high compute, external → low efficiency
        }
        selected = select_features_pareto(importances, max_features=1)
        assert selected == ["ret_1d"]


# ---------------------------------------------------------------------------
# Complexity summary
# ---------------------------------------------------------------------------


class TestSummarizeComplexity:
    """Tests for summarize_complexity."""

    def test_empty_features(self) -> None:
        summary = summarize_complexity([])
        assert summary["total"] == 0.0
        assert summary["n_external"] == 0

    def test_single_feature(self) -> None:
        summary = summarize_complexity(["rsi_14"])
        assert summary["total"] > 0
        assert summary["mean"] == summary["total"]
        assert summary["max"] == summary["total"]

    def test_counts_external(self) -> None:
        summary = summarize_complexity(["rsi_14", "vix_level", "brent_zscore_60d"])
        assert summary["n_external"] == 2  # vix_level and brent_zscore_60d

    def test_counts_high_compute(self) -> None:
        features = ["wavelet_approx_energy", "wavelet_detail1_energy", "ret_1d"]
        summary = summarize_complexity(features)
        assert summary["n_high_compute"] == 2


# ---------------------------------------------------------------------------
# Integration: select_features_efficient
# ---------------------------------------------------------------------------


class TestSelectFeaturesEfficient:
    """Tests for the efficiency-weighted MI feature selection."""

    @staticmethod
    def _make_data(
        n: int = _N_SAMPLES,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create data with features of varying predictive power."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(n)
        data = {
            "ret_1d": signal + rng.standard_normal(n) * 0.5,  # strong signal, trivial
            "rsi_14": signal + rng.standard_normal(n) * 1.0,  # medium signal, low cost
            "wavelet_approx_energy": rng.standard_normal(n),  # noise, high cost
            "vix_level": signal + rng.standard_normal(n) * 2.0,  # weak signal, external
            "constant": np.ones(n),  # zero MI
        }
        labels = (signal > 0).astype(int)
        return pd.DataFrame(data), pd.Series(labels)

    def test_returns_nonempty(self) -> None:
        from finalayze.ml.training.feature_selection import select_features_efficient

        x, y = self._make_data()
        selected = select_features_efficient(x, y, max_features=3)
        assert len(selected) > 0

    def test_prefers_cheap_informative(self) -> None:
        from finalayze.ml.training.feature_selection import select_features_efficient

        x, y = self._make_data()
        selected = select_features_efficient(x, y, max_features=2)
        # ret_1d (trivial, strong signal) should be first
        assert selected[0] == "ret_1d"

    def test_respects_max_features(self) -> None:
        from finalayze.ml.training.feature_selection import select_features_efficient

        x, y = self._make_data()
        max_f = 2
        selected = select_features_efficient(x, y, max_features=max_f)
        assert len(selected) <= max_f

    def test_empty_dataframe(self) -> None:
        from finalayze.ml.training.feature_selection import select_features_efficient

        x = pd.DataFrame()
        y = pd.Series(dtype=int)
        selected = select_features_efficient(x, y)
        assert selected == []

    def test_min_features_floor(self) -> None:
        from finalayze.ml.training.feature_selection import select_features_efficient

        x, y = self._make_data()
        # With very high MI threshold, floor should still give min_features
        selected = select_features_efficient(
            x,
            y,
            mi_threshold=999.0,
            min_features=3,
        )
        assert len(selected) >= 3
