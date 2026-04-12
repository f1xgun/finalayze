"""Feature complexity scoring for cost-aware feature selection (Layer 3).

Each feature gets a complexity score based on:
- Lookback window (bars of history required)
- Computation cost tier (trivial → high)
- External data dependency (self-contained → external feed)
- Redundancy group (features in the same group are partially redundant)

The efficiency of a feature is ``importance / complexity``.  Higher efficiency
means the feature provides more predictive signal per unit of complexity.
This enables Pareto-optimal feature selection: drop expensive features that
contribute little, keep cheap features that contribute a lot.

Inspired by the *simplicity criterion* from karpathy/autoresearch:
"A small improvement that adds ugly complexity is not worth it.
 Removing something and getting equal or better results is a great outcome."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import structlog

logger = structlog.get_logger(__name__)


class ComputeCost(IntEnum):
    """Computation cost tier for a single feature evaluation."""

    TRIVIAL = 1  # Simple arithmetic: returns, ratios, one-liners
    LOW = 2  # Rolling stats: SMA, std, RSI, z-scores
    MEDIUM = 3  # Multi-step: MACD, Bollinger, beta, OLS regression
    HIGH = 4  # Complex: wavelet decomposition, sparse reindex, structural break


class DataDependency(IntEnum):
    """External data requirement for feature computation."""

    SELF = 0  # Only the instrument's own OHLCV
    BENCHMARK = 1  # Needs benchmark (SPY / IMOEX) candles
    EXTERNAL = 2  # Needs external feed (VIX, FX, commodities, rates)


# Weights for combining sub-scores into a single complexity score.
_W_LOOKBACK = 0.3
_W_COMPUTE = 0.4
_W_DATA = 0.3
_MAX_LOOKBACK_BARS = 252


@dataclass(frozen=True)
class FeatureComplexity:
    """Complexity descriptor for one feature."""

    lookback_bars: int
    compute_cost: ComputeCost
    data_dependency: DataDependency
    redundancy_group: str | None = None

    @property
    def complexity_score(self) -> float:
        """Combined complexity score in [0, 1].  Higher = more complex."""
        lookback_norm = min(self.lookback_bars / _MAX_LOOKBACK_BARS, 1.0)
        compute_norm = (self.compute_cost - 1) / 3
        data_norm = self.data_dependency / 2
        return _W_LOOKBACK * lookback_norm + _W_COMPUTE * compute_norm + _W_DATA * data_norm


# ---------------------------------------------------------------------------
# Registry: feature_name -> FeatureComplexity
# ---------------------------------------------------------------------------

FEATURE_COMPLEXITY: dict[str, FeatureComplexity] = {
    # --- Core TA ---
    "rsi_14": FeatureComplexity(14, ComputeCost.LOW, DataDependency.SELF, "oscillator"),
    "macd_hist_pct": FeatureComplexity(26, ComputeCost.MEDIUM, DataDependency.SELF, "trend"),
    "bb_pct_b": FeatureComplexity(20, ComputeCost.MEDIUM, DataDependency.SELF, "volatility_band"),
    "volume_ratio_20d": FeatureComplexity(20, ComputeCost.TRIVIAL, DataDependency.SELF, "volume"),
    "atr_14_pct": FeatureComplexity(14, ComputeCost.LOW, DataDependency.SELF, "volatility"),
    # --- Momentum & Returns ---
    "ret_1d": FeatureComplexity(2, ComputeCost.TRIVIAL, DataDependency.SELF, "returns"),
    "ret_5d": FeatureComplexity(6, ComputeCost.TRIVIAL, DataDependency.SELF, "returns"),
    "ret_21d": FeatureComplexity(22, ComputeCost.TRIVIAL, DataDependency.SELF, "returns"),
    "ret_63d": FeatureComplexity(64, ComputeCost.TRIVIAL, DataDependency.SELF, "returns"),
    "ret_126d": FeatureComplexity(127, ComputeCost.TRIVIAL, DataDependency.SELF, "returns"),
    "mom_reversal_ratio": FeatureComplexity(21, ComputeCost.LOW, DataDependency.SELF, "momentum"),
    "roc_10": FeatureComplexity(11, ComputeCost.TRIVIAL, DataDependency.SELF, "momentum"),
    "rsi_2": FeatureComplexity(3, ComputeCost.LOW, DataDependency.SELF, "oscillator"),
    "rsi_5": FeatureComplexity(6, ComputeCost.LOW, DataDependency.SELF, "oscillator"),
    "willr_14": FeatureComplexity(14, ComputeCost.LOW, DataDependency.SELF, "oscillator"),
    "obv_slope_10": FeatureComplexity(11, ComputeCost.LOW, DataDependency.SELF, "volume"),
    # --- Volatility & Dispersion ---
    "hist_vol_20": FeatureComplexity(20, ComputeCost.LOW, DataDependency.SELF, "volatility"),
    "gk_vol_20": FeatureComplexity(20, ComputeCost.MEDIUM, DataDependency.SELF, "volatility"),
    "adx_14": FeatureComplexity(28, ComputeCost.MEDIUM, DataDependency.SELF, "trend"),
    "skew_20d": FeatureComplexity(20, ComputeCost.LOW, DataDependency.SELF, "distribution"),
    "kurt_20d": FeatureComplexity(20, ComputeCost.LOW, DataDependency.SELF, "distribution"),
    "max_ret_20d": FeatureComplexity(20, ComputeCost.TRIVIAL, DataDependency.SELF, "distribution"),
    "min_ret_20d": FeatureComplexity(20, ComputeCost.TRIVIAL, DataDependency.SELF, "distribution"),
    # --- Microstructure ---
    "proximity_rolling_high": FeatureComplexity(
        252, ComputeCost.TRIVIAL, DataDependency.SELF, "price_level"
    ),
    "amihud_20d": FeatureComplexity(252, ComputeCost.MEDIUM, DataDependency.SELF, "liquidity"),
    "corwin_schultz_spread": FeatureComplexity(
        2, ComputeCost.MEDIUM, DataDependency.SELF, "liquidity"
    ),
    # --- Z-scores ---
    "price_zscore_60d": FeatureComplexity(60, ComputeCost.LOW, DataDependency.SELF, "zscore"),
    "volume_zscore_20d": FeatureComplexity(20, ComputeCost.LOW, DataDependency.SELF, "zscore"),
    "rsi_zscore_60d": FeatureComplexity(60, ComputeCost.LOW, DataDependency.SELF, "zscore"),
    "atr_zscore_60d": FeatureComplexity(60, ComputeCost.LOW, DataDependency.SELF, "zscore"),
    # --- Wavelet ---
    "wavelet_approx_energy": FeatureComplexity(
        16, ComputeCost.HIGH, DataDependency.SELF, "wavelet"
    ),
    "wavelet_detail1_energy": FeatureComplexity(
        16, ComputeCost.HIGH, DataDependency.SELF, "wavelet"
    ),
    "wavelet_detail2_energy": FeatureComplexity(
        16, ComputeCost.HIGH, DataDependency.SELF, "wavelet"
    ),
    "wavelet_detail3_energy": FeatureComplexity(
        16, ComputeCost.HIGH, DataDependency.SELF, "wavelet"
    ),
    # --- Calendar & Regime ---
    "month_sin": FeatureComplexity(1, ComputeCost.TRIVIAL, DataDependency.SELF, "calendar"),
    "month_cos": FeatureComplexity(1, ComputeCost.TRIVIAL, DataDependency.SELF, "calendar"),
    "vix_level": FeatureComplexity(3, ComputeCost.TRIVIAL, DataDependency.EXTERNAL, "regime"),
    "vix_percentile_252d": FeatureComplexity(
        252, ComputeCost.LOW, DataDependency.EXTERNAL, "regime"
    ),
    "vix_change_5d": FeatureComplexity(7, ComputeCost.TRIVIAL, DataDependency.EXTERNAL, "regime"),
    "realized_vol_ratio": FeatureComplexity(60, ComputeCost.LOW, DataDependency.SELF, "volatility"),
    # --- Cross-Asset ---
    "relative_strength_21d": FeatureComplexity(
        22, ComputeCost.LOW, DataDependency.BENCHMARK, "cross_asset"
    ),
    "rolling_beta_63d": FeatureComplexity(
        63, ComputeCost.MEDIUM, DataDependency.BENCHMARK, "cross_asset"
    ),
    "rolling_corr_63d": FeatureComplexity(
        63, ComputeCost.MEDIUM, DataDependency.BENCHMARK, "cross_asset"
    ),
    "excess_momentum_score": FeatureComplexity(
        63, ComputeCost.MEDIUM, DataDependency.BENCHMARK, "cross_asset"
    ),
    # --- RSI Divergence ---
    "rsi_divergence": FeatureComplexity(14, ComputeCost.MEDIUM, DataDependency.SELF, "oscillator"),
    # --- MOEX External ---
    "usdrub_zscore_60d": FeatureComplexity(60, ComputeCost.HIGH, DataDependency.EXTERNAL, "fx"),
    "usdrub_return": FeatureComplexity(2, ComputeCost.LOW, DataDependency.EXTERNAL, "fx"),
    "usdrub_vol": FeatureComplexity(20, ComputeCost.LOW, DataDependency.EXTERNAL, "fx"),
    "brent_zscore_60d": FeatureComplexity(
        60, ComputeCost.HIGH, DataDependency.EXTERNAL, "commodity"
    ),
    "brent_return": FeatureComplexity(2, ComputeCost.LOW, DataDependency.EXTERNAL, "commodity"),
    "real_rate_zscore": FeatureComplexity(252, ComputeCost.HIGH, DataDependency.EXTERNAL, "macro"),
    "market_turnover_zscore": FeatureComplexity(
        60, ComputeCost.HIGH, DataDependency.EXTERNAL, "moex_market"
    ),
    "cbr_rate_level": FeatureComplexity(1, ComputeCost.LOW, DataDependency.EXTERNAL, "macro"),
    "cbr_rate_delta": FeatureComplexity(2, ComputeCost.LOW, DataDependency.EXTERNAL, "macro"),
    "cbr_direction_cut": FeatureComplexity(
        2, ComputeCost.TRIVIAL, DataDependency.EXTERNAL, "macro"
    ),
    "cbr_direction_hike": FeatureComplexity(
        2, ComputeCost.TRIVIAL, DataDependency.EXTERNAL, "macro"
    ),
    # --- Multi-Timeframe ---
    "weekly_rsi_14": FeatureComplexity(70, ComputeCost.MEDIUM, DataDependency.SELF, "multi_tf"),
    "weekly_sma_50_ratio": FeatureComplexity(
        250, ComputeCost.MEDIUM, DataDependency.SELF, "multi_tf"
    ),
    "monthly_trend_direction": FeatureComplexity(
        63, ComputeCost.MEDIUM, DataDependency.SELF, "multi_tf"
    ),
}

# Default for unknown features — assume worst-case
_UNKNOWN_COMPLEXITY = FeatureComplexity(252, ComputeCost.HIGH, DataDependency.EXTERNAL)

_MIN_COMPLEXITY = 0.01  # floor to avoid division by zero in efficiency


def get_complexity(feature_name: str) -> FeatureComplexity:
    """Get complexity for a feature.  Returns HIGH for unknown features."""
    return FEATURE_COMPLEXITY.get(feature_name, _UNKNOWN_COMPLEXITY)


def compute_efficiency(feature_name: str, importance: float) -> float:
    """Efficiency = importance / complexity_score.

    Higher efficiency means the feature provides more signal per unit of cost.
    """
    score = max(get_complexity(feature_name).complexity_score, _MIN_COMPLEXITY)
    return importance / score


def rank_by_efficiency(
    importances: dict[str, float],
) -> list[tuple[str, float, float, float]]:
    """Rank features by efficiency (descending).

    Returns:
        List of (feature_name, importance, complexity_score, efficiency).
    """
    rows: list[tuple[str, float, float, float]] = []
    for name, imp in importances.items():
        cx = get_complexity(name).complexity_score
        eff = imp / max(cx, _MIN_COMPLEXITY)
        rows.append((name, imp, cx, eff))
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows


def select_features_pareto(
    importances: dict[str, float],
    *,
    max_features: int = 15,
    min_efficiency: float = 0.0,
    max_total_complexity: float | None = None,
) -> list[str]:
    """Select features using Pareto-optimal efficiency.

    Greedily picks features in descending efficiency order until
    ``max_features`` or ``max_total_complexity`` budget is exhausted.

    Args:
        importances: Feature name -> importance score (MI or gain).
        max_features: Maximum features to select.
        min_efficiency: Skip features below this efficiency threshold.
        max_total_complexity: Optional complexity budget (sum of scores).
            When *None*, no budget constraint is applied.

    Returns:
        Selected feature names ordered by efficiency (descending).
    """
    ranked = rank_by_efficiency(importances)
    selected: list[str] = []
    total_cx = 0.0

    for name, _imp, cx, eff in ranked:
        if len(selected) >= max_features:
            break
        if eff < min_efficiency:
            continue
        if max_total_complexity is not None and total_cx + cx > max_total_complexity:
            continue
        selected.append(name)
        total_cx += cx

    logger.info(
        "pareto_feature_selection",
        selected=len(selected),
        total_complexity=round(total_cx, 3),
        budget=max_total_complexity,
    )
    return selected


def summarize_complexity(feature_names: list[str]) -> dict[str, float]:
    """Return aggregate complexity statistics for a feature set.

    Useful for comparing feature sets across experiments.

    Returns:
        Dict with keys: total, mean, max, n_external, n_high_compute.
    """
    if not feature_names:
        return {
            "total": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "n_external": 0,
            "n_high_compute": 0,
        }
    complexities = [get_complexity(n) for n in feature_names]
    scores = [c.complexity_score for c in complexities]
    return {
        "total": round(sum(scores), 4),
        "mean": round(sum(scores) / len(scores), 4),
        "max": round(max(scores), 4),
        "n_external": sum(1 for c in complexities if c.data_dependency == DataDependency.EXTERNAL),
        "n_high_compute": sum(1 for c in complexities if c.compute_cost == ComputeCost.HIGH),
    }
