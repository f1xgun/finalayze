"""Unit tests for EVT (Extreme Value Theory) tail risk estimation."""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto

from finalayze.risk.evt import EVTFit, EVTRiskEstimator

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────────
THRESHOLD_QUANTILE = 0.95
CONFIDENCE_99 = 0.99
CONFIDENCE_95 = 0.95
MIN_EXCEEDANCES = 30

# GPD parameters for synthetic data
GPD_SHAPE = 0.2  # xi > 0: heavy tail (Frechet-like)
GPD_SCALE = 0.02  # sigma
SAMPLE_SIZE = 2000
SEED = 42

# For insufficient data test
SMALL_SAMPLE_SIZE = 50

# Tolerance for numerical comparisons
RTOL = 0.3  # 30% relative tolerance for statistical estimates
ABS_TOL = 0.005  # absolute tolerance for small values


def _generate_returns_with_gpd_tail(
    n: int = SAMPLE_SIZE,
    shape: float = GPD_SHAPE,
    scale: float = GPD_SCALE,
    seed: int = SEED,
) -> list[float]:
    """Generate synthetic returns where losses follow GPD in the tail."""
    rng = np.random.default_rng(seed)
    # Normal bulk returns
    bulk = rng.normal(loc=0.0005, scale=0.01, size=n)
    # Replace worst 10% with GPD-distributed losses
    n_tail = n // 10
    tail_losses = genpareto.rvs(shape, scale=scale, size=n_tail, random_state=rng)
    # Make them negative (losses)
    bulk[:n_tail] = -tail_losses
    rng.shuffle(bulk)
    return bulk.tolist()


class TestEVTFit:
    """Test the EVTFit dataclass."""

    def test_evtfit_fields(self) -> None:
        n_total = 1000
        fit = EVTFit(
            shape=GPD_SHAPE,
            scale=GPD_SCALE,
            threshold=0.01,
            n_exceedances=50,
            n_total=n_total,
        )
        assert fit.shape == GPD_SHAPE
        assert fit.scale == GPD_SCALE
        assert fit.threshold == 0.01
        assert fit.n_exceedances == 50
        assert fit.n_total == n_total


class TestEVTRiskEstimatorFit:
    """Test GPD fitting."""

    def test_fit_returns_evtfit(self) -> None:
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is not None
        assert isinstance(fit, EVTFit)

    def test_fit_shape_positive_for_heavy_tail(self) -> None:
        """GPD shape (xi) should be positive for heavy-tailed data."""
        returns = _generate_returns_with_gpd_tail(shape=GPD_SHAPE)
        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is not None
        assert fit.shape > 0

    def test_fit_scale_positive(self) -> None:
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is not None
        assert fit.scale > 0

    def test_fit_n_exceedances_above_minimum(self) -> None:
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is not None
        assert fit.n_exceedances >= MIN_EXCEEDANCES

    def test_fit_returns_none_insufficient_data(self) -> None:
        """With very few data points, fewer than 30 exceedances -> None."""
        returns = _generate_returns_with_gpd_tail(n=SMALL_SAMPLE_SIZE)
        estimator = EVTRiskEstimator()
        # With 50 samples and 95th percentile, only ~2-3 exceedances
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is None

    def test_fit_threshold_is_positive(self) -> None:
        """The threshold should be a positive loss value."""
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is not None
        assert fit.threshold > 0


class TestVaR:
    """Test Value-at-Risk computation."""

    def test_var_is_positive(self) -> None:
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is not None
        var = estimator.var_evt(fit, confidence=CONFIDENCE_99)
        assert var > 0

    def test_var_99_exceeds_var_95(self) -> None:
        """Higher confidence should yield higher VaR."""
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is not None
        var_99 = estimator.var_evt(fit, confidence=CONFIDENCE_99)
        var_95 = estimator.var_evt(fit, confidence=CONFIDENCE_95)
        assert var_99 > var_95

    def test_var_reasonable_magnitude(self) -> None:
        """VaR at 99% should be in a plausible range for our synthetic data."""
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is not None
        var = estimator.var_evt(fit, confidence=CONFIDENCE_99)
        min_var = 0.005
        max_var = 0.50
        assert min_var < var < max_var


class TestExpectedShortfall:
    """Test Expected Shortfall (CVaR) computation."""

    def test_es_exceeds_var(self) -> None:
        """ES must always be >= VaR (by definition)."""
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is not None
        var = estimator.var_evt(fit, confidence=CONFIDENCE_99)
        es = estimator.es_evt(fit, confidence=CONFIDENCE_99)
        assert es >= var

    def test_es_is_positive(self) -> None:
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is not None
        es = estimator.es_evt(fit, confidence=CONFIDENCE_99)
        assert es > 0

    def test_es_99_exceeds_es_95(self) -> None:
        """Higher confidence should yield higher ES."""
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns, threshold_quantile=THRESHOLD_QUANTILE)
        assert fit is not None
        es_99 = estimator.es_evt(fit, confidence=CONFIDENCE_99)
        es_95 = estimator.es_evt(fit, confidence=CONFIDENCE_95)
        assert es_99 > es_95


class TestIsTailRiskElevated:
    """Test the helper that checks if current loss exceeds dynamic VaR."""

    def test_extreme_loss_is_elevated(self) -> None:
        """A very large loss should be flagged as elevated tail risk."""
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        extreme_loss = 0.50  # 50% loss -- well into the tail
        result = estimator.is_tail_risk_elevated(returns, current_loss=extreme_loss)
        assert result is True

    def test_normal_loss_not_elevated(self) -> None:
        """A small loss within normal range should NOT be flagged."""
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        small_loss = 0.001  # 0.1% loss
        result = estimator.is_tail_risk_elevated(returns, current_loss=small_loss)
        assert result is False

    def test_insufficient_data_returns_false(self) -> None:
        """When GPD cannot be fit (too few data), return False (fail-open)."""
        estimator = EVTRiskEstimator()
        short_returns = [0.01, -0.005, 0.003] * 10  # only 30 points
        result = estimator.is_tail_risk_elevated(short_returns, current_loss=0.10)
        assert result is False

    def test_zero_loss_not_elevated(self) -> None:
        returns = _generate_returns_with_gpd_tail()
        estimator = EVTRiskEstimator()
        result = estimator.is_tail_risk_elevated(returns, current_loss=0.0)
        assert result is False


class TestEdgeCases:
    """Test edge cases and branch coverage for EVT estimator."""

    def test_fit_enough_losses_but_few_exceedances(self) -> None:
        """Many losses but very high threshold quantile -> too few exceedances."""
        # Generate returns where most losses are identical (no tail variation)
        rng = np.random.default_rng(SEED)
        # 500 negative returns all clustered near -0.01
        losses = rng.uniform(-0.012, -0.008, size=500).tolist()
        gains = rng.uniform(0.001, 0.005, size=500).tolist()
        returns = losses + gains
        estimator = EVTRiskEstimator()
        # With 0.99 quantile, only ~5 exceedances out of 500 losses -> None
        high_quantile = 0.99
        fit = estimator.fit(returns, threshold_quantile=high_quantile)
        assert fit is None

    def test_var_exponential_case_xi_zero(self) -> None:
        """When shape (xi) is ~0, VaR uses the exponential formula."""
        # Construct an EVTFit with shape = 0 (exponential tail)
        xi_zero = 0.0
        scale = 0.02
        threshold = 0.01
        n_exc = 50
        n_total = 1000
        fit = EVTFit(
            shape=xi_zero,
            scale=scale,
            threshold=threshold,
            n_exceedances=n_exc,
            n_total=n_total,
        )
        estimator = EVTRiskEstimator()
        var = estimator.var_evt(fit, confidence=CONFIDENCE_99)
        # Exponential formula: u + sigma * ln(p_exceed / tail_prob)
        p_exceed = n_exc / n_total
        expected = threshold + scale * np.log(p_exceed / (1.0 - CONFIDENCE_99))
        assert abs(var - expected) < ABS_TOL

    def test_es_exponential_case_xi_zero(self) -> None:
        """When shape (xi) is ~0, ES uses the exponential formula."""
        xi_zero = 0.0
        scale = 0.02
        threshold = 0.01
        n_exc = 50
        n_total = 1000
        fit = EVTFit(
            shape=xi_zero,
            scale=scale,
            threshold=threshold,
            n_exceedances=n_exc,
            n_total=n_total,
        )
        estimator = EVTRiskEstimator()
        es = estimator.es_evt(fit, confidence=CONFIDENCE_99)
        var = estimator.var_evt(fit, confidence=CONFIDENCE_99)
        # Exponential ES = VaR + sigma
        expected = var + scale
        assert abs(es - expected) < ABS_TOL

    def test_es_extreme_heavy_tail_xi_ge_one(self) -> None:
        """When xi >= 1, ES is theoretically infinite; fallback = 2 * VaR."""
        xi_extreme = 1.5
        scale = 0.02
        threshold = 0.01
        n_exc = 50
        n_total = 1000
        fit = EVTFit(
            shape=xi_extreme,
            scale=scale,
            threshold=threshold,
            n_exceedances=n_exc,
            n_total=n_total,
        )
        estimator = EVTRiskEstimator()
        var = estimator.var_evt(fit, confidence=CONFIDENCE_99)
        es = estimator.es_evt(fit, confidence=CONFIDENCE_99)
        es_multiplier = 2.0
        expected = var * es_multiplier
        assert abs(es - expected) < ABS_TOL
        assert es >= var
