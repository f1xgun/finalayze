"""EVT (Extreme Value Theory) tail risk estimation (Layer 4).

Fits a Generalized Pareto Distribution (GPD) to losses exceeding a high
quantile threshold, then computes VaR and Expected Shortfall dynamically
per asset.  This replaces fixed drawdown thresholds with statistically
calibrated tail-risk measures.

Reference:
    McNeil & Frey (2000), "Estimation of tail-related risk measures for
    heteroscedastic financial time series: an extreme value approach."

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto

# Minimum number of exceedances required for a reliable GPD fit.
_MIN_EXCEEDANCES = 30

# Threshold for treating GPD shape parameter as zero (exponential case).
_XI_ZERO_THRESHOLD = 1e-10


@dataclass(frozen=True, slots=True)
class EVTFit:
    """Result of fitting a GPD to tail losses.

    Attributes:
        shape: GPD shape parameter (xi).  xi > 0 => heavy tail (Frechet).
        scale: GPD scale parameter (sigma).
        threshold: The loss threshold (positive) above which the GPD was fit.
        n_exceedances: Number of observations exceeding the threshold.
    """

    shape: float
    scale: float
    threshold: float
    n_exceedances: int
    n_total: int


class EVTRiskEstimator:
    """Estimate tail risk via Peaks-Over-Threshold GPD fitting.

    Typical usage::

        estimator = EVTRiskEstimator()
        fit = estimator.fit(returns)
        if fit is not None:
            var = estimator.var_evt(fit, confidence=0.99)
            es  = estimator.es_evt(fit, confidence=0.99)
    """

    def fit(
        self,
        returns: list[float],
        threshold_quantile: float = 0.95,
    ) -> EVTFit | None:
        """Fit GPD to losses exceeding the *threshold_quantile*.

        Losses are defined as ``-r`` for each return ``r < 0``.
        The threshold is the ``threshold_quantile``-th quantile of these
        losses, and the GPD is fit to exceedances above that threshold.

        Args:
            returns: Historical return series (can be positive and negative).
            threshold_quantile: Quantile (0..1) of losses to use as threshold.

        Returns:
            An :class:`EVTFit` if at least ``_MIN_EXCEEDANCES`` observations
            exceed the threshold, otherwise ``None``.
        """
        losses = np.array([-r for r in returns if r < 0])
        if len(losses) < _MIN_EXCEEDANCES:
            return None

        threshold = float(np.quantile(losses, threshold_quantile))
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < _MIN_EXCEEDANCES:
            return None

        # MLE fit of GPD to exceedances
        shape, _loc, scale = genpareto.fit(exceedances, floc=0)

        return EVTFit(
            shape=float(shape),
            scale=float(scale),
            threshold=threshold,
            n_exceedances=len(exceedances),
            n_total=len(losses),
        )

    def var_evt(self, fit: EVTFit, confidence: float = 0.99) -> float:
        """Compute Value-at-Risk using the fitted GPD.

        Uses the Peaks-Over-Threshold formula:

            VaR_p = u + (sigma / xi) * ((n/Nu * (1-p))^(-xi) - 1)

        simplified for exceedances-only fit to:

            VaR_p = u + sigma/xi * (((1-p)/p_exceed)^(-xi) - 1)

        where ``p_exceed`` is the probability of exceeding the threshold
        in the loss distribution, and ``p = confidence``.

        Args:
            fit: A fitted :class:`EVTFit`.
            confidence: Confidence level (e.g. 0.99 for 99% VaR).

        Returns:
            VaR as a positive loss magnitude.
        """
        xi = fit.shape
        sigma = fit.scale
        u = fit.threshold

        tail_prob = 1.0 - confidence
        p_exceed = fit.n_exceedances / fit.n_total

        # Standard POT-VaR (McNeil & Frey 2000):
        #   VaR_p = u + (sigma/xi) * [(tail_prob / p_exceed)^(-xi) - 1]
        if abs(xi) < _XI_ZERO_THRESHOLD:
            # Exponential case (xi -> 0):
            #   VaR_p = u + sigma * ln(p_exceed / tail_prob)
            var = u + sigma * np.log(p_exceed / tail_prob)
        else:
            var = u + (sigma / xi) * ((tail_prob / p_exceed) ** (-xi) - 1.0)

        return float(var)

    def es_evt(self, fit: EVTFit, confidence: float = 0.99) -> float:
        """Compute Expected Shortfall (CVaR) using the fitted GPD.

        ES = VaR / (1 - xi) + (sigma - xi * u) / (1 - xi)

        Simplified for the POT model:

            ES = VaR/(1-xi) + (sigma - xi * threshold) / (1-xi)

        Args:
            fit: A fitted :class:`EVTFit`.
            confidence: Confidence level (e.g. 0.99).

        Returns:
            Expected Shortfall as a positive loss magnitude (>= VaR).
        """
        xi = fit.shape
        sigma = fit.scale
        var = self.var_evt(fit, confidence=confidence)

        if abs(xi) < _XI_ZERO_THRESHOLD:
            # Exponential case
            es = var + sigma
        elif xi < 1.0:
            # Standard GPD ES formula:
            # ES = VaR/(1-xi) + (sigma - xi * u) / (1-xi)
            es = var / (1.0 - xi) + (sigma - xi * fit.threshold) / (1.0 - xi)
        else:
            # xi >= 1: ES is infinite in theory; return a large multiple of VaR
            es = var * 2.0  # pragmatic fallback

        return float(es)

    def is_tail_risk_elevated(
        self,
        returns: list[float],
        current_loss: float,
        confidence: float = 0.99,
    ) -> bool:
        """Check whether *current_loss* exceeds the dynamic EVT VaR.

        This is a convenience helper that fits the GPD and compares.

        Args:
            returns: Historical return series.
            current_loss: The current period loss (positive = loss).
            confidence: VaR confidence level.

        Returns:
            ``True`` if the loss exceeds VaR; ``False`` otherwise
            (including when the GPD cannot be fit -- fail-open).
        """
        if current_loss <= 0:
            return False

        fit = self.fit(returns)
        if fit is None:
            return False

        var = self.var_evt(fit, confidence=confidence)
        return current_loss > var
