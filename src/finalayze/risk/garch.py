"""GJR-GARCH(1,1,1) volatility forecasting (Layer 4).

Provides GJR-GARCH volatility forecasting as an alternative to rolling
realized volatility. The GJR extension captures leverage effects where
negative returns tend to increase future volatility more than positive returns.

Usage:
    forecaster = GJRGarchForecaster(p=1, o=1, q=1)
    vol = forecaster.fit_forecast(returns, horizon=1)
    # or use the safe wrapper with fallback:
    vol = forecaster.fit_forecast_safe(returns, fallback_vol=0.20)
    # or the convenience function:
    vol = forecast_garch_vol(returns)
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import structlog
from arch import arch_model

_log = structlog.get_logger()

_MIN_RETURNS = 30
_MIN_RETURNS_REALIZED = 2
_ANNUALIZATION_FACTOR = 252


def _rolling_vol_fallback(returns: list[float]) -> float:
    """Compute annualized rolling volatility as std(returns) * sqrt(252).

    Returns NaN if fewer than _MIN_RETURNS_REALIZED data points.
    """
    if len(returns) < _MIN_RETURNS_REALIZED:
        return float("nan")
    arr = np.array(returns, dtype=np.float64)
    return float(np.std(arr)) * math.sqrt(_ANNUALIZATION_FACTOR)


class GJRGarchForecaster:
    """GJR-GARCH volatility forecaster.

    Fits a GJR-GARCH(p,o,q) model with zero-mean specification and returns
    annualized volatility forecasts. The 'o' parameter controls the asymmetric
    (leverage) term that captures the tendency for negative returns to have a
    larger impact on future volatility.

    Args:
        p: GARCH lag order (default 1).
        o: Asymmetric (GJR) lag order (default 1).
        q: ARCH lag order (default 1).
    """

    def __init__(self, p: int = 1, o: int = 1, q: int = 1) -> None:
        self._p = p
        self._o = o
        self._q = q

    def fit_forecast(self, returns: list[float], horizon: int = 1) -> float:
        """Fit GJR-GARCH model and return annualized vol forecast.

        Args:
            returns: List of simple returns (e.g. daily log returns or pct returns).
            horizon: Forecast horizon in periods (default 1).

        Returns:
            Annualized volatility forecast (float). Returns rolling vol fallback
            when model fails or data is insufficient (but >= 2 points).
            Returns NaN only when fewer than 2 returns.
        """
        if len(returns) < _MIN_RETURNS:
            _log.warning(
                "garch_insufficient_data",
                n_returns=len(returns),
                fallback="rolling_vol",
            )
            return _rolling_vol_fallback(returns)

        try:
            data = np.array(returns, dtype=np.float64) * 100  # arch expects pct scale

            model = arch_model(
                data,
                vol="GARCH",
                p=self._p,
                o=self._o,
                q=self._q,
                mean="Zero",
                rescale=False,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(disp="off", show_warning=False)

            forecasts = result.forecast(horizon=horizon)
            # variance is in pct^2 scale; take last row, last horizon column
            variance_pct2 = float(forecasts.variance.iloc[-1, horizon - 1])  # type: ignore[arg-type]

            if not math.isfinite(variance_pct2) or variance_pct2 <= 0:
                _log.warning("garch_invalid_variance", fallback="rolling_vol")
                return _rolling_vol_fallback(returns)

            # Convert from pct scale back to decimal, then annualize
            daily_vol = math.sqrt(variance_pct2) / 100.0
            annualized_vol = daily_vol * math.sqrt(_ANNUALIZATION_FACTOR)

            if not math.isfinite(annualized_vol) or annualized_vol <= 0:
                _log.warning("garch_invalid_annualized_vol", fallback="rolling_vol")
                return _rolling_vol_fallback(returns)

            return annualized_vol

        except Exception:
            _log.warning("garch_fit_failed", fallback="rolling_vol", exc_info=True)
            return _rolling_vol_fallback(returns)

    def fit_forecast_safe(
        self,
        returns: list[float],
        horizon: int = 1,
        fallback_vol: float | None = None,
    ) -> float:
        """Fit GJR-GARCH with fallback to realized vol on failure.

        If the GARCH fit fails (returns NaN), falls back to either the
        provided fallback_vol or computes rolling realized vol as
        std(returns) * sqrt(252).

        Args:
            returns: List of simple returns.
            horizon: Forecast horizon in periods.
            fallback_vol: Explicit fallback annualized vol. If None, computes
                std(returns) * sqrt(252).

        Returns:
            Annualized volatility forecast (float).
        """
        vol = self.fit_forecast(returns, horizon=horizon)

        if math.isfinite(vol) and vol > 0:
            return vol

        # Fallback
        if fallback_vol is not None:
            return fallback_vol

        # Compute realized vol: std(returns) * sqrt(252)
        if len(returns) < _MIN_RETURNS_REALIZED:
            return float("nan")

        arr = np.array(returns, dtype=np.float64)
        return float(np.std(arr)) * math.sqrt(_ANNUALIZATION_FACTOR)


def forecast_garch_vol(returns: list[float], horizon: int = 1) -> float:
    """Convenience function: fit default GJR-GARCH(1,1,1) and return annualized vol.

    Args:
        returns: List of simple returns.
        horizon: Forecast horizon in periods (default 1).

    Returns:
        Annualized volatility forecast (float). Returns rolling vol fallback on failure.
    """
    forecaster = GJRGarchForecaster(p=1, o=1, q=1)
    return forecaster.fit_forecast_safe(returns, horizon=horizon)
