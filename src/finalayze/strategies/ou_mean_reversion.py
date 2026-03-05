"""Ornstein-Uhlenbeck mean reversion strategy (Layer 4).

Models price as an OU process and trades deviations from the fitted long-run mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import yaml
from scipy.optimize import minimize

if TYPE_CHECKING:
    from numpy.typing import NDArray

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.risk.regime import MarketRegime, RegimeState
from finalayze.strategies.base import BaseStrategy

_PRESETS_DIR = Path(__file__).parent / "presets"

_CONFIDENCE_BASE = 0.4
_CONFIDENCE_SCALE = 0.15
_MAX_CONFIDENCE = 0.95
_B_EPSILON = 1e-15


@dataclass(frozen=True, slots=True)
class OUParams:
    """Parameters of a fitted Ornstein-Uhlenbeck process."""

    mu: float  # mean reversion speed
    theta: float  # long-run mean (log-price)
    sigma: float  # volatility
    half_life: float  # ln(2) / mu


def fit_ou_mle(log_prices: list[float]) -> OUParams:
    """Fit OU process via MLE on log-prices.

    Uses regression: X_{t+1} - X_t = a + b*X_t + epsilon
    mu = -b, theta = -a/b, sigma from residual std.
    """
    n = len(log_prices)
    if n < 3:  # noqa: PLR2004
        msg = "Need at least 3 log-prices to fit OU"
        raise ValueError(msg)

    x = np.array(log_prices[:-1])
    dx = np.array(log_prices[1:]) - x

    # OLS: dx = a + b * x + eps
    x_mean = x.mean()
    dx_mean = dx.mean()
    ss_xx = float(np.sum((x - x_mean) ** 2))
    ss_xy = float(np.sum((x - x_mean) * (dx - dx_mean)))

    if ss_xx == 0.0:
        msg = "Degenerate log-price series (zero variance)"
        raise ValueError(msg)

    b = ss_xy / ss_xx
    a = dx_mean - b * x_mean

    # OU parameters
    mu = -b
    if mu <= 0:
        # No mean reversion detected; return with very large half-life
        mu = 1e-10

    theta = -a / b if abs(b) > _B_EPSILON else float(x_mean)
    residuals = dx - (a + b * x)
    sigma = float(np.std(residuals, ddof=1))
    half_life = math.log(2) / mu

    return OUParams(mu=mu, theta=theta, sigma=sigma, half_life=half_life)


_MIN_POINTS_FOR_MLE = 10


def _ou_neg_log_likelihood(
    params: NDArray[np.float64],
    x: NDArray[np.float64],
    dt: float,
) -> float:
    """Negative log-likelihood for discrete OU transition density.

    Transition: x_{t+1} | x_t ~ N(mean_t, var_t)
      mean_t = theta + (x_t - theta) * exp(-mu * dt)
      var_t  = sigma^2 * (1 - exp(-2*mu*dt)) / (2*mu)
    """
    mu, theta, sigma = float(params[0]), float(params[1]), float(params[2])

    if mu <= 0 or sigma <= 0:
        return 1e15

    decay = math.exp(-mu * dt)
    exp_2mu = math.exp(-2.0 * mu * dt)
    var = (sigma**2) * (1.0 - exp_2mu) / (2.0 * mu)

    if var <= 0:
        return 1e15

    x_prev = x[:-1]
    x_next = x[1:]
    means = theta + (x_prev - theta) * decay
    residuals = x_next - means

    n = len(residuals)
    nll = 0.5 * n * math.log(2.0 * math.pi * var) + float(np.sum(residuals**2)) / (2.0 * var)
    return float(nll)


def fit_ou_exact_mle(log_prices: list[float], dt: float = 1.0) -> OUParams:
    """Fit OU process via exact discrete-time MLE on log-prices.

    Uses scipy.optimize.minimize to maximise the log-likelihood of the
    exact OU transition density.  Falls back to OLS estimates if the
    optimiser fails.

    Parameters
    ----------
    log_prices:
        Log-price series (at least 10 points).
    dt:
        Time step between observations (default 1.0 for daily).

    Returns
    -------
    OUParams with (mu, theta, sigma, half_life).
    """
    n = len(log_prices)
    if n < _MIN_POINTS_FOR_MLE:
        msg = f"Need at least {_MIN_POINTS_FOR_MLE} log-prices to fit OU (got {n})"
        raise ValueError(msg)

    # Get OLS estimates as initial guess
    ols_params = fit_ou_mle(log_prices)
    x = np.array(log_prices, dtype=np.float64)

    mu0 = max(ols_params.mu, 1e-6)
    theta0 = ols_params.theta
    sigma0 = max(ols_params.sigma, 1e-6)
    initial = np.array([mu0, theta0, sigma0])

    bounds = [(1e-10, None), (None, None), (1e-10, None)]

    result = minimize(
        _ou_neg_log_likelihood,
        initial,
        args=(x, dt),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if result.success:
        mu_fit = float(result.x[0])
        theta_fit = float(result.x[1])
        sigma_fit = float(result.x[2])
    else:
        # Fallback to OLS
        mu_fit = ols_params.mu
        theta_fit = ols_params.theta
        sigma_fit = ols_params.sigma

    half_life = math.log(2) / mu_fit if mu_fit > 0 else 1e10
    return OUParams(mu=mu_fit, theta=theta_fit, sigma=sigma_fit, half_life=half_life)


class OUMeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Ornstein-Uhlenbeck process.

    Fits an OU model on log-prices and trades when the current price deviates
    significantly from the fitted long-run mean.
    """

    _SEGMENT_PARAMS: ClassVar[dict[str, dict[str, object]]] = {
        "us_tech": {
            "ou_window": 90,
            "entry_threshold": 1.5,
            "exit_threshold": 0.0,
            "half_life_range": (5, 60),
        },
        "us_broad": {
            "ou_window": 90,
            "entry_threshold": 1.5,
            "exit_threshold": 0.0,
            "half_life_range": (5, 60),
        },
        "us_healthcare": {
            "ou_window": 60,
            "entry_threshold": 1.8,
            "exit_threshold": 0.2,
            "half_life_range": (5, 45),
        },
        "us_finance": {
            "ou_window": 90,
            "entry_threshold": 1.5,
            "exit_threshold": 0.0,
            "half_life_range": (5, 60),
        },
        "ru_blue_chips": {
            "ou_window": 60,
            "entry_threshold": 2.0,
            "exit_threshold": 0.3,
            "half_life_range": (3, 90),
        },
        "ru_energy": {
            "ou_window": 60,
            "entry_threshold": 2.0,
            "exit_threshold": 0.3,
            "half_life_range": (3, 90),
        },
        "ru_tech": {
            "ou_window": 60,
            "entry_threshold": 1.8,
            "exit_threshold": 0.2,
            "half_life_range": (3, 60),
        },
        "ru_finance": {
            "ou_window": 60,
            "entry_threshold": 2.0,
            "exit_threshold": 0.3,
            "half_life_range": (3, 90),
        },
    }

    def __init__(
        self,
        ou_window: int | None = None,
        entry_threshold: float | None = None,
        exit_threshold: float | None = None,
        half_life_range: tuple[int, int] | None = None,
        use_mle: bool = False,
    ) -> None:
        self._ou_window = ou_window
        self._entry_threshold = entry_threshold
        self._exit_threshold = exit_threshold
        self._half_life_range = half_life_range
        self._use_mle = use_mle
        self._cached_params: dict[str, OUParams] = {}
        # Cache YAML params per segment to avoid reloading on every bar
        self._yaml_params_cache: dict[str, dict[str, object]] = {}

    @property
    def name(self) -> str:
        return "ou_mean_reversion"

    def supported_segments(self) -> list[str]:
        return list(self._SEGMENT_PARAMS.keys())

    def _load_yaml_params(self, segment_id: str) -> dict[str, object]:
        """Load ou_mean_reversion parameters from the YAML preset.

        Results are cached per segment_id to avoid reloading on every bar.
        """
        if segment_id in self._yaml_params_cache:
            return self._yaml_params_cache[segment_id]
        try:
            preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            params = dict(data["strategies"]["ou_mean_reversion"]["params"])
        except (FileNotFoundError, KeyError, TypeError):
            params = {}
        self._yaml_params_cache[segment_id] = params
        return params

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Return parameters for a segment.

        Priority: constructor overrides > YAML preset > _SEGMENT_PARAMS defaults.
        """
        # Start with hardcoded class defaults
        base = dict(self._SEGMENT_PARAMS.get(segment_id, self._SEGMENT_PARAMS["us_broad"]))

        # Overlay YAML preset params (higher priority than class defaults)
        yaml_params = self._load_yaml_params(segment_id)
        if yaml_params:
            base = {**base, **yaml_params}

        # Constructor overrides have highest priority
        if self._ou_window is not None:
            base["ou_window"] = self._ou_window
        if self._entry_threshold is not None:
            base["entry_threshold"] = self._entry_threshold
        if self._exit_threshold is not None:
            base["exit_threshold"] = self._exit_threshold
        if self._half_life_range is not None:
            base["half_life_range"] = self._half_life_range
        return base

    def reset(self) -> None:
        """Clear cached OU parameters between backtest runs."""
        self._cached_params = {}

    def generate_signal(  # noqa: PLR0911
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,  # noqa: ARG002
        has_open_position: bool = False,
        **kwargs: object,
    ) -> Signal | None:
        """Generate a mean reversion signal using OU process."""
        params = self.get_parameters(segment_id)
        ou_window = int(params["ou_window"])  # type: ignore[call-overload]
        entry_threshold = float(params["entry_threshold"])  # type: ignore[arg-type]
        exit_threshold = float(params["exit_threshold"])  # type: ignore[arg-type]
        hl_range = params["half_life_range"]
        hl_min = int(hl_range[0])  # type: ignore[index]
        hl_max = int(hl_range[1])  # type: ignore[index]

        # Regime gate
        regime_state: RegimeState | None = kwargs.get("regime_state")  # type: ignore[assignment]
        if regime_state is not None:
            if regime_state.regime == MarketRegime.CRISIS:
                return None
            if regime_state.regime == MarketRegime.ELEVATED:
                entry_threshold = max(entry_threshold, 2.0)

        # Need ou_window + 1 candles (window for fitting + 1 current bar)
        min_candles = ou_window + 1
        if len(candles) < min_candles:
            return None

        # Look-ahead bias fix: exclude current bar from fitting
        fitting_candles = candles[-(ou_window + 1) : -1]
        log_prices = [math.log(float(c.close)) for c in fitting_candles]

        try:
            ou_params = fit_ou_exact_mle(log_prices) if self._use_mle else fit_ou_mle(log_prices)
        except ValueError:
            return None

        # Half-life filter
        if not (hl_min <= ou_params.half_life <= hl_max):
            return None

        # Compute z-score
        current_log_price = math.log(float(candles[-1].close))
        ou_std = ou_params.sigma / math.sqrt(2 * ou_params.mu) if ou_params.mu > 0 else 0.0
        if ou_std <= 0:
            return None

        z_score = (current_log_price - ou_params.theta) / ou_std

        # Signal logic
        direction: SignalDirection | None = None

        if z_score < -entry_threshold:
            direction = SignalDirection.BUY
        elif has_open_position and z_score > exit_threshold:
            direction = SignalDirection.SELL
        else:
            return None

        # Confidence
        confidence = min(_MAX_CONFIDENCE, _CONFIDENCE_BASE + abs(z_score) * _CONFIDENCE_SCALE)

        market_id = candles[0].market_id

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={
                "z_score": round(z_score, 4),
                "ou_mu": round(ou_params.mu, 6),
                "ou_theta": round(ou_params.theta, 4),
                "half_life": round(ou_params.half_life, 2),
            },
            reasoning=(
                f"OU z-score={z_score:.2f}, half_life={ou_params.half_life:.1f}, "
                f"theta={ou_params.theta:.4f}"
            ),
        )
