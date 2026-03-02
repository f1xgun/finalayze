"""Hidden Markov Model regime detection.

Uses a 3-state Gaussian HMM trained on daily returns, rolling volatility,
and volume ratio to classify market regimes as low_vol_bull, high_vol_bull,
or bear. Includes a 3-bar persistence filter to avoid regime whipsaw.

Minimum data requirement: 252 candles (1 trading year).
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
from hmmlearn.hmm import GaussianHMM

from finalayze.risk.regime import MarketRegime, RegimeState

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from finalayze.core.schemas import Candle

# Feature engineering constants
_ROLLING_VOL_WINDOW = 20
_VOLUME_MA_WINDOW = 20


class HMMRegimeDetector:
    """Hidden Markov Model for regime detection using returns, vol, and volume."""

    _MIN_DATA_POINTS = 252  # minimum 1 year
    _N_STATES = 3
    _N_INIT = 10  # multiple initializations
    _N_ITER = 100  # max iterations (early stopping with tol)
    _TOL = 1e-4
    _PERSISTENCE_BARS = 3  # require 3 consecutive same predictions

    def __init__(self, retrain_frequency: int = 21) -> None:
        self._model: GaussianHMM | None = None
        self._state_labels: dict[int, str] = {}
        self._state_scales: dict[int, Decimal] = {}
        self._retrain_frequency = retrain_frequency
        self._last_train_bar: int = -1
        self._recent_states: list[int] = []  # for persistence filter

    def fit(self, candles: list[Candle]) -> None:
        """Train HMM on candle data.

        Performs multiple random initializations and keeps the best model
        (highest log-likelihood).

        Args:
            candles: Chronologically ordered candle data.

        Raises:
            ValueError: If fewer than 252 candles are provided.
        """
        if len(candles) < self._MIN_DATA_POINTS:
            msg = f"Need >= {self._MIN_DATA_POINTS} candles, got {len(candles)}"
            raise ValueError(msg)

        features = self._build_features(candles)

        best_score = -np.inf
        best_model: GaussianHMM | None = None

        for seed in range(self._N_INIT):
            model = GaussianHMM(
                n_components=self._N_STATES,
                covariance_type="full",
                n_iter=self._N_ITER,
                tol=self._TOL,
                random_state=42 + seed,
            )
            try:
                model.fit(features)
                score: float = model.score(features)
            except (ValueError, np.linalg.LinAlgError):
                # Degenerate covariance matrix from bad initialization -- skip
                continue
            if score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            msg = "All HMM initializations failed to converge"
            raise RuntimeError(msg)

        self._model = best_model
        self._label_states()

    def _build_features(self, candles: list[Candle]) -> NDArray[np.float64]:
        """Build feature matrix: daily return, 20-day rolling vol, volume ratio.

        Returns array of shape (n_usable_bars, 3) starting from bar
        _ROLLING_VOL_WINDOW onward (need history for rolling vol).
        """
        closes = np.array([float(c.close) for c in candles], dtype=np.float64)
        volumes = np.array([float(c.volume) for c in candles], dtype=np.float64)

        # Daily returns (log returns for normality)
        returns = np.diff(np.log(closes))

        # Rolling volatility (20-day std of returns)
        rolling_vol = np.full_like(returns, np.nan)
        for i in range(_ROLLING_VOL_WINDOW - 1, len(returns)):
            window = returns[i - _ROLLING_VOL_WINDOW + 1 : i + 1]
            rolling_vol[i] = np.std(window, ddof=1)

        # Volume ratio (current / 20-day MA)
        volume_ma = np.full_like(volumes, np.nan)
        for i in range(_VOLUME_MA_WINDOW - 1, len(volumes)):
            volume_ma[i] = np.mean(volumes[i - _VOLUME_MA_WINDOW + 1 : i + 1])

        # Volume ratio aligned with returns (shifted by 1 since returns start at index 1)
        vol_ratio = np.full_like(returns, np.nan)
        for i in range(len(returns)):
            candle_idx = i + 1  # returns[i] corresponds to candles[i+1]
            if volume_ma[candle_idx] > 0 and not np.isnan(volume_ma[candle_idx]):
                vol_ratio[i] = volumes[candle_idx] / volume_ma[candle_idx]

        # Stack features and drop rows with NaN
        features = np.column_stack([returns, rolling_vol, vol_ratio])
        valid_mask = ~np.any(np.isnan(features), axis=1)
        result: NDArray[np.float64] = features[valid_mask]
        return result

    def _label_states(self) -> None:
        """Label HMM states using BOTH mean return AND variance.

        Classification rules:
          - Positive mean + below-median variance -> low_vol_bull (scale 1.0)
          - Positive mean + above-median variance -> high_vol_bull (scale 0.5)
          - Negative mean -> bear (scale 0.25)
        """
        if self._model is None:
            return

        means = self._model.means_[:, 0]  # return means
        variances = np.array([self._model.covars_[i][0, 0] for i in range(self._N_STATES)])
        median_var = float(np.median(variances))

        self._state_labels.clear()
        self._state_scales.clear()

        for i in range(self._N_STATES):
            if means[i] > 0 and variances[i] < median_var:
                self._state_labels[i] = "low_vol_bull"
                self._state_scales[i] = Decimal("1.0")
            elif means[i] > 0:
                self._state_labels[i] = "high_vol_bull"
                self._state_scales[i] = Decimal("0.5")
            else:
                self._state_labels[i] = "bear"
                self._state_scales[i] = Decimal("0.25")

    def predict_regime(self, candles: list[Candle]) -> RegimeState:
        """Predict regime for current bar with 3-bar persistence filter.

        The persistence filter requires 3 consecutive identical predictions
        before switching regime. This prevents whipsaw in noisy conditions.

        Args:
            candles: Full candle history up to the current bar.

        Returns:
            RegimeState reflecting the current detected regime.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._model is None:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        features = self._build_features(candles)
        raw_state: int = int(self._model.predict(features[-1:].reshape(1, -1))[0])

        self._recent_states.append(raw_state)
        if len(self._recent_states) > self._PERSISTENCE_BARS:
            self._recent_states = self._recent_states[-self._PERSISTENCE_BARS :]

        # Only change regime if last 3 predictions agree
        if (
            len(self._recent_states) >= self._PERSISTENCE_BARS
            and len(set(self._recent_states[-self._PERSISTENCE_BARS :])) == 1
        ):
            active_state = raw_state
        else:
            # Keep previous stable state, or default to first seen state
            active_state = self._recent_states[0] if self._recent_states else 0

        label = self._state_labels.get(active_state, "low_vol_bull")
        scale = self._state_scales.get(active_state, Decimal("1.0"))

        if label == "bear":
            regime = MarketRegime.CRISIS
        elif label == "high_vol_bull":
            regime = MarketRegime.ELEVATED
        else:
            regime = MarketRegime.NORMAL

        return RegimeState(
            regime=regime,
            allow_new_longs=(regime != MarketRegime.CRISIS),
            position_scale=scale,
        )

    def reset_persistence(self) -> None:
        """Clear the persistence filter state."""
        self._recent_states.clear()
