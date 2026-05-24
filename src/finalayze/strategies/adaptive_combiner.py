"""Adaptive strategy combiner with rolling Sharpe-based rebalancing (Layer 4).

Extends :class:`StrategyCombiner` to dynamically adjust per-strategy weights
based on recent performance (rolling Sharpe ratio).

Supports optional Thompson sampling exploration to avoid local optima.
"""

from __future__ import annotations

import math
import random
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.strategies.combiner import StrategyCombiner

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle, Signal
    from finalayze.strategies.base import BaseStrategy

_ZERO = Decimal(0)
_ANNUALIZATION_FACTOR = math.sqrt(252)
_DEFAULT_EPSILON = 0.1
_BAYESIAN_PRIOR = 1  # Beta(1, 1) uniform prior


class AdaptiveStrategyCombiner(StrategyCombiner):
    """Dynamic strategy weighting based on rolling performance.

    Every ``_REBALANCE_FREQUENCY`` bars the combiner recomputes weights
    using the rolling Sharpe ratio of each strategy's recorded trades.
    Strategies with insufficient data keep their YAML weight.  A minimum
    weight floor of 5% prevents any strategy from being fully silenced.
    """

    _REBALANCE_FREQUENCY = 21  # monthly (trading days)
    _ROLLING_WINDOW = 63  # 3 months of trades
    _MIN_WEIGHT = Decimal("0.05")  # 5% floor

    def __init__(
        self,
        strategies: list[BaseStrategy],
        segment_id: str | None = None,
        normalize_mode: str = "firing",
        exploration_mode: str = "deterministic",
        epsilon: float = _DEFAULT_EPSILON,
    ) -> None:
        super().__init__(strategies, normalize_mode=normalize_mode)
        self._segment_id = segment_id
        self._dynamic_weights: dict[str, Decimal] = {}
        self._strategy_returns: dict[str, list[float]] = {}
        self._bars_since_rebalance: int = 0
        self._exploration_mode = exploration_mode
        self._epsilon = epsilon
        # Thompson sampling: (alpha, beta) per strategy for Beta distribution
        self._outcome_counts: dict[str, tuple[int, int]] = {}

    def record_trade_result(self, strategy_name: str, pnl_pct: float) -> None:
        """Record a trade result for weight computation.

        Args:
            strategy_name: Name of the strategy that produced the trade.
            pnl_pct: PnL as a decimal ratio (e.g. 0.02 for +2%).
        """
        if strategy_name not in self._strategy_returns:
            self._strategy_returns[strategy_name] = []
        returns = self._strategy_returns[strategy_name]
        returns.append(pnl_pct)
        # Keep only the rolling window
        if len(returns) > self._ROLLING_WINDOW:
            self._strategy_returns[strategy_name] = returns[-self._ROLLING_WINDOW :]

    def _recompute_weights(self) -> None:
        """Recompute weights based on rolling Sharpe ratios."""
        sharpes: dict[str, Decimal] = {}
        min_samples = 2  # need at least 2 returns for std

        for name in self._strategies:
            returns = self._strategy_returns.get(name, [])
            if len(returns) < min_samples:
                continue
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            std_ret = math.sqrt(variance) if variance > 0 else 0.0
            sharpe = (mean_ret / std_ret) * _ANNUALIZATION_FACTOR if std_ret > 0 else 0.0
            # Clamp negative Sharpes to zero for weighting purposes
            sharpes[name] = Decimal(str(max(sharpe, 0.0)))

        if not sharpes:
            return

        total_sharpe = sum(sharpes.values(), _ZERO)

        if total_sharpe <= _ZERO:
            # All strategies have zero or negative Sharpe -- equal weight
            n = len(sharpes)
            equal = Decimal(1) / Decimal(n)
            self._dynamic_weights = dict.fromkeys(sharpes, equal)
        else:
            raw: dict[str, Decimal] = {}
            for name, s in sharpes.items():
                raw[name] = s / total_sharpe

            # Apply minimum weight floor
            self._dynamic_weights = self._apply_floor(raw)

    @classmethod
    def _apply_floor(cls, raw: dict[str, Decimal]) -> dict[str, Decimal]:
        """Enforce minimum weight floor and re-normalize to sum to 1."""
        floored: dict[str, Decimal] = {}
        n_below = 0
        surplus = _ZERO

        for name, w in raw.items():
            if w < cls._MIN_WEIGHT:
                floored[name] = cls._MIN_WEIGHT
                surplus += cls._MIN_WEIGHT - w
                n_below += 1
            else:
                floored[name] = w

        if n_below == 0 or surplus == _ZERO:
            return floored

        # Redistribute surplus proportionally among above-floor strategies
        above_names = [n for n, w in raw.items() if w >= cls._MIN_WEIGHT]
        above_total = sum(floored[n] for n in above_names)
        if above_total > _ZERO:
            for name in above_names:
                floored[name] -= surplus * (floored[name] / above_total)
                # Ensure we don't push above-floor below the floor
                floored[name] = max(floored[name], cls._MIN_WEIGHT)

        return floored

    def record_outcome(self, strategy_name: str, *, profitable: bool) -> None:
        """Record a trade outcome for Thompson sampling exploration.

        Args:
            strategy_name: Name of the strategy that produced the trade.
            profitable: True if the trade was profitable, False otherwise.
        """
        alpha, beta = self._outcome_counts.get(strategy_name, (_BAYESIAN_PRIOR, _BAYESIAN_PRIOR))
        if profitable:
            alpha += 1
        else:
            beta += 1
        self._outcome_counts[strategy_name] = (alpha, beta)

    def _compute_thompson_weights(self) -> dict[str, float]:
        """Sample from Beta distributions to compute exploration weights.

        Each strategy is modeled as Beta(alpha, beta) where alpha = successes + 1
        and beta = failures + 1 (uniform Bayesian prior).  A sample is drawn from
        each distribution and the results are normalized to sum to 1.
        """
        samples: dict[str, float] = {}
        for name in self._strategies:
            alpha, beta = self._outcome_counts.get(name, (_BAYESIAN_PRIOR, _BAYESIAN_PRIOR))
            samples[name] = random.betavariate(alpha, beta)

        total = sum(samples.values())
        if total <= 0:
            n = len(samples)
            return dict.fromkeys(samples, 1.0 / n)
        return {name: s / total for name, s in samples.items()}

    def _blend_weights(
        self,
        exploit_weights: dict[str, Decimal],
    ) -> dict[str, Decimal]:
        """Blend exploitation weights with Thompson exploration weights.

        Returns exploit_weights unchanged when exploration_mode is deterministic.
        Otherwise: final = (1 - epsilon) * exploit + epsilon * explore.
        """
        if self._exploration_mode != "thompson":
            return exploit_weights

        epsilon = Decimal(str(self._epsilon))
        one_minus_eps = Decimal(1) - epsilon

        thompson = self._compute_thompson_weights()
        blended: dict[str, Decimal] = {}
        for name, exploit_w in exploit_weights.items():
            exploit_part = one_minus_eps * exploit_w
            explore_part = epsilon * Decimal(str(thompson.get(name, 0.0)))
            blended[name] = exploit_part + explore_part

        # Re-normalize to ensure weights sum to exactly 1
        total = sum(blended.values())
        if total > _ZERO:
            blended = {k: v / total for k, v in blended.items()}

        return blended

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
        weight_overrides: dict[str, Decimal] | None = None,
        credibility: float = 1.0,
        event_type_code: float = 0.0,
    ) -> Signal | None:
        """Override to use dynamic weights instead of YAML weights."""
        self._bars_since_rebalance += 1
        if self._bars_since_rebalance >= self._REBALANCE_FREQUENCY:
            self._recompute_weights()
            self._bars_since_rebalance = 0

        effective_overrides = weight_overrides
        if self._dynamic_weights and effective_overrides is None:
            effective_overrides = self._dynamic_weights

        # Apply Thompson sampling blending when in exploration mode
        if effective_overrides is not None:
            effective_overrides = self._blend_weights(effective_overrides)

        return super().generate_signal(
            symbol,
            candles,
            segment_id,
            sentiment_score=sentiment_score,
            has_open_position=has_open_position,
            weight_overrides=effective_overrides,
            credibility=credibility,
            event_type_code=event_type_code,
        )
