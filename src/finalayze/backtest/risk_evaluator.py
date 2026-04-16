"""Backtest risk evaluator -- segment exposure and correlation helpers.

Pure-ish functions used by :class:`~finalayze.backtest.engine.BacktestEngine`
to compute portfolio-level risk metrics during a backtest run.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle
    from finalayze.execution.simulated_broker import SimulatedBroker


class BacktestRiskEvaluator:
    """Portfolio-level risk helpers consumed by the backtest engine."""

    @staticmethod
    def compute_segment_exposure(
        broker: SimulatedBroker,
        segment_id: str,  # noqa: ARG004
    ) -> Decimal:
        """Compute the total position value for a segment (concentration check).

        In single-symbol mode, all positions belong to the same segment.
        In portfolio mode, the engine only trades one segment at a time.
        So current equity in positions approximates segment exposure.
        """
        portfolio = broker.get_portfolio()
        position_value = portfolio.equity - portfolio.cash
        return max(position_value, Decimal(0))

    @staticmethod
    def compute_correlations(
        candles_by_symbol: dict[str, list[Candle]],
        lookback: int = 60,
    ) -> dict[tuple[str, str], float]:
        """Compute trailing pairwise correlations for open positions.

        Delegates to :func:`finalayze.risk.correlation.compute_correlation_matrix`
        which uses pure-Python Pearson correlation (no numpy, no NaN risk).
        """
        from finalayze.risk.correlation import compute_correlation_matrix  # noqa: PLC0415

        return compute_correlation_matrix(candles_by_symbol, window=lookback)
