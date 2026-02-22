"""Performance analyzer for backtest results.

Computes aggregate metrics (Sharpe ratio, max drawdown, win rate, profit factor,
total return) from a list of trades and portfolio snapshots.
"""

from __future__ import annotations

import statistics
from decimal import Decimal

from finalayze.core.schemas import BacktestResult, PortfolioState, TradeResult

# Annualisation factor for daily returns.
_TRADING_DAYS_PER_YEAR = 252
_QUANTIZE_4DP = Decimal("0.0001")

# Sentinel value returned for profit_factor when there are no losing trades.
# A real profit factor is undefined (division by zero) when gross_loss == 0;
# we use 999 as a large-but-finite sentinel so the result remains a valid Decimal.
_INFINITE_PROFIT_FACTOR = Decimal(999)


class PerformanceAnalyzer:
    """Compute performance metrics from backtest trades and equity snapshots."""

    def analyze(
        self,
        trades: list[TradeResult],
        snapshots: list[PortfolioState],
    ) -> BacktestResult:
        """Return a :class:`BacktestResult` summarising the backtest run."""
        total_trades = len(trades)

        if total_trades == 0:
            # Still compute equity-curve metrics from snapshots if available
            max_drawdown = self._compute_max_drawdown(snapshots)
            sharpe = self._compute_sharpe(snapshots)
            if len(snapshots) >= 2:  # noqa: PLR2004
                initial = snapshots[0].equity
                final = snapshots[-1].equity
                total_return = (final - initial) / initial if initial > 0 else Decimal(0)
            else:
                total_return = Decimal(0)
            return BacktestResult(
                sharpe=sharpe,
                max_drawdown=max_drawdown,
                win_rate=Decimal(0),
                profit_factor=Decimal(0),
                total_return=(
                    total_return.quantize(_QUANTIZE_4DP) if total_return != 0 else Decimal(0)
                ),
                total_trades=0,
            )

        wins = [t for t in trades if t.pnl > 0]
        win_rate = Decimal(str(len(wins))) / Decimal(str(total_trades))

        gross_profit = sum((t.pnl for t in trades if t.pnl > 0), start=Decimal(0))
        gross_loss = abs(sum((t.pnl for t in trades if t.pnl < 0), start=Decimal(0)))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else _INFINITE_PROFIT_FACTOR

        # Total return from equity curve
        if len(snapshots) >= 2:  # noqa: PLR2004
            initial = snapshots[0].equity
            final = snapshots[-1].equity
            total_return = (final - initial) / initial if initial > 0 else Decimal(0)
        else:
            total_return = Decimal(0)

        max_drawdown = self._compute_max_drawdown(snapshots)
        sharpe = self._compute_sharpe(snapshots)

        return BacktestResult(
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate.quantize(_QUANTIZE_4DP),
            profit_factor=profit_factor.quantize(_QUANTIZE_4DP),
            total_return=total_return.quantize(_QUANTIZE_4DP),
            total_trades=total_trades,
        )

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _compute_max_drawdown(snapshots: list[PortfolioState]) -> Decimal:
        """Compute maximum peak-to-trough drawdown from equity snapshots."""
        if len(snapshots) < 2:  # noqa: PLR2004
            return Decimal(0)

        peak = snapshots[0].equity
        max_dd = Decimal(0)

        for snap in snapshots[1:]:
            peak = max(peak, snap.equity)
            dd = (peak - snap.equity) / peak if peak > 0 else Decimal(0)
            max_dd = max(max_dd, dd)

        return max_dd.quantize(_QUANTIZE_4DP)

    @staticmethod
    def _compute_sharpe(
        snapshots: list[PortfolioState],
        risk_free_rate: float = 0.0,
    ) -> Decimal:
        """Compute annualised Sharpe ratio from equity snapshots."""
        min_snapshots = 3
        if len(snapshots) < min_snapshots:
            return Decimal(0)

        equities = [float(s.equity) for s in snapshots]
        returns = [
            (equities[i] - equities[i - 1]) / equities[i - 1]
            for i in range(1, len(equities))
            if equities[i - 1] > 0
        ]
        if not returns:
            return Decimal(0)

        mean_return = statistics.mean(returns) - risk_free_rate / _TRADING_DAYS_PER_YEAR
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0.0

        if std_return == 0:
            return Decimal(0)

        sharpe = (mean_return / std_return) * (_TRADING_DAYS_PER_YEAR**0.5)
        return Decimal(str(round(sharpe, 4)))
