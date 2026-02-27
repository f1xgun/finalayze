"""Performance analyzer for backtest results.

Computes aggregate metrics (Sharpe ratio, max drawdown, win rate, profit factor,
total return) from a list of trades and portfolio snapshots.  Optionally computes
benchmark-relative metrics (alpha, beta, information ratio, max relative drawdown).
"""

from __future__ import annotations

import statistics
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.core.schemas import BacktestResult, PortfolioState, TradeResult

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

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
        benchmark_candles: list[Candle] | None = None,
    ) -> BacktestResult:
        """Return a :class:`BacktestResult` summarising the backtest run.

        Args:
            trades: List of completed trades.
            snapshots: Equity-curve portfolio snapshots (one per bar).
            benchmark_candles: Optional benchmark candles for relative metrics.
        """
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

            benchmark_metrics = self._compute_benchmark_metrics(snapshots, benchmark_candles)

            return BacktestResult(
                sharpe=sharpe,
                max_drawdown=max_drawdown,
                win_rate=Decimal(0),
                profit_factor=Decimal(0),
                total_return=(
                    total_return.quantize(_QUANTIZE_4DP) if total_return != 0 else Decimal(0)
                ),
                total_trades=0,
                **benchmark_metrics,
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
        benchmark_metrics = self._compute_benchmark_metrics(snapshots, benchmark_candles)

        return BacktestResult(
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate.quantize(_QUANTIZE_4DP),
            profit_factor=profit_factor.quantize(_QUANTIZE_4DP),
            total_return=total_return.quantize(_QUANTIZE_4DP),
            total_trades=total_trades,
            **benchmark_metrics,
        )

    # -- Private helpers -------------------------------------------------------

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

    @staticmethod
    def _compute_benchmark_metrics(
        snapshots: list[PortfolioState],
        benchmark_candles: list[Candle] | None,
    ) -> dict[str, Decimal | None]:
        """Compute benchmark-relative metrics when benchmark data is provided.

        Returns a dict suitable for unpacking into BacktestResult kwargs.
        """
        empty: dict[str, Decimal | None] = {
            "alpha": None,
            "beta": None,
            "information_ratio": None,
            "max_relative_drawdown": None,
            "benchmark_return": None,
        }

        min_points = 3
        if benchmark_candles is None or len(snapshots) < min_points:
            return empty

        if len(benchmark_candles) < min_points:
            return empty

        # Align lengths: use min of both series
        n = min(len(snapshots), len(benchmark_candles))
        if n < min_points:
            return empty

        # Compute daily returns for strategy
        equities = [float(s.equity) for s in snapshots[:n]]
        strat_returns = [
            (equities[i] - equities[i - 1]) / equities[i - 1]
            for i in range(1, len(equities))
            if equities[i - 1] > 0
        ]

        # Compute daily returns for benchmark (using close prices)
        bench_prices = [float(c.close) for c in benchmark_candles[:n]]
        bench_returns = [
            (bench_prices[i] - bench_prices[i - 1]) / bench_prices[i - 1]
            for i in range(1, len(bench_prices))
            if bench_prices[i - 1] > 0
        ]

        # Align return series length
        m = min(len(strat_returns), len(bench_returns))
        if m < 2:  # noqa: PLR2004
            return empty

        sr = strat_returns[:m]
        br = bench_returns[:m]

        # Benchmark total return
        bench_total = (bench_prices[n - 1] - bench_prices[0]) / bench_prices[0]
        strat_total = (equities[n - 1] - equities[0]) / equities[0]

        # Alpha = strategy return - benchmark return
        alpha = strat_total - bench_total

        # Beta = cov(strategy, benchmark) / var(benchmark)
        mean_sr = statistics.mean(sr)
        mean_br = statistics.mean(br)
        cov = statistics.mean([(s - mean_sr) * (b - mean_br) for s, b in zip(sr, br, strict=True)])
        var_bench = statistics.variance(br) if len(br) > 1 else 0.0

        beta = cov / var_bench if var_bench > 0 else Decimal(0)

        # Tracking error = stdev of excess returns
        excess = [s - b for s, b in zip(sr, br, strict=True)]
        tracking_error = statistics.stdev(excess) if len(excess) > 1 else 0.0

        # Information ratio = alpha / tracking error (annualised)
        if tracking_error > 0:
            ir = (statistics.mean(excess) / tracking_error) * (_TRADING_DAYS_PER_YEAR**0.5)
        else:
            ir = 0.0

        # Max relative drawdown: worst underperformance vs benchmark
        # Cumulative relative performance
        cumulative_relative = [0.0]
        for s, b in zip(sr, br, strict=True):
            cumulative_relative.append(cumulative_relative[-1] + (s - b))

        peak_relative = cumulative_relative[0]
        max_rel_dd = 0.0
        for val in cumulative_relative[1:]:
            peak_relative = max(peak_relative, val)
            rel_dd = peak_relative - val
            max_rel_dd = max(max_rel_dd, rel_dd)

        return {
            "alpha": Decimal(str(round(alpha, 4))),
            "beta": Decimal(str(round(float(beta), 4))),
            "information_ratio": Decimal(str(round(ir, 4))),
            "max_relative_drawdown": Decimal(str(round(max_rel_dd, 4))),
            "benchmark_return": Decimal(str(round(bench_total, 4))),
        }
