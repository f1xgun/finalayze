"""Run a backtest for a single symbol + segment.

Usage:
    uv run python scripts/run_backtest.py --symbol AAPL --segment us_tech
    uv run python scripts/run_backtest.py --symbol AAPL --segment us_tech \
        --start 2023-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal

from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.strategies.momentum import MomentumStrategy


def main() -> None:
    """Run backtest and print performance report."""
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--segment", required=True, help="Segment ID (e.g., us_tech)")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=100_000, help="Initial cash")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    print(f"Fetching {args.symbol} data from {args.start} to {args.end}...")
    fetcher = YFinanceFetcher(market_id="us")
    candles = fetcher.fetch_candles(args.symbol, start, end)

    if not candles:
        print("No candles fetched. Check symbol and date range.")
        return

    print(f"Fetched {len(candles)} candles. Running backtest...")

    strategy = MomentumStrategy()
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=Decimal(str(args.cash)),
    )
    trades, snapshots = engine.run(
        symbol=args.symbol,
        segment_id=args.segment,
        candles=candles,
    )

    analyzer = PerformanceAnalyzer()
    result = analyzer.analyze(trades, snapshots)

    print()
    print("=" * 50)
    print(f"  BACKTEST RESULTS: {args.symbol} ({args.segment})")
    print("=" * 50)
    print(f"  Period:         {args.start} to {args.end}")
    print(f"  Strategy:       {strategy.name}")
    print(f"  Initial Cash:   ${args.cash:,.2f}")
    print("-" * 50)
    print(f"  Total Return:   {float(result.total_return):.2%}")
    print(f"  Sharpe Ratio:   {result.sharpe}")
    print(f"  Max Drawdown:   {float(result.max_drawdown):.2%}")
    print(f"  Win Rate:       {float(result.win_rate):.2%}")
    print(f"  Profit Factor:  {result.profit_factor}")
    print(f"  Total Trades:   {result.total_trades}")
    print("=" * 50)


if __name__ == "__main__":
    main()
