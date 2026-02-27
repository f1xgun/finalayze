"""Run batch backtest across a symbol universe.

Usage:
    uv run python scripts/run_batch_backtest.py \
        --universe us_mega --segment us_tech \
        --start 2022-01-01 --end 2025-01-01

    uv run python scripts/run_batch_backtest.py \
        --universe sp500_sample --segment us_tech \
        --start 2018-01-01 --end 2025-01-01 \
        --output results/batch_2025.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# Ensure project root is in path for config imports
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from finalayze.backtest.costs import TransactionCosts
from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.strategies.momentum import MomentumStrategy

_UNIVERSES_DIR = Path(PROJECT_ROOT) / "config" / "universes"


def _load_universe(name: str) -> list[str]:
    """Load symbol list from config/universes/<name>.json."""
    path = _UNIVERSES_DIR / f"{name}.json"
    if not path.exists():
        print(f"Universe file not found: {path}")
        sys.exit(1)
    with path.open() as f:
        symbols: list[str] = json.load(f)
    return symbols


def main() -> None:  # noqa: PLR0915
    """Run batch backtest and print per-symbol metrics."""
    parser = argparse.ArgumentParser(description="Batch backtest across symbol universe")
    parser.add_argument("--universe", required=True, help="Universe name (e.g., us_mega)")
    parser.add_argument("--segment", required=True, help="Segment ID (e.g., us_tech)")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=Decimal, default=Decimal(100000), help="Initial cash")
    parser.add_argument("--output", default=None, help="CSV output path (optional)")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark symbol (default: SPY)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    symbols = _load_universe(args.universe)

    print(f"Universe: {args.universe} ({len(symbols)} symbols)")
    print(f"Period: {args.start} to {args.end}")
    print(f"Segment: {args.segment}")
    print()

    fetcher = YFinanceFetcher(market_id="us")
    strategy = MomentumStrategy()
    analyzer = PerformanceAnalyzer()
    costs = TransactionCosts()

    # Fetch benchmark
    print(f"Fetching benchmark ({args.benchmark})...")
    benchmark_candles = fetcher.fetch_candles(args.benchmark, start, end)

    rows: list[dict[str, object]] = []
    total_return_sum = 0.0
    total_trades_sum = 0
    symbols_with_trades = 0

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}...", end=" ", flush=True)
        try:
            candles = fetcher.fetch_candles(symbol, start, end)
        except Exception as e:
            print(f"SKIP (fetch error: {e})")
            continue

        if not candles:
            print("SKIP (no data)")
            continue

        engine = BacktestEngine(
            strategy=strategy,
            initial_cash=args.cash,
            transaction_costs=costs,
            atr_multiplier=Decimal("2.5"),
            trail_activation_atr=Decimal("1.0"),
            trail_distance_atr=Decimal("1.5"),
        )
        trades, snapshots = engine.run(
            symbol=symbol,
            segment_id=args.segment,
            candles=candles,
        )

        result = analyzer.analyze(trades, snapshots, benchmark_candles=benchmark_candles)

        row = {
            "symbol": symbol,
            "total_return": float(result.total_return),
            "sharpe": float(result.sharpe),
            "max_drawdown": float(result.max_drawdown),
            "win_rate": float(result.win_rate),
            "profit_factor": float(result.profit_factor),
            "total_trades": result.total_trades,
            "alpha": float(result.alpha) if result.alpha is not None else "",
            "beta": float(result.beta) if result.beta is not None else "",
            "info_ratio": (
                float(result.information_ratio) if result.information_ratio is not None else ""
            ),
            "benchmark_return": (
                float(result.benchmark_return) if result.benchmark_return is not None else ""
            ),
        }
        rows.append(row)

        total_return_sum += float(result.total_return)
        total_trades_sum += result.total_trades
        if result.total_trades > 0:
            symbols_with_trades += 1

        status = f"{result.total_trades} trades, {float(result.total_return):.2%}"
        print(status)

    # Summary
    n = len(rows)
    print()
    print("=" * 60)
    print("  BATCH BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Symbols tested:     {n}")
    print(f"  Symbols with trades: {symbols_with_trades}")
    print(f"  Total trades:       {total_trades_sum}")
    if n > 0:
        avg_return = total_return_sum / n
        print(f"  Avg return:         {avg_return:.2%}")
        returns = [float(r["total_return"]) for r in rows]
        print(f"  Best return:        {max(returns):.2%}")
        print(f"  Worst return:       {min(returns):.2%}")
        win_rates = [float(r["win_rate"]) for r in rows if r["total_trades"] > 0]  # type: ignore[operator]
        if win_rates:
            print(f"  Avg win rate:       {sum(win_rates) / len(win_rates):.2%}")
    print("=" * 60)

    # Write CSV
    if args.output and rows:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
