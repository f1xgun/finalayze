"""Statistical validation: walk-forward + Monte Carlo bootstrap.

Loads a symbol universe, runs walk-forward windows on each symbol, collects
out-of-sample trades, runs Monte Carlo bootstrap on the aggregated OOS trades,
and prints a PASS/FAIL verdict.

Usage:
    uv run python scripts/run_validation.py \
        --universe us_mega \
        --segment us_tech \
        --start 2018-01-01 --end 2025-01-01 \
        --bootstrap 10000 \
        --output results/validation.csv

If ``--universe`` points to a JSON file (``config/universes/<name>.json``),
the file is loaded; otherwise, the ``--symbols`` flag can supply a
comma-separated list directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so that `config.*` imports work when
# the script is invoked with ``uv run python scripts/run_validation.py``.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.monte_carlo import BootstrapResult, bootstrap_metrics
from finalayze.backtest.walk_forward import WalkForwardConfig, WalkForwardOptimizer
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.strategies.momentum import MomentumStrategy

# ---------------------------------------------------------------------------
# Thresholds for PASS/FAIL verdict
# ---------------------------------------------------------------------------
MIN_OOS_SHARPE_LOWER = 0.0  # Lower bound of Sharpe CI must be > 0
MIN_OOS_TOTAL_RETURN_LOWER = 0.0  # Lower bound of total-return CI must be > 0
MIN_TRADES = 30  # Minimum OOS trades for statistical significance

_INITIAL_CASH = Decimal(100_000)
_DEFAULT_BOOTSTRAP = 10_000
_SEPARATOR_WIDTH = 60


def _load_symbols(universe: str | None, symbols_csv: str | None) -> list[str]:
    """Resolve the list of symbols from either a universe file or CSV string."""
    if symbols_csv:
        return [s.strip().upper() for s in symbols_csv.split(",") if s.strip()]

    if universe:
        universe_path = Path(_PROJECT_ROOT) / "config" / "universes" / f"{universe}.json"
        if universe_path.exists():
            with open(universe_path) as f:
                data: Any = json.load(f)
            if isinstance(data, list):
                return [str(s) for s in data]
            if isinstance(data, dict) and "symbols" in data:
                return [str(s) for s in data["symbols"]]

    # Fallback to a small default set
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def _collect_oos_trade_returns(
    symbols: list[str],
    segment: str,
    start: datetime,
    end: datetime,
    wf_config: WalkForwardConfig,
) -> list[float]:
    """Run walk-forward on each symbol and collect OOS trade returns."""
    optimizer = WalkForwardOptimizer(config=wf_config)
    windows = optimizer.generate_windows(start.date(), end.date())

    if not windows:
        print(f"  [WARN] No walk-forward windows for {start} -> {end}")
        return []

    print(f"  Generated {len(windows)} walk-forward windows")

    fetcher = YFinanceFetcher(market_id="us")
    strategy = MomentumStrategy()
    engine = BacktestEngine(strategy=strategy, initial_cash=_INITIAL_CASH)

    all_oos_returns: list[float] = []

    for sym in symbols:
        print(f"  Processing {sym}...")
        try:
            candles = fetcher.fetch_candles(sym, start, end)
        except Exception as exc:
            print(f"    [SKIP] Failed to fetch {sym}: {exc}")
            continue

        if not candles:
            print(f"    [SKIP] No candles for {sym}")
            continue

        for window in windows:
            _train, test = optimizer.split_candles(candles, window)
            if len(test) == 0:
                continue

            # Run backtest on OOS (test) candles only
            trades, _snapshots = engine.run(sym, segment, test)
            all_oos_returns.extend(float(t.pnl_pct * 100) for t in trades)

    return all_oos_returns


def _print_bootstrap_report(result: BootstrapResult) -> None:
    """Print a human-readable bootstrap report."""
    print(f"  Simulations:      {result.n_simulations}")
    print(f"  OOS Trades:       {result.n_trades}")
    print()
    print(f"  Total Return:     {result.total_return.point_estimate:+.2f}%")
    print(
        f"    95% CI:         [{result.total_return.lower:+.2f}%,"
        f" {result.total_return.upper:+.2f}%]"
    )
    print(f"  Sharpe Ratio:     {result.sharpe_ratio.point_estimate:+.4f}")
    print(
        f"    95% CI:         [{result.sharpe_ratio.lower:+.4f}, {result.sharpe_ratio.upper:+.4f}]"
    )
    print(f"  Max Drawdown:     {result.max_drawdown.point_estimate:.2f}%")
    print(
        f"    95% CI:         [{result.max_drawdown.lower:.2f}%, {result.max_drawdown.upper:.2f}%]"
    )
    print(f"  Win Rate:         {result.win_rate.point_estimate:.2f}%")
    print(f"    95% CI:         [{result.win_rate.lower:.2f}%, {result.win_rate.upper:.2f}%]")
    print(f"  Profit Factor:    {result.profit_factor.point_estimate:.4f}")
    print(
        f"    95% CI:         [{result.profit_factor.lower:.4f}, {result.profit_factor.upper:.4f}]"
    )


def _write_csv(path: str, result: BootstrapResult) -> None:
    """Write results to a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "point_estimate", "ci_lower", "ci_upper", "confidence"])
        for name, ci in [
            ("total_return_pct", result.total_return),
            ("sharpe_ratio", result.sharpe_ratio),
            ("max_drawdown_pct", result.max_drawdown),
            ("win_rate_pct", result.win_rate),
            ("profit_factor", result.profit_factor),
        ]:
            writer.writerow(
                [
                    name,
                    f"{ci.point_estimate:.6f}",
                    f"{ci.lower:.6f}",
                    f"{ci.upper:.6f}",
                    f"{ci.confidence_level:.2f}",
                ]
            )
    print(f"  Results written to {output_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Statistical validation of trading strategy")
    parser.add_argument("--universe", default=None, help="Universe name (config/universes/<>.json)")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbol list")
    parser.add_argument("--segment", default="us_tech", help="Segment ID")
    parser.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=_DEFAULT_BOOTSTRAP,
        help="Bootstrap sims",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--train-years", type=int, default=3, help="Walk-forward train years")
    parser.add_argument("--test-years", type=int, default=1, help="Walk-forward test years")
    parser.add_argument("--step-months", type=int, default=6, help="Walk-forward step months")
    return parser.parse_args()


def _print_verdict(result: BootstrapResult) -> None:
    """Print PASS/FAIL verdict and exit with appropriate code."""
    print("-" * _SEPARATOR_WIDTH)
    sharpe_pass = result.sharpe_ratio.lower > MIN_OOS_SHARPE_LOWER
    return_pass = result.total_return.lower > MIN_OOS_TOTAL_RETURN_LOWER

    if sharpe_pass and return_pass:
        print("  VERDICT: PASS")
        print("    Sharpe CI lower bound > 0")
        print("    Total return CI lower bound > 0")
    else:
        print("  VERDICT: FAIL")
        if not sharpe_pass:
            print(f"    Sharpe CI lower bound = {result.sharpe_ratio.lower:.4f} <= 0")
        if not return_pass:
            print(f"    Total return CI lower bound = {result.total_return.lower:.2f}% <= 0")

    print("=" * _SEPARATOR_WIDTH)

    if not (sharpe_pass and return_pass):
        sys.exit(1)


def main() -> None:
    """Entry point for the validation script."""
    args = _parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    symbols = _load_symbols(args.universe, args.symbols)

    print("=" * _SEPARATOR_WIDTH)
    print("  STATISTICAL VALIDATION")
    print("=" * _SEPARATOR_WIDTH)
    print(f"  Symbols:  {', '.join(symbols)}")
    print(f"  Segment:  {args.segment}")
    print(f"  Period:   {start.date()} to {end.date()}")
    print(f"  Bootstrap: {args.bootstrap} simulations")
    print("-" * _SEPARATOR_WIDTH)

    wf_config = WalkForwardConfig(
        train_years=args.train_years,
        test_years=args.test_years,
        step_months=args.step_months,
    )

    oos_returns = _collect_oos_trade_returns(symbols, args.segment, start, end, wf_config)

    print()
    print("-" * _SEPARATOR_WIDTH)
    print(f"  Collected {len(oos_returns)} OOS trade returns")
    print("-" * _SEPARATOR_WIDTH)

    if len(oos_returns) < MIN_TRADES:
        print(f"  [FAIL] Insufficient OOS trades ({len(oos_returns)} < {MIN_TRADES})")
        sys.exit(1)

    result = bootstrap_metrics(
        oos_returns,
        n_simulations=args.bootstrap,
        seed=args.seed,
    )

    print()
    _print_bootstrap_report(result)
    print()

    # Write CSV if requested
    if args.output:
        _write_csv(args.output, result)

    _print_verdict(result)


if __name__ == "__main__":
    main()
