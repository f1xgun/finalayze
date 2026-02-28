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

The universe file supports two formats:
  - Simple list:  ["AAPL", "MSFT", ...]
  - Dict with sectors:  {"symbols": [...], "sectors": {"TECH": ["AAPL", ...], ...}}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, NamedTuple

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
_PERCENT = 100.0

_SPY_SYMBOL = "SPY"
_DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


class _SymbolReturn(NamedTuple):
    """A single OOS trade return tagged with its source symbol."""

    symbol: str
    return_pct: float


class _UniverseData(NamedTuple):
    """Parsed universe: symbols list and optional sector mapping."""

    symbols: list[str]
    sectors: dict[str, list[str]]


def _load_symbols(universe: str | None, symbols_csv: str | None) -> _UniverseData:
    """Resolve the list of symbols (and optional sectors) from a universe file or CSV."""
    if symbols_csv:
        syms = [s.strip().upper() for s in symbols_csv.split(",") if s.strip()]
        return _UniverseData(symbols=syms, sectors={})

    if universe:
        universe_path = Path(_PROJECT_ROOT) / "config" / "universes" / f"{universe}.json"
        if universe_path.exists():
            with open(universe_path) as f:
                data: Any = json.load(f)
            if isinstance(data, list):
                return _UniverseData(symbols=[str(s) for s in data], sectors={})
            if isinstance(data, dict):
                syms = [str(s) for s in data.get("symbols", [])]
                sectors_raw = data.get("sectors", {})
                sectors = {str(k): [str(v) for v in vs] for k, vs in sectors_raw.items()}
                return _UniverseData(symbols=syms, sectors=sectors)

    # Fallback to a small default set
    return _UniverseData(symbols=list(_DEFAULT_SYMBOLS), sectors={})


def _collect_oos_trade_returns(
    symbols: list[str],
    segment: str,
    start: datetime,
    end: datetime,
    wf_config: WalkForwardConfig,
) -> list[_SymbolReturn]:
    """Run walk-forward on each symbol and collect OOS trade returns with symbol tags."""
    optimizer = WalkForwardOptimizer(config=wf_config)
    windows = optimizer.generate_windows(start.date(), end.date())

    if not windows:
        print(f"  [WARN] No walk-forward windows for {start} -> {end}")
        return []

    print(f"  Generated {len(windows)} walk-forward windows")

    fetcher = YFinanceFetcher(market_id="us")
    strategy = MomentumStrategy()
    engine = BacktestEngine(strategy=strategy, initial_cash=_INITIAL_CASH)

    all_oos_returns: list[_SymbolReturn] = []

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
            all_oos_returns.extend(
                _SymbolReturn(symbol=sym, return_pct=float(t.pnl_pct * _PERCENT)) for t in trades
            )

    return all_oos_returns


def _compute_spy_return(start: datetime, end: datetime) -> float | None:
    """Compute SPY buy-and-hold return over the given period.

    Returns the percentage return, or None if data cannot be fetched.
    """
    fetcher = YFinanceFetcher(market_id="us")
    try:
        candles = fetcher.fetch_candles(_SPY_SYMBOL, start, end)
    except Exception as exc:
        print(f"  [WARN] Failed to fetch SPY benchmark: {exc}")
        return None

    if not candles:
        print("  [WARN] No SPY candle data available")
        return None

    first_close = float(candles[0].close)
    last_close = float(candles[-1].close)
    if first_close == 0:
        return None

    return (last_close - first_close) / first_close * _PERCENT


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


def _print_sector_breakdown(
    returns: list[_SymbolReturn],
    sectors: dict[str, list[str]],
) -> None:
    """Print per-sector OOS trade summary."""
    if not sectors:
        return

    print()
    print("-" * _SEPARATOR_WIDTH)
    print("  PER-SECTOR BREAKDOWN")
    print("-" * _SEPARATOR_WIDTH)
    header = f"  {'Sector':<12} {'Trades':>8} {'Win%':>8} {'AvgRet%':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Build symbol -> sector mapping
    sym_to_sector: dict[str, str] = {}
    for sector, syms in sectors.items():
        for s in syms:
            sym_to_sector[s] = sector

    # Group returns by sector
    sector_returns: dict[str, list[float]] = {}
    uncategorized: list[float] = []
    for sr in returns:
        sector = sym_to_sector.get(sr.symbol)
        if sector is not None:
            sector_returns.setdefault(sector, []).append(sr.return_pct)
        else:
            uncategorized.append(sr.return_pct)

    for sector in sorted(sector_returns):
        rets = sector_returns[sector]
        n_trades = len(rets)
        win_rate = sum(1 for r in rets if r > 0) / n_trades * _PERCENT if n_trades else 0.0
        avg_ret = sum(rets) / n_trades if n_trades else 0.0
        print(f"  {sector:<12} {n_trades:>8} {win_rate:>7.1f}% {avg_ret:>+9.2f}%")

    if uncategorized:
        n_trades = len(uncategorized)
        win_rate = sum(1 for r in uncategorized if r > 0) / n_trades * _PERCENT
        avg_ret = sum(uncategorized) / n_trades
        print(f"  {'OTHER':<12} {n_trades:>8} {win_rate:>7.1f}% {avg_ret:>+9.2f}%")


def _write_csv(path: str, result: BootstrapResult, alpha: float | None = None) -> None:
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
        if alpha is not None:
            writer.writerow(["alpha_vs_spy", f"{alpha:.6f}", "", "", ""])
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
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        default=False,
        help="Skip SPY benchmark comparison",
    )
    return parser.parse_args()


def _print_verdict(
    result: BootstrapResult,
    alpha: float | None,
) -> None:
    """Print PASS/FAIL verdict and exit with appropriate code."""
    print("-" * _SEPARATOR_WIDTH)
    sharpe_pass = result.sharpe_ratio.lower > MIN_OOS_SHARPE_LOWER
    return_pass = result.total_return.lower > MIN_OOS_TOTAL_RETURN_LOWER
    alpha_pass = alpha is None or alpha > 0

    all_pass = sharpe_pass and return_pass and alpha_pass

    if all_pass:
        print("  VERDICT: PASS")
        print("    Sharpe CI lower bound > 0")
        print("    Total return CI lower bound > 0")
        if alpha is not None:
            print(f"    Alpha vs SPY = {alpha:+.2f}% > 0")
        else:
            print("    Alpha vs SPY: N/A (benchmark skipped)")
    else:
        print("  VERDICT: FAIL")
        if not sharpe_pass:
            print(f"    Sharpe CI lower bound = {result.sharpe_ratio.lower:.4f} <= 0")
        if not return_pass:
            print(f"    Total return CI lower bound = {result.total_return.lower:.2f}% <= 0")
        if not alpha_pass:
            print(f"    Alpha vs SPY = {alpha:+.2f}% <= 0")

    print("=" * _SEPARATOR_WIDTH)

    if not all_pass:
        sys.exit(1)


def main() -> None:
    """Entry point for the validation script."""
    args = _parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    universe_data = _load_symbols(args.universe, args.symbols)
    symbols = universe_data.symbols
    sectors = universe_data.sectors

    print("=" * _SEPARATOR_WIDTH)
    print("  STATISTICAL VALIDATION")
    print("=" * _SEPARATOR_WIDTH)
    print(f"  Symbols:  {', '.join(symbols)}")
    print(f"  Segment:  {args.segment}")
    print(f"  Period:   {start.date()} to {end.date()}")
    print(f"  Bootstrap: {args.bootstrap} simulations")
    if sectors:
        print(f"  Sectors:  {', '.join(sorted(sectors))}")
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

    # Extract flat return list for bootstrap
    flat_returns = [sr.return_pct for sr in oos_returns]

    result = bootstrap_metrics(
        flat_returns,
        n_simulations=args.bootstrap,
        seed=args.seed,
    )

    print()
    _print_bootstrap_report(result)

    # SPY benchmark comparison
    alpha: float | None = None
    if not args.no_benchmark:
        print()
        print("-" * _SEPARATOR_WIDTH)
        print("  SPY BENCHMARK COMPARISON")
        print("-" * _SEPARATOR_WIDTH)
        spy_return = _compute_spy_return(start, end)
        if spy_return is not None:
            strategy_return = result.total_return.point_estimate
            alpha = strategy_return - spy_return
            print(f"  SPY buy-and-hold return:  {spy_return:+.2f}%")
            print(f"  Strategy OOS return:      {strategy_return:+.2f}%")
            print(f"  Alpha (strategy - SPY):   {alpha:+.2f}%")
        else:
            print("  [WARN] Could not compute SPY benchmark; alpha check skipped")

    # Per-sector breakdown
    _print_sector_breakdown(oos_returns, sectors)

    print()

    # Write CSV if requested
    if args.output:
        _write_csv(args.output, result, alpha=alpha)

    _print_verdict(result, alpha=alpha)


if __name__ == "__main__":
    main()
