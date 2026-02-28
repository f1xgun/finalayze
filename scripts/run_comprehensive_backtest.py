"""Comprehensive backtest: run multiple strategies across a universe.

Usage:
    uv run python scripts/run_comprehensive_backtest.py \
        --universe us_mega --segment us_tech \
        --start 2020-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from finalayze.backtest.costs import TransactionCosts
from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.monte_carlo import bootstrap_metrics
from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.strategies.combiner import StrategyCombiner
from finalayze.strategies.mean_reversion import MeanReversionStrategy
from finalayze.strategies.momentum import MomentumStrategy

_UNIVERSES_DIR = Path(PROJECT_ROOT) / "config" / "universes"
_INITIAL_CASH = Decimal(100_000)
_SEP = 70
_PERCENT = 100.0


def _load_universe(name: str) -> list[str]:
    path = _UNIVERSES_DIR / f"{name}.json"
    if not path.exists():
        print(f"Universe file not found: {path}")
        sys.exit(1)
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive multi-strategy backtest")
    parser.add_argument("--universe", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--cash", type=Decimal, default=_INITIAL_CASH)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    symbols = _load_universe(args.universe)

    strategies = {
        "Momentum (RSI+MACD)": MomentumStrategy(),
        "Mean Reversion (Bollinger)": MeanReversionStrategy(),
        "Combiner": StrategyCombiner(
            strategies=[MomentumStrategy(), MeanReversionStrategy()],
        ),
    }

    fetcher = YFinanceFetcher(market_id="us")
    analyzer = PerformanceAnalyzer()
    costs = TransactionCosts()

    print("=" * _SEP)
    print("  COMPREHENSIVE MULTI-STRATEGY BACKTEST")
    print("=" * _SEP)
    print(f"  Universe:  {args.universe} ({len(symbols)} symbols)")
    print(f"  Segment:   {args.segment}")
    print(f"  Period:    {args.start} to {args.end}")
    print(f"  Cash:      ${float(args.cash):,.0f}")
    print("=" * _SEP)

    # Pre-fetch all candles
    print("\nFetching market data...")
    candle_data: dict[str, list] = {}
    benchmark_candles = fetcher.fetch_candles("SPY", start, end)
    for sym in symbols:
        try:
            candle_data[sym] = fetcher.fetch_candles(sym, start, end)
            print(f"  {sym}: {len(candle_data[sym])} candles")
        except Exception as e:
            print(f"  {sym}: SKIP ({e})")

    for strat_name, strategy in strategies.items():
        print(f"\n{'=' * _SEP}")
        print(f"  STRATEGY: {strat_name}")
        print(f"{'=' * _SEP}")

        all_trades = []
        all_returns: list[float] = []
        strat_rows = []

        for sym in symbols:
            if sym not in candle_data:
                continue
            candles = candle_data[sym]
            engine = BacktestEngine(
                strategy=strategy,
                initial_cash=args.cash,
                transaction_costs=costs,
            )
            trades, snapshots = engine.run(sym, args.segment, candles)

            if trades:
                result = analyzer.analyze(trades, snapshots, benchmark_candles=benchmark_candles)
                all_trades.extend(trades)
                all_returns.extend(float(t.pnl_pct) * _PERCENT for t in trades)
                strat_rows.append({
                    "symbol": sym,
                    "trades": result.total_trades,
                    "return": float(result.total_return),
                    "sharpe": float(result.sharpe),
                    "max_dd": float(result.max_drawdown),
                    "win_rate": float(result.win_rate),
                    "pnl": float(sum(t.pnl for t in trades)),
                })

        # Per-symbol table
        print(f"\n  {'Symbol':<8} {'Trades':>7} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} "
              f"{'WinRate':>8} {'PnL':>12}")
        print(f"  {'-'*62}")
        for r in sorted(strat_rows, key=lambda x: x["return"], reverse=True):
            print(f"  {r['symbol']:<8} {r['trades']:>7} {r['return']:>+8.2%} "
                  f"{r['sharpe']:>+7.3f} {r['max_dd']:>7.2%} "
                  f"{r['win_rate']:>7.1%} ${r['pnl']:>+10,.0f}")

        # Aggregate
        total_trades = len(all_trades)
        if total_trades == 0:
            print("\n  No trades generated.")
            continue

        wins = sum(1 for t in all_trades if t.pnl > 0)
        losses = sum(1 for t in all_trades if t.pnl < 0)
        net_pnl = sum(float(t.pnl) for t in all_trades)
        avg_return = sum(all_returns) / len(all_returns)

        print(f"\n  AGGREGATE:")
        print(f"    Total trades:  {total_trades} ({wins}W / {losses}L)")
        print(f"    Win rate:      {wins/total_trades:.1%}")
        print(f"    Net PnL:       ${net_pnl:+,.0f}")
        print(f"    Avg trade ret: {avg_return:+.2f}%")

        # Monte Carlo bootstrap
        if len(all_returns) >= 30:
            print(f"\n  MONTE CARLO BOOTSTRAP ({args.bootstrap} simulations, 95% CI)")
            bootstrap = bootstrap_metrics(all_returns, n_simulations=args.bootstrap, seed=42)
            print(f"    Total Return:  {bootstrap.total_return.point_estimate:+.2f}% "
                  f"[{bootstrap.total_return.lower:+.2f}%, {bootstrap.total_return.upper:+.2f}%]")
            print(f"    Sharpe Ratio:  {bootstrap.sharpe_ratio.point_estimate:+.4f} "
                  f"[{bootstrap.sharpe_ratio.lower:+.4f}, {bootstrap.sharpe_ratio.upper:+.4f}]")
            print(f"    Max Drawdown:  {bootstrap.max_drawdown.point_estimate:.2f}% "
                  f"[{bootstrap.max_drawdown.lower:.2f}%, {bootstrap.max_drawdown.upper:.2f}%]")
            print(f"    Win Rate:      {bootstrap.win_rate.point_estimate:.1f}% "
                  f"[{bootstrap.win_rate.lower:.1f}%, {bootstrap.win_rate.upper:.1f}%]")
            print(f"    Profit Factor: {bootstrap.profit_factor.point_estimate:.3f} "
                  f"[{bootstrap.profit_factor.lower:.3f}, {bootstrap.profit_factor.upper:.3f}]")

            # Verdict
            sharpe_ok = bootstrap.sharpe_ratio.lower > 0
            return_ok = bootstrap.total_return.lower > 0
            verdict = "PASS" if (sharpe_ok and return_ok) else "FAIL"
            print(f"\n    Verdict: {verdict}")
            if not sharpe_ok:
                print(f"      Sharpe CI lower = {bootstrap.sharpe_ratio.lower:+.4f} <= 0")
            if not return_ok:
                print(f"      Return CI lower = {bootstrap.total_return.lower:+.2f}% <= 0")
        else:
            print(f"\n  Insufficient trades ({total_trades}) for Monte Carlo (need >= 30)")

    print(f"\n{'=' * _SEP}")
    print("  BACKTEST COMPLETE")
    print(f"{'=' * _SEP}")


if __name__ == "__main__":
    main()
