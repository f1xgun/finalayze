"""Run combined portfolio iteration across all layers.

Each portfolio layer gets its own broker instance with pre-allocated cash,
enabling per-layer PnL tracking and independent drawdown monitoring.

Layers:
  Core (45%)      - OFZ-PK carry (BondBacktestEngine)
  Strategic (27.5%) - OFZ-PD duration rotation (BondBacktestEngine)
  Tactical (17.5%)  - CBR event trading (BondBacktestEngine)
  Short (10%)       - Equity MR + dividend gap (BacktestEngine)

Usage:
    uv run python scripts/run_portfolio_iteration.py \
        --name "portfolio-baseline" \
        --description "Full 4-layer portfolio" \
        --total-cash 1500000 \
        --start 2022-01-01 \
        --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Ensure config/ at project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_bond_iteration import (  # Import bond iteration helpers (reuse fetch/build logic)
    BOND_UNIVERSE,
    _build_carry_strategy,
    _build_cbr_event_strategy,
    _build_duration_rotation_strategy,
    _fetch_bond_data,
    _load_preset,
    _make_tinkoff_fetcher,
)

from finalayze.backtest.bond_engine import BondBacktestConfig, BondBacktestEngine
from finalayze.backtest.bond_metrics import compute_bond_metrics
from finalayze.backtest.costs import MOEX_BOND_COSTS, MOEX_COSTS
from finalayze.backtest.portfolio_aggregator import (
    LayerResult,
    PortfolioAggregator,
    PortfolioResult,
)
from finalayze.core.schemas import DEFAULT_LAYER_CONFIGS, PortfolioLayer
from finalayze.data.fetchers.cbr import MacroContextProvider
from finalayze.risk.yield_stop import YieldStop

_DEFAULT_TOTAL_CASH = Decimal(1_500_000)  # 1.5M RUB default

# Short-layer equity universe (top-3 MOEX blue chips by liquidity)
# Concentrated to ensure viable position sizing (150K / 3 = 50K per symbol)
_SHORT_EQUITY_SYMBOLS = ["SBER", "GAZP", "LKOH"]


def _allocate_cash(
    total_cash: Decimal,
) -> dict[str, Decimal]:
    """Allocate cash across layers based on DEFAULT_LAYER_CONFIGS percentages.

    Returns:
        Dict mapping layer name to allocated cash (in RUB).
    """
    allocations: dict[str, Decimal] = {}
    for layer, config in DEFAULT_LAYER_CONFIGS.items():
        allocations[layer.value] = total_cash * config.capital_pct
    return allocations


def _run_core_layer(
    cash: Decimal,
    start: datetime,
    end: datetime,
    fetcher: Any,
) -> LayerResult | None:
    """Run Core layer: OFZ-PK carry strategy."""
    segment = "ru_ofz_pk"
    symbols = BOND_UNIVERSE.get(segment, [])
    if not symbols:
        return None

    print(f"\n  {'─' * 60}")
    print(f"  CORE LAYER (OFZ-PK carry) | Cash: {cash:,.0f} RUB")
    print(f"  {'─' * 60}")

    candles_by_symbol, bond_info, coupon_schedule = _fetch_bond_data(fetcher, symbols, start, end)
    if not candles_by_symbol:
        print("  No data for core layer")
        return None

    preset = _load_preset(segment)
    strategy = _build_carry_strategy(symbols, bond_info, preset)

    layer_cfg = DEFAULT_LAYER_CONFIGS[PortfolioLayer.CORE]
    config = BondBacktestConfig(
        initial_cash=cash,
        max_positions=layer_cfg.max_positions,
        yield_stop=YieldStop(threshold_bps=layer_cfg.yield_stop_bps),
        transaction_costs=MOEX_BOND_COSTS,
        max_hold_bars=252,  # quarterly rebalance, long hold
    )

    engine = BondBacktestEngine(config=config)
    result = engine.run(
        candles_by_symbol=candles_by_symbol,
        bond_info=bond_info,
        coupon_schedule=coupon_schedule,
        strategy_fn=strategy.generate_signal,
    )

    equity_float = [float(v) for v in result.equity_curve]
    metrics = compute_bond_metrics(
        equity_curve=equity_float,
        dates=result.dates,
        trades=result.trades,
        coupon_income_gross=float(result.total_coupon_income_gross),
        coupon_income_net=float(result.total_coupon_income_net),
        initial_cash=float(cash),
    )

    print(
        f"  Core: {metrics.trade_count} trades | "
        f"Return {metrics.total_return_pct:+.2f}% | "
        f"Excess Sharpe {metrics.excess_sharpe:+.4f} | "
        f"DD {metrics.max_drawdown_pct:.2f}%"
    )

    return LayerResult(
        layer_id="core",
        equity_curve=equity_float,
        dates=result.dates,
        trades=result.trades,
        total_return_pct=metrics.total_return_pct,
        max_drawdown_pct=metrics.max_drawdown_pct,
        coupon_income_net=float(result.total_coupon_income_net),
        sharpe=metrics.excess_sharpe,
    )


def _run_strategic_layer(
    cash: Decimal,
    start: datetime,
    end: datetime,
    fetcher: Any,
) -> LayerResult | None:
    """Run Strategic layer: OFZ-PD duration rotation."""
    segment = "ru_ofz_pd"
    symbols = BOND_UNIVERSE.get(segment, [])
    if not symbols:
        return None

    print(f"\n  {'─' * 60}")
    print(f"  STRATEGIC LAYER (OFZ-PD duration rotation) | Cash: {cash:,.0f} RUB")
    print(f"  {'─' * 60}")

    candles_by_symbol, bond_info, coupon_schedule = _fetch_bond_data(fetcher, symbols, start, end)
    if not candles_by_symbol:
        print("  No data for strategic layer")
        return None

    strategy = _build_duration_rotation_strategy(symbols, bond_info)

    layer_cfg = DEFAULT_LAYER_CONFIGS[PortfolioLayer.STRATEGIC]
    config = BondBacktestConfig(
        initial_cash=cash,
        max_positions=layer_cfg.max_positions,
        yield_stop=YieldStop(threshold_bps=layer_cfg.yield_stop_bps),
        transaction_costs=MOEX_BOND_COSTS,
        max_hold_bars=120,  # ~6 months
    )

    macro_provider = MacroContextProvider()
    engine = BondBacktestEngine(config=config)
    result = engine.run(
        candles_by_symbol=candles_by_symbol,
        bond_info=bond_info,
        coupon_schedule=coupon_schedule,
        strategy_fn=strategy.generate_signal,
        macro_provider=macro_provider,
    )

    equity_float = [float(v) for v in result.equity_curve]
    metrics = compute_bond_metrics(
        equity_curve=equity_float,
        dates=result.dates,
        trades=result.trades,
        coupon_income_gross=float(result.total_coupon_income_gross),
        coupon_income_net=float(result.total_coupon_income_net),
        initial_cash=float(cash),
    )

    print(
        f"  Strategic: {metrics.trade_count} trades | "
        f"Return {metrics.total_return_pct:+.2f}% | "
        f"Excess Sharpe {metrics.excess_sharpe:+.4f} | "
        f"DD {metrics.max_drawdown_pct:.2f}%"
    )

    return LayerResult(
        layer_id="strategic",
        equity_curve=equity_float,
        dates=result.dates,
        trades=result.trades,
        total_return_pct=metrics.total_return_pct,
        max_drawdown_pct=metrics.max_drawdown_pct,
        coupon_income_net=float(result.total_coupon_income_net),
        sharpe=metrics.excess_sharpe,
    )


def _run_tactical_layer(
    cash: Decimal,
    start: datetime,
    end: datetime,
    fetcher: Any,
) -> LayerResult | None:
    """Run Tactical layer: CBR event strategy on OFZ-PD."""
    segment = "ru_ofz_pd"
    symbols = BOND_UNIVERSE.get(segment, [])
    if not symbols:
        return None

    print(f"\n  {'─' * 60}")
    print(f"  TACTICAL LAYER (CBR event trading) | Cash: {cash:,.0f} RUB")
    print(f"  {'─' * 60}")

    candles_by_symbol, bond_info, coupon_schedule = _fetch_bond_data(fetcher, symbols, start, end)
    if not candles_by_symbol:
        print("  No data for tactical layer")
        return None

    preset = _load_preset(segment)
    strategy = _build_cbr_event_strategy(preset)

    layer_cfg = DEFAULT_LAYER_CONFIGS[PortfolioLayer.TACTICAL]
    config = BondBacktestConfig(
        initial_cash=cash,
        max_positions=layer_cfg.max_positions,
        yield_stop=YieldStop(threshold_bps=layer_cfg.yield_stop_bps),
        transaction_costs=MOEX_BOND_COSTS,
        max_hold_bars=20,  # short-term event trades
    )

    macro_provider = MacroContextProvider()
    engine = BondBacktestEngine(config=config)
    result = engine.run(
        candles_by_symbol=candles_by_symbol,
        bond_info=bond_info,
        coupon_schedule=coupon_schedule,
        strategy_fn=strategy.generate_signal,
        macro_provider=macro_provider,
    )

    equity_float = [float(v) for v in result.equity_curve]
    metrics = compute_bond_metrics(
        equity_curve=equity_float,
        dates=result.dates,
        trades=result.trades,
        coupon_income_gross=float(result.total_coupon_income_gross),
        coupon_income_net=float(result.total_coupon_income_net),
        initial_cash=float(cash),
    )

    print(
        f"  Tactical: {metrics.trade_count} trades | "
        f"Return {metrics.total_return_pct:+.2f}% | "
        f"Excess Sharpe {metrics.excess_sharpe:+.4f} | "
        f"DD {metrics.max_drawdown_pct:.2f}%"
    )

    return LayerResult(
        layer_id="tactical",
        equity_curve=equity_float,
        dates=result.dates,
        trades=result.trades,
        total_return_pct=metrics.total_return_pct,
        max_drawdown_pct=metrics.max_drawdown_pct,
        coupon_income_net=float(result.total_coupon_income_net),
        sharpe=metrics.excess_sharpe,
    )


def _run_short_symbol(
    symbol: str,
    segment: str,
    candles: list[Any],
    strategies: list[Any],
    cash_per_symbol: Decimal,
    max_positions: int,
) -> tuple[list[Any], list[Any]]:
    """Run equity backtest for one symbol. Returns (trades, snapshots)."""
    from finalayze.backtest.config import (  # noqa: PLC0415
        DEFAULT_STRATEGY_HOLD_BARS,
        BacktestConfig,
    )
    from finalayze.backtest.engine import BacktestEngine  # noqa: PLC0415
    from finalayze.backtest.journaling_combiner import (  # noqa: PLC0415
        JournalingStrategyCombiner,
    )
    from finalayze.backtest.performance import PerformanceAnalyzer  # noqa: PLC0415

    combiner = JournalingStrategyCombiner(
        strategies=strategies,
        allocation_mode="hrp",
    )
    engine = BacktestEngine(
        strategy=combiner,
        config=BacktestConfig(
            initial_cash=cash_per_symbol,
            max_positions=max_positions,
            stop_loss_mode="chandelier",
            max_hold_bars=DEFAULT_STRATEGY_HOLD_BARS,
            transaction_costs=MOEX_COSTS,
        ),
    )
    trades, snapshots = engine.run(symbol=symbol, segment_id=segment, candles=candles)

    sharpe_val = 0.0
    if trades:
        pa_result = PerformanceAnalyzer().analyze(trades, snapshots)
        sharpe_val = float(pa_result.sharpe) if pa_result else 0.0

    print(
        f"    {symbol:12s} | {len(candles):4d} bars | "
        f"{len(trades):3d} trades | Sharpe {sharpe_val:+.3f}"
    )
    return trades, snapshots


def _snapshots_to_equity_curve(
    all_snapshots: list[Any],
) -> tuple[list[float], list[date]]:
    """Convert portfolio snapshots to an equity curve with dates."""
    snapshot_by_date: dict[date, float] = {}
    for s in all_snapshots:
        d = s.timestamp.date()
        snapshot_by_date[d] = snapshot_by_date.get(d, 0.0) + float(s.equity)

    sorted_dates = sorted(snapshot_by_date.keys())
    equity_curve = [snapshot_by_date[d] for d in sorted_dates]
    return equity_curve, sorted_dates


def _run_short_layer(
    cash: Decimal,
    start: datetime,
    end: datetime,
) -> LayerResult | None:
    """Run Short layer: equity MR + dividend gap on MOEX blue chips."""
    print(f"\n  {'─' * 60}")
    print(f"  SHORT LAYER (equity MR + dividend gap) | Cash: {cash:,.0f} RUB")
    print(f"  {'─' * 60}")

    from run_iteration import _make_moex_fetcher, _setup_dividend_gap_strategy  # noqa: PLC0415

    from finalayze.core.schemas import PortfolioState, TradeResult  # noqa: PLC0415
    from finalayze.data.fetchers.caching import CachingFetcher  # noqa: PLC0415
    from finalayze.strategies.mean_reversion import MeanReversionStrategy  # noqa: PLC0415
    from finalayze.strategies.ou_mean_reversion import OUMeanReversionStrategy  # noqa: PLC0415
    from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy  # noqa: PLC0415

    segment = "ru_blue_chips"
    symbols = _SHORT_EQUITY_SYMBOLS
    fetcher = CachingFetcher(_make_moex_fetcher())

    strategies: list[Any] = [
        MeanReversionStrategy(),
        OUMeanReversionStrategy(use_mle=True),
        RSI2ConnorsStrategy(),
    ]

    div_gap = _setup_dividend_gap_strategy(segment, symbols, fetcher, start, end)
    if div_gap is not None:
        strategies.append(div_gap)

    all_trades: list[TradeResult] = []
    all_snapshots: list[PortfolioState] = []
    layer_cfg = DEFAULT_LAYER_CONFIGS[PortfolioLayer.SHORT]

    # Split cash equally across symbols so combined equity matches total allocation
    n_symbols = len(symbols)
    cash_per_symbol = cash / Decimal(n_symbols) if n_symbols > 0 else cash

    for symbol in symbols:
        try:
            candles = fetcher.fetch_candles(symbol, start, end)
            if not candles:
                print(f"    {symbol:12s} | no data")
                continue
        except Exception:
            print(f"    {symbol:12s} | fetch failed")
            continue

        try:
            trades, snapshots = _run_short_symbol(
                symbol,
                segment,
                candles,
                strategies,
                cash_per_symbol,
                layer_cfg.max_positions,
            )
            all_trades.extend(trades)
            if snapshots:
                all_snapshots.extend(snapshots)
        except Exception:
            print(f"    {symbol:12s} | ERROR: {traceback.format_exc().splitlines()[-1]}")

    if not all_snapshots:
        print("  Short layer: no results")
        return None

    equity_curve, sorted_dates = _snapshots_to_equity_curve(all_snapshots)
    total_return = (equity_curve[-1] / equity_curve[0] - 1.0) * 100 if equity_curve else 0.0
    max_dd = _compute_max_dd(equity_curve) * 100

    # Compute profit factor from trade PnL
    short_pf = _compute_profit_factor(all_trades)

    print(
        f"  Short: {len(all_trades)} trades | Return {total_return:+.2f}% | "
        f"DD {max_dd:.2f}% | PF {short_pf:.2f}"
    )

    return LayerResult(
        layer_id="short",
        equity_curve=equity_curve,
        dates=sorted_dates,
        trades=all_trades,
        total_return_pct=total_return,
        max_drawdown_pct=max_dd,
        profit_factor=short_pf,
    )


def _compute_profit_factor(trades: list[Any]) -> float:
    """Compute profit factor from trade list (gross_profit / gross_loss)."""
    gross_profit = 0.0
    gross_loss = 0.0
    for t in trades:
        pnl = float(getattr(t, "pnl", 0))
        if pnl > 0:
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += abs(pnl)
    if gross_loss <= 0:
        return 999.0 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _compute_max_dd(equity_curve: list[float]) -> float:
    """Compute max drawdown as a fraction."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        peak = max(peak, val)
        dd = (peak - val) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _format_gate(passed: bool) -> str:
    """Return [PASS] or [FAIL] label for a gate check."""
    return "[PASS]" if passed else "[FAIL]"


def _print_phase4_exit_criteria(result: PortfolioResult) -> None:
    """Print the Phase 4 multi-tier exit criteria evaluation."""
    print("\n  Phase 4 Exit Criteria (revised):")

    # Hard gates
    hard_1 = result.max_drawdown_pct < 10.0  # noqa: PLR2004
    hard_2 = result.core_return_vs_ruonia > -2.0  # noqa: PLR2004
    hard_3 = result.strategic_dd_ok
    hard_4 = result.tactical_has_trades

    print("  Hard Gates (all must pass):")
    print(f"    {_format_gate(hard_1)} Portfolio DD < 10%: {result.max_drawdown_pct:.2f}%")
    print(
        f"    {_format_gate(hard_2)} Core return vs RUONIA: "
        f"{result.core_return_vs_ruonia:+.2f}% above RUONIA - 200bps"
    )

    # Get strategic DD for display
    strategic_lr = result.layer_results.get("strategic")
    strategic_dd_str = f"{strategic_lr.max_drawdown_pct:.2f}%" if strategic_lr else "N/A"
    print(f"    {_format_gate(hard_3)} Strategic DD < 8%: {strategic_dd_str}")

    # Get tactical trade count for display
    tactical_lr = result.layer_results.get("tactical")
    tactical_trades_str = f"{len(tactical_lr.trades)} trades" if tactical_lr else "N/A"
    print(f"    {_format_gate(hard_4)} Tactical has trades: {tactical_trades_str}")

    # Soft gates
    # Recompute display values for soft gates
    core_lr = result.layer_results.get("core")
    if core_lr and core_lr.max_drawdown_pct > 0:
        # Approximate core Calmar from available data
        # (aggregator computed it, but we display from layer data)
        n_years = len(result.combined_dates) / 252 if result.combined_dates else 0
        if n_years > 0:
            core_ann = ((1 + core_lr.total_return_pct / 100) ** (1.0 / n_years) - 1.0) * 100
            core_calmar = core_ann / core_lr.max_drawdown_pct
        else:
            core_calmar = 0.0
    else:
        core_calmar = 0.0

    short_lr = result.layer_results.get("short")
    short_pf = short_lr.profit_factor if short_lr else 0.0

    soft_1 = core_calmar > 2.0  # noqa: PLR2004
    soft_2 = result.absolute_sharpe > 0.3  # noqa: PLR2004
    soft_3 = short_pf > 1.0

    print("  Soft Gates (2 of 3 must pass):")
    print(f"    {_format_gate(soft_1)} Core Calmar > 2.0: {core_calmar:.2f}")
    print(f"    {_format_gate(soft_2)} Absolute Sharpe > 0.3: {result.absolute_sharpe:.2f}")
    print(f"    {_format_gate(soft_3)} Short equity PF > 1.0: {short_pf:.2f}")

    # Overall result
    if result.phase4_exit_ok:
        verdict = "PASS"
    elif result.hard_gates_passed == result.hard_gates_total:
        verdict = "CONDITIONAL PASS (soft gates insufficient)"
    elif result.soft_gates_passed >= 2:  # noqa: PLR2004
        verdict = "CONDITIONAL PASS (hard gates failed)"
    else:
        verdict = "FAIL"
    print(
        f"  Result: {result.hard_gates_passed}/{result.hard_gates_total} hard, "
        f"{result.soft_gates_passed}/{result.soft_gates_total} soft -> {verdict}"
    )


def _print_portfolio_result(result: PortfolioResult) -> None:
    """Print formatted portfolio-level results."""
    print(f"\n{'=' * 72}")
    print("  PORTFOLIO RESULT (all layers combined)")
    print(f"{'=' * 72}")
    print(f"  {'Total Return:':<30} {result.total_return_pct:>+8.2f}%")
    print(f"  {'Annualized Return:':<30} {result.annualized_return_pct:>+8.2f}%")
    print(f"  {'Excess Return (vs RUONIA):':<30} {result.excess_return_pct:>+8.2f}%")
    print(f"  {'Excess Sharpe:':<30} {result.excess_sharpe:>+8.4f}")
    print(f"  {'Absolute Sharpe:':<30} {result.absolute_sharpe:>+8.4f}")
    print(f"  {'Max Drawdown:':<30} {result.max_drawdown_pct:>8.2f}%")
    print(f"  {'DD Breach (>10%):':<30} {'YES' if result.portfolio_dd_breach else 'NO':>8}")
    if result.portfolio_dd_breach_date:
        print(f"  {'DD Breach Date:':<30} {result.portfolio_dd_breach_date!s:>8}")
    print(f"  {'Total Trades:':<30} {result.total_trades:>8d}")
    print(f"  {'Total Coupon Income (net):':<30} {result.total_coupon_income_net:>12,.0f} RUB")

    if result.layer_return_contribution:
        print("\n  Per-layer PnL contribution:")
        for layer_id, pct in sorted(result.layer_return_contribution.items()):
            print(f"    {layer_id:<20s} {pct:>7.1f}%")

    if result.layer_results:
        print("\n  Per-layer summary:")
        print(f"  {'Layer':<12s} {'Return':>10s} {'DD':>8s} {'Trades':>8s} {'Sharpe':>10s}")
        print(f"  {'─' * 48}")
        for layer_id, lr in sorted(result.layer_results.items()):
            print(
                f"  {layer_id:<12s} {lr.total_return_pct:>+9.2f}% "
                f"{lr.max_drawdown_pct:>7.2f}% "
                f"{len(lr.trades):>8d} "
                f"{lr.sharpe:>+10.4f}"
            )

    # ── Phase 4 Exit Criteria (revised) ─────────────────────────────────
    _print_phase4_exit_criteria(result)
    print()


def _save_portfolio_result(
    result: PortfolioResult,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Save portfolio result as JSON."""
    summary: dict[str, Any] = {
        "name": args.name,
        "description": args.description,
        "period": {"start": args.start, "end": args.end},
        "total_cash": str(args.total_cash),
        "portfolio": {
            "total_return_pct": result.total_return_pct,
            "annualized_return_pct": result.annualized_return_pct,
            "excess_return_pct": result.excess_return_pct,
            "excess_sharpe": result.excess_sharpe,
            "absolute_sharpe": result.absolute_sharpe,
            "max_drawdown_pct": result.max_drawdown_pct,
            "portfolio_dd_breach": result.portfolio_dd_breach,
            "total_trades": result.total_trades,
            "total_coupon_income_net": result.total_coupon_income_net,
        },
        "phase4_exit_criteria": {
            "core_return_vs_ruonia": result.core_return_vs_ruonia,
            "strategic_dd_ok": result.strategic_dd_ok,
            "tactical_has_trades": result.tactical_has_trades,
            "hard_gates_passed": result.hard_gates_passed,
            "hard_gates_total": result.hard_gates_total,
            "soft_gates_passed": result.soft_gates_passed,
            "soft_gates_total": result.soft_gates_total,
            "phase4_exit_ok": result.phase4_exit_ok,
        },
        "layer_contributions": result.layer_return_contribution,
        "layers": {},
    }

    for layer_id, lr in result.layer_results.items():
        summary["layers"][layer_id] = {
            "total_return_pct": lr.total_return_pct,
            "max_drawdown_pct": lr.max_drawdown_pct,
            "trade_count": len(lr.trades),
            "coupon_income_net": lr.coupon_income_net,
            "sharpe": lr.sharpe,
        }

    output_path = output_dir / "portfolio_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Saved portfolio results to: {output_dir}")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run combined portfolio iteration")
    parser.add_argument("--name", required=True, help="Iteration name")
    parser.add_argument("--description", required=True, help="What changed")
    parser.add_argument(
        "--total-cash",
        type=int,
        default=1_500_000,
        help="Total portfolio cash in RUB (default: 1,500,000)",
    )
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="results/iterations/", help="Output root")
    parser.add_argument(
        "--skip-bonds",
        action="store_true",
        help="Skip bond layers (run equity short layer only)",
    )
    parser.add_argument(
        "--skip-equity",
        action="store_true",
        help="Skip equity short layer (run bond layers only)",
    )
    return parser.parse_args()


def main() -> None:
    """Run combined portfolio iteration across all layers."""
    args = _parse_args()
    total_cash = Decimal(args.total_cash)
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    output_dir = Path(args.output) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    allocations = _allocate_cash(total_cash)

    print(f"\nPortfolio Iteration: '{args.name}'")
    print(f"  Description: {args.description}")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Total Cash: {total_cash:,.0f} RUB")
    print("\n  Layer Allocations:")
    for layer_name, layer_cash in allocations.items():
        pct = DEFAULT_LAYER_CONFIGS[PortfolioLayer(layer_name)].capital_pct * 100
        print(f"    {layer_name:<12s} {float(pct):5.1f}%  = {layer_cash:>12,.0f} RUB")

    layer_results: list[LayerResult] = []

    # Bond layers (core, strategic, tactical)
    if not args.skip_bonds:
        fetcher = _make_tinkoff_fetcher()
        if fetcher is None:
            print("\n  WARNING: FINALAYZE_TINKOFF_TOKEN not set.")
            print("  Bond layers (core, strategic, tactical) will be skipped.")
        else:
            # Core layer
            core = _run_core_layer(allocations["core"], start, end, fetcher)
            if core is not None:
                layer_results.append(core)

            # Strategic layer
            strategic = _run_strategic_layer(allocations["strategic"], start, end, fetcher)
            if strategic is not None:
                layer_results.append(strategic)

            # Tactical layer
            tactical = _run_tactical_layer(allocations["tactical"], start, end, fetcher)
            if tactical is not None:
                layer_results.append(tactical)
    else:
        print("\n  Bond layers skipped (--skip-bonds)")

    # Short equity layer
    if not args.skip_equity:
        short = _run_short_layer(allocations["short"], start, end)
        if short is not None:
            layer_results.append(short)
    else:
        print("\n  Equity layer skipped (--skip-equity)")

    if not layer_results:
        print("\n  No layers produced results. Cannot aggregate portfolio.")
        return

    # Aggregate results
    aggregator = PortfolioAggregator()
    portfolio_result = aggregator.aggregate(layer_results)

    _print_portfolio_result(portfolio_result)
    _save_portfolio_result(portfolio_result, output_dir, args)


if __name__ == "__main__":
    main()
