"""Run bond layer backtests for MOEX OFZ portfolio.

Usage:
    uv run python scripts/run_bond_iteration.py \
        --name "bond-baseline" \
        --description "Initial OFZ bond system" \
        --segments ru_ofz_pd,ru_ofz_pk

    uv run python scripts/run_bond_iteration.py \
        --name "bond-v2" \
        --description "Tuned duration rotation" \
        --segments ru_ofz_pd \
        --start 2022-01-01 --end 2025-12-31 \
        --cash 1000000
"""

from __future__ import annotations

import argparse
import json
import os
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

# gRPC env vars MUST be set before importing grpc (via t_tech.invest).
# C-ares DNS resolver may fail; force native (system) resolver.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
# T-Bank uses Russian Trusted Root CA not in standard CA bundles.
_GRPC_ROOTS = Path(PROJECT_ROOT) / "certs" / "grpc_roots.pem"
if _GRPC_ROOTS.exists():
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", str(_GRPC_ROOTS))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from finalayze.backtest.bond_engine import (
    BondBacktestConfig,
    BondBacktestEngine,
    BondBacktestResult,
)
from finalayze.backtest.bond_metrics import BondPerformanceMetrics, compute_bond_metrics
from finalayze.backtest.costs import MOEX_BOND_COSTS
from finalayze.core.schemas import BondInfo, Candle, CouponPayment
from finalayze.data.fetchers.cbr import MacroContextProvider
from finalayze.risk.yield_stop import YieldStop
from finalayze.strategies.bond_carry import BondCarryStrategy
from finalayze.strategies.bond_duration_rotation import BondDurationRotationStrategy
from finalayze.strategies.cbr_event import CBREventStrategy

_PRESETS_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "finalayze" / "strategies" / "presets"
)

# Bond ticker universes
OFZ_PD_TICKERS = [
    "SU26238RMFS4",
    "SU26239RMFS2",
    "SU26241RMFS8",
    "SU26243RMFS4",
    "SU26244RMFS2",
    "SU26246RMFS7",
    "SU26252RMFS5",
    "SU26253RMFS3",
]

OFZ_PK_TICKERS = [
    "SU29007RMFS0",
    "SU29008RMFS8",
    "SU29009RMFS6",
    "SU29010RMFS4",
]

BOND_UNIVERSE: dict[str, list[str]] = {
    "ru_ofz_pd": OFZ_PD_TICKERS,
    "ru_ofz_pk": OFZ_PK_TICKERS,
}


def _load_preset(segment: str) -> dict[str, Any]:
    """Load YAML preset for a segment."""
    preset_path = _PRESETS_DIR / f"{segment}.yaml"
    if not preset_path.exists():
        return {}
    with preset_path.open() as f:
        return yaml.safe_load(f) or {}


def _make_tinkoff_fetcher() -> Any:
    """Create TinkoffFetcher if token available, else None."""
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    if not token:
        return None
    from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415
    from finalayze.markets.instruments import build_default_registry  # noqa: PLC0415

    registry = build_default_registry()
    use_sandbox = os.environ.get("FINALAYZE_MODE", "sandbox") == "sandbox"
    return TinkoffFetcher(token=token, registry=registry, sandbox=use_sandbox)


def _fetch_bond_data(
    fetcher: Any,
    symbols: list[str],
    start: datetime,
    end: datetime,
) -> tuple[
    dict[str, list[Candle]],
    dict[str, BondInfo],
    dict[str, list[CouponPayment]],
]:
    """Fetch candles, bond info, and coupon schedules via Tinkoff API.

    Returns:
        Tuple of (candles_by_symbol, bond_info, coupon_schedule).
    """
    candles_by_symbol: dict[str, list[Candle]] = {}
    bond_info: dict[str, BondInfo] = {}
    coupon_schedule: dict[str, list[CouponPayment]] = {}

    for symbol in symbols:
        try:
            candles = fetcher.fetch_candles(symbol, start, end)
            if not candles:
                print(f"    {symbol:20s} | no candle data")
                continue
            candles_by_symbol[symbol] = candles

            info = fetcher.fetch_bond_info(symbol)

            coupons = fetcher.fetch_bond_coupons(symbol, start, end)
            coupon_schedule[symbol] = coupons

            # Compute effective coupon_rate from actual coupon payments.
            # T-Bank API returns coupon_rate=0 for floaters, and sometimes
            # omits it for fixed-rate bonds.  Deriving from real payments
            # ensures correct YTM / DV01 calculations.
            effective_rate = _compute_effective_coupon_rate(info, coupons)
            if effective_rate != info.coupon_rate:
                info = info.model_copy(update={"coupon_rate": effective_rate})
            bond_info[symbol] = info

            print(
                f"    {symbol:20s} | {len(candles):4d} bars | "
                f"coupon={info.coupon_rate}% | "
                f"maturity={info.maturity_date} | "
                f"{len(coupons)} coupons"
            )
        except Exception:
            print(f"    {symbol:20s} | fetch failed: {traceback.format_exc().splitlines()[-1]}")
            continue

    return candles_by_symbol, bond_info, coupon_schedule


def _compute_effective_coupon_rate(
    info: BondInfo,
    coupons: list[CouponPayment],
) -> Decimal:
    """Compute annualized coupon rate from actual coupon payments.

    For floaters (OFZ-PK): API returns coupon_rate=0, so we derive
    avg_coupon * frequency / face_value as an effective annual rate.
    For fixed-rate (OFZ-PD): verify/correct against actual payment amounts.
    """
    if not coupons or info.face_value <= 0 or info.coupon_frequency <= 0:
        return info.coupon_rate

    avg_coupon = sum(c.amount_per_bond for c in coupons) / len(coupons)
    annualized = avg_coupon * info.coupon_frequency / info.face_value * Decimal(100)
    return annualized.quantize(Decimal("0.01"))


def _build_carry_strategy(
    symbols: list[str],
    bond_info: dict[str, BondInfo],
    preset: dict[str, Any],
) -> BondCarryStrategy:
    """Build a BondCarryStrategy from bond metadata and preset."""
    maturity_dates: dict[str, date] = {}
    for sym in symbols:
        info = bond_info.get(sym)
        if info is not None:
            maturity_dates[sym] = info.maturity_date

    carry_params = preset.get("strategies", {}).get("bond_carry", {}).get("params", {})
    rebalance_interval = carry_params.get("rebalance_interval_bars", 63)

    return BondCarryStrategy(
        symbols=symbols,
        maturity_dates=maturity_dates,
        rebalance_interval=rebalance_interval,
    )


def _build_duration_rotation_strategy(
    symbols: list[str],
    bond_info: dict[str, BondInfo],
) -> BondDurationRotationStrategy:
    """Build a BondDurationRotationStrategy from bond metadata."""
    bond_durations: dict[str, Decimal] = {}
    bond_maturities: dict[str, date] = {}
    coupon_rates: dict[str, Decimal] = {}

    for sym in symbols:
        info = bond_info.get(sym)
        if info is None:
            continue
        # Estimate duration from maturity (rough approximation: duration ~ 0.8 * years)
        years_to_maturity = max(
            (info.maturity_date - datetime.now(tz=UTC).date()).days / 365.25,
            Decimal("0.1"),
        )
        bond_durations[sym] = Decimal(str(round(float(years_to_maturity) * 0.8, 2)))
        bond_maturities[sym] = info.maturity_date
        coupon_rates[sym] = info.coupon_rate

    return BondDurationRotationStrategy(
        bond_durations=bond_durations,
        bond_maturities=bond_maturities,
        coupon_rates=coupon_rates,
    )


def _build_cbr_event_strategy(preset: dict[str, Any]) -> CBREventStrategy:
    """Build a CBREventStrategy from preset."""
    cbr_params = preset.get("strategies", {}).get("cbr_event", {}).get("params", {})
    preferred = cbr_params.get("preferred_symbols")
    return CBREventStrategy(preferred_symbols=preferred)


def _build_bond_backtest_config(
    cash: Decimal,
    preset: dict[str, Any],
    segment: str,
) -> BondBacktestConfig:
    """Build BondBacktestConfig from preset."""
    risk_cfg = preset.get("risk", {})
    max_positions = risk_cfg.get("max_positions", 5)
    yield_stop_bps = risk_cfg.get("yield_stop_bps", 50)
    max_hold = 120 if segment == "ru_ofz_pd" else 252

    return BondBacktestConfig(
        initial_cash=cash,
        max_positions=max_positions,
        yield_stop=YieldStop(threshold_bps=yield_stop_bps),
        transaction_costs=MOEX_BOND_COSTS,
        max_hold_bars=max_hold,
    )


def _run_bond_segment(
    segment: str,
    cash: Decimal,
    start: datetime,
    end: datetime,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Run a bond backtest for a single segment.

    Returns:
        Summary dict or None if segment cannot be run (e.g. no Tinkoff token).
    """
    symbols = BOND_UNIVERSE.get(segment, [])
    if not symbols:
        print(f"  Segment '{segment}' has no bond universe, skipping")
        return None

    preset = _load_preset(segment)
    print(f"\n{'=' * 72}")
    print(f"  BOND SEGMENT: {segment} ({len(symbols)} bonds)")
    print(f"  Cash: {cash:,.0f} RUB")
    print(f"{'=' * 72}")

    # Fetch data
    fetcher = _make_tinkoff_fetcher()
    if fetcher is None:
        print("  WARNING: FINALAYZE_TINKOFF_TOKEN not set. Cannot fetch OFZ data.")
        print("  Skipping bond segment.")
        return None

    candles_by_symbol, bond_info, coupon_schedule = _fetch_bond_data(fetcher, symbols, start, end)

    if not candles_by_symbol:
        print("  No bond data available, skipping segment")
        return None

    # Build strategy
    macro_provider: MacroContextProvider | None = None
    if segment == "ru_ofz_pk":
        carry_strategy = _build_carry_strategy(symbols, bond_info, preset)
        strategy_fn = carry_strategy.generate_signal
        strategy_name = "bond_carry"
    elif segment == "ru_ofz_pd":
        dur_strategy = _build_duration_rotation_strategy(symbols, bond_info)
        strategy_fn = dur_strategy.generate_signal
        strategy_name = "bond_duration_rotation"
        macro_provider = MacroContextProvider()
    else:
        print(f"  Unknown bond segment '{segment}', skipping")
        return None

    # Build engine config
    config = _build_bond_backtest_config(cash, preset, segment)

    # Run backtest
    print(f"\n  Running backtest with strategy={strategy_name}...")
    engine = BondBacktestEngine(config=config)
    try:
        result: BondBacktestResult = engine.run(
            candles_by_symbol=candles_by_symbol,
            bond_info=bond_info,
            coupon_schedule=coupon_schedule,
            strategy_fn=strategy_fn,
            macro_provider=macro_provider,
        )
    except Exception:
        print(f"  ERROR running bond backtest: {traceback.format_exc().splitlines()[-1]}")
        return None

    # Compute performance metrics
    equity_float = [float(v) for v in result.equity_curve]
    metrics = compute_bond_metrics(
        equity_curve=equity_float,
        dates=result.dates,
        trades=result.trades,
        coupon_income_gross=float(result.total_coupon_income_gross),
        coupon_income_net=float(result.total_coupon_income_net),
        initial_cash=float(cash),
    )

    # Print results
    _print_bond_results(segment, result, metrics)

    # Save segment results
    seg_dir = output_dir / segment
    seg_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(segment, result, metrics, config)
    (seg_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    return summary


def _print_bond_results(
    segment: str,
    result: BondBacktestResult,
    metrics: BondPerformanceMetrics,
) -> None:
    """Print formatted bond backtest results."""
    print(f"\n  {segment} Results:")
    print(f"  {'─' * 50}")
    print(f"  {'Total Return:':<30} {metrics.total_return_pct:>+8.2f}%")
    print(f"  {'Annualized Return:':<30} {metrics.annualized_return_pct:>+8.2f}%")
    print(f"  {'Excess Return (vs RUONIA):':<30} {metrics.excess_return_pct:>+8.2f}%")
    print(f"  {'Excess Sharpe:':<30} {metrics.excess_sharpe:>+8.4f}")
    print(f"  {'Max Drawdown:':<30} {metrics.max_drawdown_pct:>8.2f}%")
    print(f"  {'Volatility (ann.):':<30} {metrics.annualized_volatility_pct:>8.2f}%")
    print(f"  {'Trade Count:':<30} {metrics.trade_count:>8d}")
    print(f"  {'Win Rate:':<30} {metrics.win_rate:>8.1%}")
    print(f"  {'Profit Factor:':<30} {metrics.profit_factor:>8.2f}")
    print(f"  {'Coupon Income (gross):':<30} {float(result.total_coupon_income_gross):>12,.0f} RUB")
    print(f"  {'Coupon Income (net):':<30} {float(result.total_coupon_income_net):>12,.0f} RUB")
    print(f"  {'Tax Paid (NDFL):':<30} {float(result.total_tax_paid):>12,.0f} RUB")
    print(f"  {'Coupon Contribution:':<30} {metrics.coupon_contribution_pct:>8.1f}%")
    print()


def _build_summary(
    segment: str,
    result: BondBacktestResult,
    metrics: BondPerformanceMetrics,
    config: BondBacktestConfig,
) -> dict[str, Any]:
    """Build JSON-serializable summary dict."""
    return {
        "segment": segment,
        "config": {
            "initial_cash": str(config.initial_cash),
            "max_positions": config.max_positions,
            "max_hold_bars": config.max_hold_bars,
            "yield_stop_bps": config.yield_stop.threshold_bps,
        },
        "results": {
            "total_return_pct": float(result.total_return_pct),
            "max_drawdown_pct": float(result.max_drawdown_pct),
            "sharpe_ratio": float(result.sharpe_ratio),
            "trade_count": result.trade_count,
            "win_rate": float(result.win_rate),
            "profit_factor": float(result.profit_factor),
            "coupon_income_gross": float(result.total_coupon_income_gross),
            "coupon_income_net": float(result.total_coupon_income_net),
            "tax_paid": float(result.total_tax_paid),
        },
        "metrics": {
            "total_return_pct": metrics.total_return_pct,
            "annualized_return_pct": metrics.annualized_return_pct,
            "excess_return_pct": metrics.excess_return_pct,
            "annualized_excess_return_pct": metrics.annualized_excess_return_pct,
            "excess_sharpe": metrics.excess_sharpe,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "annualized_volatility_pct": metrics.annualized_volatility_pct,
            "trade_count": metrics.trade_count,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "avg_hold_bars": metrics.avg_hold_bars,
            "coupon_contribution_pct": metrics.coupon_contribution_pct,
        },
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run bond layer backtest iteration")
    parser.add_argument("--name", required=True, help="Iteration name")
    parser.add_argument("--description", required=True, help="What changed")
    parser.add_argument(
        "--segments",
        default="ru_ofz_pd,ru_ofz_pk",
        help="Comma-separated bond segment IDs (default: ru_ofz_pd,ru_ofz_pk)",
    )
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--cash", type=int, default=1_000_000, help="Initial cash per segment in RUB"
    )
    parser.add_argument("--output", default="results/iterations/", help="Output root")
    return parser.parse_args()


def main() -> None:
    """Run bond iteration."""
    args = _parse_args()
    segments = [s.strip() for s in args.segments.split(",")]
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    cash = Decimal(args.cash)

    output_dir = Path(args.output) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBond Iteration: '{args.name}'")
    print(f"  Description: {args.description}")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Segments: {', '.join(segments)}")
    print(f"  Cash per segment: {cash:,.0f} RUB")

    all_summaries: list[dict[str, Any]] = []

    for segment in segments:
        summary = _run_bond_segment(
            segment=segment,
            cash=cash,
            start=start,
            end=end,
            output_dir=output_dir,
        )
        if summary is not None:
            all_summaries.append(summary)

    # Save consolidated summary
    consolidated_path = output_dir / "bond_summary.json"
    consolidated = {
        "name": args.name,
        "description": args.description,
        "period": {"start": args.start, "end": args.end},
        "segments": all_summaries,
    }
    consolidated_path.write_text(json.dumps(consolidated, indent=2, default=str))
    print(f"\n  Saved bond results to: {output_dir}")

    if not all_summaries:
        print("\n  WARNING: No bond segments produced results.")
        print("  Ensure FINALAYZE_TINKOFF_TOKEN is set for OFZ data access.")


if __name__ == "__main__":
    main()
