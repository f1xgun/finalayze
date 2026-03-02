"""Run MOEX evaluation using Tinkoff Invest API for real Russian stock data.

Fetches full MOEX history (2019-2025) via Tinkoff gRPC API.
Requires TINKOFF_TOKEN environment variable.

Usage:
    TINKOFF_TOKEN=t.xxx uv run python scripts/run_moex_tinkoff_eval.py \
        --output results/moex-tinkoff/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
import subprocess

import grpc
from t_tech.invest import CandleInterval
from t_tech.invest.async_services import AsyncServices
from t_tech.invest.channels import _required_options

from finalayze.backtest.decision_journal import DecisionJournal
from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner
from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.core.schemas import Candle
from finalayze.markets.instruments import (
    DEFAULT_MOEX_INSTRUMENTS,
    Instrument,
    InstrumentRegistry,
)
from finalayze.ml.registry import MLModelRegistry
from finalayze.risk.kelly import RollingKelly
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.mean_reversion import MeanReversionStrategy
from finalayze.strategies.ml_strategy import MLStrategy
from finalayze.strategies.momentum import MomentumStrategy
from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy

# Extended MOEX instrument list (beyond defaults)
EXTRA_MOEX_INSTRUMENTS: list[Instrument] = [
    Instrument(
        symbol="MOEX",
        market_id="moex",
        name="Moscow Exchange",
        instrument_type="stock",
        figi="BBG004730JJ5",
        lot_size=10,
        currency="RUB",
    ),
    Instrument(
        symbol="MGNT",
        market_id="moex",
        name="Magnit",
        instrument_type="stock",
        figi="BBG004RVFCY3",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="POLY",
        market_id="moex",
        name="Polymetal",
        instrument_type="stock",
        figi="BBG004PYF2N3",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="TATN",
        market_id="moex",
        name="Tatneft",
        instrument_type="stock",
        figi="BBG004RVFFC0",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="NLMK",
        market_id="moex",
        name="NLMK",
        instrument_type="stock",
        figi="BBG004S681B4",
        lot_size=10,
        currency="RUB",
    ),
    Instrument(
        symbol="CHMF",
        market_id="moex",
        name="Severstal",
        instrument_type="stock",
        figi="BBG00475K6C3",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="ALRS",
        market_id="moex",
        name="Alrosa",
        instrument_type="stock",
        figi="BBG004S68B31",
        lot_size=10,
        currency="RUB",
    ),
    Instrument(
        symbol="PLZL",
        market_id="moex",
        name="Polyus Gold",
        instrument_type="stock",
        figi="BBG000R607Y3",
        lot_size=1,
        currency="RUB",
    ),
]

# Segment → symbol mapping for MOEX
MOEX_UNIVERSE: dict[str, list[str]] = {
    "ru_blue_chips": ["SBER", "GAZP", "LKOH", "GMKN", "VTBR", "MOEX", "MGNT", "TATN"],
    "ru_energy": ["GAZP", "LKOH", "NVTK", "ROSN", "TATN"],
    "ru_finance": ["SBER", "VTBR", "MOEX"],
}


_NANO_DIVISOR = Decimal(1000000000)


def _quotation_to_decimal(q: Any) -> Decimal:
    """Convert Tinkoff Quotation(units, nano) to Decimal."""
    return Decimal(q.units) + Decimal(q.nano) / _NANO_DIVISOR


_TBANK_TARGET = "invest-public-api.tbank.ru:443"


def _get_tbank_root_certs() -> bytes:
    """Fetch TLS certificate chain from tbank.ru (Russian CA not in default bundle)."""
    cache_path = Path(__file__).parent / ".tbank_certs.pem"
    if cache_path.exists():
        return cache_path.read_bytes()
    result = subprocess.run(
        [  # noqa: S607
            "openssl",
            "s_client",
            "-showcerts",
            "-connect",
            "invest-public-api.tbank.ru:443",
            "-servername",
            "invest-public-api.tbank.ru",
        ],
        input=b"",
        capture_output=True,
        timeout=10,
        check=False,
    )
    pem_blocks: list[str] = []
    in_cert = False
    current: list[str] = []
    for line in result.stdout.decode().splitlines():
        if "BEGIN CERTIFICATE" in line:
            in_cert = True
            current = [line]
        elif "END CERTIFICATE" in line:
            current.append(line)
            pem_blocks.append("\n".join(current))
            in_cert = False
        elif in_cert:
            current.append(line)
    pem_data = "\n".join(pem_blocks).encode()
    cache_path.write_bytes(pem_data)
    return pem_data


def _create_tbank_channel(_token: str) -> grpc.aio.Channel:
    """Create gRPC channel with Russian CA certs for tbank.ru."""
    root_certs = _get_tbank_root_certs()
    creds = grpc.ssl_channel_credentials(root_certificates=root_certs)
    options: list[tuple[str, Any]] = list(_required_options)
    return grpc.aio.secure_channel(_TBANK_TARGET, creds, options)


async def _fetch_candles_async(
    token: str, figi: str, symbol: str, start: datetime, end: datetime
) -> list[Candle]:
    """Fetch candles via Tinkoff API with custom TLS for tbank.ru."""
    channel = _create_tbank_channel(token)
    try:
        services = AsyncServices(channel, token)
        response = await services.market_data.get_candles(
            figi=figi,
            from_=start,
            to=end,
            interval=CandleInterval.CANDLE_INTERVAL_DAY,
        )
    finally:
        await channel.close()

    candles: list[Candle] = []
    for raw in response.candles:
        # SDK returns datetime objects directly, not protobuf Timestamps
        ts = (
            raw.time
            if isinstance(raw.time, datetime)
            else datetime.fromtimestamp(raw.time.seconds + raw.time.nanos / 1e9, tz=UTC)
        )
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        candles.append(
            Candle(
                symbol=symbol,
                market_id="moex",
                timeframe="1d",
                timestamp=ts,
                open=_quotation_to_decimal(raw.open),
                high=_quotation_to_decimal(raw.high),
                low=_quotation_to_decimal(raw.low),
                close=_quotation_to_decimal(raw.close),
                volume=int(raw.volume),
                source="tinkoff",
            )
        )
    return candles


def fetch_tinkoff_candles(
    token: str, figi: str, symbol: str, start: datetime, end: datetime
) -> list[Candle]:
    """Sync wrapper for Tinkoff candle fetch."""
    return asyncio.run(_fetch_candles_async(token, figi, symbol, start, end))


def _build_registry() -> InstrumentRegistry:
    """Build registry with all MOEX instruments."""
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    for inst in EXTRA_MOEX_INSTRUMENTS:
        registry.register(inst)
    return registry


def _setup_ml_strategy(segment: str, models_dir: Path) -> MLStrategy | None:
    """Try to load ML models for a segment."""
    segment_dir = models_dir / segment
    if not segment_dir.exists():
        return None

    from finalayze.ml.models.ensemble import EnsembleModel  # noqa: PLC0415
    from finalayze.ml.models.lightgbm_model import LightGBMModel  # noqa: PLC0415
    from finalayze.ml.models.xgboost_model import XGBoostModel  # noqa: PLC0415

    models = []
    xgb_path = segment_dir / "xgb.pkl"
    lgbm_path = segment_dir / "lgbm.pkl"
    lstm_path = segment_dir / "lstm.pkl"
    lstm_model = None

    if xgb_path.exists():
        models.append(XGBoostModel.load_from(xgb_path))
    if lgbm_path.exists():
        models.append(LightGBMModel.load_from(lgbm_path))
    if lstm_path.exists():
        from finalayze.ml.models.lstm_model import LSTMModel  # noqa: PLC0415

        lstm_model = LSTMModel(segment_id=segment)
        lstm_model.load(lstm_path)

    if not models and lstm_model is None:
        return None

    ensemble = EnsembleModel(models=models, lstm_model=lstm_model)
    registry = MLModelRegistry()
    registry.register(segment, ensemble)
    return MLStrategy(registry)


def _build_strategies(
    segment: str,
    models_dir: Path | None,
) -> list[BaseStrategy]:
    """Build strategy list for evaluation."""
    strategies: list[BaseStrategy] = [
        MomentumStrategy(),
        MeanReversionStrategy(),
        RSI2ConnorsStrategy(),
    ]

    if models_dir is not None:
        ml = _setup_ml_strategy(segment, models_dir)
        if ml is not None:
            strategies.append(ml)
            print(f"    ML ensemble loaded for {segment}")

    return strategies


def _write_tinkoff_summary(output_dir: Path, all_results: list[dict[str, Any]]) -> None:
    """Write results JSON and print aggregate stats."""
    summary_path = output_dir / "moex_tinkoff_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  DONE: {len(all_results)} successful evaluations")
    print(f"  Results: {summary_path}")
    print(f"{'=' * 70}")

    if all_results:
        sharpes = [float(r["metrics"]["sharpe"]) for r in all_results]
        win_rates = [float(r["metrics"]["win_rate"]) for r in all_results]
        trades_list = [r["total_trades"] for r in all_results]
        pos_sharpe = sum(1 for s in sharpes if s > 0)
        print("\n  MOEX (Tinkoff) AGGREGATE STATS:")
        print(f"    Symbols evaluated:  {len(all_results)}")
        print(f"    Avg trades/symbol:  {sum(trades_list) / len(trades_list):.1f}")
        print(f"    Avg win rate:       {sum(win_rates) / len(win_rates) * 100:.1f}%")
        print(f"    Avg Sharpe:         {sum(sharpes) / len(sharpes):+.3f}")
        print(f"    Positive Sharpe:    {pos_sharpe}/{len(all_results)}")
        print()
        print("  Per-symbol ranking:")
        ranked = sorted(all_results, key=lambda x: float(x["metrics"]["sharpe"]), reverse=True)
        for r in ranked:
            s = r["metrics"]
            print(
                f"    {r['symbol']:8s} | Sharpe {float(s['sharpe']):+6.3f} | "
                f"WR {float(s['win_rate']) * 100:4.1f}% | "
                f"PF {float(s['profit_factor']):5.2f} | "
                f"{r['total_trades']} trades"
            )


def _eval_symbol(
    token: str,
    symbol: str,
    figi: str,
    segment: str,
    combiner: JournalingStrategyCombiner,
    start: datetime,
    end: datetime,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Evaluate a single MOEX symbol. Returns result dict or None."""
    min_bars = 100
    print(f"  Fetching {symbol} (FIGI={figi})...", end="", flush=True)
    candles = fetch_tinkoff_candles(token, figi, symbol, start, end)
    print(f" {len(candles)} bars")

    if len(candles) < min_bars:
        print(f"  {symbol:8s} | {len(candles)} bars — insufficient, skipping")
        return None

    sym_dir = output_dir / segment / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    journal = DecisionJournal(output_path=sym_dir / "decision_journal.jsonl")

    engine = BacktestEngine(
        strategy=combiner,
        initial_cash=Decimal(1000000),  # 1M RUB
        rolling_kelly=RollingKelly(window=50),
        decision_journal=journal,
    )
    trades, snapshots = engine.run(symbol, segment, candles)
    journal.flush()

    if not trades:
        print(f"  {symbol:8s} | {len(candles):5d} bars |   0 trades")
        return None

    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(trades, snapshots)

    sharpe = float(metrics.sharpe)
    wr = float(metrics.win_rate) * 100
    ret = float(metrics.total_return) * 100
    n_trades = metrics.total_trades
    pf = float(metrics.profit_factor)

    print(
        f"  {symbol:8s} | {len(candles):5d} bars | {n_trades:3d} trades | "
        f"Sharpe {sharpe:+6.3f} | WR {wr:4.1f}% | PF {pf:5.2f} | "
        f"Ret {ret:+7.3f}%"
    )

    return {
        "symbol": symbol,
        "segment": segment,
        "total_candles": len(candles),
        "total_trades": n_trades,
        "metrics": {
            "sharpe": str(metrics.sharpe),
            "win_rate": str(metrics.win_rate),
            "profit_factor": str(metrics.profit_factor),
            "total_return": str(metrics.total_return),
            "max_drawdown": str(metrics.max_drawdown),
            "total_trades": n_trades,
        },
    }


def run_evaluation(
    output_dir: Path,
    models_dir: Path | None = None,
    start_year: int = 2019,
    end_year: int = 2025,
) -> None:
    """Run MOEX evaluation via Tinkoff API."""
    token = os.environ.get("TINKOFF_TOKEN", "")
    if not token:
        print("ERROR: TINKOFF_TOKEN environment variable not set")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    _build_registry()  # ensure all MOEX instruments registered

    start = datetime(start_year, 1, 1, tzinfo=UTC)
    end = datetime(end_year, 1, 1, tzinfo=UTC)

    # Build FIGI lookup from registry
    figi_map: dict[str, str] = {}
    for inst in DEFAULT_MOEX_INSTRUMENTS + EXTRA_MOEX_INSTRUMENTS:
        if inst.figi:
            figi_map[inst.symbol] = inst.figi

    all_results: list[dict[str, Any]] = []

    for segment, symbols in MOEX_UNIVERSE.items():
        print(f"\n{'=' * 70}")
        print(f"  SEGMENT: {segment} ({len(symbols)} symbols, market=moex)")
        print(f"  Period: {start_year}-01-01 to {end_year}-01-01")
        print(f"{'=' * 70}")

        strategies = _build_strategies(segment, models_dir)
        strat_names = [s.name for s in strategies]
        print(f"  Strategies: {', '.join(strat_names)}")

        combiner = JournalingStrategyCombiner(strategies=strategies)

        seen: set[str] = set()
        for symbol in symbols:
            if symbol in seen:
                continue
            seen.add(symbol)

            figi = figi_map.get(symbol)
            if not figi:
                print(f"  {symbol:8s} | No FIGI registered — skipping")
                continue

            try:
                result = _eval_symbol(
                    token, symbol, figi, segment, combiner, start, end, output_dir
                )
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"\n  {symbol:8s} | ERROR: {e}")
                traceback.print_exc()

    _write_tinkoff_summary(output_dir, all_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MOEX evaluation via Tinkoff API")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--models-dir", default=None, help="Directory with trained ML models")
    parser.add_argument("--start-year", type=int, default=2019, help="Start year (default: 2019)")
    parser.add_argument("--end-year", type=int, default=2025, help="End year (default: 2025)")
    args = parser.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else None
    run_evaluation(Path(args.output), models_dir, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
