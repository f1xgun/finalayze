"""Run a new iteration: backtest, measure, gate, save, compare.

Usage:
    uv run python scripts/run_iteration.py \
        --name "baseline" \
        --description "Current system before improvements"

    uv run python scripts/run_iteration.py \
        --name "add-sentiment-to-momentum" \
        --description "Integrate LLM sentiment score" \
        --baseline latest

    uv run python scripts/run_iteration.py \
        --name "test-v1" \
        --description "Initial baseline" \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

# Ensure config/ at project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from finalayze.backtest.config import DEFAULT_STRATEGY_HOLD_BARS, BacktestConfig
from finalayze.backtest.decision_journal import DecisionJournal
from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.iteration_tracker import IterationTracker
from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner
from finalayze.backtest.monte_carlo import bootstrap_from_snapshots
from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.backtest.walk_forward import WalkForwardResult
from finalayze.core.schemas import (
    Candle,
    GateResult,
    IterationMetadata,
    IterationMetrics,
    PortfolioState,
    TradeResult,
)
from finalayze.data.fetchers.base import BaseFetcher
from finalayze.data.fetchers.caching import CachingFetcher
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.markets.instruments import build_default_registry
from finalayze.risk.kelly import RollingKelly
from finalayze.risk.regime import (
    HMMRegimeProvider,
    StaticRegimeProvider,
    VIXRegimeProvider,
    compute_moex_regime_state,
    compute_realized_vol,
)
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.cbr_calendar import CBRCalendar, CBRRateEvent
from finalayze.strategies.cbr_strategy_wrapper import CBRStrategyWrapper
from finalayze.strategies.dividend_gap import DividendEntry, DividendGapStrategy
from finalayze.strategies.dual_momentum import DualMomentumStrategy
from finalayze.strategies.mean_reversion import MeanReversionStrategy
from finalayze.strategies.ml_strategy import MLStrategy
from finalayze.strategies.momentum import MomentumStrategy
from finalayze.strategies.ou_mean_reversion import OUMeanReversionStrategy
from finalayze.strategies.pairs import PairsStrategy
from finalayze.strategies.pead import EarningsSurprise, PEADStrategy
from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy

_PRESETS_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "finalayze" / "strategies" / "presets"
)

# ── Symbol universe ────────────────────────────────────────────────────────────
UNIVERSE: dict[str, list[str]] = {
    "us_tech": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSM",
        "AVGO",
        "ADBE",
        "CRM",
        "INTC",
        "AMD",
        "CSCO",
        "ORCL",
        "QCOM",
        "TXN",
        "ASML",
        "AMAT",
        "MU",
        "NOW",
    ],
    "us_broad": [
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "JNJ",
        "PG",
        "KO",
        "WMT",
        "XOM",
        "CVX",
        "PEP",
        "COST",
        "MCD",
        "NKE",
        "DIS",
        "HD",
        "LOW",
        "TGT",
        "SBUX",
        "CL",
    ],
    "us_finance": [
        "JPM",
        "BAC",
        "GS",
        "MS",
        "V",
        "MA",
        "BRK-B",
        "C",
        "SCHW",
        "AXP",
        "USB",
        "PNC",
        "TFC",
        "BLK",
        "SPGI",
    ],
    "us_healthcare": [
        "UNH",
        "LLY",
        "PFE",
        "ABBV",
        "MRK",
        "TMO",
        "ABT",
        "AMGN",
        "JNJ",
        "BMY",
        "GILD",
        "VRTX",
        "ISRG",
        "MDT",
        "ZTS",
    ],
    "us_industrial": [
        "CAT",
        "DE",
        "HON",
        "UNP",
        "BA",
        "GE",
        "RTX",
        "LMT",
        "MMM",
        "ETN",
        "ITW",
        "EMR",
        "PH",
        "WM",
        "RSG",
    ],
    "ru_blue_chips": [
        "SBER",
        "GAZP",
        "LKOH",
        "GMKN",
        "YNDX",
        "VTBR",
        "SBERP",
        "MGNT",
        "POLY",
        "ALRS",
    ],
    "ru_energy": [
        "ROSN",
        "LKOH",
        "NVTK",
        "TATN",
        "GAZP",
        "SNGS",
        "TRNFP",
        "IRAO",
    ],
}


class _MoexYFinanceFetcher(BaseFetcher):
    """YFinance wrapper that appends .ME suffix for MOEX tickers."""

    def __init__(self) -> None:
        self._inner = YFinanceFetcher(market_id="moex")

    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Candle]:
        """Fetch candles using yfinance with .ME suffix, then fix symbol back."""
        yf_symbol = f"{symbol}.ME"
        candles = self._inner.fetch_candles(yf_symbol, start, end, timeframe)
        # Replace yfinance symbol (SBER.ME) with original MOEX symbol (SBER)
        return [
            Candle(
                symbol=symbol,
                market_id="moex",
                timeframe=c.timeframe,
                timestamp=c.timestamp,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
                source=c.source,
            )
            for c in candles
        ]


def _make_moex_fetcher() -> BaseFetcher:
    """Create a MOEX data fetcher: TinkoffFetcher if token available, else yfinance .ME."""
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    if token:
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415

        registry = build_default_registry()
        return TinkoffFetcher(token=token, registry=registry, sandbox=False)
    return _MoexYFinanceFetcher()


def _load_preset(segment: str) -> dict[str, Any]:
    """Load YAML preset for a segment, returning empty dict on failure."""
    preset_path = _PRESETS_DIR / f"{segment}.yaml"
    if not preset_path.exists():
        return {}
    with preset_path.open() as f:
        return yaml.safe_load(f) or {}


def _setup_pairs_strategy(
    segment: str,
    fetcher: BaseFetcher,
    start: datetime,
    end: datetime,
) -> PairsStrategy | None:
    """Create a PairsStrategy with pre-loaded peer candles, or None."""
    preset = _load_preset(segment)
    pairs_cfg = preset.get("strategies", {}).get("pairs", {})
    if not pairs_cfg.get("enabled", False):
        return None

    raw_pairs: list[list[str]] = pairs_cfg.get("params", {}).get("pairs", [])
    if not raw_pairs:
        return None

    peer_symbols: set[str] = set()
    for pair in raw_pairs:
        for sym in pair:
            peer_symbols.add(str(sym))

    strategy = PairsStrategy()
    for sym in peer_symbols:
        try:
            candles = fetcher.fetch_candles(sym, start, end)
            if candles:
                strategy.set_peer_candles(sym, candles)
        except Exception:
            continue
    return strategy


def _setup_ml_strategy(segment: str, models_dir: Path) -> MLStrategy | None:
    """Create an MLStrategy with loaded models, or None."""
    from finalayze.ml.models.ensemble import EnsembleModel  # noqa: PLC0415

    segment_dir = models_dir / segment
    if not segment_dir.is_dir():
        return None

    xgb_path = segment_dir / "xgb.pkl"
    lgbm_path = segment_dir / "lgbm.pkl"
    lstm_path = segment_dir / "lstm.pkl"

    models = []
    lstm_model = None

    if xgb_path.exists():
        from finalayze.ml.models.xgboost_model import XGBoostModel  # noqa: PLC0415

        models.append(XGBoostModel.load_from(xgb_path))

    if lgbm_path.exists():
        from finalayze.ml.models.lightgbm_model import LightGBMModel  # noqa: PLC0415

        models.append(LightGBMModel.load_from(lgbm_path))

    if lstm_path.exists():
        from finalayze.ml.models.lstm_model import LSTMModel  # noqa: PLC0415

        lstm_model = LSTMModel(segment_id=segment)
        lstm_model.load(lstm_path)

    if not models and lstm_model is None:
        return None

    from finalayze.ml.registry import MLModelRegistry  # noqa: PLC0415

    ensemble = EnsembleModel(models=models, lstm_model=lstm_model)

    # Load calibrator if available (trained by train_models.py)
    calibrator_path = segment_dir / "calibrator.pkl"
    if calibrator_path.exists():
        import pickle  # noqa: PLC0415

        with calibrator_path.open("rb") as f:
            ensemble.calibrator = pickle.load(f)  # noqa: S301
    else:
        print(f"    [{segment}] No calibrator found, using raw ensemble probabilities")

    registry = MLModelRegistry()
    registry.register(segment, ensemble)
    return MLStrategy(registry)


def _load_event_data(event_data_dir: Path) -> dict[str, Any]:
    """Load event data JSONs from directory."""
    data: dict[str, Any] = {"dividends": {}, "earnings": {}, "cbr": []}

    # Load dividends
    div_dir = event_data_dir / "dividends"
    if div_dir.is_dir():
        for f in sorted(div_dir.glob("*.json")):
            symbol = f.stem
            with f.open() as fp:
                data["dividends"][symbol] = json.load(fp)

    # Load earnings
    earn_dir = event_data_dir / "earnings"
    if earn_dir.is_dir():
        for f in sorted(earn_dir.glob("*.json")):
            symbol = f.stem
            with f.open() as fp:
                data["earnings"][symbol] = json.load(fp)

    # Load CBR decisions
    cbr_path = event_data_dir / "cbr" / "decisions.json"
    if cbr_path.exists():
        with cbr_path.open() as fp:
            data["cbr"] = json.load(fp)

    return data


def _setup_dividend_gap_strategy(  # noqa: PLR0912
    segment: str,
    symbols: list[str],
    fetcher: BaseFetcher,
    start: datetime,
    end: datetime,
    event_data: dict[str, Any] | None = None,
) -> DividendGapStrategy | None:
    """Create DividendGapStrategy with populated dividend calendar.

    Data sources (in priority order):
    1. Tinkoff API via fetcher.fetch_dividends() — returns ex-dates with +1 bday shift
    2. Pre-built event_data JSON files (--event-data-dir)
    3. Static moex_dividends.yaml fallback for yfinance-based backtests
    """
    if not segment.startswith("ru_"):
        return None

    strategy = DividendGapStrategy()
    count = 0

    # Priority 1: Tinkoff API (fetcher has fetch_dividends)
    if hasattr(fetcher, "fetch_dividends"):
        for symbol in symbols:
            try:
                divs = fetcher.fetch_dividends(symbol, start, end)
                for div in divs:
                    strategy.add_dividend(
                        symbol,
                        DividendEntry(ex_date=div["ex_date"], amount=div["amount"]),
                    )
                    count += 1
            except Exception:
                continue

    # Priority 2: pre-built event data
    if count == 0 and event_data is not None:
        dividends = event_data.get("dividends", {})
        segment_symbols = set(symbols)
        for symbol, entries in dividends.items():
            if symbol not in segment_symbols:
                continue
            for entry in entries:
                strategy.add_dividend(
                    symbol,
                    DividendEntry(
                        ex_date=datetime.strptime(entry["ex_date"], "%Y-%m-%d").replace(tzinfo=UTC),
                        amount=float(entry["amount"]),
                    ),
                )
                count += 1

    # Priority 3: static YAML fallback
    if count == 0:
        yaml_path = _PRESETS_DIR / "moex_dividends.yaml"
        if yaml_path.exists():
            with yaml_path.open() as f:
                static_data = yaml.safe_load(f) or {}
            one_day = timedelta(days=1)
            for symbol in symbols:
                for entry in static_data.get(symbol, []):
                    ex_date_raw = entry["ex_date"]
                    if isinstance(ex_date_raw, str):
                        last_buy = datetime.strptime(ex_date_raw, "%Y-%m-%d").replace(tzinfo=UTC)
                    else:
                        last_buy = ex_date_raw
                    # Shift last_buy_date → actual ex-div date (+1 business day)
                    ex_date = last_buy + one_day
                    while ex_date.weekday() >= 5:  # noqa: PLR2004
                        ex_date += one_day
                    if start <= ex_date <= end:
                        strategy.add_dividend(
                            symbol,
                            DividendEntry(ex_date=ex_date, amount=float(entry["amount"])),
                        )
                        count += 1

    if count == 0:
        return None
    return strategy


def _setup_pead_strategy(
    segment: str,
    symbols: list[str],
    event_data: dict[str, Any],
) -> PEADStrategy | None:
    """Create PEADStrategy with loaded earnings surprises. Only for us_* segments."""
    if not segment.startswith("us_"):
        return None
    earnings = event_data.get("earnings", {})
    if not earnings:
        return None
    strategy = PEADStrategy()
    count = 0
    for symbol in symbols:
        entries = earnings.get(symbol, [])
        for entry in entries:
            if entry.get("sue_score") is None:
                continue
            strategy.add_earnings_surprise(
                EarningsSurprise(
                    symbol=symbol,
                    announcement_date=datetime.strptime(
                        entry["announcement_date"], "%Y-%m-%d"
                    ).replace(tzinfo=UTC),
                    sue_score=float(entry["sue_score"]),
                    actual_eps=float(entry.get("actual_eps", 0)),
                    expected_eps=float(entry.get("expected_eps", 0)),
                ),
            )
            count += 1
    if count == 0:
        return None
    return strategy


def _setup_cbr_strategy(
    segment: str,
    symbols: list[str],
    event_data: dict[str, Any],
) -> CBRStrategyWrapper | None:
    """Create CBRStrategyWrapper with loaded CBR rate decisions. Only for ru_* segments.

    Uses the actual segment symbols as affected_symbols so that signals are
    generated for the ETF proxies (RSX, ERUS, etc.) that the backtest uses.
    """
    if not segment.startswith("ru_"):
        return None
    cbr_events = event_data.get("cbr", [])
    if not cbr_events:
        return None
    calendar = CBRCalendar()
    for evt in cbr_events:
        calendar.add_event(
            CBRRateEvent(
                date=datetime.strptime(evt["date"], "%Y-%m-%d").replace(tzinfo=UTC).date(),
                rate_decision=float(evt["rate_decision"]),
                expected_rate=float(evt["expected_rate"]),
                surprise_bps=int(evt["surprise_bps"]),
            ),
        )
    return CBRStrategyWrapper(calendar=calendar, affected_symbols=symbols)


def _build_strategies(
    segment: str,
    fetcher: BaseFetcher,
    start: datetime,
    end: datetime,
    models_dir: Path | None,
    symbols: list[str] | None = None,
    event_data: dict[str, Any] | None = None,
) -> list[BaseStrategy]:
    """Build the full strategy list for a segment."""
    strategies: list[BaseStrategy] = [
        MomentumStrategy(),
        DualMomentumStrategy(vol_target_enabled=True),
        MeanReversionStrategy(),
        OUMeanReversionStrategy(use_mle=True),
        RSI2ConnorsStrategy(),
    ]

    pairs = _setup_pairs_strategy(segment, fetcher, start, end)
    if pairs is not None:
        strategies.append(pairs)

    if models_dir is not None:
        ml = _setup_ml_strategy(segment, models_dir)
        if ml is not None:
            strategies.append(ml)

    # Dividend gap — always try for RU segments (API > event_data > static YAML)
    div_gap = _setup_dividend_gap_strategy(
        segment, symbols or [], fetcher, start, end, event_data=event_data
    )
    if div_gap is not None:
        strategies.append(div_gap)

    # Other event-driven strategies (require event data)
    if event_data is not None:
        pead = _setup_pead_strategy(segment, symbols or [], event_data)
        if pead is not None:
            strategies.append(pead)

        cbr = _setup_cbr_strategy(segment, symbols or [], event_data)
        if cbr is not None:
            strategies.append(cbr)

    return strategies


def _build_regime_provider(  # noqa: PLR0911
    regime_type: str,
    segment: str,
    start: datetime,
    end: datetime,
) -> VIXRegimeProvider | HMMRegimeProvider | StaticRegimeProvider | None:
    """Build a RegimeProvider based on CLI flag and segment type."""
    if regime_type == "none":
        return None

    if regime_type == "hmm":
        return HMMRegimeProvider()

    # regime_type == "vix"
    if segment.startswith("ru_"):
        # For MOEX segments, compute regime from IMOEX realized volatility
        try:
            moex_fetcher = CachingFetcher(_make_moex_fetcher())
            imoex_candles = moex_fetcher.fetch_candles("IMOEX", start, end)
            if imoex_candles:
                vol = compute_realized_vol(imoex_candles)
                regime_state = compute_moex_regime_state(vol)
                print(f"    MOEX regime: {regime_state.regime.value} (vol={float(vol):.2%})")
                return StaticRegimeProvider(regime_state)
        except Exception:
            print("    Warning: failed to fetch IMOEX data, regime provider disabled")
        return None

    vix_fetcher = CachingFetcher(YFinanceFetcher(market_id="us"))
    try:
        vix_candles = vix_fetcher.fetch_candles("^VIX", start, end)
    except Exception:
        print("    Warning: failed to fetch ^VIX data, regime provider disabled")
        return None

    if not vix_candles:
        print("    Warning: no ^VIX data available, regime provider disabled")
        return None

    spy_candles = None
    try:
        spy_raw = vix_fetcher.fetch_candles("SPY", start, end)
        spy_candles = spy_raw or None
    except Exception:  # noqa: S110
        pass

    return VIXRegimeProvider(vix_candles=vix_candles, sma200_candles=spy_candles)


def _run_symbol(
    symbol: str,
    segment: str,
    candles: list[Any],
    strategies: list[BaseStrategy],
    cash: Decimal,
    output_dir: Path,
    benchmark_candles: list[Any] | None = None,
    use_evt_sizing: bool = False,
    use_copula_scaling: bool = False,
    regime_provider: VIXRegimeProvider | HMMRegimeProvider | StaticRegimeProvider | None = None,
    stop_loss_mode: str = "chandelier",
) -> tuple[list[TradeResult], list[PortfolioState], dict[str, Any] | None]:
    """Run backtest for a single symbol. Returns (trades, snapshots, summary)."""
    sym_dir = output_dir / segment / symbol.replace(".", "_")
    sym_dir.mkdir(parents=True, exist_ok=True)

    try:
        combiner = JournalingStrategyCombiner(
            strategies=strategies,
            allocation_mode="hrp",
        )
        journal = DecisionJournal(output_path=sym_dir / "decision_journal.jsonl")

        engine = BacktestEngine(
            strategy=combiner,
            config=BacktestConfig(
                initial_cash=cash,
                decision_journal=journal,
                rolling_kelly=RollingKelly(),
                use_impact_model=True,
                use_evt_sizing=use_evt_sizing,
                use_copula_scaling=use_copula_scaling,
                stop_loss_mode=stop_loss_mode,
                max_hold_bars=DEFAULT_STRATEGY_HOLD_BARS,
            ),
            regime_provider=regime_provider,
        )
        trades, snapshots = engine.run(
            symbol=symbol,
            segment_id=segment,
            candles=candles,
        )
        journal.flush()

        result = PerformanceAnalyzer().analyze(
            trades, snapshots, benchmark_candles=benchmark_candles
        )

        summary = {
            "symbol": symbol,
            "segment": segment,
            "total_candles": len(candles),
            "total_trades": len(trades),
            "metrics": result.model_dump(mode="json") if result else None,
            "journal_summary": journal.summary(),
        }

        sharpe = float(result.sharpe) if result else 0.0
        wr = float(result.win_rate) if result else 0.0
        ret = float(result.total_return) if result else 0.0
        print(
            f"    {symbol:12s} | {len(candles):4d} bars | "
            f"{len(trades):3d} trades | "
            f"Sharpe {sharpe:+7.3f} | "
            f"WR {wr:5.1%} | "
            f"Ret {ret:+7.3%}"
        )

        # Print per-strategy signal counts from engine summary
        run_summary = engine.last_run_summary
        sig_counts: dict[str, int] = run_summary.get("strategy_signals", {})  # type: ignore[assignment]
        above_thresh: int = run_summary.get("combined_above_threshold", 0)  # type: ignore[assignment]
        if sig_counts or above_thresh:
            parts = [f"{name}={count}" for name, count in sorted(sig_counts.items())]
            sig_str = " ".join(parts) if parts else "none"
            print(f"      Signals: {sig_str} | combined_above_threshold={above_thresh}")

        return trades, snapshots, summary

    except Exception:
        print(f"    {symbol:12s} | ERROR — {traceback.format_exc().splitlines()[-1]}")
        return [], [], None


def _format_comparison_table(
    current_name: str,
    baseline_name: str | None,
    metrics: IterationMetrics,
    baseline_metrics: IterationMetrics | None,
    gate_results: list[GateResult],
    verdict: str,
    git_info: dict[str, object],
) -> str:
    """Format a comparison table for terminal output."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  Iteration: {current_name}")
    if baseline_name:
        lines.append(f"  Baseline:  {baseline_name}")
    dirty_str = "dirty" if git_info.get("git_dirty") else "clean"
    sha_short = str(git_info.get("git_sha", ""))[:7]
    lines.append(f"  Git:       {sha_short} ({dirty_str})")
    lines.append(f"  Verdict:   {verdict}")
    lines.append("=" * 72)
    lines.append("")

    header = f"  {'Metric':<26} {'Current':>10}"
    if baseline_metrics:
        header += f" {'Baseline':>10} {'Delta':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header.strip()) + 2))

    bm = baseline_metrics
    _add_row(lines, "WF Sharpe", metrics.wf_sharpe, bm)
    _add_row(lines, "Max Drawdown (%)", metrics.wf_max_drawdown, bm, "wf_max_drawdown")
    _add_row(lines, "Profit Factor", metrics.profit_factor, bm)
    _add_row(lines, "Calmar Ratio", metrics.calmar_ratio, bm)
    _add_row(lines, "Trade Count", float(metrics.trade_count), bm, "trade_count")
    _add_row(lines, "Avg Hold (bars)", metrics.avg_hold_bars, bm, "avg_hold_bars")
    lines.append("  " + "-" * (len(header.strip()) + 2))
    _add_row(lines, "Sortino", metrics.sortino_ratio, bm)
    _add_row(lines, "MC 5th-pct Sharpe", metrics.mc_5th_pct_sharpe, bm)
    _add_row(lines, "Model Disagreement", metrics.model_disagreement, bm)
    _add_row(lines, "Turnover-Adj Return", metrics.turnover_adjusted_return, bm)
    _add_row(lines, "Param Stability CV", metrics.param_stability_cv, bm)

    lines.append("")
    gate_str = "  Gates: " + "  ".join(
        f"{g.name}:{'OK' if g.passed else 'FAIL'}" for g in gate_results
    )
    lines.append(gate_str)
    lines.append("")

    # Segment PnL share
    if metrics.segment_pnl_share:
        lines.append("  Segment PnL Share:")
        for seg, share in sorted(metrics.segment_pnl_share.items(), key=lambda x: -abs(x[1])):
            lines.append(f"    {seg:<20s} {share:>7.1%}")
        lines.append("")

    # Win rate by segment
    if metrics.win_rate_by_segment:
        lines.append("  Win Rate by Segment:")
        for seg, wr in sorted(metrics.win_rate_by_segment.items()):
            lines.append(f"    {seg:<20s} {wr:>7.1%}")
        lines.append("")

    return "\n".join(lines)


def _add_row(
    lines: list[str],
    name: str,
    current_val: float,
    baseline_metrics: IterationMetrics | None,
    attr: str | None = None,
) -> None:
    """Append a formatted metric row to lines."""
    row = f"  {name:<26} {current_val:>10.4f}"
    if baseline_metrics is not None:
        field = attr or name.lower().replace(" ", "_").replace("(%)", "").strip()
        base_val = getattr(baseline_metrics, field, None)
        if base_val is not None:
            base_f = float(base_val)
            delta = current_val - base_f
            row += f" {base_f:>10.4f} {delta:>+10.4f}"
    lines.append(row)


def _run_dry(
    args: argparse.Namespace,
    tracker: IterationTracker,
    git_info: dict[str, object],
) -> None:
    """Execute dry-run mode with synthetic metrics."""
    print("\n  [DRY RUN] Generating synthetic metrics...")
    metrics = IterationMetrics(
        wf_sharpe=0.0,
        wf_max_drawdown=0.0,
        profit_factor=0.0,
        calmar_ratio=0.0,
        trade_count=0,
        avg_hold_bars=0.0,
        segment_pnl_share={},
        sortino_ratio=0.0,
        win_rate_by_segment={},
        information_ratio=None,
        mc_5th_pct_sharpe=0.0,
        model_disagreement=0.0,
        turnover_adjusted_return=0.0,
        gross_sharpe=0.0,
        net_sharpe=0.0,
        param_stability_cv=0.0,
        per_model_proba_mean={},
    )
    gate_results, verdict = tracker.evaluate_gates(metrics, baseline=None)
    print(_format_comparison_table(args.name, None, metrics, None, gate_results, verdict, git_info))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a new backtest iteration")
    parser.add_argument("--name", required=True, help="Iteration name")
    parser.add_argument("--description", required=True, help="What changed")
    parser.add_argument("--baseline", default="latest", help="Baseline name (default: latest)")
    parser.add_argument("--output", default="results/iterations/", help="Output root")
    parser.add_argument("--segments", default=None, help="Comma-separated segment IDs")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--cash", type=int, default=100_000, help="Initial cash per symbol")
    parser.add_argument("--models-dir", default=None, help="Directory with trained ML models")
    parser.add_argument(
        "--event-data-dir", default=None, help="Directory with event data JSON files"
    )
    parser.add_argument("--use-evt-sizing", action="store_true", help="Enable EVT tail risk sizing")
    parser.add_argument(
        "--use-copula-scaling", action="store_true", help="Enable copula correlation scaling"
    )
    parser.add_argument(
        "--regime-provider",
        choices=["none", "vix", "hmm"],
        default="vix",
        help="Regime provider: none, vix (default), or hmm",
    )
    parser.add_argument(
        "--stop-loss-mode",
        choices=["trailing", "chandelier"],
        default="chandelier",
        help="Stop-loss mode: trailing or chandelier (default)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_baseline(
    tracker: IterationTracker,
    baseline_arg: str,
) -> tuple[IterationMetrics | None, str | None]:
    """Load baseline metrics by name or 'latest'."""
    if baseline_arg == "latest":
        latest = tracker.load_latest()
        if latest:
            return latest.metrics, latest.name
        return None, None
    try:
        meta = tracker.load(baseline_arg)
        return meta.metrics, baseline_arg
    except FileNotFoundError:
        print(f"  Warning: baseline '{baseline_arg}' not found")
        return None, None


def main() -> None:  # noqa: PLR0912, PLR0915
    """Run a new iteration."""
    args = _parse_args()
    output_root = Path(args.output)
    tracker = IterationTracker(results_root=output_root)

    try:
        git_info = tracker.snapshot_context()
    except Exception:
        git_info = {
            "git_sha": "unknown",
            "git_describe": "unknown",
            "git_dirty": True,
        }

    all_segments = list(UNIVERSE.keys())
    segments = args.segments.split(",") if args.segments else all_segments

    if args.dry_run:
        _run_dry(args, tracker, git_info)
        return

    start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    cash = Decimal(args.cash)
    models_dir = Path(args.models_dir) if args.models_dir else None
    event_data_dir = Path(args.event_data_dir) if args.event_data_dir else None
    event_data: dict[str, Any] | None = None
    if event_data_dir is not None:
        event_data = _load_event_data(event_data_dir)
        n_div = sum(len(v) for v in event_data["dividends"].values())
        n_earn = sum(len(v) for v in event_data["earnings"].values())
        n_cbr = len(event_data["cbr"])
        print(f"  Event data: {n_div} dividends, {n_earn} earnings, {n_cbr} CBR decisions")
    use_evt_sizing = args.use_evt_sizing
    use_copula_scaling = args.use_copula_scaling

    # Load strategy configs from YAML presets
    strategy_configs: dict[str, Any] = {}
    for seg in segments:
        strategy_configs[seg] = _load_preset(seg)

    config = BacktestConfig(initial_cash=cash)
    backtest_config_dict = {
        "initial_cash": str(config.initial_cash),
        "max_positions": config.max_positions,
        "kelly_fraction": str(config.kelly_fraction),
        "atr_multiplier": str(config.atr_multiplier),
        "regime_provider": args.regime_provider,
        "stop_loss_mode": args.stop_loss_mode,
        "max_hold_bars": "per_strategy",
    }
    config_hash = tracker.compute_config_hash(backtest_config_dict, strategy_configs)

    # Cache benchmark candles per market
    benchmark_cache: dict[str, list[Any] | None] = {}

    all_trades: list[TradeResult] = []
    all_snapshots: list[PortfolioState] = []
    segment_trades: dict[str, list[TradeResult]] = {}
    all_summaries: list[dict[str, Any]] = []

    print(f"\nRunning iteration '{args.name}'")
    print(f"  Period: {args.start_date} to {args.end_date}")
    print(f"  Segments: {', '.join(segments)}")
    print(f"  Cash: ${cash:,.0f}")
    print()

    for segment in segments:
        symbols = UNIVERSE.get(segment, [])
        if not symbols:
            print(f"  Segment '{segment}' not found in universe, skipping")
            continue

        market_id = "moex" if segment.startswith("ru_") else "us"
        print(f"{'=' * 72}")
        print(f"  SEGMENT: {segment} ({len(symbols)} symbols, market={market_id})")
        print(f"{'=' * 72}")

        # Fetch benchmark
        bench_symbol = "IMOEX.ME" if segment.startswith("ru_") else "SPY"
        if bench_symbol not in benchmark_cache:
            try:
                bench_fetcher = CachingFetcher(YFinanceFetcher(market_id="us"))
                benchmark_cache[bench_symbol] = bench_fetcher.fetch_candles(
                    bench_symbol, start, end
                )
                n_bars = len(benchmark_cache[bench_symbol] or [])
                print(f"  Benchmark: {bench_symbol} ({n_bars} bars)")
            except Exception:
                benchmark_cache[bench_symbol] = None
                print(f"  Benchmark: {bench_symbol} (fetch failed)")
        bench_candles = benchmark_cache[bench_symbol]

        # Create appropriate fetcher for the market
        is_moex = segment.startswith("ru_")
        base_fetcher = _make_moex_fetcher() if is_moex else YFinanceFetcher(market_id=market_id)

        # Build strategies once per segment
        seg_fetcher = CachingFetcher(base_fetcher)
        strategies = _build_strategies(
            segment,
            seg_fetcher,
            start,
            end,
            models_dir,
            symbols=symbols,
            event_data=event_data,
        )
        strat_names = [s.name for s in strategies]
        print(f"  Strategies: {', '.join(strat_names)}")

        # Build regime provider once per segment
        regime_provider = _build_regime_provider(args.regime_provider, segment, start, end)
        if regime_provider is not None:
            print(f"  Regime provider: {type(regime_provider).__name__}")
        print()

        segment_trades[segment] = []

        for symbol in symbols:
            # Fetch candles
            try:
                fetcher = CachingFetcher(base_fetcher)
                candles = fetcher.fetch_candles(symbol, start, end)
                if not candles:
                    print(f"    {symbol:12s} | no data")
                    continue
            except Exception:
                print(f"    {symbol:12s} | fetch failed")
                continue

            iter_dir = output_root / args.name
            trades, snapshots, summary = _run_symbol(
                symbol=symbol,
                segment=segment,
                candles=candles,
                strategies=strategies,
                cash=cash,
                output_dir=iter_dir,
                benchmark_candles=bench_candles,
                use_evt_sizing=use_evt_sizing,
                use_copula_scaling=use_copula_scaling,
                regime_provider=regime_provider,
                stop_loss_mode=args.stop_loss_mode,
            )

            all_trades.extend(trades)
            segment_trades[segment].extend(trades)
            if snapshots:
                all_snapshots.extend(snapshots)
            if summary:
                all_summaries.append(summary)

        print()

    if not all_trades:
        print("\n  No trades generated across all segments.")
        print("  Saving iteration with zero metrics for tracking purposes.\n")

    # Save consolidated summary
    iter_dir = output_root / args.name
    iter_dir.mkdir(parents=True, exist_ok=True)
    summary_path = iter_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2, default=str))

    # Compute iteration metrics
    if all_snapshots and all_trades:
        mc_result = bootstrap_from_snapshots(all_snapshots, n_simulations=1000, seed=42)
    else:
        # Create a minimal MC result for zero-trade iterations
        from finalayze.backtest.monte_carlo import BootstrapCI, BootstrapResult  # noqa: PLC0415

        zero_ci = BootstrapCI(point_estimate=0.0, lower=0.0, upper=0.0, confidence_level=0.95)
        mc_result = BootstrapResult(
            total_return=zero_ci,
            sharpe_ratio=zero_ci,
            max_drawdown=zero_ci,
            win_rate=zero_ci,
            profit_factor=zero_ci,
            n_simulations=0,
            n_trades=0,
        )

    # Build a synthetic WalkForwardResult from the direct backtest run
    # (full walk-forward takes much longer; this gives immediate baseline)
    from finalayze.backtest.performance import PerformanceAnalyzer as PA  # noqa: PLC0415, N817

    full_result = PA().analyze(all_trades, all_snapshots) if all_trades else None
    oos_sharpe = float(full_result.sharpe) if full_result else 0.0
    oos_dd = float(full_result.max_drawdown) if full_result else 0.0

    wf_result = WalkForwardResult(
        oos_sharpe=oos_sharpe,
        oos_max_drawdown_pct=oos_dd * 100,  # convert to percent
        oos_snapshots=all_snapshots,
        per_fold_sharpes=[oos_sharpe] if all_trades else [],
        per_fold_trade_counts=[len(all_trades)] if all_trades else [],
    )

    metrics = tracker.compute_metrics(
        wf_result=wf_result,
        trades=all_trades,
        snapshots=all_snapshots,
        segment_trades=segment_trades,
        mc_result=mc_result,
    )

    baseline_metrics, baseline_name = _load_baseline(tracker, args.baseline)
    gate_results, verdict = tracker.evaluate_gates(metrics, baseline=baseline_metrics)

    metadata = IterationMetadata(
        name=args.name,
        description=args.description,
        created_at=datetime.now(UTC),
        git_describe=str(git_info["git_describe"]),
        git_sha=str(git_info["git_sha"]),
        git_dirty=bool(git_info["git_dirty"]),
        config_hash=config_hash,
        strategy_configs=strategy_configs,
        backtest_config=backtest_config_dict,
        metrics=metrics,
        gate_results=gate_results,
        verdict=verdict,
    )

    result_path = tracker.save(metadata)
    print(f"\n  Saved to: {result_path}")
    print(
        _format_comparison_table(
            args.name,
            baseline_name,
            metrics,
            baseline_metrics,
            gate_results,
            verdict,
            git_info,
        )
    )


if __name__ == "__main__":
    main()
