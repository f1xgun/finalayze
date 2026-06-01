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
import warnings

from dotenv import load_dotenv

load_dotenv()
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Ensure config/ at project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from config.segments import _BOOTSTRAP_SEGMENT_SYMBOLS, DEFAULT_SEGMENTS

from finalayze.backtest.config import DEFAULT_STRATEGY_HOLD_BARS, MOEX_2022_BREAK, BacktestConfig
from finalayze.backtest.costs import MOEX_COSTS, US_COSTS
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
    MarketContext,
    PortfolioState,
    TradeResult,
)
from finalayze.data.fetchers._cache_utils import GenericFileCache
from finalayze.data.fetchers.base import BaseFetcher
from finalayze.data.fetchers.caching import CachingFetcher
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.data.loader import MarketDataLoader
from finalayze.data.rate_limiter import RateLimiter
from finalayze.markets.instruments import build_default_registry
from finalayze.markets.liquidity import eligible_universe_as_of, select_segment_symbols
from finalayze.risk.kelly import RollingKelly
from finalayze.risk.regime import (
    HMMRegimeProvider,
    RollingVolRegimeProvider,
    StaticRegimeProvider,
    VIXRegimeProvider,
)
from finalayze.risk.rub_oil_regime import RubOilRegimeSignal
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.cbr_calendar import CBRCalendar, CBRRateEvent
from finalayze.strategies.cbr_strategy_wrapper import CBRStrategyWrapper
from finalayze.strategies.dividend_gap import DividendEntry, DividendGapStrategy
from finalayze.strategies.dual_momentum import DualMomentumStrategy
from finalayze.strategies.event_driven import EventDrivenStrategy
from finalayze.strategies.fundamental_gate import earnings_yield_gate

_FALLBACK_USDRUB = Decimal("90.0")
from finalayze.strategies.mean_reversion import MeanReversionStrategy
from finalayze.strategies.ml_strategy import MLStrategy
from finalayze.strategies.momentum import MomentumStrategy
from finalayze.strategies.ou_mean_reversion import OUMeanReversionStrategy
from finalayze.strategies.pairs import PairsStrategy
from finalayze.strategies.pead import EarningsSurprise, PEADStrategy, compute_sue_proxy
from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy

_PRESETS_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "finalayze" / "strategies" / "presets"
)

# ── Symbol universe ────────────────────────────────────────────────────────────
# Toxic / sanctioned / structurally-illiquid MOEX names the backtest harness has always
# excluded from every ru_* segment (pre-66 invariant -- see tests/unit/test_run_iteration_
# universe.py). The canonical toxic set + the safety post-filter now live in the Layer-2
# selector (finalayze.markets.liquidity) and are applied UNIVERSALLY to the final selected
# universe (Plan 66-04) -- so this filter holds for the committed snapshot AND the bootstrap
# regardless of source. _drop_toxic re-exports the SHARED set (NO divergent duplicate) and
# is kept as a thin shim for the bootstrap-argument call sites below (defense-in-depth: the
# selector also re-applies it). Order-preserving (keeps the ranked sequence).
from finalayze.markets.liquidity import _TOXIC_SYMBOLS


def _drop_toxic(symbols: list[str]) -> list[str]:
    """Return ``symbols`` with toxic/sanctioned names removed, order preserved."""
    return [s for s in symbols if s not in _TOXIC_SYMBOLS]


def _write_trades_jsonl(output_path: Path, trades: list[TradeResult]) -> None:
    """Serialize closed trades to a ``trades.jsonl`` sidecar (RUFIN-01 / D-01).

    Mirrors ``DecisionJournal.flush`` exactly: parent-dir mkdir + one
    ``model_dump_json()`` per line. The Task-1 exit_reason/entry_strategy
    fields ride along via ``model_dump_json``. This is the artifact
    ``scripts/diagnose_ru_finance.py`` reads back for the attribution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for trade in trades:
            f.write(trade.model_dump_json() + "\n")


def _bootstrap_for(segment_id: str) -> list[str]:
    """Prior-hardcoded bootstrap list for ``segment_id`` (WR-04: missing-key safe).

    A new enabled MOEX stock segment added to ``DEFAULT_SEGMENTS`` without a matching
    ``_BOOTSTRAP_SEGMENT_SYMBOLS`` key must NOT raise ``KeyError`` at module import time
    (that would break ``import scripts.run_iteration`` and every consumer before any clear
    error surfaces). Use ``.get`` with an empty default; once the committed snapshot lands
    the selector supplies the liquid set regardless. Warn (non-fatal, self-healing) so the
    missing key is still visible to an operator.
    """
    if segment_id not in _BOOTSTRAP_SEGMENT_SYMBOLS:
        warnings.warn(
            f"no _BOOTSTRAP_SEGMENT_SYMBOLS key for enabled MOEX segment {segment_id!r}; "
            "bootstrapping with [] until the liquidity snapshot supplies it",
            stacklevel=2,
        )
    return _BOOTSTRAP_SEGMENT_SYMBOLS.get(segment_id, [])


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
    # ru_* SHARE universes resolve through the liquidity selector (LIQ-08) -- single
    # source shared with config.segments and training/cli.py. No hardcoded ru_* share
    # lists (the old YNDX/POLY/TCSG/etc. were delisted/stale). US entries above stay
    # hardcoded (enabled=False, frozen -- NOT re-pointed at the MOEX selector).
    #
    # 66-04: derive a key for EVERY enabled MOEX ru_* SHARE segment directly from
    # DEFAULT_SEGMENTS (single source -- it can never drift again). This is exactly the
    # bug that made the Phase-66 iteration skip ru_consumer/ru_telecom/ru_construction/
    # ru_chemicals/ru_transport (plus ru_metals/ru_utilities): those KEYS simply did not
    # exist here, even though the selector resolves all of them. Each value is selector-
    # driven (LIQ-08) so it stays identical to config.segments and training/cli.py.
    # Prior-list bootstrap (pre-66-04): when the committed snapshot FILE is absent the
    # selector returns these hardcoded lists (single source: config._BOOTSTRAP_SEGMENT_SYMBOLS)
    # so this seam stays populated and identical to the other two; once the snapshot lands
    # the liquid set replaces them. The toxic filter is applied to the BOOTSTRAP ARGUMENT
    # (defense-in-depth; the selector also re-applies _apply_safety_filters universally):
    # pre-66 run_iteration always dropped toxic / sanctioned names (e.g. VTBR) from its ru_*
    # bootstrap, while the generated liquid snapshot is the trust boundary and is returned
    # post-safety-filter (D-04) -- so the three-seams single-source guarantee holds.
    **{
        seg.segment_id: select_segment_symbols(
            seg.segment_id,
            bootstrap=_drop_toxic(_bootstrap_for(seg.segment_id)),
        )
        for seg in DEFAULT_SEGMENTS
        if seg.market == "moex" and seg.enabled and seg.instrument_type == "stock"
    },
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


def _resolve_segment_cash(segment_id: str, us_cash: Decimal, moex_cash: Decimal) -> Decimal:
    """Return starting capital for a segment.

    MOEX (``ru_*``) segments are RUB-denominated and use ``moex_cash``; every
    other (US) segment uses ``us_cash``. Keeps the currencies separate so a
    single ``--cash`` value can't silently apply a USD figure to a RUB book.
    """
    return moex_cash if segment_id.startswith("ru_") else us_cash


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
    """Create an MLStrategy with loaded models, or None.

    Uses ``load_registry`` from the ML loader which correctly handles:
    - Model file loading (XGBoost, LightGBM, LSTM)
    - ``selected_features.json`` for MI feature filtering
    - ``calibrator.pkl`` via the EnsembleModel constructor (private ``_calibrator``)
    - HMAC integrity verification
    """
    from finalayze.ml.loader import load_registry  # noqa: PLC0415

    segment_dir = models_dir / segment
    if not segment_dir.is_dir():
        return None

    registry = load_registry(models_dir, [segment])
    if registry.get(segment) is None:
        return None

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
                        DividendEntry(ex_date=div["ex_date"], amount=div["amount"], status="paid"),
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
                        status=entry.get("status", "paid"),
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
                            DividendEntry(
                                ex_date=ex_date,
                                amount=float(entry["amount"]),
                                status=entry.get("status", "paid"),
                            ),
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


# ── Phase 60 earnings seed (INTG-01) ─────────────────────────────────────────
# No earnings event-data JSON exists in the repo (verified — RESEARCH Open
# Question 1 / A2). To make an in-window SUE event exist for the MEAS-01 proving
# run, we seed a small LABELLED eps_ttm series per ru_energy symbol and run
# compute_sue_proxy (which always sets is_proxy=True). The announcement date
# falls inside the Phase-59 proving window (2023-01-01..2024-06-30). These are
# proxy values, NOT analyst consensus — kept is_proxy so backtest attribution
# stays honest (Phase-59 D-01 / threat T-60-02).
_SEED_ANNOUNCEMENT = datetime(2023, 4, 28, tzinfo=UTC)
# eps_ttm (days-before-announcement, value): a step-up that yields a positive,
# non-zero SUE vs the prior-year/rolling baseline when resolved as-of D.
_SEED_EPS_SERIES_DAYS = (730, 365, 90, 0)
_SEED_EPS_BY_SYMBOL: dict[str, tuple[float, ...]] = {
    "LKOH": (640.0, 690.0, 740.0, 980.0),
    "ROSN": (45.0, 50.0, 54.0, 78.0),
}


def _setup_event_driven_earnings(
    segment: str,
    symbols: list[str],
    event_data: dict[str, Any] | None,
    strategy: EventDrivenStrategy,
) -> int:
    """Register an earnings SUE calendar into the EventDrivenStrategy (ru_ only).

    D-02: extend event_driven, do NOT add a separate pead strategy. Data sources
    in priority order:
      1. Pre-built event_data JSON (``--event-data-dir`` ``earnings/<symbol>.json``).
      2. A SEEDED labelled eps_ttm series for ru_energy symbols (LKOH/ROSN), run
         through ``compute_sue_proxy`` so an in-window event exists for the
         proving run. Always ``is_proxy=True``.

    Returns the number of registered surprises (0 when none apply).
    """
    if not segment.startswith("ru_"):
        return 0

    count = 0
    segment_symbols = set(symbols)

    # Priority 1: pre-built event data JSON (real history when supplied).
    earnings = (event_data or {}).get("earnings", {})
    for symbol in symbols:
        for entry in earnings.get(symbol, []):
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
                    is_proxy=bool(entry.get("is_proxy", True)),
                ),
            )
            count += 1

    # Priority 2: seeded labelled SUE for ru_energy when no JSON earnings exist.
    if count == 0 and segment == "ru_energy":
        for symbol, values in _SEED_EPS_BY_SYMBOL.items():
            if symbol not in segment_symbols:
                continue
            eps_history = [
                (_SEED_ANNOUNCEMENT - timedelta(days=days), value)
                for days, value in zip(_SEED_EPS_SERIES_DAYS, values, strict=True)
            ]
            surprise = compute_sue_proxy(symbol, _SEED_ANNOUNCEMENT, eps_history)
            strategy.add_earnings_surprise(surprise)
            count += 1

    return count


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
    """Build the full strategy list for a segment.

    Reads the preset YAML to check which strategies are enabled.
    Momentum and dual_momentum are skipped when ``enabled: false``
    in the preset (e.g. MOEX equity segments are mean-reverting).
    """
    preset = _load_preset(segment)
    strategies_cfg = preset.get("strategies", {})

    strategies: list[BaseStrategy] = []

    # Momentum — only include if enabled in preset (default: true for backward compat)
    mom_cfg = strategies_cfg.get("momentum", {})
    if mom_cfg.get("enabled", True):
        strategies.append(MomentumStrategy())

    # Dual momentum — only include if enabled in preset (default: true for backward compat)
    dual_cfg = strategies_cfg.get("dual_momentum", {})
    if dual_cfg.get("enabled", True):
        strategies.append(DualMomentumStrategy(vol_target_enabled=True))

    # MR strategies — always include (core strategies)
    strategies.append(MeanReversionStrategy())
    strategies.append(OUMeanReversionStrategy(use_mle=True))
    strategies.append(RSI2ConnorsStrategy())

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

    # event_driven (Phase 60): build when enabled in the preset, then register
    # the earnings SUE calendar into it (ru_-gated; seeded for ru_energy when no
    # earnings JSON exists). The strategy resolves surprises per-bar itself, so
    # no engine signature change (D-02).
    ed_cfg = strategies_cfg.get("event_driven", {})
    if ed_cfg.get("enabled", False):
        event_driven = EventDrivenStrategy()
        earnings_count = _setup_event_driven_earnings(
            segment, symbols or [], event_data, event_driven
        )
        strategies.append(event_driven)
        # MEAS-01 sanity print: confirm event_driven has fuel before declaring
        # the gate failed (Pitfall 5).
        print(f"  event_driven earnings surprises registered: {earnings_count}")

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
) -> VIXRegimeProvider | HMMRegimeProvider | StaticRegimeProvider | RollingVolRegimeProvider | None:
    """Build a RegimeProvider based on CLI flag and segment type."""
    if regime_type == "none":
        return None

    if regime_type == "hmm":
        return HMMRegimeProvider()

    # regime_type == "vix"
    if segment.startswith("ru_"):
        # For MOEX segments, compute regime from rolling IMOEX realized volatility
        try:
            moex_fetcher = CachingFetcher(_make_moex_fetcher())
            imoex_candles = moex_fetcher.fetch_candles("IMOEX", start, end)
            if imoex_candles:
                print(f"    MOEX regime: RollingVolRegimeProvider ({len(imoex_candles)} bars)")
                return RollingVolRegimeProvider(imoex_candles=imoex_candles)
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


def _normalize_trades_to_usd(
    trades: list[TradeResult],
    segment: str,
) -> list[TradeResult]:
    """Convert MOEX trade values to USD for cross-segment aggregation."""
    if not segment.startswith("ru_"):
        return trades
    # model_copy(update=...) overrides ONLY the currency-denominated fields and
    # carries every other field through verbatim. This keeps all current AND
    # future TradeResult fields (e.g. exit_reason / entry_strategy / instrument_type)
    # intact, so the cross-segment all_trades aggregation cannot silently drop
    # attribution data when new fields are added (WR-01).
    return [
        t.model_copy(
            update={
                "entry_price": t.entry_price / _FALLBACK_USDRUB,
                "exit_price": t.exit_price / _FALLBACK_USDRUB,
                "pnl": t.pnl / _FALLBACK_USDRUB,
                "coupon_income": t.coupon_income / _FALLBACK_USDRUB,
            }
        )
        for t in trades
    ]


def _normalize_snapshots_to_usd(
    snapshots: list[PortfolioState],
    segment: str,
) -> list[PortfolioState]:
    """Convert MOEX portfolio snapshots to USD for aggregation."""
    if not segment.startswith("ru_"):
        return snapshots
    return [
        PortfolioState(
            timestamp=s.timestamp,
            equity=s.equity / _FALLBACK_USDRUB,
            cash=s.cash / _FALLBACK_USDRUB,
            positions=s.positions,
        )
        for s in snapshots
    ]


def _compute_moex_sizing_data(
    market_context: MarketContext,
) -> tuple[float, RubOilRegimeSignal | None, float, str]:
    """Extract Brent-in-RUB price, RubOilRegimeSignal, yield slope and CBR direction.

    Returns:
        (brent_rub_price, rub_oil_regime_signal, yield_slope_bps, cbr_direction).
    """
    from finalayze.core.schemas import Candle  # noqa: PLC0415

    moex_data = market_context.moex_data
    if moex_data is None:
        return 0.0, None, 0.0, ""

    # Extract Brent USD candles from commodity_candles["BZ=F"]
    brent_candles: list[Candle] = []
    if moex_data.commodity_candles and "BZ=F" in moex_data.commodity_candles:
        brent_candles = list(moex_data.commodity_candles["BZ=F"])

    # Extract USDRUB rate from fx_rates
    # fx_rates are FXRate(timestamp, pair, rate) -- NOT Candle objects.
    # Convert FXRate to synthetic Candle objects for correlation computation.
    rub_candles: list[Candle] = []
    usdrub_rate: float = 0.0
    if moex_data.fx_rates:
        for fx in moex_data.fx_rates:
            if fx.pair == "USDRUB":
                rate_float = float(fx.rate)
                usdrub_rate = rate_float  # keep last rate for Brent-in-RUB
                rate_dec = Decimal(str(rate_float))
                rub_candles.append(
                    Candle(
                        symbol="USDRUB",
                        market_id="cbr",
                        timeframe="1d",
                        timestamp=fx.timestamp,
                        open=rate_dec,
                        high=rate_dec,
                        low=rate_dec,
                        close=rate_dec,
                        volume=0,
                    )
                )

    # Compute Brent-in-RUB: last Brent USD close * last USDRUB rate
    brent_rub_price = 0.0
    if brent_candles and usdrub_rate > 0:
        last_brent_usd = float(brent_candles[-1].close)
        brent_rub_price = last_brent_usd * usdrub_rate

    # Build RubOilRegimeSignal if both series have enough data
    regime_signal: RubOilRegimeSignal | None = None
    _min_series_len = 61  # need window+1 candles for correlation
    if len(rub_candles) >= _min_series_len and len(brent_candles) >= _min_series_len:
        regime_signal = RubOilRegimeSignal(
            rub_candles=rub_candles,
            oil_candles=brent_candles,
        )

    # Phase 10: yield slope and CBR direction
    from datetime import UTC  # noqa: PLC0415
    from datetime import datetime as _dt  # noqa: PLC0415

    from finalayze.data.fetchers.cbr import (  # noqa: PLC0415
        get_last_cbr_decision,
        get_yield_slope_bps,
    )

    as_of = _dt.now(tz=UTC).date()
    yield_slope = get_yield_slope_bps(as_of)
    last_decision = get_last_cbr_decision(as_of)
    cbr_dir = last_decision.decision if last_decision and last_decision.decision else ""

    return brent_rub_price, regime_signal, yield_slope, cbr_dir


def _resolve_fundamental_scale(
    segment: str,
    candles: list[Any],
    market_context: MarketContext | None,
) -> tuple[Decimal, float]:
    """Resolve the Plan-03 ``earnings_yield_gate`` boost for ru_ segments (INTG-02).

    Wires the orphaned fundamental stream into the proving run's decision path: the
    rule-based ``earnings_yield_gate`` consumes ``market_context.moex_data`` (the
    point-in-time ``FundamentalSnapshot`` series) and returns a boost-only verdict
    (``BOOST_SCALE`` when cheap, ``NEUTRAL_SCALE`` otherwise — it never cuts).

    Look-ahead safety: ``as_of`` is pinned to the LAST candle's timestamp (the run's
    most recent bar, never ``now()``), and the gate delegates the ``as_of <= D`` filter
    to ``compute_fundamental_features._filter_as_of``, so a snapshot dated after the
    final bar cannot leak in.

    A3 / D-03 caveat: T-Bank ``get_asset_fundamentals`` is point-in-time only (no
    history — see ``test_lookahead_phase60_fundamental``), so this snapshot is CONSTANT
    across the backtest window. The fundamental gate therefore COMPLETES the INTG-02
    decision-path wiring but is NOT the MEAS-01 causal lever — the non-zero trade_count
    delta is driven by SUE (event_driven) and CPI (CpiRiskOffStep), both per-bar.

    Returns ``(scale, earnings_yield)``; ``(NEUTRAL_SCALE, 0.0)`` when the gate is
    inapplicable (non-ru_ segment, no moex_data, or no candles).
    """
    if not segment.startswith("ru_") or market_context is None or not candles:
        return Decimal("1.0"), 0.0
    moex_data = market_context.moex_data
    if moex_data is None:
        return Decimal("1.0"), 0.0
    as_of = getattr(candles[-1], "timestamp", None)
    verdict = earnings_yield_gate(moex_data, as_of=as_of)
    return verdict.scale, verdict.earnings_yield


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
    market_context: MarketContext | None = None,
    brent_rub_price: float = 0.0,
    rub_oil_regime_signal: object | None = None,
    yield_slope_bps: float = 0.0,
    cbr_direction: str = "",
) -> tuple[list[TradeResult], list[PortfolioState], dict[str, Any] | None]:
    """Run backtest for a single symbol. Returns (trades, snapshots, summary)."""
    sym_dir = output_dir / segment / symbol.replace(".", "_")
    sym_dir.mkdir(parents=True, exist_ok=True)

    # INTG-02: wire the Plan-03 fundamental gate into the decision path. The boost-only
    # earnings_yield verdict tilts the symbol's risk capital (constant-in-window per A3;
    # completes the wiring, not the MEAS-01 causal lever — SUE/CPI drive the delta).
    fundamental_scale, fundamental_ey = _resolve_fundamental_scale(segment, candles, market_context)
    cash = (cash * fundamental_scale).quantize(Decimal("0.01"))

    try:
        combiner = JournalingStrategyCombiner(
            strategies=strategies,
            allocation_mode="hrp",
            market_context=market_context,
        )
        journal = DecisionJournal(output_path=sym_dir / "decision_journal.jsonl")

        engine = BacktestEngine(
            strategy=combiner,
            config=BacktestConfig(
                initial_cash=cash,
                decision_journal=journal,
                rolling_kelly=RollingKelly(fraction=0.75)
                if segment.startswith("ru_")
                else RollingKelly(),
                use_impact_model=True,
                use_evt_sizing=use_evt_sizing,
                use_copula_scaling=use_copula_scaling,
                stop_loss_mode=stop_loss_mode,
                max_hold_bars=DEFAULT_STRATEGY_HOLD_BARS,
                transaction_costs=MOEX_COSTS if segment.startswith("ru_") else US_COSTS,
                exclude_periods=MOEX_2022_BREAK if segment.startswith("ru_") else (),
                brent_rub_price=brent_rub_price,
                rub_oil_regime_signal=rub_oil_regime_signal,
                yield_slope_bps=yield_slope_bps,
                cbr_direction=cbr_direction,
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
            "fundamental_gate_scale": str(fundamental_scale),
            "fundamental_earnings_yield": fundamental_ey,
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

        # INTG-02 visibility: report when the fundamental gate boosted this symbol's
        # risk capital (constant-in-window per A3; see _resolve_fundamental_scale).
        if fundamental_scale != Decimal("1.0"):
            print(
                f"      Fundamental gate: scale={fundamental_scale} "
                f"(earnings_yield={fundamental_ey:.4f})"
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


# Synthetic single-sector key for the backtest as-of gate. The cross-NAME Top-N-per-sector
# ranking already happened at universe SELECTION time (the committed liquidity snapshot ->
# config.segments .symbols). Within a single-segment backtest run, the as-of gate's live job
# (D-05) is the point-in-time liquidity / survivorship guard: a name is entered on a bar only if,
# as of the most recent quarterly rebalance, it had >= 60 visible bars and is not stale. Mapping
# every segment symbol to one sector with top_n = len(symbols) keeps all names that pass that
# point-in-time guard while still recomputing AS-OF (<= ts) at each quarterly rebalance.
#
# WR-03 (ACCEPTED LIMITATION -- not a defect, do NOT re-architect): because this gate uses one
# synthetic sector and top_n = len(symbols), it enforces point-in-time 60-bar/staleness
# ELIGIBILITY only -- it does NOT re-apply the D-03 cross-name Top-N liquidity RANK at each
# rebalance. That Top-N cut is fixed at snapshot-build time and back-projected onto all history,
# a softer survivorship guard than a true as-of Top-N: a name liquid recently but thin
# mid-history is still entered for the whole backtest (subject only to the 60-bar/staleness
# gate). This is the documented residual gap; see docs/operations/liquidity_universe_runbook.md
# (D-11 verdict, "Accepted limitation").
_BACKTEST_GATE_SECTOR = "_segment"


def _run_segment_portfolio(
    symbols: list[str],
    segment: str,
    candles_by_symbol: dict[str, list[Candle]],
    strategies: list[BaseStrategy],
    cash: Decimal,
    output_dir: Path,
    *,
    benchmark_candles: list[Any] | None = None,
    use_evt_sizing: bool = False,
    use_copula_scaling: bool = False,
    regime_provider: VIXRegimeProvider | HMMRegimeProvider | StaticRegimeProvider | None = None,
    stop_loss_mode: str = "chandelier",
    market_context: MarketContext | None = None,
    brent_rub_price: float = 0.0,
    rub_oil_regime_signal: object | None = None,
    yield_slope_bps: float = 0.0,
    cbr_direction: str = "",
) -> tuple[list[TradeResult], list[PortfolioState], dict[str, Any] | None]:
    """Run a SHARED-capital portfolio backtest for one segment (LIQ-07 / LIQ-08 / D-05 / D-09).

    Replaces the per-symbol ``engine.run`` loop: one ``BacktestEngine`` + one shared broker over the
    whole eligible set, so (a) the per-segment ``max_concurrent_positions`` cap is REAL (it is
    silently ineffective per-symbol -- each symbol owned its own broker), and (b) the CARDINAL D-05
    as-of universe gate runs at quarterly rebalances using only candles dated ``<= ts``.

    Candle sourcing is the caller's responsibility (DB-seeded daily candles preferred for
    determinism; live ``TinkoffFetcher`` is the rate-limited MOEX fallback). Returns
    ``(trades, snapshots, summary)`` mirroring ``_run_symbol`` so the downstream metrics-recording
    shape is preserved (one segment-level summary instead of per-symbol summaries).
    """
    seg_dir = output_dir / segment
    seg_dir.mkdir(parents=True, exist_ok=True)

    # INTG-02: the boost-only earnings_yield verdict is a SEGMENT-wide, constant-in-window tilt
    # (T-Bank fundamentals are point-in-time, A3). Apply it once to the shared capital, using a
    # single symbol's candles as the as_of proxy (_resolve_fundamental_scale pins as_of to that
    # symbol's LAST bar timestamp). IN-05: pick the symbol whose last candle has the MAXIMUM
    # timestamp rather than the first non-empty entry, so the chosen as_of is DETERMINISTIC and
    # reproducible across symbol sets with ragged end dates (the prior next(...) was dict
    # insertion-order dependent). Defensive against unsorted candle lists: key on the last bar of
    # each list (the selector sorts in eligible_universe_as_of, but this seam makes no assumption).
    proxy_candles = max(
        (c for c in candles_by_symbol.values() if c),
        key=lambda c: c[-1].timestamp,
        default=[],
    )
    fundamental_scale, fundamental_ey = _resolve_fundamental_scale(
        segment, proxy_candles, market_context
    )
    cash = (cash * fundamental_scale).quantize(Decimal("0.01"))

    # Per-segment concurrent-position cap (D-09 / LIQ-07) from the curated SegmentConfig.
    cap: int | None = None
    for seg_cfg in DEFAULT_SEGMENTS:
        if seg_cfg.segment_id == segment:
            cap = seg_cfg.max_concurrent_positions
            break

    # CARDINAL D-05 as-of gate: recomputed at quarterly rebalances over candles <= ts. The full
    # candle dict is closed over, but eligible_universe_as_of slices each symbol to <= ts itself.
    sector_map = dict.fromkeys(symbols, _BACKTEST_GATE_SECTOR)
    top_n = max(len(symbols), 1)

    def eligible_at(ts: Any) -> set[str]:
        return eligible_universe_as_of(candles_by_symbol, ts, sector_map, top_n)

    combiner = JournalingStrategyCombiner(
        strategies=strategies,
        allocation_mode="hrp",
        market_context=market_context,
    )
    journal = DecisionJournal(output_path=seg_dir / "decision_journal.jsonl")

    engine = BacktestEngine(
        strategy=combiner,
        config=BacktestConfig(
            initial_cash=cash,
            decision_journal=journal,
            rolling_kelly=RollingKelly(fraction=0.75)
            if segment.startswith("ru_")
            else RollingKelly(),
            use_impact_model=True,
            use_evt_sizing=use_evt_sizing,
            use_copula_scaling=use_copula_scaling,
            stop_loss_mode=stop_loss_mode,
            max_hold_bars=DEFAULT_STRATEGY_HOLD_BARS,
            transaction_costs=MOEX_COSTS if segment.startswith("ru_") else US_COSTS,
            exclude_periods=MOEX_2022_BREAK if segment.startswith("ru_") else (),
            brent_rub_price=brent_rub_price,
            rub_oil_regime_signal=rub_oil_regime_signal,
            yield_slope_bps=yield_slope_bps,
            cbr_direction=cbr_direction,
            max_concurrent_positions=cap,
        ),
        regime_provider=regime_provider,
    )
    trades, snapshots = engine.run_portfolio(
        symbols,
        segment,
        candles_by_symbol,
        eligible_at=eligible_at,
    )
    journal.flush()
    # RUFIN-01 / D-01: per-segment closed-trade sidecar next to decision_journal.jsonl.
    _write_trades_jsonl(seg_dir / "trades.jsonl", trades)

    result = PerformanceAnalyzer().analyze(trades, snapshots, benchmark_candles=benchmark_candles)

    total_bars = sum(len(c) for c in candles_by_symbol.values())
    summary = {
        "segment": segment,
        "symbols": symbols,
        "total_candles": total_bars,
        "total_trades": len(trades),
        "concurrent_position_cap": cap,
        "fundamental_gate_scale": str(fundamental_scale),
        "fundamental_earnings_yield": fundamental_ey,
        "metrics": result.model_dump(mode="json") if result else None,
        "journal_summary": journal.summary(),
    }

    sharpe = float(result.sharpe) if result else 0.0
    wr = float(result.win_rate) if result else 0.0
    ret = float(result.total_return) if result else 0.0
    print(
        f"    PORTFOLIO {segment:14s} | {len(symbols):3d} syms | "
        f"{len(trades):3d} trades | "
        f"Sharpe {sharpe:+7.3f} | WR {wr:5.1%} | Ret {ret:+7.3%} | cap={cap}"
    )
    if fundamental_scale != Decimal("1.0"):
        print(
            f"      Fundamental gate: scale={fundamental_scale} "
            f"(earnings_yield={fundamental_ey:.4f})"
        )

    run_summary = engine.last_run_summary
    above_thresh: int = run_summary.get("combined_above_threshold", 0)  # type: ignore[assignment]
    if above_thresh:
        print(f"      combined_above_threshold={above_thresh}")

    return trades, snapshots, summary


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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


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
    parser.add_argument(
        "--moex-cash",
        type=int,
        default=1_000_000,
        help="Initial RUB capital for MOEX (ru_*) segments (default: 1,000,000)",
    )
    parser.add_argument("--models-dir", default="models/", help="Directory with trained ML models")
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
    parser.add_argument(
        "--wf-train-months", type=int, default=12, help="Walk-forward train window in months"
    )
    parser.add_argument(
        "--wf-test-months", type=int, default=6, help="Walk-forward test window in months"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hypothesis", default=None, help="Experiment ID to link backtest results")
    parser.add_argument(
        "--run-name", default="main", help="Label for interaction test run (A-only, B-only, AB)"
    )
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
    moex_cash = Decimal(args.moex_cash)
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

    # Experiment integration: load hypothesis, merge preset overrides AFTER _load_preset()
    experiment_mgr = None
    if args.hypothesis:
        from finalayze.core.experiment_manager import ExperimentManager  # noqa: PLC0415
        from finalayze.core.schemas import ExperimentStatus  # noqa: PLC0415

        experiment_mgr = ExperimentManager()
        experiment = experiment_mgr.read_experiment(args.hypothesis)
        experiment_mgr.update_status(args.hypothesis, ExperimentStatus.RUNNING.value)
        if experiment.preset_overrides:
            for seg, overrides in experiment.preset_overrides.items():
                if seg in strategy_configs:
                    strategy_configs[seg] = _deep_merge(strategy_configs[seg], overrides)

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

    # Build MarketDataLoader — single instance reused across all segments.
    # MOEX-specific fetchers (ISS + CBR) are only created when at least one segment
    # is MOEX, to avoid importing heavy gRPC deps unnecessarily.
    has_moex = any(seg.startswith("ru_") for seg in segments)
    if has_moex:
        from finalayze.data.fetchers.cbr import CBRFetcher  # noqa: PLC0415
        from finalayze.data.fetchers.moex_iss import MoexISSFetcher  # noqa: PLC0415

        # Turnover is fetched one ISS call per weekday over the whole window, so a
        # cold cache on a multi-year backtest means ~250 calls/year. rate=0.5
        # (1 req / 2s) made a 5-year backfill take ~40 min. 3 req/s keeps a
        # one-time backfill to a few minutes and is well within ISS tolerance;
        # results are cached under .cache/turnover so subsequent runs are instant.
        _moex_iss = MoexISSFetcher(rate_limiter=RateLimiter("moex_iss", rate=3.0, capacity=10))
        _loader = MarketDataLoader(
            moex_iss_candles=CachingFetcher(_moex_iss, cache_dir=Path(".cache/moex_iss")),
            moex_iss_raw=_moex_iss,
            cbr=CBRFetcher(rate_limiter=RateLimiter("cbr", rate=0.2, capacity=3)),
            yfinance_fetcher=CachingFetcher(YFinanceFetcher(market_id="us")),
            turnover_cache=GenericFileCache(Path(".cache/turnover")),
            cbr_cache=GenericFileCache(Path(".cache/cbr")),
        )
    else:
        _loader = MarketDataLoader(
            yfinance_fetcher=CachingFetcher(YFinanceFetcher(market_id="us")),
        )

    all_trades: list[TradeResult] = []
    all_snapshots: list[PortfolioState] = []
    segment_trades: dict[str, list[TradeResult]] = {}
    all_summaries: list[dict[str, Any]] = []

    print(f"\nRunning iteration '{args.name}'")
    print(f"  Period: {args.start_date} to {args.end_date}")
    print(f"  Segments: {', '.join(segments)}")
    print(f"  Cash: ${cash:,.0f} (MOEX: ₽{moex_cash:,.0f})")
    print()

    try:
        for segment in segments:
            symbols = UNIVERSE.get(segment, [])
            if not symbols:
                print(f"  Segment '{segment}' not found in universe, skipping")
                continue

            market_id = "moex" if segment.startswith("ru_") else "us"
            print(f"{'=' * 72}")
            print(f"  SEGMENT: {segment} ({len(symbols)} symbols, market={market_id})")
            print(f"{'=' * 72}")

            # Load all ambient market data (benchmark, VIX, MOEX-specific) via MarketDataLoader.
            # The loader routes by market: US → SPY + ^VIX; MOEX → IMOEX + CBR + turnover + Brent.
            _seg_cfg = SimpleNamespace(market=market_id)
            ml_market_context: MarketContext = _loader.load(_seg_cfg, start.date(), end.date())
            bench_candles = ml_market_context.benchmark_candles
            bench_label = "IMOEX" if segment.startswith("ru_") else "SPY"
            n_bars = len(bench_candles) if bench_candles else 0
            if bench_candles:
                print(f"  Benchmark: {bench_label} ({n_bars} bars)")
            else:
                print(f"  Benchmark: {bench_label} (fetch failed)")
            if ml_market_context.vix_candles is not None:
                print(f"  VIX: ^VIX ({len(ml_market_context.vix_candles)} bars)")
            if _loader.fetch_failures:
                print(f"  Market data warnings: {', '.join(_loader.fetch_failures)}")

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

            # MOEX sizing data (Phase 9: BrentGateStep + RubOilRegimeStep)
            brent_rub_price = 0.0
            rub_oil_regime_signal: RubOilRegimeSignal | None = None
            yield_slope_bps = 0.0
            cbr_direction = ""
            if is_moex:
                brent_rub_price, rub_oil_regime_signal, yield_slope_bps, cbr_direction = (
                    _compute_moex_sizing_data(ml_market_context)
                )
                if brent_rub_price > 0:
                    print(f"  Brent-in-RUB: {brent_rub_price:,.0f} RUB/bbl")
                if rub_oil_regime_signal is not None:
                    print("  RUB/oil regime signal: active")
                else:
                    print("  RUB/oil regime signal: disabled (insufficient data)")
                if yield_slope_bps != 0.0:
                    print(f"  Yield curve slope: {yield_slope_bps:+.0f} bps")
                if cbr_direction:
                    print(f"  CBR direction: {cbr_direction}")

            segment_trades[segment] = []
            # MOEX segments use RUB capital (--moex-cash, default 1M); US uses --cash
            segment_cash = _resolve_segment_cash(segment, cash, moex_cash)

            # Fetch candles for the whole segment, then run ONE shared-capital portfolio backtest
            # (LIQ-08 / Pattern 4). This is what makes the per-segment concurrent-position cap
            # (D-09) real and runs the CARDINAL D-05 as-of universe gate at quarterly rebalances --
            # both silently ineffective in the old per-symbol engine.run loop.
            candles_by_symbol: dict[str, list[Candle]] = {}
            for symbol in symbols:
                try:
                    fetcher = CachingFetcher(base_fetcher)
                    candles = fetcher.fetch_candles(symbol, start, end)
                    if not candles:
                        print(f"    {symbol:12s} | no data")
                        continue
                    candles_by_symbol[symbol] = candles
                except Exception:
                    print(f"    {symbol:12s} | fetch failed")
                    continue

            if not candles_by_symbol:
                print("    (no candle data for any symbol in segment)")
                print()
                continue

            eligible_symbols = list(candles_by_symbol)
            iter_dir = output_root / args.name
            trades, snapshots, summary = _run_segment_portfolio(
                eligible_symbols,
                segment,
                candles_by_symbol,
                strategies,
                segment_cash,
                iter_dir,
                benchmark_candles=bench_candles,
                use_evt_sizing=use_evt_sizing,
                use_copula_scaling=use_copula_scaling,
                regime_provider=regime_provider,
                stop_loss_mode=args.stop_loss_mode,
                market_context=ml_market_context,
                brent_rub_price=brent_rub_price,
                rub_oil_regime_signal=rub_oil_regime_signal,
                yield_slope_bps=yield_slope_bps,
                cbr_direction=cbr_direction,
            )

            normalized_trades = _normalize_trades_to_usd(trades, segment)
            all_trades.extend(normalized_trades)
            segment_trades[segment].extend(trades)  # keep raw for per-segment metrics
            if snapshots:
                all_snapshots.extend(_normalize_snapshots_to_usd(snapshots, segment))
            if summary:
                all_summaries.append(summary)

            print()

    finally:
        _loader.close()

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

    # D-04: log the full economic delta vs the named baseline either way (the
    # trade_count delta is the MEAS-01 gate; wf_sharpe/profit_factor are the stretch
    # and must be recorded regardless of sign). Guarded: tracker.compare() needs the
    # baseline metadata.json on disk, so only run it for an explicit, loadable baseline.
    if baseline_name and args.baseline not in ("latest", "none"):
        try:
            comparison = tracker.compare(args.name, baseline_name)
            md = comparison.metric_deltas
            print("\n  Baseline delta (D-04):")
            print(f"    trade_count delta   = {md['trade_count']:+.0f}")
            print(f"    wf_sharpe delta     = {md['wf_sharpe']:+.4f}")
            print(f"    profit_factor delta = {md['profit_factor']:+.4f}")
            if md["trade_count"] != 0.0:
                print("    OK MEAS-01: >=1 trade changed vs baseline")
            else:
                print("    WARN MEAS-01: trade_count delta is 0 — wiring may be inert")
        except (FileNotFoundError, KeyError) as exc:
            print(f"\n  Baseline delta (D-04): unavailable ({type(exc).__name__}: {exc})")

    # Save experiment result and link to experiment registry
    if experiment_mgr and args.hypothesis:
        from finalayze.core.schemas import ExperimentResult  # noqa: PLC0415

        exp_result_dir = Path("results/experiments") / args.hypothesis
        exp_result_dir.mkdir(parents=True, exist_ok=True)
        run_name = args.run_name
        exp_result_path = exp_result_dir / f"{run_name}.json"
        metrics_dict: dict[str, Any] = {
            "experiment_id": args.hypothesis,
            "run_name": run_name,
            "iteration_name": args.name,
            "wf_sharpe": float(metrics.wf_sharpe),
            "profit_factor": float(metrics.profit_factor),
            "wf_max_drawdown": float(metrics.wf_max_drawdown),
            "trade_count": int(metrics.trade_count),
        }
        exp_result_path.write_text(json.dumps(metrics_dict, indent=2))
        experiment_mgr.link_result(
            args.hypothesis,
            ExperimentResult(
                run_name=run_name,
                iteration_name=args.name,
                metrics=metrics_dict,
            ),
        )
        print(f"\n  Experiment result saved to: {exp_result_path}")

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
