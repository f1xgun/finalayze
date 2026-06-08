"""Shared Pydantic schemas (Layer 0).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import date, datetime  # noqa: TC003
from decimal import Decimal
from enum import IntEnum, StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type InstrumentType = Literal["stock", "etf", "bond", "future", "currency"]


class SignalDirection(StrEnum):
    """Direction of a trading signal."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class PortfolioLayer(StrEnum):
    """Portfolio layer in the multi-asset multi-timeframe system."""

    CORE = "core"  # 40-50%, OFZ-PK floaters, 6-12+ months
    STRATEGIC = "strategic"  # 25-30%, OFZ-PD duration rotation, 1-6 months
    TACTICAL = "tactical"  # 15-20%, OFZ-PD + stocks, 1-4 weeks
    SHORT = "short"  # 10-15%, stocks only, 1-5 days


class RiskProfile(StrEnum):
    """SAA risk profile (D-01/D-02). Maps to a fixed {deposit, ofz_pk, equity} weight vector."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    GROWTH = "growth"


class AssetClass(StrEnum):
    """The three SAA asset classes merged by the AllocationOrchestrator (D-01)."""

    DEPOSIT = "deposit"
    OFZ_PK = "ofz_pk"
    EQUITY = "equity"


@dataclass(frozen=True)
class AllocationProfile:
    """A risk profile's fixed target weights + its MaxDD cap (SAA-01/SAA-05, D-01/D-04).

    Weights are FIXED config vectors (D-03 -- never solver output). The vector MUST
    sum to 1.0 and be non-negative; validation is enforced by the L1 loader (Plan 03,
    V5 fail-closed), not here, so this stays a pure carrier mirroring LayerConfig.
    """

    profile: RiskProfile
    weights: dict[AssetClass, Decimal]
    max_drawdown_pct: Decimal


class Candle(BaseModel):
    """OHLCV candle for a single timeframe bar."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    market_id: str
    timeframe: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int = Field(ge=0)
    source: str | None = None

    @field_validator("timestamp")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes; all timestamps must be UTC-aware."""
        if v.tzinfo is None:
            msg = "timestamp must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v


class EventType(IntEnum):
    """Event type code carried on Signal.metadata.

    Read by ``StrategyCombiner._dedup_event_signals``: same ticker + cycle +
    event_type collapses to the highest-weighted contributor.
    """

    NONE = 0
    CBR = 1
    DIVIDEND = 2
    EARNINGS = 3


class AdxRegime(StrEnum):
    """ADX-derived market regime carried on Signal.metadata.

    Written by ``StrategyCombiner`` after computing ADX. Read by display/
    alerting layers.
    """

    TREND = "trend"
    MR = "mr"
    AMBIGUOUS = "ambiguous"


class SignalMetadata(BaseModel):
    """Cross-module protocol fields attached to every Signal.

    Only fields read by a module *other than* the producer belong here. Per-
    strategy internal numbers live on ``Signal.strategy_payload``; per-strategy
    confidence contributions from the combiner live on ``Signal.contributions``.
    """

    model_config = ConfigDict(frozen=True)

    event_type: EventType = EventType.NONE
    ml_confidence: float | None = None
    adx_value: float | None = None
    adx_regime: AdxRegime | None = None


class Signal(BaseModel):
    """Trading signal produced by a strategy.

    Notes:
        ``confidence`` is typed as ``float`` (not ``Decimal``) because it
        represents a probability/ratio in [0.0, 1.0], not a monetary value.
        The "Decimal for money fields" rule does not apply here.

        Signal carries three distinct payloads, replacing the old single
        ``features: dict[str, float]`` bag (see ADR Candidate 3):

        - ``metadata`` — typed cross-module protocol (event_type, ml_confidence,
          ADX). Read by non-producers.
        - ``strategy_payload`` — strategy-internal numbers (e.g. sma_ratio,
          ou_spread, kalman). Single-writer, read only by display/tests.
        - ``contributions`` — per-strategy confidence contributions written by
          ``StrategyCombiner`` keyed by strategy name.
    """

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    symbol: str
    market_id: str
    segment_id: str
    direction: SignalDirection
    confidence: float
    reasoning: str
    metadata: SignalMetadata = Field(default_factory=SignalMetadata)
    strategy_payload: dict[str, float] = Field(default_factory=dict)
    contributions: dict[str, float] = Field(default_factory=dict)
    instrument_type: str = "stock"  # "stock" or "bond"
    signal_price: Decimal | None = None

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_probability(cls, v: float) -> float:
        """Validate that confidence is a probability in [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            msg = f"confidence must be in [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v


class ExitReason(StrEnum):
    """Which backtest exit path closed a position.

    Recorded on every closed ``TradeResult`` so post-hoc attribution
    (RUFIN-01 / D-01) can split realised PnL by exit mechanism. StrEnum
    members equal their lowercase string value, so the stored field is a
    plain str.
    """

    STOP = "stop"
    PROFIT_TARGET = "profit_target"
    TIME = "time"
    SIGNAL = "signal"
    FORCE_CLOSE = "force_close"


class TradeResult(BaseModel):
    """Result of an executed trade."""

    model_config = ConfigDict(frozen=True)

    signal_id: UUID
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    pnl_pct: Decimal
    hold_bars: int | None = None
    coupon_income: Decimal = Decimal(0)  # bond coupon income during hold
    instrument_type: str = "stock"  # "stock" or "bond"
    # RUFIN-01 attribution (append-only optional fields; never reorder/require):
    exit_reason: str | None = None  # ExitReason value (str-compatible StrEnum)
    entry_strategy: str | None = None  # strategy that opened the position


class PortfolioState(BaseModel):
    """Snapshot of portfolio at a point in time."""

    model_config = ConfigDict(frozen=True)

    cash: Decimal
    positions: dict[str, Decimal]
    equity: Decimal
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes; all timestamps must be UTC-aware."""
        if v.tzinfo is None:
            msg = "timestamp must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v


class BacktestResult(BaseModel):
    """Aggregate metrics from a backtest run."""

    model_config = ConfigDict(frozen=True)

    sharpe: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    total_return: Decimal
    total_trades: int
    # Risk-adjusted ratios (computed from equity snapshots)
    sortino_ratio: Decimal | None = None
    calmar_ratio: Decimal | None = None
    turnover_ratio: Decimal | None = None
    # Sharpe statistical significance (t-test under IID normal returns)
    sharpe_n_samples: int | None = None
    sharpe_t_statistic: Decimal | None = None
    sharpe_p_value: Decimal | None = None
    # Benchmark comparison fields (populated when benchmark_candles are provided)
    alpha: Decimal | None = None
    beta: Decimal | None = None
    information_ratio: Decimal | None = None
    max_relative_drawdown: Decimal | None = None
    benchmark_return: Decimal | None = None


class NewsArticle(BaseModel):
    """A news article fetched from an external source."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    source: str
    title: str
    content: str
    url: str
    language: str  # "en" | "ru"
    published_at: datetime
    symbols: list[str] = []
    affected_segments: list[str] = []
    scope: str | None = None  # "global" | "us" | "russia" | "sector"
    raw_sentiment: float | None = None
    credibility_score: float | None = None

    @field_validator("published_at")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes."""
        if v.tzinfo is None:
            msg = "published_at must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v

    @field_validator("raw_sentiment")
    @classmethod
    def sentiment_in_range(cls, v: float | None) -> float | None:
        """Validate sentiment is in [-1.0, 1.0] when provided."""
        if v is not None and not (-1.0 <= v <= 1.0):
            msg = f"raw_sentiment must be in [-1.0, 1.0], got {v}"
            raise ValueError(msg)
        return v


class SentimentResult(BaseModel):
    """Result of LLM sentiment analysis on a news article."""

    model_config = ConfigDict(frozen=True)

    sentiment: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    tickers: list[str] = []  # LLM-extracted ticker symbols from the article
    is_fallback: bool = False  # True when this result is a fallback (LLM failed)

    @field_validator("sentiment")
    @classmethod
    def sentiment_in_range(cls, v: float) -> float:
        """Validate sentiment is in [-1.0, 1.0]."""
        if not (-1.0 <= v <= 1.0):
            msg = f"sentiment must be in [-1.0, 1.0], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        """Validate confidence is in [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            msg = f"confidence must be in [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v


# ── Iteration Tracking Schemas ──────────────────────────────────────────────


class GateResult(BaseModel):
    """Result of a single acceptance gate."""

    model_config = ConfigDict(frozen=True)

    name: str
    gate_type: str  # "safety" | "calibration"
    passed: bool
    value: float
    threshold: float
    message: str


class IterationMetrics(BaseModel):
    """All tracked metrics for one iteration."""

    model_config = ConfigDict(frozen=True)

    # Primary (6)
    wf_sharpe: float
    wf_max_drawdown: float
    profit_factor: float
    calmar_ratio: float
    trade_count: int
    avg_hold_bars: float
    segment_pnl_share: dict[str, float]

    # Secondary (6)
    sortino_ratio: float
    win_rate_by_segment: dict[str, float]
    information_ratio: float | None
    mc_5th_pct_sharpe: float
    model_disagreement: float
    turnover_adjusted_return: float

    # Diagnostic
    gross_sharpe: float
    net_sharpe: float
    param_stability_cv: float
    per_model_proba_mean: dict[str, float]


class IterationMetadata(BaseModel):
    """Complete snapshot of one iteration."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    name: str
    description: str
    created_at: datetime
    git_describe: str
    git_sha: str
    git_dirty: bool
    config_hash: str
    strategy_configs: dict[str, Any]
    backtest_config: dict[str, Any]
    metrics: IterationMetrics
    gate_results: list[GateResult]
    verdict: str  # "PASS" | "WARN" | "REJECT"
    tags: list[str] = []


class IterationComparison(BaseModel):
    """Delta between two iterations."""

    model_config = ConfigDict(frozen=True)

    current: str
    baseline: str
    metric_deltas: dict[str, float]
    gate_results: list[GateResult]
    verdict: str


class FXRate(BaseModel):
    """Daily official FX rate from CBR."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime  # UTC midnight for the date
    pair: str  # "USDRUB", "EURRUB"
    rate: Decimal

    @field_validator("timestamp")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes; all timestamps must be UTC-aware."""
        if v.tzinfo is None:
            msg = "timestamp must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v


class KeyRateRecord(BaseModel):
    """CBR key rate effective from a given date."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime  # UTC midnight of effective date
    rate: Decimal  # Annual rate as decimal fraction: 0.16 = 16%

    @field_validator("timestamp")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes; all timestamps must be UTC-aware."""
        if v.tzinfo is None:
            msg = "timestamp must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v


class TurnoverRecord(BaseModel):
    """Aggregate MOEX market turnover for a trading day."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime  # UTC midnight for the date
    volume_rub: Decimal

    @field_validator("timestamp")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes; all timestamps must be UTC-aware."""
        if v.tzinfo is None:
            msg = "timestamp must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v


class BondInfo(BaseModel):
    """Static metadata for an OFZ bond."""

    model_config = ConfigDict(frozen=True)

    figi: str
    ticker: str
    isin: str
    name: str
    face_value: Decimal
    coupon_rate: Decimal  # annual % (e.g. 7.10 for 7.10%)
    coupon_frequency: int  # payments per year (2 for semiannual)
    maturity_date: date
    floating_coupon: bool = False
    class_code: str = "TQOB"
    currency: str = "RUB"
    # Extended fields (Phase 3)
    amortization_flag: bool = False
    inflation_linked: bool = False
    initial_nominal: Decimal | None = None  # original face value before amortization
    day_count_convention: str = "actual/365"  # from T-Invest metadata
    bond_type: str = "fixed"  # "fixed", "floating", "amortizing", "inflation_linked"


class CouponPayment(BaseModel):
    """A single coupon payment event."""

    model_config = ConfigDict(frozen=True)

    bond_figi: str
    coupon_date: date  # payment date
    record_date: date  # T-2 business days before payment
    amount_per_bond: Decimal  # gross RUB per bond
    coupon_number: int
    is_floating: bool = False


class CouponEvent(BaseModel):
    """Coupon event emitted on ex-coupon date for bond coupon scheduling."""

    model_config = ConfigDict(frozen=True)

    bond_figi: str
    bond_ticker: str
    coupon_date: date
    record_date: date
    amount_per_bond: Decimal
    coupon_number: int
    is_floating: bool = False


class AccruedInterest(BaseModel):
    """Daily accrued interest (NKD) for a bond."""

    model_config = ConfigDict(frozen=True)

    bond_figi: str
    date: date
    value: Decimal  # RUB per bond
    value_percent: Decimal  # % of face value


@dataclass(frozen=True)
class BondPositionRecord:
    """Immutable record for a bond position in a portfolio layer.

    Stores entry conditions (YTM, price, clean price) for P&L tracking
    and risk management.
    """

    symbol: str
    quantity: Decimal
    entry_ytm_pct: Decimal  # yield-to-maturity at entry (%)
    entry_date: date
    entry_price: Decimal  # dirty price at entry (RUB)
    entry_clean_pct: Decimal  # clean price as % of face at entry
    layer_id: str


@dataclass(frozen=True)
class MultiTimeframeContext:
    """Higher-timeframe context derived from daily candles.

    All values use COMPLETED periods only (no partial bars).
    Weekly: last completed Mon-Fri week. Monthly: last completed calendar month.
    A 2-bar lag (_EXTERNAL_DATA_LAG_BARS) is applied on top.
    """

    weekly_completed: Candle | None = dc_field(default=None)
    monthly_completed: Candle | None = dc_field(default=None)
    # Derived features
    weekly_rsi_14: float | None = dc_field(default=None)
    weekly_sma_50_ratio: float | None = dc_field(default=None)  # close / SMA50 ratio
    monthly_trend_direction: int | None = dc_field(default=None)  # +1, 0, -1


class FundamentalSnapshot(BaseModel):
    """Point-in-time fundamental snapshot for one symbol (FUND-01).

    ``as_of`` is the publication/fetch date and is the look-ahead filter key
    (downstream: ``as_of <= D``). Every fundamental field is Optional and
    defaults to ``None`` (unavailable) — values are never fabricated.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    as_of: datetime
    pe_ratio: float | None = None
    ev_ebitda: float | None = None
    revenue_ttm: float | None = None
    net_margin: float | None = None
    roe: float | None = None
    eps_ttm: float | None = None
    dividend_yield: float | None = None
    market_cap: float | None = None
    currency: str | None = None

    @field_validator("as_of")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes; ``as_of`` must be UTC-aware."""
        if v.tzinfo is None:
            msg = "as_of must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v


class ReportEvent(BaseModel):
    """Earnings/report calendar event (EARN-01).

    Calendar-only: ``get_asset_reports`` carries no actuals. ``report_date`` is
    the publication date and is usable as an ``as_of`` look-ahead key.
    ``period_type`` is a plain string mapped from the SDK enum's ``.name``
    ("ANNUAL" | "QUARTER" | "SEMIANNUAL" | "UNSPECIFIED").
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    report_date: datetime
    period_year: int
    period_num: int
    period_type: str

    @field_validator("report_date")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes; ``report_date`` must be UTC-aware."""
        if v.tzinfo is None:
            msg = "report_date must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v


@dataclass(frozen=True)
class MoexMarketData:
    """MOEX-specific ambient data. None = unavailable."""

    fx_rates: tuple[FXRate, ...] | None = dc_field(default=None)
    key_rates: tuple[KeyRateRecord, ...] | None = dc_field(default=None)
    commodity_candles: dict[str, tuple[Candle, ...]] | None = dc_field(default=None)
    turnover: tuple[TurnoverRecord, ...] | None = dc_field(default=None)
    fundamentals: tuple[FundamentalSnapshot, ...] | None = dc_field(default=None)


@dataclass(frozen=True)
class MarketContext:
    """Ambient market data passed to strategies for cross-asset / regime features.

    Both fields are optional: MOEX segments will have vix_candles=None,
    and benchmark_candles may be absent if the benchmark fetch failed.
    """

    # TODO: Design specifies tuple[Candle, ...] for immutability. Kept as list
    # for backward compatibility with existing consumers. Migrate to tuple in v0.2.0.
    benchmark_candles: list[Candle] | None = dc_field(default=None)
    vix_candles: list[Candle] | None = dc_field(default=None)
    moex_data: MoexMarketData | None = dc_field(default=None)


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for a portfolio layer."""

    layer: PortfolioLayer
    capital_pct: Decimal  # target allocation (e.g. 0.40 for 40%)
    max_drawdown_pct: Decimal  # max peak-to-trough DD (e.g. 0.03 for 3%)
    max_positions: int
    rebalance_interval: str  # "daily", "weekly", "monthly", "quarterly", "event"
    allowed_instrument_types: tuple[str, ...] = ("stock",)
    yield_stop_bps: int = 0  # 0 = no yield stop (for Core)


# Default layer configurations per plan
DEFAULT_LAYER_CONFIGS: dict[PortfolioLayer, LayerConfig] = {
    PortfolioLayer.CORE: LayerConfig(
        layer=PortfolioLayer.CORE,
        capital_pct=Decimal("0.45"),
        max_drawdown_pct=Decimal("0.03"),
        max_positions=4,
        rebalance_interval="quarterly",
        allowed_instrument_types=("bond",),
        yield_stop_bps=0,
    ),
    PortfolioLayer.STRATEGIC: LayerConfig(
        layer=PortfolioLayer.STRATEGIC,
        capital_pct=Decimal("0.275"),
        max_drawdown_pct=Decimal("0.05"),
        max_positions=5,
        rebalance_interval="monthly",
        allowed_instrument_types=("bond",),
        yield_stop_bps=50,
    ),
    PortfolioLayer.TACTICAL: LayerConfig(
        layer=PortfolioLayer.TACTICAL,
        capital_pct=Decimal("0.175"),
        max_drawdown_pct=Decimal("0.05"),
        max_positions=5,
        rebalance_interval="weekly",
        allowed_instrument_types=("bond", "stock"),
        yield_stop_bps=30,
    ),
    PortfolioLayer.SHORT: LayerConfig(
        layer=PortfolioLayer.SHORT,
        capital_pct=Decimal("0.10"),
        max_drawdown_pct=Decimal("0.02"),  # risk review: 2% not 5%
        max_positions=6,
        rebalance_interval="daily",
        allowed_instrument_types=("stock", "bond"),  # was ("stock",)
        yield_stop_bps=0,
    ),
}


# ── Debate Protocol Schemas ─────────────────────────────────────────────────


class DebateStatus(StrEnum):
    """Status of a structured debate."""

    OPEN = "open"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class FileLineSource(BaseModel):
    """Source reference pointing to a specific file and line."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["file"] = "file"
    path: str
    line: int
    excerpt: str
    snapshot_sha: str | None = None
    """SHA-256 digest of the file content at claim creation time.

    Used by the arbiter to detect file changes between claim creation and
    verification. If the file's current SHA differs from snapshot_sha, the
    arbiter skips the line-level check and marks the claim as UNTESTABLE
    rather than incorrectly CONTRADICTED.
    """


class MetricSource(BaseModel):
    """Source reference citing a metric value from iteration history."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["metric"] = "metric"
    metric_name: str
    value: float
    iteration: str


ClaimSource = Annotated[
    FileLineSource | MetricSource,
    Field(discriminator="kind"),
]


class Claim(BaseModel):
    """A verifiable assertion made by an agent."""

    model_config = ConfigDict(frozen=True)

    statement: str
    source: ClaimSource
    confidence: float

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_probability(cls, v: float) -> float:
        """Validate that confidence is a probability in [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            msg = f"confidence must be in [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v


class AgentOutput(BaseModel):
    """Structured agent recommendation with verifiable evidence."""

    model_config = ConfigDict(frozen=True)

    agent_name: str
    recommendation: str
    claims: list[Claim] = Field(min_length=1)
    timestamp: datetime


class ConflictType(StrEnum):
    """Type of conflict detected between agent outputs."""

    DIRECTION = "direction"
    METRIC = "metric"
    STATEMENT = "statement"


class ConflictSeverity(StrEnum):
    """Severity level of a detected conflict."""

    CRITICAL = "critical"
    HIGH = "high"
    LOW = "low"


class ConflictReport(BaseModel):
    """Structured report of a detected conflict between agent claims."""

    model_config = ConfigDict(frozen=True)

    conflict_id: str  # SHA-256 hex digest dedup key
    conflict_type: ConflictType
    severity: ConflictSeverity
    involved_claims: list[Claim] = Field(min_length=2)
    agent_names: list[str] = Field(min_length=2)
    detected_at: datetime
    confidence_delta: float | None = None


class ClaimVerdict(StrEnum):
    """Verdict from arbiter fact-checking a claim."""

    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNTESTABLE = "untestable"


class ClaimCheckResult(BaseModel):
    """Result of fact-checking a single claim."""

    model_config = ConfigDict(frozen=True)

    claim: Claim
    verdict: ClaimVerdict
    evidence: str


class FactCheckReport(BaseModel):
    """Arbiter's structured fact-check report on a set of claims."""

    model_config = ConfigDict(frozen=True)

    debate_id: str
    arbiter_timestamp: datetime
    results: list[ClaimCheckResult]

    @property
    def has_contradictions(self) -> bool:
        """Return True if any claim was CONTRADICTED."""
        return any(r.verdict == ClaimVerdict.CONTRADICTED for r in self.results)

    def to_markdown(self) -> str:
        """Render the fact-check report as structured Markdown."""
        sections: dict[str, list[ClaimCheckResult]] = {
            "Verified": [],
            "Contradicted": [],
            "Untestable": [],
        }
        for r in self.results:
            if r.verdict == ClaimVerdict.VERIFIED:
                sections["Verified"].append(r)
            elif r.verdict == ClaimVerdict.CONTRADICTED:
                sections["Contradicted"].append(r)
            else:
                sections["Untestable"].append(r)

        lines: list[str] = [f"# Fact-Check Report: {self.debate_id}", ""]
        for heading, items in sections.items():
            lines.append(f"## {heading}")
            lines.append("")
            if not items:
                lines.append("_None_")
            for item in items:
                lines.append(f"- **{item.claim.statement}**")
                lines.append(f"  Evidence: {item.evidence}")
            lines.append("")
        return "\n".join(lines)


class DebateState(BaseModel):
    """Persistent state of a structured debate between agents."""

    model_config = ConfigDict(frozen=True)

    debate_id: str
    topic: str
    status: DebateStatus
    created: str  # ISO date string "YYYY-MM-DD"
    agents: list[str]
    arbiter_report: FactCheckReport | None = None
    resolution: str | None = None
    experiment_id: str | None = None

    @model_validator(mode="after")
    def escalated_requires_experiment_id(self) -> DebateState:
        """Validate that escalated debates have an experiment_id."""
        if self.status == DebateStatus.ESCALATED and self.experiment_id is None:
            msg = "experiment_id is required when status is 'escalated'"
            raise ValueError(msg)
        return self


# ── Experiment Registry Schemas ──────────────────────────────────────────────


class ExperimentStatus(StrEnum):
    """Lifecycle status of an experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


_VALID_OPERATORS = frozenset({">=", "<=", ">", "<"})


class SuccessCriteria(BaseModel):
    """Defines a single metric threshold for experiment success/failure."""

    model_config = ConfigDict(frozen=True)

    metric: str
    threshold: float
    operator: str = ">="

    @field_validator("operator")
    @classmethod
    def operator_must_be_valid(cls, v: str) -> str:
        """Validate that operator is in the allowed whitelist."""
        if v not in _VALID_OPERATORS:
            msg = f"operator must be one of {sorted(_VALID_OPERATORS)}, got '{v}'"
            raise ValueError(msg)
        return v


class ExperimentResult(BaseModel):
    """Result of a single backtest run within an experiment."""

    model_config = ConfigDict(frozen=True)

    run_name: str
    iteration_name: str
    metrics: dict[str, Any]


_EXPERIMENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class ExperimentState(BaseModel):
    """Persistent state of an experiment in the registry."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    hypothesis: str
    success_criteria: SuccessCriteria
    status: ExperimentStatus
    created: str  # ISO date "YYYY-MM-DD"
    debate_id: str | None = None
    results: list[ExperimentResult] = []
    verdict: str | None = None
    reasoning: str | None = None
    preset_overrides: dict[str, Any] | None = None

    @field_validator("experiment_id")
    @classmethod
    def experiment_id_safe(cls, v: str) -> str:
        """Validate experiment_id is safe for use as a filename."""
        if not _EXPERIMENT_ID_PATTERN.match(v):
            msg = f"experiment_id must match [a-zA-Z0-9_-]+, got '{v}'"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def terminal_status_requires_verdict(self) -> ExperimentState:
        """Validate that terminal statuses have a verdict set."""
        terminal = {
            ExperimentStatus.ACCEPTED,
            ExperimentStatus.REJECTED,
            ExperimentStatus.INCONCLUSIVE,
        }
        if self.status in terminal and self.verdict is None:
            msg = "verdict is required when status is terminal (ACCEPTED/REJECTED/INCONCLUSIVE)"
            raise ValueError(msg)
        return self


# ── Snapshot helpers ──────────────────────────────────────────────────────────


def compute_file_sha(path: str) -> str:
    """Compute SHA-256 digest of a file's content.

    Used by agent definitions when creating FileLineSource claims to capture
    the file's integrity at claim-creation time. The arbiter compares this
    digest against the current file content to detect post-claim edits.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        Hex-encoded SHA-256 digest string (64 characters).

    Raises:
        FileNotFoundError: if the file does not exist.
        OSError: if the file cannot be read.
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# Deposit sleeve types (Phase 71 -- total-return accounting + deposit ladder)
# ---------------------------------------------------------------------------
@dataclass
class DepositTranche:
    """One rung of the deposit ladder (D-01).

    Mutable on purpose: ``accrued_net``/``accrued_gross`` mutate per bar as the
    sleeve broker compounds interest, and ``broken`` flips once a pre-maturity
    break resets the tranche to the demand rate (D-03).
    """

    principal: Decimal
    term_months: int  # 3 / 6 / 12
    annual_rate: Decimal  # fraction; key_rate - spread at open (D-04)
    open_date: date
    maturity_date: date
    accrued_net: Decimal = Decimal(0)
    accrued_gross: Decimal = Decimal(0)
    broken: bool = False


@dataclass(frozen=True)
class BankAllocation:
    """ASV per-bank insured-exposure slice (D-09 / R-5)."""

    bank_id: str
    principal: Decimal
    accrued_net: Decimal = Decimal(0)

    @property
    def insured_exposure(self) -> Decimal:
        """Principal plus accrued net interest (both count toward the cap, D-09)."""
        return self.principal + self.accrued_net
