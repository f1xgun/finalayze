"""Bond backtest engine (Layer 5).

Purpose-built engine for OFZ bond backtesting. Separate from equity
BacktestEngine because bond mechanics are fundamentally different:
- Clean/dirty price accounting
- Coupon income tracking (with 13% NDFL tax)
- DV01-based position sizing
- Yield-based stop-loss
- Duration monitoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

import structlog

from finalayze.backtest.costs import MOEX_BOND_COSTS, bond_total_cost
from finalayze.core.bond_math import dirty_price, dv01, modified_duration, nkd, ytm
from finalayze.core.schemas import (
    Candle,
    CouponPayment,
    LayerConfig,
    PortfolioLayer,
    Signal,
    SignalDirection,
    TradeResult,
)
from finalayze.execution.bond_simulated_broker import BondSimulatedBroker
from finalayze.risk.dv01_sizing import DV01BudgetStep, EqualWeightBondSizer
from finalayze.risk.yield_stop import YieldStop
from finalayze.strategies.bond_duration_rotation import CBRRegime, classify_regime

if TYPE_CHECKING:
    from finalayze.backtest.costs import TransactionCosts
    from finalayze.core.schemas import BondInfo
    from finalayze.data.fetchers.cbr import MacroContextProvider

logger = structlog.get_logger(__name__)

_OFZ_FACE_VALUE = Decimal(1000)
_HUNDRED = Decimal(100)
_TRADING_DAYS_PER_YEAR = 252
_MIN_EQUITY_CURVE_FOR_SHARPE = 3
_MIN_RETURNS_FOR_SHARPE = 2
_STD_EPSILON = 1e-12


class BondStrategyFn(Protocol):
    """Protocol for bond strategy callables.

    Strategies that need macro context accept keyword-only arguments
    (key_rate, ruonia_7d_avg, etc.) via **kwargs.
    """

    def __call__(
        self,
        symbol: str,
        candles: list[Candle],
        positions: dict[str, BondPosition],
        bar_idx: int,
        **kwargs: Any,
    ) -> Signal | None: ...


@dataclass
class BondPosition:
    """Tracks an open bond position."""

    symbol: str
    quantity: int
    entry_clean_pct: Decimal
    entry_nkd: Decimal
    entry_ytm_pct: Decimal
    entry_bar_idx: int
    entry_date: date
    coupon_income: Decimal = Decimal(0)  # cumulative gross during hold


@dataclass
class BondBacktestConfig:
    """Configuration for bond backtest."""

    initial_cash: Decimal = Decimal(1_000_000)
    max_positions: int = 5
    dv01_sizer: DV01BudgetStep | EqualWeightBondSizer = field(default_factory=DV01BudgetStep)
    yield_stop: YieldStop = field(default_factory=lambda: YieldStop(threshold_bps=50))
    transaction_costs: TransactionCosts = field(default_factory=lambda: MOEX_BOND_COSTS)
    face_value: Decimal = _OFZ_FACE_VALUE
    max_hold_bars: int = 120  # ~6 months for strategic layer


@dataclass
class BondBacktestResult:
    """Results from a bond backtest run."""

    trades: list[TradeResult]
    equity_curve: list[Decimal]
    dates: list[date]
    total_coupon_income_gross: Decimal
    total_coupon_income_net: Decimal
    total_tax_paid: Decimal
    total_return_pct: Decimal
    max_drawdown_pct: Decimal
    sharpe_ratio: Decimal
    trade_count: int
    win_rate: Decimal
    profit_factor: Decimal
    ofz_rotation_active: bool = False


class BondBacktestEngine:
    """Backtest engine for OFZ bond strategies.

    Processes daily candles (clean price as % of face value), generates
    signals from a bond strategy, executes via BondSimulatedBroker, and
    tracks portfolio value at dirty prices.
    """

    def __init__(
        self,
        config: BondBacktestConfig | None = None,
    ) -> None:
        self._config = config or BondBacktestConfig()

    def run(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        bond_info: dict[str, BondInfo],
        coupon_schedule: dict[str, list[CouponPayment]],
        strategy_fn: Any,  # BondStrategyFn -- use Any to accept plain callables
        nkd_series: dict[str, dict[date, Decimal]] | None = None,
        macro_provider: MacroContextProvider | None = None,
        layer_configs: dict[PortfolioLayer, LayerConfig] | None = None,
        as_of_date: date | None = None,
    ) -> BondBacktestResult:
        """Run bond backtest.

        Args:
            candles_by_symbol: Daily candles per bond symbol (close = clean price %).
            bond_info: Static bond metadata per symbol.
            coupon_schedule: Coupon payment schedule per symbol.
            strategy_fn: Callable that returns a Signal or None for each bar.
                Signature: (symbol, candles[:bar_idx+1], open_positions, bar_idx) -> Signal | None
            nkd_series: Optional pre-computed NKD per symbol per date.
                If None, NKD is estimated from coupon schedule.
            macro_provider: Optional macro context provider. When supplied,
                key_rate, ruonia_7d_avg, cpi_yoy, and last_cbr_decision are
                forwarded to strategy_fn as keyword arguments.

        Returns:
            BondBacktestResult with trades, equity curve, and metrics.
        """
        cfg = self._config

        # Align all symbols to common date index
        all_dates = self._build_date_index(candles_by_symbol)

        # OFZ rotation check
        rotation_active = False
        if layer_configs is not None:
            from finalayze.core.bond_cycle import apply_ofz_rotation  # noqa: PLC0415

            _as_of = as_of_date or (all_dates[-1] if all_dates else datetime.now(tz=UTC).date())
            effective = apply_ofz_rotation(layer_configs, _as_of)
            rotation_active = effective != layer_configs
            if rotation_active:
                logger.info(
                    "ofz_rotation_active",
                    core_pct=str(effective[PortfolioLayer.CORE].capital_pct),
                    strategic_pct=str(effective[PortfolioLayer.STRATEGIC].capital_pct),
                )

        if not all_dates:
            return self._empty_result(rotation_active=rotation_active)

        # Initialize broker with coupon schedule
        broker = BondSimulatedBroker(
            initial_cash=cfg.initial_cash,
            coupon_schedule=coupon_schedule,
            face_value=cfg.face_value,
        )

        # Build candle lookup: symbol -> date -> Candle
        candle_lookup: dict[str, dict[date, Candle]] = {}
        for symbol, candles in candles_by_symbol.items():
            candle_lookup[symbol] = {self._candle_date(c): c for c in candles}

        positions: dict[str, BondPosition] = {}
        trades: list[TradeResult] = []
        equity_curve: list[Decimal] = []
        dates_out: list[date] = []

        for bar_idx, current_date in enumerate(all_dates):
            # 1. Process coupon payments via broker (credits net to cash)
            broker.process_coupons(current_date)

            # Update coupon tracking for open positions (gross amount)
            self._update_coupon_tracking(positions, coupon_schedule, current_date)

            # 2. Compute CBR regime for regime-adaptive yield stops
            regime = self._get_current_regime(current_date, macro_provider)

            # 3. Check yield-based stops and max hold bars
            close_trades = self._check_stops_and_limits(
                positions,
                bar_idx,
                current_date,
                candle_lookup,
                nkd_series,
                broker,
                bond_info,
                coupon_schedule,
                regime=regime,
            )
            trades.extend(close_trades)

            # 3. Generate new signals (only if we have capacity)
            if len(positions) < cfg.max_positions:
                self._process_signals(
                    positions,
                    bar_idx,
                    current_date,
                    all_dates,
                    candles_by_symbol,
                    candle_lookup,
                    nkd_series,
                    bond_info,
                    coupon_schedule,
                    broker,
                    strategy_fn,
                    macro_provider,
                )

            # 4. Record equity curve (dirty price valuation)
            equity = self._compute_equity(
                positions,
                current_date,
                candle_lookup,
                nkd_series,
                bond_info,
                coupon_schedule,
                broker,
            )
            equity_curve.append(equity)
            dates_out.append(current_date)

        # Close remaining positions at last bar
        if all_dates:
            last_date = all_dates[-1]
            last_bar_idx = len(all_dates) - 1
            for sym in list(positions):
                trade = self._close_position(
                    sym,
                    positions[sym],
                    last_date,
                    last_bar_idx,
                    candle_lookup,
                    nkd_series,
                    broker,
                    bond_info,
                    coupon_schedule,
                )
                if trade:
                    trades.append(trade)
            positions.clear()

        return self._build_result(
            trades,
            equity_curve,
            dates_out,
            broker,
            cfg.initial_cash,
            rotation_active=rotation_active,
        )

    # ------------------------------------------------------------------
    # Bar-level processing helpers (extracted to reduce branch count)
    # ------------------------------------------------------------------

    @staticmethod
    def _update_coupon_tracking(
        positions: dict[str, BondPosition],
        coupon_schedule: dict[str, list[CouponPayment]],
        current_date: date,
    ) -> None:
        """Update cumulative coupon income for open positions."""
        for sym, pos in positions.items():
            for coupon in coupon_schedule.get(sym, []):
                if coupon.coupon_date == current_date:
                    pos.coupon_income += coupon.amount_per_bond * pos.quantity

    def _check_stops_and_limits(
        self,
        positions: dict[str, BondPosition],
        bar_idx: int,
        current_date: date,
        candle_lookup: dict[str, dict[date, Candle]],
        nkd_series: dict[str, dict[date, Decimal]] | None,
        broker: BondSimulatedBroker,
        bond_info: dict[str, BondInfo],
        coupon_schedule: dict[str, list[CouponPayment]],
        *,
        regime: int = 1,
    ) -> list[TradeResult]:
        """Check yield stops and max hold bars, close triggered positions."""
        cfg = self._config
        symbols_to_close: list[tuple[str, str]] = []

        for sym, pos in list(positions.items()):
            candle = candle_lookup.get(sym, {}).get(current_date)
            if candle is None:
                continue

            info = bond_info.get(sym)
            if info is None:
                continue

            # Compute current YTM for stop check
            try:
                current_ytm = ytm(
                    candle.close,
                    info.coupon_rate,
                    cfg.face_value,
                    info.coupon_frequency,
                    current_date,
                    info.maturity_date,
                )
            except (ValueError, ZeroDivisionError):
                current_ytm = pos.entry_ytm_pct  # fallback: no stop

            if cfg.yield_stop.is_stopped_with_regime(pos.entry_ytm_pct, current_ytm, regime=regime):
                symbols_to_close.append((sym, "yield_stop"))
                continue

            if bar_idx - pos.entry_bar_idx >= cfg.max_hold_bars:
                symbols_to_close.append((sym, "max_hold"))

        closed_trades: list[TradeResult] = []
        for sym, reason in symbols_to_close:
            trade = self._close_position(
                sym,
                positions[sym],
                current_date,
                bar_idx,
                candle_lookup,
                nkd_series,
                broker,
                bond_info,
                coupon_schedule,
            )
            if trade:
                closed_trades.append(trade)
                del positions[sym]
                logger.debug(
                    "bond_position_closed",
                    symbol=sym,
                    reason=reason,
                    bar_idx=bar_idx,
                )
        return closed_trades

    def _process_signals(
        self,
        positions: dict[str, BondPosition],
        bar_idx: int,
        current_date: date,
        all_dates: list[date],
        candles_by_symbol: dict[str, list[Candle]],
        candle_lookup: dict[str, dict[date, Candle]],
        nkd_series: dict[str, dict[date, Decimal]] | None,
        bond_info: dict[str, BondInfo],
        coupon_schedule: dict[str, list[CouponPayment]],
        broker: BondSimulatedBroker,
        strategy_fn: Any,
        macro_provider: MacroContextProvider | None = None,
    ) -> None:
        """Generate signals and open new positions."""
        cfg = self._config

        # Build macro kwargs once per bar (same snapshot for all symbols)
        macro_kwargs: dict[str, Any] = {}
        if macro_provider is not None:
            snapshot = macro_provider.get_snapshot(current_date)
            macro_kwargs = {
                "key_rate": snapshot.key_rate,
                "ruonia_7d_avg": snapshot.ruonia_7d_avg,
                "cpi_yoy": snapshot.cpi_yoy,
                "last_cbr_decision": snapshot.last_cbr_decision,
            }

        portfolio_dv01 = self._compute_portfolio_dv01(
            positions,
            current_date,
            candle_lookup,
            nkd_series,
            bond_info,
            coupon_schedule,
        )

        for sym in candles_by_symbol:
            if sym in positions:
                continue
            candle = candle_lookup.get(sym, {}).get(current_date)
            if candle is None:
                continue

            # Get candles up to current bar (no look-ahead)
            sym_candles = [
                candle_lookup[sym][d]
                for d in all_dates[: bar_idx + 1]
                if d in candle_lookup.get(sym, {})
            ]

            signal = strategy_fn(sym, sym_candles, positions, bar_idx, **macro_kwargs)
            if signal is None or signal.direction != SignalDirection.BUY:
                continue

            opened = self._try_open_position(
                sym,
                candle,
                current_date,
                bar_idx,
                positions,
                nkd_series,
                bond_info,
                coupon_schedule,
                broker,
                portfolio_dv01,
            )
            if opened:
                portfolio_dv01 = self._compute_portfolio_dv01(
                    positions,
                    current_date,
                    candle_lookup,
                    nkd_series,
                    bond_info,
                    coupon_schedule,
                )

            if len(positions) >= cfg.max_positions:
                break

    def _try_open_position(
        self,
        sym: str,
        candle: Candle,
        current_date: date,
        bar_idx: int,
        positions: dict[str, BondPosition],
        nkd_series: dict[str, dict[date, Decimal]] | None,
        bond_info: dict[str, BondInfo],
        coupon_schedule: dict[str, list[CouponPayment]],
        broker: BondSimulatedBroker,
        portfolio_dv01: Decimal,
    ) -> bool:
        """Try to size and open a new bond position. Returns True if filled."""
        cfg = self._config
        info = bond_info.get(sym)
        if info is None:
            return False

        current_clean_pct = candle.close
        current_nkd = self._get_nkd(sym, current_date, nkd_series, coupon_schedule, info)

        try:
            current_ytm = ytm(
                current_clean_pct,
                info.coupon_rate,
                cfg.face_value,
                info.coupon_frequency,
                current_date,
                info.maturity_date,
            )
            mod_dur = modified_duration(
                current_ytm,
                info.coupon_rate,
                cfg.face_value,
                info.coupon_frequency,
                current_date,
                info.maturity_date,
            )
            dirty_px = dirty_price(current_clean_pct, current_nkd, cfg.face_value)
            bond_dv01 = dv01(mod_dur, dirty_px)
        except (ValueError, ZeroDivisionError):
            return False

        # Get portfolio value for sizing
        pv_prices: dict[str, Decimal] = {}
        pv_nkds: dict[str, Decimal] = {}
        for s in [*list(positions), sym]:
            s_info = bond_info.get(s)
            if s_info is None:
                continue
            pv_prices[s] = current_clean_pct  # use current candle as proxy
            pv_nkds[s] = self._get_nkd(s, current_date, nkd_series, coupon_schedule, s_info)

        layer_equity = broker.portfolio_value_at(pv_prices, pv_nkds)

        quantity = cfg.dv01_sizer.compute_position_size(
            layer_equity=layer_equity,
            bond_dv01_per_unit=bond_dv01,
            current_portfolio_dv01=portfolio_dv01,
            unit_cost=dirty_px,
        )

        if quantity <= 0:
            return False

        cost = bond_total_cost(
            cfg.transaction_costs,
            current_clean_pct,
            cfg.face_value,
            Decimal(quantity),
            sym,
        )

        filled = broker.buy_bond(sym, quantity, current_clean_pct, current_nkd, cost)
        if filled:
            positions[sym] = BondPosition(
                symbol=sym,
                quantity=quantity,
                entry_clean_pct=current_clean_pct,
                entry_nkd=current_nkd,
                entry_ytm_pct=current_ytm,
                entry_bar_idx=bar_idx,
                entry_date=current_date,
            )
            logger.debug(
                "bond_position_opened",
                symbol=sym,
                quantity=quantity,
                clean_pct=str(current_clean_pct),
                ytm_pct=str(current_ytm),
                bar_idx=bar_idx,
            )
        return filled

    def _compute_equity(
        self,
        positions: dict[str, BondPosition],
        current_date: date,
        candle_lookup: dict[str, dict[date, Candle]],
        nkd_series: dict[str, dict[date, Decimal]] | None,
        bond_info: dict[str, BondInfo],
        coupon_schedule: dict[str, list[CouponPayment]],
        broker: BondSimulatedBroker,
    ) -> Decimal:
        """Compute portfolio equity at dirty prices for current bar."""
        prices: dict[str, Decimal] = {}
        nkd_vals: dict[str, Decimal] = {}
        for sym in positions:
            c = candle_lookup.get(sym, {}).get(current_date)
            if c:
                prices[sym] = c.close
                info = bond_info.get(sym)
                if info:
                    nkd_vals[sym] = self._get_nkd(
                        sym, current_date, nkd_series, coupon_schedule, info
                    )
        return broker.portfolio_value_at(prices, nkd_vals)

    @staticmethod
    def _get_current_regime(
        current_date: date,
        macro_provider: MacroContextProvider | None,
    ) -> int:
        """Get CBR regime as int for regime-adaptive yield stops."""
        if macro_provider is None:
            return int(CBRRegime.NEUTRAL)
        snapshot = macro_provider.get_snapshot(current_date)
        if (
            snapshot.key_rate is None
            or snapshot.ruonia_7d_avg is None
            or snapshot.cpi_yoy is None
            or snapshot.last_cbr_decision is None
        ):
            return int(CBRRegime.NEUTRAL)
        return int(
            classify_regime(
                key_rate=snapshot.key_rate,
                ruonia_7d_avg=snapshot.ruonia_7d_avg,
                cpi_yoy_latest_published=snapshot.cpi_yoy,
                last_cbr_decision=snapshot.last_cbr_decision,
            )
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _candle_date(candle: Candle) -> date:
        """Extract date from candle timestamp."""
        return candle.timestamp.date()

    def _build_date_index(self, candles_by_symbol: dict[str, list[Candle]]) -> list[date]:
        """Build sorted union of all trading dates across symbols."""
        all_dates: set[date] = set()
        for candles in candles_by_symbol.values():
            for c in candles:
                all_dates.add(self._candle_date(c))
        return sorted(all_dates)

    def _get_nkd(
        self,
        symbol: str,
        current_date: date,
        nkd_series: dict[str, dict[date, Decimal]] | None,
        coupon_schedule: dict[str, list[CouponPayment]],
        bond_info: BondInfo | None,
    ) -> Decimal:
        """Get NKD for a symbol on a given date.

        Uses pre-computed series if available, otherwise estimates from
        coupon schedule.
        """
        if nkd_series is not None:
            sym_nkd = nkd_series.get(symbol, {})
            if current_date in sym_nkd:
                return sym_nkd[current_date]

        # Estimate from coupon schedule
        coupons = coupon_schedule.get(symbol, [])
        if not coupons or bond_info is None:
            return Decimal(0)

        return self._estimate_nkd(current_date, coupons)

    @staticmethod
    def _estimate_nkd(
        current_date: date,
        coupon_schedule: list[CouponPayment],
    ) -> Decimal:
        """Estimate NKD from coupon schedule when API data unavailable."""
        # Find the last coupon on or before current_date
        past_coupons = [c for c in coupon_schedule if c.coupon_date < current_date]
        if not past_coupons:
            return Decimal(0)
        last_coupon = past_coupons[-1]

        # Find next coupon
        future_coupons = [c for c in coupon_schedule if c.coupon_date >= current_date]
        if not future_coupons:
            return Decimal(0)
        next_coupon = future_coupons[0]

        days_since = (current_date - last_coupon.coupon_date).days
        period_days = (next_coupon.coupon_date - last_coupon.coupon_date).days
        if period_days <= 0:
            return Decimal(0)

        return nkd(next_coupon.amount_per_bond, days_since, period_days)

    def _compute_portfolio_dv01(
        self,
        positions: dict[str, BondPosition],
        current_date: date,
        candle_lookup: dict[str, dict[date, Candle]],
        nkd_series: dict[str, dict[date, Decimal]] | None,
        bond_info: dict[str, BondInfo],
        coupon_schedule: dict[str, list[CouponPayment]],
    ) -> Decimal:
        """Sum DV01 across all open bond positions."""
        cfg = self._config
        total_dv01 = Decimal(0)

        for sym, pos in positions.items():
            candle = candle_lookup.get(sym, {}).get(current_date)
            info = bond_info.get(sym)
            if candle is None or info is None:
                continue

            current_clean_pct = candle.close
            current_nkd = self._get_nkd(sym, current_date, nkd_series, coupon_schedule, info)

            try:
                current_ytm = ytm(
                    current_clean_pct,
                    info.coupon_rate,
                    cfg.face_value,
                    info.coupon_frequency,
                    current_date,
                    info.maturity_date,
                )
                mod_dur = modified_duration(
                    current_ytm,
                    info.coupon_rate,
                    cfg.face_value,
                    info.coupon_frequency,
                    current_date,
                    info.maturity_date,
                )
                dirty_px = dirty_price(current_clean_pct, current_nkd, cfg.face_value)
                bond_dv01 = dv01(mod_dur, dirty_px)
            except (ValueError, ZeroDivisionError):
                continue

            total_dv01 += bond_dv01 * pos.quantity

        return total_dv01

    def _close_position(
        self,
        symbol: str,
        pos: BondPosition,
        current_date: date,
        bar_idx: int,
        candle_lookup: dict[str, dict[date, Candle]],
        nkd_series: dict[str, dict[date, Decimal]] | None,
        broker: BondSimulatedBroker,
        bond_info: dict[str, BondInfo],
        coupon_schedule: dict[str, list[CouponPayment]],
    ) -> TradeResult | None:
        """Close a bond position and return TradeResult."""
        candle = candle_lookup.get(symbol, {}).get(current_date)
        if candle is None:
            return None

        info = bond_info.get(symbol)
        exit_clean_pct = candle.close
        exit_nkd = (
            self._get_nkd(symbol, current_date, nkd_series, coupon_schedule, info)
            if info
            else Decimal(0)
        )

        cost = bond_total_cost(
            self._config.transaction_costs,
            exit_clean_pct,
            self._config.face_value,
            Decimal(pos.quantity),
            symbol,
        )

        broker.sell_bond(symbol, pos.quantity, exit_clean_pct, exit_nkd, cost)

        # Compute PnL
        entry_dirty = dirty_price(pos.entry_clean_pct, pos.entry_nkd, self._config.face_value)
        exit_dirty = dirty_price(exit_clean_pct, exit_nkd, self._config.face_value)
        price_pnl = (exit_dirty - entry_dirty) * pos.quantity
        total_pnl = price_pnl + pos.coupon_income - cost
        entry_value = entry_dirty * pos.quantity
        pnl_pct = total_pnl / entry_value if entry_value > 0 else Decimal(0)

        return TradeResult(
            signal_id=uuid4(),
            symbol=symbol,
            side="SELL",
            quantity=Decimal(pos.quantity),
            entry_price=entry_dirty,
            exit_price=exit_dirty,
            pnl=total_pnl,
            pnl_pct=pnl_pct,
            hold_bars=bar_idx - pos.entry_bar_idx,
            coupon_income=pos.coupon_income,
            instrument_type="bond",
        )

    def _build_result(
        self,
        trades: list[TradeResult],
        equity_curve: list[Decimal],
        dates: list[date],
        broker: BondSimulatedBroker,
        initial_cash: Decimal,
        *,
        rotation_active: bool = False,
    ) -> BondBacktestResult:
        """Compute final metrics from trades and equity curve."""
        if not equity_curve:
            return self._empty_result(rotation_active=rotation_active)

        final_equity = equity_curve[-1]
        total_return_pct = (
            (final_equity - initial_cash) / initial_cash * _HUNDRED
            if initial_cash > 0
            else Decimal(0)
        )

        max_dd_pct = self._compute_max_drawdown(equity_curve)
        sharpe = self._compute_sharpe(equity_curve)
        win_rate = self._compute_win_rate(trades)
        profit_factor = self._compute_profit_factor(trades)

        return BondBacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            dates=dates,
            total_coupon_income_gross=broker.coupon_income_gross,
            total_coupon_income_net=broker.coupon_income_net,
            total_tax_paid=broker.tax_paid,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            trade_count=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            ofz_rotation_active=rotation_active,
        )

    def _empty_result(self, *, rotation_active: bool = False) -> BondBacktestResult:
        """Return empty result when no data."""
        return BondBacktestResult(
            trades=[],
            equity_curve=[],
            dates=[],
            total_coupon_income_gross=Decimal(0),
            total_coupon_income_net=Decimal(0),
            total_tax_paid=Decimal(0),
            total_return_pct=Decimal(0),
            max_drawdown_pct=Decimal(0),
            sharpe_ratio=Decimal(0),
            trade_count=0,
            win_rate=Decimal(0),
            profit_factor=Decimal(0),
            ofz_rotation_active=rotation_active,
        )

    @staticmethod
    def _compute_max_drawdown(equity_curve: list[Decimal]) -> Decimal:
        """Compute max peak-to-trough drawdown as a positive percentage."""
        if not equity_curve:
            return Decimal(0)

        peak = equity_curve[0]
        max_dd = Decimal(0)

        for equity in equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                dd = (peak - equity) / peak * _HUNDRED
                max_dd = max(max_dd, dd)

        return max_dd

    @staticmethod
    def _compute_sharpe(equity_curve: list[Decimal]) -> Decimal:
        """Compute annualised Sharpe ratio from equity curve daily returns."""
        if len(equity_curve) < _MIN_EQUITY_CURVE_FOR_SHARPE:
            return Decimal(0)

        # Compute daily returns
        returns: list[float] = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]
            if prev > 0:
                ret = float((equity_curve[i] - prev) / prev)
                returns.append(ret)

        if len(returns) < _MIN_RETURNS_FOR_SHARPE:
            return Decimal(0)

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_ret = variance**0.5

        if std_ret < _STD_EPSILON:
            return Decimal(0)

        sharpe = (mean_ret / std_ret) * (_TRADING_DAYS_PER_YEAR**0.5)
        return Decimal(str(round(sharpe, 4)))

    @staticmethod
    def _compute_win_rate(trades: list[TradeResult]) -> Decimal:
        """Compute win rate as fraction of winning trades."""
        if not trades:
            return Decimal(0)

        winners = sum(1 for t in trades if t.pnl > 0)
        return Decimal(str(round(winners / len(trades), 4)))

    @staticmethod
    def _compute_profit_factor(trades: list[TradeResult]) -> Decimal:
        """Compute profit factor = gross profits / gross losses."""
        if not trades:
            return Decimal(0)

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = sum(abs(t.pnl) for t in trades if t.pnl < 0)

        if gross_loss == 0:
            return Decimal("Infinity") if gross_profit > 0 else Decimal(0)

        return Decimal(str(round(float(gross_profit / gross_loss), 4)))
