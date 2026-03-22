"""Layer ledger -- tracks per-layer cash, positions, and drawdown (Layer 0).

Each portfolio layer operates as a virtual sub-account with:
- Own cash allocation
- Own position tracking (stocks as Decimal quantities, bonds as BondPositionRecord)
- Own peak-to-trough drawdown monitoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from finalayze.core.schemas import BondPositionRecord

if TYPE_CHECKING:
    from finalayze.api.alerts import TelegramAlerter
    from finalayze.core.models import LayerLedgerModel
    from finalayze.core.schemas import PortfolioState
    from finalayze.markets.instruments import InstrumentRegistry

_log = structlog.get_logger()


@dataclass
class LayerLedger:
    """Mutable ledger for a single portfolio layer.

    Tracks cash, positions, equity, and peak-to-trough drawdown.
    ``bond_positions`` stores :class:`BondPositionRecord` objects keyed by symbol,
    separate from the plain ``positions`` dict used for stock quantities.
    """

    layer_id: str  # PortfolioLayer value
    cash: Decimal
    positions: dict[str, Decimal] = field(default_factory=dict)  # symbol -> quantity
    bond_positions: dict[str, BondPositionRecord] = field(default_factory=dict)
    peak_equity: Decimal = Decimal(0)
    current_equity: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if self.peak_equity == 0:
            self.peak_equity = self.cash
        if self.current_equity == 0:
            self.current_equity = self.cash

    def update_equity(self, equity: Decimal) -> None:
        """Update current equity and peak tracking."""
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    @property
    def drawdown_pct(self) -> Decimal:
        """Current peak-to-trough drawdown as decimal fraction."""
        if self.peak_equity <= 0:
            return Decimal(0)
        return (self.peak_equity - self.current_equity) / self.peak_equity

    @property
    def is_empty(self) -> bool:
        """True if no positions held."""
        return not self.positions or all(q == 0 for q in self.positions.values())

    def add_position(self, symbol: str, quantity: Decimal) -> None:
        """Add to a position (or create new)."""
        current = self.positions.get(symbol, Decimal(0))
        self.positions[symbol] = current + quantity

    def remove_position(self, symbol: str, quantity: Decimal) -> None:
        """Reduce a position. Removes entry if quantity reaches 0."""
        current = self.positions.get(symbol, Decimal(0))
        new_qty = current - quantity
        if new_qty <= 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_qty

    def debit_cash(self, amount: Decimal) -> bool:
        """Debit cash. Returns False if insufficient."""
        if amount > self.cash:
            return False
        self.cash -= amount
        return True

    def credit_cash(self, amount: Decimal) -> None:
        """Credit cash."""
        self.cash += amount

    # ── Bond position management ──────────────────────────────────────────

    def add_bond_position(self, record: BondPositionRecord) -> None:
        """Add a bond position record.

        If the symbol already exists, creates a new record with summed quantity
        but preserves the original entry conditions.
        """
        existing = self.bond_positions.get(record.symbol)
        if existing is not None:
            # Merge: keep original entry data, sum quantities
            merged = BondPositionRecord(
                symbol=record.symbol,
                quantity=existing.quantity + record.quantity,
                entry_ytm_pct=existing.entry_ytm_pct,
                entry_date=existing.entry_date,
                entry_price=existing.entry_price,
                entry_clean_pct=existing.entry_clean_pct,
                layer_id=existing.layer_id,
            )
            self.bond_positions[record.symbol] = merged
        else:
            self.bond_positions[record.symbol] = record

    def remove_bond_position(self, symbol: str, quantity: Decimal) -> None:
        """Reduce or remove a bond position by quantity."""
        existing = self.bond_positions.get(symbol)
        if existing is None:
            return
        new_qty = existing.quantity - quantity
        if new_qty <= 0:
            self.bond_positions.pop(symbol, None)
        else:
            self.bond_positions[symbol] = BondPositionRecord(
                symbol=existing.symbol,
                quantity=new_qty,
                entry_ytm_pct=existing.entry_ytm_pct,
                entry_date=existing.entry_date,
                entry_price=existing.entry_price,
                entry_clean_pct=existing.entry_clean_pct,
                layer_id=existing.layer_id,
            )

    # ── ORM persistence ───────────────────────────────────────────────────

    def to_orm_rows(self) -> list[LayerLedgerModel]:
        """Convert bond positions to ORM model instances for DB persistence."""
        from finalayze.core.models import LayerLedgerModel as LedgerModel  # noqa: PLC0415

        now = datetime.now(tz=UTC)
        return [
            LedgerModel(
                layer_id=self.layer_id,
                symbol=record.symbol,
                quantity=record.quantity,
                entry_ytm_pct=record.entry_ytm_pct,
                entry_price=record.entry_price,
                entry_clean_pct=record.entry_clean_pct,
                entry_date=datetime.combine(record.entry_date, datetime.min.time(), tzinfo=UTC),
                updated_at=now,
            )
            for record in self.bond_positions.values()
        ]

    @classmethod
    def from_orm_rows(
        cls,
        layer_id: str,
        cash: Decimal,
        rows: list[LayerLedgerModel],
    ) -> LayerLedger:
        """Restore a LayerLedger from ORM rows."""
        ledger = cls(layer_id=layer_id, cash=cash)
        for row in rows:
            entry_date = row.entry_date.date()
            record = BondPositionRecord(
                symbol=row.symbol,
                quantity=row.quantity,
                entry_ytm_pct=row.entry_ytm_pct,
                entry_date=entry_date,
                entry_price=row.entry_price,
                entry_clean_pct=row.entry_clean_pct,
                layer_id=layer_id,
            )
            ledger.bond_positions[row.symbol] = record
        return ledger


def _extract_broker_bonds(
    portfolio: PortfolioState,
    registry: InstrumentRegistry,
) -> dict[str, Decimal]:
    """Extract bond positions from broker portfolio (FIGI -> symbol mapping)."""
    broker_bonds: dict[str, Decimal] = {}
    for figi, qty in portfolio.positions.items():
        try:
            instrument = registry.get_by_figi(figi)
        except Exception:
            _log.warning("reconcile_unknown_figi", figi=figi)
            continue
        if getattr(instrument, "instrument_type", "") == "bond":
            broker_bonds[instrument.symbol] = qty
    return broker_bonds


def _collect_ledger_bonds(
    ledgers: dict[str, LayerLedger],
) -> dict[str, tuple[str, BondPositionRecord]]:
    """Collect all bond positions across ledger layers: symbol -> (layer_id, record)."""
    result: dict[str, tuple[str, BondPositionRecord]] = {}
    for layer_id, ledger in ledgers.items():
        for symbol, record in ledger.bond_positions.items():
            result[symbol] = (layer_id, record)
    return result


def reconcile_with_broker(
    portfolio: PortfolioState,
    ledgers: dict[str, LayerLedger],
    registry: InstrumentRegistry,
    alerter: TelegramAlerter | None = None,
) -> list[str]:
    """Reconcile broker portfolio against persisted ledger state.

    Filters bond positions from the broker portfolio (via registry),
    compares against ledger bond_positions across all layers, and:
    - Unknown positions: added to Core layer with alert
    - Quantity mismatch: trusts broker, updates ledger with alert
    - Missing positions (in ledger but not broker): removed with alert

    Args:
        portfolio: Broker portfolio state (FIGI-keyed positions).
        ledgers: Dict of layer_id -> LayerLedger.
        registry: InstrumentRegistry for FIGI -> symbol lookup.
        alerter: Optional TelegramAlerter for sending discrepancy alerts.

    Returns:
        List of alert message strings describing all discrepancies found.
    """
    alerts: list[str] = []
    broker_bonds = _extract_broker_bonds(portfolio, registry)
    ledger_bonds = _collect_ledger_bonds(ledgers)
    core_ledger = ledgers.get("core")

    # Check broker positions against ledger
    for symbol, broker_qty in broker_bonds.items():
        if symbol in ledger_bonds:
            layer_id, record = ledger_bonds[symbol]
            if record.quantity != broker_qty:
                msg = (
                    f"Reconciliation mismatch: {symbol} ledger={record.quantity} "
                    f"broker={broker_qty} (trusting broker, layer={layer_id})"
                )
                alerts.append(msg)
                updated = BondPositionRecord(
                    symbol=record.symbol,
                    quantity=broker_qty,
                    entry_ytm_pct=record.entry_ytm_pct,
                    entry_date=record.entry_date,
                    entry_price=record.entry_price,
                    entry_clean_pct=record.entry_clean_pct,
                    layer_id=record.layer_id,
                )
                ledgers[layer_id].bond_positions[symbol] = updated
        elif core_ledger is not None:
            msg = (
                f"Reconciliation: unknown bond {symbol} qty={broker_qty} "
                f"found in broker, adding to Core layer"
            )
            alerts.append(msg)
            core_ledger.bond_positions[symbol] = BondPositionRecord(
                symbol=symbol,
                quantity=broker_qty,
                entry_ytm_pct=Decimal(0),
                entry_date=datetime.now(tz=UTC).date(),
                entry_price=Decimal(0),
                entry_clean_pct=Decimal(0),
                layer_id="core",
            )

    # Check ledger positions not in broker (missing)
    for symbol, (layer_id, _record) in ledger_bonds.items():
        if symbol not in broker_bonds:
            msg = (
                f"Reconciliation: {symbol} in ledger (layer={layer_id}) "
                f"but not in broker portfolio, removing"
            )
            alerts.append(msg)
            ledgers[layer_id].bond_positions.pop(symbol, None)

    # Send alerts via Telegram if alerter provided
    if alerter is not None:
        for msg in alerts:
            alerter.on_error("reconciliation", msg)

    return alerts
