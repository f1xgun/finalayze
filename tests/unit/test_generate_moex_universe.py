"""Unit tests for scripts/generate_moex_universe.py (mocked SDK, no live network).

The generator is the single live-gRPC caller of Phase 65; these tests stub every
fetcher/coupon call so nothing touches the network. They prove:
  - required_symbols() derives from config/segments.py with the TCSG->T alias (UNIV-02 / Pitfall 2)
  - validate() refuses to write on ANY missing required symbol (UNIV-08 / D-04)
  - the coupon-rate derivation matches the OFZ formula (UNIV-06 / Pitfall 1)
  - the happy-path orchestrator writes a snapshot containing every required symbol
    with YTM-able traded-OFZ rows.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
import scripts.generate_moex_universe as gen

# ── Named constants (ruff PLR2004: no magic numbers in tests) ─────────────────
_PAY_ONE_BOND = Decimal("35.40")
_COUPON_QTY_PER_YEAR = 2
_NOMINAL = Decimal(1000)
_EXPECTED_COUPON_RATE = Decimal("7.08")  # 35.40 * 2 / 1000 * 100 (~hand-list 7.10)
_OFZ_LOT_SIZE = 1
_SHARE_LOT_SIZE = 10
_FACE_VALUE = Decimal(1000)
_SAMPLE_MISSING_SYMBOL = "SBER"

# A traded OFZ-PD (fixed) and OFZ-PK (floating) used by the happy-path fixtures.
_OFZ_PD_SYMBOL = "SU26239RMFS2"
_OFZ_PD_FIGI = "BBG011FHF1F7"
_OFZ_PK_SYMBOL = "SU29007RMFS0"
_OFZ_PK_FIGI = "BBG007Z5DF79"

# A HYPOTHETICAL new floating OFZ-PK that is traded but NOT yet in the hand-list.
# Used to exercise the WR-02/WR-03 unknown-floater paths (no live impact today --
# all 4 current floaters are in _OFZ_PK_HANDLIST_RATE).
_UNKNOWN_FLOATER_SYMBOL = "SU29099RMFS0"
_UNKNOWN_FLOATER_FIGI = "BBG00UNKNOWNPK"


class _StubCoupon:
    """Minimal stand-in for a CouponPayment with amount_per_bond."""

    def __init__(self, amount: Decimal) -> None:
        self.amount_per_bond = amount


class _StubFetcher:
    """Stubs the TinkoffFetcher surface the generator calls. No live network."""

    def __init__(self, rows_by_class: dict[str, list[dict[str, object]]]) -> None:
        self._rows = rows_by_class
        self.closed = False

    def fetch_all_shares(self) -> list[dict[str, object]]:
        return self._rows.get("shares", [])

    def fetch_all_etfs(self) -> list[dict[str, object]]:
        return self._rows.get("etfs", [])

    def fetch_all_bonds(self) -> list[dict[str, object]]:
        return self._rows.get("bonds", [])

    def fetch_all_futures(self) -> list[dict[str, object]]:
        return self._rows.get("futures", [])

    def fetch_all_currencies(self) -> list[dict[str, object]]:
        return self._rows.get("currencies", [])

    def close(self) -> None:
        self.closed = True


def _share_row(ticker: str, figi: str) -> dict[str, object]:
    return {
        "figi": figi,
        "ticker": ticker,
        "isin": f"ISIN{ticker}",
        "class_code": "TQBR",
        "name": ticker,
        "lot": _SHARE_LOT_SIZE,
        "currency": "rub",  # lowercase on purpose (Pitfall 3 normalization)
        "asset_uid": f"uid-{ticker}",
        "first_1day_candle_date": None,
    }


def _bond_row(ticker: str, figi: str, *, floating: bool) -> dict[str, object]:
    return {
        "figi": figi,
        "ticker": ticker,
        "isin": f"ISIN{ticker}",
        "name": ticker,
        "lot": _OFZ_LOT_SIZE,
        "currency": "rub",
        "nominal": _NOMINAL,
        "coupon_quantity_per_year": _COUPON_QTY_PER_YEAR,
        "maturity_date": "2031-07-23",
        "floating_coupon_flag": floating,
        "class_code": "TQOB",
    }


def _complete_rows() -> dict[str, list[dict[str, object]]]:
    """Build mock SDK rows covering EVERY required symbol + the traded OFZ."""
    req = gen.required_symbols()
    shares = [_share_row(sym, f"FIGI{sym}") for sym in sorted(req) if not sym.startswith("SU")]
    bonds = [
        _bond_row(sym, f"FIGI{sym}", floating=sym.startswith("SU29"))
        for sym in sorted(req)
        if sym.startswith("SU")
    ]
    # Ensure the two FIGIs the coupon stub keys on are present verbatim.
    bonds.append(_bond_row(_OFZ_PD_SYMBOL, _OFZ_PD_FIGI, floating=False))
    bonds.append(_bond_row(_OFZ_PK_SYMBOL, _OFZ_PK_FIGI, floating=True))
    return {
        "shares": shares,
        "etfs": [],
        "bonds": bonds,
        "futures": [],
        "currencies": [],
    }


def _coupon_lookup(_figi: str) -> list[_StubCoupon]:
    return [_StubCoupon(_PAY_ONE_BOND)]


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_alias_tcsg_to_t() -> None:
    """required_symbols() reconciles TCSG->T and never emits TCSG (Pitfall 2)."""
    req = gen.required_symbols()
    assert "T" in req
    assert "TCSG" not in req
    assert gen._ALIAS["TCSG"] == "T"


def test_rejects_missing() -> None:
    """validate() raises SystemExit (REFUSING ...) when a required symbol is absent (UNIV-08)."""
    complete = gen.required_symbols()
    incomplete = complete - {_SAMPLE_MISSING_SYMBOL}
    with pytest.raises(SystemExit) as exc_info:
        gen.validate(incomplete)
    assert str(exc_info.value).startswith("REFUSING to write snapshot")
    assert _SAMPLE_MISSING_SYMBOL in str(exc_info.value)


def test_accepts_complete() -> None:
    """validate() returns None when the snapshot set is a superset of required (no raise)."""
    superset = gen.required_symbols() | {"EXTRA1", "EXTRA2"}
    assert gen.validate(superset) is None


def test_coupon_rate_formula() -> None:
    """derive_coupon_rate computes pay_one_bond * qty / nominal * 100 (UNIV-06)."""
    rate = gen.derive_coupon_rate(_PAY_ONE_BOND, _COUPON_QTY_PER_YEAR, _NOMINAL)
    assert rate == _EXPECTED_COUPON_RATE


def test_happy_path_write(tmp_path: Path) -> None:
    """Orchestrator writes a JSON snapshot with every required symbol + YTM-able OFZ rows."""
    fetcher = _StubFetcher(_complete_rows())
    out = tmp_path / "moex_universe.json"

    gen.build_and_write(
        fetcher=fetcher,
        coupon_lookup=_coupon_lookup,
        out_path=out,
        dry_run=False,
    )

    raw = json.loads(out.read_text(encoding="utf-8"))
    assert "generated_at" in raw
    assert "sdk_universe_counts" in raw
    symbols = {row["symbol"] for row in raw["instruments"]}
    assert gen.required_symbols() <= symbols

    # Currency normalized to upper-case (Pitfall 3).
    assert all(row["currency"] == "RUB" for row in raw["instruments"])

    # Every traded OFZ row is YTM-able (non-None coupon fields) (UNIV-06).
    traded_ofz = gen.traded_ofz_symbols()
    ofz_rows = [r for r in raw["instruments"] if r["symbol"] in traded_ofz]
    assert ofz_rows
    for row in ofz_rows:
        assert row["coupon_rate"] is not None
        assert row["coupon_frequency"] is not None
        assert row["face_value"] is not None
        assert row["maturity_date"] is not None


def test_happy_path_dry_run_writes_nothing(tmp_path: Path) -> None:
    """--dry-run validates and counts but does NOT write the snapshot file."""
    fetcher = _StubFetcher(_complete_rows())
    out = tmp_path / "moex_universe.json"

    gen.build_and_write(
        fetcher=fetcher,
        coupon_lookup=_coupon_lookup,
        out_path=out,
        dry_run=True,
    )

    assert not out.exists()


# ── WR-02: unknown traded floater derives NO fixed coupon rate ──────────────────


def test_unknown_floater_derives_no_fixed_rate() -> None:
    """A traded floating OFZ outside the hand-list leaves coupon_rate None (WR-02).

    The derive branch is gated on `not floating`, so an unknown floater must NOT
    back-compute a (misleading, constant) fixed rate from a single coupon. The
    coupon_lookup would return a payment if reached -- proving it is NOT reached.
    """
    row = _bond_row(_UNKNOWN_FLOATER_SYMBOL, _UNKNOWN_FLOATER_FIGI, floating=True)
    traded = {_UNKNOWN_FLOATER_SYMBOL}

    out = gen._bond_row(row, _coupon_lookup, traded)

    assert out["floating_coupon"] is True
    assert out["coupon_rate"] is None  # WR-02: no fixed rate fabricated for a floater


def test_known_floater_uses_handlist_rate() -> None:
    """A traded floater IN the hand-list still gets its RUONIA spread (behavior-preserving)."""
    row = _bond_row(_OFZ_PK_SYMBOL, _OFZ_PK_FIGI, floating=True)
    traded = {_OFZ_PK_SYMBOL}

    out = gen._bond_row(row, _coupon_lookup, traded)

    assert out["coupon_rate"] == str(gen._OFZ_PK_HANDLIST_RATE[_OFZ_PK_SYMBOL])


# ── WR-03: unknown traded floater trips the targeted A3 error ────────────────────


def test_unknown_traded_floater_refuses_with_a3_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """_assert_ofz_yieldable raises a targeted A3 SystemExit for an unknown floater (WR-03)."""
    monkeypatch.setattr(gen, "traded_ofz_symbols", lambda: {_UNKNOWN_FLOATER_SYMBOL})
    rows = [
        {
            "symbol": _UNKNOWN_FLOATER_SYMBOL,
            "figi": _UNKNOWN_FLOATER_FIGI,
            "floating_coupon": True,
            "coupon_rate": None,
            "coupon_frequency": _COUPON_QTY_PER_YEAR,
            "face_value": str(_FACE_VALUE),
            "maturity_date": "2031-07-23",
        }
    ]

    with pytest.raises(SystemExit) as exc_info:
        gen._assert_ofz_yieldable(rows)

    message = str(exc_info.value)
    assert _UNKNOWN_FLOATER_SYMBOL in message
    assert "_OFZ_PK_HANDLIST_RATE" in message
    assert "A3" in message


# ── WR-05: a single bond's coupon-lookup failure does not abort enumeration ─────


def test_coupon_lookup_failure_isolated_per_bond() -> None:
    """One bond's gRPC failure logs and yields [] instead of aborting build_rows (WR-05)."""

    class _BoomFetcher:
        """Stub whose _run_async raises -- simulates a transient gRPC error per bond."""

        def _get_services_async(self) -> object:  # pragma: no cover - never awaited
            raise AssertionError("should not be reached in the failure path")

        def _money_to_decimal(self, _m: object) -> Decimal:  # pragma: no cover
            return Decimal(0)

        def _run_async(self, _coro: object) -> list[object]:
            # Close the un-awaited coroutine to avoid a RuntimeWarning, then fail.
            if hasattr(_coro, "close"):
                _coro.close()
            raise RuntimeError("transient gRPC timeout")

    lookup = gen._live_coupon_lookup(_BoomFetcher())

    # The failure is swallowed: returns [] rather than propagating (which would abort
    # the whole 1500+ bond enumeration in build_rows).
    assert lookup(_UNKNOWN_FLOATER_FIGI) == []
