"""Tests for SectorTickerMapper (Layer 3)."""

from __future__ import annotations

from finalayze.analysis.sector_ticker_mapper import SECTOR_TICKERS, SectorTickerMapper


class TestSectorTickerMapper:
    """Tests for SectorTickerMapper.map_sectors() and all_tickers()."""

    def setup_method(self) -> None:
        self.mapper = SectorTickerMapper()

    def test_map_single_sector_oil_gas(self) -> None:
        result = self.mapper.map_sectors(["oil_gas"])
        assert result == ["ROSN", "LKOH", "TATN", "TATNP", "NVTK", "SNGS", "SNGSP"]

    def test_map_single_sector_banking(self) -> None:
        result = self.mapper.map_sectors(["banking"])
        assert result == ["SBER", "VTBR", "TCSG"]

    def test_map_unknown_sector_returns_empty(self) -> None:
        result = self.mapper.map_sectors(["unknown_sector"])
        assert result == []

    def test_map_empty_list_returns_empty(self) -> None:
        result = self.mapper.map_sectors([])
        assert result == []

    def test_map_multiple_sectors_combined_no_duplicates(self) -> None:
        result = self.mapper.map_sectors(["oil_gas", "banking"])
        assert "ROSN" in result
        assert "SBER" in result
        # No duplicates
        assert len(result) == len(set(result))

    def test_all_tickers_returns_frozenset(self) -> None:
        tickers = self.mapper.all_tickers()
        assert isinstance(tickers, frozenset)
        assert "SBER" in tickers
        assert "ROSN" in tickers
        assert "LKOH" in tickers

    def test_sector_tickers_has_14_sectors(self) -> None:
        expected_sectors = {
            "oil_gas",
            "banking",
            "metals_mining",
            "telecom",
            "tech",
            "utilities",
            "real_estate",
            "retail",
            "transport",
            "fertilizers",
            "conglomerate",
            "exchange",
            "bonds_fixed",
            "bonds_floating",
        }
        assert set(SECTOR_TICKERS.keys()) == expected_sectors
        assert len(SECTOR_TICKERS) == 14

    def test_all_tickers_count(self) -> None:
        """All tickers across all sectors (excluding empty bond sectors)."""
        all_t = self.mapper.all_tickers()
        # Count unique tickers from the constant
        expected = set()
        for tickers in SECTOR_TICKERS.values():
            expected.update(tickers)
        assert all_t == frozenset(expected)

    def test_bond_sectors_have_no_tickers(self) -> None:
        assert self.mapper.map_sectors(["bonds_fixed"]) == []
        assert self.mapper.map_sectors(["bonds_floating"]) == []

    def test_map_sectors_preserves_order(self) -> None:
        """Tickers appear in sector order, not random."""
        result = self.mapper.map_sectors(["banking", "oil_gas"])
        # Banking tickers first, then oil_gas
        banking_end = max(result.index(t) for t in ["SBER", "VTBR", "TCSG"])
        oil_start = min(result.index(t) for t in ["ROSN", "LKOH"])
        assert banking_end < oil_start
