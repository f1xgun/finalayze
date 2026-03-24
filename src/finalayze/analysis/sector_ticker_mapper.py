"""Static sector-to-MOEX-ticker mapping (Layer 3).

Maps LLM-returned sector names to concrete MOEX ticker symbols.
No LLM needed -- pure static registry.
"""

from __future__ import annotations

SECTOR_TICKERS: dict[str, list[str]] = {
    "oil_gas": ["ROSN", "LKOH", "TATN", "TATNP", "NVTK", "SNGS", "SNGSP"],
    "banking": ["SBER", "VTBR", "TCSG"],
    "metals_mining": ["GMKN", "NLMK", "CHMF", "MAGN", "RUAL", "ALRS", "PLZL"],
    "telecom": ["MTSS"],
    "tech": ["YDEX", "OZON", "HHRU"],
    "utilities": ["IRAO", "MSNG", "HYDR"],
    "real_estate": ["PIKK"],
    "retail": ["MGNT"],
    "transport": ["AFLT", "TRNFP"],
    "fertilizers": ["PHOR"],
    "conglomerate": ["AFKS"],
    "exchange": ["MOEX"],
    "bonds_fixed": [],  # OFZ-PD -- no equity tickers
    "bonds_floating": [],  # OFZ-PK -- no equity tickers
}


class SectorTickerMapper:
    """Maps sector names from LLM output to MOEX ticker symbols."""

    def __init__(self) -> None:
        self._all_tickers: frozenset[str] = frozenset(
            t for tickers in SECTOR_TICKERS.values() for t in tickers
        )

    def map_sectors(self, sectors: list[str]) -> list[str]:
        """Return unique tickers for given sector names. Unknown sectors are silently skipped."""
        seen: set[str] = set()
        result: list[str] = []
        for sector in sectors:
            for ticker in SECTOR_TICKERS.get(sector, []):
                if ticker not in seen:
                    seen.add(ticker)
                    result.append(ticker)
        return result

    def all_tickers(self) -> frozenset[str]:
        """Return frozenset of all mapped tickers across all sectors."""
        return self._all_tickers
