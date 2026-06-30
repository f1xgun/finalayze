"""Live geopolitical-risk alert (forward-only, advisory).

Aggregates the live news/sentiment store across the MOEX bellwethers, computes the
geopolitical-risk assessment, prints it, and — when the level is ELEVATED/HIGH —
sends a Telegram alert via ``scripts/notify_telegram.py`` (fail-soft).

ADVISORY ONLY: it recommends trimming the equity sleeve toward the deposit/OFZ
anchor; it does NOT change allocations or place orders. Real money is a hard stop.
Cannot be backtested (no historical sentiment) — it is a live risk-awareness aid.

    uv run python scripts/geo_risk_alert.py            # print only
    uv run python scripts/geo_risk_alert.py --notify   # also Telegram on ELEVATED+

Cron example (weekday mornings):
    13 6 * * 1-5  cd /path/to/finalayze && uv run python scripts/geo_risk_alert.py --notify
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from finalayze.analysis.geopolitical_risk import (
    GeoRiskAssessment,
    GeoRiskLevel,
)
from finalayze.orchestration.geo_risk_monitor import MOEX_BELLWETHERS, assess_live

_NOTIFY = Path(__file__).resolve().parent / "notify_telegram.py"


async def _assess() -> tuple[GeoRiskAssessment, str]:
    """Return (assessment, note). Fail-soft to a NORMAL no-data result if the DB is down."""
    from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
    from finalayze.data.sentiment_store import SentimentStore  # noqa: PLC0415

    try:
        store = SentimentStore(get_async_session_factory())
        assessment = await assess_live(store, MOEX_BELLWETHERS)
    except Exception as exc:
        return (
            GeoRiskAssessment(
                level=GeoRiskLevel.NORMAL,
                score=0.0,
                recommended_equity_trim_pct=Decimal(0),
                recommended_fx_hedge_pct=Decimal(0),
            ),
            f"no live sentiment data ({type(exc).__name__}); reporting NORMAL",
        )
    return assessment, "ok"


def _notify(level: GeoRiskLevel, body: str) -> None:
    priority = "high" if level is GeoRiskLevel.HIGH else "normal"
    try:
        subprocess.run(  # noqa: S603 — fixed local script, no shell
            [
                sys.executable,
                str(_NOTIFY),
                "--title",
                f"Geopolitical risk: {level.value.upper()}",
                "--body",
                body,
                "--priority",
                priority,
            ],
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # fail-soft
        print(f"telegram notify failed (skipped): {exc}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notify", action="store_true", help="send Telegram on ELEVATED/HIGH")
    args = ap.parse_args(argv)

    assessment, note = asyncio.run(_assess())
    level = assessment.level
    body = (
        f"level={level.value} score={assessment.score:.2f} "
        f"recommended_equity_trim={assessment.recommended_equity_trim_pct} "
        f"recommended_zo_fx_hedge={assessment.recommended_fx_hedge_pct}\n"
        f"drivers: {'; '.join(assessment.drivers) or 'none'}\n"
        f"{assessment.disclaimer}"
    )
    print(f"[{note}]")
    print(body)

    if args.notify and level is not GeoRiskLevel.NORMAL:
        _notify(level, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
