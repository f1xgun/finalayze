"""Sandbox validation orchestration script.

Prints a step-by-step checklist for running the 5-day sandbox
validation. This is NOT an automated runner -- the actual run
happens via Docker Compose over 5 days.

Optionally queries the /health endpoint for current status.

Usage:
    python scripts/run_sandbox_validation.py
    python scripts/run_sandbox_validation.py --check-status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _print_preflight() -> None:
    """Print the pre-flight checklist section."""
    print("## PRE-FLIGHT CHECKLIST")
    print()

    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    mode = os.environ.get("FINALAYZE_MODE", "")
    print(
        f"  [{'x' if token else ' '}] FINALAYZE_TINKOFF_TOKEN is set"
        f" {'(detected)' if token else '(NOT SET -- required!)'}"
    )
    mode_label = mode or "unset"
    mode_detail = "(detected)" if mode == "sandbox" else f"(current: {mode_label})"
    print(f"  [{'x' if mode == 'sandbox' else ' '}] FINALAYZE_MODE=sandbox {mode_detail}")
    print()

    print("  [ ] Docker is running")
    print("      Verify: docker info")
    print()

    print("  [ ] Sandbox account funded with 1,000,000 RUB starting capital")
    print("      The 1M RUB matches Phase 1 capital assumption and backtested parameters.")
    print("      Fund via T-Invest sandbox API or web interface before starting.")
    print()

    print("  [ ] Only ru_* segments are active in config/segments.py")
    print("      MOEX-only validation -- disable us_tech, us_broad, us_finance, us_healthcare.")
    print("      Verify: grep 'enabled.*True' config/segments.py")
    print()

    print("  [ ] Bond cycle is enabled")
    print("      Both equity + bond cycles must run for full Phase 5 integration validation.")
    print("      Verify: bond_enabled=True in config or trading_loop settings.")
    print()


def _print_startup() -> None:
    """Print the startup section."""
    print("-" * 72)
    print()
    print("## STARTUP")
    print()
    print("  Start the sandbox stack:")
    print("    docker compose -f docker/docker-compose.sandbox.yml up -d")
    print()
    print("  Verify services are healthy:")
    print("    docker compose -f docker/docker-compose.sandbox.yml ps")
    print()


def _print_monitoring() -> None:
    """Print the monitoring section."""
    print("-" * 72)
    print()
    print("## MONITORING")
    print()
    print("  Grafana dashboard: http://localhost:3000")
    print("    Default credentials: admin / admin")
    print("    Look for: Finalayze Trading Dashboard")
    print()
    print("  API health: http://localhost:8000/health")
    print("    curl http://localhost:8000/health | python -m json.tool")
    print()
    print("  Cycle logs:")
    print(
        "    docker compose -f docker/docker-compose.sandbox.yml exec app "
        "cat results/validation/cycles.jsonl | tail -5"
    )
    print()


def _print_kill_test() -> None:
    """Print the kill test section."""
    print("-" * 72)
    print()
    print("## KILL TEST (Day 2-3)")
    print()
    print("  Simulate crash and verify recovery:")
    print("    docker kill finalayze-app")
    print("    # Wait 30s, Docker restart policy should bring it back")
    print("    docker compose -f docker/docker-compose.sandbox.yml ps")
    print("    # Verify /health returns ok and trading resumes")
    print("    curl http://localhost:8000/health")
    print()


def _print_completion() -> None:
    """Print the completion section."""
    print("-" * 72)
    print()
    print("## COMPLETION (After 5 days)")
    print()
    print("  1. Save Docker logs:")
    print(
        "     docker compose -f docker/docker-compose.sandbox.yml logs app "
        "> results/validation/docker_logs.txt"
    )
    print()
    print("  2. Generate validation report:")
    print("     python scripts/generate_validation_report.py")
    print()
    print("  3. Review report:")
    print("     cat results/validation/VALIDATION-REPORT.md")
    print()
    print("  4. Stop sandbox stack:")
    print("     docker compose -f docker/docker-compose.sandbox.yml down")
    print()
    print("=" * 72)


def print_checklist() -> None:
    """Print the pre-flight checklist and operational guide."""
    print("=" * 72)
    print("  FINALAYZE SANDBOX VALIDATION -- 5-DAY RUN CHECKLIST")
    print("=" * 72)
    print()
    _print_preflight()
    _print_startup()
    _print_monitoring()
    _print_kill_test()
    _print_completion()


def check_status() -> None:
    """Query the /health endpoint and print current status."""
    import httpx  # noqa: PLC0415  # lazy import -- optional dependency

    url = "http://localhost:8000/health"
    print(f"Checking health endpoint: {url}")
    print()

    try:
        resp = httpx.get(url, timeout=5.0)
        print(f"Status code: {resp.status_code}")
        print()
        data = resp.json()
        print(json.dumps(data, indent=2))
    except httpx.ConnectError:
        print("ERROR: Cannot connect to http://localhost:8000")
        print("Is the sandbox stack running?")
        print("  docker compose -f docker/docker-compose.sandbox.yml up -d")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sandbox validation orchestration -- 5-day run checklist."
    )
    parser.add_argument(
        "--check-status",
        action="store_true",
        help="Query /health endpoint and print current status",
    )
    args = parser.parse_args()

    if args.check_status:
        check_status()
    else:
        print_checklist()


if __name__ == "__main__":
    main()
