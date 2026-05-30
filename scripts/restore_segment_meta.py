#!/usr/bin/env python3
"""Regenerate ``segment_meta.json`` for a model segment (audit #4 / S6.3).

When a segment directory under ``models/`` has model artefacts but the
``segment_meta.json`` sidecar has been lost (corrupted disk, mis-merged
branch, manual cleanup), this helper rebuilds the file so the loader's
feature-schema version guard works again on next boot.

Usage::

    uv run python scripts/restore_segment_meta.py \\
        --models-dir models \\
        --segment ru_energy \\
        --base-rate 0.4923

The ``--base-rate`` argument is the empirical positive-class rate from
the training labels; the operator obtains it from the most recent training
log or by running the labeller on calibration candles. When omitted the
helper writes ``base_rate=null`` and the consuming code falls back to
loader defaults.

``feature_schema_version`` is taken from the in-repo loader constant so
that the restored file matches the running code — if the operator wants
to gate-fail on purpose, they can pass ``--schema-version`` explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _build_meta(*, base_rate: float | None, schema_version: int) -> dict[str, object]:
    meta: dict[str, object] = {"feature_schema_version": schema_version}
    if base_rate is not None:
        meta["base_rate"] = base_rate
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Path to the ``models/`` root (e.g. ``models`` or ``./models``)",
    )
    parser.add_argument(
        "--segment",
        required=True,
        help="Segment ID matching the subdirectory name under --models-dir",
    )
    parser.add_argument(
        "--base-rate",
        type=float,
        default=None,
        help="Empirical positive-class rate from training (0.0-1.0). "
        "Omit to write ``base_rate=null``.",
    )
    parser.add_argument(
        "--schema-version",
        type=int,
        default=None,
        help="Override feature_schema_version. Defaults to the in-repo "
        "``finalayze.ml.loader.FEATURE_SCHEMA_VERSION``.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing segment_meta.json. Default is to refuse.",
    )

    args = parser.parse_args(argv)

    if args.schema_version is None:
        from finalayze.ml.loader import FEATURE_SCHEMA_VERSION  # noqa: PLC0415

        schema_version = FEATURE_SCHEMA_VERSION
    else:
        schema_version = args.schema_version

    segment_dir = args.models_dir / args.segment
    if not segment_dir.is_dir():
        print(f"error: segment directory not found: {segment_dir}", file=sys.stderr)
        return 2

    meta_path = segment_dir / "segment_meta.json"
    if meta_path.exists() and not args.force:
        print(
            f"error: {meta_path} already exists. Pass --force to overwrite.",
            file=sys.stderr,
        )
        return 3

    if args.base_rate is not None and not 0.0 <= args.base_rate <= 1.0:
        print(
            f"error: --base-rate must be in [0.0, 1.0], got {args.base_rate}",
            file=sys.stderr,
        )
        return 4

    meta = _build_meta(base_rate=args.base_rate, schema_version=schema_version)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"wrote {meta_path} (feature_schema_version={schema_version}, base_rate={args.base_rate})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
