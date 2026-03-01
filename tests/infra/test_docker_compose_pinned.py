"""Tests for pinned Docker image tags (6D.10)."""

from __future__ import annotations

from pathlib import Path

import yaml

_COMPOSE_PROD = Path(__file__).resolve().parents[2] / "docker" / "docker-compose.prod.yml"


class TestDockerImagesPinned:
    def test_no_latest_tags(self) -> None:
        """All images in docker-compose.prod.yml must use pinned tags (no 'latest')."""
        data = yaml.safe_load(_COMPOSE_PROD.read_text())
        services = data.get("services", {})
        for name, svc in services.items():
            image = svc.get("image", "")
            if not image:
                continue  # build-only service
            assert "latest" not in image, f"Service '{name}' uses unpinned image: {image}"

    def test_timescaledb_pinned(self) -> None:
        """TimescaleDB image must be pinned to a specific version."""
        data = yaml.safe_load(_COMPOSE_PROD.read_text())
        pg_image = data["services"]["postgres"]["image"]
        assert pg_image == "timescale/timescaledb:2.17.2-pg16"

    def test_redis_pinned(self) -> None:
        """Redis image must be pinned to a specific version."""
        data = yaml.safe_load(_COMPOSE_PROD.read_text())
        redis_image = data["services"]["redis"]["image"]
        assert redis_image == "redis:7.4-alpine"
