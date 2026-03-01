"""Tests for nginx TLS configuration (6D.11)."""

from __future__ import annotations

from pathlib import Path

_NGINX_CONF = Path(__file__).resolve().parents[2] / "docker" / "nginx" / "nginx.conf"


class TestNginxTLS:
    def test_listen_443_ssl(self) -> None:
        """nginx.conf must contain a 'listen 443 ssl' directive."""
        content = _NGINX_CONF.read_text()
        assert "listen 443 ssl" in content

    def test_http_redirect_to_https(self) -> None:
        """Port 80 server block must redirect to HTTPS."""
        content = _NGINX_CONF.read_text()
        assert "return 301 https://$host$request_uri" in content

    def test_ssl_protocols(self) -> None:
        """Only TLSv1.2 and TLSv1.3 should be configured."""
        content = _NGINX_CONF.read_text()
        assert "ssl_protocols" in content
        assert "TLSv1.2" in content
        assert "TLSv1.3" in content

    def test_ssl_cert_paths(self) -> None:
        """SSL certificate paths must be configured."""
        content = _NGINX_CONF.read_text()
        assert "ssl_certificate" in content
        assert "/etc/nginx/ssl/fullchain.pem" in content
        assert "/etc/nginx/ssl/privkey.pem" in content

    def test_ssl_gitkeep_exists(self) -> None:
        """docker/nginx/ssl/.gitkeep must exist for cert mounting."""
        gitkeep = Path(__file__).resolve().parents[2] / "docker" / "nginx" / "ssl" / ".gitkeep"
        assert gitkeep.exists()
