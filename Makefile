.PHONY: sandbox-up sandbox-down sandbox-stop sandbox-nuke sandbox-logs \
       sandbox-restart sandbox-ps sandbox-local \
       test test-all lint typecheck fmt \
       news-test rss-test tg-test help

# ── Docker Sandbox ──────────────────────────────

COMPOSE = docker compose --env-file .env -f docker/docker-compose.sandbox.yml

sandbox-up: ## Start sandbox stack (PostgreSQL, Redis, Prometheus, Grafana, App)
	$(COMPOSE) up -d --build

sandbox-down: ## Stop sandbox stack
	$(COMPOSE) down

sandbox-logs: ## Tail sandbox app logs
	$(COMPOSE) logs -f app

sandbox-restart: ## Restart app container only
	$(COMPOSE) restart app

sandbox-ps: ## Show sandbox container status
	$(COMPOSE) ps

sandbox-stop: ## Stop containers without removing them
	$(COMPOSE) stop

sandbox-nuke: ## Stop, remove containers, volumes, and images
	$(COMPOSE) down -v --rmi local

# ── Local Sandbox (no Docker) ───────────────────

sandbox-local: ## Run sandbox locally (requires PostgreSQL + Redis running)
	FINALAYZE_MODE=sandbox uv run python scripts/run_sandbox.py

sandbox-nosleep: ## Start sandbox + prevent Mac sleep (Ctrl+C to stop both)
	caffeinate -d -i -s $(COMPOSE) up --build

# ── Quick checks ────────────────────────────────

test: ## Run unit tests
	uv run pytest tests/unit/ -x -q --no-cov

test-all: ## Run all tests
	uv run pytest tests/ -q --no-cov

lint: ## Run ruff linter
	uv run ruff check src/ config/ tests/

fmt: ## Format code with ruff
	uv run ruff format src/ config/ tests/

typecheck: ## Run mypy type checker
	uv run mypy src/

# ── News pipeline checks ───────────────────────

rss-test: ## Test RSS feed fetching (live)
	uv run python -c "\
	import feedparser; \
	f = feedparser.parse('https://rssexport.rbc.ru/rbcnews/news/30/full.rss'); \
	print(f'RBC: {len(f.entries)} articles'); \
	[print(f'  - {e.title[:80]}') for e in f.entries[:3]]"

tg-test: ## Test Telegram channel parsing (live)
	uv run python -c "\
	import httpx; from bs4 import BeautifulSoup; \
	channels = ['rbc_news', 'ifax_go', 'kommersant']; \
	[print(f'@{c}: {len(BeautifulSoup(httpx.get(f\"https://t.me/s/{c}\", headers={\"User-Agent\": \"Mozilla/5.0\"}).text, \"html.parser\").select(\".tgme_widget_message_text\"))} msgs') for c in channels]"

news-test: rss-test tg-test ## Test all news sources (live)

# ── Help ────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
