# Changelog

All notable changes to the Finalayze project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2026-02-21

### Added

- Initial project scaffolding and repository setup.
- `pyproject.toml` with full dependency specification (Python 3.12, FastAPI,
  SQLAlchemy 2.0 async, XGBoost, LightGBM, PyTorch, pandas-ta, ruff, mypy, pytest).
- Package manager: uv with lockfile.
- Directory structure: `src/`, `tests/`, `config/`, `docs/`, `scripts/`,
  `alembic/`, `docker/`.
- Pre-commit hooks configuration (`.pre-commit-config.yaml`).
- Environment template (`.env.example`).
- `.gitignore` for Python, IDE, and environment files.
- Full documentation suite in `docs/` covering architecture, design, quality,
  operations, and plans.
- Phase 0 (Code Quality Foundation) completed: ruff, mypy, pre-commit, project
  structure established.
