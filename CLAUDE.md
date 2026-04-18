# Claude Code — Entry Shim

Claude Code auto-loads this file. The canonical agent entry point for this project is
[**AGENTS.md**](AGENTS.md) (Harness Engineering graph root).

Read `AGENTS.md` first; it forwards you to the correct area (`src/`, `config/`, `tests/`,
`scripts/`, `docs/`) and then to the module-level AGENTS.md that owns the files you need
to edit. Machine-readable index: [`.agents/manifest.jsonl`](.agents/manifest.jsonl).

## Hard invariants (repeated here so Claude Code sees them even before walking the graph)

1. Imports flow downward only across dependency layers 0→6.
2. TDD mandatory: failing test first.
3. MOEX data = Tinkoff Invest gRPC only. Never `yfinance` for MOEX tickers.
4. Any change to strategies / risk / backtest / ML triggers the `backtest-iteration` skill.
5. `uv run ruff check .` and `uv run mypy src/` must stay green.

Nothing else belongs in this file — everything that used to live here now lives in a node of
the graph. If you find yourself adding content here, add it to the appropriate `AGENTS.md`
instead, and update `.agents/manifest.jsonl`.
