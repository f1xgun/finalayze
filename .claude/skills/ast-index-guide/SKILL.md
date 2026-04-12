---
name: ast-index-guide
description: Guide for using ast-index to search code in this Python trading project. Use when needing to find classes, symbols, usages, implementations, callers, or explore project structure. Invoke before any code search task.
---

# ast-index Guide for Finalayze

Fast structural code search using tree-sitter AST parsing. 17-69x faster than grep.

## Quick Start

The index is auto-rebuilt on session start. If you need a manual refresh:

```bash
ast-index update    # incremental (changed files only)
ast-index rebuild   # full rebuild
```

## Common Tasks in This Project

### Find a strategy class

```bash
ast-index class "BaseStrategy"
ast-index implementations "BaseStrategy"    # all strategies
ast-index hierarchy "BaseStrategy"          # inheritance tree
```

### Find where a symbol is used

```bash
ast-index usages "StrategyCombiner"
ast-index usages "generate_signal"
ast-index refs "PositionSizer"            # definitions + imports + usages
```

### Explore a module

```bash
ast-index outline "src/finalayze/strategies/combiner.py"
ast-index imports "src/finalayze/risk/position_sizing_pipeline.py"
ast-index deps "src/finalayze/strategies/"
ast-index map --module src/finalayze/strategies
```

### Find callers of a function

```bash
ast-index callers "calculate_position_size"
ast-index call-tree "generate_signal" --depth 3
```

### Project overview

```bash
ast-index map                    # compact project overview
ast-index conventions            # architecture patterns, frameworks, naming
ast-index stats                  # index statistics
```

### Review what changed

```bash
ast-index changed --base main    # symbols changed vs main branch
```

### Find TODOs

```bash
ast-index todo                   # all TODO/FIXME/HACK comments
```

## Advanced: SQL Queries

For complex analysis that can't be done with built-in commands:

```bash
# Find classes with most methods (complexity hotspots)
ast-index query "SELECT s.name, COUNT(*) as methods FROM symbols s JOIN files f ON s.file_id = f.id WHERE s.kind = 'function' GROUP BY f.id ORDER BY methods DESC LIMIT 20"

# Find unused classes
ast-index query "SELECT s.name, f.path FROM symbols s JOIN files f ON s.file_id = f.id WHERE s.kind = 'class' AND s.name NOT IN (SELECT name FROM refs)"

# Symbols in a specific directory
ast-index query "SELECT s.name, s.kind, f.path FROM symbols s JOIN files f ON s.file_id = f.id WHERE f.path LIKE 'src/finalayze/risk/%' AND s.kind = 'class'"
```

## Key Project Paths

| Area | Path |
|------|------|
| Strategies | `src/finalayze/strategies/` |
| Risk | `src/finalayze/risk/` |
| ML | `src/finalayze/ml/` |
| Backtest | `src/finalayze/backtest/` |
| Data | `src/finalayze/data/` |
| Execution | `src/finalayze/execution/` |
| Core | `src/finalayze/core/` |
| Config | `config/` |
| Tests | `tests/` |

## When to Fall Back to Grep

ast-index is the primary tool. Use Grep only when:

1. **Regex patterns** — `ast-index` does literal match only
2. **String literals in code** — searching for `"some error message"`
3. **Comment content** — ast-index indexes symbols, not comment text
4. **ast-index returned empty** — fallback is allowed
5. **Counting occurrences** — `output_mode: "count"`

## Flags

| Flag | Description |
|------|-------------|
| `--fuzzy` | Three-stage: exact -> prefix -> contains match |
| `--in-file <PATH>` | Filter results by file path |
| `--module <PATH>` | Filter results by module path |
| `--limit <N>` | Max results |
| `--format json` | Structured JSON output |
