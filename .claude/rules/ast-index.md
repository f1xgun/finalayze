# ast-index Rules

## Mandatory Search Rules

1. **ALWAYS use ast-index FIRST** for any code search task
2. **NEVER duplicate results** — if ast-index found usages/implementations, that IS the complete answer
3. **DO NOT run grep/Grep "for completeness"** after ast-index returns results
4. **Use Grep ONLY when:**
   - ast-index returns empty results
   - Searching for regex patterns (ast-index uses literal match)
   - Searching for string literals inside code (`"some text"`)
   - Searching in comments content
   - Counting occurrences (`output_mode: "count"`)

## Why ast-index

ast-index is 17-69x faster than grep (1-10ms vs 200ms-3s) and returns structured, accurate results.
It uses tree-sitter parsing — understands Python AST, not just text patterns.

## Command Reference (Python)

| Task | Command | Time |
|------|---------|------|
| Universal search | `ast-index search "query"` | ~10ms |
| Find class | `ast-index class "ClassName"` | ~1ms |
| Find symbol (func/class/var) | `ast-index symbol "name"` | ~1ms |
| Find all usages | `ast-index usages "SymbolName"` | ~8ms |
| Find implementations | `ast-index implementations "BaseClass"` | ~5ms |
| Class hierarchy | `ast-index hierarchy "BaseClass"` | ~5ms |
| Find callers | `ast-index callers "function_name"` | ~1s |
| File outline | `ast-index outline "path/to/file.py"` | ~1ms |
| File imports | `ast-index imports "path/to/file.py"` | ~0.3ms |
| Module deps | `ast-index deps "src/finalayze/strategies/"` | ~10ms |
| Project map | `ast-index map` | ~1-3s |
| Conventions | `ast-index conventions` | ~1-4s |
| Changed symbols | `ast-index changed --base main` | ~1s |
| Cross-references | `ast-index refs "Symbol"` | ~10ms |
| TODO/FIXME | `ast-index todo` | ~1s |

## Python-Specific Patterns

| Task | Command |
|------|---------|
| Find Python class | `ast-index class "BaseStrategy"` |
| Find async functions | `ast-index symbol "fetch"` |
| Find decorators | `ast-index search "@router"` |
| Find all strategies | `ast-index implementations "BaseStrategy"` |
| Show file structure | `ast-index outline "src/finalayze/strategies/combiner.py"` |
| Find unused symbols | `ast-index unused-symbols --module src/finalayze/` |

## Index Management

- Index is auto-rebuilt on SessionStart (hook: `ast-index-rebuild.sh`)
- `ast-index update` — incremental update (changed files only)
- `ast-index rebuild` — full rebuild (after major refactoring)
- `ast-index stats` — show index statistics

## Search Decision Tree

```
Need to find code? 
  |
  +-- Symbol/class/function name? --> ast-index search/class/symbol
  +-- Who uses this symbol? -------> ast-index usages
  +-- Who calls this function? ----> ast-index callers
  +-- Class inheritance? ----------> ast-index hierarchy/implementations
  +-- File structure? -------------> ast-index outline
  +-- Regex pattern? --------------> Grep (allowed)
  +-- String literal in code? -----> Grep (allowed)
  +-- Comment content? ------------> Grep (allowed)
  +-- ast-index returned empty? ---> Grep (fallback allowed)
```
