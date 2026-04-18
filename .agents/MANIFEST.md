# AGENTS.md Graph Manifest

`manifest.jsonl` is the machine-readable index of every node in the Harness-Engineering graph.
One JSON object per line; each object describes a single `AGENTS.md` file.

## Why

Agents should not need to read prose to decide which node to open. They:
1. Load the manifest.
2. Filter by `kind`, `layer`, or `keywords` against the user's task.
3. Open the best-matching node's `AGENTS.md` directly.

Result: typical "where do I edit?" hops are ≤2 files, never a full-tree read.

## Schema (per line)

```jsonc
{
  "path": "src/finalayze/strategies/AGENTS.md",   // repo-relative path to the node
  "kind": "module",                                // one of: root | area | package | module
  "layer": 4,                                      // dependency layer (0-6); null for non-src nodes
  "owner": "strategies-agent",                     // primary Claude Code sub-agent
  "parent": "src/finalayze/AGENTS.md",             // repo-relative path or null (root only)
  "children": [],                                  // paths of child AGENTS.md nodes
  "depends_on": ["core", "config", "data", "markets", "analysis", "ml"],
  "imported_by": ["backtest", "orchestration", "execution", "api"],
  "keywords": ["BaseStrategy", "StrategyCombiner", "ADX", "..."],
  "summary": "8 strategies + ADX regime routing + combiner with preset weights"
}
```

## Field semantics

| Field | Type | Meaning |
|---|---|---|
| `path` | string | Canonical node location. Must exist on disk. |
| `kind` | enum | `root` (/AGENTS.md) · `area` (src/, tests/, docs/, config/, scripts/) · `package` (src/finalayze/) · `module` (leaf in src/finalayze/<m>/) |
| `layer` | int \| null | Dependency layer from OVERVIEW.md. `null` for non-src nodes. |
| `owner` | string | Sub-agent that owns edits here (see `docs/AGENTS.md` dispatch table). |
| `parent` | string \| null | Exactly one parent; `null` only for the root. |
| `children` | string[] | Ordered list of direct child node paths. |
| `depends_on` | string[] | Module names this node may import from (dependency-layer enforcement). |
| `imported_by` | string[] | Module names that consume this node. |
| `keywords` | string[] | Lookup terms — class names, concepts, CLI flags. |
| `summary` | string | ≤100 char one-liner. |

## Invariants (checked manually today; to be linted later)

1. Every `AGENTS.md` file in the repo has exactly one manifest entry.
2. Every manifest entry's `path` exists on disk.
3. Every `parent` / `children` reference points to a path that exists in the manifest.
4. `depends_on` respects the dependency-layer rule (no upward imports).
5. The graph is a tree: exactly one `root`, every other node reachable via `parent` links.

## Consumers

- **Human agents** grep `keywords` to locate the right node.
- **Automation** (future `scripts/graph_check.py`) walks the tree to enforce invariants.
- **Dashboards / indexers** can render the graph as a sitemap.

## Updating

When you add / remove / rename an `AGENTS.md`:
1. Edit the file (or `git mv` it).
2. Update its manifest line (path, children, keywords).
3. Update the parent's `children` array.
4. Run `scripts/graph_check.py` (TODO — not yet written; for now, manually verify).
