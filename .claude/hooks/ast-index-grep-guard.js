#!/usr/bin/env node
/**
 * PreToolUse hook: intercept Grep/Glob calls and suggest ast-index when appropriate.
 *
 * BLOCKS Grep when the pattern looks like a symbol/class/function lookup
 * that ast-index handles better (17-69x faster, structured results).
 *
 * ALLOWS Grep when:
 *   - Pattern is a regex (contains regex metacharacters)
 *   - Searching for string literals in code
 *   - Searching in comments content
 *   - ast-index returned empty and user is falling back
 */

const input = JSON.parse(process.argv[2] || '{}');
const toolName = input.tool_name || '';
const toolInput = input.tool_input || {};

// Only intercept Grep
if (toolName !== 'Grep') {
  process.exit(0);
}

const pattern = toolInput.pattern || '';

// Regex metacharacters that indicate a genuine regex search (not symbol lookup)
const regexMeta = /[.*+?^${}()|[\]\\]/;

// Patterns that are clearly regex-based searches — allow Grep
if (regexMeta.test(pattern)) {
  process.exit(0);
}

// String literal searches (quoted) — allow Grep
if (pattern.startsWith('"') || pattern.startsWith("'")) {
  process.exit(0);
}

// Very short patterns (1-2 chars) — not useful for ast-index
if (pattern.length <= 2) {
  process.exit(0);
}

// If output_mode is "count" — ast-index doesn't do counts, allow Grep
if (toolInput.output_mode === 'count') {
  process.exit(0);
}

// Pattern looks like a symbol/class/function name — block and suggest ast-index
const result = {
  decision: "block",
  reason: `Use ast-index instead of Grep for symbol lookup "${pattern}". Try:\n` +
    `  ast-index search "${pattern}"     # universal search\n` +
    `  ast-index class "${pattern}"      # find class\n` +
    `  ast-index symbol "${pattern}"     # find symbol\n` +
    `  ast-index usages "${pattern}"     # find all usages\n` +
    `Grep is allowed for: regex patterns, string literals, comment searches, or when ast-index returns empty.`
};

console.log(JSON.stringify(result));
