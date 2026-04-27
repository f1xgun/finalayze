#!/usr/bin/env node
/**
 * PreToolUse hook: enforce the ast-index-first rule on code searches.
 *
 * BLOCKS when a search targets Python source (ast-index's domain):
 *   - Grep tool with pattern that looks like a symbol and scope is Python (or unspecified)
 *   - Bash invocations of grep / rg / find that target Python files or src/
 *
 * ALLOWS freely when:
 *   - Search is over non-Python files (md, yaml, json, toml, txt, log, etc.)
 *   - Pattern is a regex (contains regex metacharacters) or a string literal
 *   - Output mode is count
 *   - Pattern is too short (≤ 2 chars) to be useful for ast-index
 *
 * The hook reads the Claude Code PreToolUse JSON payload from:
 *   1. process.argv[2]                — legacy path
 *   2. stdin (JSON)                    — current Claude Code behaviour
 * Whichever is present first is used.
 */

'use strict';

const fs = require('fs');

// ---------- payload loading ----------------------------------------------

function readPayload() {
  if (process.argv[2]) {
    try { return JSON.parse(process.argv[2]); } catch (_) {}
  }
  try {
    const raw = fs.readFileSync(0, 'utf8');
    if (raw && raw.trim()) return JSON.parse(raw);
  } catch (_) {}
  return {};
}

const input = readPayload();
const toolName = input.tool_name || input.toolName || '';
const toolInput = input.tool_input || input.toolInput || {};

// Only inspect Grep and Bash. Anything else → allow.
if (toolName !== 'Grep' && toolName !== 'Bash') {
  process.exit(0);
}

// ---------- helpers -------------------------------------------------------

const regexMeta = /[.*+?^${}()|[\]\\]/;

// File-type / path markers that indicate the search is NOT over Python code.
// Examples: markdown docs, yaml config, json payloads, planning artefacts.
const NON_PY_TYPES = new Set([
  'md', 'markdown', 'rst',
  'yaml', 'yml',
  'json', 'jsonl', 'json5',
  'toml', 'ini', 'cfg',
  'txt', 'log',
  'html', 'css', 'scss',
  'sh', 'bash',
  'sql', 'csv',
]);

const NON_PY_PATH_MARKERS = [
  '.md', '.yaml', '.yml', '.json', '.jsonl', '.toml', '.ini', '.cfg',
  '.txt', '.log', '.html', '.csv', '.sql',
  '/docs/', '/.planning/', '/.claude/', '/config/',
  'ROADMAP', 'STATE', 'REQUIREMENTS', 'PLAN.md', 'SUMMARY.md',
  'CHANGELOG', 'README', 'CLAUDE.md', 'AGENTS.md',
];

const PY_PATH_MARKERS = [
  '.py', '/src/', 'src/finalayze', '/tests/', '/scripts/',
];

function hasAnyMarker(haystack, needles) {
  if (!haystack) return false;
  const s = String(haystack);
  return needles.some((m) => s.includes(m));
}

function block(reason) {
  // Claude Code PreToolUse: { decision: "block", reason: "..." } stops the call
  // and feeds the reason back to the model.
  process.stdout.write(JSON.stringify({ decision: 'block', reason }));
  process.exit(0);
}

function suggestAstIndex(pattern, extra = '') {
  const tip =
    `Use ast-index instead of Grep/grep/rg/find for Python code search.\n` +
    `  ast-index search "${pattern}"     # universal search (files + symbols)\n` +
    `  ast-index class   "${pattern}"    # class definition\n` +
    `  ast-index symbol  "${pattern}"    # function/variable\n` +
    `  ast-index usages  "${pattern}"    # all references\n` +
    `  ast-index outline path/to/file.py\n\n` +
    `Grep/grep is still OK for: regex patterns, string literals, comments,\n` +
    `markdown/yaml/json/toml/log files, or when ast-index returned empty.\n` +
    (extra ? `\n${extra}\n` : '');
  return tip;
}

// ---------- Grep tool handling -------------------------------------------

function handleGrep() {
  const pattern = toolInput.pattern || '';
  const path = toolInput.path || '';
  const glob = toolInput.glob || '';
  const type = (toolInput.type || '').toLowerCase();
  const outputMode = toolInput.output_mode || '';

  // Short patterns → ast-index doesn't help.
  if (pattern.length <= 2) return;

  // Count queries → ast-index doesn't do counts.
  if (outputMode === 'count') return;

  // Regex / string-literal patterns → genuine Grep work.
  if (regexMeta.test(pattern)) return;
  if (pattern.startsWith('"') || pattern.startsWith("'")) return;

  // Explicit non-Python scope → allow.
  if (type && NON_PY_TYPES.has(type)) return;
  if (glob && !/\.py\b/.test(glob) && !/\*\*\/\*\.py/.test(glob)) {
    // glob mentions a specific non-py extension → allow.
    if (/\.(md|yaml|yml|json|jsonl|toml|ini|cfg|txt|log|html|csv|sql)\b/.test(glob)) return;
  }
  if (path && hasAnyMarker(path, NON_PY_PATH_MARKERS) &&
      !hasAnyMarker(path, PY_PATH_MARKERS)) {
    return;
  }

  // Otherwise the search targets (or may target) Python code → block.
  block(suggestAstIndex(pattern));
}

// ---------- Bash tool handling -------------------------------------------

function handleBash() {
  const command = toolInput.command || '';
  if (!command) return;

  // Strip leading `cd ... &&` segments so we inspect the actual search cmd.
  const tail = command.replace(/^\s*cd\s+[^&|;]+(&&|;)\s*/g, '').trim();

  // Detect grep / rg / find / egrep / fgrep invocations anywhere in a pipeline.
  const cmdRe = /(^|[|;&\s])(r?grep|rg|egrep|fgrep|find)\b/;
  if (!cmdRe.test(tail)) return;

  // Allow if the command clearly targets non-Python files.
  if (hasAnyMarker(tail, NON_PY_PATH_MARKERS) &&
      !hasAnyMarker(tail, PY_PATH_MARKERS)) {
    // Common benign cases: grep on ROADMAP.md, .planning/, docs/.
    return;
  }

  // Allow typical pipeline searches over command output (e.g., `git log | grep foo`).
  // Heuristic: no file path argument, reading from stdin.
  if (/\|\s*r?grep\b/.test(tail) && !/\b(src|tests|scripts)\b/.test(tail)) {
    return;
  }

  // Allow `find ... -name '*.md'` style lookups that aren't targeting Python.
  if (/find\b[^|;]*-name\s+['"]\*\.(md|yaml|yml|json|toml|ini|cfg|txt|log|html|csv|sql)['"]/.test(tail)) {
    return;
  }

  // Allow when the user explicitly set --include for non-Python extensions.
  if (/--include=?['"]?\*\.(md|yaml|yml|json|toml|ini|cfg|txt|log|html|csv|sql)/.test(tail)) {
    return;
  }

  // Heuristics pointing at Python code search:
  const pythonSignals = [
    /\*\.py\b/,
    /\bsrc\/finalayze\b/,
    /\bsrc\/\b/,
    /--include=?['"]?\*\.py/,
    /-name\s+['"]\*\.py['"]/,
    /\.py\b/,
  ];
  const looksPython = pythonSignals.some((re) => re.test(tail));

  // If it's a raw `grep <pattern> <file-or-dir>` with no type filter at all,
  // assume it may touch Python and suggest ast-index — cheap to suggest.
  if (!looksPython) {
    // Without a Python signal AND without a non-Python marker, still recommend
    // ast-index — grep'ing unknown targets is the exact case ast-index exists for.
    const patternMatch = tail.match(/(r?grep|rg|egrep|fgrep)\b\s+(?:-[A-Za-z]+\s+)*(?:-[-\w]+=?['"]?\S*['"]?\s+)*['"]?([^'"|;&\s][^'"|;&\s]*)['"]?/);
    const pattern = patternMatch ? patternMatch[2] : '<pattern>';
    block(suggestAstIndex(pattern, `Command blocked: \`${tail.slice(0, 120)}\``));
    return;
  }

  // Extract the search pattern (best-effort) for a helpful message.
  const patternMatch = tail.match(/(r?grep|rg|egrep|fgrep)\b\s+(?:-[A-Za-z]+\s+)*(?:-[-\w]+=?['"]?\S*['"]?\s+)*['"]?([^'"|;&\s][^'"|;&\s]*)['"]?/);
  const pattern = patternMatch ? patternMatch[2] : '<pattern>';

  block(suggestAstIndex(pattern, `Command blocked: \`${tail.slice(0, 120)}\``));
}

// ---------- dispatch ------------------------------------------------------

try {
  if (toolName === 'Grep') handleGrep();
  else if (toolName === 'Bash') handleBash();
} catch (err) {
  // Fail-open: a buggy hook must not block legitimate tool use.
  process.stderr.write(`ast-index-grep-guard: ${err && err.message ? err.message : err}\n`);
  process.exit(0);
}
