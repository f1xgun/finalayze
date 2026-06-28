export const meta = {
  name: 'nightly-audit',
  description: 'Autonomous nightly/weekly R&D: sense recent log+metric anomalies, diagnose across audit dimensions, adversarially verify every finding, return triaged findings (each tagged with a path-based risk hint) for the autonomous-audit skill to act on',
  whenToUse: 'Invoked by the autonomous-audit skill from cron (end of day / end of week). Diagnoses only -- it opens no PRs and merges nothing; the skill decides safe-auto-merge vs human-escalation.',
  phases: [
    { title: 'Sense' },
    { title: 'Diagnose' },
    { title: 'Verify' },
  ],
}

// args: { mode?: 'daily' | 'weekly' }  (daily = lighter, fewer finders; weekly = full sweep)
const MODE = (args && args.mode) === 'weekly' ? 'weekly' : 'daily'

const FINDING_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'] },
          file: { type: 'string', description: 'repo-relative path most likely to change for the fix, or "n/a"' },
          line: { type: 'string' },
          evidence: { type: 'string' },
          recommendation: { type: 'string' },
        },
        required: ['title', 'severity', 'file', 'line', 'evidence', 'recommendation'],
      },
    },
  },
  required: ['summary', 'findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    is_real: { type: 'boolean' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    corrected_severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'] },
    reasoning: { type: 'string' },
  },
  required: ['is_real', 'confidence', 'corrected_severity', 'reasoning'],
}

const COMMON = `You are part of the finalayze autonomous nightly audit (multi-market trading system: US/Alpaca, MOEX/Tinkoff gRPC). Read the REAL code/logs in the current repo (prefer ast-index; grep on .py is blocked -- use Read/ast-index). Cite exact file:line. This system handles real money -- prioritize financial-correctness, live-trade safety, and security. Report only REAL, evidenced issues. For the 'file' field, give the repo-relative path most likely to change to FIX the issue (this drives the safe-vs-risky auto-merge decision downstream).`

// ── Sense: recent operational anomalies from the running stack ────────────────
const SENSE_PROMPT = `${COMMON}

DIMENSION: Operational sense -- what is going wrong RIGHT NOW.
Gather and triage recent signals (best-effort; note what is unavailable):
- Container logs: "docker logs finalayze-sandbox-app --since 24h" (weekly: --since 168h) -- find tracebacks, repeated ERROR/WARNING, gRPC failures, DB errors, unhandled exceptions.
- App health: "curl -s http://localhost:8000/api/v1/health".
- Cycle anomalies: read results/validation/cycles.jsonl + results/iterations/history.jsonl against config/pipelines.yaml anomaly_thresholds (max_drawdown_warn_pct, min_fill_rate_pct, signal_drop_pct, max_errors_per_day, min_cycles_per_day).
Report each distinct operational problem as a finding (severity by impact; file = the source path most likely responsible, or "n/a" for pure ops).`

// ── Diagnose dimensions (code audit) ─────────────────────────────────────────
const ALL_DIMENSIONS = [
  { key: 'security', agentType: 'gsd-code-reviewer', focus: 'secrets, the real-money LIVE hard stop (no path may place a live order unattended), API auth, untrusted-input/pickle handling. Re-verify the triple gate + safe-by-default flags still hold.' },
  { key: 'risk_live', agentType: 'risk-officer', focus: 'pre-trade pipeline completeness/ordering, circuit breakers, kill switch, position sizing bounds, stop-loss.' },
  { key: 'financial', agentType: 'quant-analyst', focus: 'Decimal money math, NDFL/tax, deposit accrual, look-ahead bias, currency conversion.' },
  { key: 'ml_data', agentType: 'ml-engineer', focus: 'look-ahead in features, MOEX=Tinkoff-only invariant (no yfinance for ru_*), disabled/force-saved models, calibration.' },
  { key: 'architecture', agentType: 'systems-architect', focus: 'dependency-layer 0->6 (tests/test_architecture.py), async correctness, DB session/engine lifecycle (leaks), dead modules.' },
  { key: 'quality', agentType: 'gsd-code-reviewer', focus: 'dead code, drifted tests, swallowed errors, complexity in money/execution paths, TODO/FIXME debt.' },
]
// Daily = the highest-signal subset; weekly = the full sweep.
const DIMENSIONS = MODE === 'weekly' ? ALL_DIMENSIONS : ALL_DIMENSIONS.slice(0, 4)

log('nightly-audit (' + MODE + '): sensing operational anomalies, then diagnosing ' + DIMENSIONS.length + ' dimensions, then adversarially verifying')

// Sense first (one agent) -- its output frames the diagnosis.
const sense = await agent(SENSE_PROMPT, { label: 'sense:ops', phase: 'Sense', schema: FINDING_SCHEMA, agentType: 'live-monitor-agent' })

const diagnose = await pipeline(
  DIMENSIONS,
  d => agent(`${COMMON}\n\nDIMENSION: ${d.key}.\nFocus: ${d.focus}\nRecent operational context to consider: ${(sense && sense.summary) || 'n/a'}`,
    { label: 'audit:' + d.key, phase: 'Diagnose', schema: FINDING_SCHEMA, agentType: d.agentType }),
  (review, d) => {
    if (!review || !Array.isArray(review.findings) || review.findings.length === 0) return []
    return parallel(review.findings.map(f => () =>
      agent(`Adversarially verify this finding against the real code in the current repo. Default to is_real=false if you cannot reproduce it at the cited location.\n\nDimension: ${d.key}\nTitle: ${f.title}\nSeverity: ${f.severity}\nFile: ${f.file}:${f.line}\nEvidence: ${f.evidence}\nRecommendation: ${f.recommendation}`,
        { label: 'verify:' + d.key, phase: 'Verify', schema: VERDICT_SCHEMA })
        .then(v => ({ ...f, dimension: d.key, verdict: v })).catch(() => null)
    )).then(xs => xs.filter(Boolean))
  }
)

// Path-based risk hint (mirrors scripts/audit_triage.py; the skill re-checks the REAL diff before merging).
const safe = p => {
  const n = String(p || '').trim().replace(/^\.?\//, '')
  return n.endsWith('.md') || n === 'uv.lock' || n.startsWith('docs/') || n.startsWith('tests/')
}

const confirmed = diagnose.flat().filter(f => f && f.verdict && f.verdict.is_real).map(f => ({
  dimension: f.dimension,
  severity: f.verdict.corrected_severity,
  title: f.title,
  file: f.file,
  line: f.line,
  evidence: f.evidence,
  recommendation: f.recommendation,
  confidence: f.verdict.confidence,
  risk_hint: safe(f.file) ? 'safe' : 'risky',
}))
const order = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3, INFO: 4 }
confirmed.sort((a, b) => (order[a.severity] ?? 9) - (order[b.severity] ?? 9))

log('nightly-audit: ' + confirmed.length + ' confirmed finding(s); ops summary: ' + ((sense && sense.summary) || 'n/a').slice(0, 200))

return {
  mode: MODE,
  ops_summary: (sense && sense.summary) || '',
  ops_findings: (sense && sense.findings) || [],
  confirmed,
}
