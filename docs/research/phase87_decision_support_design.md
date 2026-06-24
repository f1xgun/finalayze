# Phase 87 — Honest Decision-Support View for Deposit-Anchored SAA

**Status:** implementation-ready design
**Author:** portfolio-strategist (adjudication of three proposals)
**Scope:** read-only cert-reader + composer + API endpoint + dashboard block. No frozen
allocator/gate is touched, no cert is re-run.

---

## 1. Problem + the honest-product principle

The SAA product recommends regime-tilted target weights (e.g. easing balanced = deposit 0.25 /
ofz 0.40 / equity 0.35). But the FROZEN binding cert
(`src/finalayze/backtest/allocation_gate.py`, run by `scripts/run_allocation_gate.py`) measured
that the allocator **HARD_FAILs** against the best naive benchmark in BOTH rate regimes and over the
full window. The operator's real question is "вклады vs облигации vs акции — what + when?". A view
that shows only the recommendation is dishonest by omission: in a 16-21% regime the deposit wins.

**The honest-product principle (the deliverable):** every number and every verdict the view shows
MUST be DERIVED from the real committed cert `summary.json`. No pre-baked verdict literal, no
fabricated number, no softened HARD_FAIL. A hardcoded verdict is a fixture, not a measurement (the
Phase 72/75 anti-hollow lesson). The view surfaces the cert's verdict and the deposit-anchor
benchmark ALONGSIDE the recommendation, plus an honest-qualitative "when do risk assets pay" framing
that introduces NO fabricated rate threshold (the cert computes none).

### Adjudication of the three proposals — and the one number that decides honesty

The three proposals converge on the same architecture (cert-reader → composer → API + dashboard,
read-only, fail-closed). They differ on ONE load-bearing fact, and getting it wrong would itself be
a fabrication:

| | full-window "best naive" framing | verdict |
|---|---|---|
| CERT-READER proposal | implies the global "deposit wins big" headline is +0.89 | WRONG framing |
| API/DASHBOARD proposal | same +0.89 "deposit wins" headline | WRONG framing |
| **HONEST-MESSAGING proposal** | **full-window best_naive = −0.6506 (equity is best-of-three); +0.89 is the high_rate SUB-window only** | **CORRECT** |

Verified against `results/iterations/allocation-gate-73-20260622T220628Z/summary.json`:

- **Full window** `per_profile.balanced.best_naive_sharpe = −0.6505587…` — the best-of-three bar is
  `equity_100` (`naive.equity_100_sharpe = −0.6505587…`), NOT the deposit. Over the full window the
  100%-deposit leg is `naive.deposit_100_sharpe = −4.6264…` (the deposit's flat ~18% nominal return
  is a deeply negative *RUONIA-excess* Sharpe because the excess basis is ~15%/yr).
- **high_rate sub-window** `per_regime.high_rate.balanced.best_naive_sharpe = +0.8904…` — HERE the
  deposit wins big (the easing-period equity drawdown is excluded), and the allocator is −0.7830.
- **early_cut (easing) sub-window** `per_regime.early_cut.balanced.best_naive_sharpe = −0.7638…` vs
  allocator −1.0511 — ALL sleeves negative.

**Decision: adopt the HONEST-MESSAGING composer's per-regime story model.** The "deposit wins big
(+0.89)" claim is TRUE only as a per-regime (high_rate) statement and must never be presented as the
full-window headline. This distinction is the single biggest honesty risk in the whole view (§7).

**Location decision:** the cert-reader is a pure data hydrator with no allocation logic, so it sits
in **`src/finalayze/backtest/cert_reader.py`** (next to the gate it reads; importable from the API
layer; reads JSON + raises an L0 exception — no upward import). Both other proposals' alternate
homes (`api/cert_reader.py`, `dashboard/cert_reader.py`) work, but `backtest/` keeps the reader
co-located with `allocation_gate.py` and reusable by any future surface (CLI, report) without an API
or dashboard dependency.

---

## 2. Cert-reader module (`src/finalayze/backtest/cert_reader.py`)

Pure stdlib (`pathlib` + `json` + `dataclasses` + `datetime` + `re`). Read-only. Imports nothing
from the allocator/gate logic — it only reads the committed artifact.

### Exception (L0, add to `src/finalayze/core/exceptions.py`)

```python
class CertNotFoundError(ConfigurationError):
    """No committed allocation-gate cert was found, or the latest one is malformed."""
```

Subclassing the existing `ConfigurationError` (exceptions.py:19) keeps it fail-closed and lets the
API translate it to an HTTP 503 with a clean message. (The existing gate already raises
`ConfigurationError` for a missing/corrupt snapshot — same family.)

### Constants

```python
_ITER_DIR = Path(__file__).resolve().parents[2] / "results" / "iterations"  # <repo>/results/iterations
_RUN_PREFIX = "allocation-gate-73"
_TS_RE = re.compile(r"allocation-gate-73-(\d{8}T\d{6}Z)$")   # dirname → YYYYMMDDTHHMMSSZ
_TS_FMT = "%Y%m%dT%H%M%SZ"
# The required top-level keys the latest cert MUST carry (Phase 75 additive set). A missing
# key fails closed — never surface a partial/older-schema cert silently.
_REQUIRED_KEYS = (
    "git_sha", "per_profile", "naive", "regime_split",
    "per_regime", "escalation", "n1_caveat", "phase_verdict",
)
# Slice the BALANCED profile as the representative middle profile for the per-regime stories
# (the cert carries conservative/balanced/growth; balanced is the headline middle).
_REPRESENTATIVE_PROFILE = "balanced"
# Unit keys as emitted by allocation_gate.regime_split (NOT re-declared from the gate to avoid an
# upward import; these are the data contract of the JSON, asserted by a pinned test in §6).
_HIGH_RATE_UNIT = "high_rate"
_EASING_UNIT = "early_cut"        # the post-cut binding unit
_EASING_LABEL = "easing"          # human-facing label (matches the gate's _EASING_UNIT_LABEL)
```

### Dataclasses (frozen)

```python
@dataclass(frozen=True)
class RegimeStory:
    unit_key: str            # "high_rate" | "early_cut"  (raw cert key, for tests)
    unit_label: str          # "high_rate" | "easing"     (human label; early_cut→easing)
    window_start: str        # summary.json["regime_split"][unit][0]  (ISO date string, verbatim)
    window_end: str          # summary.json["regime_split"][unit][1]
    allocation_sharpe: float # per_regime[unit][balanced]["sharpe"]
    best_naive_sharpe: float # per_regime[unit][balanced]["best_naive_sharpe"]
    allocation_sortino: float
    best_naive_sortino: float
    unit_verdict: str        # per_regime[unit][balanced]["verdict"]  (read verbatim)

@dataclass(frozen=True)
class CertDecision:
    # Provenance
    cert_path: str           # absolute path of the selected summary.json
    cert_timestamp: str      # ISO datetime parsed from the dir name suffix
    git_sha: str             # summary.json["git_sha"]
    staleness_days: int      # (today - cert_date).days  (computed via an injected clock)

    # Binding verdict — sourced verbatim, never hardcoded
    phase_verdict: str       # summary.json["phase_verdict"]
    escalation: str | None   # summary.json["escalation"]
    n1_caveat: bool          # summary.json["n1_caveat"]

    # Full-window representative metrics (BALANCED profile)
    alloc_sharpe_full: float       # per_profile[balanced]["sharpe"]
    best_naive_sharpe_full: float  # per_profile[balanced]["best_naive_sharpe"]  (= equity_100 here)
    full_verdict: str              # per_profile[balanced]["verdict"]

    # Per-regime stories (high_rate first, then easing if present)
    regime_stories: list[RegimeStory]

    # Operator-facing strings — DERIVED inline from the fields above (see §3), not pre-baked
    headline: str
    when_framing: str

    # Verbatim caveat from the cert (the gate's _HIGH_RATE_CAVEAT, written to the JSON)
    high_rate_caveat: str          # summary.json["high_rate_caveat"]
```

### Function signatures

```python
def select_latest_cert_dir(iter_dir: Path = _ITER_DIR) -> Path:
    """Return the most-recent allocation-gate-73-* dir. Fail-closed.

    Lists iter_dir for subdirs whose name matches _TS_RE, sorts by the captured
    YYYYMMDDTHHMMSSZ suffix (ISO-8601 → lexicographic == chronological), returns the LAST.
    Raises CertNotFoundError if iter_dir is missing or no matching dir exists. Deterministic:
    same filesystem state → same dir.
    """

def parse_cert_json(cert_dir: Path) -> dict[str, Any]:
    """Read + validate {cert_dir}/summary.json. Fail-closed.

    json.loads; on malformed JSON, missing file, or any missing key in _REQUIRED_KEYS,
    raise CertNotFoundError(diagnostic). Returns the parsed dict (floats/strings as-is).
    """

def _cert_timestamp(cert_dir: Path) -> datetime:
    """Parse the UTC timestamp from the dir name suffix via _TS_RE / _TS_FMT (tz=UTC)."""

def load_latest_cert(
    iter_dir: Path = _ITER_DIR, *, today: date | None = None
) -> CertDecision:
    """Select → parse → hydrate the latest committed cert into a frozen CertDecision.

    `today` is injectable (defaults to RealClock().now().date()) so staleness_days is
    deterministic in tests. Composes headline + when_framing via the §3 derivation. Raises
    CertNotFoundError on any failure (the API/dashboard catch it and show a fail-closed state).
    """
```

**Fail-closed contract (constraint 4):** `select_latest_cert_dir` raises when `results/iterations/`
is absent or empty of matching dirs; `parse_cert_json` raises on bad JSON or any missing required
key. The view layers catch `CertNotFoundError` and render "no committed cert" — never invent
numbers.

---

## 3. The composer — anti-fabrication contract (cert field → operator-facing output)

Every row below maps ONE operator-facing output to the EXACT `summary.json` field it derives from.
Line numbers reference `results/iterations/allocation-gate-73-20260622T220628Z/summary.json`. **No
output exists that is not in this table.**

| # | Operator-facing output | EXACT cert field (or inline derivation) | summary.json line | Verbatim value (this cert) |
|---|---|---|---|---|
| 1 | `phase_verdict` (binding headline) | `summary.json["phase_verdict"]` | 134 | `"HARD_FAIL"` |
| 2 | `headline` text wrapper | DERIVED: `if phase_verdict == "HARD_FAIL": "HOLD DEPOSIT-HEAVY: the allocator does not beat the deposit benchmark (verdict: " + phase_verdict + ")" else "Gate result: " + phase_verdict` | 134 | derived |
| 3 | `escalation` | `summary.json["escalation"]` | 132 | `"deposit_anchor_vs_redesign"` |
| 4 | `n1_caveat` | `summary.json["n1_caveat"]` | 133 | `true` |
| 5 | `git_sha` (provenance) | `summary.json["git_sha"]` | 2 | `44ef26ff…` |
| 6 | `cert_timestamp` | parsed from dir name `allocation-gate-73-20260622T220628Z` | (dir) | `2026-06-22T22:06:28Z` |
| 7 | `staleness_days` | `(today − cert_date).days` (injected clock) | (derived) | derived |
| 8 | `alloc_sharpe_full` | `per_profile["balanced"]["sharpe"]` | 18 | `−0.8589…` |
| 9 | `best_naive_sharpe_full` | `per_profile["balanced"]["best_naive_sharpe"]` (= `equity_100`, NOT deposit) | 19 | `−0.6506…` |
| 10 | `full_verdict` | `per_profile["balanced"]["verdict"]` | 16 | `"HARD_FAIL"` |
| 11 | high_rate story window | `regime_split["high_rate"]` | 50–53 | `["2024-01-02","2025-06-05"]` |
| 12 | high_rate alloc Sharpe | `per_regime["high_rate"]["balanced"]["sharpe"]` | 74 | `−0.7830…` |
| 13 | high_rate best-naive Sharpe (**deposit wins here, +0.89**) | `per_regime["high_rate"]["balanced"]["best_naive_sharpe"]` | 77 | `+0.8904…` |
| 14 | high_rate alloc/best Sortino | `per_regime["high_rate"]["balanced"]["sortino"]` / `["best_naive_sortino"]` | 79–80 | `−1.1200…` / `+1.3121…` |
| 15 | high_rate unit verdict | `per_regime["high_rate"]["balanced"]["verdict"]` | 74 | `"HARD_FAIL"` |
| 16 | easing story window | `regime_split["early_cut"]` | 54–57 | `["2025-06-06","2026-06-08"]` |
| 17 | easing alloc Sharpe | `per_regime["early_cut"]["balanced"]["sharpe"]` | 111 | `−1.0511…` |
| 18 | easing best-naive Sharpe | `per_regime["early_cut"]["balanced"]["best_naive_sharpe"]` | 112 | `−0.7638…` |
| 19 | easing unit verdict | `per_regime["early_cut"]["balanced"]["verdict"]` | 109 | `"HARD_FAIL"` |
| 20 | "all sleeves negative" claim (easing) | DERIVED inline: `alloc_sharpe < 0 AND best_naive_sharpe < 0` for early_cut | 109–112 | `True` (both < 0) |
| 21 | `high_rate_caveat` (verbatim) | `summary.json["high_rate_caveat"]` | 59 | "100% deposit winning raw return in a 16-21% high-rate regime is NOT a failure" |
| 22 | `when_framing` (honest-qualitative) | DERIVED from regime outcomes + n1_caveat + escalation — see below | 59,77,112,132,133 | derived |
| 23 | **rate threshold** ("below X%") | **ABSENT — not in the cert (constraint 2)** | N/A | **never emitted** |

### `headline` derivation (constraint 1 + 6 — derived, never softened)

```python
if data["phase_verdict"] == "HARD_FAIL":
    headline = (
        "HOLD DEPOSIT-HEAVY: the allocator does not beat the deposit benchmark "
        f"(verdict: {data['phase_verdict']})"
    )
else:
    headline = f"Gate result: {data['phase_verdict']}"
```

The literal `"HARD_FAIL"` comes from `data["phase_verdict"]`; the surrounding sentence is a display
wrapper. If the cert ever becomes PASS, the headline flips automatically — it is not a constant.

### `when_framing` derivation (constraint 2 — honest-qualitative, NO fabricated rate threshold)

```python
all_hard_fail = all(s.unit_verdict == "HARD_FAIL" for s in regime_stories)
high_rate = next((s for s in regime_stories if s.unit_key == _HIGH_RATE_UNIT), None)
deposit_won_high_rate = high_rate is not None and high_rate.best_naive_sharpe > 0.0

if all_hard_fail and deposit_won_high_rate:
    when_framing = (
        "Risk assets have not beaten the deposit in either measured regime. "
        f"In the high-rate plateau ({high_rate.window_start}..{high_rate.window_end}) the "
        f"deposit's risk-adjusted return was strongly positive (best-naive Sharpe "
        f"{high_rate.best_naive_sharpe:+.2f}) while the allocator was deeply negative "
        f"({high_rate.allocation_sharpe:+.2f}). "
        "In the single observed easing cycle all sleeves were negative — the allocator still "
        "trailed its best benchmark. "
        + ("This easing read is N=1: suggestive, not statistically robust. " if data["n1_caveat"] else "")
        + ("The recorded escalation is deposit-anchor-vs-redesign: anchor on the near-vol-free "
           "deposit for now; a redesign is the documented next step when conditions change. "
           if data["escalation"] == "deposit_anchor_vs_redesign" else "")
        + "No rate threshold is available from the measurement — the cert computes no key-rate "
          "cutoff; this is a qualitative regime read, not a 'rates below X%' rule."
    )
else:
    # Honest fallback for any OTHER cert outcome — still derived, still no fabricated threshold.
    when_framing = (
        "Per-regime outcomes (each verdict and Sharpe sourced from the cert above) determine "
        "when risk assets pay; no numeric rate threshold is computed by the measurement."
    )
```

Every number in `when_framing` (`{high_rate.best_naive_sharpe:+.2f}` = +0.89 from line 77;
`{high_rate.allocation_sharpe:+.2f}` = −0.78 from line 74; the windows from lines 50–57) is a cert
field. The N=1 clause fires off `data["n1_caveat"]` (line 133); the redesign clause off
`data["escalation"]` (line 132). **No "rates below X%" string is ever produced** (constraint 2).

---

## 4. API model + endpoint + dashboard block

### 4a. API — extend `src/finalayze/api/v1/saa.py` (the existing, already-registered router)

The `saa` router is already included at `router.py:25` and carries
`dependencies=[Depends(api_key_auth)]`. **"Token-free" here means the same thing as the Phase 81
`target-allocation` endpoint: NO Tinkoff/broker token, no DB, no network — a pure filesystem read of
the committed `results/` artifact.** It still sits behind the gateway `X-API-Key` like every other
`/api/v1` route (the dashboard `ApiClient` injects that key on every request, api_client.py:18).
This is an honest framing: the endpoint does not require operator broker credentials and places no
orders — but it is NOT publicly unauthenticated. Do not claim "public" / "unauthenticated".

Add to `saa.py` (frozen Pydantic models mirroring `LegTarget`/`SaaTargetAllocation`,
saa.py:35/45):

```python
_HTTP_NO_CERT = 503  # add alongside the existing _HTTP_NOT_FOUND = 404

class RegimeStoryOut(BaseModel):
    model_config = ConfigDict(frozen=True)
    unit_key: str
    unit_label: str
    window_start: str
    window_end: str
    allocation_sharpe: float
    best_naive_sharpe: float
    allocation_sortino: float
    best_naive_sortino: float
    unit_verdict: str

class CertDecisionResponse(BaseModel):
    """The binding allocation-gate cert verdict + per-regime benchmark stories (read-only)."""
    model_config = ConfigDict(frozen=True)
    cert_path: str
    cert_timestamp: str
    git_sha: str
    staleness_days: int
    phase_verdict: str
    escalation: str | None
    n1_caveat: bool
    alloc_sharpe_full: float
    best_naive_sharpe_full: float
    full_verdict: str
    high_rate_caveat: str
    headline: str
    when_framing: str
    regime_stories: list[RegimeStoryOut]

@router.get("/cert-decision", response_model=CertDecisionResponse)
async def cert_decision() -> CertDecisionResponse:
    """Return the latest binding allocation-gate cert verdict (read-only, no Tinkoff token).

    Surfaces the FROZEN allocator's honest binding verdict measured on real net-of-tax curves
    (Phase 74) ALONGSIDE the deposit-anchor benchmark, so the operator sees the honest truth: in
    a 16-21% regime the deposit wins, and in the single easing cycle so far all sleeves are
    negative (N=1). Every number + verdict is DERIVED from the committed cert summary.json — no
    pre-baked literal, no softened HARD_FAIL. Returns 503 when no committed cert exists
    (fail-closed). Sole side-effect: a filesystem read of results/iterations/.
    """
    from finalayze.backtest.cert_reader import CertNotFoundError, load_latest_cert  # noqa: PLC0415

    try:
        decision = load_latest_cert()
    except CertNotFoundError as exc:
        raise HTTPException(status_code=_HTTP_NO_CERT, detail=str(exc)) from exc

    return CertDecisionResponse(
        cert_path=decision.cert_path,
        cert_timestamp=decision.cert_timestamp,
        git_sha=decision.git_sha,
        staleness_days=decision.staleness_days,
        phase_verdict=decision.phase_verdict,
        escalation=decision.escalation,
        n1_caveat=decision.n1_caveat,
        alloc_sharpe_full=decision.alloc_sharpe_full,
        best_naive_sharpe_full=decision.best_naive_sharpe_full,
        full_verdict=decision.full_verdict,
        high_rate_caveat=decision.high_rate_caveat,
        headline=decision.headline,
        when_framing=decision.when_framing,
        regime_stories=[RegimeStoryOut(**asdict(s)) for s in decision.regime_stories],
    )
```

**ApiClient method** (add to `api_client.py`, mirroring `saa_target_allocation`, api_client.py:92,
fail-soft → empty dict so the dashboard never crashes):

```python
def saa_cert_decision(self) -> dict[str, object]:
    """Fetch the binding cert decision from /api/v1/saa/cert-decision (Phase 87).

    Returns the parsed JSON on success; an empty dict on a 503 (no committed cert) / non-2xx /
    non-dict response so the dashboard renders a friendly empty state rather than crashing.
    """
    resp = self.get("/api/v1/saa/cert-decision")
    result = resp.json()
    if resp.is_success and isinstance(result, dict):
        return result
    return {}
```

### 4b. Dashboard — extend `src/finalayze/dashboard/pages/saa_allocation.py`

Add a `render_cert_decision(api)` section, called from the existing module-level guard
(saa_allocation.py:66) right after `render(_api)`. Wrap the API call in `try/except` (mirror
saa_allocation.py:37-41 — never crash the dashboard).

```python
def render_cert_decision(api: ApiClient) -> None:
    st.divider()
    st.subheader("Binding Verdict — deposit vs bonds vs equity (honest measurement)")
    try:
        cert = api.saa_cert_decision()
    except Exception:  # connection failure — never crash the dashboard
        st.error("Cannot reach API server")
        return
    if not cert or "phase_verdict" not in cert:
        st.info(
            "No committed allocation-gate cert. Run `scripts/run_allocation_gate.py --live` "
            "to produce one, then refresh."
        )
        return

    # 1. Verdict banner — HARD_FAIL is shown as HARD_FAIL (NOT softened, constraint 6).
    if cert["phase_verdict"] == "HARD_FAIL":
        st.error(cert["headline"])
    else:
        st.success(cert["headline"])

    # 2. Benchmark comparison — per-regime rows + a full-window row, numbers from the cert.
    rows = [
        {
            "Regime": s["unit_label"],
            "Period": f'{s["window_start"]} → {s["window_end"]}',
            "Allocation Sharpe": f'{s["allocation_sharpe"]:.4f}',
            "Best-naive Sharpe": f'{s["best_naive_sharpe"]:.4f}',
            "Verdict": s["unit_verdict"],
        }
        for s in cert["regime_stories"]
    ]
    rows.append(
        {
            "Regime": "full window",
            "Period": "—",
            "Allocation Sharpe": f'{cert["alloc_sharpe_full"]:.4f}',
            "Best-naive Sharpe": f'{cert["best_naive_sharpe_full"]:.4f}',
            "Verdict": cert["full_verdict"],
        }
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(cert["high_rate_caveat"])  # verbatim honesty caveat from the cert

    # 3. "When do risk assets pay" — honest-qualitative, no fabricated threshold.
    st.info(cert["when_framing"])

    # 4. Escalation + N=1 caveat — rendered only when present (both sourced from the cert).
    if cert.get("escalation") or cert.get("n1_caveat"):
        bits = []
        if cert.get("escalation"):
            bits.append(f'Escalation: `{cert["escalation"]}`')
        if cert.get("n1_caveat"):
            bits.append("N=1: single observed easing cycle — suggestive, not robust.")
        st.warning(" · ".join(bits))

    # 5. Provenance footer (staleness visible).
    st.caption(
        f'Cert: {cert["cert_timestamp"]} · sha {str(cert["git_sha"])[:8]} · '
        f'{cert["staleness_days"]}d ago · {cert["cert_path"]}'
    )
```

Wire it into the module guard:

```python
if (_api := st.session_state.get("api")) is not None:
    render(_api)
    render_cert_decision(_api)
```

---

## 5. Confirmation: the frozen allocator/gate are UNTOUCHED

- **NOT modified:** `src/finalayze/backtest/allocation_gate.py` (FROZEN measurement gate),
  `src/finalayze/orchestration/allocation.py` (FROZEN W2 allocator),
  `scripts/run_allocation_gate.py` (NOT re-run — the view reads the already-committed artifact).
- **NEW (read-only):** `src/finalayze/backtest/cert_reader.py` — pure stdlib; reads
  `results/iterations/`, parses JSON, raises an L0 exception. Imports no allocation logic, performs
  no money math, mutates nothing (frozen dataclasses).
- **EXTENDED (additive only):** `src/finalayze/core/exceptions.py` (add `CertNotFoundError`),
  `src/finalayze/api/v1/saa.py` (add models + one route on the existing router),
  `src/finalayze/dashboard/api_client.py` (add `saa_cert_decision`),
  `src/finalayze/dashboard/pages/saa_allocation.py` (add `render_cert_decision`).
- **Decimal/float care:** the cert stores Sharpe/Sortino/MaxDD as floats — these are MEASUREMENT
  outputs, not money math; surface them as floats (format to 4dp for display, never round to hide a
  borderline FAIL). No `Decimal` arithmetic is introduced (there is no cash flow here).
- **Smallest blast radius:** one new module + one new exception + three additive extensions. No
  router rename, no new router, no DB, no network, no broker, no token.

---

## 6. TDD test list (write tests first; RED → GREEN)

Tests live in `tests/unit/test_cert_reader.py` (reader) and `tests/unit/test_api_cert_decision.py`
(endpoint + page). Every metric assertion reads the REAL committed cert file and compares — never a
hardcoded fixture value (anti-hollow).

1. **`test_select_latest_cert_dir_picks_most_recent`** — given the 8 committed
   `allocation-gate-73-*` dirs, `select_latest_cert_dir` returns `…20260622T220628Z` (the
   lexicographically/chronologically last). Deterministic across repeated calls.
2. **`test_parse_cert_json_reads_required_keys`** — `parse_cert_json` returns a dict containing all
   `_REQUIRED_KEYS`; values are the JSON types (floats for Sharpe, str for verdict).
3. **`test_surfaced_phase_verdict_equals_committed_cert`** (ANTI-HOLLOW, the key test) —
   `load_latest_cert().phase_verdict == json.load(open(latest/summary.json))["phase_verdict"]`. The
   surfaced verdict BYTE-MATCHES the committed cert; it is not a literal.
4. **`test_hard_fail_not_softened`** (constraint 6) — when the committed cert's `phase_verdict ==
   "HARD_FAIL"`, `CertDecision.phase_verdict == "HARD_FAIL"` and `headline` contains "does not beat
   the deposit". Assert the headline NEVER contains "PASS"/"beats"/"outperforms" when verdict is
   HARD_FAIL.
5. **`test_full_window_best_naive_is_equity_not_deposit`** (the §1 adjudication trap) —
   `best_naive_sharpe_full == per_profile["balanced"]["best_naive_sharpe"]` AND it equals
   `naive["equity_100_sharpe"]`, NOT `naive["deposit_100_sharpe"]`. Guards against re-introducing the
   "deposit wins big" full-window framing error.
6. **`test_high_rate_story_deposit_wins`** — the `high_rate` `RegimeStory.best_naive_sharpe` is
   `> 0` and equals `per_regime["high_rate"]["balanced"]["best_naive_sharpe"]` (≈ +0.89), while
   `allocation_sharpe < 0` — the deposit-wins claim is per-regime, sourced, sign-derived.
7. **`test_easing_story_all_sleeves_negative_derived`** — for `early_cut`, both `allocation_sharpe`
   and `best_naive_sharpe` are `< 0`, and the "all sleeves negative" boolean is computed inline (not
   a string constant). Values equal the cert's `early_cut.balanced` fields.
8. **`test_escalation_passthrough`** — `CertDecision.escalation ==
   summary.json["escalation"]` (= `"deposit_anchor_vs_redesign"` here); it is read verbatim, NOT
   recomputed by the reader.
9. **`test_n1_caveat_passthrough_and_in_framing`** — `n1_caveat == summary.json["n1_caveat"]`
   (True), and when True the `when_framing` string contains "N=1".
10. **`test_high_rate_caveat_verbatim`** — `CertDecision.high_rate_caveat ==
    summary.json["high_rate_caveat"]` exactly (the gate's pinned literal).
11. **`test_when_framing_has_no_fabricated_rate_threshold`** (constraint 2 GUARD) — assert
    `when_framing` does NOT match a rate-threshold regex (e.g.
    `r"below\s+\d+(\.\d+)?\s*%|rates?\s+(under|below)\s+\d"`), and DOES contain "no rate threshold"
    / "qualitative". No "rates below X%" is ever produced.
12. **`test_cert_timestamp_and_staleness_from_dirname`** — `cert_timestamp` parses to
    `2026-06-22T22:06:28+00:00`; with an injected `today=date(2026, 7, 2)`, `staleness_days == 10`.
13. **`test_load_latest_cert_fail_closed_no_dir`** — pointing `iter_dir` at an empty/missing temp
    dir raises `CertNotFoundError` (never returns fabricated numbers).
14. **`test_parse_cert_json_fail_closed_malformed`** — a temp cert dir with malformed JSON, or
    missing a required key, raises `CertNotFoundError` with a diagnostic message.
15. **`test_endpoint_returns_cert_fields`** — FastAPI `TestClient` GET `/api/v1/saa/cert-decision`
    returns 200 with `phase_verdict`, `headline`, `regime_stories` populated and matching the
    committed cert.
16. **`test_endpoint_503_when_no_cert`** — monkeypatch `load_latest_cert` to raise
    `CertNotFoundError`; the endpoint returns HTTP 503 (fail-closed), not 200-with-zeros.
17. **`test_dashboard_render_cert_decision_handles_empty`** — `render_cert_decision` with an
    `ApiClient` stub returning `{}` shows the "No committed cert" info message and never raises
    (mirror the existing page's empty-state test).

(Items 1–14 are the reader; 15–17 the surfaces. 10–14 items minimum; this list is 17 for full
coverage — trim 14/17 if a leaner suite is required, but keep 3, 4, 5, and 11 — they are the
anti-hollow / anti-fabrication core.)

---

## 7. Executive summary + the single biggest honesty risk

**Executive summary (10 lines):**

1. Build a read-only `CertDecision` view that surfaces the FROZEN binding cert's verdict ALONGSIDE
   the deposit-anchor benchmark, so the operator sees the honest truth next to the recommendation.
2. New `src/finalayze/backtest/cert_reader.py` dynamically selects the latest committed
   `allocation-gate-73-*` cert (sort by timestamp suffix) and parses `summary.json` fail-closed.
3. Every number and verdict is DERIVED from the real cert via the §3 field-mapping table — no
   pre-baked literal, no fabricated number, HARD_FAIL shown AS HARD_FAIL.
4. The `headline` and the honest-qualitative `when_framing` are composed inline from cert fields
   (verdict, per-regime Sharpe signs, n1_caveat, escalation) — they flip automatically if the cert
   changes.
5. No fabricated rate threshold: the cert computes none, so the framing stays qualitative ("risk
   assets haven't beaten the deposit in either regime; easing is N=1").
6. Surface via a new `GET /api/v1/saa/cert-decision` on the existing `saa` router (no Tinkoff token,
   pure `results/` read) + a `render_cert_decision` block on the Phase 81 dashboard page.
7. Fail-closed: 503 / "no committed cert" when none exists; the dashboard renders an empty state and
   never crashes (mirrors Phase 81).
8. The FROZEN allocator, the FROZEN gate, and `run_allocation_gate.py` are NOT touched and the cert
   is NOT re-run — smallest blast radius (one module + one exception + three additive extensions).
9. ~17 TDD tests, including the anti-hollow "surfaced == committed cert" test (#3), "HARD_FAIL not
   softened" (#4), "no fabricated rate threshold" (#11), and the full-window-best-naive trap (#5).
10. The verdict this cert reports — HARD_FAIL, deposit_anchor_vs_redesign, N=1 — IS the deliverable:
    the honest finding shipped honestly.

**The single biggest honesty risk to watch:** conflating the **per-regime** "deposit wins big
(+0.89)" with the **full-window** benchmark. The full-window best-naive Sharpe is **−0.6506
(equity_100)**, not the deposit — the +0.89 deposit win is ONLY the high_rate sub-window (which
excludes the easing equity drawdown). Two of the three input proposals framed +0.89 as the global
"deposit wins" headline, which would silently overstate the deposit's full-window advantage and
quietly hide that over the WHOLE window every sleeve — including the deposit's RUONIA-excess Sharpe
(−4.63) — is negative. The view must show the per-regime +0.89 strictly as a high_rate-labeled row,
keep `best_naive_sharpe_full` sourced from `per_profile.balanced.best_naive_sharpe` (= equity_100),
and lock both with tests #5 and #6. Get this wrong and the "honest" view becomes a subtler fixture
than a hardcoded verdict.
