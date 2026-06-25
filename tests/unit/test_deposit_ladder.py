"""Tests for the recommendation-only deposit-ladder optimizer (Phase 88).

Anti-hollow discipline: the integration tests are driven by the REAL committed snapshot, the
single-allowance-pool invariant is checked against a reference computation, and scope is locked
by an import-graph test + a negative-call guard (the wrapper must never touch an order method).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from finalayze.core.constants import (
    ASV_CAP_PER_BANK,
    ASV_RAISED_TIER_2_8M,
    ASV_RAISED_TIER_2M,
)
from finalayze.core.exceptions import ConfigurationError
from finalayze.orchestration import deposit_ladder as dl
from finalayze.orchestration.deposit_ladder import (
    LadderConstraints,
    LockinVerdict,
    OptimizerRequest,
    TermOffer,
    TermStructure,
    assess_lockin,
    asv_tier_cap,
    load_term_structure,
    make_default_scenarios,
    optimize_deposit_ladder,
    rank_ladders,
    simulate_candidate,
)

_BUDGET = Decimal(2500000)
_AS_OF = date(2025, 4, 25)  # 21.00% key rate, last hold before the first 2025 cut


def _ts(
    offers: list[tuple[int, str, str]],
    scenarios: dict[str, object],
    *,
    mode: str = "backtest",
    as_of: date = _AS_OF,
    horizon: int = 6,
    git_sha: str | None = "abc1234",
) -> TermStructure:
    return TermStructure(
        as_of=as_of,
        source="test offered rates",
        git_sha=git_sha,
        snapshot_mode=mode,
        horizon_months=horizon,
        offers=tuple(
            TermOffer(term_months=t, annual_rate=Decimal(r), roll_spread_pp=Decimal(s))
            for t, r, s in offers
        ),
        raw_scenarios=scenarios,
    )


def _req(
    ts: TermStructure, *, terms: tuple[int, ...] = (3, 12), horizon: int | None = None, **kw: object
) -> OptimizerRequest:
    return OptimizerRequest(
        budget=_BUDGET,
        start=ts.as_of,
        horizon_months=horizon or ts.horizon_months,
        term_structure=ts,
        constraints=LadderConstraints(allowed_terms=terms),
        **kw,  # type: ignore[arg-type]
    )


# A small, internally-consistent inverted curve anchored to the 21% 2025 start.
_INVERTED = [(3, "0.20", "-1.0"), (12, "0.19", "-2.0")]
_REALIZED_PLUS_HOLD = {
    "REALIZED": "use_committed_cbr_calendar",
    "HOLD": [["2025-04-25", "21.00"]],
}


# ---------------------------------------------------------------- loader (fail-closed)


def test_t01_loader_happy_path() -> None:
    ts = load_term_structure()
    assert ts.snapshot_mode == "backtest"
    assert len(ts.offers) == 4
    assert all(o.roll_spread_pp is not None for o in ts.offers)
    assert ts.source and "REALIZED" in ts.raw_scenarios


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "snap.json"
    p.write_text(body)
    return p


def test_t02_fail_closed_missing_roll_spread(tmp_path: Path) -> None:
    body = """{"as_of":"2025-04-25","source":"x","snapshot_mode":"backtest","horizon_months":6,
      "offers":[{"term_months":3,"annual_rate":"0.20"}],
      "key_rate_scenarios":{"REALIZED":"use_committed_cbr_calendar"}}"""
    with pytest.raises(ConfigurationError, match="roll_spread_pp"):
        load_term_structure(_write(tmp_path, body))


def test_t03_fail_closed_missing_allowed_term() -> None:
    ts = _ts(_INVERTED, _REALIZED_PLUS_HOLD)  # only 3 & 12mo offered
    req = _req(ts, terms=(3, 12, 36))  # 36mo requested but not offered
    with pytest.raises(ConfigurationError, match="have no offer"):
        optimize_deposit_ladder(req)


def test_t04_fail_closed_stale_forward(tmp_path: Path) -> None:
    body = """{"as_of":"2025-04-25","source":"x","snapshot_mode":"forward","horizon_months":6,
      "offers":[{"term_months":3,"annual_rate":"0.20","roll_spread_pp":"-1.0"}],
      "key_rate_scenarios":{"REALIZED":"use_committed_cbr_calendar"}}"""
    with pytest.raises(ConfigurationError, match="stale"):
        load_term_structure(_write(tmp_path, body), today=date(2026, 6, 25))


def test_t04b_forward_fresh_ok(tmp_path: Path) -> None:
    as_of = date(2026, 6, 20)
    body = f"""{{"as_of":"{as_of}","source":"x","snapshot_mode":"forward","horizon_months":6,
      "offers":[{{"term_months":3,"annual_rate":"0.20","roll_spread_pp":"-1.0"}}],
      "key_rate_scenarios":{{"REALIZED":"use_committed_cbr_calendar"}}}}"""
    ts = load_term_structure(_write(tmp_path, body), today=as_of + timedelta(days=3))
    assert ts.snapshot_mode == "forward"


def test_t05_fail_closed_corrupt_and_nonpositive(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="corrupt JSON"):
        load_term_structure(_write(tmp_path, "{not json"))
    body = """{"as_of":"2025-04-25","source":"x","snapshot_mode":"backtest","horizon_months":6,
      "offers":[{"term_months":3,"annual_rate":"0","roll_spread_pp":"-1.0"}],
      "key_rate_scenarios":{"REALIZED":"use_committed_cbr_calendar"}}"""
    with pytest.raises(ConfigurationError, match="positive"):
        load_term_structure(_write(tmp_path, body))


def test_t05b_fail_closed_missing_key(tmp_path: Path) -> None:
    body = """{"as_of":"2025-04-25","source":"x","snapshot_mode":"backtest",
      "offers":[{"term_months":3,"annual_rate":"0.20","roll_spread_pp":"-1.0"}],
      "key_rate_scenarios":{"REALIZED":"use_committed_cbr_calendar"}}"""
    with pytest.raises(ConfigurationError, match="horizon_months"):
        load_term_structure(_write(tmp_path, body))


# ---------------------------------------------------------------- invariants


def test_t06_realized_anchor_invariant() -> None:
    ts = _ts(_INVERTED, _REALIZED_PLUS_HOLD)
    req = _req(ts)
    scenarios = make_default_scenarios(req)
    no_realized = [s for s in scenarios if not s.is_realized_anchor]
    with pytest.raises(ConfigurationError, match="REALIZED"):
        rank_ladders(req, [], no_realized)


def test_t07_curve_implied_breakeven_and_invariant() -> None:
    ts = _ts(_INVERTED, _REALIZED_PLUS_HOLD)
    req = _req(ts)
    scenarios = make_default_scenarios(req)
    report = assess_lockin(req, scenarios)
    # under the derived breakeven path, locking long ~ ties rolling short (within noise)
    assert abs(report.curve_implied_bps) <= req.constraints.noise_bps
    # absent the curve-implied anchor -> fail closed
    no_ci = [s for s in scenarios if not s.is_curve_implied]
    with pytest.raises(ConfigurationError, match="CURVE_IMPLIED"):
        assess_lockin(req, no_ci)


def test_t08_degenerate_set_makes_no_claim() -> None:
    # cuts-only scenario set (REALIZED cuts, FAST cuts) -> cannot falsify -> NO_EDGE
    scen = {
        "REALIZED": "use_committed_cbr_calendar",
        "FAST": [["2025-04-25", "21.00"], ["2025-07-25", "16.00"]],
    }
    req = _req(_ts(_INVERTED, scen))
    report = assess_lockin(req, make_default_scenarios(req))
    assert report.scenario_set_degenerate is True
    assert report.verdict == LockinVerdict.NO_EDGE_CURVE_PRICES_CUTS


def test_t09_no_fabricated_edge_on_flat_curve() -> None:
    # a flat curve must NOT be sold as a real lock-in edge (anti-hollow)
    flat = [(3, "0.18", "-1.0"), (12, "0.18", "-1.0")]
    req = _req(_ts(flat, _REALIZED_PLUS_HOLD))
    report = assess_lockin(req, make_default_scenarios(req))
    assert report.verdict != LockinVerdict.REAL_LOCKIN_EDGE


def test_t10_inverted_curve_long_loses_is_liquidity_cost() -> None:
    # deeply inverted: long locked far below short -> long loses under both cut and hold
    deep = [(3, "0.21", "-1.0"), (12, "0.10", "-1.0")]
    req = _req(_ts(deep, _REALIZED_PLUS_HOLD))
    report = assess_lockin(req, make_default_scenarios(req))
    assert report.curve_inverted is True
    assert report.curve_slope_bps < 0
    assert report.verdict == LockinVerdict.LIQUIDITY_COST


def test_t11_path_fragile_flag_tracks_dispersion() -> None:
    req = _req(_ts(_INVERTED, _REALIZED_PLUS_HOLD))
    ranked = rank_ladders(req, dl.generate_candidates(req), make_default_scenarios(req))
    by_arch = {r.candidate.archetype: r for r in ranked}
    # a rolling all-short ladder swings with the path; a locked all-long does not
    assert by_arch["ALL_SHORT"].path_fragile is True
    assert by_arch["ALL_LONG"].path_fragile is False


def test_t13_single_broker_one_allowance_pool() -> None:
    # one broker = one YTD tax-free allowance pool shared across tranches. A barbell of two
    # half-budget tranches in ONE broker is taxed MORE (shares one 1M-floor allowance) than the
    # two tranches simulated in isolation (each gets a full allowance), so its terminal is lower.
    req = _req(_ts(_INVERTED, _REALIZED_PLUS_HOLD), horizon=12)
    offer_by_term = {o.term_months: o for o in req.term_structure.offers}
    scen = make_default_scenarios(req)
    hold = next(s for s in scen if s.scenario_id == "HOLD")  # flat 21% -> interest >> 210k floor

    barbell = dl.LadderCandidate("BB", "BARBELL", {3: Decimal("0.5"), 12: Decimal("0.5")})
    shared = simulate_candidate(barbell, hold, req, offer_by_term).terminal_value

    half = req.budget / 2
    isolated = Decimal(0)
    for term in (3, 12):
        leg_req = OptimizerRequest(
            budget=half,
            start=req.start,
            horizon_months=12,
            term_structure=req.term_structure,
            constraints=LadderConstraints(allowed_terms=(3, 12)),
        )
        leg = dl.LadderCandidate(f"L{term}", "ALL", {term: Decimal(1)})
        isolated += simulate_candidate(leg, hold, leg_req, offer_by_term).terminal_value

    assert shared < isolated  # shared allowance -> more tax -> strictly lower terminal


def test_t14_asv_tier_minfin_verified() -> None:
    assert asv_tier_cap(TermOffer(36, Decimal("0.1"), Decimal(0), "deposit")) == ASV_CAP_PER_BANK
    assert asv_tier_cap(TermOffer(60, Decimal("0.1"), Decimal(0), "deposit")) == ASV_RAISED_TIER_2M
    assert (
        asv_tier_cap(TermOffer(24, Decimal("0.1"), Decimal(0), "irrevocable_cert"))
        == ASV_RAISED_TIER_2M
    )
    assert (
        asv_tier_cap(TermOffer(48, Decimal("0.1"), Decimal(0), "irrevocable_cert"))
        == ASV_RAISED_TIER_2_8M
    )


def test_t15_progressive_band_cross_sleeve_lower_bound() -> None:
    ts = _ts(_INVERTED, _REALIZED_PLUS_HOLD, horizon=12)
    # operator already near the 2.4M band from other sleeves -> deposit interest tips it over
    req_high = _req(ts, horizon=12, ytd_other_taxable_income=Decimal(2350000))
    plan_high = optimize_deposit_ladder(req_high)
    assert plan_high.progressive_band_caveat is True
    # at operator scale with no other income, the band is not crossed
    req_zero = _req(ts, horizon=12)
    plan_zero = optimize_deposit_ladder(req_zero)
    assert plan_zero.progressive_band_caveat is False


# ---------------------------------------------------------------- integration / scope


@pytest.fixture(scope="module")
def committed_plan() -> dl.LadderPlan:
    ts = load_term_structure()
    req = OptimizerRequest(
        budget=_BUDGET, start=ts.as_of, horizon_months=ts.horizon_months, term_structure=ts
    )
    return optimize_deposit_ladder(req)


def test_t16_committed_fixture_is_regime_bet(committed_plan: dl.LadderPlan) -> None:
    r = committed_plan.lockin_report
    # the realized 2025 cuts ran deeper than the inverted curve priced -> a regime bet, not alpha
    assert r.verdict == LockinVerdict.REGIME_BET_NOT_EDGE
    assert r.curve_inverted is True
    assert r.min_lockin_bps < 0 < r.max_lockin_bps  # wins under deep cuts, loses if rates hold
    assert r.n1_caveat is True


def test_t17_import_graph_scope_lock() -> None:
    src = Path(dl.__file__).read_text()
    tree = ast.parse(src)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(n.name for n in node.names)
    joined = " ".join(modules)
    for forbidden in (
        "alpaca_broker",
        "tinkoff_broker",
        "broker_router",
        "AsyncClient",
        "AsyncSandboxClient",
    ):
        assert forbidden not in joined, f"forbidden execution import: {forbidden}"


def test_t18_negative_call_guard_no_order_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("the optimizer must never call an order method")

    monkeypatch.setattr(dl.DepositSimulatedBroker, "submit_order", _boom, raising=False)
    monkeypatch.setattr(dl.DepositSimulatedBroker, "cancel_order", _boom, raising=False)
    req = _req(_ts(_INVERTED, _REALIZED_PLUS_HOLD))
    plan = optimize_deposit_ladder(req)  # completes iff no order method is touched
    assert plan.recommended is not None


def test_t19_deterministic(committed_plan: dl.LadderPlan) -> None:
    ts = load_term_structure()
    req = OptimizerRequest(
        budget=_BUDGET, start=ts.as_of, horizon_months=ts.horizon_months, term_structure=ts
    )
    again = optimize_deposit_ladder(req)
    assert again.recommended.mean_eatv == committed_plan.recommended.mean_eatv
    assert again.lockin_report.verdict == committed_plan.lockin_report.verdict


def test_t20_provenance_stamped(committed_plan: dl.LadderPlan) -> None:
    prov = committed_plan.snapshot_provenance
    assert "deposit_term_structure.json" in prov
    assert "as_of=2025-04-25" in prov
    assert "mode=backtest" in prov


def test_t21_cli_is_read_only_and_runs() -> None:
    script = Path(dl.__file__).resolve().parents[3] / "scripts" / "recommend_deposit_ladder.py"
    source = script.read_text()
    # strip the module docstring (which legitimately *documents the absence* of these) and check
    # the actual CODE never implements a mode/confirm/broker path.
    doc = ast.get_docstring(ast.parse(source)) or ""
    code = source.replace(doc, "")
    for forbidden in (
        "--mode",
        "--confirm",
        "--live",
        "--sandbox",
        "submit_order",
        "AsyncClient",
        "TinkoffBroker",
        "BrokerRouter",
        "FINALAYZE_TINKOFF_TOKEN",
    ):
        assert forbidden not in code, f"CLI must be read-only; found {forbidden}"
    result = subprocess.run(  # noqa: S603 - trusted: our own script via the venv interpreter
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "RECOMMENDED LADDER" in result.stdout
    assert "LOCK-IN VERDICT" in result.stdout
    assert "No money was moved" in result.stdout


# ---------------------------------------------------------------- review-fix regressions


def test_b3_rolled_tranche_accrues_on_open_day() -> None:
    # Under a FLAT key path where the 3mo roll reproduces the offered rate ((21-1)/100 = 0.20), a
    # rolling all-short ladder must earn the SAME 20% continuously as a locked 12mo at 0.20 -- i.e.
    # no roll silently skips its open day (pre-fix the rolling leg lost ~1 day per roll).
    ts = _ts([(3, "0.20", "-1.0"), (12, "0.20", "-1.0")], _REALIZED_PLUS_HOLD, horizon=12)
    req = _req(ts, horizon=12)
    obt = {o.term_months: o for o in ts.offers}
    flat = dl.RatePathScenario("FLAT21", lambda _d: Decimal("21.0"))
    short = simulate_candidate(
        dl.LadderCandidate("S", "ALL_SHORT", {3: Decimal(1)}), flat, req, obt
    )
    locked = simulate_candidate(
        dl.LadderCandidate("L", "ALL_LONG", {12: Decimal(1)}), flat, req, obt
    )
    assert short.roll_count >= 3
    assert abs(short.terminal_value - locked.terminal_value) / req.budget < Decimal("0.0005")


def test_b2_progressive_band_is_scenario_lower_bound() -> None:
    ts = _ts(_INVERTED, _REALIZED_PLUS_HOLD, horizon=12)
    req = _req(ts, horizon=12, ytd_other_taxable_income=Decimal(2300000))
    plan = optimize_deposit_ladder(req)
    rec = plan.recommended
    any_crosses = any(
        rec.per_scenario[sid].progressive_band_caveat
        for sid in plan.scenarios_used
        if sid != "CURVE_IMPLIED"
    )
    # plan caveat is the OR across real-world scenarios -> a true LOWER bound (never under-warns)
    assert plan.progressive_band_caveat == any_crosses


def test_w2_min_liquid_fraction_excludes_locked() -> None:
    ts = _ts(_INVERTED, _REALIZED_PLUS_HOLD)
    req = OptimizerRequest(
        budget=_BUDGET,
        start=ts.as_of,
        horizon_months=6,
        term_structure=ts,
        constraints=LadderConstraints(
            allowed_terms=(3, 12), min_liquid_fraction=Decimal(1), liquidity_horizon_months=6
        ),
    )
    plan = optimize_deposit_ladder(req)
    assert all(
        term <= 6 for term in plan.recommended.candidate.weights
    )  # only liquid ladders survive


def test_w2_impossible_min_liquid_raises() -> None:
    ts = _ts(_INVERTED, _REALIZED_PLUS_HOLD)
    req = OptimizerRequest(
        budget=_BUDGET,
        start=ts.as_of,
        horizon_months=6,
        term_structure=ts,
        constraints=LadderConstraints(
            allowed_terms=(3, 12), min_liquid_fraction=Decimal(1), liquidity_horizon_months=2
        ),
    )
    with pytest.raises(ConfigurationError, match="min_liquid_fraction"):
        optimize_deposit_ladder(req)


def test_b1_recommendation_caveat_reconciles_with_verdict(committed_plan: dl.LadderPlan) -> None:
    # the committed fixture recommends a long lock but the verdict is REGIME_BET -> the plan must
    # carry an explicit reconciliation caveat (recommendation must not silently contradict verdict).
    assert committed_plan.lockin_report.verdict == LockinVerdict.REGIME_BET_NOT_EDGE
    assert committed_plan.recommendation_caveat
    assert "REGIME_BET" in committed_plan.recommendation_caveat
