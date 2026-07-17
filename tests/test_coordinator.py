"""
Tests for agents/crisis_coordinator.py + the coordinator constitution branch.

Covers:
  * deterministic response_plan assembly (gap/coverage/residual, escalation dial,
    committed actions, priority actions) — the load-bearing, LLM-free core;
  * the narrative: template fallback when the LLM returns nothing, and the node
    never depending on a live call for a safe answer;
  * the integrity-aggregator behaviour: block flags are reconstructed from the
    append-only audit_trail, not the serially-overwritten constitution_flags key;
  * the coordinator constitution (COORD-01..05): no all-clear over an open gap,
    no sanctioned cargo laundered into the plan, plan↔twin arithmetic, dropped
    upstream flags, escalation vocabulary.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from eib_guardrails.constitution_checker import check as constitution_check
import agents.crisis_coordinator as cc


# ── Fixtures / builders ─────────────────────────────────────────────────────────

def _clean_bid(supplier="Spot cargo (Atlantic)", supplier_id="spot_lula_atlantic",
               volume=1.0, price=82.0, corridor="cape_of_good_hope",
               sanctions="clear"):
    return {
        "supplier": supplier, "supplier_id": supplier_id, "region": "spot",
        "grade": "lula", "delivery_corridor": corridor, "volume_mbd": volume,
        "price_per_bbl": price, "transit_days_to_india": 20,
        "sanctions_status": sanctions,
    }


def _state(gap=1.0, covered=1.0, covers_gap=True, critical=1, stressed=0,
           components=None, corridor_risk=None, audit_trail=None,
           constitution_flags=None, disrupted=True):
    if components is None:
        components = [_clean_bid(volume=covered)] if covered > 0 else []
    refineries = ([{"name": f"crit{i}", "status": "critical"} for i in range(critical)]
                  + [{"name": f"str{i}", "status": "stressed"} for i in range(stressed)])
    return {
        "query": "Iran closes the Strait of Hormuz",
        "corridor_risk": corridor_risk or {"strait_of_hormuz": 0.9},
        "corridor_events": {"strait_of_hormuz": "war_conflict"},
        "scenarios": [{"corridor": "strait_of_hormuz"}],
        "twin_state": {
            "total_india_shortfall_mbd": gap,
            "critical_count": critical, "stressed_count": stressed,
            "refineries": refineries,
            "corridors": [{"id": "strait_of_hormuz",
                           "disruption_fraction": 1.0 if disrupted else 0.0}],
        },
        "recommended_mix": {
            "total_volume_mbd": covered, "coverage_ratio": (covered / gap) if gap else None,
            "covers_gap": covers_gap, "components": components,
            "est_daily_cost_usd": 1000,
        },
        "audit_trail": audit_trail or [],
        "constitution_flags": constitution_flags or [],
    }


def _run_node(state):
    """Run the node fully offline: LLM returns nothing (→ template), memory empty."""
    client = MagicMock()
    client.chat.completions.create.return_value = _resp({})
    mem = MagicMock()
    mem.recall_similar.return_value = []
    with patch.object(cc, "_client", client), patch.object(cc, "_xmemory", mem):
        return cc.coordinator_node(state)


def _resp(content):
    msg = MagicMock(); msg.content = json.dumps(content)
    choice = MagicMock(); choice.message = msg
    r = MagicMock(); r.choices = [choice]
    return r


# ── Deterministic plan assembly ─────────────────────────────────────────────────

def test_covered_gap_plan_is_populated_and_clean():
    out = _run_node(_state(gap=1.0, covered=1.0, covers_gap=True, critical=1))
    plan = out["response_plan"]
    assert plan["situation"]["gap_mbd"] == 1.0
    assert plan["procurement"]["covered_mbd"] == 1.0
    assert plan["procurement"]["residual_gap_mbd"] == 0.0
    assert plan["escalation_level"] == "critical"        # a critical refinery
    assert len(plan["procurement"]["committed_actions"]) == 1
    # a clean, reconciled plan trips no coordinator rule
    assert out["constitution_flags"] == []


def test_uncovered_gap_escalates_critical_and_flags_residual():
    out = _run_node(_state(gap=1.0, covered=0.4, covers_gap=False, critical=0, stressed=0,
                           components=[_clean_bid(volume=0.4)]))
    plan = out["response_plan"]
    assert plan["procurement"]["residual_gap_mbd"] == pytest.approx(0.6)
    assert plan["escalation_level"] == "critical"        # COORD-01 by construction
    assert any("UNCOVERED" in a for a in plan["priority_actions"])
    assert any("uncovered" in u.lower() for u in plan["unresolved_issues"])
    assert out["constitution_flags"] == []               # its own plan is consistent


def test_no_gap_low_risk_is_routine_and_needs_no_action():
    out = _run_node(_state(gap=0.0, covered=0.0, covers_gap=True, critical=0,
                           components=[], disrupted=False,
                           corridor_risk={"strait_of_hormuz": 0.2}))
    plan = out["response_plan"]
    assert plan["escalation_level"] == "routine"
    assert plan["procurement"]["committed_actions"] == []
    assert "No action required" in plan["priority_actions"][0]
    assert "No India-bound crude shortfall" in out["final_recommendation"]


def test_no_gap_elevated_risk_is_watch_not_routine():
    """Real tension with zero projected shortfall must read 'watch' and name the
    corridor — never a 'routine / corridors nominal' all-clear."""
    out = _run_node(_state(gap=0.0, covered=0.0, covers_gap=True, critical=0,
                           components=[], disrupted=False,
                           corridor_risk={"strait_of_hormuz": 0.9}))
    plan = out["response_plan"]
    assert plan["escalation_level"] == "watch"
    assert any("strait_of_hormuz" in a for a in plan["priority_actions"])
    rec = out["final_recommendation"]
    assert "strait_of_hormuz" in rec
    assert "nominal" not in rec.lower()
    assert "No India-bound crude shortfall" in rec


def test_no_gap_blind_run_recommendation_is_caveated():
    """Zero news articles retrieved → the all-clear must carry a low-confidence
    caveat ('no evidence looked at' ≠ 'no disruption found')."""
    blind = _state(gap=0.0, covered=0.0, covers_gap=True, critical=0,
                   components=[], disrupted=False,
                   corridor_risk={"strait_of_hormuz": 0.2})
    blind["risk_signals"] = []
    out = _run_node(blind)
    assert out["response_plan"]["situation"]["news_articles"] == 0
    assert "low confidence" in out["final_recommendation"].lower()

    informed = _state(gap=0.0, covered=0.0, covers_gap=True, critical=0,
                      components=[], disrupted=False,
                      corridor_risk={"strait_of_hormuz": 0.2})
    informed["risk_signals"] = [{"title": "Gulf calm as talks progress"}]
    out2 = _run_node(informed)
    assert out2["response_plan"]["situation"]["news_articles"] == 1
    assert "low confidence" not in out2["final_recommendation"].lower()


def test_stressed_only_is_elevated():
    out = _run_node(_state(gap=0.5, covered=0.5, covers_gap=True, critical=0, stressed=2))
    assert out["response_plan"]["escalation_level"] == "elevated"


def test_risky_cargo_is_disclosed_and_coverage_risk_discounted():
    """A committed cargo through a 30%-disrupted corridor: covered counts
    expected delivery (0.7, not 1.0), the residual is honest, the secure line
    carries a CAUTION, and the narrative discloses the risk-adjustment."""
    state = _state(gap=1.0, covered=1.0, covers_gap=False, critical=1)
    state["recommended_mix"]["components"] = [{
        **_clean_bid(volume=1.0, corridor="strait_of_hormuz"),
        "delivery_risk_fraction": 0.3, "effective_volume_mbd": 0.7,
    }]
    state["recommended_mix"]["effective_volume_mbd"] = 0.7
    out = _run_node(state)
    plan = out["response_plan"]
    assert plan["procurement"]["covered_mbd"] == pytest.approx(0.7)
    assert plan["procurement"]["residual_gap_mbd"] == pytest.approx(0.3)
    assert plan["escalation_level"] == "critical"    # honest uncovered residual
    secure = next(a for a in plan["priority_actions"] if a.startswith("Secure"))
    assert "CAUTION" in secure and "30%" in secure
    assert "partially disrupted" in out["final_recommendation"]


# ── Narrative ────────────────────────────────────────────────────────────────────

def test_recommendation_falls_back_to_template_when_llm_empty():
    # LLM returns {} → node must still emit the deterministic draft, not blank.
    out = _run_node(_state(gap=1.0, covered=1.0, critical=1))
    rec = out["final_recommendation"]
    assert rec and rec.startswith("CRITICAL:")
    assert "1.0 mbd" in rec


def test_recommendation_uses_llm_text_when_present():
    client = MagicMock()
    client.chat.completions.create.return_value = _resp({"recommendation": "Phrased by the model."})
    mem = MagicMock(); mem.recall_similar.return_value = []
    with patch.object(cc, "_client", client), patch.object(cc, "_xmemory", mem):
        out = cc.coordinator_node(_state(gap=1.0, covered=1.0, critical=1))
    assert out["final_recommendation"] == "Phrased by the model."


# ── Integrity aggregation from the audit trail ───────────────────────────────────

def test_block_flags_reconstructed_from_audit_trail():
    # constitution_flags (plain key) holds only the last writer's; the durable
    # record is the audit_trail, where each agent embedded its constitution_check.
    audit = [
        {"agent": "dsm_agent", "constitution_check": {
            "violations": [{"rule_id": "DSM-07", "severity": "block",
                            "message": "volume != baseline×fraction"}]}},
        {"agent": "sctd_agent", "constitution_check": {
            "violations": [{"rule_id": "SCTD-LIVENESS", "severity": "block",
                            "message": "signal lost"}]}},
    ]
    flags = cc._collect_block_flags({"audit_trail": audit, "constitution_flags": []})
    rules = {f["rule_id"] for f in flags}
    assert rules == {"DSM-07", "SCTD-LIVENESS"}


def test_upstream_block_flags_surface_in_plan_unresolved():
    audit = [{"agent": "dsm_agent", "constitution_check": {
        "violations": [{"rule_id": "DSM-07", "severity": "block",
                        "message": "broken math"}]}}]
    out = _run_node(_state(gap=1.0, covered=1.0, critical=1, audit_trail=audit))
    issues = out["response_plan"]["unresolved_issues"]
    assert any("DSM-07" in u for u in issues)
    # COORD-04 is satisfied (issue surfaced), so no coordinator warning about it
    assert not any(v["rule_id"] == "COORD-04" for v in out["constitution_flags"])


# ── Coordinator constitution (rule-level) ────────────────────────────────────────

def _plan(escalation="critical", gap=1.0, covered=1.0, residual=0.0,
          covers_gap=True, committed=None, unresolved=None):
    return {
        "escalation_level": escalation,
        "situation": {"gap_mbd": gap, "top_corridor_risks": [], "critical_refineries": [],
                      "stressed_refineries": [], "disrupted_corridors": []},
        "procurement": {"covered_mbd": covered, "residual_gap_mbd": residual,
                        "covers_gap": covers_gap,
                        "committed_actions": committed if committed is not None
                        else [_clean_bid()]},
        "unresolved_issues": unresolved if unresolved is not None else [],
    }


def _check(plan, twin_gap=1.0, upstream=None):
    return constitution_check("coordinator", {
        "response_plan": plan,
        "twin_state": {"total_india_shortfall_mbd": twin_gap},
        "upstream_block_flags": upstream or [],
    })


def test_coord01_blocks_allclear_over_open_gap():
    res = _check(_plan(escalation="routine", gap=1.0, covered=0.4, residual=0.6,
                       covers_gap=False))
    assert not res["passed"]
    assert any(v["rule_id"] == "COORD-01" for v in res["violations"])


def test_coord02_blocks_sanctioned_cargo_by_status():
    committed = [_clean_bid(sanctions="blocked", supplier="NIOC (Iran)",
                            supplier_id="nioc")]
    res = _check(_plan(committed=committed))
    assert not res["passed"]
    assert any(v["rule_id"] == "COORD-02" for v in res["violations"])


def test_coord02_blocks_sanctioned_cargo_by_independent_rescreen():
    # status claims clear, but the supplier name matches the SDN seed → re-screen bites
    committed = [_clean_bid(sanctions="clear", supplier="NIOC (Iran)",
                            supplier_id="nioc")]
    res = _check(_plan(committed=committed))
    assert any(v["rule_id"] == "COORD-02" for v in res["violations"])


def test_coord03_blocks_gap_mismatch_with_twin():
    res = _check(_plan(gap=1.0), twin_gap=2.5)
    assert not res["passed"]
    assert any(v["rule_id"] == "COORD-03" for v in res["violations"])


def test_coord03_blocks_wrong_residual():
    # gap 1.0, covered 1.0 → residual should be 0.0, not 0.5
    res = _check(_plan(gap=1.0, covered=1.0, residual=0.5))
    assert any(v["rule_id"] == "COORD-03" for v in res["violations"])


def test_coord04_warns_when_upstream_flags_dropped():
    res = _check(_plan(unresolved=[]), upstream=[{"agent": "dsm_agent",
                       "rule_id": "DSM-07", "message": "x"}])
    ids = {v["rule_id"] for v in res["violations"]}
    assert "COORD-04" in ids
    assert res["passed"]  # COORD-04 is warn-only, not a block

def test_coord05_warns_on_unknown_escalation():
    res = _check(_plan(escalation="apocalypse"))
    assert any(v["rule_id"] == "COORD-05" for v in res["violations"])


def test_clean_plan_passes_all_rules():
    res = _check(_plan())
    assert res["passed"]
    assert res["violations"] == []
