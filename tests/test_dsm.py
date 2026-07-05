"""
Unit tests for agents/dsm_agent.py — dsm_node.
Numbers are deterministic; the LLM narrative is decoration. Tests assert the
numbers hold regardless of the LLM, and that the constitution catches broken math.
All external calls (corridor tool, LLM, memory) are mocked.
"""
from unittest.mock import MagicMock, patch

import pytest

from agents.dsm_agent import dsm_node, _disruption_fraction, _severity
from eib_guardrails.constitution_checker import check as constitution_check


# ── Auto-mock long-term memory + LLM so unit tests never hit the network ───────

@pytest.fixture(autouse=True)
def _mock_xmemory():
    with patch("agents.dsm_agent._xmemory") as m:
        yield m


# ── Shared mock data ──────────────────────────────────────────────────────────

MOCK_CORRIDORS = [
    {"id": "strait_of_hormuz", "name": "Strait of Hormuz", "chokepoint": True,
     "baseline_flow_mbd": 21.0, "alternative_routes": ["cape_of_good_hope"]},
    {"id": "suez_canal", "name": "Suez Canal", "chokepoint": True,
     "baseline_flow_mbd": 5.5, "alternative_routes": ["cape_of_good_hope"]},
    {"id": "malacca_strait", "name": "Strait of Malacca", "chokepoint": True,
     "baseline_flow_mbd": 16.0, "alternative_routes": ["lombok_strait"]},
]

MOCK_CORRIDOR_RESULT = {
    "tool": "corridor_status", "status": "ok",
    "data": {"corridors": MOCK_CORRIDORS},
}


def _make_llm_response(content: dict) -> MagicMock:
    import json
    msg = MagicMock(); msg.content = json.dumps(content)
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


def _run_dsm(state, narratives=None):
    """Run dsm_node with corridor tool + LLM patched."""
    narratives = narratives if narratives is not None else {}
    corr_p = patch("agents.dsm_agent.get_corridor_status", return_value=MOCK_CORRIDOR_RESULT)
    client_p = patch("agents.dsm_agent._client")
    with corr_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(narratives)
        return dsm_node(state)


# ── Pure-function: deterministic model ────────────────────────────────────────

def test_fraction_full_closure_for_high_physical_risk():
    assert _disruption_fraction(0.9, "war_conflict") == 1.0

def test_fraction_dampened_for_sanctions():
    # sanctions are targeted, not a physical closure → smaller fraction
    assert _disruption_fraction(0.9, "sanctions") == round(0.9 * 0.6, 3)
    assert _disruption_fraction(0.9, "sanctions") < _disruption_fraction(0.9, "war_conflict")

def test_severity_scales_with_volume_and_duration():
    assert _severity(21.0, 42) == "critical"
    assert _severity(0.5, 7) == "low"


# ── Scenario A: Hormuz blockade (war) ─────────────────────────────────────────

def test_hormuz_war_full_baseline_at_risk():
    state = {"corridor_risk": {"strait_of_hormuz": 0.9},
             "corridor_events": {"strait_of_hormuz": "war_conflict"}}
    result = _run_dsm(state)
    sc = next(s for s in result["scenarios"] if s["corridor"] == "strait_of_hormuz")
    assert sc["volume_at_risk_mbd"] == 21.0        # full closure
    assert sc["duration_days"] == 42               # ~6 weeks
    assert sc["severity"] == "critical"
    assert sc["reroute"]["alt_route"] == "cape_of_good_hope"

def test_hormuz_india_exposure_computed():
    state = {"corridor_risk": {"strait_of_hormuz": 0.9},
             "corridor_events": {"strait_of_hormuz": "war_conflict"}}
    result = _run_dsm(state)
    sc = result["scenarios"][0]
    assert sc["india_exposure_mbd"] == round(21.0 * 0.62, 3)
    assert sc["india_exposure_mbd"] <= sc["volume_at_risk_mbd"]


# ── Scenario C: sanctions (targeted, long-lived) ──────────────────────────────

def test_sanctions_dampened_volume_and_long_duration():
    state = {"corridor_risk": {"strait_of_hormuz": 0.7},
             "corridor_events": {"strait_of_hormuz": "sanctions"}}
    result = _run_dsm(state)
    sc = result["scenarios"][0]
    assert sc["disruption_fraction"] == round(0.7 * 0.6, 3)
    assert sc["volume_at_risk_mbd"] < 21.0         # not a full closure
    assert sc["duration_days"] == 90               # sanctions persist


# ── Threshold gating ──────────────────────────────────────────────────────────

def test_low_score_corridor_not_modelled():
    state = {"corridor_risk": {"suez_canal": 0.3},
             "corridor_events": {"suez_canal": "none"}}
    result = _run_dsm(state)
    assert result["scenarios"] == []

def test_unknown_corridor_ignored():
    state = {"corridor_risk": {"red_sea_new": 0.9},
             "corridor_events": {"red_sea_new": "war_conflict"}}
    result = _run_dsm(state)
    assert result["scenarios"] == []


# ── State shape + stigmergy ───────────────────────────────────────────────────

def test_returns_expected_keys():
    state = {"corridor_risk": {"strait_of_hormuz": 0.9},
             "corridor_events": {"strait_of_hormuz": "war_conflict"}}
    result = _run_dsm(state)
    for key in ("current_agent", "scenarios", "stigmergy_markers",
                "audit_trail", "constitution_flags"):
        assert key in result
    assert result["current_agent"] == "dsm_agent"

def test_demand_pheromone_deposited_by_dsm():
    state = {"corridor_risk": {"strait_of_hormuz": 0.9},
             "corridor_events": {"strait_of_hormuz": "war_conflict"}}
    result = _run_dsm(state)
    assert result["stigmergy_markers"], "expected a demand marker"
    for m in result["stigmergy_markers"]:
        assert m["deposited_by"] == "dsm_agent"
        assert m["type"] == "demand"
        assert 0.0 <= m["intensity"] <= 1.0


# ── LLM is decoration only ────────────────────────────────────────────────────

def test_llm_failure_leaves_numbers_intact():
    state = {"corridor_risk": {"strait_of_hormuz": 0.9},
             "corridor_events": {"strait_of_hormuz": "war_conflict"}}
    corr_p = patch("agents.dsm_agent.get_corridor_status", return_value=MOCK_CORRIDOR_RESULT)
    client_p = patch("agents.dsm_agent._client")
    with corr_p, client_p as mock_client:
        mock_client.chat.completions.create.side_effect = Exception("LLM down")
        result = dsm_node(state)
    sc = result["scenarios"][0]
    assert sc["volume_at_risk_mbd"] == 21.0        # unchanged
    assert sc["cascade_narrative"] == ""           # only the decoration is missing

def test_llm_narrative_attached_when_available():
    state = {"corridor_risk": {"strait_of_hormuz": 0.9},
             "corridor_events": {"strait_of_hormuz": "war_conflict"}}
    result = _run_dsm(state, narratives={"strait_of_hormuz": "4 refineries face shortage."})
    assert result["scenarios"][0]["cascade_narrative"] == "4 refineries face shortage."


# ── Memory best-effort ────────────────────────────────────────────────────────

def test_memory_failure_does_not_break_node(_mock_xmemory):
    _mock_xmemory.remember.side_effect = RuntimeError("cloud down")
    state = {"corridor_risk": {"strait_of_hormuz": 0.9},
             "corridor_events": {"strait_of_hormuz": "war_conflict"}}
    result = _run_dsm(state)
    assert result["scenarios"][0]["volume_at_risk_mbd"] == 21.0


# ── Constitution catches broken math (would-be silent failure) ────────────────

def test_constitution_flags_volume_exceeding_baseline():
    bad = {"scenarios": [{
        "corridor": "suez_canal", "baseline_flow_mbd": 5.5,
        "volume_at_risk_mbd": 9.0, "duration_days": 42,
        "disruption_fraction": 0.5, "india_exposure_mbd": 1.0,
        "event_type": "war_conflict",
    }]}
    res = constitution_check("dsm", bad)
    assert not res["passed"]                       # DSM-01 is a block
    assert "DSM-01" in {v["rule_id"] for v in res["violations"]}

def test_constitution_flags_zero_duration():
    bad = {"scenarios": [{
        "corridor": "suez_canal", "baseline_flow_mbd": 5.5,
        "volume_at_risk_mbd": 3.0, "duration_days": 0,
        "disruption_fraction": 0.5, "india_exposure_mbd": 1.0,
        "event_type": "war_conflict",
    }]}
    res = constitution_check("dsm", bad)
    assert "DSM-02" in {v["rule_id"] for v in res["violations"]}

def test_computed_scenarios_pass_constitution():
    state = {"corridor_risk": {"strait_of_hormuz": 0.9, "suez_canal": 0.7},
             "corridor_events": {"strait_of_hormuz": "war_conflict",
                                 "suez_canal": "sanctions"}}
    result = _run_dsm(state)
    assert result["constitution_flags"] == []


# ── Quarantine: a block-flagged scenario is NOT narrated/propagated ───────────

def test_block_flagged_scenario_is_quarantined_and_not_narrated():
    """If the constitution blocks a scenario (broken math), the LLM must not
    narrate it and it must not deposit a demand pheromone — but it stays visible
    in the output, flagged, never silently dropped."""
    state = {"corridor_risk": {"strait_of_hormuz": 0.9},
             "corridor_events": {"strait_of_hormuz": "war_conflict"}}
    blocked = {"passed": False, "violations": [
        {"rule_id": "DSM-01", "severity": "block",
         "corridor": "strait_of_hormuz", "message": "forced block"},
    ]}
    corr_p = patch("agents.dsm_agent.get_corridor_status", return_value=MOCK_CORRIDOR_RESULT)
    client_p = patch("agents.dsm_agent._client")
    check_p = patch("agents.dsm_agent.constitution_check", return_value=blocked)
    with corr_p, client_p as mock_client, check_p:
        mock_client.chat.completions.create.return_value = _make_llm_response(
            {"strait_of_hormuz": "this narrative must never be attached"}
        )
        result = dsm_node(state)

    sc = next(s for s in result["scenarios"] if s["corridor"] == "strait_of_hormuz")
    assert sc["quarantined"] is True
    assert sc["cascade_narrative"] == ""            # LLM did not narrate broken math
    assert result["stigmergy_markers"] == []        # no demand signal from a bad scenario
    assert sc in result["scenarios"]                # still visible, not dropped


# ── Recompute cross-check: catches arithmetic drift that stays in bounds ──────

def test_recompute_catches_volume_drift_within_bounds():
    # 12.6 is <= baseline (21) and positive → DSM-01 passes. But 21 × 1.0 = 21,
    # not 12.6 → only the recompute (DSM-07) catches it.
    bad = {"scenarios": [{
        "corridor": "strait_of_hormuz", "baseline_flow_mbd": 21.0,
        "disruption_fraction": 1.0, "volume_at_risk_mbd": 12.6,
        "india_import_share": 0.62, "india_exposure_mbd": round(12.6 * 0.62, 3),
        "duration_days": 42, "event_type": "war_conflict",
    }]}
    res = constitution_check("dsm", bad)
    ids = {v["rule_id"] for v in res["violations"]}
    assert "DSM-07" in ids
    assert not res["passed"]                         # block
    assert "DSM-01" not in ids                       # bound check missed it; recompute caught it

def test_recompute_catches_india_exposure_drift():
    # india 5.0 <= volume 21 → DSM-03 passes. But 21 × 0.62 = 13.02, not 5.0.
    bad = {"scenarios": [{
        "corridor": "strait_of_hormuz", "baseline_flow_mbd": 21.0,
        "disruption_fraction": 1.0, "volume_at_risk_mbd": 21.0,
        "india_import_share": 0.62, "india_exposure_mbd": 5.0,
        "duration_days": 42, "event_type": "war_conflict",
    }]}
    res = constitution_check("dsm", bad)
    ids = {v["rule_id"] for v in res["violations"]}
    assert "DSM-08" in ids
    assert "DSM-03" not in ids

def test_scenario_carries_india_import_share():
    state = {"corridor_risk": {"strait_of_hormuz": 0.9},
             "corridor_events": {"strait_of_hormuz": "war_conflict"}}
    result = _run_dsm(state)
    assert result["scenarios"][0]["india_import_share"] == 0.62

def test_real_scenarios_pass_recompute():
    # a genuinely computed run must satisfy DSM-07/08 (self-consistent by construction)
    state = {"corridor_risk": {"strait_of_hormuz": 0.9, "suez_canal": 0.7},
             "corridor_events": {"strait_of_hormuz": "war_conflict",
                                 "suez_canal": "sanctions"}}
    result = _run_dsm(state)
    assert result["constitution_flags"] == []
