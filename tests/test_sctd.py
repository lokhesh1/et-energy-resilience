"""
Unit tests for agents/sctd_agent.py — sctd_node.
SCTD is fully deterministic (no LLM): feed_at_risk = capacity × Σ(share × fraction).
Tests assert the math, the reroute/bottleneck detection, and the three guardrails
(quarantine skip, contract-drift flag, liveness). All external calls (corridor
tool, memory) are mocked; no network.
"""
from unittest.mock import MagicMock, patch

import pytest

import agents.sctd_agent as sctd
from agents.sctd_agent import sctd_node, _project_refinery, _status_for


# ── Auto-mock long-term memory so unit tests never hit the network ─────────────

@pytest.fixture(autouse=True)
def _mock_xmemory():
    with patch.object(sctd, "_xmemory", MagicMock()) as m:
        yield m


# ── Shared mock corridor status ────────────────────────────────────────────────

MOCK_CORRIDORS = [
    {"id": "strait_of_hormuz", "name": "Strait of Hormuz", "baseline_flow_mbd": 21.0,
     "current_flow_mbd": 0.0, "status": "closed", "risk_score": 0.9, "lat": 26.57, "lon": 56.25},
    {"id": "cape_of_good_hope", "name": "Cape of Good Hope", "baseline_flow_mbd": 4.5,
     "current_flow_mbd": 4.5, "status": "open", "risk_score": 0.2, "lat": -34.36, "lon": 18.47},
    {"id": "malacca_strait", "name": "Strait of Malacca", "baseline_flow_mbd": 16.0,
     "current_flow_mbd": 16.0, "status": "open", "risk_score": 0.2, "lat": 2.5, "lon": 101.3},
    {"id": "panama_canal", "name": "Panama Canal", "baseline_flow_mbd": 0.9,
     "current_flow_mbd": 0.0, "status": "closed", "risk_score": 0.9, "lat": 9.08, "lon": -79.68},
]
MOCK_RESULT = {"tool": "corridor_status", "status": "ok",
               "data": {"corridors": MOCK_CORRIDORS}}


def _run_sctd(state):
    with patch.object(sctd, "get_corridor_status", return_value=MOCK_RESULT):
        return sctd_node(state)


def _hormuz_war():
    return {"scenarios": [{
        "corridor": "strait_of_hormuz", "disruption_fraction": 1.0,
        "volume_at_risk_mbd": 21.0, "quarantined": False,
        "reroute": {"alt_route": "cape_of_good_hope",
                    "added_transit_days": 14, "freight_cost_mult": 1.6},
    }]}


# ── Pure functions: deterministic projection ───────────────────────────────────

def test_status_bands():
    assert _status_for(0.30) == "critical"
    assert _status_for(0.29) == "stressed"
    assert _status_for(0.10) == "stressed"
    assert _status_for(0.09) == "normal"
    assert _status_for(0.0) == "normal"


def test_project_refinery_full_closure():
    ref = {"id": "r", "name": "R", "capacity_mbd": 1.0,
           "corridor_dependency": {"strait_of_hormuz": 0.45, "malacca_strait": 0.10}}
    imp = _project_refinery(ref, {"strait_of_hormuz": 1.0})   # only hormuz disrupted
    assert imp["at_risk_share"] == 0.45                       # 0.45×1.0 + 0.10×0.0
    assert imp["feed_at_risk_mbd"] == 0.45                    # capacity × share
    assert imp["top_corridor"] == "strait_of_hormuz"
    assert imp["status"] == "critical"


def test_project_refinery_no_disruption_is_normal():
    ref = {"id": "r", "name": "R", "capacity_mbd": 1.0,
           "corridor_dependency": {"strait_of_hormuz": 0.45}}
    imp = _project_refinery(ref, {})                          # nothing disrupted
    assert imp["at_risk_share"] == 0.0
    assert imp["feed_at_risk_mbd"] == 0.0
    assert imp["status"] == "normal"
    assert imp["top_corridor"] is None


def test_feed_equals_capacity_times_share():
    # traceability: the stored number is recomputable from its primitives
    result = _run_sctd(_hormuz_war())
    for imp in result["twin_state"]["refineries"]:
        assert abs(imp["feed_at_risk_mbd"] - imp["capacity_mbd"] * imp["at_risk_share"]) < 1e-6


# ── Scenario: Hormuz full closure ──────────────────────────────────────────────

def test_hormuz_closure_hits_jamnagar_critical():
    result = _run_sctd(_hormuz_war())
    jam = next(i for i in result["twin_state"]["refineries"] if i["id"] == "jamnagar_ril")
    assert jam["at_risk_share"] == 0.45          # Jamnagar's Hormuz dependency
    assert jam["feed_at_risk_mbd"] == round(1.24 * 0.45, 4)
    assert jam["status"] == "critical"
    assert "jamnagar_ril" in result["affected_refineries"]


def test_partial_fraction_scales_impact():
    # sanctions-style dampened fraction → proportionally smaller feed at risk
    state = {"scenarios": [{"corridor": "strait_of_hormuz", "disruption_fraction": 0.42,
                            "volume_at_risk_mbd": 8.82, "quarantined": False, "reroute": None}]}
    result = _run_sctd(state)
    jam = next(i for i in result["twin_state"]["refineries"] if i["id"] == "jamnagar_ril")
    assert jam["at_risk_share"] == round(0.45 * 0.42, 4)


# ── Routing + bottleneck detection ─────────────────────────────────────────────

def test_reroute_overloaded_when_volume_exceeds_alt_baseline():
    result = _run_sctd(_hormuz_war())                # 21 mbd onto Cape's 4.5 mbd
    route = result["affected_routes"][0]
    assert route["from_corridor"] == "strait_of_hormuz"
    assert route["to_corridor"] == "cape_of_good_hope"
    assert route["overloaded"] is True

def test_reroute_not_overloaded_when_within_capacity():
    state = {"scenarios": [{
        "corridor": "strait_of_hormuz", "disruption_fraction": 0.1,
        "volume_at_risk_mbd": 2.1, "quarantined": False,
        "reroute": {"alt_route": "cape_of_good_hope",
                    "added_transit_days": 14, "freight_cost_mult": 1.6},
    }]}
    result = _run_sctd(state)
    assert result["affected_routes"][0]["overloaded"] is False


# ── Guardrail 1: contract drift ────────────────────────────────────────────────

def test_missing_disruption_fraction_is_flagged_not_zeroed():
    state = {"scenarios": [{"corridor": "strait_of_hormuz",
                            "volume_at_risk_mbd": 21.0, "quarantined": False}]}  # no fraction
    result = _run_sctd(state)
    ids = {f["rule_id"] for f in result["constitution_flags"]}
    assert "SCTD-CONTRACT" in ids
    assert result["affected_refineries"] == []       # scenario skipped, not modelled as 0-impact


# ── Guardrail 2: liveness (and its false-positive suppression) ─────────────────

def test_panama_scenario_does_not_trip_liveness():
    # No Indian refinery depends on Panama → zero impact is legitimate, not a lost signal
    state = {"scenarios": [{"corridor": "panama_canal", "disruption_fraction": 1.0,
                            "volume_at_risk_mbd": 0.9, "quarantined": False, "reroute": None}]}
    result = _run_sctd(state)
    ids = {f["rule_id"] for f in result["constitution_flags"]}
    assert "SCTD-LIVENESS" not in ids
    assert result["twin_state"]["total_india_shortfall_mbd"] == 0.0

def test_liveness_fires_when_depended_corridor_yields_zero():
    # Force the projection path to disagree with the depended-set path: a real
    # Hormuz disruption that maps to zero impact = signal lost → must flag.
    zero = lambda r, f: {**_project_refinery(r, {}), "status": "normal"}
    with patch.object(sctd, "get_corridor_status", return_value=MOCK_RESULT), \
         patch.object(sctd, "_project_refinery", side_effect=zero):
        result = sctd_node(_hormuz_war())
    ids = {f["rule_id"] for f in result["constitution_flags"]}
    assert "SCTD-LIVENESS" in ids


# ── Guardrail 3: quarantine ────────────────────────────────────────────────────

def test_quarantined_scenario_does_not_propagate():
    state = {"scenarios": [{
        "corridor": "strait_of_hormuz", "disruption_fraction": 1.0,
        "volume_at_risk_mbd": 21.0, "quarantined": True,
        "reroute": {"alt_route": "cape_of_good_hope",
                    "added_transit_days": 14, "freight_cost_mult": 1.6},
    }]}
    result = _run_sctd(state)
    assert result["affected_refineries"] == []       # broken math never mapped
    assert result["affected_routes"] == []           # no reroute from a quarantined scenario
    assert result["stigmergy_markers"] == []         # no bottleneck signal


# ── State shape / stigmergy / twin baseline ────────────────────────────────────

def test_returns_expected_keys():
    result = _run_sctd(_hormuz_war())
    for key in ("current_agent", "affected_refineries", "affected_routes",
                "twin_state", "stigmergy_markers", "audit_trail", "constitution_flags"):
        assert key in result
    assert result["current_agent"] == "sctd_agent"


def test_bottleneck_pheromones_deposited_by_sctd():
    result = _run_sctd(_hormuz_war())
    assert result["stigmergy_markers"], "expected bottleneck markers"
    for m in result["stigmergy_markers"]:
        assert m["deposited_by"] == "sctd_agent"
        assert m["type"] == "bottleneck"
        assert 0.0 <= m["intensity"] <= 1.0


def test_twin_state_has_geojson_and_shortfall():
    result = _run_sctd(_hormuz_war())
    twin = result["twin_state"]
    assert twin["geojson"]["type"] == "FeatureCollection"
    assert twin["geojson"]["features"]
    assert twin["total_india_shortfall_mbd"] > 0

def test_twin_renders_with_zero_scenarios():
    # baseline-healthy twin: no scenarios → still a full map, all refineries normal
    result = _run_sctd({"scenarios": []})
    twin = result["twin_state"]
    assert result["affected_refineries"] == []
    assert twin["total_india_shortfall_mbd"] == 0.0
    assert twin["critical_count"] == 0
    assert twin["geojson"]["features"]               # map still renders
    assert result["constitution_flags"] == []        # no active scenarios → no liveness flag


# ── Memory best-effort ─────────────────────────────────────────────────────────

def test_memory_failure_does_not_break_node(_mock_xmemory):
    _mock_xmemory.remember.side_effect = RuntimeError("cloud down")
    result = _run_sctd(_hormuz_war())
    assert result["twin_state"]["total_india_shortfall_mbd"] > 0


# ── Per-corridor shortfall decomposition (attribution, debugger.md #20) ────────

def test_shortfall_by_corridor_decomposes_total():
    state = {"scenarios": [
        {"corridor": "strait_of_hormuz", "disruption_fraction": 1.0,
         "volume_at_risk_mbd": 21.0, "quarantined": False, "reroute": None},
        {"corridor": "bab_el_mandeb", "disruption_fraction": 1.0,
         "volume_at_risk_mbd": 6.2, "quarantined": False, "reroute": None},
    ]}
    result = _run_sctd(state)
    twin = result["twin_state"]
    parts = twin["shortfall_by_corridor"]
    assert set(parts) == {"strait_of_hormuz", "bab_el_mandeb"}
    # Hormuz dependency shares dominate the refinery base — its contribution must
    # exceed bab's even though both corridors are fully disrupted
    assert parts["strait_of_hormuz"] > parts["bab_el_mandeb"]
    assert sum(parts.values()) == pytest.approx(
        twin["total_india_shortfall_mbd"], abs=0.01)


def test_shortfall_by_corridor_empty_when_no_disruption():
    result = _run_sctd({"scenarios": []})
    assert result["twin_state"]["shortfall_by_corridor"] == {}


# ── Voyage-level refinery reroutes ─────────────────────────────────────────────

def test_hormuz_closure_yields_no_sea_detour_per_refinery():
    result = _run_sctd(_hormuz_war())
    rr = result["twin_state"]["refinery_reroutes"]
    assert rr, "affected refineries must carry reroute advice"
    jam = next(x for x in rr if x["refinery"] == "jamnagar_ril")
    assert jam["port"] == "Sikka (Jamnagar)"
    hormuz_lane = next(l for l in jam["lanes"] if l["corridor"] == "strait_of_hormuz")
    # the geographic truth: Hormuz has NO maritime alternative — bypass + advice
    assert hormuz_lane["no_maritime_alternative"] is True
    assert hormuz_lane["bypass"]["capacity_mbd"] > 0
    assert "re-source" in hormuz_lane["mitigation"]
    assert hormuz_lane["at_risk_mbd"] > 0


def test_bab_closure_yields_cape_option_per_refinery():
    state = {"scenarios": [{
        "corridor": "bab_el_mandeb", "disruption_fraction": 1.0,
        "volume_at_risk_mbd": 6.2, "quarantined": False, "reroute": None}]}
    result = _run_sctd(state)
    rr = result["twin_state"]["refinery_reroutes"]
    assert rr
    lane = rr[0]["lanes"][0]
    assert lane["corridor"] == "bab_el_mandeb"
    assert lane["no_maritime_alternative"] is False
    assert any(o.get("modeled_corridor") == "cape_of_good_hope"
               for o in lane["options"])


def test_refinery_reroutes_empty_when_no_disruption():
    result = _run_sctd({"scenarios": []})
    assert result["twin_state"]["refinery_reroutes"] == []
