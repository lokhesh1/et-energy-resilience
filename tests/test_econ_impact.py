"""
Tests for agents/economic_impact.py — the Economic Impact Modeller.

Covers:
  * Damage formulas (premium spend, residual loss, refinery losses, reroute
    freight, SPR refill, Brent spike, import bill, CPI/CAD, do-nothing)
  * Recovery lever ranking (net benefit order, feasibility gates)
  * Recovery timeline (days-to-normal, cumulative curves)
  * Guardrails (ECON-LIVENESS, ECON-CONTRACT, ECON-DOMINANCE)
  * Zero-gap passthrough (no damage, no levers)
  * Params fallback (missing file → defaults, loud)
  * Coordinator integration (econ block in response_plan + narrative)
  * Distiller digest carry (economic_impact present in trajectory)
"""
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

import agents.economic_impact as eim
from agents.economic_impact import eim_node, _brent_spike, _micro_damage, _macro_damage
from agents.economic_impact import _do_nothing_cost, _loss_per_bbl, _recovery_timeline
from agents.economic_impact import _lever_procurement, _lever_spr, _lever_demand_restraint
from agents.economic_impact import _lever_export_curtailment, _lever_overbuy
from agents.economic_impact import _subsidy_vs_passthrough

_PARAMS = eim._DEFAULT_PARAMS


def _make_state(gap=1.2, duration=42, brent=85.0, covered=1.2,
                components=None, scenarios=None, quarantined=False):
    if components is None:
        components = [{
            "supplier": "NNPC", "price_per_bbl": 92.0,
            "volume_mbd": 0.6, "transit_days_to_india": 18,
            "delivery_corridor": "cape_of_good_hope",
            "effective_volume_mbd": 0.6,
        }, {
            "supplier": "Petrobras", "price_per_bbl": 88.0,
            "volume_mbd": 0.6, "transit_days_to_india": 25,
            "delivery_corridor": "cape_of_good_hope",
            "effective_volume_mbd": 0.6,
        }]
    if scenarios is None:
        scenarios = [{
            "corridor": "strait_of_hormuz",
            "event_type": "war_conflict",
            "volume_at_risk_mbd": 21.0,
            "india_exposure_mbd": 13.0,
            "duration_days": duration,
            "reroute": None,
            "quarantined": quarantined,
        }]
    return {
        "twin_state": {
            "total_india_shortfall_mbd": gap,
            "critical_count": 3,
            "stressed_count": 4,
            "refineries": [
                {"name": "Jamnagar", "feed_at_risk_mbd": 0.5, "status": "critical",
                 "capacity_mbd": 1.36},
                {"name": "Mangalore", "feed_at_risk_mbd": 0.3, "status": "stressed",
                 "capacity_mbd": 0.3},
            ],
            "corridors": [{"id": "strait_of_hormuz", "disruption_fraction": 1.0}],
        },
        "recommended_mix": {
            "gap_mbd": gap,
            "total_volume_mbd": covered,
            "effective_volume_mbd": covered,
            "coverage_ratio": 1.0 if gap > 0 else None,
            "covers_gap": covered >= gap,
            "components": components,
            "est_daily_cost_usd": 100_000_000,
        },
        "scenarios": scenarios,
        "corridor_risk": {"strait_of_hormuz": 0.95},
        "corridor_events": {"strait_of_hormuz": "war_conflict"},
        "audit_trail": [],
        "constitution_flags": [],
        "pheromone_field": {},
        "stigmergy_markers": [],
    }


# ── Micro damage ──────────────────────────────────────────────────────────────

def test_premium_spend():
    state = _make_state()
    brent = 85.0
    micro = _micro_damage(_PARAMS, state["twin_state"], state["recommended_mix"],
                          state["scenarios"], brent, 42)
    # Two cargoes: (92-85)*0.6e6*42 + (88-85)*0.6e6*42
    expected = (7 * 600_000 * 42) + (3 * 600_000 * 42)
    assert micro["premium_spend_usd"] == expected


def test_residual_loss_with_gap():
    state = _make_state(gap=2.0, covered=1.5)
    micro = _micro_damage(_PARAMS, state["twin_state"], state["recommended_mix"],
                          state["scenarios"], 85.0, 42)
    loss_bbl = _loss_per_bbl(_PARAMS)
    expected = round(0.5 * 1e6 * 42 * loss_bbl)
    assert micro["residual_loss_usd"] == expected


def test_refinery_losses_sorted_by_impact():
    state = _make_state()
    micro = _micro_damage(_PARAMS, state["twin_state"], state["recommended_mix"],
                          state["scenarios"], 85.0, 42)
    losses = micro["refinery_losses"]
    assert len(losses) == 2
    assert losses[0]["name"] == "Jamnagar"
    assert losses[0]["throughput_loss_usd"] > losses[1]["throughput_loss_usd"]


def test_reroute_freight_with_reroute():
    scenarios = [{
        "corridor": "suez_canal",
        "event_type": "war_conflict",
        "volume_at_risk_mbd": 5.0,
        "india_exposure_mbd": 0.6,
        "duration_days": 42,
        "reroute": {"alt_route": "cape_of_good_hope",
                    "added_transit_days": 14, "freight_cost_mult": 1.5},
        "quarantined": False,
    }]
    micro = _micro_damage(_PARAMS, {"refineries": []}, {"components": []},
                          scenarios, 85.0, 42)
    freight_rate = _PARAMS["freight_rate_usd_per_bbl_per_day"]
    expected = round(0.6 * 1e6 * 14 * freight_rate)
    assert micro["reroute_freight_usd"] == expected


def test_quarantined_scenario_excluded():
    scenarios = [{
        "corridor": "suez_canal", "event_type": "war_conflict",
        "volume_at_risk_mbd": 5.0, "india_exposure_mbd": 0.6,
        "duration_days": 42,
        "reroute": {"added_transit_days": 14, "freight_cost_mult": 1.5},
        "quarantined": True,
    }]
    micro = _micro_damage(_PARAMS, {"refineries": []}, {"components": []},
                          scenarios, 85.0, 42)
    assert micro["reroute_freight_usd"] == 0


# ── Brent spike ───────────────────────────────────────────────────────────────

def test_brent_spike_basic():
    scenarios = [{"volume_at_risk_mbd": 10.0, "reroute": None, "quarantined": False}]
    spike = _brent_spike(_PARAMS, scenarios, 80.0)
    assert spike["delta_usd"] > 0
    assert spike["basis"] == "elasticity_model"
    assert spike["net_supply_loss_mbd"] == 10.0


def test_brent_spike_capped():
    scenarios = [{"volume_at_risk_mbd": 50.0, "reroute": None, "quarantined": False}]
    spike = _brent_spike(_PARAMS, scenarios, 80.0)
    assert spike["delta_usd"] <= _PARAMS["brent_spike_cap_usd"]
    assert spike["capped"] is True


def test_brent_spike_reroute_recovers_volume():
    scenarios = [{"volume_at_risk_mbd": 5.0,
                  "reroute": {"freight_cost_mult": 1.5}, "quarantined": False}]
    spike = _brent_spike(_PARAMS, scenarios, 80.0)
    assert spike["net_supply_loss_mbd"] < 5.0


# ── Macro damage ──────────────────────────────────────────────────────────────

def test_macro_import_bill():
    spike = {"delta_usd": 10.0}
    macro = _macro_damage(_PARAMS, spike, 42, 80.0)
    imports = _PARAMS["india_crude_imports_mbd"]
    expected = round((imports * 1e6 * (80 + 10) - imports * 1e6 * 80) * 42)
    assert macro["import_bill_delta_usd"] == expected


def test_macro_cpi():
    spike = {"delta_usd": 20.0}
    macro = _macro_damage(_PARAMS, spike, 42, 80.0)
    expected = round(_PARAMS["cpi_bps_per_10usd_brent"] * 20.0 / 10.0, 1)
    assert macro["cpi_impact_bps"] == expected


def test_macro_cad():
    spike = {"delta_usd": 10.0}
    macro = _macro_damage(_PARAMS, spike, 42, 80.0)
    expected = round(_PARAMS["cad_pct_gdp_per_10usd_brent"] * 10.0 / 10.0, 2)
    assert macro["cad_gdp_impact_pct"] == expected


# ── Do-nothing ────────────────────────────────────────────────────────────────

def test_do_nothing_cost():
    loss_bbl = _loss_per_bbl(_PARAMS)
    cost = _do_nothing_cost(_PARAMS, 1.0, 42)
    assert cost == round(1.0 * 1e6 * 42 * loss_bbl)


def test_do_nothing_zero_gap():
    assert _do_nothing_cost(_PARAMS, 0.0, 42) == 0.0


# ── Recovery levers ───────────────────────────────────────────────────────────

def test_procurement_lever():
    micro = {"premium_spend_usd": 1_000_000}
    loss_bbl = _loss_per_bbl(_PARAMS)
    lev = _lever_procurement(micro, 10_000_000, gap=1.0, covered=1.0,
                             duration=42, loss_bbl=loss_bbl)
    assert lev is not None
    assert lev["lever"] == "committed_procurement"
    assert lev["net_benefit_usd"] == lev["avoided_loss_usd"] - lev["lever_cost_usd"]


def test_spr_lever():
    lev = _lever_spr(gap=1.0, residual=0.5, duration=42,
                     loss_bbl=12.0, refill_exposure=500_000)
    assert lev is not None
    assert lev["lever"] == "spr_drawdown"
    assert lev["drawdown_mbd"] > 0


def test_demand_restraint_lever():
    lev = _lever_demand_restraint(_PARAMS, brent=85.0, spike_delta=10.0, duration=42, gap=1.0)
    assert lev is not None
    assert lev["lever"] == "demand_restraint"
    assert lev["restraint_pct"] == _PARAMS["max_feasible_restraint_pct"]


def test_export_curtailment_lever():
    lev = _lever_export_curtailment(_PARAMS, brent=85.0, duration=42, gap=1.0)
    assert lev is not None
    assert lev["lever"] == "export_curtailment"
    assert lev["curtailable_mbd"] <= _PARAMS["india_product_export_mbd"]


def test_overbuy_lever_no_headroom():
    lev = _lever_overbuy(_PARAMS, gap=1.0, covered=1.3, brent=85.0,
                         spike_delta=10.0, duration=42)
    assert lev is None


def test_subsidy_vs_passthrough():
    sub = _subsidy_vs_passthrough(_PARAMS, spike_delta=10.0, duration=42)
    assert sub is not None
    assert sub["subsidy_fiscal_cost_usd"] > 0
    assert sub["passthrough_cpi_bps"] > 0


def test_subsidy_no_spike():
    sub = _subsidy_vs_passthrough(_PARAMS, spike_delta=0.0, duration=42)
    assert sub is None


# ── Recovery timeline ─────────────────────────────────────────────────────────

def test_timeline_days_to_normal():
    mix = {"components": [
        {"transit_days_to_india": 10, "effective_volume_mbd": 0.6},
        {"transit_days_to_india": 20, "effective_volume_mbd": 0.6},
    ]}
    tl = _recovery_timeline(gap=1.0, duration=42, loss_bbl=12.0,
                            mix=mix, scenarios=[])
    assert tl["days_to_normal"] == 20
    assert tl["savings_usd"] > 0
    assert len(tl["daily_loss_curve"]) > 0


def test_timeline_zero_gap():
    tl = _recovery_timeline(gap=0.0, duration=42, loss_bbl=12.0,
                            mix={}, scenarios=[])
    assert tl["days_to_normal"] == 0
    assert tl["cumulative_do_nothing_usd"] == 0


# ── Guardrails ────────────────────────────────────────────────────────────────

@patch.object(eim, "_fetch_brent", return_value=(85.0, False))
def test_econ_contract_missing_twin(mock_brent):
    state = _make_state()
    state["twin_state"] = {}
    result = eim_node(state)
    econ = result["economic_impact"]
    flag_names = [f["flag"] for f in econ["guardrail_flags"]]
    assert "ECON-CONTRACT" in flag_names


@patch("agents.economic_impact.calculate_drawdown", side_effect=Exception("mocked"))
@patch.object(eim, "_fetch_brent", return_value=(85.0, False))
def test_econ_liveness_gap_but_zero_exposure(mock_brent, mock_spr):
    # Pathological handoff failure: gap > 0, duration > 0, but every cost
    # component is zero (covered at brent = zero premium, zero-volume scenario
    # = no spike/reroute, no refineries, SPR calc fails). The gap says
    # something is wrong but the economics see nothing → ECON-LIVENESS fires.
    scenarios = [{
        "corridor": "strait_of_hormuz", "event_type": "war_conflict",
        "volume_at_risk_mbd": 0.0, "india_exposure_mbd": 0.0,
        "duration_days": 42, "reroute": None, "quarantined": False,
    }]
    components = [{"supplier": "test", "price_per_bbl": 85.0,
                   "volume_mbd": 1.0, "transit_days_to_india": 15,
                   "delivery_corridor": "cape_of_good_hope",
                   "effective_volume_mbd": 1.0}]
    state = _make_state(gap=1.0, duration=42, covered=1.0,
                        components=components, scenarios=scenarios)
    state["twin_state"]["refineries"] = []
    result = eim_node(state)
    econ = result["economic_impact"]
    flag_names = [f["flag"] for f in econ["guardrail_flags"]]
    assert "ECON-LIVENESS" in flag_names


# ── Node output shape ─────────────────────────────────────────────────────────

@patch.object(eim, "_fetch_brent", return_value=(85.0, True))
def test_node_output_shape(mock_brent):
    state = _make_state()
    result = eim_node(state)
    assert result["current_agent"] == "economic_impact"
    econ = result["economic_impact"]
    assert "total_exposure_usd" in econ
    assert "do_nothing_cost_usd" in econ
    assert "plan_net_benefit_usd" in econ
    assert "brent_spike_estimate" in econ
    assert "micro" in econ
    assert "macro" in econ
    assert "recovery_actions" in econ
    assert "recovery_timeline" in econ
    assert econ["params_source"] == "file"
    assert len(result["audit_trail"]) == 1


@patch.object(eim, "_fetch_brent", return_value=(85.0, True))
def test_node_zero_gap(mock_brent):
    state = _make_state(gap=0.0, covered=0.0, components=[], scenarios=[])
    result = eim_node(state)
    econ = result["economic_impact"]
    assert econ["total_exposure_usd"] == 0
    assert econ["do_nothing_cost_usd"] == 0
    assert econ["recovery_actions"] == []


@patch.object(eim, "_fetch_brent", return_value=(85.0, True))
def test_recovery_actions_sorted_by_net_benefit(mock_brent):
    state = _make_state()
    result = eim_node(state)
    actions = result["economic_impact"]["recovery_actions"]
    if len(actions) > 1:
        benefits = [a["net_benefit_usd"] for a in actions]
        assert benefits == sorted(benefits, reverse=True), \
            "Recovery actions must be sorted by net_benefit descending"


@patch.object(eim, "_fetch_brent", return_value=(80.0, False))
def test_brent_fallback_flagged(mock_brent):
    state = _make_state()
    result = eim_node(state)
    econ = result["economic_impact"]
    assert econ["brent_source"] == "fallback"


# ── Params fallback ───────────────────────────────────────────────────────────

def test_params_fallback():
    with patch.object(eim, "_PARAMS_PATH", Path("/nonexistent/econ_params.json")):
        params, from_file = eim._load_params()
        assert not from_file
        assert params == eim._DEFAULT_PARAMS


# ── Coordinator integration ───────────────────────────────────────────────────

def test_coordinator_extracts_econ():
    from agents.crisis_coordinator import _extract_economic_impact
    state = {"economic_impact": {
        "total_exposure_usd": 5_000_000_000,
        "do_nothing_cost_usd": 10_000_000_000,
        "plan_net_benefit_usd": 4_000_000_000,
        "brent_spike_estimate": {"delta_usd": 15.0},
        "macro": {"import_bill_delta_usd": 3e9, "cpi_impact_bps": 45.0,
                  "cad_gdp_impact_pct": 0.6},
        "recovery_actions": [{"lever": "procurement", "net_benefit_usd": 2e9}],
        "subsidy_vs_passthrough": {"subsidy_fiscal_cost_usd": 1e9,
                                   "passthrough_cpi_bps": 45.0},
        "recovery_timeline": {"days_to_normal": 20},
        "micro": {"refinery_losses": [{"name": "Jamnagar", "throughput_loss_usd": 1e9}]},
        "guardrail_flags": [],
    }}
    result = _extract_economic_impact(state)
    assert result is not None
    assert result["total_exposure_usd"] == 5_000_000_000
    assert result["import_bill_delta_usd"] == 3e9


def test_coordinator_extracts_none_when_absent():
    from agents.crisis_coordinator import _extract_economic_impact
    assert _extract_economic_impact({}) is None
    assert _extract_economic_impact({"economic_impact": {}}) is None


# ── Distiller digest ──────────────────────────────────────────────────────────

@patch.object(eim, "_fetch_brent", return_value=(85.0, True))
def test_distiller_digest_carries_econ(mock_brent):
    from agents.distiller.experience_distiller import build_trajectory
    state = _make_state()
    state["economic_impact"] = eim_node(state)["economic_impact"]
    state["response_plan"] = {"situation": {}, "procurement": {}}
    state["final_recommendation"] = "test"
    state["query"] = "test query"
    digest = build_trajectory(state)
    assert "economic_impact" in digest
    assert digest["economic_impact"] is not None
    assert digest["economic_impact"]["total_exposure_usd"] > 0


# ── Summary components ────────────────────────────────────────────────────────

def test_summary_econ_tiles():
    from api.summary import build_components
    summary = {
        "response_plan": {
            "escalation_level": "critical",
            "situation": {"disruption_drivers": []},
            "procurement": {"coverage_ratio": 1.0, "covers_gap": True,
                            "residual_gap_mbd": 0, "committed_actions": []},
            "economic_impact": {
                "total_exposure_usd": 5e9,
                "do_nothing_cost_usd": 10e9,
                "plan_net_benefit_usd": 4e9,
                "brent_spike_estimate": {"delta_usd": 15.0},
                "import_bill_delta_usd": 3e9,
                "cpi_impact_bps": 45.0,
                "recovery_actions": [{"lever": "procurement",
                                      "net_benefit_usd": 2e9,
                                      "description": "Buy cargoes",
                                      "avoided_loss_usd": 3e9,
                                      "lever_cost_usd": 1e9,
                                      "time_to_effect_days": 0,
                                      "feasible": True}],
                "subsidy_vs_passthrough": {
                    "subsidy_fiscal_cost_usd": 1e9,
                    "passthrough_cpi_bps": 45.0,
                },
            },
        },
        "twin_summary": {"total_india_shortfall_mbd": 1.2,
                         "critical_count": 3, "stressed_count": 4},
        "escalation_level": "critical",
        "corridor_risk": {},
        "constitution_flags": [],
    }
    twin = {"geojson": {}}
    comps = build_components(summary, twin)
    types = [c["type"] for c in comps]
    assert "recovery_table" in types
    assert "policy_tradeoff" in types
    econ_metrics = [c for c in comps
                    if c.get("type") == "metrics" and c.get("title") == "Economic impact"]
    assert len(econ_metrics) == 1
    assert len(econ_metrics[0]["items"]) >= 3
