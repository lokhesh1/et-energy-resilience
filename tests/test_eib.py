import operator
import pytest
from graph.eib_state import EnergyIntelligenceBoard, StigmergyMarker


def make_marker(**kwargs) -> StigmergyMarker:
    defaults = dict(
        type="risk",
        target="strait_of_hormuz",
        intensity=0.8,
        deposited_by="gri_agent",
        timestamp="2026-06-30T00:00:00Z",
        decay_rate=0.1,
    )
    return {**defaults, **kwargs}


def test_import():
    assert EnergyIntelligenceBoard is not None
    assert StigmergyMarker is not None


def test_stigmergy_marker_structure():
    marker = make_marker()
    assert marker["type"] == "risk"
    assert marker["intensity"] == 0.8
    assert marker["decay_rate"] == 0.1


def test_stigmergy_marker_types():
    for t in ("risk", "bottleneck", "demand", "bid", "alert"):
        m = make_marker(type=t)
        assert m["type"] == t


def test_operator_add_reducer():
    # simulates concurrent procurement agents appending bids
    bids_a = [{"agent": "west_africa", "price": 82.0}]
    bids_b = [{"agent": "americas", "price": 79.5}]
    bids_c = [{"agent": "spot_market", "price": 84.0}]
    merged = operator.add(operator.add(bids_a, bids_b), bids_c)
    assert len(merged) == 3
    agents = [b["agent"] for b in merged]
    assert "west_africa" in agents
    assert "americas" in agents
    assert "spot_market" in agents


def test_operator_add_stigmergy_markers():
    m1 = [make_marker(deposited_by="gri_agent", intensity=0.8)]
    m2 = [make_marker(deposited_by="dsm_agent", type="bottleneck", intensity=0.6)]
    m3 = [make_marker(deposited_by="sctd_agent", type="demand", intensity=0.5)]
    merged = operator.add(operator.add(m1, m2), m3)
    assert len(merged) == 3
    assert merged[0]["deposited_by"] == "gri_agent"
    assert merged[1]["type"] == "bottleneck"
    assert merged[2]["type"] == "demand"


def test_pheromone_field_structure():
    field: dict[str, float] = {
        "strait_of_hormuz": 0.8,
        "suez_canal": 0.4,
        "malacca_strait": 0.6,
    }
    assert all(isinstance(v, float) for v in field.values())
    assert field["strait_of_hormuz"] > field["suez_canal"]


def test_eib_state_instantiation():
    state: EnergyIntelligenceBoard = {
        "query": "Hormuz closure impact",
        "scenario_params": {"corridor": "strait_of_hormuz", "severity": "high"},
        "messages": [],
        "risk_signals": [],
        "corridor_risk": {},
        "scenarios": [],
        "affected_refineries": [],
        "affected_routes": [],
        "twin_state": {},
        "bids": [],
        "evaluated_bids": [],
        "response_plan": {},
        "final_recommendation": "",
        "retrieved_memories": [],
        "working_context": {},
        "audit_trail": [],
        "constitution_flags": [],
        "current_agent": "",
        "stigmergy_markers": [],
        "pheromone_field": {},
    }
    assert state["query"] == "Hormuz closure impact"
    assert isinstance(state["stigmergy_markers"], list)
    assert isinstance(state["pheromone_field"], dict)
