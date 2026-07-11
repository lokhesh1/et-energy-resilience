"""
Tests for graph/workflow.py + graph/nodes.py — the EIB orchestrator.

Covers:
  * the pheromone-field rebuild (decay math, max-per-target, evaporation floor,
    fail-open on bad timestamps) — the loop nothing used to close;
  * the wrapper contracts (sequential wrapper writes the field back; bidder
    wrapper returns ONLY reducer keys, so the parallel fan-out is safe);
  * a full end-to-end run (all network/LLM mocked): fan-out → fan-in, a real
    non-empty pheromone_field mid-run, a covered mix, an accumulated audit trail;
  * a no-risk run flowing clean to a zero-gap plan;
  * the twin-only sub-graph stopping after SCTD.
"""
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from graph.nodes import rebuild_pheromone_field, wrap, wrap_bidder
from graph.workflow import (
    build_graph, build_twin_graph, initial_state, run_board_with_learning, SPOT,
)


# ── Shared mock data (mirrors tests/test_agents.py) ────────────────────────────

_NOW = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)


def _mk(target, intensity, decay_rate=0.1, age_h=0.0):
    ts = (_NOW - timedelta(hours=age_h)).isoformat()
    return {"type": "risk", "target": target, "intensity": intensity,
            "deposited_by": "x", "timestamp": ts, "decay_rate": decay_rate}


def _fresh_marker(target, intensity):
    """A marker timestamped at real `now`, so it survives the internal `now=None`
    rebuild inside wrap/wrap_bidder (which use the wall clock, not `_NOW`)."""
    return {"type": "risk", "target": target, "intensity": intensity,
            "deposited_by": "gri_agent",
            "timestamp": datetime.now(timezone.utc).isoformat(), "decay_rate": 0.1}


MOCK_CORRIDORS = [
    {"id": "strait_of_hormuz", "name": "Strait of Hormuz", "region": "Middle East",
     "chokepoint": True, "baseline_flow_mbd": 21.0, "risk_factors": ["Iran"]},
    {"id": "suez_canal", "name": "Suez Canal", "region": "North Africa",
     "chokepoint": True, "baseline_flow_mbd": 5.5, "risk_factors": ["Egypt"]},
]

MOCK_ARTICLES = [
    {"title": "Hormuz shipping lanes disrupted", "url": "https://reuters.com/a1",
     "source": "reuters.com", "trust_score": 0.95, "trusted": True},
    {"title": "Iran navy exercises near strait", "url": "https://ft.com/a2",
     "source": "ft.com", "trust_score": 0.90, "trusted": True},
]

MOCK_NEWS_RESULT = {
    "tool": "news_fetcher", "status": "ok",
    "data": {"articles": MOCK_ARTICLES, "errors": []},
    "source_trust_avg": 0.92, "low_trust_sources_flagged": 0,
    "retrieved_at": "2026-07-09T00:00:00+00:00", "staleness_seconds": 2,
}

MOCK_CORRIDOR_RESULT = {
    "tool": "corridor_status", "status": "ok",
    "data": {"corridors": MOCK_CORRIDORS},
    "source_trust_avg": 1.0, "low_trust_sources_flagged": 0,
    "retrieved_at": "2026-07-09T00:00:00+00:00", "staleness_seconds": 0,
}

# Hormuz high (war_conflict) → DSM models it, SCTD sees a shortfall, bidders bid.
HIGH_RISK_LLM = {
    "corridor_risk": {
        "strait_of_hormuz": {
            "score": 0.90, "confidence": 0.92, "evidence_count": 2,
            "key_signals": ["Hormuz shipping lanes disrupted", "Iran navy exercises near strait"],
            "reasoning": "Two high-trust signals with direct corridor reference.",
            "event_type": "war_conflict",
        },
    },
    "novel_corridor_alerts": [], "overall_assessment": "Elevated Hormuz risk.",
    "low_trust_signals_flagged": 0,
}

# All corridors calm → no scenario, zero gap, no bids.
NO_RISK_LLM = {
    "corridor_risk": {
        "strait_of_hormuz": {
            "score": 0.10, "confidence": 0.80, "evidence_count": 1,
            "key_signals": ["routine traffic"], "reasoning": "Nominal.",
            "event_type": "none",
        },
    },
    "novel_corridor_alerts": [], "overall_assessment": "All nominal.",
    "low_trust_signals_flagged": 0,
}


def _llm_response(content: dict) -> MagicMock:
    msg = MagicMock(); msg.content = json.dumps(content)
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


class _run_offline:
    """Patch every external edge (GRI tools+LLM, DSM narrative LLM, bidder price,
    all three xmemory handles) so a graph invocation is fully network-free."""

    def __init__(self, llm_output):
        self.llm_output = llm_output
        self._patches = []

    def __enter__(self):
        gri_client = MagicMock()
        gri_client.chat.completions.create.return_value = _llm_response(self.llm_output)
        dsm_client = MagicMock()
        dsm_client.chat.completions.create.return_value = _llm_response({})  # empty narratives
        coord_client = MagicMock()
        coord_client.chat.completions.create.return_value = _llm_response({})  # template fallback
        coord_mem = MagicMock()
        coord_mem.recall_similar.return_value = []

        self._patches = [
            patch("agents.gri_agent._fetch_tools",
                  return_value=(MOCK_NEWS_RESULT, MOCK_CORRIDOR_RESULT)),
            patch("agents.gri_agent._client", gri_client),
            patch("agents.dsm_agent._client", dsm_client),
            patch("agents.crisis_coordinator._client", coord_client),
            patch("agents.procurement._sourcing_base.fetch_price",
                  return_value={"status": "ok", "data": {"current_price": 80.0}}),
            patch("agents.gri_agent._xmemory", MagicMock()),
            patch("agents.dsm_agent._xmemory", MagicMock()),
            patch("agents.sctd_agent._xmemory", MagicMock()),
            patch("agents.crisis_coordinator._xmemory", coord_mem),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False


def _invoke(llm_output):
    graph = build_graph()
    with _run_offline(llm_output):
        return graph.invoke(
            initial_state(query="Iran closes the Strait of Hormuz"),
            config={"configurable": {"thread_id": "test-run"}},
        )


# ── rebuild_pheromone_field ────────────────────────────────────────────────────

def test_fresh_markers_equal_raw_intensity_max_per_target():
    field = rebuild_pheromone_field(
        [_mk("hormuz", 0.9), _mk("hormuz", 0.5), _mk("suez", 0.7)], _NOW)
    assert field == {"hormuz": 0.9, "suez": 0.7}


def test_aged_marker_decays_exponentially():
    field = rebuild_pheromone_field([_mk("hormuz", 0.9, 0.1, age_h=24)], _NOW)
    assert field["hormuz"] == pytest.approx(0.9 * pow(2.718281828, -2.4), abs=1e-3)


def test_fully_evaporated_marker_dropped_below_floor():
    field = rebuild_pheromone_field([_mk("hormuz", 0.9, 0.1, age_h=72)], _NOW)
    assert "hormuz" not in field


def test_malformed_timestamp_fails_open_to_raw_intensity():
    field = rebuild_pheromone_field(
        [{"target": "x", "intensity": 0.8, "decay_rate": 0.1, "timestamp": "garbage"}], _NOW)
    assert field["x"] == 0.8


def test_empty_and_targetless_markers_ignored():
    assert rebuild_pheromone_field([], _NOW) == {}
    assert rebuild_pheromone_field([{"intensity": 0.9, "timestamp": _NOW.isoformat()}], _NOW) == {}


# ── wrapper contracts ──────────────────────────────────────────────────────────

def test_sequential_wrapper_injects_field_and_writes_it_back():
    seen = {}

    def fake_agent(state):
        seen["field"] = state["pheromone_field"]
        return {"stigmergy_markers": [_fresh_marker("suez", 0.6)]}

    node = wrap(fake_agent, "fake")
    out = node({"stigmergy_markers": [_fresh_marker("hormuz", 0.9)]})

    assert seen["field"] == {"hormuz": 0.9}           # agent saw current field
    assert out["pheromone_field"]["hormuz"] == 0.9    # written back
    assert out["pheromone_field"]["suez"] == 0.6      # incl. this node's own deposit
    assert out["current_agent"] == "fake"


def test_bidder_wrapper_injects_field_but_returns_only_agent_output():
    seen = {}

    def fake_bidder(state):
        seen["field"] = state["pheromone_field"]
        return {"bids": [{"x": 1}], "audit_trail": [{"a": 1}]}

    node = wrap_bidder(fake_bidder, "spot")
    out = node({"stigmergy_markers": [_fresh_marker("hormuz", 0.9)]})

    assert seen["field"] == {"hormuz": 0.9}
    # must NOT write plain keys — three bidders share one superstep
    assert "pheromone_field" not in out
    assert "current_agent" not in out
    assert set(out.keys()) == {"bids", "audit_trail"}


# ── end-to-end: high-risk run ──────────────────────────────────────────────────

def test_full_run_completes_and_populates_response_plan():
    final = _invoke(HIGH_RISK_LLM)
    assert final["response_plan"], "coordinator produced no response_plan"
    assert final["final_recommendation"]


def test_full_run_field_is_non_empty_regression():
    # The whole reason this layer exists: nothing used to build the field.
    final = _invoke(HIGH_RISK_LLM)
    assert final["pheromone_field"], "pheromone_field empty — rebuild loop is dead"
    assert final["pheromone_field"].get("strait_of_hormuz", 0) > 0


def test_full_run_fans_out_bids_and_composes_mix():
    final = _invoke(HIGH_RISK_LLM)
    assert final["twin_state"]["total_india_shortfall_mbd"] > 0
    assert len(final["bids"]) > 0                 # bidders fanned out
    assert final["evaluated_bids"]               # evaluator ran (fan-in)
    assert final["recommended_mix"].get("components") is not None


def test_full_run_audit_trail_spans_all_agents():
    final = _invoke(HIGH_RISK_LLM)
    agents = {a["agent"] for a in final["audit_trail"]}
    assert {"gri_agent", "dsm_agent", "sctd_agent", "bid_evaluator"} <= agents
    # at least one regional bidder recorded a sourcing action
    assert any(a["agent"].endswith("_agent") and a.get("action") == "sourcing"
               for a in final["audit_trail"])


# ── end-to-end: no-risk run ────────────────────────────────────────────────────

def test_no_risk_run_flows_clean_to_zero_gap():
    final = _invoke(NO_RISK_LLM)
    assert final["twin_state"]["total_india_shortfall_mbd"] == 0
    assert final["response_plan"]                # still produced a plan
    # bidders self-skip on zero gap → no committed cargo
    assert final["recommended_mix"].get("components") in ([], None)


# ── causal coordination through pheromones ─────────────────────────────────────
# The strong claim: a marker deposited upstream doesn't just make the field
# non-empty — it CHANGES a downstream agent's decision. The spot bidder prices
# its premium off _scarcity(field) = max pheromone intensity, so the same bidder,
# on the same gap, must bid HIGHER when a strong marker is present than when the
# field is empty. Everything else is held fixed; the marker pile is the only
# independent variable.

def _spot_bids_for_markers(markers):
    """Run the wrapped spot bidder over a fixed 1.0-mbd gap, varying ONLY the
    marker pile. wrap_bidder rebuilds the field the bidder sees from `markers`."""
    from agents.procurement.spot_market_agent import spot_market_node
    state = {**initial_state(), "twin_state": {"total_india_shortfall_mbd": 1.0},
             "stigmergy_markers": markers}
    node = wrap_bidder(spot_market_node, SPOT)
    with patch("agents.procurement._sourcing_base.fetch_price",
               return_value={"status": "ok", "data": {"current_price": 80.0}}):
        return node(state)["bids"]


def test_pheromone_marker_raises_spot_premium_causally():
    # Same bidder, same gap, same price — only the field differs.
    calm = {b["supplier_id"]: b for b in _spot_bids_for_markers([])}
    hot = {b["supplier_id"]: b for b in
           _spot_bids_for_markers([_fresh_marker("strait_of_hormuz", 0.9)])}

    # A clean (non-sanctioned) spot cargo to compare like-for-like.
    sid = "spot_bonny_light"
    assert calm[sid]["scarcity_surcharge_applied"] is False
    assert hot[sid]["scarcity_surcharge_applied"] is True
    # 0.9 scarcity × $5 max surcharge = +$4.5 on the premium (and thus the price).
    assert hot[sid]["price_premium_usd"] == pytest.approx(calm[sid]["price_premium_usd"] + 4.5, abs=1e-3)
    assert hot[sid]["price_per_bbl"] > calm[sid]["price_per_bbl"]


def test_stronger_marker_yields_stronger_response_monotonic():
    # The response tracks intensity: a hotter field prices strictly higher.
    sid = "spot_bonny_light"
    weak = {b["supplier_id"]: b for b in
            _spot_bids_for_markers([_fresh_marker("strait_of_hormuz", 0.3)])}[sid]
    strong = {b["supplier_id"]: b for b in
              _spot_bids_for_markers([_fresh_marker("strait_of_hormuz", 0.9)])}[sid]
    assert strong["price_premium_usd"] > weak["price_premium_usd"]


def test_full_run_spot_bid_carries_upstream_scarcity_signal():
    # End-to-end: GRI deposits the Hormuz risk marker → it survives DSM/SCTD →
    # the spot bidder, three superstesps later, prices against it. The coordination
    # is INDIRECT (no agent calls another) yet the signal provably lands.
    final = _invoke(HIGH_RISK_LLM)
    spot_bids = [b for b in final["bids"] if b.get("region") == "spot"]
    assert spot_bids, "spot bidder produced no bids"
    assert any(b["scarcity_surcharge_applied"] for b in spot_bids), \
        "spot never saw the upstream pheromone — causal channel is dead"

    # The effect is specifically pheromone-driven: the non-reactive regional
    # bidders on the SAME run carry no surcharge.
    non_spot = [b for b in final["bids"] if b.get("region") != "spot"]
    assert non_spot and all(not b["scarcity_surcharge_applied"] for b in non_spot)


# ── runner: answer now, learn in the background ──────────────────────────────────

def test_run_board_with_learning_returns_answer_and_fires_pod():
    # Real compiled graph, offline. learn_async is patched so no background thread /
    # network fires — we only assert the runner hands the pod the real final state.
    with patch("graph.workflow.learn_async") as la, _run_offline(HIGH_RISK_LLM):
        final = run_board_with_learning(
            "Iran closes the Strait of Hormuz",
            thread_id="learn-run",
        )
    assert final["response_plan"]                       # answer produced
    assert final["final_recommendation"]
    la.assert_called_once()                             # learning kicked off
    learned_state, _ = la.call_args
    assert learned_state[0] is final                    # …with THIS run's result


# ── twin sub-graph ─────────────────────────────────────────────────────────────

def test_twin_graph_runs_core_and_stops_after_sctd():
    graph = build_twin_graph()
    with _run_offline(HIGH_RISK_LLM):
        final = graph.invoke(
            initial_state(query="Hormuz"),
            config={"configurable": {"thread_id": "twin-run"}},
        )
    assert final["twin_state"], "twin not projected"
    # procurement never ran in the twin-only graph
    assert final["bids"] == []
    assert final["recommended_mix"] == {}
