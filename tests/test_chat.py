"""
Tests for the multi-turn chat layer (api/chat.py), the extended summary
(api/summary.py), the /audit/verify endpoint (api/main.py), and the
pure Folium builder (ui/map_view.py).

All offline: the board run and LLM calls are mocked — no network.
"""
import json

import pytest
from fastapi.testclient import TestClient

import api.main as main
import api.chat as chat
from api.main import app
from api.chat import ChatStore
from api.summary import summarize_final, build_components, suggest_follow_ups


# ── fixtures ────────────────────────────────────────────────────────────────────

_FINAL = {
    "query": "Iran closes the Strait of Hormuz",
    "response_plan": {
        "escalation_level": "critical",
        "situation": {
            "top_corridor_risks": [
                {"corridor": "strait_of_hormuz", "score": 0.9, "event_type": "war_conflict"},
            ],
            "gap_mbd": 3.2,
            "critical_refineries": ["Jamnagar"],
            "stressed_refineries": ["Mangalore"],
            "disrupted_corridors": ["strait_of_hormuz"],
        },
        "procurement": {
            "covered_mbd": 3.0,
            "coverage_ratio": 0.94,
            "covers_gap": False,
            "residual_gap_mbd": 0.2,
            "committed_actions": [
                {"supplier": "NNPC", "region": "west_africa", "grade": "Bonny Light",
                 "volume_mbd": 1.5, "price_per_bbl": 82.0,
                 "delivery_corridor": "cape_of_good_hope", "transit_days": 25,
                 "sanctions_status": "clear"},
                {"supplier": "Petrobras", "region": "americas", "grade": "Tupi",
                 "volume_mbd": 1.5, "price_per_bbl": 84.0,
                 "delivery_corridor": "cape_of_good_hope", "transit_days": 30,
                 "sanctions_status": "clear"},
            ],
            "spr_bridge": {
                "drawdown_mbd": 0.2, "days_of_cover": 90,
                "bridge_fraction": 1.0, "unbridged_mbd": 0.0,
            },
        },
        "priority_actions": ["Secure 1.5 mbd Bonny Light from NNPC."],
        "unresolved_issues": [],
    },
    "final_recommendation": "CRITICAL: Hormuz blockade; West Africa + Americas cargo closes the gap.",
    "twin_state": {
        "total_india_shortfall_mbd": 3.2,
        "critical_count": 1,
        "stressed_count": 1,
        "refineries": [
            {"name": "Jamnagar", "status": "critical"},
            {"name": "Mangalore", "status": "stressed"},
        ],
        "corridors": [
            {"id": "strait_of_hormuz", "disruption_fraction": 1.0},
        ],
        "geojson": {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [56.0, 26.5]},
                 "properties": {"kind": "corridor", "tooltip": "Hormuz", "marker_color": "red"}},
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [72.8, 22.3]},
                 "properties": {"kind": "refinery", "tooltip": "Jamnagar", "marker_color": "red"}},
            ],
        },
    },
    "recommended_mix": {"covers_gap": False, "total_volume_mbd": 3.0,
                        "coverage_ratio": 0.94,
                        "components": [
                            {"supplier": "NNPC", "volume_mbd": 1.5},
                            {"supplier": "Petrobras", "volume_mbd": 1.5},
                        ]},
    "retrieved_memories": [],
    "constitution_flags": [],
    "corridor_risk": {"strait_of_hormuz": 0.9, "suez_canal": 0.1},
    "stigmergy_markers": [
        {"type": "risk", "target": "strait_of_hormuz", "intensity": 0.9,
         "deposited_by": "gri", "timestamp": "2026-07-14T00:00:00Z", "decay_rate": 0.1},
    ],
    "pheromone_field": {"strait_of_hormuz": 0.85},
    "audit_trail": [{"agent": "gri", "action": "scored"}],
    "scenarios": [{"corridor": "strait_of_hormuz", "duration_days": 42}],
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.settings, "TWIN_LOOP_ENABLED", False)
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _fresh_chat_store(monkeypatch):
    """Each test gets a clean ChatStore so sessions don't leak across tests."""
    monkeypatch.setattr(chat, "store", ChatStore())


def _mock_board(monkeypatch, final=None):
    """Patch run_board_with_learning on the chat module namespace."""
    calls = []
    def fake(*a, **k):
        calls.append({"args": a, "kwargs": k})
        return final or _FINAL
    monkeypatch.setattr(chat, "run_board_with_learning", fake)
    return calls


def _mock_route(monkeypatch, intent="run_board", query="standalone Q"):
    """Patch the router LLM."""
    monkeypatch.setattr(chat, "_route",
                        lambda msg, turns: {"intent": intent,
                                            "standalone_query": query})


def _mock_answer(monkeypatch, reply="Grounded answer."):
    """Patch the answer-from-digest LLM."""
    monkeypatch.setattr(chat, "_answer_from_digest",
                        lambda msg, digest, turns: reply)


# ── ChatStore unit tests ────────────────────────────────────────────────────────

def test_chat_mints_session_id(client, monkeypatch):
    _mock_board(monkeypatch)
    r = client.post("/chat", json={"message": "Hormuz crisis"})
    body = r.json()
    assert r.status_code == 200
    assert body["session_id"]
    assert len(body["session_id"]) == 32  # uuid4().hex


def test_chat_reuses_session(client, monkeypatch):
    _mock_board(monkeypatch)
    r1 = client.post("/chat", json={"message": "Hormuz crisis"})
    sid = r1.json()["session_id"]
    _mock_route(monkeypatch, intent="answer_from_last_run")
    _mock_answer(monkeypatch)
    r2 = client.post("/chat", json={"session_id": sid, "message": "Why Bonny Light?"})
    assert r2.json()["session_id"] == sid


# ── Router + intent gating ──────────────────────────────────────────────────────

def test_first_turn_skips_router_runs_board(client, monkeypatch):
    calls = _mock_board(monkeypatch)
    # Mock _client to track if the router LLM was called
    llm_calls = []
    orig_route = chat._route
    monkeypatch.setattr(chat, "_route",
                        lambda msg, turns: (llm_calls.append(1) or
                                            orig_route(msg, turns)))
    r = client.post("/chat", json={"message": "Hormuz crisis"})
    assert r.json()["mode"] == "run_board"
    assert len(calls) == 1
    assert len(llm_calls) == 0  # router was never invoked


def test_run_board_uses_rewritten_standalone_query(client, monkeypatch):
    calls = _mock_board(monkeypatch)
    # Seed a first run
    client.post("/chat", json={"message": "Hormuz crisis"})
    sid = client.post("/chat", json={"message": "Hormuz crisis"}).json()["session_id"]

    # Second turn: router rewrites
    _mock_route(monkeypatch, intent="run_board", query="Hormuz closed; Americas only")
    calls.clear()
    r = client.post("/chat", json={"session_id": sid, "message": "Americas only?"})
    assert r.json()["mode"] == "run_board"
    # The rewritten query is what reached the board
    assert calls[0]["args"][0] == "Hormuz closed; Americas only"


def test_answer_mode_skips_board(client, monkeypatch):
    calls = _mock_board(monkeypatch)
    r1 = client.post("/chat", json={"message": "Hormuz crisis"})
    sid = r1.json()["session_id"]
    calls.clear()

    _mock_route(monkeypatch, intent="answer_from_last_run")
    _mock_answer(monkeypatch, reply="Bonny Light is cheapest impact.")
    r2 = client.post("/chat", json={"session_id": sid, "message": "Why Bonny Light?"})
    body = r2.json()
    assert body["mode"] == "answer_from_last_run"
    assert len(calls) == 0  # board NOT invoked
    assert body["run_summary"] is None
    assert "Bonny Light" in body["reply"]


def test_answer_grounded_on_digest_only(client, monkeypatch):
    _mock_board(monkeypatch)
    r1 = client.post("/chat", json={"message": "Hormuz crisis"})
    sid = r1.json()["session_id"]

    captured = {}
    def fake_answer(msg, digest, turns):
        captured["digest"] = digest
        return "Grounded."
    monkeypatch.setattr(chat, "_answer_from_digest", fake_answer)
    _mock_route(monkeypatch, intent="answer_from_last_run")

    client.post("/chat", json={"session_id": sid, "message": "explain"})
    d = captured["digest"]
    # Digest has the trajectory keys, NOT raw geojson/audit
    assert "corridor_risks" in d
    assert "twin" in d
    assert "procurement" in d
    assert "geojson" not in d
    assert "audit_trail" not in d


# ── Fallbacks ───────────────────────────────────────────────────────────────────

def test_router_failure_falls_back_to_run_board(client, monkeypatch):
    """Broken LLM client → _route catches internally → run_board with raw message."""
    calls = _mock_board(monkeypatch)
    r1 = client.post("/chat", json={"message": "Hormuz crisis"})
    sid = r1.json()["session_id"]

    class BrokenClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("LLM down")
    monkeypatch.setattr(chat, "_client", BrokenClient())
    calls.clear()
    r2 = client.post("/chat", json={"session_id": sid, "message": "new crisis"})
    assert r2.status_code == 200
    assert r2.json()["mode"] == "run_board"
    assert calls[0]["args"][0] == "new crisis"


def test_router_internal_fallback(monkeypatch):
    """_route itself catches exceptions and falls back to run_board."""
    # Patch the OpenAI client to raise
    class BrokenClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("LLM down")
    monkeypatch.setattr(chat, "_client", BrokenClient())
    result = chat._route("new crisis", [{"role": "user", "content": "hello"}])
    assert result["intent"] == "run_board"
    assert result["standalone_query"] == "new crisis"


def test_answer_llm_failure_template_fallback(client, monkeypatch):
    _mock_board(monkeypatch)
    r1 = client.post("/chat", json={"message": "Hormuz crisis"})
    sid = r1.json()["session_id"]

    _mock_route(monkeypatch, intent="answer_from_last_run")
    monkeypatch.setattr(chat, "_answer_from_digest",
                        lambda msg, digest, turns: None)  # LLM failed

    r2 = client.post("/chat", json={"session_id": sid, "message": "explain"})
    body = r2.json()
    # Template fallback contains escalation + numbers from the stored summary
    assert "CRITICAL" in body["reply"]
    assert "3.2" in body["reply"]  # gap


# ── Components ──────────────────────────────────────────────────────────────────

def test_run_turn_components_shape(client, monkeypatch):
    _mock_board(monkeypatch)
    r = client.post("/chat", json={"message": "Hormuz crisis"})
    body = r.json()
    types = {c["type"] for c in body["components"]}
    assert types == {"map", "metrics", "mix_table", "follow_ups"}

    # Map geojson is from twin_state
    map_c = next(c for c in body["components"] if c["type"] == "map")
    assert len(map_c["geojson"]["features"]) == 2
    assert map_c["counts"]["corridor"] == 1
    assert map_c["counts"]["refinery"] == 1

    # Metrics values match the twin
    metrics_c = next(c for c in body["components"] if c["type"] == "metrics")
    labels = {it["label"] for it in metrics_c["items"]}
    assert "Escalation" in labels
    assert "India shortfall" in labels
    shortfall = next(it for it in metrics_c["items"]
                     if it["label"] == "India shortfall")
    assert shortfall["value"] == 3.2

    # Mix table has the committed actions
    mix_c = next(c for c in body["components"] if c["type"] == "mix_table")
    assert len(mix_c["rows"]) == 2
    assert mix_c["rows"][0]["supplier"] == "NNPC"
    assert mix_c["spr_bridge"]["days_of_cover"] == 90


def test_answer_turn_reuses_last_components(client, monkeypatch):
    _mock_board(monkeypatch)
    r1 = client.post("/chat", json={"message": "Hormuz crisis"})
    sid = r1.json()["session_id"]
    orig_components = r1.json()["components"]

    _mock_route(monkeypatch, intent="answer_from_last_run")
    _mock_answer(monkeypatch)
    r2 = client.post("/chat", json={"session_id": sid, "message": "explain"})
    assert r2.json()["components"] == orig_components


# ── build_components / suggest_follow_ups unit tests ────────────────────────────

def test_build_components_deterministic():
    summary = summarize_final(_FINAL)
    twin = _FINAL["twin_state"]
    c1 = build_components(summary, twin)
    c2 = build_components(summary, twin)
    assert c1 == c2


def test_build_components_omits_map_when_no_features():
    modified = {**_FINAL, "twin_state": {"geojson": {"features": []}}}
    summary = summarize_final(modified)
    comps = build_components(summary, modified["twin_state"])
    types = {c["type"] for c in comps}
    assert "map" not in types
    assert "metrics" in types


def test_build_components_omits_mix_table_on_quiet_board():
    quiet = {
        **_FINAL,
        "response_plan": {"escalation_level": "routine",
                          "procurement": {"committed_actions": [],
                                          "residual_gap_mbd": 0}},
        "twin_state": {**_FINAL["twin_state"],
                       "total_india_shortfall_mbd": 0,
                       "critical_count": 0, "stressed_count": 0},
    }
    summary = summarize_final(quiet)
    comps = build_components(summary, quiet["twin_state"])
    types = {c["type"] for c in comps}
    assert "mix_table" not in types


def test_follow_ups_residual_gap():
    summary = summarize_final(_FINAL)
    fups = suggest_follow_ups(summary)
    assert any("SPR" in f for f in fups)
    assert len(fups) <= 4


def test_follow_ups_quiet_board():
    quiet = {
        **_FINAL,
        "response_plan": {"escalation_level": "routine",
                          "procurement": {"committed_actions": [],
                                          "residual_gap_mbd": 0}},
        "twin_state": {**_FINAL["twin_state"],
                       "total_india_shortfall_mbd": 0,
                       "critical_count": 0, "stressed_count": 0},
        "corridor_risk": {},
    }
    fups = suggest_follow_ups(summarize_final(quiet))
    assert any("Hormuz" in f for f in fups)


# ── Checkpointer + thread IDs ──────────────────────────────────────────────────

def test_shared_checkpointer_and_thread_ids(client, monkeypatch):
    seen = []
    def fake(query, thread_id="default", checkpointer=None, learn=True, **kw):
        seen.append({"query": query, "thread_id": thread_id,
                     "checkpointer": checkpointer})
        return _FINAL
    monkeypatch.setattr(chat, "run_board_with_learning", fake)

    r1 = client.post("/chat", json={"message": "turn 1"})
    sid = r1.json()["session_id"]
    _mock_route(monkeypatch, intent="run_board", query="standalone turn 2")
    r2 = client.post("/chat", json={"session_id": sid, "message": "turn 2"})

    assert seen[0]["thread_id"] == f"chat-{sid}-t1"
    assert seen[1]["thread_id"] == f"chat-{sid}-t2"
    assert seen[0]["checkpointer"] is chat._CHECKPOINTER
    assert seen[1]["checkpointer"] is chat._CHECKPOINTER


# ── History budget ──────────────────────────────────────────────────────────────

def test_history_budget(monkeypatch):
    s = ChatStore()
    sid = s.ensure(None)
    for i in range(12):
        s.append_turn(sid, "user", f"msg {i}")
        s.append_turn(sid, "assistant", f"reply {i}")
    ctx = s.context(sid)
    # Only the last CHAT_HISTORY_TURNS turns in the context
    from config.settings import CHAT_HISTORY_TURNS
    assert len(ctx["turns"]) == CHAT_HISTORY_TURNS
    # But all 24 are stored
    assert s._sessions[sid]["turns"][-1]["content"] == "reply 11"


# ── learn flag passthrough ──────────────────────────────────────────────────────

def test_learn_flag_passthrough(client, monkeypatch):
    seen = {}
    def fake(*a, **kw):
        seen.update(kw)
        return _FINAL
    monkeypatch.setattr(chat, "run_board_with_learning", fake)
    client.post("/chat", json={"message": "test", "learn": False})
    assert seen["learn"] is False


# ── summarize_final adds stigmergy ──────────────────────────────────────────────

def test_summarize_final_adds_stigmergy():
    out = summarize_final(_FINAL)
    assert "pheromone_field" in out
    assert out["pheromone_field"]["strait_of_hormuz"] == 0.85
    assert "stigmergy" in out
    assert out["stigmergy"]["marker_count"] == 1
    assert out["stigmergy"]["top_markers"][0]["target"] == "strait_of_hormuz"
    # audit_trail still excluded
    assert "audit_trail" not in out


# ── /audit/verify endpoint ──────────────────────────────────────────────────────

def test_audit_verify_ok(client, monkeypatch):
    monkeypatch.setattr(main, "verify_chain",
                        lambda **kw: {"status": "ok", "valid": True, "entries": 3})
    r = client.get("/audit/verify")
    assert r.status_code == 200
    assert r.json()["valid"] is True


def test_audit_verify_denied_403(client, monkeypatch):
    monkeypatch.setattr(main, "check_permission",
                        lambda p, c: {"allowed": False, "principal": p,
                                      "capability": c, "reason": "denied"})
    r = client.get("/audit/verify")
    assert r.status_code == 403


# ── ui/map_view.py (pure Folium, no Streamlit) ─────────────────────────────────

from ui.map_view import build_folium_map, feature_counts


def test_map_view_builds_map():
    geojson = _FINAL["twin_state"]["geojson"]
    m = build_folium_map(geojson)
    html = m._repr_html_()
    assert "Hormuz" in html or "Digital Twin" in html
    assert isinstance(m, __import__("folium").Map)


def test_map_view_empty_geojson_never_raises():
    m = build_folium_map({})
    assert isinstance(m, __import__("folium").Map)
    m2 = build_folium_map(None)
    assert isinstance(m2, __import__("folium").Map)
    m3 = build_folium_map({"features": []})
    assert isinstance(m3, __import__("folium").Map)


def test_map_view_feature_counts():
    geojson = _FINAL["twin_state"]["geojson"]
    counts = feature_counts(geojson)
    assert counts["corridor"] == 1
    assert counts["refinery"] == 1
    assert feature_counts({}) == {}
    assert feature_counts(None) == {}
