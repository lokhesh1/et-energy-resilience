"""
Tests for the A2A layer (protocols/agent_cards.py + protocols/a2a_server.py).

All offline: the board runner is mocked, so no LLM / news / network. Covers:
  * agent_cards  - board card shape + skills, per-node registry, base-url override;
  * discovery    - /.well-known/agent.json, /a2a/card, /a2a/agents;
  * invocation   - /a2a/tasks/send accepts an A2A message AND a bare query, returns a
    completed Task with a curated artifact (no audit_trail/geojson leak), threads
    scenario_params + learn through, and downgrades a board failure to a `failed`
    task instead of a 500.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import protocols.a2a_server as a2a
from protocols.a2a_server import router, create_a2a_app
from protocols.agent_cards import board_card, agent_cards, BOARD_SKILLS


_FINAL = {
    "query": "Iran closes the Strait of Hormuz",
    "response_plan": {"escalation_level": "critical"},
    "final_recommendation": "CRITICAL: Hormuz war; West Africa cargo closes the gap.",
    "twin_state": {"total_india_shortfall_mbd": 1.0, "critical_count": 1,
                   "stressed_count": 0, "geojson": {"features": []}},
    "recommended_mix": {"covers_gap": True},
    "corridor_risk": {"strait_of_hormuz": 0.9},
    "constitution_flags": [],
    "audit_trail": [{"agent": "gri"}],
}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ── agent_cards ──────────────────────────────────────────────────────────────────

def test_board_card_shape_and_skills():
    card = board_card()
    assert card["name"] == "Energy Intelligence Board"
    assert card["capabilities"]["streaming"] is False
    ids = {s["id"] for s in card["skills"]}
    assert {"run_crisis_board", "assess_corridor_risk",
            "model_disruption", "source_shortfall"} <= ids
    assert card["skills"] is BOARD_SKILLS


def test_board_card_base_url_override():
    card = board_card("https://board.example.com/")
    assert card["url"] == "https://board.example.com/a2a"   # trailing slash trimmed


def test_agent_registry_lists_all_nodes():
    names = {c["name"] for c in agent_cards()}
    assert {"crisis_coordinator", "gri", "dsm", "sctd",
            "procurement", "distiller"} <= names
    # per-node cards are flagged internal (not external A2A peers)
    assert all(c["internal"] for c in agent_cards())


# ── discovery ────────────────────────────────────────────────────────────────────

def test_well_known_agent_card(client):
    card = client.get("/.well-known/agent.json").json()
    assert card["name"] == "Energy Intelligence Board"
    # url advertises the address actually serving it
    assert card["url"].endswith("/a2a")


def test_card_alias_matches_well_known(client):
    assert client.get("/a2a/card").json()["name"] == \
        client.get("/.well-known/agent.json").json()["name"]


def test_agents_registry_endpoint(client):
    names = {a["name"] for a in client.get("/a2a/agents").json()["agents"]}
    assert "crisis_coordinator" in names


# ── invocation ───────────────────────────────────────────────────────────────────

def test_tasks_send_runs_board_and_returns_completed(client, monkeypatch):
    monkeypatch.setattr(a2a, "run_board_with_learning", lambda *a, **k: _FINAL)
    body = client.post("/a2a/tasks/send", json={
        "id": "task-1",
        "message": {"role": "user",
                    "parts": [{"type": "text", "text": "Iran closes Hormuz"}]},
    }).json()

    assert body["id"] == "task-1"
    assert body["status"]["state"] == "completed"
    art = body["artifacts"][0]
    # text part = the recommendation
    assert art["parts"][0]["text"].startswith("CRITICAL")
    # data part = curated fields, NO raw audit_trail / geojson
    data = art["parts"][1]["data"]
    assert data["escalation_level"] == "critical"
    assert data["twin_summary"]["total_india_shortfall_mbd"] == 1.0
    assert "geojson" not in data["twin_summary"]
    assert "audit_trail" not in data


def test_tasks_send_accepts_bare_query_and_mints_id(client, monkeypatch):
    seen = {}
    def fake(query, scenario_params=None, learn=True, **k):
        seen.update(query=query, scenario_params=scenario_params, learn=learn)
        return _FINAL
    monkeypatch.setattr(a2a, "run_board_with_learning", fake)

    body = client.post("/a2a/tasks/send",
                       json={"query": "what if Suez closes", "learn": False}).json()
    assert seen["query"] == "what if Suez closes"
    assert seen["learn"] is False
    assert body["id"]                       # minted a uuid
    assert body["status"]["state"] == "completed"


def test_tasks_send_threads_scenario_params(client, monkeypatch):
    seen = {}
    def fake(query, scenario_params=None, learn=True, **k):
        seen["scenario_params"] = scenario_params
        return _FINAL
    monkeypatch.setattr(a2a, "run_board_with_learning", fake)
    client.post("/a2a/tasks/send",
                json={"query": "q", "scenario_params": {"shock": 0.5}})
    assert seen["scenario_params"] == {"shock": 0.5}


def test_tasks_send_downgrades_board_failure_to_failed_task(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("board blew up")
    monkeypatch.setattr(a2a, "run_board_with_learning", boom)

    r = client.post("/a2a/tasks/send", json={"query": "q"})
    assert r.status_code == 200                          # not a 500
    body = r.json()
    assert body["status"]["state"] == "failed"
    assert "board blew up" in body["status"]["message"]["parts"][0]["text"]
    assert body["artifacts"] == []


# ── standalone app ───────────────────────────────────────────────────────────────

def test_standalone_app_serves_card():
    c = TestClient(create_a2a_app())
    assert c.get("/.well-known/agent.json").json()["name"] == "Energy Intelligence Board"
