"""
Tests for the FastAPI backend (api/main.py) + the continuous twin loop
(api/twin_loop.py).

All offline: the board run and the twin graph are mocked, so no LLM/news/network.
Covers:
  * TwinSnapshot — cold read, ok update, and the key safety property: a failed
    refresh keeps the last good twin instead of blanking it;
  * refresh_twin — stores a projection; never raises on a graph failure;
  * the lifespan — launches the loop when enabled, cancels it on shutdown;
  * the endpoints — /health, /agents, /query, /scenario, /twin, /twin/refresh.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

import api.main as main
from api.main import app, _summarize, lifespan
from api import twin_loop as tl
from api.twin_loop import TwinSnapshot, refresh_twin


# ── TwinSnapshot ─────────────────────────────────────────────────────────────────

def test_snapshot_starts_cold():
    snap = TwinSnapshot().read()
    assert snap["status"] == "cold"
    assert snap["refresh_count"] == 0
    assert snap["twin_state"] == {}


def test_snapshot_update_ok_records_twin_and_meta():
    s = TwinSnapshot()
    s.update_ok({"total_india_shortfall_mbd": 1.2})
    snap = s.read()
    assert snap["status"] == "ok"
    assert snap["refresh_count"] == 1
    assert snap["twin_state"]["total_india_shortfall_mbd"] == 1.2
    assert snap["last_error"] is None


def test_failed_refresh_preserves_last_good_twin():
    # The safety property: a stale-but-real twin beats a blank one.
    s = TwinSnapshot()
    s.update_ok({"total_india_shortfall_mbd": 2.0})
    s.update_error("BoomError: news feed down")
    snap = s.read()
    assert snap["status"] == "stale"                       # not "cold"
    assert snap["twin_state"]["total_india_shortfall_mbd"] == 2.0  # kept
    assert "BoomError" in snap["last_error"]


def test_error_before_any_good_refresh_stays_cold():
    s = TwinSnapshot()
    s.update_error("startup failure")
    assert s.read()["status"] == "cold"                    # never had a good twin


def test_read_returns_a_copy():
    s = TwinSnapshot()
    s.update_ok({"k": 1})
    snap = s.read()
    snap["twin_state"]["k"] = 999                          # mutate the copy
    assert s.read()["twin_state"]["k"] == 1                # source untouched


# ── refresh_twin ─────────────────────────────────────────────────────────────────

def test_refresh_twin_stores_projection(monkeypatch):
    fake_graph = _fake_graph({"twin_state": {"total_india_shortfall_mbd": 3.3}})
    monkeypatch.setattr(tl, "_get_twin_graph", lambda: fake_graph)
    monkeypatch.setattr(tl, "snapshot", TwinSnapshot())

    out = refresh_twin()
    assert out["status"] == "ok"
    assert out["twin_state"]["total_india_shortfall_mbd"] == 3.3


def test_refresh_twin_survives_graph_failure(monkeypatch):
    boom = _fake_graph(None, raises=RuntimeError("twin blew up"))
    monkeypatch.setattr(tl, "_get_twin_graph", lambda: boom)
    fresh = TwinSnapshot()
    monkeypatch.setattr(tl, "snapshot", fresh)

    out = refresh_twin()                                   # must not raise
    assert out["status"] == "cold"
    assert "twin blew up" in out["last_error"]


# ── lifespan (loop start/stop) ───────────────────────────────────────────────────

def test_lifespan_starts_and_cancels_loop(monkeypatch):
    calls = {"started": False, "cancelled": False}

    async def fake_loop(interval):
        calls["started"] = True
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            calls["cancelled"] = True
            raise

    monkeypatch.setattr(main.settings, "TWIN_LOOP_ENABLED", True)
    monkeypatch.setattr(main.tl, "twin_loop", fake_loop)

    async def drive():
        async with lifespan(app):
            for _ in range(20):                            # let the task get scheduled
                if calls["started"]:
                    break
                await asyncio.sleep(0.01)
        # context exit cancels + awaits the task

    asyncio.run(drive())
    assert calls["started"] and calls["cancelled"]


def test_lifespan_skips_loop_when_disabled(monkeypatch):
    called = {"n": 0}

    async def fake_loop(interval):
        called["n"] += 1

    monkeypatch.setattr(main.settings, "TWIN_LOOP_ENABLED", False)
    monkeypatch.setattr(main.tl, "twin_loop", fake_loop)

    async def drive():
        async with lifespan(app):
            await asyncio.sleep(0.02)

    asyncio.run(drive())
    assert called["n"] == 0


# ── endpoints ────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    # never launch the real loop during endpoint tests
    monkeypatch.setattr(main.settings, "TWIN_LOOP_ENABLED", False)
    with TestClient(app) as c:
        yield c


_FINAL = {
    "query": "Iran closes the Strait of Hormuz",
    "response_plan": {"escalation_level": "critical",
                      "situation": {"gap_mbd": 1.0},
                      "procurement": {"covered_mbd": 1.0, "residual_gap_mbd": 0.0}},
    "final_recommendation": "CRITICAL: Hormuz war; West Africa cargo closes the gap.",
    "twin_state": {"total_india_shortfall_mbd": 1.0, "critical_count": 1,
                   "stressed_count": 0, "geojson": {"features": []}},
    "recommended_mix": {"covers_gap": True},
    "retrieved_memories": [],
    "constitution_flags": [],
    "corridor_risk": {"strait_of_hormuz": 0.9},
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_agents_lists_the_board(client):
    names = {a["name"] for a in client.get("/agents").json()["agents"]}
    assert {"crisis_coordinator", "gri", "dsm", "sctd", "procurement", "distiller"} <= names


def test_query_runs_board_and_summarizes(client, monkeypatch):
    monkeypatch.setattr(main, "run_board_with_learning", lambda *a, **k: _FINAL)
    body = client.post("/query", json={"query": "Iran closes Hormuz"}).json()
    assert body["escalation_level"] == "critical"
    assert body["twin_summary"]["total_india_shortfall_mbd"] == 1.0
    assert body["final_recommendation"].startswith("CRITICAL")
    # curated: the raw geojson blob is NOT dumped into a /query response
    assert "geojson" not in body["twin_summary"]


def test_query_honours_learn_flag(client, monkeypatch):
    seen = {}
    def fake(query, learn=True, consolidate=True, **k):
        seen.update(query=query, learn=learn, consolidate=consolidate)
        return _FINAL
    monkeypatch.setattr(main, "run_board_with_learning", fake)
    client.post("/query", json={"query": "q", "learn": False})
    assert seen["learn"] is False


def test_scenario_passes_params_through(client, monkeypatch):
    seen = {}
    def fake(query, scenario_params=None, learn=False, **k):
        seen.update(query=query, scenario_params=scenario_params, learn=learn)
        return _FINAL
    monkeypatch.setattr(main, "run_board_with_learning", fake)
    client.post("/scenario", json={"query": "what if", "scenario_params": {"shock": 0.5}})
    assert seen["scenario_params"] == {"shock": 0.5}
    assert seen["learn"] is False                          # what-ifs don't learn by default


def test_corridor_status_wraps_the_tool(client, monkeypatch):
    monkeypatch.setattr(main, "get_corridor_status",
                        lambda: {"tool": "corridor_status", "status": "ok",
                                 "data": {"corridors": [{"id": "strait_of_hormuz"}],
                                          "highest_risk_corridor": "strait_of_hormuz"}})
    body = client.get("/corridor-status").json()
    assert body["status"] == "ok"
    assert body["data"]["highest_risk_corridor"] == "strait_of_hormuz"


def test_corridor_status_is_live_by_default(client):
    # No mock: hits the real offline tool (reads data/corridors.json) — 8 corridors.
    body = client.get("/corridor-status").json()
    assert body["status"] == "ok"
    assert len(body["data"]["corridors"]) == 8


def test_twin_serves_latest_snapshot(client, monkeypatch):
    snap = TwinSnapshot()
    snap.update_ok({"total_india_shortfall_mbd": 4.2})
    monkeypatch.setattr(main.tl, "snapshot", snap)
    body = client.get("/twin").json()
    assert body["status"] == "ok"
    assert body["twin_state"]["total_india_shortfall_mbd"] == 4.2


def test_twin_refresh_endpoint_triggers_refresh(client, monkeypatch):
    monkeypatch.setattr(main.tl, "refresh_twin",
                        lambda: {"status": "ok", "twin_state": {"forced": True}})
    body = client.post("/twin/refresh").json()
    assert body["twin_state"]["forced"] is True


# ── helpers ──────────────────────────────────────────────────────────────────────

class _FakeGraph:
    def __init__(self, result, raises=None):
        self._result = result
        self._raises = raises

    def invoke(self, state, config=None):
        if self._raises:
            raise self._raises
        return self._result


def _fake_graph(result, raises=None):
    return _FakeGraph(result, raises)


def test_summarize_curates_state():
    out = _summarize(_FINAL)
    assert set(out) >= {"escalation_level", "final_recommendation", "response_plan",
                        "twin_summary", "recommended_mix", "corridor_risk"}
    assert "audit_trail" not in out
