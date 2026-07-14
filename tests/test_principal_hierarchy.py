"""
Tests for eib_guardrails/principal_hierarchy.py + its two enforcement points.

Covers:
  * the grant table: strict nesting (external ⊂ operator ⊂ system), the specific
    denials that define the external boundary (refresh_twin, read_audit,
    write_memory);
  * closed-world / least-privilege: unknown principal and unknown capability are
    both denied, `require` raises on denial;
  * A2A wiring: `tasks/send` passes the gate before the board runs — a denial
    (table patched) comes back as a `failed` Task without invoking the board;
  * REST wiring: /twin/refresh is allowed for the operator, 403 when denied.
"""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eib_guardrails.principal_hierarchy import (
    CAPABILITIES, PRINCIPALS, check_permission, grants, require,
)
import protocols.a2a_server as a2a


# ── The grant table ─────────────────────────────────────────────────────────────

def test_grants_are_strictly_nested():
    # everything a less-trusted principal may do, a more-trusted one may too
    assert grants("external_agent") < grants("operator") < grants("system")
    assert grants("system") == CAPABILITIES


def test_external_agent_boundary():
    assert check_permission("external_agent", "run_board")["allowed"] is True
    assert check_permission("external_agent", "discover")["allowed"] is True
    for denied in ("refresh_twin", "read_audit", "write_memory"):
        assert check_permission("external_agent", denied)["allowed"] is False


def test_operator_cannot_write_memory_directly():
    assert check_permission("operator", "refresh_twin")["allowed"] is True
    assert check_permission("operator", "write_memory")["allowed"] is False


def test_unknown_principal_and_capability_denied():
    assert check_permission("mystery_caller", "run_board")["allowed"] is False
    assert check_permission("operator", "launch_missiles")["allowed"] is False


def test_require_raises_on_denial_and_passes_on_grant():
    require("system", "write_memory")  # no raise
    with pytest.raises(PermissionError):
        require("external_agent", "read_audit")


def test_every_grant_is_a_known_capability():
    for p in PRINCIPALS:
        assert grants(p) <= CAPABILITIES


# ── A2A enforcement (external_agent at tasks/send) ──────────────────────────────

@pytest.fixture
def a2a_client():
    app = FastAPI()
    app.include_router(a2a.router)
    return TestClient(app)


def test_a2a_denial_is_a_failed_task_and_board_never_runs(a2a_client, monkeypatch):
    board = MagicMock()
    monkeypatch.setattr(a2a, "run_board_with_learning", board)
    monkeypatch.setattr(
        a2a, "check_permission",
        lambda p, c: {"allowed": False, "principal": p, "capability": c,
                      "reason": "'external_agent' is not granted 'run_board'"})

    body = a2a_client.post("/a2a/tasks/send", json={"query": "q"}).json()
    assert body["status"]["state"] == "failed"
    msg = body["status"]["message"]["parts"][0]["text"]
    assert "permission denied" in msg
    board.assert_not_called()


def test_a2a_allowed_path_runs_the_board(a2a_client, monkeypatch):
    final = {"final_recommendation": "ok", "response_plan": {}, "twin_state": {},
             "corridor_risk": {}, "recommended_mix": {}, "constitution_flags": []}
    monkeypatch.setattr(a2a, "run_board_with_learning", lambda *a, **k: final)
    body = a2a_client.post("/a2a/tasks/send", json={"query": "q"}).json()
    assert body["status"]["state"] == "completed"


# ── REST enforcement (/twin/refresh as operator) ────────────────────────────────

@pytest.fixture
def api_client():
    import api.main as m
    return m, TestClient(m.app)


def test_twin_refresh_allowed_for_operator(api_client, monkeypatch):
    m, client = api_client
    monkeypatch.setattr(m.tl, "refresh_twin", lambda: {"status": "ok"})
    assert client.post("/twin/refresh").json() == {"status": "ok"}


def test_twin_refresh_403_when_denied(api_client, monkeypatch):
    m, client = api_client
    monkeypatch.setattr(
        m, "check_permission",
        lambda p, c: {"allowed": False, "principal": p, "capability": c,
                      "reason": "revoked"})
    r = client.post("/twin/refresh")
    assert r.status_code == 403
    assert r.json()["detail"] == "revoked"
