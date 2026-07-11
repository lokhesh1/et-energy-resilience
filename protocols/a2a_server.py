"""
A2A server — the board's front door for OTHER agents.

This is a thin protocol adapter, not a second brain: it exposes the Energy
Intelligence Board over the Agent-to-Agent (A2A) protocol so an external agent can
(1) DISCOVER the board via its capability card and (2) INVOKE it with a task. Both
map straight onto machinery we already have:

  * discovery  -> the static cards in `agent_cards.py`
  * invocation -> `graph.workflow.run_board_with_learning` (the same runner the
    REST /query endpoint uses)

We implement the minimal, honest A2A subset for a demo:
  * GET  /.well-known/agent.json  - agent-card discovery (spec's well-known path)
  * GET  /a2a/card                - same card, convenience alias
  * GET  /a2a/agents              - the per-node capability registry
  * POST /a2a/tasks/send          - synchronous task: run the board, return a Task

Deliberately deferred (not needed to demo): streaming (`tasks/sendSubscribe`), the
full task lifecycle/polling, push notifications, and auth. `capabilities.streaming`
is advertised False so a compliant client won't expect them.

Mount into the main app:  `app.include_router(a2a_server.router)`  — or run it
standalone via `create_a2a_app()`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel, Field

from graph.workflow import run_board_with_learning
from protocols.agent_cards import board_card, agent_cards


router = APIRouter(tags=["a2a"])


# ── A2A message/task models (minimal subset) ─────────────────────────────────────

class Part(BaseModel):
    type: str = "text"
    text: str = ""


class Message(BaseModel):
    role: str = "user"
    parts: list[Part] = Field(default_factory=list)


class TaskSendRequest(BaseModel):
    """An A2A `tasks/send` payload. `message.parts[*].text` carries the query; we
    also accept a bare top-level `query` for convenience. `id` is optional (the
    client's task id); we mint one if absent."""
    id: str | None = None
    message: Message | None = None
    query: str | None = None
    scenario_params: dict = Field(default_factory=dict)
    learn: bool = Field(True, description="Fire the distiller pod after the run.")


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _base_url(request: Request) -> str:
    """The address this server is actually reachable at, so the card advertises the
    right URL rather than a hard-coded default."""
    return str(request.base_url).rstrip("/")


def _extract_text(req: TaskSendRequest) -> str:
    """Pull the user's text out of an A2A message (or the convenience `query`)."""
    if req.query:
        return req.query
    if req.message and req.message.parts:
        return " ".join(p.text for p in req.message.parts if p.text).strip()
    return ""


def _artifact_from_board(final: dict) -> dict:
    """Turn the big board state into one A2A artifact: a human-readable text part
    (the final recommendation) + a compact data part (the decision-relevant fields).
    The heavy audit_trail / geojson are intentionally NOT shipped over A2A."""
    plan = final.get("response_plan", {}) or {}
    twin = final.get("twin_state", {}) or {}
    data = {
        "escalation_level":     plan.get("escalation_level"),
        "corridor_risk":        final.get("corridor_risk", {}),
        "twin_summary": {
            "total_india_shortfall_mbd": twin.get("total_india_shortfall_mbd"),
            "critical_count":            twin.get("critical_count"),
            "stressed_count":            twin.get("stressed_count"),
        },
        "recommended_mix":     final.get("recommended_mix", {}),
        "constitution_flags":  final.get("constitution_flags", []),
    }
    return {
        "name": "response_plan",
        "parts": [
            {"type": "text", "text": final.get("final_recommendation", "")},
            {"type": "data", "data": data},
        ],
    }


def _completed_task(task_id: str, text: str, artifact: dict) -> dict:
    """An A2A Task in terminal `completed` state (sync path — no polling needed)."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": task_id,
        "status": {"state": "completed", "timestamp": now},
        "history": [
            {"role": "user", "parts": [{"type": "text", "text": text}]},
        ],
        "artifacts": [artifact],
    }


def _failed_task(task_id: str, text: str, error: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": task_id,
        "status": {
            "state": "failed",
            "timestamp": now,
            "message": {"role": "agent",
                        "parts": [{"type": "text", "text": error}]},
        },
        "history": [{"role": "user", "parts": [{"type": "text", "text": text}]}],
        "artifacts": [],
    }


# ── Discovery ────────────────────────────────────────────────────────────────────

@router.get("/.well-known/agent.json")
def well_known_agent_card(request: Request) -> dict:
    """A2A discovery: the board's capability card at the spec's well-known path."""
    return board_card(_base_url(request))


@router.get("/a2a/card")
def a2a_card(request: Request) -> dict:
    """Convenience alias for the board card."""
    return board_card(_base_url(request))


@router.get("/a2a/agents")
def a2a_agents(request: Request) -> dict:
    """The per-node capability registry (internal descriptors, not A2A peers)."""
    return {"agents": agent_cards(_base_url(request))}


# ── Invocation ───────────────────────────────────────────────────────────────────

@router.post("/a2a/tasks/send")
def tasks_send(req: TaskSendRequest) -> dict:
    """Synchronous A2A task: run the full board on the message text and return a
    completed Task with the response plan as an artifact. Best-effort — a board
    failure comes back as a `failed` task, not an HTTP 500."""
    task_id = req.id or str(uuid.uuid4())
    text = _extract_text(req)
    try:
        final = run_board_with_learning(
            text, scenario_params=req.scenario_params or None, learn=req.learn,
        )
        return _completed_task(task_id, text, _artifact_from_board(final))
    except Exception as exc:  # keep the protocol surface stable
        return _failed_task(task_id, text, f"{type(exc).__name__}: {exc}")


# ── Standalone app (optional) ────────────────────────────────────────────────────

def create_a2a_app() -> FastAPI:
    """Serve A2A on its own, e.g. `uvicorn protocols.a2a_server:create_a2a_app`."""
    app = FastAPI(
        title="Energy Intelligence Board - A2A",
        description="Agent-to-Agent interface to the crisis board.",
        version="0.1.0",
    )
    app.include_router(router)
    return app
