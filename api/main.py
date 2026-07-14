"""
FastAPI backend — the Energy Intelligence Board's front door.

The board is a compiled graph; this exposes it as an HTTP service a UI / another
agent / curl can call. Two distinct surfaces:

  * ON-DEMAND  — POST /query and POST /scenario run the FULL board (GRI→…→coordinator)
    for a specific question and return the response plan. Learning fires in the
    background (run_board_with_learning), so the answer is never delayed by memory
    writes.
  * ALWAYS-ON  — GET /twin serves the LATEST twin snapshot maintained by the
    continuous twin loop (api/twin_loop.py). This read is cheap and instant: SCTD
    recomputes on its own clock in the background, the query just reads the freshest
    projection. This is the decoupling the "live twin" TODO calls for — reads don't
    trigger recompute.

The twin loop is launched/stopped by the lifespan handler (gated by
TWIN_LOOP_ENABLED). Board endpoints are sync `def` so FastAPI runs each blocking
graph invocation in its threadpool without blocking the event loop / the twin loop.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import settings
from graph.workflow import run_board_with_learning
from tools.corridor_status import get_corridor_status
from protocols.a2a_server import router as a2a_router
from eib_guardrails.principal_hierarchy import check_permission
from eib_guardrails.audit_logger import verify_chain
from api import twin_loop as tl
from api.summary import summarize_final


# ── Request models ───────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="The crisis/question to run the board on.")
    learn: bool = Field(True, description="Fire the distiller pod after the run.")
    consolidate: bool = Field(True, description="Also run consolidation in the pod.")


class ScenarioRequest(BaseModel):
    query: str = Field("", description="Optional framing text for the what-if run.")
    scenario_params: dict = Field(default_factory=dict,
                                  description="Scenario overrides fed to the board.")
    learn: bool = Field(False, description="What-ifs default to NOT polluting memory.")


# ── The board agents, for the capability endpoint ────────────────────────────────

_AGENTS = [
    {"name": "crisis_coordinator",
     "role": "Orchestrates the board; assembles the response plan + final recommendation."},
    {"name": "gri", "role": "Geopolitical Risk Intelligence — scores corridor risk from news."},
    {"name": "dsm", "role": "Disruption Scenario Modeller — volume/duration/India exposure."},
    {"name": "sctd", "role": "Supply Chain Digital Twin — projects disruption onto refineries."},
    {"name": "procurement",
     "role": "West Africa / Americas / Spot bidders + Bid Evaluator — sources the shortfall."},
    {"name": "distiller",
     "role": "Experience Distiller + Consolidation — the learning loop (runs off the response path)."},
]


_summarize = summarize_final


# ── Lifespan: run the continuous twin loop for the life of the app ───────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    if settings.TWIN_LOOP_ENABLED:
        task = asyncio.create_task(tl.twin_loop(settings.TWIN_REFRESH_INTERVAL))
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title="Energy Intelligence Board",
    description="Multi-agent energy supply-chain resilience board.",
    version="0.1.0",
    lifespan=lifespan,
)

# A2A front door: discovery (/.well-known/agent.json, /a2a/card, /a2a/agents) +
# invocation (/a2a/tasks/send). Lets an external agent discover and call the board.
app.include_router(a2a_router)


# ── Endpoints ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "twin": tl.snapshot.read()["status"]}


@app.get("/agents")
def agents() -> dict:
    return {"agents": _AGENTS}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    """Run the full board on a query; learning fires in the background."""
    final = run_board_with_learning(
        req.query, learn=req.learn, consolidate=req.consolidate,
    )
    return _summarize(final)


@app.post("/scenario")
def scenario(req: ScenarioRequest) -> dict:
    """Run a what-if with explicit scenario_params (memory write off by default)."""
    final = run_board_with_learning(
        req.query, scenario_params=req.scenario_params, learn=req.learn,
    )
    return _summarize(final)


@app.get("/corridor-status")
def corridor_status() -> dict:
    """Live status of the 8 shipping corridors (baselines + active incident
    overrides) — the same feed GRI/SCTD consume, exposed directly. Cheap read: it
    reads the corridor tool, it does NOT run the board."""
    return get_corridor_status()


@app.get("/twin")
def twin() -> dict:
    """The latest twin snapshot — served instantly from the continuous loop, not
    recomputed on the request."""
    return tl.snapshot.read()


@app.post("/twin/refresh")
def twin_refresh() -> dict:
    """Force an immediate twin refresh (useful for demos / a cold start).

    Principal-hierarchy gate: forcing the compute clock is an operator capability
    (the REST surface has no auth yet, so REST = operator by construction; the
    check makes the boundary explicit and testable)."""
    perm = check_permission("operator", "refresh_twin")
    if not perm["allowed"]:
        raise HTTPException(status_code=403, detail=perm["reason"])
    return tl.refresh_twin()


@app.get("/audit/verify")
def audit_verify() -> dict:
    """Verify the hash-chained audit log end-to-end.

    Principal-hierarchy gate: reading the audit trail is an operator capability
    (same pattern as /twin/refresh — the REST surface = operator by construction)."""
    perm = check_permission("operator", "read_audit")
    if not perm["allowed"]:
        raise HTTPException(status_code=403, detail=perm["reason"])
    return verify_chain()
