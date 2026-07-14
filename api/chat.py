"""Multi-turn conversation layer for the Energy Intelligence Board.

Adds session-aware chat over the existing board runner.  Design choices:

  * Conversational query rewriting into standalone form before board execution
    (conversational-search line, EMNLP 2025).
  * Adaptive-RAG intent gating — only run the expensive board when the turn
    needs new computation; answer-from-context otherwise (LLM-Based Dialogue
    Labeling for Multiturn Adaptive RAG, EMNLP 2025 industry).
  * Context compression — ground follow-up answers on a compact digest, not
    the raw ~20-key board state (A Survey of Context Engineering for LLMs,
    arXiv 2507.13334); cap history at CHAT_HISTORY_TURNS (working-memory
    budget).

Numbers are always deterministic (from the board / digest); the LLM only
classifies intent, rewrites queries, and phrases grounded answers — with a
template fallback if it's unavailable.
"""
from __future__ import annotations

import json
import threading
import uuid

from fastapi import APIRouter
from openai import OpenAI
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver

from config.settings import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    CHAT_MODEL, CHAT_HISTORY_TURNS,
)
from graph.workflow import run_board_with_learning
from agents.distiller.experience_distiller import build_trajectory
from api.summary import summarize_final, build_components, suggest_follow_ups

router = APIRouter(tags=["chat"])

_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# ONE shared checkpointer across all chat sessions.  Each board run uses a
# UNIQUE thread_id (chat-{sid}-t{n}) so LangGraph's reducer channels (bids,
# audit_trail, stigmergy_markers — all operator.add) never accumulate stale
# data across turns.  Session continuity lives in ChatStore (digest + history +
# query rewriting), not in graph-state replay.
_CHECKPOINTER = MemorySaver()

_MAX_STORED_TURNS = 24


# ── Request model ───────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str | None = Field(None, description="Omit to start a new session.")
    message: str = Field(..., description="The user turn.")
    learn: bool = Field(True, description="Fire the distiller pod after a board run.")


# ── Session store ───────────────────────────────────────────────────────────────

class ChatStore:
    """Thread-safe in-process session store (mirrors the TwinSnapshot lock
    pattern from api/twin_loop.py).  Sessions live as long as the server
    process — restart loses them.  Same status as the twin snapshot; swap for
    Redis/SQLite later if persistence is needed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}

    def _blank(self) -> dict:
        return {
            "turns": [],
            "summary": None,
            "digest": None,
            "components": [],
            "follow_ups": [],
            "run_count": 0,
        }

    def ensure(self, session_id: str | None) -> str:
        with self._lock:
            if session_id and session_id in self._sessions:
                return session_id
            sid = session_id or uuid.uuid4().hex
            if sid not in self._sessions:
                self._sessions[sid] = self._blank()
            return sid

    def append_turn(self, sid: str, role: str, content: str) -> None:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return
            sess["turns"].append({"role": role, "content": content})
            if len(sess["turns"]) > _MAX_STORED_TURNS:
                sess["turns"] = sess["turns"][-_MAX_STORED_TURNS:]

    def record_run(self, sid: str, summary: dict, digest: dict,
                   components: list, follow_ups: list) -> None:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return
            sess["summary"] = summary
            sess["digest"] = digest
            sess["components"] = components
            sess["follow_ups"] = follow_ups
            sess["run_count"] += 1

    def context(self, sid: str) -> dict:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return {**self._blank(), "turns": []}
            return {
                "turns": list(sess["turns"][-CHAT_HISTORY_TURNS:]),
                "summary": sess["summary"],
                "digest": sess["digest"],
                "components": list(sess["components"]),
                "follow_ups": list(sess["follow_ups"]),
                "run_count": sess["run_count"],
            }


store = ChatStore()


# ── LLM helpers (gri_agent.py pattern: response_format json, broad except) ──

_ROUTER_SYSTEM = """\
You are the intent router for an energy-crisis AI board.
Given the user's latest message and recent conversation history, return JSON:
{"intent": "<run_board or answer_from_last_run>", "standalone_query": "<...>"}

Rules:
- "run_board": a NEW crisis scenario, what-if, or request that needs fresh
  computation.  Rewrite the message into a STANDALONE query that folds in any
  relevant context from the conversation (e.g. "what about Americas only?" after
  a Hormuz discussion → "Strait of Hormuz closed; source the shortfall only
  from Americas suppliers").
- "answer_from_last_run": a clarification or question about the LAST result
  ("why Bonny Light?", "explain the SPR bridge", "show me the flagged
  suppliers").  Set standalone_query to the user's message as-is.
- When in doubt, choose "run_board" — it's the safe default.
"""

_ANSWER_SYSTEM = """\
You are a crisis-briefing assistant grounded on the run digest below.
Answer the user's question using ONLY the data in the digest — cite specific
numbers.  Do NOT invent data not present in the digest.  Be concise (2-4
sentences).

Digest:
{digest_json}
"""


def _route(message: str, turns: list[dict]) -> dict:
    """Classify intent and rewrite the query into standalone form."""
    history = [{"role": t["role"], "content": t["content"]}
               for t in turns[-CHAT_HISTORY_TURNS:]]
    try:
        resp = _client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                *history,
                {"role": "user", "content": message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        parsed = json.loads(resp.choices[0].message.content)
        intent = parsed.get("intent", "run_board")
        if intent not in ("run_board", "answer_from_last_run"):
            intent = "run_board"
        query = parsed.get("standalone_query") or message
        return {"intent": intent, "standalone_query": query}
    except Exception:
        return {"intent": "run_board", "standalone_query": message}


def _answer_from_digest(message: str, digest: dict,
                        turns: list[dict]) -> str | None:
    """Answer a follow-up grounded ONLY on the stored run digest."""
    try:
        digest_json = json.dumps(digest, default=str)
        history = [{"role": t["role"], "content": t["content"]}
                   for t in turns[-CHAT_HISTORY_TURNS:]]
        resp = _client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system",
                 "content": _ANSWER_SYSTEM.format(digest_json=digest_json)},
                *history,
                {"role": "user", "content": message},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def _template_answer(summary: dict) -> str:
    """Deterministic fallback when the LLM is unavailable."""
    plan = summary.get("response_plan", {}) or {}
    proc = plan.get("procurement", {}) or {}
    sit = plan.get("situation", {}) or {}
    esc = (summary.get("escalation_level") or "routine").upper()
    gap = float((summary.get("twin_summary") or {}).get(
        "total_india_shortfall_mbd") or 0)
    covered = float(proc.get("covered_mbd") or 0)
    residual = float(proc.get("residual_gap_mbd") or 0)

    top = (sit.get("top_corridor_risks") or [{}])[0] if sit.get(
        "top_corridor_risks") else {}
    corridor = top.get("corridor", "unknown corridor")

    if gap <= 0:
        return (f"{esc}: No India-bound crude shortfall projected. "
                f"Corridors nominal; no procurement action required.")

    parts = [
        f"{esc}: {corridor} disruption projects a {gap} mbd India shortfall.",
        f"Procurement covers {covered} mbd.",
    ]
    if residual > 0:
        parts.append(f"{residual} mbd remains uncovered — SPR / demand-side "
                     f"measures recommended.")
    else:
        parts.append("Gap fully covered by market bids.")
    return " ".join(parts)


# ── Endpoint ────────────────────────────────────────────────────────────────────

@router.post("/chat")
def chat(req: ChatRequest) -> dict:
    """Multi-turn conversation with the board.

    First turn always runs the board.  Subsequent turns are routed: a new
    crisis/what-if re-runs the board with a rewritten standalone query; a
    question about the last result is answered from the stored digest (no
    board re-run — instant and free).  Sync def so FastAPI uses its threadpool
    (same as /query).
    """
    sid = store.ensure(req.session_id)
    ctx = store.context(sid)
    store.append_turn(sid, "user", req.message)

    # First turn: skip the router — no prior run to answer from.
    if ctx["run_count"] == 0:
        intent, query = "run_board", req.message
    else:
        route = _route(req.message, ctx["turns"])
        intent, query = route["intent"], route["standalone_query"]

    if intent == "run_board":
        run_n = ctx["run_count"] + 1
        final = run_board_with_learning(
            query,
            thread_id=f"chat-{sid}-t{run_n}",
            checkpointer=_CHECKPOINTER,
            learn=req.learn,
        )
        summary = summarize_final(final)
        digest = build_trajectory(final)
        twin_state = final.get("twin_state", {}) or {}
        components = build_components(summary, twin_state)
        follow_ups = suggest_follow_ups(summary)
        store.record_run(sid, summary, digest, components, follow_ups)
        reply = final.get("final_recommendation") or _template_answer(summary)
        run_summary = summary
    else:
        reply = (_answer_from_digest(req.message, ctx["digest"], ctx["turns"])
                 or _template_answer(ctx["summary"] or {}))
        components = ctx["components"]
        follow_ups = ctx["follow_ups"]
        run_summary = None

    store.append_turn(sid, "assistant", reply)
    return {
        "session_id": sid,
        "mode": intent,
        "reply": reply,
        "run_summary": run_summary,
        "components": components,
        "follow_ups": follow_ups,
    }
