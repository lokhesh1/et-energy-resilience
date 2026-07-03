"""
XMemory — the single facade the agents talk to.

Wires the 4 memory layers + decay + distillation behind one interface so agents
call xmemory.remember(...) / xmemory.recall_*(...) instead of juggling four stores:

    Working    — per-run scratchpad (in-process, token-budgeted)
    Episodic   — durable event log (Supabase)
    Semantic   — similarity search (Pinecone)
    Procedural — reusable skill cookbook (Supabase)

Writes are dual (episodic + semantic, shared id). Reads apply event-type decay so
slow-moving crises (wars, sanctions) stay relevant longer than fast ones (weather).
Fire-and-forget throughout — nothing here raises; the underlying stores already
return status dicts / [] on failure.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from memory.working_memory import WorkingMemory
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from memory.procedural_store import ProceduralStore
from memory.distillation import Distiller
from memory import decay as decay_mod


def _derive_text(payload: dict) -> str:
    """Best-effort human-readable text for semantic embedding."""
    if not isinstance(payload, dict):
        return str(payload)
    return payload.get("summary") or payload.get("text") or json.dumps(payload, default=str)


class XMemory:
    def __init__(self, working_budget: int = 4000) -> None:
        self.working    = WorkingMemory(budget=working_budget)
        self.episodic   = EpisodicStore()
        self.semantic   = SemanticStore()
        self.procedural = ProceduralStore()
        self.distiller  = Distiller()

    # ── Working (per-run scratchpad) ───────────────────────────────────────────

    def scratch_set(self, key: str, value: Any, token_cost: int) -> bool:
        return self.working.set(key, value, token_cost)

    def scratch_get(self, key: str) -> Any | None:
        return self.working.get(key)

    def working_snapshot(self) -> dict:
        return self.working.snapshot()

    def reset_working(self) -> None:
        """Start a fresh run — clears the scratchpad."""
        self.working.clear()

    # ── Long-term write (dual: episodic + semantic, shared id) ─────────────────

    def remember(
        self,
        event_type: str,
        agent: str,
        payload: dict[str, Any],
        outcome: Optional[str] = None,
        text: Optional[str] = None,
    ) -> dict:
        """
        Persist one event to episodic (log) and semantic (searchable), sharing an id.
        Returns {"episodic_id", "semantic_id", "status"}. Never raises.
        """
        result = {"episodic_id": None, "semantic_id": None, "status": "ok"}

        epi_res = self.episodic.store(event_type, agent, payload, outcome=outcome)
        if epi_res.get("status") == "ok":
            result["episodic_id"] = epi_res.get("id")
        else:
            result["status"] = "partial"

        meta = {"event_type": event_type, "agent": agent}
        if outcome is not None:
            meta["outcome"] = outcome
        sem_res = self.semantic.store(
            text or _derive_text(payload), metadata=meta, id=result["episodic_id"]
        )
        if sem_res.get("status") == "ok":
            result["semantic_id"] = sem_res.get("id")
        else:
            result["status"] = "partial"

        return result

    # ── Long-term read ─────────────────────────────────────────────────────────

    def recall_similar(self, text: str, top_k: int = 3, filter: Optional[dict] = None) -> list[dict]:
        """Semantic recall — past events closest in meaning. [] on error."""
        return self.semantic.query(text, top_k=top_k, filter=filter)

    def recall_events(
        self,
        agent: Optional[str] = None,
        event_type: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 20,
        decay: bool = True,
    ) -> list[dict]:
        """
        Episodic recall. When decay=True, each row gets a `decayed_relevance` score
        (intensity from payload['score'] or 1.0, aged by its event_type half-life)
        and rows are returned most-relevant-first — so a 60-day-old war still
        outranks yesterday's weather blip. decay=False keeps chronological order.
        """
        rows = self.episodic.query(
            agent=agent, event_type=event_type, outcome=outcome, limit=limit
        )
        if not decay or not rows:
            return rows

        scored = []
        for row in rows:
            payload = row.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {}
            intensity = float(payload.get("score", 1.0))
            # Decay half-life keys on the geopolitical classification (war_conflict,
            # sanctions, ...) which GRI writes into the payload — NOT the episodic
            # record type ("risk_assessment"). Fall back to the record type, then none.
            et = payload.get("event_type") or row.get("event_type") or "none"
            age = decay_mod.age_days(row.get("timestamp", ""))
            row["decayed_relevance"] = decay_mod.compute_decay(intensity, age, et)
            scored.append(row)

        scored.sort(key=lambda r: r["decayed_relevance"], reverse=True)
        return scored

    # ── Procedural (cookbook) ──────────────────────────────────────────────────

    def get_skill(self, name: str) -> Optional[dict]:
        return self.procedural.get_skill(name)

    def list_skills(self, agent: Optional[str] = None, limit: int = 50) -> list[dict]:
        return self.procedural.list_skills(agent=agent, limit=limit)

    def record_skill_use(self, name: str, success: bool = False) -> dict:
        return self.procedural.increment_use(name, success=success)

    # ── Learning loop ──────────────────────────────────────────────────────────

    def distill_run(self, trajectory: Optional[dict] = None) -> dict:
        """
        Close the learning loop: distill the run's trajectory into durable learnings
        and route them into episodic/semantic/procedural. If no trajectory is given,
        one is built from the working snapshot + recent episodic events.
        Returns the persist report. Never raises.
        """
        if trajectory is None:
            trajectory = {
                "working_memory": self.working_snapshot(),
                "episodic_events": self.episodic.recent(20),
            }
        distilled = self.distiller.distill(trajectory)
        return self.distiller.persist(
            distilled,
            episodic=self.episodic,
            semantic=self.semantic,
            procedural=self.procedural,
        )
