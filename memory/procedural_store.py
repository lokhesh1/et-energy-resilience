"""
ProceduralStore — the system's "cookbook" of reusable skill templates,
backed by Supabase (PostgreSQL).

Where EpisodicStore is a diary (append-only log of what happened), this is a
cookbook (one row per named recipe, refined over time). A skill is a reusable
how-to template plus efficacy counters, so agents can pick strategies that have
actually worked before.

Table: procedural_skills
    id            UUID   PK, auto
    skill_name    TEXT   UNIQUE — upsert key (one row per recipe)
    agent         TEXT   which agent owns/uses it
    template      JSONB  the recipe body (trigger, steps, notes, source, ...)
    use_count     INT    times applied
    success_count INT    times it worked
    created_at    TIMESTAMPTZ auto
    updated_at    TIMESTAMPTZ auto

Design: lazy client, fire-and-forget resilient — writes return status dicts,
reads return None/[] on error. Never crashes an agent.
"""
from __future__ import annotations

from typing import Any, Optional

from config import settings

_TABLE = "procedural_skills"


class ProceduralStore:
    def __init__(self) -> None:
        self._client = None  # lazy: connect on first use, not on import

    # ── Connection ─────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            return None
        try:
            from supabase import create_client
            self._client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        except Exception:
            self._client = None
        return self._client

    # ── Write ──────────────────────────────────────────────────────────────────

    def store_skill(self, skill_name: str, agent: str, template: dict[str, Any]) -> dict:
        """
        Upsert a skill by skill_name — updates the template if it exists, inserts
        otherwise. One row per recipe (never duplicates). Never raises.
        Returns {"status": "ok", "id": ...} or {"status": "error", "error": ...}.
        """
        client = self._get_client()
        if client is None:
            return {"status": "error", "error": "supabase unavailable"}

        row = {"skill_name": skill_name, "agent": agent, "template": template}
        try:
            resp = client.table(_TABLE).upsert(row, on_conflict="skill_name").execute()
            record = resp.data[0] if resp.data else {}
            return {"status": "ok", "id": record.get("id")}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def increment_use(self, skill_name: str, success: bool = False) -> dict:
        """
        Record that a skill was applied: bump use_count (+ success_count if it
        worked). Read-modify-write on the single named row. Never raises.
        """
        client = self._get_client()
        if client is None:
            return {"status": "error", "error": "supabase unavailable"}
        try:
            existing = (
                client.table(_TABLE)
                .select("use_count, success_count")
                .eq("skill_name", skill_name)
                .execute()
            )
            if not existing.data:
                return {"status": "error", "error": f"skill not found: {skill_name}"}

            row = existing.data[0]
            updates = {
                "use_count": (row.get("use_count") or 0) + 1,
                "success_count": (row.get("success_count") or 0) + (1 if success else 0),
            }
            client.table(_TABLE).update(updates).eq("skill_name", skill_name).execute()
            return {"status": "ok", **updates}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_skill(self, skill_name: str) -> Optional[dict]:
        """Return one recipe by name, or None if missing/error."""
        client = self._get_client()
        if client is None:
            return None
        try:
            resp = (
                client.table(_TABLE)
                .select("*")
                .eq("skill_name", skill_name)
                .limit(1)
                .execute()
            )
            return resp.data[0] if resp.data else None
        except Exception:
            return None

    def list_skills(self, agent: Optional[str] = None, limit: int = 50) -> list[dict]:
        """All recipes, most battle-tested first (use_count desc). [] on error."""
        client = self._get_client()
        if client is None:
            return []
        try:
            q = client.table(_TABLE).select("*")
            if agent is not None:
                q = q.eq("agent", agent)
            q = q.order("use_count", desc=True).limit(limit)
            resp = q.execute()
            return resp.data or []
        except Exception:
            return []
