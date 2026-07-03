"""
EpisodicStore — durable, cross-run event ledger backed by Supabase (PostgreSQL).

Every meaningful agent action is written here as one row and kept permanently,
so future runs can query what has happened before (including past failures).

Table: episodic_events
    id          UUID   PK, auto
    event_type  TEXT   e.g. "risk_assessment", "procurement_bid"
    agent       TEXT   e.g. "gri_agent"
    payload     JSONB  free-form event body (put `reason` here)
    outcome     TEXT   "success" | "failure" | NULL (non-attempt events)
    timestamp   TIMESTAMPTZ auto

Design: writes are fire-and-forget resilient — a memory failure returns an error
dict instead of raising, so it can never crash an agent mid-run. Reads return []
on error.
"""
from __future__ import annotations

from typing import Any, Optional

from config import settings

_TABLE = "episodic_events"


class EpisodicStore:
    def __init__(self) -> None:
        self._client = None  # lazy: connect on first use, not on import

    # ── Connection ─────────────────────────────────────────────────────────────

    def _get_client(self):
        """Lazily create the Supabase client. Returns None if unconfigured/unavailable."""
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

    def store(
        self,
        event_type: str,
        agent: str,
        payload: dict[str, Any],
        outcome: Optional[str] = None,
    ) -> dict:
        """
        Insert one event. Never raises.
        Returns {"status": "ok", "id": ...} or {"status": "error", "error": ...}.
        """
        client = self._get_client()
        if client is None:
            return {"status": "error", "error": "supabase unavailable"}

        row = {
            "event_type": event_type,
            "agent": agent,
            "payload": payload,
            "outcome": outcome,
        }
        try:
            resp = client.table(_TABLE).insert(row).execute()
            record = resp.data[0] if resp.data else {}
            return {"status": "ok", "id": record.get("id")}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ── Read ───────────────────────────────────────────────────────────────────

    def query(
        self,
        agent: Optional[str] = None,
        event_type: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Filtered lookup, newest first. Returns [] on error."""
        client = self._get_client()
        if client is None:
            return []
        try:
            q = client.table(_TABLE).select("*")
            if agent is not None:
                q = q.eq("agent", agent)
            if event_type is not None:
                q = q.eq("event_type", event_type)
            if outcome is not None:
                q = q.eq("outcome", outcome)
            q = q.order("timestamp", desc=True).limit(limit)
            resp = q.execute()
            return resp.data or []
        except Exception:
            return []

    def recent(self, n: int = 10) -> list[dict]:
        """Latest n events across all agents. Returns [] on error."""
        client = self._get_client()
        if client is None:
            return []
        try:
            resp = (
                client.table(_TABLE)
                .select("*")
                .order("timestamp", desc=True)
                .limit(n)
                .execute()
            )
            return resp.data or []
        except Exception:
            return []
