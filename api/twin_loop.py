"""
Continuous SCTD twin loop — keeps the digital twin LIVE, not query-triggered.

Today `twin_state` only exists for the duration of one graph run. This module makes
the twin always-current: a background task re-runs the GRI→DSM→SCTD segment
(`build_twin_graph`) on a cadence and stores the result in a durable in-process
snapshot the API serves instantly. Reads are cheap (snapshot lookup); the expensive
recompute happens on the twin's OWN clock, decoupled from any user query — the
concrete form of the "24/7 digital crisis team" framing.

Design:
  * `TwinSnapshot` — one thread-safe holder of the latest twin + refresh metadata.
    The API reads it; the loop (running the blocking graph in a worker thread) writes
    it. A single lock guards the swap.
  * `refresh_twin()` — one projection cycle. Best-effort: on any failure it records
    the error in the metadata and LEAVES the last good snapshot in place, so a
    transient news/LLM hiccup never blanks the twin (a stale-but-real twin beats an
    empty one). Never raises.
  * `twin_loop()` — the async cadence driver the FastAPI lifespan launches; refreshes
    once immediately (so /twin is warm at startup) then every interval until cancelled.

Note: each refresh is a live GRI news+LLM read, so it is gated by TWIN_LOOP_ENABLED
and paced by TWIN_REFRESH_INTERVAL (config/settings.py).
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone

from graph.workflow import build_twin_graph, initial_state

# The generic monitoring prompt the twin runs under — it is NOT answering a user, it
# is reading the world (live corridor news) and reprojecting the physical twin.
_TWIN_QUERY = "continuous background twin refresh — current corridor monitoring"


class TwinSnapshot:
    """Thread-safe holder of the latest twin projection + refresh bookkeeping."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._twin: dict = {}
        self._meta: dict = {
            "status":            "cold",   # cold → ok / stale (last refresh failed)
            "refresh_count":     0,
            "last_refreshed_at": None,
            "last_error":        None,
        }

    def read(self) -> dict:
        """A snapshot copy safe to serialise while the loop keeps writing."""
        with self._lock:
            return {"twin_state": dict(self._twin), **self._meta}

    def update_ok(self, twin_state: dict) -> None:
        with self._lock:
            self._twin = twin_state or {}
            self._meta["status"] = "ok"
            self._meta["refresh_count"] += 1
            self._meta["last_refreshed_at"] = datetime.now(timezone.utc).isoformat()
            self._meta["last_error"] = None

    def update_error(self, error: str) -> None:
        """Record a failed refresh WITHOUT discarding the last good twin — a
        stale-but-real snapshot is safer than a blank one."""
        with self._lock:
            # only downgrade to "stale" if we ever had a good one; else stay "cold"
            self._meta["status"] = "stale" if self._meta["refresh_count"] else "cold"
            self._meta["last_error"] = error
            self._meta["last_refreshed_at"] = datetime.now(timezone.utc).isoformat()


# Module singleton — the one live twin the whole API shares.
snapshot = TwinSnapshot()

# Compiled twin graph, built once and reused across refreshes.
_twin_graph = None


def _get_twin_graph():
    global _twin_graph
    if _twin_graph is None:
        _twin_graph = build_twin_graph()
    return _twin_graph


def refresh_twin() -> dict:
    """Run ONE twin projection and store it. Never raises — errors are captured in
    the snapshot metadata and the last good twin is preserved. Returns the snapshot."""
    try:
        graph = _get_twin_graph()
        final = graph.invoke(
            initial_state(_TWIN_QUERY),
            config={"configurable": {"thread_id": "twin-loop"}},
        )
        snapshot.update_ok(final.get("twin_state", {}) or {})
    except Exception as exc:  # pragma: no cover — defensive; nodes are best-effort
        snapshot.update_error(f"{type(exc).__name__}: {exc}")
    return snapshot.read()


async def twin_loop(interval_seconds: int) -> None:
    """Refresh the twin on a cadence until the task is cancelled. Runs the blocking
    graph in a worker thread so the event loop stays free for API requests. Refreshes
    once immediately so /twin is warm at startup, then every `interval_seconds`."""
    try:
        while True:
            await asyncio.to_thread(refresh_twin)
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:  # graceful shutdown
        raise
