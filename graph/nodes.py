"""Graph-level node adapters + the stigmergy field refresh.

The agent functions (`gri_node`, `dsm_node`, ...) are written to READ
`state["pheromone_field"]` but nothing in the codebase ever DERIVES that field
from the raw `stigmergy_markers` pile — tests inject it by hand. Without the
refresh below, an end-to-end run leaves the field empty and the whole
ant-trail coordination channel is silently dead.

This module closes that loop. Two pieces:

  1. `rebuild_pheromone_field(markers, now)` — pure: fade every marker by its
     age, keep the strongest per target. This is "the field agents sniff".
  2. `wrap` / `wrap_bidder` — thin adapters that recompute the field at node
     entry and inject it into the state the agent sees, so every agent reads a
     CURRENT field instead of an empty dict. Agent code is untouched.

Parallel-write discipline (LangGraph): `pheromone_field` and `current_agent`
are plain keys (no reducer). Two nodes writing a plain key in the SAME
superstep raises `InvalidUpdateError`. The three bidders fan out in one
superstep, so `wrap_bidder` returns ONLY the reducer keys the bidder itself
returns (`bids`, `audit_trail`) — it never writes the field back. The
sequential nodes are each their own superstep, so `wrap` may write it back
(useful as the latest snapshot for the API/UI).
"""

import math
from datetime import datetime, timezone

from graph.eib_state import EnergyIntelligenceBoard

# Effective intensity below this is treated as fully evaporated and dropped from
# the field. In a single query run every marker is seconds old (age ~ 0), so the
# field ~ raw max intensity and nothing is dropped — matching what the agents
# expect today. The floor only bites across a long-running twin loop, where a
# marker deposited hours ago should stop influencing fresh decisions.
_EVAPORATION_FLOOR = 0.05


def _age_hours(timestamp: str, now: datetime) -> float:
    """Hours between a marker's deposit time and `now`. Fails open to 0.0 (raw
    intensity) on a missing/malformed timestamp — a parse error must never blank
    out a real signal."""
    if not timestamp:
        return 0.0
    try:
        ts = datetime.fromisoformat(timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = (now - ts).total_seconds() / 3600.0
        return max(0.0, delta)
    except (ValueError, TypeError):
        return 0.0


def rebuild_pheromone_field(
    markers: list[dict], now: datetime | None = None
) -> dict[str, float]:
    """Collapse the append-only marker pile into the current field.

    For each marker: strength = intensity * exp(-decay_rate * age_hours).
    Keep the MAX strength per target (strongest live signal wins), drop anything
    that has evaporated below the floor. Pure function of its inputs.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    field: dict[str, float] = {}
    for m in markers or []:
        target = m.get("target")
        if not target:
            continue
        intensity = float(m.get("intensity", 0.0) or 0.0)
        decay_rate = float(m.get("decay_rate", 0.0) or 0.0)
        age = _age_hours(m.get("timestamp", ""), now)
        strength = intensity * math.exp(-decay_rate * age)
        if strength < _EVAPORATION_FLOOR:
            continue
        if strength > field.get(target, 0.0):
            field[target] = round(strength, 4)
    return field


def wrap(agent_fn, name: str):
    """Adapter for a SEQUENTIAL node (single writer in its own superstep).

    Recomputes the field from the current marker pile, injects it into the state
    the agent sees, runs the agent, then writes back a field refreshed to
    include the agent's own freshly-deposited markers (so the persisted
    `pheromone_field` is the latest snapshot for downstream nodes and the API).
    """
    def node(state: EnergyIntelligenceBoard) -> dict:
        markers = state.get("stigmergy_markers", []) or []
        field_in = rebuild_pheromone_field(markers)

        out = agent_fn({**state, "pheromone_field": field_in})

        # Refresh the field to include markers this node just deposited, so the
        # next sequential node (and any observer) sees an up-to-date field.
        new_markers = out.get("stigmergy_markers", []) or []
        field_out = rebuild_pheromone_field(markers + new_markers)

        merged = dict(out)
        merged["pheromone_field"] = field_out
        merged.setdefault("current_agent", name)
        return merged

    return node


def wrap_bidder(agent_fn, name: str):
    """Adapter for a PARALLEL bidder node (one of several in a single superstep).

    Same field injection so the bidder reads a current field (spot scales its
    premium off it), but returns ONLY what the bidder returns — reducer keys
    (`bids`, `audit_trail`). It must NOT write `pheromone_field` or
    `current_agent`: those are plain keys and concurrent writes from the three
    bidders would raise LangGraph's InvalidUpdateError.
    """
    def node(state: EnergyIntelligenceBoard) -> dict:
        field_in = rebuild_pheromone_field(state.get("stigmergy_markers", []) or [])
        return agent_fn({**state, "pheromone_field": field_in})

    return node
