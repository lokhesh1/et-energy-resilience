"""
A2A capability cards — the board's "menu", in Agent-to-Agent (A2A) shape.

A2A is the interop layer: how an OUTSIDE agent (another org, a client orchestrator)
discovers what this system can do and how to call it. Internally our 6 agents are
LangGraph nodes that coordinate through shared state + stigmergy — they do NOT need
A2A. So the star of this file is the ONE board-level card an external caller
consumes; the per-agent cards are a declarative registry of what each node does
(a structured version of GET /agents), useful for docs/observability.

An A2A AgentCard is just JSON. We follow the spec's core shape:
  name, description, url, version, capabilities, defaultInput/OutputModes,
  skills[] (each: id, name, description, tags, examples).

Nothing here runs the board — these are static descriptors. `a2a_server.py` serves
them and maps an inbound task onto the real board runner.
"""
from __future__ import annotations

# Base URL the board is reachable at. Overridable by the server at serve-time so the
# card advertises the address it is actually mounted on.
DEFAULT_BASE_URL = "http://localhost:8000"

A2A_VERSION = "0.1.0"


def _skill(id: str, name: str, description: str, tags: list[str],
           examples: list[str]) -> dict:
    return {
        "id": id,
        "name": name,
        "description": description,
        "tags": tags,
        "examples": examples,
    }


# ── The board-level card: what an EXTERNAL agent calls ───────────────────────────
# One coherent skill — "run the crisis board" — plus the sub-capabilities it wraps,
# advertised so a caller knows what a run delivers.

BOARD_SKILLS = [
    _skill(
        "run_crisis_board",
        "Run the energy crisis board",
        "Run the full Energy Intelligence Board on a crisis/question: geopolitical "
        "risk -> disruption scenarios -> digital-twin refinery impact -> procurement "
        "sourcing -> coordinated response plan.",
        ["energy", "supply-chain", "crisis", "orchestration"],
        [
            "Iran closes the Strait of Hormuz - assess impact on Indian refineries "
            "and recommend a response.",
            "A drone strike shuts Ras Tanura for 3 weeks. What is the shortfall and "
            "how do we cover it?",
        ],
    ),
    _skill(
        "assess_corridor_risk",
        "Assess shipping-corridor risk",
        "Score geopolitical disruption risk across the 8 key global oil corridors "
        "from live news signals (GRI).",
        ["geopolitics", "risk", "corridors"],
        ["What is the current risk on the Strait of Hormuz and Bab-el-Mandeb?"],
    ),
    _skill(
        "model_disruption",
        "Model a disruption scenario",
        "Model volume-at-risk, duration and India import exposure for a corridor "
        "disruption (DSM).",
        ["scenario", "modelling", "disruption"],
        ["If Suez closes, how much India-bound crude is at risk and for how long?"],
    ),
    _skill(
        "source_shortfall",
        "Source a crude shortfall",
        "Evaluate procurement alternatives (West Africa / Americas / Spot) against "
        "sanctions + crude-grade compatibility and recommend a covering mix.",
        ["procurement", "sourcing", "sanctions", "crude-grades"],
        ["Find sanctions-clean cargoes to cover a 1.0 mbd medium-sour shortfall."],
    ),
]


def board_card(base_url: str = DEFAULT_BASE_URL) -> dict:
    """The single card an external A2A client discovers to use the whole board."""
    return {
        "name": "Energy Intelligence Board",
        "description": "Multi-agent energy supply-chain resilience board - assesses "
                       "corridor risk, models disruptions, projects refinery impact, "
                       "and sources procurement to coordinate a crisis response.",
        "url": f"{base_url.rstrip('/')}/a2a",
        "version": A2A_VERSION,
        "provider": {"organization": "et-energy-resilience"},
        "capabilities": {
            "streaming": False,          # sync tasks/send only, for now
            "pushNotifications": False,
        },
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text", "data"],
        "skills": BOARD_SKILLS,
    }


# ── Per-agent cards: an internal registry of what each node does ─────────────────
# NOT a network boundary (the agents talk via shared state internally) - a declarative
# capability catalogue. Kept honest: these describe nodes, not A2A endpoints.

_AGENT_DEFS = [
    ("crisis_coordinator",
     "Orchestrates the board; assembles the response plan + final recommendation "
     "and owns this A2A capability card.",
     ["orchestration", "synthesis"]),
    ("gri",
     "Geopolitical Risk Intelligence - scores corridor risk from live news signals.",
     ["geopolitics", "risk"]),
    ("dsm",
     "Disruption Scenario Modeller - volume-at-risk, duration and India exposure.",
     ["scenario", "modelling"]),
    ("sctd",
     "Supply Chain Digital Twin - projects a disruption onto Indian refineries.",
     ["digital-twin", "refineries"]),
    ("procurement",
     "West Africa / Americas / Spot bidders + Bid Evaluator - sources the shortfall "
     "against sanctions + grade compatibility.",
     ["procurement", "sourcing"]),
    ("distiller",
     "Experience Distiller + Consolidation - the learning loop (runs off the "
     "response path).",
     ["memory", "learning"]),
]


def agent_card(name: str, description: str, tags: list[str],
               base_url: str = DEFAULT_BASE_URL) -> dict:
    return {
        "name": name,
        "description": description,
        "url": f"{base_url.rstrip('/')}/a2a",   # all routed through the board front door
        "version": A2A_VERSION,
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text", "data"],
        "internal": True,   # a node, not an external A2A peer
        "skills": [_skill(name, name, description, tags, [])],
    }


def agent_cards(base_url: str = DEFAULT_BASE_URL) -> list[dict]:
    """The per-node capability registry."""
    return [agent_card(n, d, t, base_url) for (n, d, t) in _AGENT_DEFS]
