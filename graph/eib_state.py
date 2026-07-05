import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class StigmergyMarker(TypedDict):
    type: str          # "risk" | "bottleneck" | "demand" | "bid" | "alert"
    target: str        # corridor_id / route_id / refinery_id
    intensity: float   # pheromone strength [0.0, 1.0]
    deposited_by: str  # agent name
    timestamp: str     # ISO-8601
    decay_rate: float  # per-step evaporation fraction


class EnergyIntelligenceBoard(TypedDict):
    # ── Input / trigger ──
    query: str
    scenario_params: dict
    messages: Annotated[list, add_messages]

    # ── GRI ──
    risk_signals: list[dict]
    corridor_risk: dict[str, float]   # corridor_id → risk score
    corridor_events: dict[str, str]   # corridor_id → event_type (drives DSM/decay)

    # ── DSM ──
    scenarios: list[dict]

    # ── SCTD ──
    affected_refineries: list[str]
    affected_routes: list[dict]
    twin_state: dict

    # ── Procurement pod (parallel agents append concurrently) ──
    bids: Annotated[list[dict], operator.add]
    evaluated_bids: list[dict]

    # ── Coordinator ──
    response_plan: dict
    final_recommendation: str

    # ── Memory ──
    retrieved_memories: list[dict]
    working_context: dict

    # ── Guardrails ──
    audit_trail: Annotated[list[dict], operator.add]
    constitution_flags: list[dict]
    current_agent: str

    # ── Stigmergy: indirect coordination substrate ──
    stigmergy_markers: Annotated[list[StigmergyMarker], operator.add]
    pheromone_field: dict[str, float]  # target → decayed intensity; rebuilt each step
