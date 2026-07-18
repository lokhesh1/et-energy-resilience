"""The compiled EIB graph — the orchestrator's skeleton.

The topology IS the routing. No LLM ever decides "which agent next"; the data
dependencies fix the order:

    GRI ─► DSM ─► SCTD ─► [ west_africa ∥ americas ∥ spot ] ─► bid_evaluator ─► coordinator

The only parallel step is the three regional bidders (fan-out from SCTD,
fan-in to the evaluator) — they have no dependency on each other and append to
the `bids` reducer, so they run concurrently and safely. Everything else is
sequential because each stage consumes the previous stage's output.

Two builders over the same node functions:
  * `build_graph()`      — the full board, for answering a query.
  * `build_twin_graph()` — just GRI→DSM→SCTD, the "keep the twin fresh" segment
    a background scheduler will re-run on a cadence (the continuous-twin TODO).

Coordinator note: `agents/crisis_coordinator.py` is the next build step. Until
it exists, a built-in passthrough placeholder keeps the full graph runnable
end-to-end today; the real node is picked up automatically once importable.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.eib_state import EnergyIntelligenceBoard
from graph.nodes import wrap, wrap_bidder

from agents.gri_agent import gri_node
from agents.dsm_agent import dsm_node
from agents.sctd_agent import sctd_node
from agents.procurement.west_africa_agent import west_africa_node
from agents.procurement.americas_agent import americas_node
from agents.procurement.spot_market_agent import spot_market_node
from agents.procurement.bid_evaluator import bid_evaluator_node
from agents.distiller.pod import learn_async
from eib_guardrails.audit_logger import log_run

try:  # real coordinator once it lands; placeholder keeps the graph runnable now
    from agents.crisis_coordinator import coordinator_node
    _COORDINATOR_IS_STUB = False
except Exception:  # ImportError today, or any import-time failure in the new file
    def coordinator_node(state: EnergyIntelligenceBoard) -> dict:
        """Placeholder: emit a minimal, honest response plan from existing state
        so the pipeline produces an end-to-end result before the real
        coordinator is built. Deliberately does no reasoning."""
        twin = state.get("twin_state", {}) or {}
        mix = state.get("recommended_mix", {}) or {}
        return {
            "current_agent": "coordinator",
            "response_plan": {
                "placeholder": True,
                "gap_mbd": twin.get("total_india_shortfall_mbd", 0.0),
                "critical_count": twin.get("critical_count", 0),
                "covers_gap": mix.get("covers_gap"),
            },
            "final_recommendation": "(placeholder coordinator — real node pending)",
        }
    _COORDINATOR_IS_STUB = True


# Node names, kept in one place so both builders and any observer agree.
GRI = "gri"
DSM = "dsm"
SCTD = "sctd"
WEST_AFRICA = "west_africa"
AMERICAS = "americas"
SPOT = "spot"
BID_EVALUATOR = "bid_evaluator"
COORDINATOR = "coordinator"

_BIDDERS = [WEST_AFRICA, AMERICAS, SPOT]


def _add_core_nodes(g: StateGraph) -> None:
    """GRI → DSM → SCTD, shared by both builders."""
    g.add_node(GRI, wrap(gri_node, GRI))
    g.add_node(DSM, wrap(dsm_node, DSM))
    g.add_node(SCTD, wrap(sctd_node, SCTD))
    g.add_edge(START, GRI)
    g.add_edge(GRI, DSM)
    g.add_edge(DSM, SCTD)


def build_twin_graph(checkpointer=None):
    """The always-on twin segment: GRI → DSM → SCTD → END.

    This is what a background scheduler re-runs on its own clock to keep the
    digital twin current, independent of any user query. Same node functions as
    the full graph — no duplicated logic.
    """
    g = StateGraph(EnergyIntelligenceBoard)
    _add_core_nodes(g)
    g.add_edge(SCTD, END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def build_graph(checkpointer=None):
    """The full Energy Intelligence Board, for answering a query end-to-end."""
    g = StateGraph(EnergyIntelligenceBoard)
    _add_core_nodes(g)

    # Procurement pod: fan-out (three edges from SCTD → one superstep, parallel)
    g.add_node(WEST_AFRICA, wrap_bidder(west_africa_node, WEST_AFRICA))
    g.add_node(AMERICAS, wrap_bidder(americas_node, AMERICAS))
    g.add_node(SPOT, wrap_bidder(spot_market_node, SPOT))
    for bidder in _BIDDERS:
        g.add_edge(SCTD, bidder)

    # Fan-in: the evaluator waits for ALL three bidders (list-of-sources = join).
    g.add_node(BID_EVALUATOR, wrap(bid_evaluator_node, BID_EVALUATOR))
    g.add_edge(_BIDDERS, BID_EVALUATOR)

    g.add_node(COORDINATOR, wrap(coordinator_node, COORDINATOR))
    g.add_edge(BID_EVALUATOR, COORDINATOR)
    g.add_edge(COORDINATOR, END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


def run_board_with_learning(
    query: str = "",
    scenario_params: dict | None = None,
    *,
    thread_id: str = "default",
    checkpointer=None,
    learn: bool = True,
    consolidate: bool = True,
) -> EnergyIntelligenceBoard:
    """Answer one query end-to-end, then learn from the run in the BACKGROUND.

    The full board runs synchronously and its final state is returned immediately;
    if `learn` is set, the distiller pod is fired on a daemon thread (`learn_async`)
    so distillation/consolidation never delay the answer. This is the seam
    `api/main.py` calls — reads are cheap, learning happens on its own clock.

    The run's audit_trail is also flushed to the durable hash-chained audit log
    here (best-effort, returns a status — a broken audit DB never blocks the
    answer). This seam sees every board run: /query, /scenario, and A2A.
    """
    graph = build_graph(checkpointer=checkpointer)
    final = graph.invoke(
        initial_state(query, scenario_params),
        config={"configurable": {"thread_id": thread_id}},
    )
    log_run(final)
    if learn:
        learn_async(final, consolidate=consolidate)
    return final


def initial_state(query: str = "", scenario_params: dict | None = None) -> EnergyIntelligenceBoard:
    """A fully-populated blank board. Every key the reducers/agents touch is
    pre-seeded so `graph.invoke(initial_state(...))` never KeyErrors."""
    return {
        "query": query,
        "scenario_params": scenario_params or {},
        "messages": [],
        "risk_signals": [],
        "corridor_risk": {},
        "corridor_events": {},
        "assessment_failed": False,
        "root_causes": [],
        "scenarios": [],
        "affected_refineries": [],
        "affected_routes": [],
        "twin_state": {},
        "bids": [],
        "evaluated_bids": [],
        "recommended_mix": {},
        "response_plan": {},
        "final_recommendation": "",
        "retrieved_memories": [],
        "working_context": {},
        "audit_trail": [],
        "constitution_flags": [],
        "current_agent": "",
        "stigmergy_markers": [],
        "pheromone_field": {},
    }
