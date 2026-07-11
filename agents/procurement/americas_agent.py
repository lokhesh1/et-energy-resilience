"""
Americas sourcing agent — long-haul crude from the US Gulf, Canada and Brazil. The
farthest but most geopolitically insulated barrels (US WTI, Canadian WCS via Panama,
Brazilian Lula), the deepest "diversify away from the Middle East" option.

A thin wrapper over the shared deterministic sourcing logic; all bidding, pricing,
sanctions-screening and grade checks live in _sourcing_base.run_sourcing.
"""
from graph.eib_state import EnergyIntelligenceBoard
from agents.procurement._sourcing_base import run_sourcing


def americas_node(state: EnergyIntelligenceBoard) -> dict:
    return run_sourcing(state, "americas")
