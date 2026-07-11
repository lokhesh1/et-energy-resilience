"""
West Africa sourcing agent — Atlantic-basin crude (Nigeria/Angola) that sails to
India around the Cape of Good Hope, structurally clear of the Persian Gulf. The
"diversify away from Hormuz" option with steady contract-style premiums.

A thin wrapper over the shared deterministic sourcing logic; all bidding, pricing,
sanctions-screening and grade checks live in _sourcing_base.run_sourcing.
"""
from graph.eib_state import EnergyIntelligenceBoard
from agents.procurement._sourcing_base import run_sourcing


def west_africa_node(state: EnergyIntelligenceBoard) -> dict:
    return run_sourcing(state, "west_africa")
