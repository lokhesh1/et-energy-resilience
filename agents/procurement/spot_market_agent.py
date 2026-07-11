"""
Spot Market sourcing agent — fast, prompt cargoes (Fujairah/Ras Tanura Gulf spot,
Atlantic and West-Africa storage). The reactive bidder: shortest transit, but its
premium RISES with scarcity — the stronger the pheromone field (DSM demand + SCTD
bottleneck markers), the higher the spot surcharge, mirroring how a real spot market
spikes when a corridor is choked. Gulf spot cargoes route through Hormuz, so the
constitution flags them (PROC-05) when the strait is itself the disruption.

A thin wrapper over the shared deterministic sourcing logic; scarcity pricing is the
only difference from the other two bidders (premium_scales_with_scarcity=True).
"""
from graph.eib_state import EnergyIntelligenceBoard
from agents.procurement._sourcing_base import run_sourcing


def spot_market_node(state: EnergyIntelligenceBoard) -> dict:
    return run_sourcing(state, "spot", premium_scales_with_scarcity=True)
