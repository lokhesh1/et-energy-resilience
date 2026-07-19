"""
Procurement sourcing — shared bidder logic for the West Africa, Americas and
Spot Market agents.

A "bidder" answers one question: given the current India crude shortfall (from
SCTD's twin) and today's Brent price, what can MY region's suppliers offer to help
close it, and at what landed price? It is FULLY DETERMINISTIC — a filter over the
seeded supplier catalog (data/suppliers.json) plus ONE live input (Brent spot via
tools.price_feed). No LLM, so there is nothing to hallucinate: every bid number is
either read from the catalog or arithmetic (price_per_bbl = brent_ref + premium).

The three regional agents are THIS one function with a different `region` filter;
Spot additionally reacts to scarcity — its premium rises with pheromone intensity,
mirroring how a real spot market spikes when a corridor is choked.

Each bid is INDEPENDENTLY sanctions-screened (tools.sanctions_check) and stamped
sanctions_status "blocked" if it matches the SDN seed — surfaced but never cleared.
Grade compatibility against the disruption-affected refineries (tools.grade_lookup)
and a disrupted-delivery-corridor check are attached as ranking hints; the
procurement constitution re-verifies all of this at the evaluator stage rather than
trusting these flags.

Bid sizing: volume_mbd = min(supplier max_volume_mbd, gap). Each supplier offers up
to the whole gap, capped at what it can lift; the Bid Evaluator composes the actual
cross-region mix (and enforces coverage via PROC-06).

Concurrency: the three bidders run as a parallel fan-out, so a node here writes ONLY
reducer-annotated state keys (`bids`, `audit_trail` — both operator.add). It must
NOT write plain fields like `current_agent`/`constitution_flags`, which two
concurrent bidders would clobber (LangGraph InvalidUpdateError). The single-writer
Bid Evaluator owns those. Bidders are pheromone CONSUMERS (spot reads the scarcity
field); the evaluator, not a bidder, deposits the 'bid' pheromone.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from graph.eib_state import EnergyIntelligenceBoard
from tools.price_feed import fetch_price
from tools import sanctions_check
from tools import grade_lookup

_SUPPLIERS_PATH = Path(__file__).parent.parent.parent / "data" / "suppliers.json"

# Used only when the live Brent fetch fails — a plausible standing price so a bid
# is still priced (and the audit records that a fallback was used, loud not silent).
_BRENT_FALLBACK = 80.0

# Spot only: extra USD/bbl added to the premium at full scarcity (pheromone = 1.0).
# Kept well inside the constitution's |premium| <= 40 band.
_SPOT_SCARCITY_SURCHARGE_MAX = 5.0


def _load_suppliers() -> tuple[list[dict], bool]:
    """Return (suppliers, loaded_from_file). Missing/broken file -> [] so the node
    still runs (produces no bids) and the audit records which source was used."""
    try:
        with open(_SUPPLIERS_PATH, encoding="utf-8") as f:
            return json.load(f).get("suppliers", []), True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return [], False


def _fetch_brent() -> tuple[float, bool, str]:
    """Return (brent_price, is_live, status). Falls back to a fixed price on any
    failure so a bidder never crashes on a dead network."""
    try:
        res = fetch_price()
        if res["status"] == "ok":
            return float(res["data"]["current_price"]), True, "ok"
        return _BRENT_FALLBACK, False, res["status"]
    except Exception:
        return _BRENT_FALLBACK, False, "exception"


def _disrupted_corridors(state: EnergyIntelligenceBoard) -> set[str]:
    """Corridors the twin currently marks as disrupted (disruption_fraction > 0).
    A bid whose delivery corridor is in this set does not escape the disruption it
    is meant to relieve (PROC-05 warn)."""
    twin = state.get("twin_state", {}) or {}
    return {
        c.get("id")
        for c in (twin.get("corridors", []) or [])
        if float(c.get("disruption_fraction", 0.0) or 0.0) > 0.0
    }


def _scarcity(pheromone_field: dict) -> float:
    """Strongest pheromone in the field, clamped to [0, 1]. DSM deposits 'demand'
    and SCTD 'bottleneck' markers; their peak intensity is the scarcity signal the
    spot market prices against."""
    if not pheromone_field:
        return 0.0
    return max(0.0, min(1.0, max(pheromone_field.values())))


def _build_bid(
    supplier: dict,
    brent: float,
    gap: float,
    affected: list[str],
    disrupted: set[str],
    scarcity: float,
    scales_with_scarcity: bool,
) -> dict:
    """Turn one catalog supplier into a fully-specified, independently-screened bid.

    Every field the procurement constitution recomputes is carried explicitly:
    brent_ref + price_premium_usd == price_per_bbl, and volume_mbd <= max_volume_mbd.
    """
    premium = float(supplier.get("price_premium_usd", 0.0))
    surcharge_applied = False
    if scales_with_scarcity and scarcity > 0:
        premium = round(premium + scarcity * _SPOT_SCARCITY_SURCHARGE_MAX, 3)
        surcharge_applied = True

    price = round(brent + premium, 4)

    max_vol = float(supplier.get("max_volume_mbd", 0.0))
    volume = round(min(max_vol, gap), 4)

    supplier_name = supplier.get("supplier", "")
    screen = sanctions_check.check_supplier(supplier_name)
    sanctioned = screen["status"] == "ok" and screen["data"]["sanctioned"]

    grade = supplier.get("grade")
    grade_compatible = None  # None = unknown (no affected refineries to check against)
    if grade and affected:
        gres = grade_lookup.check_grade_against_refineries(grade, affected)
        if gres["status"] == "ok":
            grade_compatible = gres["data"]["any_compatible"]

    corridor = supplier.get("delivery_corridor")

    return {
        "supplier_id":               supplier.get("id"),
        "supplier":                  supplier_name,
        "region":                    supplier.get("region"),
        "grade":                     grade,
        "delivery_corridor":         corridor,
        "load_port":                 supplier.get("load_port"),
        "transit_days_to_india":     supplier.get("transit_days_to_india"),
        "max_volume_mbd":            max_vol,
        "volume_mbd":                volume,
        "brent_ref":                 brent,
        "price_premium_usd":         premium,
        "price_per_bbl":             price,
        # Surfaced-but-blocked: a sanctioned supplier is still returned (visible) but
        # stamped so the evaluator can never clear it into a recommended mix.
        "sanctions_status":          "blocked" if sanctioned else "clear",
        "sanctions_matched_entity":  screen["data"].get("matched_entity") if sanctioned else None,
        # Ranking hints for the evaluator (the constitution re-verifies these):
        "grade_compatible":          grade_compatible,
        "routes_through_disrupted":  corridor in disrupted,
        "scarcity_surcharge_applied": surcharge_applied,
        "trade_terms":               supplier.get("trade_terms", "FOB"),
    }


def run_sourcing(
    state: EnergyIntelligenceBoard,
    region: str,
    *,
    premium_scales_with_scarcity: bool = False,
) -> dict:
    """Shared bidder entry point. Reads the gap from the twin, filters the catalog
    to `region`, and emits one bid per supplier into the `bids` reducer field.

    Returns a partial state dict with ONLY reducer-annotated keys (`bids`,
    `audit_trail`) — safe to run concurrently with the other regional bidders.
    """
    now = datetime.now(timezone.utc).isoformat()
    agent_name = f"{region}_agent"

    twin = state.get("twin_state", {}) or {}
    gap = float(twin.get("total_india_shortfall_mbd", 0.0) or 0.0)

    suppliers, from_file = _load_suppliers()
    region_suppliers = [s for s in suppliers if s.get("region") == region]

    # No shortfall → nothing to source. Emit no bids, but record why (an empty
    # bid list with no audit trail is indistinguishable from the node never running).
    if gap <= 0:
        return {
            "bids": [],
            "audit_trail": [{
                "agent":             agent_name,
                "action":            "sourcing_skipped",
                "reason":            "no_shortfall",
                "gap_mbd":           gap,
                "region_suppliers":  len(region_suppliers),
                "suppliers_source":  "file" if from_file else "empty",
                "timestamp":         now,
            }],
        }

    affected = state.get("affected_refineries", []) or []
    disrupted = _disrupted_corridors(state)
    scarcity = (
        _scarcity(state.get("pheromone_field", {}) or {})
        if premium_scales_with_scarcity else 0.0
    )

    brent, is_live, price_status = _fetch_brent()

    bids = [
        _build_bid(s, brent, gap, affected, disrupted, scarcity, premium_scales_with_scarcity)
        for s in region_suppliers
    ]

    audit = [{
        "agent":            agent_name,
        "action":           "sourcing",
        "region":           region,
        "gap_mbd":          round(gap, 4),
        "brent_ref":        brent,
        "brent_source":     "live" if is_live else "fallback",
        "price_status":     price_status,
        "scarcity":         round(scarcity, 4),
        "bids_generated":   len(bids),
        "blocked_bids":     sum(1 for b in bids if b["sanctions_status"] == "blocked"),
        "suppliers_source": "file" if from_file else "empty",
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }]

    return {"bids": bids, "audit_trail": audit}
