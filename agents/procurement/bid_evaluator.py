"""
Bid Evaluator — the procurement pod's fan-in node.

The three regional bidders fan OUT and append raw bids to `bids` concurrently. The
evaluator is the single JOIN that reads them all, ranks them, and composes the
cross-region mix that actually covers SCTD's shortfall. Being the sole writer after
the fan-in, it is the ONE procurement node allowed to write plain state fields
(`current_agent`, `evaluated_bids`, `recommended_mix`, `constitution_flags`) — the
bidders may not, or concurrent writes would clobber (LangGraph InvalidUpdateError).

Selection ranks bids on lowest TOTAL ECONOMIC IMPACT, not lowest sticker price:
  1. Independently re-screen every bid via the procurement constitution (never trust
     the flags a bidder attached). Sanctioned bids are EXCLUDED from the mix outright.
  2. Score each eligible bid = price/bbl + penalties (grade no affected refinery can
     run, routing through the disrupted corridor) + a COST-OF-DELAY term:
     transit_days × impact_per_day, where impact_per_day scales with how urgent the
     shortfall is (SCTD's critical/stressed refinery counts). A mild shortfall keeps
     impact_per_day small so the cheapest cargo wins; a critical one makes every extra
     day expensive, so a faster-but-pricier cargo can win.
  3. Greedily fill the gap lowest-impact-first, trimming the last cargo so the mix
     total lands on the gap (coverage ~1.0x, inside the constitution's 0.8x–1.3x band).

Then deposit a 'bid' pheromone per committed cargo so the rest of the board sees
supply has been marshalled against the gap. Fully deterministic — no LLM.

Scoring weights live in data/procurement_params.json (illustrative until calibrated,
same status as dsm_params.json); a baked-in fallback keeps the node running if the
file is missing. Urgency ties to SCTD status bands now, and to spr_calculator.py once
it exists (backlog).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from graph.eib_state import EnergyIntelligenceBoard, StigmergyMarker
from eib_guardrails.constitution_checker import check as constitution_check

_PARAMS_PATH = Path(__file__).parent.parent.parent / "data" / "procurement_params.json"

_DEFAULT_PARAMS = {
    "grade_mismatch_penalty_usd": 15.0,
    "disrupted_route_penalty_usd": 10.0,
    "delay_cost": {
        "base_per_bbl_per_day": 0.15,
        "urgency_extra_per_bbl_per_day": 2.0,
        "critical_refinery_weight": 0.5,
        "stressed_refinery_weight": 0.15,
        "max_urgency": 1.0,
    },
}


def _load_params() -> tuple[dict, bool]:
    """Return (params, loaded_from_file). Missing/broken file -> documented defaults;
    the audit records which was used (loud, not silent) — same pattern as dsm_agent."""
    try:
        with open(_PARAMS_PATH, encoding="utf-8") as f:
            return json.load(f), True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _DEFAULT_PARAMS, False


_PARAMS, _PARAMS_FROM_FILE = _load_params()

_GRADE_MISMATCH_PENALTY = float(_PARAMS.get("grade_mismatch_penalty_usd", 15.0))
_DISRUPTED_ROUTE_PENALTY = float(_PARAMS.get("disrupted_route_penalty_usd", 10.0))
_DELAY = _PARAMS.get("delay_cost", _DEFAULT_PARAMS["delay_cost"])

# Coverage band the mix must land in (mirrors PROC-06 in the constitution).
_COVERAGE_MIN = 0.8
_COVERAGE_MAX = 1.3


def _disrupted_corridors(state: EnergyIntelligenceBoard) -> set[str]:
    twin = state.get("twin_state", {}) or {}
    return {
        c.get("id")
        for c in (twin.get("corridors", []) or [])
        if float(c.get("disruption_fraction", 0.0) or 0.0) > 0.0
    }


def _urgency(twin: dict) -> float:
    """Shortfall severity in [0, 1] from SCTD's status bands. A critical refinery
    weighs more than a stressed one; urgency 0 means a relaxed shortfall (cheapest
    cargo wins), urgency 1 means every open day is very expensive (speed wins)."""
    critical = int(twin.get("critical_count", 0) or 0)
    stressed = int(twin.get("stressed_count", 0) or 0)
    raw = (critical * float(_DELAY["critical_refinery_weight"])
           + stressed * float(_DELAY["stressed_refinery_weight"]))
    return round(min(float(_DELAY["max_urgency"]), max(0.0, raw)), 4)


def _impact_per_day(urgency: float) -> float:
    """$/bbl cost of each extra day the gap stays open, scaled by urgency."""
    return round(float(_DELAY["base_per_bbl_per_day"])
                 + urgency * float(_DELAY["urgency_extra_per_bbl_per_day"]), 4)


def _score(bid: dict, impact_per_day: float) -> float:
    """Lower is better. Total economic impact per bbl: landed price, plus penalties
    for an unusable grade / self-defeating route, plus the cost of delay
    (transit_days × impact_per_day — the urgency-scaled cost-of-waiting term)."""
    score = float(bid.get("price_per_bbl", 0.0))
    if bid.get("grade_compatible") is False:      # None (unknown) is not penalised
        score += _GRADE_MISMATCH_PENALTY
    if bid.get("routes_through_disrupted"):
        score += _DISRUPTED_ROUTE_PENALTY
    score += float(bid.get("transit_days_to_india", 0) or 0) * impact_per_day
    return round(score, 4)


def _eligible(bid: dict) -> tuple[bool, str | None]:
    """Can this bid enter the mix? Sanctioned or zero-volume bids cannot."""
    if bid.get("sanctions_status") == "blocked":
        return False, "sanctioned"
    try:
        if float(bid.get("volume_mbd", 0.0)) <= 0:
            return False, "non_positive_volume"
    except (TypeError, ValueError):
        return False, "non_numeric_volume"
    return True, None


def _compose_mix(ranked: list[dict], gap: float) -> tuple[list[dict], float]:
    """Greedy cheapest-first fill. Returns (components, total_volume). The last
    cargo is trimmed so the running total lands on the gap — keeping coverage near
    1.0x and the mix total exactly equal to the sum of its components (PROC-07)."""
    components: list[dict] = []
    total = 0.0
    for bid in ranked:
        if not bid.get("_eligible"):
            continue
        need = round(gap - total, 6)
        if need <= 0:
            break
        take = round(min(float(bid["volume_mbd"]), need), 4)
        if take <= 0:
            continue
        component = {**bid, "volume_mbd": take}
        component.pop("_eligible", None)
        component.pop("_exclude_reason", None)
        components.append(component)
        total = round(total + take, 4)
    return components, total


def bid_evaluator_node(state: EnergyIntelligenceBoard) -> dict:
    now = datetime.now(timezone.utc).isoformat()

    bids = state.get("bids", []) or []
    twin = state.get("twin_state", {}) or {}
    gap = float(twin.get("total_india_shortfall_mbd", 0.0) or 0.0)
    affected = state.get("affected_refineries", []) or []
    disrupted = _disrupted_corridors(state)

    # ── 0. Cost-of-delay: how expensive is each extra day the gap stays open? ──
    urgency = _urgency(twin)
    impact_per_day = _impact_per_day(urgency)

    # ── 1. Rank every bid (annotate eligibility + score; keep the full list) ──
    ranked: list[dict] = []
    for bid in bids:
        ok, reason = _eligible(bid)
        ranked.append({
            **bid,
            "score":          _score(bid, impact_per_day),
            "_eligible":      ok,
            "_exclude_reason": reason,
        })
    ranked.sort(key=lambda b: (not b["_eligible"], b["score"]))

    # ── 2. Compose the mix that covers the gap (cheapest eligible first) ──
    if gap > 0:
        components, total = _compose_mix(ranked, gap)
    else:
        components, total = [], 0.0

    coverage_ratio = round(total / gap, 4) if gap > 0 else None
    est_daily_cost_usd = round(
        sum(float(c["volume_mbd"]) * 1e6 * float(c["price_per_bbl"]) for c in components)
    )
    selected_ids = {c.get("supplier_id") for c in components}

    recommended_mix = {
        "gap_mbd":            round(gap, 4),
        "total_volume_mbd":   round(total, 4),
        "coverage_ratio":     coverage_ratio,
        "covers_gap":         gap <= 0 or (_COVERAGE_MIN <= (coverage_ratio or 0) <= _COVERAGE_MAX),
        "components":         components,
        "est_daily_cost_usd": est_daily_cost_usd,
        "urgency":            urgency,           # shortfall severity that shaped the ranking
        "impact_per_day_usd": impact_per_day,    # cost-of-delay applied per transit day
        "generated_at":       now,
    }

    # ── 3. Independent constitution re-verification (never trust bidder flags) ──
    check_result = constitution_check("procurement", {
        "bids":                bids,
        "recommended_mix":     recommended_mix,
        "disrupted_corridors": list(disrupted),
        "affected_refineries": affected,
    })
    violations = check_result.get("violations", [])

    # ── 4. Present the ranked bids (strip internal annotations) ──
    evaluated_bids = []
    for b in ranked:
        clean = {k: v for k, v in b.items() if not k.startswith("_")}
        clean["selected"] = clean.get("supplier_id") in selected_ids
        clean["eligible"] = b["_eligible"]
        clean["exclude_reason"] = b["_exclude_reason"]
        evaluated_bids.append(clean)

    # ── 5. Deposit a 'bid' pheromone per committed cargo (supply marshalled) ──
    markers: list[StigmergyMarker] = []
    for c in components:
        corridor = c.get("delivery_corridor")
        if not corridor:
            continue
        markers.append({
            "type":         "bid",
            "target":       corridor,
            "intensity":    round(min(1.0, float(c["volume_mbd"]) / gap), 4) if gap > 0 else 0.0,
            "deposited_by": "bid_evaluator",
            "timestamp":    now,
            "decay_rate":   0.1,
        })

    audit = [{
        "agent":            "bid_evaluator",
        "action":           "evaluate_bids",
        "bids_received":    len(bids),
        "eligible_bids":    sum(1 for b in ranked if b["_eligible"]),
        "excluded_bids":    sum(1 for b in ranked if not b["_eligible"]),
        "urgency":          urgency,
        "impact_per_day_usd": impact_per_day,
        "params_source":    "file" if _PARAMS_FROM_FILE else "defaults",
        "gap_mbd":          round(gap, 4),
        "mix_volume_mbd":   round(total, 4),
        "coverage_ratio":   coverage_ratio,
        "covers_gap":       recommended_mix["covers_gap"],
        "components":       len(components),
        "constitution_check": check_result,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }]

    return {
        "current_agent":     "bid_evaluator",
        "evaluated_bids":    evaluated_bids,
        "recommended_mix":   recommended_mix,
        "stigmergy_markers": markers,
        "audit_trail":       audit,
        "constitution_flags": violations,
    }
