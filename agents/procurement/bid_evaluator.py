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
     the flags a bidder attached). Sanctioned bids are EXCLUDED from the mix outright,
     and so are bids delivering through an effectively-CLOSED corridor (twin
     disruption fraction >= 0.75) — volume that cannot physically arrive must never
     count as coverage. A degraded-but-passable corridor costs a $ penalty instead.
  2. Score each eligible bid = price/bbl + penalties (grade no affected refinery can
     run) + a DELIVERY-RISK uplift (price × f/(1−f), f = the delivery corridor's twin
     disruption fraction — expected cost per DELIVERED barrel; flat penalty only as
     fallback when f is unknown) + a COST-OF-DELAY term: transit_days ×
     impact_per_day, where impact_per_day scales with how urgent the shortfall is
     (SCTD's critical/stressed refinery counts). A mild shortfall keeps
     impact_per_day small so the cheapest cargo wins; a critical one makes every extra
     day expensive, so a faster-but-pricier cargo can win.
  3. Greedily fill the gap lowest-impact-first on RISK-DISCOUNTED volume: a cargo
     through a fraction-f corridor counts as volume×(1−f) toward the gap, so the mix
     buys nominally more when risky cargo is involved (nominal capped at the 1.3x
     band edge — PROC-06). The mix reports both totals; an effective shortfall
     surfaces as residual at the coordinator instead of fake 1.0x coverage.

Then deposit a 'bid' pheromone per committed cargo so the rest of the board sees
supply has been marshalled against the gap. Fully deterministic — no LLM.

Scoring weights live in data/procurement_params.json (illustrative until calibrated,
same status as dsm_params.json); a baked-in fallback keeps the node running if the
file is missing. Urgency ties to SCTD status bands + SPR days-of-cover: a gap the
strategic reserve could bridge for months is calmer than one that would drain it in
a week, even at the same refinery-status severity.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from graph.eib_state import EnergyIntelligenceBoard, StigmergyMarker
from eib_guardrails.constitution_checker import check as constitution_check
from tools.spr_calculator import days_of_cover as spr_days_of_cover

_PARAMS_PATH = Path(__file__).parent.parent.parent / "data" / "procurement_params.json"

_DEFAULT_PARAMS = {
    "grade_mismatch_penalty_usd": 15.0,
    "disrupted_route_penalty_usd": 10.0,
    "delay_cost": {
        "base_per_bbl_per_day": 0.15,
        "urgency_extra_per_bbl_per_day": 2.0,
        "critical_refinery_weight": 0.5,
        "stressed_refinery_weight": 0.06,
        "spr_weight": 0.6,
        "spr_comfort_days": 30.0,
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

# A delivery corridor at/above this disruption fraction is effectively CLOSED
# (mirrors corridor_status's "closed" band at 75%): cargo routed through it
# cannot escape the disruption it is meant to relieve, so such bids are
# EXCLUDED from the mix outright — not just penalized. Below the threshold the
# corridor is degraded-but-passable and the $ route penalty applies instead.
_CLOSED_CORRIDOR_FRACTION = 0.75


def _disrupted_corridors(state: EnergyIntelligenceBoard) -> set[str]:
    twin = state.get("twin_state", {}) or {}
    return {
        c.get("id")
        for c in (twin.get("corridors", []) or [])
        if float(c.get("disruption_fraction", 0.0) or 0.0) > 0.0
    }


def _closed_corridors(state: EnergyIntelligenceBoard) -> set[str]:
    """Corridors the twin marks as effectively closed — derived from state, never
    from bidder-attached flags (same never-trust discipline as the constitution)."""
    twin = state.get("twin_state", {}) or {}
    return {
        c.get("id")
        for c in (twin.get("corridors", []) or [])
        if float(c.get("disruption_fraction", 0.0) or 0.0) >= _CLOSED_CORRIDOR_FRACTION
    }


def _corridor_fractions(state: EnergyIntelligenceBoard) -> dict[str, float]:
    """corridor_id → twin disruption fraction. The SAME number DSM used to size
    the shortfall must price the relief cargo that transits that corridor —
    otherwise the board asserts '30% of Hormuz flow is at risk' while counting
    Hormuz-transiting relief volume at 100% face value."""
    twin = state.get("twin_state", {}) or {}
    return {
        c.get("id"): float(c.get("disruption_fraction", 0.0) or 0.0)
        for c in (twin.get("corridors", []) or [])
        if float(c.get("disruption_fraction", 0.0) or 0.0) > 0.0
    }


def _spr_pressure(gap_mbd: float) -> float:
    """Extra urgency in [0, 1] from strategic-reserve cover. If the SPR could
    bridge the whole gap for `spr_comfort_days` or more, pressure is 0; as
    days-of-cover shrinks below that horizon, pressure rises linearly toward 1.
    Best-effort — no SPR signal (no gap / tool failure) means no extra pressure."""
    try:
        days = spr_days_of_cover(gap_mbd)
    except Exception:
        return 0.0
    if days is None:
        return 0.0
    comfort = float(_DELAY.get("spr_comfort_days", 30.0))
    if comfort <= 0:
        return 0.0
    return max(0.0, 1.0 - days / comfort)


def _urgency(twin: dict) -> float:
    """Shortfall severity in [0, 1]: SCTD's status bands (a critical refinery
    weighs more than a stressed one) plus SPR pressure (thin days-of-cover on the
    gap). Urgency 0 means a relaxed shortfall (cheapest cargo wins), urgency 1
    means every open day is very expensive (speed wins)."""
    critical = int(twin.get("critical_count", 0) or 0)
    stressed = int(twin.get("stressed_count", 0) or 0)
    gap = float(twin.get("total_india_shortfall_mbd", 0.0) or 0.0)
    raw = (critical * float(_DELAY["critical_refinery_weight"])
           + stressed * float(_DELAY["stressed_refinery_weight"])
           + float(_DELAY.get("spr_weight", 0.6)) * _spr_pressure(gap))
    return round(min(float(_DELAY["max_urgency"]), max(0.0, raw)), 4)


def _impact_per_day(urgency: float) -> float:
    """$/bbl cost of each extra day the gap stays open, scaled by urgency."""
    return round(float(_DELAY["base_per_bbl_per_day"])
                 + urgency * float(_DELAY["urgency_extra_per_bbl_per_day"]), 4)


def _score(bid: dict, impact_per_day: float) -> float:
    """Lower is better. Total economic impact per bbl: landed price, plus penalties
    for an unusable grade / self-defeating route, plus the cost of delay
    (transit_days × impact_per_day — the urgency-scaled cost-of-waiting term).

    Delivery risk is priced by expectation, not a flat fee: with fraction f of
    the delivery corridor choked, only (1−f) of the cargo is expected to arrive,
    so the cost per DELIVERED barrel is price/(1−f) — an uplift of price·f/(1−f)
    that scales with how disrupted the corridor actually is. The flat
    `disrupted_route_penalty_usd` remains only as the fallback when the fraction
    is unknown (bidder flag without twin data)."""
    price = float(bid.get("price_per_bbl", 0.0))
    score = price
    if bid.get("grade_compatible") is False:      # None (unknown) is not penalised
        score += _GRADE_MISMATCH_PENALTY
    fraction = float(bid.get("delivery_risk_fraction", 0.0) or 0.0)
    if 0.0 < fraction < 1.0:
        score += price * fraction / (1.0 - fraction)
    elif bid.get("routes_through_disrupted"):
        score += _DISRUPTED_ROUTE_PENALTY
    score += float(bid.get("transit_days_to_india", 0) or 0) * impact_per_day
    return round(score, 4)


def _eligible(bid: dict, closed: set[str] = frozenset()) -> tuple[bool, str | None]:
    """Can this bid enter the mix? Sanctioned, zero-volume, or routed through an
    effectively-closed corridor — cannot. (Closed-corridor cargo would inflate
    coverage with volume that physically cannot arrive.)"""
    if bid.get("sanctions_status") == "blocked":
        return False, "sanctioned"
    if bid.get("delivery_corridor") in closed:
        return False, "delivery_corridor_closed"
    try:
        if float(bid.get("volume_mbd", 0.0)) <= 0:
            return False, "non_positive_volume"
    except (TypeError, ValueError):
        return False, "non_numeric_volume"
    return True, None


def _compose_mix(ranked: list[dict], gap: float,
                 fractions: dict[str, float] | None = None,
                 ) -> tuple[list[dict], float, float]:
    """Greedy cheapest-first fill on RISK-DISCOUNTED volume. Returns
    (components, total_nominal, total_effective).

    A cargo through a corridor with disruption fraction f is expected to deliver
    only (1−f) of what was bought, so the fill targets EFFECTIVE volume landing
    on the gap — buying nominally more when risky cargo is in the mix ("buy extra
    because some won't arrive"). Nominal total is capped at the PROC-06 band edge
    (_COVERAGE_MAX × gap); if effective coverage still falls short, the shortfall
    surfaces as residual at the coordinator instead of being papered over.

    PROC-07 invariant preserved: total_volume_mbd == sum of component volume_mbd
    (both nominal). Every component carries its delivery_risk_fraction and
    effective_volume_mbd so downstream consumers see both numbers."""
    fractions = fractions or {}
    components: list[dict] = []
    total_nominal = 0.0
    total_effective = 0.0
    nominal_cap = round(_COVERAGE_MAX * gap, 6)
    for bid in ranked:
        if not bid.get("_eligible"):
            continue
        need_effective = round(gap - total_effective, 6)
        if need_effective <= 0:
            break
        fraction = float(fractions.get(bid.get("delivery_corridor"), 0.0))
        survival = 1.0 - fraction
        if survival <= 0:            # fully closed — the eligibility gate owns this
            continue
        take = round(min(float(bid["volume_mbd"]),
                         need_effective / survival,
                         nominal_cap - total_nominal), 4)
        if take <= 0:
            continue
        effective = round(take * survival, 4)
        component = {**bid,
                     "volume_mbd": take,
                     "delivery_risk_fraction": round(fraction, 3),
                     "effective_volume_mbd": effective}
        component.pop("_eligible", None)
        component.pop("_exclude_reason", None)
        components.append(component)
        total_nominal = round(total_nominal + take, 4)
        total_effective = round(total_effective + effective, 4)
    return components, total_nominal, total_effective


def bid_evaluator_node(state: EnergyIntelligenceBoard) -> dict:
    now = datetime.now(timezone.utc).isoformat()

    bids = state.get("bids", []) or []
    twin = state.get("twin_state", {}) or {}
    gap = float(twin.get("total_india_shortfall_mbd", 0.0) or 0.0)
    affected = state.get("affected_refineries", []) or []
    disrupted = _disrupted_corridors(state)
    closed = _closed_corridors(state)
    fractions = _corridor_fractions(state)

    # ── 0. Cost-of-delay: how expensive is each extra day the gap stays open? ──
    urgency = _urgency(twin)
    impact_per_day = _impact_per_day(urgency)

    # ── 1. Rank every bid (annotate eligibility + score; keep the full list) ──
    ranked: list[dict] = []
    for bid in bids:
        ok, reason = _eligible(bid, closed)
        # Delivery risk from twin state (never the bidder's flag): the fraction
        # prices the risk in the score and discounts the volume in the fill.
        fraction = round(fractions.get(bid.get("delivery_corridor"), 0.0), 3)
        annotated = {**bid, "delivery_risk_fraction": fraction}
        ranked.append({
            **annotated,
            "score":          _score(annotated, impact_per_day),
            "_eligible":      ok,
            "_exclude_reason": reason,
        })
    ranked.sort(key=lambda b: (not b["_eligible"], b["score"]))

    # ── 2. Compose the mix that covers the gap (cheapest eligible first) ──
    if gap > 0:
        components, total, total_effective = _compose_mix(ranked, gap, fractions)
    else:
        components, total, total_effective = [], 0.0, 0.0

    coverage_ratio = round(total / gap, 4) if gap > 0 else None
    effective_coverage = round(total_effective / gap, 4) if gap > 0 else None
    est_daily_cost_usd = round(
        sum(float(c["volume_mbd"]) * 1e6 * float(c["price_per_bbl"]) for c in components)
    )
    selected_ids = {c.get("supplier_id") for c in components}

    recommended_mix = {
        "gap_mbd":            round(gap, 4),
        "total_volume_mbd":   round(total, 4),
        # Risk-discounted expected delivery: what actually counts against the gap.
        "effective_volume_mbd": round(total_effective, 4),
        "coverage_ratio":     coverage_ratio,          # nominal (PROC-06 band basis)
        "effective_coverage_ratio": effective_coverage,
        "covers_gap":         gap <= 0 or (
            (_COVERAGE_MIN <= (coverage_ratio or 0) <= _COVERAGE_MAX)
            and total_effective >= gap - 1e-3),  # 4dp-rounding headroom
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
        "closed_corridor_exclusions": sum(
            1 for b in ranked if b["_exclude_reason"] == "delivery_corridor_closed"),
        "urgency":          urgency,
        "impact_per_day_usd": impact_per_day,
        "params_source":    "file" if _PARAMS_FROM_FILE else "defaults",
        "gap_mbd":          round(gap, 4),
        "mix_volume_mbd":   round(total, 4),
        "mix_effective_mbd": round(total_effective, 4),
        "coverage_ratio":   coverage_ratio,
        "effective_coverage_ratio": effective_coverage,
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
