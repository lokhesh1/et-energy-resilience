"""
SCTD — Supply Chain Digital Twin.

Projects DSM's *corridor-level* disruption scenarios onto *physical assets*: the
12 major Indian refineries. DSM says "Hormuz war → 21 mbd at risk"; SCTD says
"…therefore Jamnagar loses 45% of its feed, Mangalore 55%, and all rerouted
volume piles onto the Cape of Good Hope → bottleneck."

Design: FULLY DETERMINISTIC — no LLM. A digital twin is a state projection; every
number is `capacity × dependency_share × disruption_fraction`, traceable back to
GRI's risk score. There is nothing to hallucinate, so there is no LLM to guard.

Guardrails are placed where a *deterministic consumer* node can actually fail —
not bound-checks on our own static data (a CI test catches those earlier), but:
  1. Contract drift  — a scenario missing `disruption_fraction` is skipped AND
     flagged, never silently read as 0.0 (which would understate risk).
  2. Liveness        — active scenarios but zero total impact means the signal
     died in the handoff; flag it (a false "all clear" is the worst failure here).
Data integrity (shares sum ≤ 1, capacities > 0) lives in tests/test_refineries_data.py.

Memory + twin baseline are fire-and-forget / always-render, like GRI and DSM.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from graph.eib_state import EnergyIntelligenceBoard, StigmergyMarker
from tools.corridor_status import get_corridor_status
from tools.geospatial_mapper import build_supply_chain_map
from tools.route_ranker import rank_routes
from memory.xmemory import XMemory

_REFINERIES_PATH = Path(__file__).parent.parent / "data" / "refineries.json"

_xmemory = XMemory()

# Persist only the worst refinery impacts so the episodic log stays signal.
_MEMORY_PERSIST_STATUSES = {"critical"}

# Fraction of a refinery's capacity at risk → twin status band.
_CRITICAL_BAND = 0.30
_STRESSED_BAND = 0.10


def _load_refineries() -> tuple[list[dict], bool]:
    """Return (refineries, loaded_from_file). Missing/broken file -> [] so the
    node still produces a (degraded) twin; the audit records which was used."""
    try:
        with open(_REFINERIES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("refineries", []), True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return [], False


_REFINERIES, _REFINERIES_FROM_FILE = _load_refineries()

# Corridors that at least one refinery actually sources crude through. A scenario
# on a corridor NOT in this set (e.g. Panama — no Indian refinery depends on it)
# legitimately produces zero refinery impact, so it must NOT trip the liveness
# check. Only a zero on a *depended* corridor means the signal was lost.
_DEPENDED_CORRIDORS = {
    cid
    for r in _REFINERIES
    for cid in (r.get("corridor_dependency") or {})
}


# ── Deterministic twin projection ───────────────────────────────────────────────

def _status_for(at_risk_share: float) -> str:
    if at_risk_share >= _CRITICAL_BAND:
        return "critical"
    if at_risk_share >= _STRESSED_BAND:
        return "stressed"
    return "normal"


def _project_refinery(refinery: dict, fraction_by_corridor: dict[str, float]) -> dict:
    """Distribute corridor disruptions onto one refinery. Deterministic:
    feed_at_risk = capacity × Σ(dependency_share × disruption_fraction).
    Stores at_risk_share alongside the volume so the number is traceable."""
    capacity = float(refinery.get("capacity_mbd", 0.0))
    deps = refinery.get("corridor_dependency", {}) or {}

    at_risk_share = 0.0
    top_corridor = None
    top_contrib = 0.0
    for cid, share in deps.items():
        frac = fraction_by_corridor.get(cid, 0.0)
        contrib = float(share) * float(frac)
        at_risk_share += contrib
        if contrib > top_contrib:
            top_contrib = contrib
            top_corridor = cid

    at_risk_share = round(min(at_risk_share, 1.0), 4)
    feed_at_risk = round(capacity * at_risk_share, 4)

    return {
        "id":               refinery["id"],
        "name":             refinery.get("name", refinery["id"]),
        "operator":         refinery.get("operator"),
        "coast":            refinery.get("coast"),
        "lat":              refinery.get("lat"),
        "lon":              refinery.get("lon"),
        "capacity_mbd":     round(capacity, 4),
        "at_risk_share":    at_risk_share,   # stored so the number is recomputable
        "feed_at_risk_mbd": feed_at_risk,
        "at_risk_pct":      round(at_risk_share * 100),
        "top_corridor":     top_corridor if top_contrib > 0 else None,
        "status":           _status_for(at_risk_share),
    }


def _build_routes(scenarios: list[dict], corridors_by_id: dict[str, dict]) -> list[dict]:
    """One reroute per scenario that carries an alt route. `overloaded` flags the
    real-world bottleneck DSM can't see: rerouted volume exceeding the alternative
    corridor's own baseline capacity."""
    routes: list[dict] = []
    for sc in scenarios:
        reroute = sc.get("reroute")
        if not reroute:
            continue
        alt = reroute.get("alt_route")
        volume = float(sc.get("volume_at_risk_mbd", 0.0))
        alt_baseline = float(corridors_by_id.get(alt, {}).get("baseline_flow_mbd", 0.0))
        routes.append({
            "from_corridor":      sc["corridor"],
            "to_corridor":        alt,
            "added_transit_days": reroute.get("added_transit_days"),
            "freight_cost_mult":  reroute.get("freight_cost_mult"),
            "volume_mbd":         round(volume, 3),
            "alt_baseline_mbd":   round(alt_baseline, 3),
            "overloaded":         alt_baseline > 0 and volume > alt_baseline,
        })
    return routes


def _build_refinery_reroutes(impacts: list[dict],
                             fraction_by_corridor: dict[str, float]) -> list[dict]:
    """Voyage-level reroute options per AFFECTED refinery — a reroute belongs to
    a voyage (loading zone → the refinery's own harbour), not to a corridor.
    For each stressed/critical refinery, each disrupted supply lane it depends
    on gets the route_ranker's answer: feasible alternates, or the honest
    `no_maritime_alternative` (e.g. Hormuz — a dead end) with the pipeline
    bypass + fallback advice. Deterministic; the lane advice is computed once
    per corridor and shared."""
    disrupted = {cid: f for cid, f in fraction_by_corridor.items() if f > 0.0}
    if not disrupted:
        return []
    lane_advice = {cid: rank_routes(cid, fraction_by_corridor)["data"]
                   for cid in disrupted}
    by_id = {r["id"]: r for r in _REFINERIES}
    out: list[dict] = []
    for imp in impacts:
        if imp["status"] == "normal":
            continue
        raw = by_id.get(imp["id"], {})
        deps = raw.get("corridor_dependency", {}) or {}
        lanes: list[dict] = []
        for cid, share in deps.items():
            frac = disrupted.get(cid, 0.0)
            if frac <= 0.0:
                continue
            advice = lane_advice[cid]
            lanes.append({
                "corridor":                cid,
                "share":                   float(share),
                "at_risk_mbd":             round(imp["capacity_mbd"]
                                                 * float(share) * frac, 4),
                "no_maritime_alternative": advice.get("no_maritime_alternative", False),
                "bypass":                  advice.get("bypass"),
                "options":                 advice.get("options", []),
                "mitigation":              advice.get("fallback_advice"),
            })
        if lanes:
            lanes.sort(key=lambda l: l["at_risk_mbd"], reverse=True)
            out.append({
                "refinery":         imp["id"],
                "name":             imp["name"],
                "port":             raw.get("port") or raw.get("import_port"),
                "feed_at_risk_mbd": imp["feed_at_risk_mbd"],
                "lanes":            lanes,
            })
    return out


def _deposit_bottleneck_pheromones(
    impacts: list[dict], routes: list[dict]
) -> list[StigmergyMarker]:
    """SCTD signals the procurement pod indirectly: a 'bottleneck' marker per
    stressed/critical refinery (intensity ∝ at-risk share) and per overloaded
    reroute, so sourcing agents amplify against the real physical gap."""
    now = datetime.now(timezone.utc).isoformat()
    markers: list[StigmergyMarker] = []
    for imp in impacts:
        if imp["status"] not in ("stressed", "critical"):
            continue
        markers.append({
            "type":         "bottleneck",
            "target":       imp["id"],
            "intensity":    round(min(1.0, imp["at_risk_share"]), 4),
            "deposited_by": "sctd_agent",
            "timestamp":    now,
            "decay_rate":   0.1,
        })
    for route in routes:
        if route.get("overloaded"):
            markers.append({
                "type":         "bottleneck",
                "target":       route["to_corridor"],
                "intensity":    1.0,
                "deposited_by": "sctd_agent",
                "timestamp":    now,
                "decay_rate":   0.1,
            })
    return markers


# ── Node ────────────────────────────────────────────────────────────────────────

def sctd_node(state: EnergyIntelligenceBoard) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    flags: list[dict] = []

    scenarios = state.get("scenarios", []) or []

    # ── 1. Corridor baselines / live status ────────────────────────────────
    corridor_result = get_corridor_status()
    corridors = corridor_result["data"]["corridors"] if corridor_result["status"] == "ok" else []
    corridors_by_id = {c["id"]: c for c in corridors}

    audit: list[dict] = [{
        "agent":              "sctd_agent",
        "action":             "twin_input",
        "corridor_status":    corridor_result["status"],
        "scenario_count":     len(scenarios),
        "refineries_source":  "file" if _REFINERIES_FROM_FILE else "empty",
        "refinery_count":     len(_REFINERIES),
        "timestamp":          now,
    }]

    # ── 2. Build disruption fraction per corridor (skip quarantined) ────────
    # Guardrail 1 (contract drift): read the field as MISSING, not as 0.0. A
    # scenario without disruption_fraction is skipped AND flagged — never
    # silently understated to zero.
    fraction_by_corridor: dict[str, float] = {}
    impacting_scenarios = 0  # active scenarios that SHOULD move a refinery number
    for sc in scenarios:
        if sc.get("quarantined"):
            continue  # broken math — never propagate downstream
        cid = sc.get("corridor")
        frac = sc.get("disruption_fraction")
        if frac is None:
            flags.append({
                "rule_id":  "SCTD-CONTRACT",
                "severity": "block",
                "corridor": cid,
                "message":  f"{cid}: scenario missing 'disruption_fraction' "
                            f"— upstream contract drift, not modelled",
            })
            continue
        frac = float(frac)
        # a corridor could appear twice; keep the largest disruption
        fraction_by_corridor[cid] = max(fraction_by_corridor.get(cid, 0.0), frac)
        # only count it as "should-impact" if it's a real disruption on a corridor
        # some refinery depends on — otherwise a legitimate zero (e.g. Panama).
        if frac > 0.0 and cid in _DEPENDED_CORRIDORS:
            impacting_scenarios += 1

    # ── 3. Project onto refineries (deterministic) ─────────────────────────
    impacts = [_project_refinery(r, fraction_by_corridor) for r in _REFINERIES]
    affected = [i for i in impacts if i["status"] != "normal"]
    total_shortfall = round(sum(i["feed_at_risk_mbd"] for i in impacts), 4)

    # Per-corridor decomposition of that shortfall, so the fan-in can attribute
    # the gap honestly (which corridor is COSTING how much) instead of crediting
    # the whole number to whichever corridor has the loudest risk score.
    # shortfall_by_corridor[cid] = Σ_refineries capacity × dependency_share ×
    # disruption_fraction; sums to total_shortfall except where a refinery's
    # at_risk_share was capped at 1.0 (then the parts slightly overstate).
    shortfall_by_corridor: dict[str, float] = {}
    for r in _REFINERIES:
        cap = float(r.get("capacity_mbd", 0.0))
        for cid, share in (r.get("corridor_dependency") or {}).items():
            frac = fraction_by_corridor.get(cid, 0.0)
            if frac > 0.0:
                shortfall_by_corridor[cid] = round(
                    shortfall_by_corridor.get(cid, 0.0)
                    + cap * float(share) * frac, 4)

    # ── 4. Routes + bottleneck detection ───────────────────────────────────
    non_quarantined = [s for s in scenarios if not s.get("quarantined")]
    routes = _build_routes(non_quarantined, corridors_by_id)

    # ── 5. Guardrail 2 (liveness): active scenarios but zero impact means the
    # signal died in the handoff. A bound/recompute check is blind to this (0 is
    # in-bounds and self-consistent); only the input↔output contradiction sees it.
    if impacting_scenarios > 0 and total_shortfall == 0.0:
        flags.append({
            "rule_id":  "SCTD-LIVENESS",
            "severity": "block",
            "corridor": None,
            "message":  f"{impacting_scenarios} scenario(s) on depended corridors but "
                        f"total twin impact is 0.0 — refinery mapping produced no "
                        f"effect (signal lost?)",
        })

    # ── 6. Corridor view for the twin (mark disrupted corridors) ───────────
    corridor_view = []
    for c in corridors:
        frac = fraction_by_corridor.get(c["id"], 0.0)
        corridor_view.append({**c, "disruption_fraction": round(frac, 3)})

    # ── 7. GeoJSON for the Folium map ──────────────────────────────────────
    geo = build_supply_chain_map(corridors=corridor_view, refineries=impacts, routes=routes)

    twin_state = {
        "refineries":               impacts,
        "corridors":                corridor_view,
        "routes":                   routes,
        "total_india_shortfall_mbd": total_shortfall,
        "shortfall_by_corridor":    shortfall_by_corridor,
        "refinery_reroutes":        _build_refinery_reroutes(impacts, fraction_by_corridor),
        "critical_count":           sum(1 for i in impacts if i["status"] == "critical"),
        "stressed_count":           sum(1 for i in impacts if i["status"] == "stressed"),
        "geojson":                  geo["data"]["geojson"],
        "generated_at":             now,
    }

    # ── 8. Stigmergy: bottleneck markers for the procurement pod ────────────
    markers = _deposit_bottleneck_pheromones(affected, routes)

    audit.append({
        "agent":             "sctd_agent",
        "action":            "twin_projection",
        "affected_count":    len(affected),
        "total_shortfall":   total_shortfall,
        "overloaded_routes": sum(1 for r in routes if r.get("overloaded")),
        "flags":             flags,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    })

    # ── 9. Persist critical impacts (best-effort, never breaks the node) ────
    try:
        for imp in affected:
            if imp["status"] not in _MEMORY_PERSIST_STATUSES:
                continue
            _xmemory.remember(
                event_type="refinery_impact",
                agent="sctd_agent",
                payload={
                    "refinery":         imp["id"],
                    "status":           imp["status"],
                    "feed_at_risk_mbd": imp["feed_at_risk_mbd"],
                    "at_risk_pct":      imp["at_risk_pct"],
                    "top_corridor":     imp["top_corridor"],
                },
                outcome="success",
                text=(
                    f"{imp['name']} {imp['status']}: "
                    f"{imp['feed_at_risk_mbd']} mbd feed at risk ({imp['at_risk_pct']}%) "
                    f"via {imp['top_corridor']}"
                ),
            )
    except Exception:
        pass  # memory is best-effort; never break the node

    return {
        "current_agent":      "sctd_agent",
        "affected_refineries": [i["id"] for i in affected],
        "affected_routes":     routes,
        "twin_state":          twin_state,
        "stigmergy_markers":   markers,
        "audit_trail":         audit,
        "constitution_flags":  flags,
    }
