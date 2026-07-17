"""
Experience Distiller — the agent that turns a completed run into durable memory.

Every other agent solves the *current* crisis; this one makes the board remember,
so run N+1 starts warmer than run N. It sits at the very end (after the Crisis
Coordinator), reads the final board state, and hands a compact trajectory to the
distillation engine, which extracts the reusable lesson and routes it into memory.

Division of labour (deliberate):
    - The distillation ENGINE (`memory/distillation.py` + `XMemory.distill_run`)
      already exists and is tested: LLM-extract → persist into episodic/semantic/
      procedural. This agent does NOT re-implement any of that.
    - This agent owns the two things the engine can't decide for itself:
        1. `build_trajectory(state)` — reduce the ~20-key board (with its giant
           audit_trail) into a small, high-signal digest the LLM can actually reason
           over. Feeding raw state would bury the signal and blow the token budget.
        2. `_run_outcome(state)` — label whether the run SUCCEEDED, deterministically
           from state facts (did the mix cover the gap? any block flags?). The LLM
           never judges its own outcome — that would be marking its own homework.

Best-effort, like the rest of the memory path: `distill_run` never raises, and a
failed/empty LLM extraction degrades to "nothing learned this run", never a crash.
The deterministic per-agent `remember()` calls (GRI/DSM/SCTD/coordinator) have
already logged the run's *facts*; the distiller adds the *lesson* on top.
"""
from datetime import datetime, timezone

from graph.eib_state import EnergyIntelligenceBoard
from memory.xmemory import XMemory

_xmemory = XMemory()

# Cap each list in the digest so a pathological run can't bloat the LLM prompt.
_MAX_ITEMS = 8


def _score_of(val) -> float:
    """corridor_risk values may be a bare float or a {'score': ...} dict."""
    if isinstance(val, dict):
        val = val.get("score", 0.0)
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return 0.0


def _run_outcome(state: EnergyIntelligenceBoard) -> str:
    """Did the run resolve the crisis? A pure function of state facts — no LLM.

    - No shortfall projected      → success (board correctly stood down).
    - Gap fully covered, no block → success (supply marshalled).
    - Otherwise                   → failure (residual gap or an integrity block).
    """
    twin = state.get("twin_state", {}) or {}
    gap = float(twin.get("total_india_shortfall_mbd", 0.0) or 0.0)
    if gap <= 0:
        return "success"

    plan = state.get("response_plan", {}) or {}
    proc = plan.get("procurement", {}) or {}
    mix = state.get("recommended_mix", {}) or {}

    covers = proc.get("covers_gap", mix.get("covers_gap", False))
    try:
        residual = float(proc.get("residual_gap_mbd", gap) or 0.0)
    except (TypeError, ValueError):
        residual = gap

    return "success" if covers and residual <= 0 else "failure"


def build_trajectory(state: EnergyIntelligenceBoard) -> dict:
    """Reduce the final board to a compact, high-signal digest for distillation.

    Keeps only what a future run would want to recall: the risk that drove it, the
    scenarios modelled, the physical impact, what procurement did, and how it ended.
    Drops the noise (raw audit_trail, geojson, per-refinery coordinates). Every value
    is copied from an upstream deterministic field — the digest invents nothing.
    """
    corridor_risk = state.get("corridor_risk", {}) or {}
    corridor_events = state.get("corridor_events", {}) or {}
    scenarios = state.get("scenarios", []) or []
    twin = state.get("twin_state", {}) or {}
    mix = state.get("recommended_mix", {}) or {}
    plan = state.get("response_plan", {}) or {}

    risks = sorted(
        ({"corridor": cid, "score": _score_of(v),
          "event_type": corridor_events.get(cid, "none")}
         for cid, v in corridor_risk.items()),
        key=lambda r: r["score"], reverse=True,
    )[:_MAX_ITEMS]

    scenario_digest = [{
        "corridor":           s.get("corridor"),
        "event_type":         s.get("event_type"),
        "volume_at_risk_mbd": s.get("volume_at_risk_mbd"),
        "india_exposure_mbd": s.get("india_exposure_mbd"),
        "duration_days":      s.get("duration_days"),
        "severity":           s.get("severity"),
    } for s in scenarios[:_MAX_ITEMS]]

    refineries = twin.get("refineries", []) or []

    # Reroutes are follow-up gold ("what are the reroute options?") — carry them
    # compactly. Corridor-level by nature: reroutes belong to corridors, not to
    # individual refineries.
    routes = [{
        "from_corridor":      r.get("from_corridor"),
        "to_corridor":        r.get("to_corridor"),
        "added_transit_days": r.get("added_transit_days"),
        "freight_cost_mult":  r.get("freight_cost_mult"),
        "volume_mbd":         r.get("volume_mbd"),
        "overloaded":         r.get("overloaded"),
    } for r in (twin.get("routes", []) or [])[:_MAX_ITEMS]]

    cargoes = [{
        "supplier":          c.get("supplier"),
        "region":            c.get("region"),
        "grade":             c.get("grade"),
        "volume_mbd":        c.get("volume_mbd"),
        "delivery_corridor": c.get("delivery_corridor"),
    } for c in (mix.get("components", []) or [])[:_MAX_ITEMS]]

    return {
        "query":            state.get("query", ""),
        "outcome":          _run_outcome(state),
        "escalation_level": plan.get("escalation_level"),
        "corridor_risks":   risks,
        "scenarios":        scenario_digest,
        "twin": {
            "gap_mbd":             round(float(twin.get("total_india_shortfall_mbd", 0.0) or 0.0), 4),
            "critical_count":      twin.get("critical_count", 0),
            "stressed_count":      twin.get("stressed_count", 0),
            # NOT capped: bounded by the physical refineries file (12), and a
            # truncated list makes follow-up answers contradict the board's count.
            # Stressed names carried too — a tension run can have 12 stressed and
            # 0 critical, and "which refineries?" must still be answerable.
            "critical_refineries": [r.get("name") for r in refineries
                                    if r.get("status") == "critical"],
            "stressed_refineries": [r.get("name") for r in refineries
                                    if r.get("status") == "stressed"],
            "disrupted_corridors": [c.get("id") for c in (twin.get("corridors", []) or [])
                                    if float(c.get("disruption_fraction", 0.0) or 0.0) > 0.0],
            "reroutes":            routes,
        },
        "procurement": {
            "covered_mbd":      mix.get("total_volume_mbd"),
            "coverage_ratio":   mix.get("coverage_ratio"),
            "covers_gap":       mix.get("covers_gap"),
            "residual_gap_mbd": (plan.get("procurement", {}) or {}).get("residual_gap_mbd"),
            "cargoes":          cargoes,
        },
        "unresolved_issues": (plan.get("unresolved_issues", []) or [])[:_MAX_ITEMS],
        "recommendation":    state.get("final_recommendation", ""),
    }


def experience_distiller_node(state: EnergyIntelligenceBoard) -> dict:
    """Distill one completed run into memory. Best-effort: `distill_run` never
    raises, so this always returns a clean audit entry describing what was learned
    (or why nothing was). Meant to run AFTER the coordinator — on its own clock
    (async), so it never blocks the answer the board just produced."""
    now = datetime.now(timezone.utc).isoformat()

    trajectory = build_trajectory(state)
    report = _xmemory.distill_run(trajectory)  # distill → persist; never raises

    audit = [{
        "agent":                "experience_distiller",
        "action":               "distill",
        "outcome":              trajectory["outcome"],
        "episodic_written":     report.get("episodic_written", 0),
        "semantic_written":     report.get("semantic_written", 0),
        "skill_written":        report.get("skill_written", False),
        "skill_skipped_reason": report.get("skill_skipped_reason"),
        "timestamp":            now,
    }]

    return {
        "current_agent": "experience_distiller",
        "audit_trail":   audit,
    }
