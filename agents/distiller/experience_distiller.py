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

    # Impact-ordered disruption drivers from the coordinator (debugger.md #20).
    # The chat MUST use this ordering (contribution to gap), not the score-sorted
    # corridor_risks below, when answering "which corridor causes the most shortfall."
    sit = plan.get("situation", {}) or {}
    disruption_drivers = sit.get("disruption_drivers") or []
    root_causes = sit.get("root_causes") or []

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
    # compactly: the corridor-level flow view AND the voyage-level per-refinery
    # options (which lanes are hit, feasible alternates or the honest "no sea
    # detour — pipeline bypass / re-source / SPR").
    routes = [{
        "from_corridor":      r.get("from_corridor"),
        "to_corridor":        r.get("to_corridor"),
        "added_transit_days": r.get("added_transit_days"),
        "freight_cost_mult":  r.get("freight_cost_mult"),
        "volume_mbd":         r.get("volume_mbd"),
        "overloaded":         r.get("overloaded"),
    } for r in (twin.get("routes", []) or [])[:_MAX_ITEMS]]

    refinery_reroutes = [{
        "refinery":         rr.get("name"),
        "port":             rr.get("port"),
        "feed_at_risk_mbd": rr.get("feed_at_risk_mbd"),
        "lanes": [{
            "corridor":                l.get("corridor"),
            "at_risk_mbd":             l.get("at_risk_mbd"),
            "no_maritime_alternative": l.get("no_maritime_alternative"),
            "bypass_capacity_mbd":     (l.get("bypass") or {}).get("capacity_mbd"),
            "options":                 [o.get("alt_route")
                                        for o in (l.get("options") or [])[:2]],
            "mitigation":              l.get("mitigation"),
        } for l in (rr.get("lanes") or [])[:2]],
    } for rr in (twin.get("refinery_reroutes", []) or [])[:_MAX_ITEMS]]

    cargoes = [{
        "supplier":          c.get("supplier"),
        "region":            c.get("region"),
        "grade":             c.get("grade"),
        "volume_mbd":        c.get("volume_mbd"),
        "delivery_corridor": c.get("delivery_corridor"),
        "transit_days":      c.get("transit_days_to_india"),
        "trade_terms":       c.get("trade_terms", "FOB"),
    } for c in (mix.get("components", []) or [])[:_MAX_ITEMS]]

    # News evidence: per-corridor article counts so the chat can answer
    # "which corridors have zero coverage?" without fabricating (Finding #1).
    # Also carries a small sample of article titles/sources for citation.
    news_evidence_by_corridor: dict = {}
    news_articles_sample: list[dict] = []
    for entry in state.get("audit_trail", []) or []:
        if (entry.get("agent") == "gri_agent"
                and entry.get("action") == "tool_fetch"):
            news_evidence_by_corridor = entry.get("evidence_by_corridor", {}) or {}
            break
    raw_signals = state.get("risk_signals", []) or []
    # Compact article sample: title + source + corridors, highest-trust first.
    for a in sorted(raw_signals, key=lambda x: float(x.get("trust_score") or 0),
                    reverse=True)[:_MAX_ITEMS]:
        news_articles_sample.append({
            "title":    a.get("title", ""),
            "source":   a.get("source", "unknown"),
            "corridors": a.get("corridors", []),
        })

    # Retrieved memories / precedents — so "have we seen this before?" is answerable.
    precedents = []
    for p in (state.get("retrieved_memories", []) or [])[:4]:
        precedents.append({
            "text":  p.get("text", ""),
            "score": p.get("score"),
        })

    proc_plan = plan.get("procurement", {}) or {}

    # Economic impact: compact block so cost/recovery questions are answerable.
    econ_raw = state.get("economic_impact", {}) or {}
    econ_digest: dict | None = None
    if econ_raw.get("total_exposure_usd") or econ_raw.get("do_nothing_cost_usd"):
        micro = econ_raw.get("micro", {}) or {}
        macro = econ_raw.get("macro", {}) or {}
        spike = econ_raw.get("brent_spike_estimate", {}) or {}
        timeline = econ_raw.get("recovery_timeline", {}) or {}
        econ_digest = {
            "total_exposure_usd":     econ_raw.get("total_exposure_usd"),
            "do_nothing_cost_usd":    econ_raw.get("do_nothing_cost_usd"),
            "plan_net_benefit_usd":   econ_raw.get("plan_net_benefit_usd"),
            "brent_spike_delta_usd":  spike.get("delta_usd"),
            "import_bill_delta_usd":  macro.get("import_bill_delta_usd"),
            "cpi_impact_bps":         macro.get("cpi_impact_bps"),
            "cad_gdp_impact_pct":     macro.get("cad_gdp_impact_pct"),
            "premium_spend_usd":      micro.get("premium_spend_usd"),
            "residual_loss_usd":      micro.get("residual_loss_usd"),
            "spr_refill_exposure_usd": micro.get("spr_refill_exposure_usd"),
            "days_to_normal":         timeline.get("days_to_normal"),
            "daily_loss_do_nothing_usd": (timeline.get("daily_loss_curve", [{}])[0]
                                          .get("daily_loss_do_nothing_usd")
                                          if timeline.get("daily_loss_curve") else None),
            "recovery_actions": [{
                "lever":           a.get("lever"),
                "net_benefit_usd": a.get("net_benefit_usd"),
                "description":     a.get("description"),
            } for a in (econ_raw.get("recovery_actions") or [])[:6]],
            "subsidy_vs_passthrough": econ_raw.get("subsidy_vs_passthrough"),
            "top_refinery_loss": (micro.get("refinery_losses", [{}])[0]
                                  if micro.get("refinery_losses") else None),
        }

    return {
        "query":            state.get("query", ""),
        "outcome":          _run_outcome(state),
        "escalation_level": plan.get("escalation_level"),
        "corridor_risks":   risks,
        "disruption_drivers": [{
            "corridor":             d.get("corridor"),
            "gap_contribution_mbd": d.get("gap_contribution_mbd"),
            "risk_score":           d.get("risk_score"),
            "event_type":           d.get("event_type"),
        } for d in disruption_drivers[:_MAX_ITEMS]],
        "root_causes": [{
            "origin":     rc.get("origin"),
            "driven":     rc.get("driven"),
            "reasoning":  rc.get("reasoning", ""),
        } for rc in root_causes[:4]],
        "scenarios":        scenario_digest,
        "twin": {
            "gap_mbd":             round(float(twin.get("total_india_shortfall_mbd", 0.0) or 0.0), 4),
            "critical_count":      twin.get("critical_count", 0),
            "stressed_count":      twin.get("stressed_count", 0),
            "critical_refineries": [r.get("name") for r in refineries
                                    if r.get("status") == "critical"],
            "stressed_refineries": [r.get("name") for r in refineries
                                    if r.get("status") == "stressed"],
            "disrupted_corridors": [c.get("id") for c in (twin.get("corridors", []) or [])
                                    if float(c.get("disruption_fraction", 0.0) or 0.0) > 0.0],
            "reroutes":            routes,
            "refinery_reroutes":   refinery_reroutes,
        },
        "procurement": {
            "covered_mbd":        mix.get("total_volume_mbd"),
            "coverage_ratio":     mix.get("coverage_ratio"),
            "covers_gap":         mix.get("covers_gap"),
            "residual_gap_mbd":   proc_plan.get("residual_gap_mbd"),
            "est_daily_cost_usd": proc_plan.get("est_daily_cost_usd"),
            "delivery_lag":       proc_plan.get("delivery_lag"),
            "cargoes":            cargoes,
        },
        "economic_impact":  econ_digest,
        "news_evidence": {
            "article_count":  len(raw_signals),
            "by_corridor":    news_evidence_by_corridor,
            "top_articles":   news_articles_sample,
        },
        "precedents":        precedents,
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
