"""Deterministic projections of board state for the API and chat layers.

Extracts from _summarize (moved here so both main.py and chat.py can import it
without a circular dependency) plus component and follow-up builders that turn
the raw board into a typed payload the Streamlit UI renders 1:1.

No LLM, no FastAPI — pure functions only.
"""
from __future__ import annotations

_TOL = 1e-6


# ── summarize_final ─────────────────────────────────────────────────────────────

def summarize_final(final: dict) -> dict:
    """Curate the big final board state into a useful response.

    Drops the raw audit_trail and keeps the decision-relevant fields; the twin
    snapshot (with geojson for the map) is served separately by GET /twin.
    """
    plan = final.get("response_plan", {}) or {}
    twin = final.get("twin_state", {}) or {}
    markers = final.get("stigmergy_markers", []) or []

    top_markers = sorted(
        markers, key=lambda m: float(m.get("intensity", 0)), reverse=True,
    )[:5]

    # Evidence base of the run: zero articles means GRI assessed blind, and any
    # "routine / all-clear" must be read as low confidence, not as a calm world.
    news_status = None
    for e in final.get("audit_trail", []) or []:
        if e.get("agent") == "gri_agent" and e.get("action") == "tool_fetch":
            news_status = e.get("news_status")
            break

    return {
        "query":                final.get("query", ""),
        "escalation_level":     plan.get("escalation_level"),
        "final_recommendation": final.get("final_recommendation", ""),
        "response_plan":        plan,
        "corridor_risk":        final.get("corridor_risk", {}),
        "twin_summary": {
            "total_india_shortfall_mbd": twin.get("total_india_shortfall_mbd"),
            "critical_count":            twin.get("critical_count"),
            "stressed_count":            twin.get("stressed_count"),
        },
        "recommended_mix":     final.get("recommended_mix", {}),
        "news_evidence": {
            "article_count": len(final.get("risk_signals", []) or []),
            "news_status":   news_status,
        },
        "retrieved_memories":  final.get("retrieved_memories", []),
        "constitution_flags":  final.get("constitution_flags", []),
        "pheromone_field":     final.get("pheromone_field", {}) or {},
        "stigmergy": {
            "marker_count": len(markers),
            "top_markers": [
                {
                    "type":         m.get("type"),
                    "target":       m.get("target"),
                    "intensity":    m.get("intensity"),
                    "deposited_by": m.get("deposited_by"),
                }
                for m in top_markers
            ],
        },
    }


# ── build_components ────────────────────────────────────────────────────────────

def _tone(value: float | int | str | None, kind: str) -> str:
    """Map a value to a severity tone for the UI."""
    if kind == "escalation":
        s = str(value or "routine").lower()
        if s == "critical":
            return "critical"
        if s in ("elevated", "watch"):
            return "elevated"
        return "ok"
    if kind == "count":
        return "critical" if (value or 0) > 0 else "ok"
    if kind == "gap":
        return "critical" if (value or 0) > _TOL else "ok"
    if kind == "residual":
        return "elevated" if (value or 0) > _TOL else "ok"
    if kind == "coverage":
        r = float(value or 0)
        if r >= 0.95:
            return "ok"
        if r >= 0.8:
            return "elevated"
        return "critical"
    return "ok"


def build_components(summary: dict, twin_state: dict) -> list[dict]:
    """Build the typed components[] payload the Streamlit UI renders 1:1.

    Every value is copied from summary / twin_state — nothing invented.
    Component types: map, metrics, mix_table, follow_ups.
    """
    components: list[dict] = []

    # ── Map ──
    geojson = twin_state.get("geojson", {}) or {}
    features = geojson.get("features", []) or []
    if features:
        counts: dict[str, int] = {}
        for f in features:
            kind = (f.get("properties") or {}).get("kind", "unknown")
            counts[kind] = counts.get(kind, 0) + 1
        components.append({
            "type": "map",
            "title": "Digital twin — corridors, refineries, reroutes",
            "geojson": geojson,
            "counts": counts,
        })

    # ── Metrics ──
    plan = summary.get("response_plan", {}) or {}
    ts = summary.get("twin_summary", {}) or {}
    proc = plan.get("procurement", {}) or {}
    gap = float(ts.get("total_india_shortfall_mbd") or 0)
    critical = int(ts.get("critical_count") or 0)
    stressed = int(ts.get("stressed_count") or 0)
    has_gap = gap > _TOL
    coverage = float(proc.get("coverage_ratio") or 0) if has_gap else None
    residual = float(proc.get("residual_gap_mbd") or 0)

    components.append({
        "type": "metrics",
        "title": "Board readout",
        "items": [
            {"label": "Escalation",          "value": summary.get("escalation_level") or "routine",
             "unit": None, "tone": _tone(summary.get("escalation_level"), "escalation")},
            {"label": "India shortfall",     "value": round(gap, 3),
             "unit": "mbd", "tone": _tone(gap, "gap")},
            {"label": "Critical refineries", "value": critical,
             "unit": None, "tone": _tone(critical, "count")},
            {"label": "Stressed refineries", "value": stressed,
             "unit": None, "tone": _tone(stressed, "count")},
            {"label": "Coverage",            "value": round(coverage, 2) if coverage is not None else "—",
             "unit": "×" if coverage is not None else None,
             "tone": _tone(coverage, "coverage") if coverage is not None else "ok"},
            {"label": "Residual gap",        "value": round(residual, 3),
             "unit": "mbd", "tone": _tone(residual, "residual")},
        ],
    })

    # Evidence base — zero articles means the run was blind; flag it, so a green
    # board can never masquerade as a verified calm world.
    news = summary.get("news_evidence", {}) or {}
    articles = news.get("article_count")
    if articles is not None:
        components[-1]["items"].append({
            "label": "News evidence", "value": articles, "unit": "articles",
            "tone": "elevated" if articles == 0 else "ok",
        })

    # ── Mix table ──
    actions = proc.get("committed_actions", []) or []
    if actions or gap > _TOL:
        components.append({
            "type": "mix_table",
            "title": "Recommended procurement mix",
            "rows": actions,
            "coverage_ratio": proc.get("coverage_ratio"),
            "covers_gap": proc.get("covers_gap"),
            "residual_gap_mbd": residual,
            "spr_bridge": proc.get("spr_bridge"),
        })

    # ── Follow-ups ──
    components.append({
        "type": "follow_ups",
        "options": suggest_follow_ups(summary),
    })

    return components


# ── suggest_follow_ups ──────────────────────────────────────────────────────────

def suggest_follow_ups(summary: dict) -> list[str]:
    """Deterministic follow-up suggestions from board state.

    Ordered rules, first 4 win.  No LLM — follow-ups re-enter the router as user
    turns, so they must be reproducible, free, and reliable.
    """
    plan = summary.get("response_plan", {}) or {}
    ts = summary.get("twin_summary", {}) or {}
    proc = plan.get("procurement", {}) or {}
    sit = plan.get("situation", {}) or {}
    corridor_risk = summary.get("corridor_risk", {}) or {}
    flags = summary.get("constitution_flags", []) or []

    gap = float(ts.get("total_india_shortfall_mbd") or 0)
    residual = float(proc.get("residual_gap_mbd") or 0)
    spr = proc.get("spr_bridge")
    critical = int(ts.get("critical_count") or 0)

    # Any sanctioned bid in the committed actions?
    has_flag = bool(flags)
    for a in proc.get("committed_actions", []) or []:
        if (a.get("sanctions_status") or "clear") != "clear":
            has_flag = True
            break

    pool: list[str] = []

    if residual > _TOL:
        pool.append("What if we bridge the residual gap with an SPR drawdown?")
    if spr:
        pool.append("How many days can the SPR bridge hold?")
    if has_flag:
        pool.append("Which suppliers or outputs were flagged, and why?")
    if critical > 0:
        pool.append("Which refineries are critical and what are their reroute options?")

    # Top corridor by score
    if corridor_risk:
        top_c = max(corridor_risk, key=lambda c: _risk_score(corridor_risk[c]))
        if _risk_score(corridor_risk[top_c]) >= 0.5:
            pool.append(f"What if the {top_c} disruption lasts twice as long?")

    # Quiet board
    if gap <= _TOL:
        pool.append("What if the Strait of Hormuz closes tomorrow?")
        pool.append("Which corridor is currently highest-risk?")

    return pool[:4]


def _risk_score(v) -> float:
    if isinstance(v, dict):
        return float(v.get("score", 0))
    return float(v or 0)
