"""Deterministic projections of board state for the API and chat layers.

Extracts from _summarize (moved here so both main.py and chat.py can import it
without a circular dependency) plus component and follow-up builders that turn
the raw board into a typed payload the Streamlit UI renders 1:1.

No LLM, no FastAPI — pure functions only.
"""
from __future__ import annotations

_TOL = 1e-6


# ── summarize_final ─────────────────────────────────────────────────────────────

# The inspectable evidence sample: an article COUNT the user can't open is not
# evidence. Capped, and round-robin'd across corridor tags so the sample covers
# every corridor that had news, not just the loudest one.
_EVIDENCE_ARTICLE_CAP = 12


def _shape_article(a: dict) -> dict:
    return {
        "title":       a.get("title", ""),
        "url":         a.get("url", ""),
        "source":      a.get("source", "unknown"),
        "trust_score": a.get("trust_score"),
        "trust_rated": a.get("trust_rated"),
        "corridors":   a.get("corridors", []) or [],
    }


def _trust(a: dict) -> float:
    return float(a.get("trust_score") or 0)


def _evidence_articles(signals: list[dict], cap: int = _EVIDENCE_ARTICLE_CAP) -> list[dict]:
    # Priority: corridor-tagged articles (actual corridor evidence) before
    # untagged sweep noise; within each bucket, highest-trust sources first —
    # a Reuters article at position 15 of the fetch must not lose its slot to
    # an unrated market wrap-up at position 1.
    tagged: dict[str, list[dict]] = {}
    untagged: list[dict] = []
    for a in signals:
        tags = a.get("corridors") or []
        if tags:
            tagged.setdefault(tags[0], []).append(a)
        else:
            untagged.append(a)
    for bucket in tagged.values():
        bucket.sort(key=_trust, reverse=True)
    untagged.sort(key=_trust, reverse=True)

    out: list[dict] = []
    depth = 0
    while len(out) < cap and any(depth < len(b) for b in tagged.values()):
        for tag in sorted(tagged):
            bucket = tagged[tag]
            if depth < len(bucket):
                out.append(_shape_article(bucket[depth]))
                if len(out) >= cap:
                    return out
        depth += 1
    for a in untagged:
        if len(out) >= cap:
            break
        out.append(_shape_article(a))
    return out

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
    # Also lift GRI's failure/tripwire signals out of the audit trail — a failed
    # scoring step or scores-contradict-evidence warning buried in the audit DB
    # is invisible; the UI must be able to show it (debugger.md #21).
    news_status = None
    by_corridor: dict = {}
    assessment = {
        "failed":                     bool(final.get("assessment_failed")),
        "failure_reason":             None,
        "evidence_ignored_corridors": [],
    }
    for e in final.get("audit_trail", []) or []:
        if e.get("agent") != "gri_agent":
            continue
        action = e.get("action")
        if action == "tool_fetch" and news_status is None:
            news_status = e.get("news_status")
            by_corridor = e.get("evidence_by_corridor", {}) or {}
        elif action == "llm_assessment" and e.get("llm_failure"):
            assessment["failed"] = True
            assessment["failure_reason"] = e.get("llm_failure")
        elif action == "evidence_ignored_warning":
            assessment["evidence_ignored_corridors"] = sorted(
                (e.get("corridors") or {}).keys())

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
            "by_corridor":   by_corridor,
            "articles":      _evidence_articles(final.get("risk_signals", []) or []),
        },
        "assessment":          assessment,
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

    # Failed risk scoring is the loudest fact on the board: every green tile
    # beside it is an unassessed default, so it goes FIRST (debugger.md #21).
    assessment = summary.get("assessment", {}) or {}
    if assessment.get("failed"):
        components[-1]["items"].insert(0, {
            "label": "Risk scoring", "value": "FAILED", "unit": None,
            "tone": "critical",
        })

    # Disrupted corridors, so the cause is countable at a glance — the names +
    # per-corridor impact render below the tiles (debugger.md #20).
    sit = plan.get("situation", {}) or {}
    drivers = sit.get("disruption_drivers") or []
    if drivers:
        components[-1]["items"].append({
            "label": "Disrupted corridors", "value": len(drivers), "unit": None,
            "tone": "critical",
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

    # Top corridor by IMPACT when the twin decomposed the gap (the corridor
    # costing the most, debugger.md #20); by score otherwise.
    drivers = sit.get("disruption_drivers") or []
    if drivers:
        pool.append(f"What if the {drivers[0]['corridor']} disruption lasts twice as long?")
    elif corridor_risk:
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
