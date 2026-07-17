import json
from datetime import datetime, timezone
from pathlib import Path

CORRIDORS_PATH = Path(__file__).parent.parent / "data" / "corridors.json"

# Manual incident overrides (ops/testing hook via apply_incident/clear_incident).
# Format: corridor_id → {disruption_pct, status, last_incident}
# NOTE: nothing in the automated pipeline writes this — corridor disruption is
# driven by GRI risk scores → DSM scenarios, not this table. That is deliberate:
# a hypothetical query ("what if Hormuz closes") must never mutate global
# corridor state that the live twin would then serve as fact.
_ACTIVE_INCIDENTS: dict[str, dict] = {}


def _load_baselines() -> list[dict]:
    with open(CORRIDORS_PATH) as f:
        return json.load(f)


def _compute_risk_score(corridor: dict, disruption_pct: float) -> float:
    base = disruption_pct / 100.0
    factor_weight = len(corridor["risk_factors"]) * 0.05
    chokepoint_bonus = 0.15 if corridor["chokepoint"] else 0.0
    return round(min(base + factor_weight + chokepoint_bonus, 1.0), 4)


def _resolve_status(disruption_pct: float) -> str:
    if disruption_pct >= 75:
        return "closed"
    if disruption_pct >= 20:
        return "restricted"
    return "open"


def get_corridor_status(corridor_overrides: dict[str, dict] | None = None) -> dict:
    retrieved_at = datetime.now(timezone.utc)
    overrides = corridor_overrides or _ACTIVE_INCIDENTS

    try:
        baselines = _load_baselines()
    except Exception as e:
        return {
            "tool":                      "corridor_status",
            "status":                    "failed",
            "data":                      {"error": str(e)},
            "source_trust_avg":          1.0,
            "low_trust_sources_flagged": 0,
            "retrieved_at":              retrieved_at.isoformat(),
            "staleness_seconds":         0,
        }

    corridors = []
    trust_scores = []

    for c in baselines:
        cid = c["id"]
        override = overrides.get(cid, {})

        disruption_pct = float(override.get("disruption_pct", 0.0))
        current_flow   = round(c["baseline_flow_mbd"] * (1 - disruption_pct / 100), 4)
        risk_score     = _compute_risk_score(c, disruption_pct)
        status         = override.get("status") or _resolve_status(disruption_pct)
        last_incident  = override.get("last_incident", None)

        # EIA data would replace current_flow here when wired in
        trust_scores.append(1.0)  # baseline = authoritative

        corridors.append({
            "id":                  cid,
            "name":                c["name"],
            "region":              c["region"],
            "chokepoint":          c["chokepoint"],
            "baseline_flow_mbd":   c["baseline_flow_mbd"],
            "current_flow_mbd":    current_flow,
            "disruption_pct":      disruption_pct,
            "risk_score":          risk_score,
            "risk_factors":        c["risk_factors"],
            "status":              status,
            "last_incident":       last_incident,
            "alternative_routes":  c["alternative_routes"],
        })

    highest_risk = max(corridors, key=lambda x: x["risk_score"])
    total_disrupted = round(sum(
        c["baseline_flow_mbd"] - c["current_flow_mbd"] for c in corridors
    ), 4)

    trust_avg = round(sum(trust_scores) / len(trust_scores), 4)

    return {
        "tool":   "corridor_status",
        "status": "ok",
        "data": {
            "corridors":                corridors,
            "highest_risk_corridor":    highest_risk["id"],
            "total_disrupted_flow_mbd": total_disrupted,
        },
        "source_trust_avg":          trust_avg,
        "low_trust_sources_flagged": 0,
        "retrieved_at":              retrieved_at.isoformat(),
        "staleness_seconds":         0,
    }


def apply_incident(corridor_id: str, disruption_pct: float, last_incident: str) -> None:
    """Manual/ops hook to inject an incident override. NOT called by the
    automated pipeline (see _ACTIVE_INCIDENTS note)."""
    _ACTIVE_INCIDENTS[corridor_id] = {
        "disruption_pct": disruption_pct,
        "last_incident":  last_incident,
    }


def clear_incident(corridor_id: str) -> None:
    _ACTIVE_INCIDENTS.pop(corridor_id, None)
