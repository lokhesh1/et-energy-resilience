"""
DSM — Disruption Scenario Modeller.

Takes GRI's corridor risk scores + event types and turns each risky corridor into
a *quantified* disruption scenario: how much oil is at risk, for how long, India's
exposure, and the rerouting penalty. SCTD then maps these to specific refineries.

Design: HYBRID, but numbers are deterministic and the LLM is decoration only.
    - Volume / duration / reroute / india-exposure are computed from fixed tables
      and corridor baselines — same input always gives the same output, and every
      figure is traceable (volume = baseline x fraction). This is the DSM analogue
      of GRI's chain-of-evidence: no hallucinated numbers.
    - The LLM writes ONLY the qualitative cascade_narrative sentence. If it fails,
      every number in the scenario is unchanged — the narrative is simply empty.
      Guarantee: delete the LLM entirely and the numbers are identical.

Fire-and-forget on memory, like GRI — persistence never breaks the node.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from config.settings import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DSM_MODEL, DSM_MODEL_THRESHOLD,
)
from graph.eib_state import EnergyIntelligenceBoard, StigmergyMarker
from eib_guardrails.constitution_checker import check as constitution_check
from tools.corridor_status import get_corridor_status
from memory.xmemory import XMemory

KNOWN_CORRIDORS = {
    "strait_of_hormuz", "suez_canal", "malacca_strait", "bab_el_mandeb",
    "turkish_straits", "danish_straits", "cape_of_good_hope", "panama_canal",
}

_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# Shared long-term memory facade. Lazy inside — no cloud connection on import.
_xmemory = XMemory()

# Persist only high-severity scenarios so the episodic log stays signal.
_MEMORY_PERSIST_SEVERITIES = {"critical", "high"}

# ── Deterministic model parameters (with baked-in fallback if the file is gone) ──

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "dsm_params.json"

_DEFAULT_PARAMS = {
    "duration_days": {
        "war_conflict": 42, "sanctions": 90, "political_tension": 30,
        "infrastructure_failure": 21, "market_spike": 14, "piracy": 14,
        "weather_disruption": 7, "none": 14,
    },
    "india_import_share": {
        "strait_of_hormuz": 0.62, "bab_el_mandeb": 0.20, "suez_canal": 0.12,
        "malacca_strait": 0.05, "cape_of_good_hope": 0.10, "turkish_straits": 0.05,
        "danish_straits": 0.03, "panama_canal": 0.02,
    },
    "reroute_deltas": {},
}
_DEFAULT_DURATION = 14
_DEFAULT_INDIA_SHARE = 0.05
_DEFAULT_REROUTE = {"added_transit_days": 7, "freight_cost_mult": 1.2}


def _load_params() -> tuple[dict, bool]:
    """Return (params, loaded_from_file). Missing/broken file -> documented
    defaults; the audit records which was used (loud, not silent)."""
    try:
        with open(_PARAMS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data, True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _DEFAULT_PARAMS, False


_PARAMS, _PARAMS_FROM_FILE = _load_params()


# ── Deterministic scenario math ─────────────────────────────────────────────────

def _disruption_fraction(score: float, event_type: str) -> float:
    """Fraction of corridor flow lost.

    Physical events (war, weather, piracy, infrastructure) scale toward a full
    closure in bands. Legal/political events (sanctions, tension) are *targeted* —
    they choke a subset of flow rather than shutting the strait — so they use a
    dampened linear fraction. This is why a sanctions scenario yields a smaller
    volume than a war at the same risk score.
    """
    if event_type in ("sanctions", "political_tension"):
        return round(min(1.0, max(0.0, score) * 0.6), 3)
    if score >= 0.8:
        return 1.0
    if score >= 0.6:
        return 0.6
    if score >= 0.4:
        return 0.3
    return 0.1


def _severity(volume_mbd: float, duration_days: float) -> str:
    # weight volume by how long it persists (mbd carried for ~a month = 1 unit)
    weighted = volume_mbd * (duration_days / 30.0)
    if volume_mbd >= 10.0 or weighted >= 12.0:
        return "critical"
    if volume_mbd >= 4.0 or weighted >= 5.0:
        return "high"
    if volume_mbd >= 1.5:
        return "moderate"
    return "low"


def _build_scenario(corridor: dict, score: float, event_type: str) -> dict:
    cid = corridor["id"]
    baseline = float(corridor.get("baseline_flow_mbd", 0.0))

    fraction = _disruption_fraction(score, event_type)
    volume = round(baseline * fraction, 3)

    india_share = _PARAMS.get("india_import_share", {}).get(cid, _DEFAULT_INDIA_SHARE)
    india_exposure = round(volume * india_share, 3)

    duration = _PARAMS.get("duration_days", {}).get(event_type, _DEFAULT_DURATION)

    alt_routes = corridor.get("alternative_routes") or []
    reroute = None
    if alt_routes:
        delta = _PARAMS.get("reroute_deltas", {}).get(cid, _DEFAULT_REROUTE)
        reroute = {
            "alt_route": alt_routes[0],
            "added_transit_days": delta.get("added_transit_days", _DEFAULT_REROUTE["added_transit_days"]),
            "freight_cost_mult": delta.get("freight_cost_mult", _DEFAULT_REROUTE["freight_cost_mult"]),
        }

    return {
        "corridor":            cid,
        "corridor_name":       corridor.get("name", cid),
        "event_type":          event_type,
        "risk_score":          round(float(score), 4),
        "baseline_flow_mbd":   round(baseline, 3),
        "disruption_fraction": fraction,
        "volume_at_risk_mbd":  volume,
        "india_import_share":  india_share,   # stored so the constitution can recompute exposure
        "india_exposure_mbd":  india_exposure,
        "duration_days":       duration,
        "reroute":             reroute,
        "severity":            _severity(volume, duration),
        "quarantined":         False,  # set True if a block-severity rule rejects it
        "cascade_narrative":   "",   # filled by the LLM decoration layer (best-effort)
    }


def _deposit_demand_pheromones(scenarios: list[dict]) -> list[StigmergyMarker]:
    """DSM signals the procurement pod indirectly: a 'demand' marker per scenario,
    intensity scaled by India's exposed volume, so sourcing agents amplify against
    the real gap without any direct call."""
    now = datetime.now(timezone.utc).isoformat()
    markers: list[StigmergyMarker] = []
    for sc in scenarios:
        if sc["severity"] not in ("critical", "high", "moderate"):
            continue
        intensity = round(min(1.0, sc["india_exposure_mbd"] / 5.0), 4)
        markers.append({
            "type":         "demand",
            "target":       sc["corridor"],
            "intensity":    intensity,
            "deposited_by": "dsm_agent",
            "timestamp":    now,
            "decay_rate":   0.1,
        })
    return markers


# ── LLM decoration: narrative only, never numbers ───────────────────────────────

_SYSTEM_PROMPT = """You are a Disruption Scenario Modeller for Indian energy supply chains.
You are given already-computed disruption numbers. Do NOT change, recompute, or
question any number. Write one concise sentence per corridor describing the
downstream cascade (refineries, freight, timeline) implied by those numbers.
Respond with valid JSON only: {"<corridor_id>": "<one sentence>"}."""


def _add_narratives(scenarios: list[dict]) -> None:
    """Best-effort in-place enrichment. On any failure, narratives stay empty and
    every number is untouched."""
    if not scenarios:
        return
    summary = "\n".join(
        f"{s['corridor']}: {s['event_type']}, {s['volume_at_risk_mbd']} mbd at risk "
        f"({s['india_exposure_mbd']} mbd India-bound), {s['duration_days']} days, "
        f"severity={s['severity']}"
        for s in scenarios
    )
    try:
        response = _client.chat.completions.create(
            model=DSM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"SCENARIOS:\n{summary}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        narratives = json.loads(response.choices[0].message.content)
        if isinstance(narratives, dict):
            for s in scenarios:
                text = narratives.get(s["corridor"])
                if isinstance(text, str):
                    s["cascade_narrative"] = text.strip()
    except Exception:
        pass  # narrative is decoration; numbers already stand on their own


# ── Node ────────────────────────────────────────────────────────────────────────

def dsm_node(state: EnergyIntelligenceBoard) -> dict:
    now = datetime.now(timezone.utc).isoformat()

    corridor_risk   = state.get("corridor_risk", {}) or {}
    corridor_events = state.get("corridor_events", {}) or {}
    pheromone_field = state.get("pheromone_field", {}) or {}

    # ── 1. Baselines (reuse the corridor tool GRI uses) ────────────────────
    corridor_result = get_corridor_status()
    corridors_by_id = {c["id"]: c for c in corridor_result["data"]["corridors"]}

    # ── 2. Constitution check on inputs ────────────────────────────────────
    audit: list[dict] = [{
        "agent":            "dsm_agent",
        "action":           "input_fetch",
        "corridor_status":  corridor_result["status"],
        "corridors_scored": len(corridor_risk),
        "params_source":    "file" if _PARAMS_FROM_FILE else "defaults",
        "timestamp":        now,
    }]

    # ── 3. Deterministic scenario modelling ────────────────────────────────
    scenarios: list[dict] = []
    for cid, score in corridor_risk.items():
        if cid not in KNOWN_CORRIDORS or cid not in corridors_by_id:
            continue
        # stigmergy amplification: a strong pheromone raises the modelled score
        effective = max(float(score), float(pheromone_field.get(cid, 0.0)))
        if effective < DSM_MODEL_THRESHOLD:
            continue
        event_type = corridor_events.get(cid, "none")
        scenarios.append(_build_scenario(corridors_by_id[cid], effective, event_type))

    scenarios.sort(key=lambda s: s["volume_at_risk_mbd"], reverse=True)

    # ── 4. Constitution check + quarantine block-flagged scenarios ─────────
    # A block-severity violation means the math is broken. Such a scenario must
    # NOT be narrated (a fluent sentence would lend it false credibility), nor
    # signal procurement, nor enter memory. It stays in the output flagged, so
    # the failure is visible — never silently dropped.
    check_result = constitution_check("dsm", {"scenarios": scenarios})
    blocked_corridors = {
        v["corridor"] for v in check_result.get("violations", [])
        if v.get("severity") == "block" and v.get("corridor")
    }
    for sc in scenarios:
        if sc["corridor"] in blocked_corridors:
            sc["quarantined"] = True

    valid_scenarios = [s for s in scenarios if not s["quarantined"]]

    audit.append({
        "agent":             "dsm_agent",
        "action":            "scenario_model",
        "scenario_count":    len(scenarios),
        "quarantined_count": len(scenarios) - len(valid_scenarios),
        "constitution_check": check_result,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    })

    # ── 5. LLM narrative — ONLY on scenarios that passed (numbers already final) ─
    _add_narratives(valid_scenarios)

    # ── 6. Deposit demand pheromones for the procurement pod (valid only) ──
    markers = _deposit_demand_pheromones(valid_scenarios)

    # ── 7. Persist notable scenarios to long-term memory (best-effort) ─────
    try:
        for sc in valid_scenarios:
            if sc["severity"] not in _MEMORY_PERSIST_SEVERITIES:
                continue
            _xmemory.remember(
                event_type="disruption_scenario",
                agent="dsm_agent",
                payload={
                    "corridor":           sc["corridor"],
                    "event_type":         sc["event_type"],
                    "score":              sc["risk_score"],
                    "volume_at_risk_mbd": sc["volume_at_risk_mbd"],
                    "india_exposure_mbd": sc["india_exposure_mbd"],
                    "duration_days":      sc["duration_days"],
                    "severity":           sc["severity"],
                },
                outcome="success",
                text=(
                    f"{sc['corridor_name']} {sc['event_type']}: "
                    f"{sc['volume_at_risk_mbd']} mbd at risk for {sc['duration_days']} days "
                    f"({sc['severity']})"
                ),
            )
    except Exception:
        pass  # memory is best-effort; never break the node

    return {
        "current_agent":     "dsm_agent",
        "scenarios":         scenarios,
        "stigmergy_markers": markers,
        "audit_trail":       audit,
        "constitution_flags": check_result.get("violations", []),
    }
