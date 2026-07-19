"""
Route Ranker — voyage-level reroute options for a disrupted corridor.

A reroute belongs to a VOYAGE (loading zone → discharge port), not to a corridor:
"Hormuz blocked → sail around the Cape" is geographically impossible (Hormuz is
the only exit from the Persian Gulf), while "Suez blocked → round Africa" is
real. This tool answers, per disrupted corridor: what actually happens to cargo
that normally transits it — ranked feasible alternates (added days / freight
multiplier), or the honest `no_maritime_alternative` with the physical bypass
(pipelines) and its capacity, plus the fallback advice (re-source / SPR).

Deterministic, offline, standard `{status, data}` envelope, never raises.
Seed data: `data/routes.json` (illustrative — same status as dsm_params.json)
with a baked-in fallback so a missing file degrades loudly, not fatally.
"""
import copy
import json
from pathlib import Path

_ROUTES_PATH = Path(__file__).parent.parent / "data" / "routes.json"

# An alternate whose own modeled corridor is effectively CLOSED is not an
# option. Mirrors the bid_evaluator's closed-corridor band (debugger.md #14).
_CLOSED_FRACTION = 0.75

# Minimal baked-in fallback: the two lanes whose truth matters most — the
# Hormuz dead-end and the classic Suez/Bab → Cape diversion.
_FALLBACK_LANES = {
    "strait_of_hormuz": {
        "origin_zone": "persian_gulf",
        "no_maritime_alternative": True,
        "bypass": {"kind": "pipeline",
                   "description": "Saudi East-West + ADNOC Fujairah pipelines "
                                  "(loads outside the strait)",
                   "capacity_mbd": 6.5, "added_days": 4,
                   "freight_cost_mult": 1.2},
        "fallback_advice": "re-source from alternate origins or bridge from SPR",
        "alternatives": [],
    },
    "suez_canal": {
        "origin_zone": "atlantic_mediterranean",
        "alternatives": [{"alt_route": "cape_of_good_hope",
                          "modeled_corridor": "cape_of_good_hope",
                          "added_days": 14, "freight_cost_mult": 1.35}],
    },
    "bab_el_mandeb": {
        "origin_zone": "atlantic_mediterranean",
        "alternatives": [{"alt_route": "cape_of_good_hope",
                          "modeled_corridor": "cape_of_good_hope",
                          "added_days": 14, "freight_cost_mult": 1.35}],
    },
}


def _load(path: Path = _ROUTES_PATH) -> tuple[dict, bool]:
    """Return (corridor_lanes, loaded_from_file). Missing/broken file → the
    baked-in fallback; the envelope records which was used (loud, not silent)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        lanes = data.get("corridor_lanes")
        if isinstance(lanes, dict) and lanes:
            return lanes, True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return _FALLBACK_LANES, False


_LANES, _LANES_FROM_FILE = _load()


def rank_routes(corridor_id: str,
                disruption_fractions: dict[str, float] | None = None) -> dict:
    """Feasible reroute options for cargo that normally transits `corridor_id`.

    `disruption_fractions` (corridor_id → fraction, e.g. the twin's view) prunes
    alternates that are themselves effectively closed (≥ 0.75) — a detour into a
    blockade is not an option. Below that band the alternate stays, degraded.
    """
    fractions = disruption_fractions or {}
    lane = _LANES.get(corridor_id)
    if lane is None:
        return {
            "tool": "route_ranker", "status": "ok",
            "data": {"corridor": corridor_id, "known": False,
                     "no_maritime_alternative": False, "bypass": None,
                     "fallback_advice": None, "options": [], "excluded": [],
                     "params_source": "file" if _LANES_FROM_FILE else "fallback"},
        }

    options: list[dict] = []
    excluded: list[dict] = []
    for alt in lane.get("alternatives", []) or []:
        alt = copy.deepcopy(alt)
        via = alt.get("modeled_corridor")
        frac = float(fractions.get(via, 0.0) or 0.0) if via else 0.0
        if via and frac >= _CLOSED_FRACTION:
            excluded.append({**alt, "excluded_reason":
                             f"{via} effectively closed ({frac:.2f} disrupted)"})
            continue
        if via and frac > 0.0:
            alt["via_disruption_fraction"] = round(frac, 3)
        options.append(alt)
    options.sort(key=lambda a: (float(a.get("added_days", 0) or 0),
                                float(a.get("freight_cost_mult", 1.0) or 1.0)))

    return {
        "tool": "route_ranker", "status": "ok",
        "data": {
            "corridor":                corridor_id,
            "known":                   True,
            "origin_zone":             lane.get("origin_zone"),
            "no_maritime_alternative": bool(lane.get("no_maritime_alternative"))
                                       or not (options or lane.get("alternatives")),
            "bypass":                  copy.deepcopy(lane.get("bypass")),
            "fallback_advice":         lane.get("fallback_advice"),
            "options":                 options,
            "excluded":                excluded,
            "params_source":           "file" if _LANES_FROM_FILE else "fallback",
        },
    }
