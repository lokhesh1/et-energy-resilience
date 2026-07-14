"""
SPR Calculator — Strategic Petroleum Reserve drawdown modelling.

Answers "how long can India's strategic reserves bridge a supply gap?" Fully
deterministic and offline — a reserve drawdown is arithmetic over seeded cavern
parameters, nothing to fetch and nothing to hallucinate. Standard {status, data}
tool envelope, same discipline as corridor_status.

Two consumers:
  * bid_evaluator — `days_of_cover()` feeds the urgency dial: thin SPR cover makes
    every day the gap stays open more expensive, so faster cargo wins;
  * crisis_coordinator — `calculate_drawdown()` sizes the SPR bridge for whatever
    residual gap the market mix leaves uncovered.

Two numbers matter and they are different:
  * `days_of_cover` — usable reserve ÷ actual drawdown rate: how long the SPR can
    sustain the bridge it is physically able to pump;
  * `bridge_fraction` — drawdown ÷ gap: how much of the gap that bridge closes.
    A 4 mbd gap against a 1 mbd max drawdown is only a quarter-bridge no matter
    how many barrels sit in the caverns.

Parameters live in data/spr_params.json (ILLUSTRATIVE — only the ISPRL Phase I
~39 mmbbl anchor is real; see the file's _note). Loaded per call with a baked-in
fallback so a missing/broken file degrades loudly (`params_source: "defaults"`),
never crashes. Units: mmbbl for stock, mbd for rates → days = mmbbl / mbd.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

SPR_PARAMS_PATH = Path(__file__).parent.parent / "data" / "spr_params.json"

_DEFAULT_PARAMS = {
    "strategic_reserve_mmbbl": 39.0,
    "fill_fraction": 0.9,
    "max_drawdown_mbd": 1.0,
}

# A bridge_fraction this close to 1.0 counts as a full bridge (float rounding).
_FULL_BRIDGE_EPS = 1e-3


def _load_params() -> tuple[dict, bool]:
    """Return (params, loaded_from_file). Missing/broken file → documented
    defaults; the envelope records which was used (loud, not silent)."""
    try:
        with open(SPR_PARAMS_PATH, encoding="utf-8") as f:
            return json.load(f), True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _DEFAULT_PARAMS, False


def _usable_reserve_mmbbl(params: dict) -> float:
    return (float(params.get("strategic_reserve_mmbbl", 39.0))
            * float(params.get("fill_fraction", 0.9)))


def days_of_cover(gap_mbd: float) -> float | None:
    """How many days the usable strategic reserve could bridge `gap_mbd` if it
    alone had to cover the whole gap (ignores the drawdown-rate cap — this is the
    urgency signal, not the operational plan). None when there is no gap."""
    try:
        gap = float(gap_mbd)
    except (TypeError, ValueError):
        return None
    if gap <= 0:
        return None
    params, _ = _load_params()
    return round(_usable_reserve_mmbbl(params) / gap, 2)


def calculate_drawdown(gap_mbd: float, duration_days: float | None = None) -> dict:
    """Model an SPR drawdown against a supply gap.

    `gap_mbd` — the shortfall to bridge (typically the residual the market mix
    leaves uncovered). `duration_days` — optional expected disruption length
    (e.g. DSM's scenario duration); when given, `covers_duration` says whether
    the SPR bridge outlasts it."""
    retrieved_at = datetime.now(timezone.utc).isoformat()
    params, from_file = _load_params()

    usable = round(_usable_reserve_mmbbl(params), 4)
    max_drawdown = float(params.get("max_drawdown_mbd", 1.0))

    try:
        gap = max(0.0, float(gap_mbd))
    except (TypeError, ValueError):
        gap = 0.0

    if gap <= 0:
        data = {
            "gap_mbd":              0.0,
            "usable_reserve_mmbbl": usable,
            "max_drawdown_mbd":     max_drawdown,
            "drawdown_mbd":         0.0,
            "bridge_fraction":      None,
            "unbridged_mbd":        0.0,
            "days_of_cover":        None,
            "duration_days":        duration_days,
            "covers_duration":      None,
            "adequacy":             "not_needed",
            "params_source":        "file" if from_file else "defaults",
        }
    else:
        drawdown = round(min(gap, max_drawdown), 4)
        bridge_fraction = round(drawdown / gap, 4)
        cover = round(usable / drawdown, 2) if drawdown > 0 else None
        full_bridge = bridge_fraction >= 1.0 - _FULL_BRIDGE_EPS

        covers_duration = None
        if duration_days is not None and cover is not None:
            # A partial bridge never "covers" the disruption, however long the
            # barrels last — the unbridged flow is lost every single day.
            covers_duration = bool(full_bridge and cover >= float(duration_days))

        data = {
            "gap_mbd":              round(gap, 4),
            "usable_reserve_mmbbl": usable,
            "max_drawdown_mbd":     max_drawdown,
            "drawdown_mbd":         drawdown,
            "bridge_fraction":      bridge_fraction,
            "unbridged_mbd":        round(gap - drawdown, 4),
            "days_of_cover":        cover,
            "duration_days":        duration_days,
            "covers_duration":      covers_duration,
            "adequacy":             "full_bridge" if full_bridge else "partial_bridge",
            "params_source":        "file" if from_file else "defaults",
        }

    return {
        "tool":                      "spr_calculator",
        "status":                    "ok",
        "data":                      data,
        "source_trust_avg":          1.0,   # seeded params = our own data
        "low_trust_sources_flagged": 0,
        "retrieved_at":              retrieved_at,
        "staleness_seconds":         0,
    }
