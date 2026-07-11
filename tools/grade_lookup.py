"""
grade_lookup — refinery × crude-grade compatibility.

Pure, deterministic lookup over data/grade_matrix.json + data/refineries.json.
No threshold math at runtime: each grade already carries its explicit
light_sweet/medium_sour/heavy_sour `type`, and each refinery carries a
`crude_flexibility` level (high/medium/low). Compatibility is just
"is the grade's type in the set the refinery's flexibility accepts?".

Used by the procurement bidders/evaluator to answer "can the refineries hit by
this disruption actually run the crude this bid offers?". Standard {status,data}
envelope with an offline fallback — never raises, needs no network.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

_DATA = Path(__file__).parent.parent / "data"
_GRADE_MATRIX_PATH = _DATA / "grade_matrix.json"
_REFINERIES_PATH = _DATA / "refineries.json"

# Loaded lazily and cached so repeated bid checks don't re-read disk.
_GRADES: dict | None = None
_FLEX_RULES: dict | None = None
_REFINERIES: dict | None = None


def _envelope(status: str, data: dict) -> dict:
    return {
        "tool":                      "grade_lookup",
        "status":                    status,
        "data":                      data,
        "source_trust_avg":          1.0,
        "low_trust_sources_flagged": 0,
        "retrieved_at":              datetime.now(timezone.utc).isoformat(),
        "staleness_seconds":         0,
    }


def _load() -> None:
    """Populate the module caches. Raises on I/O so callers can fall back."""
    global _GRADES, _FLEX_RULES, _REFINERIES
    if _GRADES is not None:
        return
    with open(_GRADE_MATRIX_PATH, encoding="utf-8") as f:
        gm = json.load(f)
    with open(_REFINERIES_PATH, encoding="utf-8") as f:
        rj = json.load(f)
    _GRADES = gm["grades"]
    _FLEX_RULES = gm["flexibility_rules"]
    _REFINERIES = {r["id"]: r for r in rj["refineries"]}


def get_grade(grade_id: str) -> dict:
    """Return a single grade's properties, or a failed envelope if unknown."""
    try:
        _load()
    except Exception as e:
        return _envelope("failed", {"error": str(e), "grade_id": grade_id})

    grade = _GRADES.get(grade_id)
    if grade is None:
        return _envelope("failed", {
            "error":    f"unknown grade '{grade_id}'",
            "grade_id": grade_id,
        })
    return _envelope("ok", {"grade_id": grade_id, **grade})


def check_compatibility(grade_id: str, refinery_id: str) -> dict:
    """
    Can `refinery_id` process crude `grade_id`?

    data → {grade_id, refinery_id, grade_type, refinery_flexibility,
            accepted_types, compatible}
    A failed envelope means an unknown grade/refinery/flexibility level (a data
    contract violation) — the caller should treat that as "cannot confirm".
    """
    try:
        _load()
    except Exception as e:
        return _envelope("failed", {
            "error":       str(e),
            "grade_id":    grade_id,
            "refinery_id": refinery_id,
        })

    grade = _GRADES.get(grade_id)
    if grade is None:
        return _envelope("failed", {
            "error":       f"unknown grade '{grade_id}'",
            "grade_id":    grade_id,
            "refinery_id": refinery_id,
        })

    refinery = _REFINERIES.get(refinery_id)
    if refinery is None:
        return _envelope("failed", {
            "error":       f"unknown refinery '{refinery_id}'",
            "grade_id":    grade_id,
            "refinery_id": refinery_id,
        })

    flexibility = refinery.get("crude_flexibility")
    rule = _FLEX_RULES.get(flexibility)
    if rule is None:
        return _envelope("failed", {
            "error":       f"refinery '{refinery_id}' has unknown "
                           f"crude_flexibility '{flexibility}'",
            "grade_id":    grade_id,
            "refinery_id": refinery_id,
        })

    grade_type = grade["type"]
    accepted = rule["accepts"]
    return _envelope("ok", {
        "grade_id":             grade_id,
        "refinery_id":          refinery_id,
        "grade_type":           grade_type,
        "refinery_flexibility": flexibility,
        "accepted_types":       accepted,
        "compatible":           grade_type in accepted,
    })


def check_grade_against_refineries(grade_id: str, refinery_ids: list[str]) -> dict:
    """
    Check one grade against a set of (typically disruption-affected) refineries.

    data → {grade_id, grade_type, compatible_refineries, incompatible_refineries,
            unknown_refineries, any_compatible}
    Lets the evaluator/constitution answer "is this bid's crude usable by ANY
    affected refinery, or should it be warned as a grade mismatch?".
    """
    try:
        _load()
    except Exception as e:
        return _envelope("failed", {"error": str(e), "grade_id": grade_id})

    grade = _GRADES.get(grade_id)
    if grade is None:
        return _envelope("failed", {
            "error":    f"unknown grade '{grade_id}'",
            "grade_id": grade_id,
        })

    compatible, incompatible, unknown = [], [], []
    for rid in refinery_ids:
        res = check_compatibility(grade_id, rid)
        if res["status"] != "ok":
            unknown.append(rid)
        elif res["data"]["compatible"]:
            compatible.append(rid)
        else:
            incompatible.append(rid)

    return _envelope("ok", {
        "grade_id":                grade_id,
        "grade_type":              grade["type"],
        "compatible_refineries":   compatible,
        "incompatible_refineries": incompatible,
        "unknown_refineries":      unknown,
        "any_compatible":          bool(compatible),
    })
