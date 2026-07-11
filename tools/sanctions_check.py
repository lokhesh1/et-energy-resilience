"""
sanctions_check — offline OFAC SDN screening for procurement bids.

Matches a bid's supplier string against a SEEDED, cached SDN subset
(data/sdn_seed.json) — the Iran/Russia/Venezuela entities we actually source
from. This is deliberately NOT the live ~50MB treasury.gov SDN XML: parsing that
per call would be slow and fragile. A scheduled refresh job that downloads the
full list into this same cache is a deferred production TODO (see CLAUDE.md);
the runtime lookup here always stays offline and fast.

Matching is normalised token containment: lowercase, strip punctuation, and flag
if every token of an SDN name/alias appears in the supplier string. So
'NIOC (Iran)' matches alias 'NIOC'. This mirrors the reference matcher pinned in
tests/test_suppliers_data.py. Standard {status,data} envelope — never raises.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

_SDN_PATH = Path(__file__).parent.parent / "data" / "sdn_seed.json"

# Loaded lazily and cached — the seed never changes within a run.
_ENTITIES: list[dict] | None = None


def _envelope(status: str, data: dict) -> dict:
    return {
        "tool":                      "sanctions_check",
        "status":                    status,
        "data":                      data,
        "source_trust_avg":          1.0,
        "low_trust_sources_flagged": 0,
        "retrieved_at":              datetime.now(timezone.utc).isoformat(),
        "staleness_seconds":         0,
    }


def _normalise(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def _load() -> list[dict]:
    """Return the cached SDN entities. Raises on I/O so callers can fall back."""
    global _ENTITIES
    if _ENTITIES is None:
        with open(_SDN_PATH, encoding="utf-8") as f:
            _ENTITIES = json.load(f)["entities"]
    return _ENTITIES


def check_supplier(supplier_name: str) -> dict:
    """
    Screen a supplier string against the seeded SDN list.

    data → {supplier, sanctioned, matched_entity, matched_alias, programs}
    `matched_entity` is None when clean. A failed envelope means the SDN cache
    itself couldn't be read — callers should treat "cannot screen" conservatively
    (a bid that can't be cleared should not be auto-approved).
    """
    try:
        entities = _load()
    except Exception as e:
        return _envelope("failed", {"error": str(e), "supplier": supplier_name})

    tokens = _normalise(supplier_name)
    for ent in entities:
        names = [ent["name"], *ent.get("aliases", [])]
        for n in names:
            n_tokens = _normalise(n)
            if n_tokens and n_tokens <= tokens:
                return _envelope("ok", {
                    "supplier":       supplier_name,
                    "sanctioned":     True,
                    "matched_entity": ent["name"],
                    "matched_alias":  n,
                    "programs":       ent.get("programs", []),
                    "country":        ent.get("country"),
                })

    return _envelope("ok", {
        "supplier":       supplier_name,
        "sanctioned":     False,
        "matched_entity": None,
        "matched_alias":  None,
        "programs":       [],
    })
