"""
Data-integrity tests for the procurement data layer:
  data/grade_matrix.json · data/suppliers.json · data/sdn_seed.json

Same CI-guard philosophy as tests/test_refineries_data.py — these are static files
WE author, so a bad edit is our own regression, caught at `git push` before it ever
reaches a running bidder/evaluator. They encode the contracts the (not-yet-built)
procurement pod relies on:
  - every supplier.grade exists in the grade matrix         (grade_lookup contract)
  - every supplier.delivery_corridor is a known corridor    (corridor id contract)
  - every supplier.region is one of the 3 sourcing regions  (bidder routing contract)
  - the sanctioned traps actually match the SDN seed, and clean suppliers do NOT
    (sanctions_check is neither dead nor over-firing)
"""
import json
import re
from pathlib import Path

import pytest

_DATA = Path(__file__).parent.parent / "data"

# Same closed set the rest of the board uses.
KNOWN_CORRIDORS = {
    "strait_of_hormuz", "suez_canal", "malacca_strait", "bab_el_mandeb",
    "turkish_straits", "danish_straits", "cape_of_good_hope", "panama_canal",
}

# The 3 sourcing regions a bidder can claim.
KNOWN_REGIONS = {"west_africa", "americas", "spot"}

# The crude-type taxonomy grade_matrix + flexibility_rules are keyed on.
KNOWN_TYPES = {"light_sweet", "medium_sour", "heavy_sour"}

# refineries.json uses these three flexibility levels; flexibility_rules must cover them.
KNOWN_FLEXIBILITY = {"high", "medium", "low"}


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def grade_matrix() -> dict:
    with open(_DATA / "grade_matrix.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def suppliers_doc() -> dict:
    with open(_DATA / "suppliers.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def suppliers(suppliers_doc) -> list[dict]:
    return suppliers_doc["suppliers"]


@pytest.fixture(scope="module")
def sdn() -> dict:
    with open(_DATA / "sdn_seed.json", encoding="utf-8") as f:
        return json.load(f)


# ── Shared matcher (mirror of what tools/sanctions_check.py will implement) ───────

def _normalise(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def _is_sanctioned(supplier_name: str, sdn_entities: list[dict]) -> bool:
    tokens = _normalise(supplier_name)
    for ent in sdn_entities:
        names = [ent["name"], *ent.get("aliases", [])]
        for n in names:
            n_tokens = _normalise(n)
            # every token of the SDN name/alias must appear in the supplier string
            if n_tokens and n_tokens <= tokens:
                return True
    return False


# ── grade_matrix.json ─────────────────────────────────────────────────────────

def test_grade_matrix_illustrative_caveat(grade_matrix):
    assert "illustrative" in grade_matrix.get("_note", "").lower()


def test_grades_have_valid_types(grade_matrix):
    for gid, g in grade_matrix["grades"].items():
        assert g["type"] in KNOWN_TYPES, f"{gid}: unknown type {g['type']}"
        assert isinstance(g["api_gravity"], (int, float))
        assert isinstance(g["sulfur_pct"], (int, float))
        assert g["api_gravity"] > 0 and g["sulfur_pct"] >= 0


def test_flexibility_rules_cover_all_levels(grade_matrix):
    rules = grade_matrix["flexibility_rules"]
    assert KNOWN_FLEXIBILITY <= rules.keys(), \
        f"flexibility_rules missing levels: {KNOWN_FLEXIBILITY - rules.keys()}"
    for level, rule in rules.items():
        for t in rule["accepts"]:
            assert t in KNOWN_TYPES, f"{level}: accepts unknown type {t}"


# ── suppliers.json ──────────────────────────────────────────────────────────────

def test_suppliers_illustrative_caveat(suppliers_doc):
    assert "illustrative" in suppliers_doc.get("_note", "").lower()


def test_supplier_ids_unique(suppliers):
    ids = [s["id"] for s in suppliers]
    assert len(ids) == len(set(ids)), "duplicate supplier id"


def test_supplier_required_fields(suppliers):
    required = {
        "id", "supplier", "region", "grade", "price_premium_usd",
        "max_volume_mbd", "transit_days_to_india", "delivery_corridor",
    }
    for s in suppliers:
        missing = required - s.keys()
        assert not missing, f"{s.get('id', '?')} missing fields: {missing}"


def test_supplier_regions_known(suppliers):
    for s in suppliers:
        assert s["region"] in KNOWN_REGIONS, f"{s['id']}: unknown region {s['region']}"


def test_supplier_grades_exist_in_matrix(suppliers, grade_matrix):
    grades = grade_matrix["grades"]
    for s in suppliers:
        assert s["grade"] in grades, \
            f"{s['id']}: grade '{s['grade']}' not in grade_matrix"


def test_supplier_delivery_corridors_known(suppliers):
    for s in suppliers:
        assert s["delivery_corridor"] in KNOWN_CORRIDORS, \
            f"{s['id']}: unknown delivery_corridor '{s['delivery_corridor']}'"


def test_supplier_volumes_and_transit_positive(suppliers):
    for s in suppliers:
        assert isinstance(s["max_volume_mbd"], (int, float)) and not isinstance(s["max_volume_mbd"], bool)
        assert s["max_volume_mbd"] > 0, f"{s['id']}: max_volume_mbd must be > 0"
        assert s["transit_days_to_india"] > 0, f"{s['id']}: transit must be > 0"
        assert isinstance(s["price_premium_usd"], (int, float)) and not isinstance(s["price_premium_usd"], bool)


def test_each_region_has_suppliers(suppliers):
    regions = {s["region"] for s in suppliers}
    assert KNOWN_REGIONS <= regions, f"missing suppliers for regions: {KNOWN_REGIONS - regions}"


# ── sdn_seed.json + the sanctions gate contract ──────────────────────────────────

def test_sdn_seed_caveat_and_shape(sdn):
    note = sdn.get("_note", "").lower()
    assert "seeded" in note or "sdn" in note
    assert isinstance(sdn["entities"], list) and len(sdn["entities"]) >= 1
    for ent in sdn["entities"]:
        assert ent.get("name")
        assert isinstance(ent.get("programs", []), list)


def test_every_sdn_entity_matches_a_supplier(sdn, suppliers):
    # The traps are actually wired: each seeded entity must catch >= 1 supplier,
    # otherwise the SDN seed has dead rows and the gate proves nothing.
    for ent in sdn["entities"]:
        matched = [s["id"] for s in suppliers if _is_sanctioned(s["supplier"], [ent])]
        assert matched, f"SDN entity '{ent['name']}' matches no supplier (dead seed row)"


def test_expected_sanctioned_suppliers_flagged(sdn, suppliers):
    # The three deliberately-planted sanctioned suppliers must be caught.
    flagged = {s["id"] for s in suppliers if _is_sanctioned(s["supplier"], sdn["entities"])}
    expected = {"nioc_iranian_heavy", "rosneft_urals", "pdvsa_merey"}
    assert expected <= flagged, f"expected sanctioned suppliers not flagged: {expected - flagged}"


def test_clean_suppliers_not_flagged(sdn, suppliers):
    # The gate must not over-fire: every non-planted supplier is clean.
    sanctioned_ids = {"nioc_iranian_heavy", "rosneft_urals", "pdvsa_merey"}
    for s in suppliers:
        if s["id"] in sanctioned_ids:
            continue
        assert not _is_sanctioned(s["supplier"], sdn["entities"]), \
            f"{s['id']} ('{s['supplier']}') wrongly flagged as sanctioned"
