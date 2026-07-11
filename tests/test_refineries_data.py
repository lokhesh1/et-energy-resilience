"""
Data-integrity tests for data/refineries.json.

This is the CI guard SCTD deliberately does NOT do at runtime: refineries.json is
static data WE author, so a bad edit is our own regression — and a test catches it
at `git push`, before it ever reaches a running twin. These assertions encode the
invariants SCTD's math relies on (esp. shares ≤ 1.0, which makes feed ≤ capacity
true by construction) plus the refinery↔corridor id contract.
"""
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).parent.parent / "data" / "refineries.json"

# Same closed set the rest of the board uses. A refinery may only depend on these.
KNOWN_CORRIDORS = {
    "strait_of_hormuz", "suez_canal", "malacca_strait", "bab_el_mandeb",
    "turkish_straits", "danish_straits", "cape_of_good_hope", "panama_canal",
}


@pytest.fixture(scope="module")
def data() -> dict:
    with open(_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def refineries(data) -> list[dict]:
    return data["refineries"]


def test_file_parses_and_is_nonempty(refineries):
    assert isinstance(refineries, list)
    assert len(refineries) >= 1


def test_illustrative_caveat_present(data):
    # The "_note" flags these as illustrative/unsourced. Losing it silently would
    # let someone mistake hand-seeded shares for real data.
    note = data.get("_note", "").lower()
    assert "illustrative" in note


def test_ids_unique(refineries):
    ids = [r["id"] for r in refineries]
    assert len(ids) == len(set(ids)), "duplicate refinery id"


def test_required_fields_present(refineries):
    required = {"id", "name", "lat", "lon", "capacity_mbd", "corridor_dependency"}
    for r in refineries:
        missing = required - r.keys()
        assert not missing, f"{r.get('id', '?')} missing fields: {missing}"


def test_capacities_positive_numbers(refineries):
    for r in refineries:
        cap = r["capacity_mbd"]
        assert isinstance(cap, (int, float)) and not isinstance(cap, bool)
        assert cap > 0, f"{r['id']}: capacity must be > 0, got {cap}"


def test_coordinates_numeric_and_in_range(refineries):
    for r in refineries:
        lat, lon = r["lat"], r["lon"]
        assert isinstance(lat, (int, float)) and isinstance(lon, (int, float))
        assert -90.0 <= lat <= 90.0, f"{r['id']}: lat out of range {lat}"
        assert -180.0 <= lon <= 180.0, f"{r['id']}: lon out of range {lon}"
        # sanity: all refineries are in/around India
        assert 5.0 <= lat <= 37.0 and 65.0 <= lon <= 100.0, \
            f"{r['id']}: coordinates {lat},{lon} not plausibly in India"


def test_dependency_shares_are_valid_fractions(refineries):
    for r in refineries:
        for cid, share in r["corridor_dependency"].items():
            assert isinstance(share, (int, float)) and not isinstance(share, bool)
            assert 0.0 <= share <= 1.0, f"{r['id']}/{cid}: share out of [0,1]: {share}"


def test_dependency_shares_sum_at_most_one(refineries):
    # THE load-bearing invariant: shares sum ≤ 1.0 so feed_at_risk (= capacity ×
    # Σ share×fraction, fraction ≤ 1) can never exceed capacity. This is what the
    # runtime SCTD-01 bound would have guarded — caught here at CI instead.
    for r in refineries:
        total = sum(r["corridor_dependency"].values())
        assert total <= 1.0 + 1e-9, \
            f"{r['id']}: corridor shares sum to {total:.4f} > 1.0 " \
            f"(would let feed_at_risk exceed capacity)"


def test_dependencies_reference_known_corridors(refineries):
    # The refinery↔corridor id contract. A typo here (or a corridor rename) would
    # make SCTD's lookup silently miss and understate risk — catch it at CI.
    for r in refineries:
        for cid in r["corridor_dependency"]:
            assert cid in KNOWN_CORRIDORS, \
                f"{r['id']}: unknown corridor id '{cid}' in corridor_dependency"
