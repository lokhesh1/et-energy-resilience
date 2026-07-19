"""
Unit tests for tools/route_ranker.py — voyage-level reroute options.
Pure/offline: seed file + baked-in fallback, no network.
"""
from pathlib import Path

from tools.route_ranker import rank_routes, _load, _FALLBACK_LANES


def test_envelope_shape():
    r = rank_routes("suez_canal")
    assert r["tool"] == "route_ranker"
    assert r["status"] == "ok"
    assert "options" in r["data"] and "no_maritime_alternative" in r["data"]


def test_hormuz_is_a_dead_end_with_pipeline_bypass():
    d = rank_routes("strait_of_hormuz")["data"]
    assert d["no_maritime_alternative"] is True
    assert d["options"] == []                       # no sea detour exists
    assert d["bypass"]["capacity_mbd"] > 0          # pipelines skip the strait
    assert "re-source" in d["fallback_advice"]


def test_suez_and_bab_divert_around_the_cape():
    for cid in ("suez_canal", "bab_el_mandeb"):
        d = rank_routes(cid)["data"]
        assert d["no_maritime_alternative"] is False
        assert any(o.get("modeled_corridor") == "cape_of_good_hope"
                   for o in d["options"])


def test_closed_alternate_is_excluded():
    # a detour into a blockade is not an option (mirrors the >= 0.75 band)
    d = rank_routes("suez_canal", {"cape_of_good_hope": 0.8})["data"]
    assert not any(o.get("modeled_corridor") == "cape_of_good_hope"
                   for o in d["options"])
    assert any("cape_of_good_hope" in e["excluded_reason"] for e in d["excluded"])


def test_degraded_alternate_stays_with_disclosure():
    d = rank_routes("suez_canal", {"cape_of_good_hope": 0.5})["data"]
    cape = next(o for o in d["options"]
                if o.get("modeled_corridor") == "cape_of_good_hope")
    assert cape["via_disruption_fraction"] == 0.5


def test_options_ranked_by_added_days():
    d = rank_routes("malacca_strait")["data"]
    days = [o.get("added_days", 0) for o in d["options"]]
    assert days == sorted(days)


def test_unknown_corridor_is_honest_not_fabricated():
    d = rank_routes("atlantis_channel")["data"]
    assert d["known"] is False
    assert d["options"] == []
    assert d["no_maritime_alternative"] is False


def test_missing_file_falls_back_and_says_so():
    lanes, from_file = _load(Path("does/not/exist.json"))
    assert from_file is False
    assert lanes is _FALLBACK_LANES
    assert lanes["strait_of_hormuz"]["no_maritime_alternative"] is True


def test_never_raises_on_junk_input():
    assert rank_routes("")["status"] == "ok"
    assert rank_routes("suez_canal", {"cape_of_good_hope": None})["status"] == "ok"
