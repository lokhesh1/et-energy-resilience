"""
Tests for the voyage simulation map builder (ui/sim_map.py) and
data/sea_routes.json integrity.

All offline — no network, no Streamlit. Tests the pure-function data
assembly layer, not the HTML rendering.
"""
import json
from pathlib import Path

import pytest

from ui.sim_map import (
    build_voyages,
    build_sim_map_html,
    _build_voyage,
    _build_reroute_voyage,
    _sea_routes,
    _suppliers_by_id,
    _corridors_map,
    _baseline_voyages,
)

_DATA = Path(__file__).resolve().parent.parent / "data"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sea_routes():
    return _sea_routes()


@pytest.fixture
def suppliers():
    return _suppliers_by_id()


@pytest.fixture
def corridors():
    return _corridors_map()


@pytest.fixture
def sample_mix_row():
    return {
        "supplier": "NNPC (Nigeria)",
        "supplier_id": "nnpc_bonny_light",
        "grade": "bonny_light",
        "volume_mbd": 0.5,
        "effective_volume_mbd": 0.5,
        "price_per_bbl": 82.50,
        "delivery_corridor": "cape_of_good_hope",
        "transit_days": 20,
        "delivery_risk_fraction": 0,
        "trade_terms": "FOB",
        "sanctions_status": "clear",
    }


@pytest.fixture
def risky_mix_row():
    return {
        "supplier": "Spot cargo (Ras Tanura)",
        "supplier_id": "spot_arab_heavy",
        "grade": "arab_heavy",
        "volume_mbd": 0.3,
        "effective_volume_mbd": 0.15,
        "price_per_bbl": 78.00,
        "delivery_corridor": "strait_of_hormuz",
        "transit_days": 7,
        "delivery_risk_fraction": 0.5,
        "trade_terms": "FOB",
    }


@pytest.fixture
def sample_route():
    return {
        "from_corridor": "bab_el_mandeb",
        "to_corridor": "cape_of_good_hope",
        "volume_mbd": 1.2,
        "added_transit_days": 14,
        "freight_cost_mult": 1.35,
        "overloaded": False,
    }


@pytest.fixture
def sample_twin_state():
    return {
        "corridor_risks": [
            {"corridor": "strait_of_hormuz", "risk_score": 0.9},
            {"corridor": "bab_el_mandeb", "risk_score": 0.7},
            {"corridor": "cape_of_good_hope", "risk_score": 0.1},
        ],
        "impacts": [
            {"id": "jamnagar_ril", "name": "Jamnagar (RIL)", "lat": 22.35,
             "lon": 69.83, "status": "critical", "capacity_mbd": 1.36,
             "feed_at_risk_mbd": 0.8},
            {"id": "kochi_bpcl", "name": "Kochi (BPCL)", "lat": 9.96,
             "lon": 76.27, "status": "normal", "capacity_mbd": 0.31,
             "feed_at_risk_mbd": 0},
        ],
        "routes": [
            {"from_corridor": "bab_el_mandeb", "to_corridor": "cape_of_good_hope",
             "volume_mbd": 1.2, "added_transit_days": 14, "overloaded": False},
        ],
    }


# ── data/sea_routes.json integrity ───────────────────────────────────────────

class TestSeaRoutesIntegrity:
    def test_file_loads(self, sea_routes):
        assert sea_routes, "sea_routes.json failed to load or is empty"
        assert "ports" in sea_routes
        assert "lanes" in sea_routes

    def test_every_supplier_load_port_has_coords(self, sea_routes):
        """Every load_port in suppliers.json must have an entry in ports."""
        with open(_DATA / "suppliers.json", encoding="utf-8") as f:
            data = json.load(f)
        ports = sea_routes["ports"]
        for s in data.get("suppliers", []):
            lp = s.get("load_port")
            if lp:
                assert lp in ports, f"load_port '{lp}' (supplier {s['id']}) missing from sea_routes.ports"

    def test_every_delivery_corridor_has_a_lane(self, sea_routes):
        """Every delivery_corridor in suppliers.json must have a lane."""
        with open(_DATA / "suppliers.json", encoding="utf-8") as f:
            data = json.load(f)
        lanes = sea_routes["lanes"]
        corridors_used = {s["delivery_corridor"] for s in data.get("suppliers", [])
                          if s.get("delivery_corridor")}
        for c in corridors_used:
            assert c in lanes, f"delivery_corridor '{c}' has no lane in sea_routes.json"

    def test_all_eight_corridors_have_lanes(self, sea_routes):
        lanes = sea_routes["lanes"]
        expected = {"strait_of_hormuz", "suez_canal", "bab_el_mandeb",
                    "cape_of_good_hope", "malacca_strait", "turkish_straits",
                    "danish_straits", "panama_canal"}
        for c in expected:
            assert c in lanes, f"corridor '{c}' missing from lanes"

    def test_waypoints_are_valid_lat_lon(self, sea_routes):
        for name, lane in sea_routes["lanes"].items():
            if name.startswith("_"):
                continue
            assert isinstance(lane, list), f"lane '{name}' is not a list"
            assert len(lane) >= 2, f"lane '{name}' has fewer than 2 waypoints"
            for i, wp in enumerate(lane):
                assert len(wp) == 2, f"lane '{name}' waypoint {i} doesn't have 2 coords"
                lat, lon = wp
                assert -90 <= lat <= 90, f"lane '{name}' waypoint {i} lat {lat} out of range"
                assert -180 <= lon <= 180, f"lane '{name}' waypoint {i} lon {lon} out of range"

    def test_port_coords_valid(self, sea_routes):
        for name, port in sea_routes["ports"].items():
            if not isinstance(port, dict):
                continue
            if name.startswith("_"):
                if "lat" in port:
                    assert -90 <= port["lat"] <= 90
                continue
            assert "lat" in port and "lon" in port, f"port '{name}' missing lat/lon"
            assert -90 <= port["lat"] <= 90, f"port '{name}' lat out of range"
            assert -180 <= port["lon"] <= 180, f"port '{name}' lon out of range"


# ── Voyage building ──────────────────────────────────────────────────────────

class TestBuildVoyage:
    def test_voyage_from_mix_row(self, sample_mix_row, sea_routes, suppliers):
        v = _build_voyage(sample_mix_row, sea_routes, suppliers)
        assert v is not None
        assert v["type"] == "cargo"
        assert v["supplier"] == "NNPC (Nigeria)"
        assert v["grade"] == "bonny_light"
        assert v["volume_mbd"] == 0.5
        assert v["barrels_per_day"] == 500_000
        assert v["transit_days"] == 20
        assert v["status"] == "clear"
        assert len(v["path"]) >= 3

    def test_voyage_path_starts_at_load_port(self, sample_mix_row, sea_routes, suppliers):
        v = _build_voyage(sample_mix_row, sea_routes, suppliers)
        ports = sea_routes["ports"]
        bonny = ports["Bonny"]
        assert v["path"][0] == [bonny["lat"], bonny["lon"]]

    def test_voyage_path_ends_in_india(self, sample_mix_row, sea_routes, suppliers):
        v = _build_voyage(sample_mix_row, sea_routes, suppliers)
        end = v["path"][-1]
        assert 5 <= end[0] <= 30, "discharge latitude should be in India range"
        assert 65 <= end[1] <= 90, "discharge longitude should be in India range"

    def test_risky_voyage_status(self, risky_mix_row, sea_routes, suppliers):
        v = _build_voyage(risky_mix_row, sea_routes, suppliers)
        assert v is not None
        assert v["status"] == "risky"
        assert v["delivery_risk_fraction"] == 0.5

    def test_unknown_corridor_returns_none(self, sea_routes, suppliers):
        row = {"delivery_corridor": "nonexistent_corridor"}
        v = _build_voyage(row, sea_routes, suppliers)
        assert v is None

    def test_barrels_per_day_conversion(self, sample_mix_row, sea_routes, suppliers):
        sample_mix_row["volume_mbd"] = 1.0
        v = _build_voyage(sample_mix_row, sea_routes, suppliers)
        assert v["barrels_per_day"] == 1_000_000


# ── Reroute voyage ───────────────────────────────────────────────────────────

class TestBuildRerouteVoyage:
    def test_reroute_voyage(self, sample_route, sea_routes, corridors):
        rv = _build_reroute_voyage(sample_route, sea_routes, corridors)
        assert rv is not None
        assert rv["type"] == "reroute"
        assert rv["from_corridor"] == "bab_el_mandeb"
        assert rv["to_corridor"] == "cape_of_good_hope"
        assert rv["volume_mbd"] == 1.2
        assert rv["added_transit_days"] == 14
        assert rv["overloaded"] is False
        assert len(rv["path"]) >= 2

    def test_reroute_has_blocked_at(self, sample_route, sea_routes, corridors):
        rv = _build_reroute_voyage(sample_route, sea_routes, corridors)
        assert rv["blocked_at"] is not None
        assert rv["blocked_at"][0] == corridors["bab_el_mandeb"]["lat"]

    def test_reroute_has_original_path(self, sample_route, sea_routes, corridors):
        rv = _build_reroute_voyage(sample_route, sea_routes, corridors)
        assert rv["original_path"] is not None
        assert len(rv["original_path"]) >= 2

    def test_overloaded_flag(self, sample_route, sea_routes, corridors):
        sample_route["overloaded"] = True
        rv = _build_reroute_voyage(sample_route, sea_routes, corridors)
        assert rv["overloaded"] is True

    def test_unknown_alt_corridor_returns_none(self, sea_routes, corridors):
        route = {"from_corridor": "bab_el_mandeb", "to_corridor": "nonexistent"}
        rv = _build_reroute_voyage(route, sea_routes, corridors)
        assert rv is None


# ── Full build_voyages ───────────────────────────────────────────────────────

class TestBuildVoyages:
    def test_with_mix_rows(self, sample_mix_row, sample_twin_state):
        result = build_voyages([sample_mix_row], sample_twin_state, {})
        assert result["is_baseline"] is False
        assert len(result["voyages"]) >= 1
        assert result["voyages"][0]["type"] == "cargo"

    def test_with_reroutes(self, sample_twin_state):
        result = build_voyages([], sample_twin_state, {})
        assert len(result["reroutes"]) >= 1

    def test_empty_inputs_gives_baseline(self):
        result = build_voyages(None, None, None)
        assert result["is_baseline"] is True
        assert len(result["voyages"]) >= 1
        assert all(v["type"] == "baseline" for v in result["voyages"])

    def test_corridor_pins_from_twin(self, sample_twin_state):
        result = build_voyages([], sample_twin_state, {})
        disrupted = [p for p in result["corridor_pins"] if p["status"] == "disrupted"]
        assert len(disrupted) >= 2

    def test_refinery_pins_from_twin(self, sample_twin_state):
        result = build_voyages([], sample_twin_state, {})
        assert len(result["refinery_pins"]) == 2
        crit = [r for r in result["refinery_pins"] if r["status"] == "critical"]
        assert len(crit) == 1


# ── HTML output ──────────────────────────────────────────────────────────────

class TestBuildSimMapHtml:
    def test_returns_html_string(self):
        html = build_sim_map_html()
        assert isinstance(html, str)
        assert "<html>" in html or "<!DOCTYPE html>" in html

    def test_html_contains_leaflet(self):
        html = build_sim_map_html()
        assert "leaflet" in html.lower()

    def test_html_embeds_voyage_data(self, sample_mix_row, sample_twin_state):
        html = build_sim_map_html(mix_rows=[sample_mix_row],
                                   twin_state=sample_twin_state)
        assert "NNPC" in html
        assert "bonny_light" in html

    def test_empty_inputs_produce_baseline_html(self):
        html = build_sim_map_html()
        assert "baseline" in html.lower()

    def test_html_has_controls(self):
        html = build_sim_map_html()
        assert "Play" in html
        assert "Pause" in html

    def test_html_has_legend(self):
        html = build_sim_map_html()
        assert "Voyage Simulation" in html
        assert "Safe route" in html

    def test_html_never_raises_on_none(self):
        html = build_sim_map_html(None, None, None)
        assert isinstance(html, str)
        assert len(html) > 100


# ── Baseline voyages ─────────────────────────────────────────────────────────

class TestBaselineVoyages:
    def test_baseline_has_three_corridors(self, sea_routes, corridors):
        baselines = _baseline_voyages(sea_routes, corridors)
        corridor_ids = {v["delivery_corridor"] for v in baselines}
        assert "strait_of_hormuz" in corridor_ids
        assert "cape_of_good_hope" in corridor_ids
        assert "malacca_strait" in corridor_ids

    def test_baseline_voyages_are_baseline_type(self, sea_routes, corridors):
        baselines = _baseline_voyages(sea_routes, corridors)
        assert all(v["type"] == "baseline" for v in baselines)

    def test_baseline_has_flow_volumes(self, sea_routes, corridors):
        baselines = _baseline_voyages(sea_routes, corridors)
        for v in baselines:
            assert v["volume_mbd"] > 0
