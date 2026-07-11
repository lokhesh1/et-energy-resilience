"""
Unit tests for the procurement pod — the 3 regional bidders + the Bid Evaluator.

Everything here is deterministic: the bidders are a filter over data/suppliers.json
plus ONE live input (Brent), and the evaluator is scoring + a greedy gap-fill. The
only external call is price_feed.fetch_price (yfinance), which is MOCKED so tests
never hit the network. sanctions_check / grade_lookup read local JSON and run for
real (offline, deterministic).

Coverage:
  bidders   — volume sizing (min(max, gap)), price = brent+premium, Brent fallback,
              sanctioned→"blocked" stamp, spot scarcity surcharge, gap=0 skip,
              region filter, reducer-only writes.
  evaluator — cheapest-first score, gap coverage within band, PROC-07 sum invariant,
              last-cargo trim, sanctioned excluded from mix, bid pheromone deposit,
              clean constitution on a valid mix.
  integration — fan-out (3 bidders) → fan-in (evaluator) yields a clean mix.

NOTE: a few tests lock the CURRENT price-first behaviour (marked "behaviour"). The
cost-of-delay scoring upgrade (CLAUDE.md backlog) will deliberately revise those;
the invariant tests (coverage, PROC-07, sanctions exclusion, pheromones) survive it.
"""
import operator
from unittest.mock import patch

import pytest

import agents.procurement._sourcing_base as base
from agents.procurement.west_africa_agent import west_africa_node
from agents.procurement.americas_agent import americas_node
from agents.procurement.spot_market_agent import spot_market_node
from agents.procurement.bid_evaluator import (
    bid_evaluator_node, _score, _eligible, _urgency, _impact_per_day,
    _GRADE_MISMATCH_PENALTY, _DISRUPTED_ROUTE_PENALTY, _DELAY,
)

_BRENT = 80.0


def _price_ok(price=_BRENT):
    return {"tool": "price_feed", "status": "ok",
            "data": {"ticker": "BZ=F", "current_price": price}}


def _price_failed():
    return {"tool": "price_feed", "status": "failed", "data": {"error": "no net"}}


@pytest.fixture(autouse=True)
def _mock_brent():
    """Pin Brent to a fixed price so bid arithmetic is deterministic."""
    with patch.object(base, "fetch_price", return_value=_price_ok()) as m:
        yield m


def _state(gap=1.2, hormuz_disrupted=True, scarcity=0.9,
           affected=("jamnagar_ril", "mathura_iocl"),
           critical_count=0, stressed_count=0):
    return {
        "twin_state": {
            "total_india_shortfall_mbd": gap,
            "critical_count": critical_count,
            "stressed_count": stressed_count,
            "corridors": [
                {"id": "strait_of_hormuz",
                 "disruption_fraction": 0.8 if hormuz_disrupted else 0.0},
                {"id": "cape_of_good_hope", "disruption_fraction": 0.0},
            ],
        },
        "affected_refineries": list(affected),
        "pheromone_field": {"strait_of_hormuz": scarcity},
    }


def _all_bids(state):
    """Simulate the parallel fan-out: concat each bidder's bids via operator.add."""
    bids = []
    for node in (west_africa_node, americas_node, spot_market_node):
        bids = operator.add(bids, node(state)["bids"])
    return bids


# ── Bidders: sizing & pricing ──────────────────────────────────────────────────

def test_volume_capped_at_gap_when_gap_small():
    out = west_africa_node(_state(gap=0.1))
    assert out["bids"], "expected west-africa bids"
    for b in out["bids"]:
        assert b["volume_mbd"] == pytest.approx(0.1)   # min(max_volume, 0.1) = 0.1


def test_volume_capped_at_max_when_gap_large():
    out = west_africa_node(_state(gap=5.0))
    for b in out["bids"]:
        assert b["volume_mbd"] == pytest.approx(b["max_volume_mbd"])


def test_price_equals_brent_plus_premium():
    out = west_africa_node(_state())
    for b in out["bids"]:
        assert b["brent_ref"] == _BRENT
        assert b["price_per_bbl"] == pytest.approx(_BRENT + b["price_premium_usd"])


def test_brent_fallback_on_failed_fetch(_mock_brent):
    _mock_brent.return_value = _price_failed()
    out = west_africa_node(_state())
    assert out["audit_trail"][0]["brent_source"] == "fallback"
    for b in out["bids"]:
        assert b["brent_ref"] == base._BRENT_FALLBACK


# ── Bidders: sanctions, scarcity, region ───────────────────────────────────────

def test_sanctioned_supplier_stamped_blocked():
    bids = {b["supplier_id"]: b for b in spot_market_node(_state())["bids"]}
    assert bids["nioc_iranian_heavy"]["sanctions_status"] == "blocked"
    assert bids["rosneft_urals"]["sanctions_status"] == "blocked"
    assert bids["spot_arab_light"]["sanctions_status"] == "clear"


def test_spot_premium_scales_with_scarcity():
    # spot surcharge = scarcity × _SPOT_SCARCITY_SURCHARGE_MAX (0.9 × 5.0 = 4.5)
    spot = {b["supplier_id"]: b for b in spot_market_node(_state(scarcity=0.9))["bids"]}
    assert spot["spot_arab_light"]["price_premium_usd"] == pytest.approx(3.5 + 4.5)
    assert spot["spot_arab_light"]["scarcity_surcharge_applied"] is True


def test_non_spot_bidder_does_not_scale_premium():
    wa = {b["supplier_id"]: b for b in west_africa_node(_state(scarcity=0.9))["bids"]}
    assert wa["nnpc_bonny_light"]["price_premium_usd"] == pytest.approx(2.5)  # base, unscaled
    assert wa["nnpc_bonny_light"]["scarcity_surcharge_applied"] is False


def test_region_filter():
    for node, region in ((west_africa_node, "west_africa"),
                         (americas_node, "americas"),
                         (spot_market_node, "spot")):
        for b in node(_state())["bids"]:
            assert b["region"] == region


def test_grade_compatible_hint_set():
    wa = {b["supplier_id"]: b for b in west_africa_node(_state())["bids"]}
    # bonny_light (light_sweet) is runnable by at least one affected refinery
    assert wa["nnpc_bonny_light"]["grade_compatible"] is True


def test_routes_through_disrupted_flag():
    spot = {b["supplier_id"]: b for b in spot_market_node(_state(hormuz_disrupted=True))["bids"]}
    assert spot["spot_arab_light"]["routes_through_disrupted"] is True   # Hormuz cargo
    assert spot["spot_lula_atlantic"]["routes_through_disrupted"] is False  # Cape cargo


# ── Bidders: control flow & concurrency safety ─────────────────────────────────

def test_zero_gap_skips_sourcing():
    out = west_africa_node(_state(gap=0.0))
    assert out["bids"] == []
    assert out["audit_trail"][0]["action"] == "sourcing_skipped"


def test_bidder_writes_only_reducer_keys():
    # Parallel fan-out safety: a bidder may ONLY touch operator.add keys.
    out = west_africa_node(_state())
    assert set(out.keys()) <= {"bids", "audit_trail"}


# ── Evaluator: pure scoring / eligibility ──────────────────────────────────────

def _bid(**kw):
    b = {"supplier_id": "x", "supplier": "X", "price_per_bbl": 80.0,
         "volume_mbd": 0.3, "max_volume_mbd": 0.5, "sanctions_status": "clear",
         "transit_days_to_india": 0, "grade_compatible": True,
         "routes_through_disrupted": False, "delivery_corridor": "cape_of_good_hope"}
    b.update(kw)
    return b


def test_score_is_price_plus_cost_of_delay():
    ipd = 0.5
    assert _score(_bid(price_per_bbl=80.0, transit_days_to_india=10), ipd) == \
        pytest.approx(80.0 + 10 * ipd)


def test_score_grade_mismatch_penalty():
    clean = _score(_bid(), 0.15)
    mismatch = _score(_bid(grade_compatible=False), 0.15)
    assert mismatch == pytest.approx(clean + _GRADE_MISMATCH_PENALTY)


def test_score_disrupted_route_penalty():
    clean = _score(_bid(), 0.15)
    disrupted = _score(_bid(routes_through_disrupted=True), 0.15)
    assert disrupted == pytest.approx(clean + _DISRUPTED_ROUTE_PENALTY)


# ── Evaluator: cost-of-delay / urgency ─────────────────────────────────────────

def test_urgency_zero_without_critical_refineries():
    assert _urgency({}) == 0.0
    assert _impact_per_day(0.0) == pytest.approx(_DELAY["base_per_bbl_per_day"])


def test_urgency_rises_and_caps_with_criticals():
    # weight 0.5 → 1 critical = 0.5 urgency; 3 criticals cap at max_urgency (1.0)
    assert _urgency({"critical_count": 1}) == pytest.approx(0.5)
    assert _urgency({"critical_count": 3}) == pytest.approx(_DELAY["max_urgency"])
    assert _impact_per_day(1.0) == pytest.approx(
        _DELAY["base_per_bbl_per_day"] + _DELAY["urgency_extra_per_bbl_per_day"])


def _delay_flip_state(critical_count):
    """Two cargoes, same volume=gap: cheap-but-slow vs pricey-but-fast."""
    cheap_slow = {"supplier_id": "cheap_slow", "supplier": "Cheap Slow",
                  "grade": "wti", "delivery_corridor": "cape_of_good_hope",
                  "transit_days_to_india": 32, "max_volume_mbd": 1.2, "volume_mbd": 1.2,
                  "brent_ref": 80.0, "price_premium_usd": -6.0, "price_per_bbl": 74.0,
                  "sanctions_status": "clear", "grade_compatible": True,
                  "routes_through_disrupted": False}
    pricey_fast = {**cheap_slow, "supplier_id": "pricey_fast", "supplier": "Pricey Fast",
                   "transit_days_to_india": 6, "price_premium_usd": 2.0, "price_per_bbl": 82.0}
    return {
        "twin_state": {"total_india_shortfall_mbd": 1.2,
                       "critical_count": critical_count, "corridors": []},
        "affected_refineries": [],
        "bids": [cheap_slow, pricey_fast],
    }


def test_mild_shortfall_prefers_cheaper_cargo():
    out = bid_evaluator_node(_delay_flip_state(critical_count=0))
    assert out["recommended_mix"]["components"][0]["supplier_id"] == "cheap_slow"


def test_critical_shortfall_prefers_faster_cargo():
    # The cost-of-delay flip: with the shortfall critical, the faster cargo wins
    # despite costing $8/bbl more, because 26 fewer transit days is worth more.
    out = bid_evaluator_node(_delay_flip_state(critical_count=3))
    assert out["recommended_mix"]["components"][0]["supplier_id"] == "pricey_fast"
    assert out["recommended_mix"]["urgency"] == pytest.approx(1.0)


def test_eligible_rejects_sanctioned():
    ok, reason = _eligible(_bid(sanctions_status="blocked"))
    assert ok is False and reason == "sanctioned"


def test_eligible_rejects_nonpositive_volume():
    ok, reason = _eligible(_bid(volume_mbd=0.0))
    assert ok is False and reason == "non_positive_volume"


# ── Evaluator: mix composition ─────────────────────────────────────────────────

def test_mix_covers_gap_within_band():
    state = _state(gap=1.2)
    out = bid_evaluator_node({**state, "bids": _all_bids(state)})
    mix = out["recommended_mix"]
    assert 0.8 <= mix["coverage_ratio"] <= 1.3
    assert mix["covers_gap"] is True


def test_mix_total_equals_sum_of_components():
    # PROC-07 invariant — survives the scoring upgrade.
    state = _state(gap=1.2)
    out = bid_evaluator_node({**state, "bids": _all_bids(state)})
    mix = out["recommended_mix"]
    assert mix["total_volume_mbd"] == pytest.approx(
        sum(c["volume_mbd"] for c in mix["components"]))


def test_last_cargo_trimmed_to_gap():
    state = _state(gap=1.2)
    out = bid_evaluator_node({**state, "bids": _all_bids(state)})
    assert out["recommended_mix"]["total_volume_mbd"] == pytest.approx(1.2)


def test_sanctioned_never_in_mix():
    # Even though Rosneft/NIOC are the CHEAPEST bids, the guardrail excludes them.
    state = _state(gap=2.0)
    out = bid_evaluator_node({**state, "bids": _all_bids(state)})
    for c in out["recommended_mix"]["components"]:
        assert c["sanctions_status"] != "blocked"


def test_cheapest_eligible_selected_first():  # behaviour (price-first)
    state = _state(gap=1.2)
    out = bid_evaluator_node({**state, "bids": _all_bids(state)})
    # Canadian WCS carries a −$6 premium → cheapest landed cost → first component.
    assert out["recommended_mix"]["components"][0]["supplier_id"] == "canada_wcs"


def test_zero_gap_yields_empty_mix():
    state = _state(gap=0.0)
    out = bid_evaluator_node({**state, "bids": _all_bids(state)})
    assert out["recommended_mix"]["components"] == []
    assert out["recommended_mix"]["total_volume_mbd"] == 0.0


# ── Evaluator: outputs, pheromones, constitution ───────────────────────────────

def test_evaluator_deposits_bid_pheromones():
    state = _state(gap=1.2)
    out = bid_evaluator_node({**state, "bids": _all_bids(state)})
    assert out["stigmergy_markers"], "expected bid pheromones"
    for m in out["stigmergy_markers"]:
        assert m["type"] == "bid"
        assert m["deposited_by"] == "bid_evaluator"
        assert 0.0 <= m["intensity"] <= 1.0


def test_evaluated_bids_mark_selected_and_eligibility():
    state = _state(gap=1.2)
    out = bid_evaluator_node({**state, "bids": _all_bids(state)})
    ev = {b["supplier_id"]: b for b in out["evaluated_bids"]}
    assert len(ev) == len(_all_bids(state))                # every bid represented
    assert ev["canada_wcs"]["selected"] is True
    assert ev["rosneft_urals"]["eligible"] is False        # sanctioned
    assert ev["rosneft_urals"]["exclude_reason"] == "sanctioned"


def test_valid_mix_has_no_block_violations():
    state = _state(gap=1.2)
    out = bid_evaluator_node({**state, "bids": _all_bids(state)})
    blocks = [v for v in out["constitution_flags"] if v["severity"] == "block"]
    assert blocks == [], f"unexpected block violations: {blocks}"


def test_evaluator_writes_expected_state_keys():
    state = _state(gap=1.2)
    out = bid_evaluator_node({**state, "bids": _all_bids(state)})
    for key in ("current_agent", "evaluated_bids", "recommended_mix",
                "stigmergy_markers", "audit_trail", "constitution_flags"):
        assert key in out
    assert out["current_agent"] == "bid_evaluator"


# ── Integration: fan-out → fan-in ──────────────────────────────────────────────

def test_fanout_fanin_produces_clean_covered_mix():
    state = _state(gap=1.5)
    bids = _all_bids(state)
    assert len(bids) == 14                                 # 4 WA + 4 Americas + 6 spot = catalog
    out = bid_evaluator_node({**state, "bids": bids})
    mix = out["recommended_mix"]
    assert mix["covers_gap"] is True
    assert all(c["sanctions_status"] != "blocked" for c in mix["components"])
    assert [v for v in out["constitution_flags"] if v["severity"] == "block"] == []
