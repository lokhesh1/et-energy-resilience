"""
Tests for tools/spr_calculator.py + its two wiring points.

Covers:
  * the deterministic drawdown model — full vs partial bridge (the drawdown-rate
    cap), days-of-cover arithmetic, covers_duration, the no-gap case;
  * the params fallback (missing file → documented defaults, loudly flagged);
  * the bid_evaluator urgency coupling — thin SPR cover on a big gap raises
    urgency, a comfortably-covered gap adds nothing;
  * the coordinator wiring — an uncovered residual gets a sized SPR bridge in the
    plan, the priority actions, and the template narrative.

Seeded params: 39 mmbbl × 0.9 fill = 35.1 mmbbl usable, max drawdown 1.0 mbd.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tools.spr_calculator as spr
from tools.spr_calculator import calculate_drawdown, days_of_cover
from agents.procurement.bid_evaluator import _urgency, _spr_pressure, _DELAY
import agents.crisis_coordinator as cc

_USABLE = 39.0 * 0.9  # 35.1 mmbbl


# ── Tool envelope + drawdown model ──────────────────────────────────────────────

def test_envelope_shape():
    out = calculate_drawdown(2.0)
    assert out["tool"] == "spr_calculator"
    assert out["status"] == "ok"
    assert out["source_trust_avg"] == 1.0
    assert "retrieved_at" in out
    assert out["data"]["params_source"] == "file"


def test_full_bridge_small_gap():
    # 0.5 mbd is under the 1.0 mbd drawdown cap → SPR bridges the whole gap
    d = calculate_drawdown(0.5)["data"]
    assert d["drawdown_mbd"] == pytest.approx(0.5)
    assert d["bridge_fraction"] == pytest.approx(1.0)
    assert d["unbridged_mbd"] == pytest.approx(0.0)
    assert d["days_of_cover"] == pytest.approx(_USABLE / 0.5)  # 70.2
    assert d["adequacy"] == "full_bridge"


def test_partial_bridge_capped_by_drawdown_rate():
    # A 4 mbd gap against a 1 mbd max drawdown: only a quarter-bridge, however
    # many barrels sit in the caverns.
    d = calculate_drawdown(4.0)["data"]
    assert d["drawdown_mbd"] == pytest.approx(1.0)
    assert d["bridge_fraction"] == pytest.approx(0.25)
    assert d["unbridged_mbd"] == pytest.approx(3.0)
    assert d["days_of_cover"] == pytest.approx(_USABLE)  # 35.1 at the capped rate
    assert d["adequacy"] == "partial_bridge"


def test_no_gap_is_not_needed():
    d = calculate_drawdown(0.0)["data"]
    assert d["adequacy"] == "not_needed"
    assert d["drawdown_mbd"] == 0.0
    assert d["days_of_cover"] is None
    assert d["covers_duration"] is None


def test_covers_duration_full_bridge():
    # full bridge, 70.2 days of cover vs a 42-day disruption → covered
    assert calculate_drawdown(0.5, duration_days=42)["data"]["covers_duration"] is True
    # ... but not a 100-day one
    assert calculate_drawdown(0.5, duration_days=100)["data"]["covers_duration"] is False


def test_partial_bridge_never_covers_duration():
    # the unbridged flow is lost every day — long cover doesn't make it "covered"
    d = calculate_drawdown(4.0, duration_days=5)["data"]
    assert d["covers_duration"] is False


def test_missing_params_file_falls_back_to_defaults(monkeypatch):
    monkeypatch.setattr(spr, "SPR_PARAMS_PATH", Path("does_not_exist.json"))
    out = calculate_drawdown(2.0)
    assert out["status"] == "ok"
    assert out["data"]["params_source"] == "defaults"
    assert out["data"]["usable_reserve_mmbbl"] == pytest.approx(_USABLE)


def test_days_of_cover_helper():
    assert days_of_cover(2.0) == pytest.approx(_USABLE / 2.0)
    assert days_of_cover(0.0) is None
    assert days_of_cover(-1.0) is None
    assert days_of_cover("not a number") is None


# ── Bid-evaluator urgency coupling ──────────────────────────────────────────────

def test_spr_pressure_zero_when_comfortably_covered():
    # 1.0 mbd gap → 35.1 days of cover > 30-day comfort horizon → no pressure
    assert _spr_pressure(1.0) == 0.0
    assert _spr_pressure(0.0) == 0.0


def test_spr_pressure_rises_with_gap():
    # 10 mbd gap → 3.51 days of cover → pressure ≈ 1 − 3.51/30
    p10 = _spr_pressure(10.0)
    assert p10 == pytest.approx(1.0 - (_USABLE / 10.0) / 30.0)
    assert _spr_pressure(3.0) < p10 < _spr_pressure(20.0)


def test_urgency_includes_spr_pressure():
    # No refinery bands at all — a huge gap alone raises urgency via the SPR term
    calm = _urgency({"total_india_shortfall_mbd": 1.0})
    tight = _urgency({"total_india_shortfall_mbd": 10.0})
    assert calm == 0.0
    assert tight == pytest.approx(
        min(_DELAY["max_urgency"], _DELAY["spr_weight"] * _spr_pressure(10.0)))
    assert tight > 0


# ── Coordinator wiring (offline: LLM → template, memory empty) ──────────────────

def _resp(content):
    msg = MagicMock(); msg.content = json.dumps(content)
    choice = MagicMock(); choice.message = msg
    r = MagicMock(); r.choices = [choice]
    return r


def _run_node(state):
    client = MagicMock()
    client.chat.completions.create.return_value = _resp({})
    mem = MagicMock()
    mem.recall_similar.return_value = []
    with patch.object(cc, "_client", client), patch.object(cc, "_xmemory", mem):
        return cc.coordinator_node(state)


def _uncovered_state():
    return {
        "query": "Iran closes the Strait of Hormuz",
        "corridor_risk": {"strait_of_hormuz": 0.9},
        "corridor_events": {"strait_of_hormuz": "war_conflict"},
        "scenarios": [{"corridor": "strait_of_hormuz", "duration_days": 42}],
        "twin_state": {
            "total_india_shortfall_mbd": 1.0,
            "critical_count": 1, "stressed_count": 0,
            "refineries": [{"name": "crit0", "status": "critical"}],
            "corridors": [{"id": "strait_of_hormuz", "disruption_fraction": 1.0}],
        },
        "recommended_mix": {
            "total_volume_mbd": 0.4, "coverage_ratio": 0.4, "covers_gap": False,
            "components": [{"supplier": "Spot cargo", "supplier_id": "spot_x",
                            "region": "spot", "grade": "lula",
                            "delivery_corridor": "cape_of_good_hope",
                            "volume_mbd": 0.4, "price_per_bbl": 82.0,
                            "transit_days_to_india": 20,
                            "sanctions_status": "clear"}],
            "est_daily_cost_usd": 1000,
        },
        "audit_trail": [],
        "constitution_flags": [],
    }


def test_coordinator_plan_carries_spr_bridge_for_residual():
    out = _run_node(_uncovered_state())
    bridge = out["response_plan"]["procurement"]["spr_bridge"]
    # residual 0.6 mbd < 1.0 mbd cap → full bridge, 35.1/0.6 = 58.5 days
    assert bridge["drawdown_mbd"] == pytest.approx(0.6)
    assert bridge["days_of_cover"] == pytest.approx(_USABLE / 0.6)
    assert bridge["adequacy"] == "full_bridge"
    assert bridge["covers_duration"] is True  # 58.5 days ≥ 42-day scenario
    uncovered = [a for a in out["response_plan"]["priority_actions"]
                 if "UNCOVERED" in a]
    assert uncovered and "SPR can bridge 0.6 mbd" in uncovered[0]
    assert "SPR can bridge" in out["final_recommendation"]


def test_coordinator_no_bridge_when_gap_covered():
    state = _uncovered_state()
    state["recommended_mix"]["total_volume_mbd"] = 1.0
    state["recommended_mix"]["covers_gap"] = True
    state["recommended_mix"]["components"][0]["volume_mbd"] = 1.0
    out = _run_node(state)
    assert out["response_plan"]["procurement"]["spr_bridge"] is None
    assert not any("SPR can bridge" in a
                   for a in out["response_plan"]["priority_actions"])
