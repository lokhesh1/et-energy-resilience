"""
Tests for agents/distiller/experience_distiller.py — the learning-loop agent.

The distillation ENGINE (memory/distillation.py) is covered by test_memory.py; this
suite covers the AGENT layer the distiller owns:
  * build_trajectory — a compact, high-signal digest (drops audit/geojson noise,
    caps list sizes, copies numbers faithfully);
  * _run_outcome — deterministic success/failure labelling from state facts, never
    the LLM's opinion;
  * experience_distiller_node — hands the trajectory to distill_run and reports what
    was learned in the audit; best-effort (never raises, even on engine failure).
"""
from unittest.mock import MagicMock, patch

import pytest

import agents.distiller.experience_distiller as ed
from agents.distiller.experience_distiller import (
    build_trajectory, _run_outcome, experience_distiller_node,
)


# ── State builders ───────────────────────────────────────────────────────────────

def _covered_state():
    """A crisis that was fully resolved: gap closed, no residual, no block flags."""
    return {
        "query": "Iran closes the Strait of Hormuz",
        "corridor_risk": {"strait_of_hormuz": 0.9, "suez_canal": 0.3},
        "corridor_events": {"strait_of_hormuz": "war_conflict", "suez_canal": "none"},
        "scenarios": [
            {"corridor": "strait_of_hormuz", "event_type": "war_conflict",
             "volume_at_risk_mbd": 21.0, "india_exposure_mbd": 13.0,
             "duration_days": 42, "severity": "critical"},
        ],
        "twin_state": {
            "total_india_shortfall_mbd": 1.0, "critical_count": 1, "stressed_count": 0,
            "refineries": [{"name": "Jamnagar", "status": "critical"},
                           {"name": "Chennai", "status": "normal"}],
            "corridors": [{"id": "strait_of_hormuz", "disruption_fraction": 1.0},
                          {"id": "suez_canal", "disruption_fraction": 0.0}],
        },
        "recommended_mix": {
            "total_volume_mbd": 1.0, "coverage_ratio": 1.0, "covers_gap": True,
            "components": [{"supplier": "Bonny Light", "region": "west_africa",
                            "grade": "bonny_light", "volume_mbd": 1.0,
                            "delivery_corridor": "cape_of_good_hope"}],
        },
        "response_plan": {
            "escalation_level": "critical",
            "procurement": {"covers_gap": True, "residual_gap_mbd": 0.0},
            "unresolved_issues": [],
        },
        "final_recommendation": "CRITICAL: Hormuz war; West Africa cargo closes the gap.",
    }


def _uncovered_state():
    s = _covered_state()
    s["recommended_mix"].update({"total_volume_mbd": 0.4, "coverage_ratio": 0.4,
                                 "covers_gap": False})
    s["response_plan"]["procurement"] = {"covers_gap": False, "residual_gap_mbd": 0.6}
    s["response_plan"]["unresolved_issues"] = ["0.6 mbd shortfall uncovered by market supply."]
    return s


def _no_gap_state():
    return {
        "query": "routine check",
        "corridor_risk": {"strait_of_hormuz": 0.1},
        "corridor_events": {"strait_of_hormuz": "none"},
        "scenarios": [],
        "twin_state": {"total_india_shortfall_mbd": 0.0, "critical_count": 0,
                       "stressed_count": 0, "refineries": [], "corridors": []},
        "recommended_mix": {},
        "response_plan": {"escalation_level": "routine", "procurement": {},
                          "unresolved_issues": []},
        "final_recommendation": "ROUTINE: no shortfall.",
    }


# ── _run_outcome (deterministic) ─────────────────────────────────────────────────

def test_outcome_success_when_gap_covered():
    assert _run_outcome(_covered_state()) == "success"


def test_outcome_failure_when_gap_uncovered():
    assert _run_outcome(_uncovered_state()) == "failure"


def test_outcome_success_when_no_shortfall():
    assert _run_outcome(_no_gap_state()) == "success"


def test_outcome_failure_on_residual_despite_covers_flag():
    # Contradiction guard: covers_gap True but a residual remains → still a failure.
    s = _covered_state()
    s["response_plan"]["procurement"] = {"covers_gap": True, "residual_gap_mbd": 0.3}
    assert _run_outcome(s) == "failure"


# ── build_trajectory ─────────────────────────────────────────────────────────────

def test_trajectory_carries_signal_and_outcome():
    traj = build_trajectory(_covered_state())
    assert traj["outcome"] == "success"
    assert traj["escalation_level"] == "critical"
    assert traj["twin"]["gap_mbd"] == 1.0
    assert traj["twin"]["critical_refineries"] == ["Jamnagar"]
    assert traj["twin"]["disrupted_corridors"] == ["strait_of_hormuz"]
    assert traj["procurement"]["covers_gap"] is True
    assert traj["procurement"]["cargoes"][0]["supplier"] == "Bonny Light"


def test_trajectory_ranks_risks_and_attaches_event_type():
    traj = build_trajectory(_covered_state())
    assert traj["corridor_risks"][0]["corridor"] == "strait_of_hormuz"   # highest score first
    assert traj["corridor_risks"][0]["event_type"] == "war_conflict"


def test_trajectory_drops_noise():
    # The raw audit_trail / geojson must never reach the digest.
    s = _covered_state()
    s["audit_trail"] = [{"agent": "x"}] * 50
    s["twin_state"]["geojson"] = {"features": [1, 2, 3]}
    traj = build_trajectory(s)
    assert "audit_trail" not in traj
    assert "geojson" not in traj["twin"]


def test_trajectory_caps_list_sizes():
    s = _covered_state()
    s["scenarios"] = [dict(corridor=f"c{i}", severity="low") for i in range(20)]
    traj = build_trajectory(s)
    assert len(traj["scenarios"]) == ed._MAX_ITEMS


def test_trajectory_handles_dict_score_shape():
    s = _covered_state()
    s["corridor_risk"] = {"strait_of_hormuz": {"score": 0.85}}
    traj = build_trajectory(s)
    assert traj["corridor_risks"][0]["score"] == 0.85


# ── experience_distiller_node ────────────────────────────────────────────────────

def _mock_xmemory(report):
    mem = MagicMock()
    mem.distill_run.return_value = report
    return mem


def test_node_reports_what_was_learned():
    report = {"episodic_written": 2, "semantic_written": 2,
              "skill_written": True, "skill_skipped_reason": None}
    with patch.object(ed, "_xmemory", _mock_xmemory(report)):
        out = experience_distiller_node(_covered_state())
    entry = out["audit_trail"][0]
    assert out["current_agent"] == "experience_distiller"
    assert entry["outcome"] == "success"
    assert entry["episodic_written"] == 2
    assert entry["skill_written"] is True


def test_node_passes_trajectory_to_engine():
    mem = _mock_xmemory({})
    with patch.object(ed, "_xmemory", mem):
        experience_distiller_node(_covered_state())
    (traj,), _ = mem.distill_run.call_args
    assert traj["query"] == "Iran closes the Strait of Hormuz"
    assert traj["outcome"] == "success"


def test_node_records_skill_skip_reason():
    report = {"episodic_written": 1, "semantic_written": 1,
              "skill_written": False, "skill_skipped_reason": "no candidate_skill"}
    with patch.object(ed, "_xmemory", _mock_xmemory(report)):
        out = experience_distiller_node(_uncovered_state())
    assert out["audit_trail"][0]["skill_skipped_reason"] == "no candidate_skill"
    assert out["audit_trail"][0]["outcome"] == "failure"


def test_node_survives_empty_engine_report():
    # distill_run degraded to {} (LLM failed upstream) → node still clean, no raise.
    with patch.object(ed, "_xmemory", _mock_xmemory({})):
        out = experience_distiller_node(_covered_state())
    entry = out["audit_trail"][0]
    assert entry["episodic_written"] == 0
    assert entry["skill_written"] is False
