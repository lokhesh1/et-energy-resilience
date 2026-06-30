"""
Unit tests for eib_guardrails/constitution_checker.py — GRI rules.
Pure tests: no network, no LLM, no file writes.
"""
import pytest
from eib_guardrails.constitution_checker import check


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _good_article():
    return {"title": "Hormuz tensions rise", "trust_score": 0.95, "trusted": True}


def _corridor_entry(score=0.75, signals=None):
    signals = signals or ["Hormuz tensions rise"]
    return {
        "score":          score,
        "confidence":     0.8,
        "evidence_count": len(signals),
        "key_signals":    signals,
        "reasoning":      "Signal count × trust weight.",
    }


def _good_output():
    return {
        "risk_signals": [_good_article()],
        "corridor_risk": {
            "strait_of_hormuz": _corridor_entry(0.75),
            "suez_canal":       _corridor_entry(0.30, ["Suez closure reported"]),
        },
        "novel_corridor_alerts":     [],
        "overall_assessment":        "Elevated risk in Persian Gulf.",
        "low_trust_signals_flagged": 0,
    }


def _rule_ids(result):
    return {v["rule_id"] for v in result["violations"]}


# ── Happy path ────────────────────────────────────────────────────────────────

def test_happy_path_passes():
    result = check("gri", _good_output())
    assert result["passed"] is True
    assert result["violations"] == []


# ── GRI-02 ────────────────────────────────────────────────────────────────────

def test_gri02_missing_trust_score_blocks():
    out = _good_output()
    out["risk_signals"] = [{"title": "No trust field"}]
    result = check("gri", out)
    assert "GRI-02" in _rule_ids(result)
    assert result["passed"] is False


def test_gri02_passes_when_all_articles_have_trust_score():
    out = _good_output()
    out["risk_signals"] = [_good_article(), _good_article()]
    result = check("gri", out)
    assert "GRI-02" not in _rule_ids(result)


# ── GRI-03 ────────────────────────────────────────────────────────────────────

def test_gri03_score_above_one_warns():
    out = _good_output()
    out["corridor_risk"]["strait_of_hormuz"]["score"] = 1.5
    result = check("gri", out)
    assert "GRI-03" in _rule_ids(result)


def test_gri03_negative_score_warns():
    out = _good_output()
    out["corridor_risk"]["strait_of_hormuz"]["score"] = -0.1
    result = check("gri", out)
    assert "GRI-03" in _rule_ids(result)


def test_gri03_boundary_zero_and_one_pass():
    out = _good_output()
    out["corridor_risk"]["strait_of_hormuz"]["score"] = 0.0
    out["corridor_risk"]["suez_canal"]["score"] = 1.0
    result = check("gri", out)
    assert "GRI-03" not in _rule_ids(result)


# ── GRI-04 ────────────────────────────────────────────────────────────────────

def test_gri04_unknown_corridor_warns():
    out = _good_output()
    out["corridor_risk"]["red_sea_new"] = _corridor_entry(0.5)
    result = check("gri", out)
    assert "GRI-04" in _rule_ids(result)


def test_gri04_all_eight_known_corridors_pass():
    known = [
        "strait_of_hormuz", "suez_canal", "malacca_strait", "bab_el_mandeb",
        "turkish_straits", "danish_straits", "cape_of_good_hope", "panama_canal",
    ]
    out = _good_output()
    out["corridor_risk"] = {c: _corridor_entry(0.3, [f"Signal {c}"]) for c in known}
    result = check("gri", out)
    assert "GRI-04" not in _rule_ids(result)


# ── GRI-05 ────────────────────────────────────────────────────────────────────

def test_gri05_low_trust_not_flagged_warns():
    out = _good_output()
    out["risk_signals"] = [{"title": "RT piece", "trust_score": 0.30, "trusted": False}]
    out["low_trust_signals_flagged"] = 0
    result = check("gri", out)
    assert "GRI-05" in _rule_ids(result)


def test_gri05_passes_when_flagged():
    out = _good_output()
    out["risk_signals"] = [{"title": "RT piece", "trust_score": 0.30, "trusted": False}]
    out["low_trust_signals_flagged"] = 1
    result = check("gri", out)
    assert "GRI-05" not in _rule_ids(result)


def test_gri05_passes_when_no_low_trust_articles():
    result = check("gri", _good_output())
    assert "GRI-05" not in _rule_ids(result)


# ── GRI-06 ────────────────────────────────────────────────────────────────────

def test_gri06_count_mismatch_blocks():
    out = _good_output()
    out["corridor_risk"]["strait_of_hormuz"]["evidence_count"] = 5
    out["corridor_risk"]["strait_of_hormuz"]["key_signals"] = ["Only one"]
    result = check("gri", out)
    assert "GRI-06" in _rule_ids(result)
    assert result["passed"] is False


def test_gri06_passes_when_count_matches():
    out = _good_output()
    signals = ["A", "B", "C"]
    out["corridor_risk"]["strait_of_hormuz"]["evidence_count"] = 3
    out["corridor_risk"]["strait_of_hormuz"]["key_signals"] = signals
    result = check("gri", out)
    assert "GRI-06" not in _rule_ids(result)


# ── GRI-07 ────────────────────────────────────────────────────────────────────

def test_gri07_zero_evidence_blocks():
    out = _good_output()
    out["corridor_risk"]["strait_of_hormuz"]["evidence_count"] = 0
    out["corridor_risk"]["strait_of_hormuz"]["key_signals"] = []
    result = check("gri", out)
    assert "GRI-07" in _rule_ids(result)
    assert result["passed"] is False


def test_gri07_one_evidence_passes():
    out = _good_output()
    out["corridor_risk"]["strait_of_hormuz"]["evidence_count"] = 1
    out["corridor_risk"]["strait_of_hormuz"]["key_signals"] = ["One signal"]
    result = check("gri", out)
    assert "GRI-07" not in _rule_ids(result)


# ── Missing constitution ───────────────────────────────────────────────────────

def test_missing_constitution_returns_passed_with_warning():
    result = check("nonexistent_agent", {"foo": "bar"})
    assert result["passed"] is True
    assert "warning" in result
