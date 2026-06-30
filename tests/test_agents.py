"""
Unit tests for agents/gri_agent.py — gri_node.
All external calls (tools + LLM) are mocked unless marked @pytest.mark.integration.
"""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agents.gri_agent import gri_node


# ── Shared mock data ──────────────────────────────────────────────────────────

MOCK_ARTICLES = [
    {"title": "Hormuz shipping lanes disrupted", "url": "https://reuters.com/a1",
     "source": "reuters.com", "trust_score": 0.95, "trusted": True},
    {"title": "Iran navy exercises near strait", "url": "https://ft.com/a2",
     "source": "ft.com", "trust_score": 0.90, "trusted": True},
    {"title": "RT claims no disruption", "url": "https://rt.com/a3",
     "source": "rt.com", "trust_score": 0.30, "trusted": False},
]

MOCK_CORRIDORS = [
    {"id": "strait_of_hormuz", "name": "Strait of Hormuz", "region": "Middle East",
     "chokepoint": True, "baseline_flow_mbd": 21.0, "risk_factors": ["Iran", "US tension"]},
    {"id": "suez_canal", "name": "Suez Canal", "region": "North Africa",
     "chokepoint": True, "baseline_flow_mbd": 5.5, "risk_factors": ["Egypt stability"]},
    {"id": "malacca_strait", "name": "Malacca Strait", "region": "Southeast Asia",
     "chokepoint": True, "baseline_flow_mbd": 16.0, "risk_factors": ["Piracy"]},
]

MOCK_NEWS_RESULT = {
    "tool":                      "news_fetcher",
    "status":                    "ok",
    "data":                      {"articles": MOCK_ARTICLES, "errors": []},
    "source_trust_avg":          0.72,
    "low_trust_sources_flagged": 1,
    "retrieved_at":              "2026-07-01T00:00:00+00:00",
    "staleness_seconds":         2,
}

MOCK_CORRIDOR_RESULT = {
    "tool":                      "corridor_status",
    "status":                    "ok",
    "data":                      {"corridors": MOCK_CORRIDORS},
    "source_trust_avg":          1.0,
    "low_trust_sources_flagged": 0,
    "retrieved_at":              "2026-07-01T00:00:00+00:00",
    "staleness_seconds":         0,
}

MOCK_LLM_OUTPUT = {
    "corridor_risk": {
        "strait_of_hormuz": {
            "score":          0.85,
            "confidence":     0.90,
            "evidence_count": 2,
            "key_signals":    [
                "Hormuz shipping lanes disrupted",
                "Iran navy exercises near strait",
            ],
            "reasoning": "Two high-trust signals with direct corridor reference.",
        },
        "suez_canal": {
            "score":          0.20,
            "confidence":     0.60,
            "evidence_count": 1,
            "key_signals":    ["Hormuz shipping lanes disrupted"],
            "reasoning":      "Indirect signal only — baseline default applied.",
        },
    },
    "novel_corridor_alerts":     [],
    "overall_assessment":        "Elevated risk at Hormuz; other corridors nominal.",
    "low_trust_signals_flagged": 1,
}

MOCK_STATE = {"query": "oil supply disruption Middle East"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm_response(content: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(content)
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _patch_gri(llm_output=None):
    """Context manager that patches both _fetch_tools and the OpenAI client."""
    llm_output = llm_output or MOCK_LLM_OUTPUT

    fetch_patch = patch(
        "agents.gri_agent._fetch_tools",
        return_value=(MOCK_NEWS_RESULT, MOCK_CORRIDOR_RESULT),
    )
    client_patch = patch("agents.gri_agent._client")

    return fetch_patch, client_patch, llm_output


# ── State shape ───────────────────────────────────────────────────────────────

def test_gri_node_returns_expected_keys():
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)

    assert "current_agent"     in result
    assert "risk_signals"      in result
    assert "corridor_risk"     in result
    assert "stigmergy_markers" in result
    assert "audit_trail"       in result
    assert "constitution_flags" in result


def test_gri_node_sets_current_agent():
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)

    assert result["current_agent"] == "gri_agent"


# ── corridor_risk filtering ───────────────────────────────────────────────────

def test_corridor_risk_values_are_floats():
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)

    for cid, score in result["corridor_risk"].items():
        assert isinstance(score, float), f"{cid} score is not a float"


def test_unknown_corridor_excluded_from_corridor_risk():
    llm_out = {**MOCK_LLM_OUTPUT, "corridor_risk": {
        **MOCK_LLM_OUTPUT["corridor_risk"],
        "red_sea_new": {
            "score": 0.55, "confidence": 0.5,
            "evidence_count": 1, "key_signals": ["Red Sea incident"], "reasoning": "test",
        },
    }}
    fetch_p, client_p, _ = _patch_gri(llm_out)
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)

    assert "red_sea_new" not in result["corridor_risk"]


def test_known_corridor_retained_in_corridor_risk():
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)

    assert "strait_of_hormuz" in result["corridor_risk"]


# ── Stigmergy pheromones ──────────────────────────────────────────────────────

def test_high_risk_corridor_deposits_pheromone():
    # strait_of_hormuz has score=0.85 → must deposit marker
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)

    targets = [m["target"] for m in result["stigmergy_markers"]]
    assert "strait_of_hormuz" in targets


def test_low_risk_corridor_does_not_deposit_pheromone():
    # suez_canal score=0.20 → below 0.6 threshold
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)

    targets = [m["target"] for m in result["stigmergy_markers"]]
    assert "suez_canal" not in targets


def test_pheromone_deposited_by_gri_agent():
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)

    for marker in result["stigmergy_markers"]:
        assert marker["deposited_by"] == "gri_agent"
        assert marker["type"] == "risk"
        assert 0.0 <= marker["intensity"] <= 1.0


# ── Audit trail ───────────────────────────────────────────────────────────────

def test_audit_trail_has_two_entries():
    # one for tool_fetch, one for llm_assessment
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)

    assert len(result["audit_trail"]) == 2
    actions = [e["action"] for e in result["audit_trail"]]
    assert "tool_fetch" in actions
    assert "llm_assessment" in actions


# ── LLM failure fallback ──────────────────────────────────────────────────────

def test_llm_failure_does_not_crash_node():
    fetch_p = patch(
        "agents.gri_agent._fetch_tools",
        return_value=(MOCK_NEWS_RESULT, MOCK_CORRIDOR_RESULT),
    )
    client_p = patch("agents.gri_agent._client")

    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.side_effect = Exception("LLM timeout")
        result = gri_node(MOCK_STATE)

    assert "corridor_risk" in result
    assert "overall_assessment" not in result or True  # node must not raise


def test_llm_failure_produces_empty_corridor_risk():
    fetch_p = patch(
        "agents.gri_agent._fetch_tools",
        return_value=(MOCK_NEWS_RESULT, MOCK_CORRIDOR_RESULT),
    )
    client_p = patch("agents.gri_agent._client")

    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.side_effect = RuntimeError("network error")
        result = gri_node(MOCK_STATE)

    assert result["corridor_risk"] == {}


def test_llm_failure_still_returns_risk_signals():
    fetch_p = patch(
        "agents.gri_agent._fetch_tools",
        return_value=(MOCK_NEWS_RESULT, MOCK_CORRIDOR_RESULT),
    )
    client_p = patch("agents.gri_agent._client")

    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.side_effect = RuntimeError("network error")
        result = gri_node(MOCK_STATE)

    assert result["risk_signals"] == MOCK_ARTICLES


# ── #1: Constitution violations from LLM output ───────────────────────────────

def test_llm_evidence_count_mismatch_sets_constitution_flags():
    # LLM returns evidence_count=5 but only 1 key_signal → GRI-06 block
    bad_llm_out = {
        "corridor_risk": {
            "strait_of_hormuz": {
                "score":          0.75,
                "confidence":     0.8,
                "evidence_count": 5,           # mismatch: only 1 signal below
                "key_signals":    ["One signal"],
                "reasoning":      "test",
            }
        },
        "novel_corridor_alerts":     [],
        "overall_assessment":        "Test.",
        "low_trust_signals_flagged": 1,
    }
    fetch_p = patch(
        "agents.gri_agent._fetch_tools",
        return_value=(MOCK_NEWS_RESULT, MOCK_CORRIDOR_RESULT),
    )
    client_p = patch("agents.gri_agent._client")
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(bad_llm_out)
        result = gri_node(MOCK_STATE)

    assert len(result["constitution_flags"]) > 0
    rule_ids = {v["rule_id"] for v in result["constitution_flags"]}
    assert "GRI-06" in rule_ids


def test_llm_unknown_corridor_in_output_sets_constitution_flags():
    # LLM puts an unknown corridor inside corridor_risk → GRI-04 warn
    bad_llm_out = {
        "corridor_risk": {
            "red_sea_new": {
                "score":          0.60,
                "confidence":     0.7,
                "evidence_count": 1,
                "key_signals":    ["Red Sea incident"],
                "reasoning":      "Novel corridor reported.",
            }
        },
        "novel_corridor_alerts":     [],
        "overall_assessment":        "Test.",
        "low_trust_signals_flagged": 0,
    }
    fetch_p = patch(
        "agents.gri_agent._fetch_tools",
        return_value=(MOCK_NEWS_RESULT, MOCK_CORRIDOR_RESULT),
    )
    client_p = patch("agents.gri_agent._client")
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(bad_llm_out)
        result = gri_node(MOCK_STATE)

    rule_ids = {v["rule_id"] for v in result["constitution_flags"]}
    assert "GRI-04" in rule_ids


# ── #2: Malformed JSON from LLM ───────────────────────────────────────────────

def test_malformed_json_from_llm_hits_fallback():
    # LLM returns a non-JSON string → json.loads raises, exception handler fires
    broken_msg = MagicMock()
    broken_msg.content = "Sorry, I cannot provide that. { broken json <<"
    broken_choice = MagicMock()
    broken_choice.message = broken_msg
    broken_response = MagicMock()
    broken_response.choices = [broken_choice]

    fetch_p = patch(
        "agents.gri_agent._fetch_tools",
        return_value=(MOCK_NEWS_RESULT, MOCK_CORRIDOR_RESULT),
    )
    client_p = patch("agents.gri_agent._client")
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = broken_response
        result = gri_node(MOCK_STATE)

    assert result["corridor_risk"] == {}
    assert result["risk_signals"] == MOCK_ARTICLES  # tool data still returned


def test_malformed_json_audit_trail_still_written():
    broken_msg = MagicMock()
    broken_msg.content = "not json"
    broken_choice = MagicMock()
    broken_choice.message = broken_msg
    broken_response = MagicMock()
    broken_response.choices = [broken_choice]

    fetch_p = patch(
        "agents.gri_agent._fetch_tools",
        return_value=(MOCK_NEWS_RESULT, MOCK_CORRIDOR_RESULT),
    )
    client_p = patch("agents.gri_agent._client")
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = broken_response
        result = gri_node(MOCK_STATE)

    assert len(result["audit_trail"]) == 2


# ── #3: Live integration test ─────────────────────────────────────────────────

@pytest.mark.integration
def test_gri_node_live_llm_real_api():
    """
    Calls the real OpenRouter LLM with canned tool data.
    Skipped automatically if OPENROUTER_API_KEY is not set.
    Run explicitly: pytest -m integration
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    fetch_p = patch(
        "agents.gri_agent._fetch_tools",
        return_value=(MOCK_NEWS_RESULT, MOCK_CORRIDOR_RESULT),
    )
    with fetch_p:
        result = gri_node(MOCK_STATE)

    # Shape checks
    assert result["current_agent"] == "gri_agent"
    assert isinstance(result["corridor_risk"], dict)
    assert isinstance(result["risk_signals"], list)
    assert len(result["audit_trail"]) == 2

    # Score validity — whatever the LLM returned must be in range
    for cid, score in result["corridor_risk"].items():
        assert isinstance(score, float), f"{cid} score not a float"
        assert 0.0 <= score <= 1.0, f"{cid} score out of range: {score}"

    # Pheromone markers must be well-formed
    for marker in result["stigmergy_markers"]:
        assert marker["deposited_by"] == "gri_agent"
        assert 0.0 <= marker["intensity"] <= 1.0
