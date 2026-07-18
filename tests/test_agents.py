"""
Unit tests for agents/gri_agent.py — gri_node.
All external calls (tools + LLM) are mocked unless marked @pytest.mark.integration.
"""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agents.gri_agent import gri_node


# ── Auto-mock long-term memory so unit tests never touch the real cloud ────────

@pytest.fixture(autouse=True)
def _mock_xmemory():
    with patch("agents.gri_agent._xmemory") as m:
        yield m


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
            "reasoning":   "Two high-trust signals with direct corridor reference.",
            "event_type":  "war_conflict",
        },
        "suez_canal": {
            "score":          0.20,
            "confidence":     0.60,
            "evidence_count": 1,
            "key_signals":    ["Hormuz shipping lanes disrupted"],
            "reasoning":      "Indirect signal only — baseline default applied.",
            "event_type":     "none",
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


# ── Long-term memory persistence (xMemory wiring) ─────────────────────────────

def test_notable_corridor_persisted_low_risk_not(_mock_xmemory):
    # hormuz 0.85 (>= 0.6) persisted; suez 0.20 (< 0.6) skipped
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        gri_node(MOCK_STATE)

    persisted = [c.kwargs["payload"]["corridor"] for c in _mock_xmemory.remember.call_args_list]
    assert "strait_of_hormuz" in persisted
    assert "suez_canal" not in persisted


def test_persisted_payload_carries_event_type_for_decay(_mock_xmemory):
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        gri_node(MOCK_STATE)

    hormuz_call = next(
        c for c in _mock_xmemory.remember.call_args_list
        if c.kwargs["payload"]["corridor"] == "strait_of_hormuz"
    )
    assert hormuz_call.kwargs["payload"]["event_type"] == "war_conflict"
    assert hormuz_call.kwargs["event_type"] == "risk_assessment"


def test_memory_failure_does_not_break_node(_mock_xmemory):
    # even if persistence raises, the node must still return normally
    _mock_xmemory.remember.side_effect = RuntimeError("cloud down")
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)

    assert result["current_agent"] == "gri_agent"
    assert "strait_of_hormuz" in result["corridor_risk"]


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
    # LLM puts an unknown corridor inside corridor_risk (alongside a known one —
    # an unknown-ONLY scorecard is now a failed assessment, GRI-09) → GRI-04 warn
    bad_llm_out = {
        "corridor_risk": {
            "strait_of_hormuz": {
                "score":          0.85,
                "confidence":     0.9,
                "evidence_count": 1,
                "key_signals":    ["Hormuz shipping lanes disrupted"],
                "reasoning":      "Direct corridor reference.",
                "event_type":     "war_conflict",
            },
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


# ── Per-corridor evidence coverage (fan-out honesty) ──────────────────────────

def test_audit_carries_evidence_by_corridor():
    news = {**MOCK_NEWS_RESULT,
            "data": {**MOCK_NEWS_RESULT["data"],
                     "evidence_by_corridor": {"strait_of_hormuz": 3, "suez_canal": 0}}}
    fetch_p = patch("agents.gri_agent._fetch_tools",
                    return_value=(news, MOCK_CORRIDOR_RESULT))
    client_p = patch("agents.gri_agent._client")
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(MOCK_LLM_OUTPUT)
        result = gri_node(MOCK_STATE)

    entry = next(e for e in result["audit_trail"] if e["action"] == "tool_fetch")
    assert entry["evidence_by_corridor"] == {"strait_of_hormuz": 3, "suez_canal": 0}


def test_prompt_marks_zero_evidence_corridors_unverified():
    from agents.gri_agent import _build_user_prompt
    p = _build_user_prompt("status?", [], MOCK_CORRIDORS,
                           {"strait_of_hormuz": 2, "suez_canal": 0})
    assert "EVIDENCE COVERAGE" in p
    assert "strait_of_hormuz: 2" in p
    assert "UNVERIFIED" in p          # 0-article corridors ≠ confirmed calm


def test_select_articles_balances_across_corridors():
    # 40 Hormuz articles must not crowd the single Suez article out of the
    # LLM's evidence window.
    from agents.gri_agent import _select_articles
    arts = ([{"title": f"h{i}", "corridors": ["strait_of_hormuz"]} for i in range(40)]
            + [{"title": "s0", "corridors": ["suez_canal"]}])
    sel = _select_articles(arts, cap=10)
    assert len(sel) == 10
    assert any("suez_canal" in a["corridors"] for a in sel)


def test_select_articles_serves_highest_trust_first():
    from agents.gri_agent import _select_articles
    arts = [{"title": "low", "trust_score": 0.2, "corridors": ["strait_of_hormuz"]},
            {"title": "high", "trust_score": 0.95, "corridors": ["strait_of_hormuz"]},
            {"title": "untagged-high", "trust_score": 0.95, "corridors": []}]
    sel = _select_articles(arts, cap=2)
    # trust order within the bucket, and tagged evidence beats untagged noise
    assert [a["title"] for a in sel] == ["high", "low"]


def test_select_articles_proportional_not_diluted():
    """A crisis corridor with 26 articles must dominate the judgment window —
    equal-share balancing diluted the crisis and flipped boards to routine."""
    from agents.gri_agent import _select_articles
    arts = ([{"title": f"h{i}", "corridors": ["strait_of_hormuz"]} for i in range(26)]
            + [{"title": f"p{i}", "corridors": ["panama_canal"]} for i in range(3)]
            + [{"title": f"s{i}", "corridors": ["suez_canal"]} for i in range(3)])
    sel = _select_articles(arts, cap=16)
    hormuz = sum(1 for a in sel if "strait_of_hormuz" in a["corridors"])
    assert hormuz >= 10                          # crisis dominates the window
    assert any("panama_canal" in a["corridors"] for a in sel)   # floor keeps
    assert any("suez_canal" in a["corridors"] for a in sel)     # everyone visible


def test_prompt_carries_rich_evidence_metrics():
    from agents.gri_agent import _build_user_prompt
    p = _build_user_prompt(
        "status?", [], MOCK_CORRIDORS,
        {"strait_of_hormuz": 26},
        {"strait_of_hormuz": {"articles": 26, "independent_domains": 14,
                              "fresh_72h": 12, "top_trust": 0.90,
                              "evidence_weight": 14.2}})
    assert "14 domains" in p and "12 fresh(72h)" in p and "top trust 0.90" in p


def test_prompt_signal_lines_carry_age_and_attribution():
    from agents.gri_agent import _build_user_prompt
    arts = [{"title": "Hormuz strike", "source": "reuters.com", "trust_score": 0.95,
             "age_days": 0.4, "attribution": "attributed",
             "corridors": ["strait_of_hormuz"]}]
    p = _build_user_prompt("status?", arts, MOCK_CORRIDORS, {})
    assert "age 0.4d" in p and "attributed" in p


def test_evidence_ignored_warning_in_audit():
    """Fresh high-trust evidence + baseline LLM score → audit tripwire fires;
    a high score on the same evidence → it doesn't."""
    strong = {"strait_of_hormuz": {"articles": 5, "independent_domains": 4,
                                   "fresh_72h": 3, "top_trust": 0.9,
                                   "evidence_weight": 4.0}}
    news = {**MOCK_NEWS_RESULT,
            "data": {**MOCK_NEWS_RESULT["data"], "corridor_evidence": strong}}
    low_llm = {**MOCK_LLM_OUTPUT, "corridor_risk": {
        "strait_of_hormuz": {"score": 0.2, "confidence": 0.6, "evidence_count": 0,
                             "key_signals": [], "reasoning": "baseline",
                             "event_type": "none"}}}
    fetch_p = patch("agents.gri_agent._fetch_tools",
                    return_value=(news, MOCK_CORRIDOR_RESULT))
    with fetch_p, patch("agents.gri_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(low_llm)
        result = gri_node(MOCK_STATE)
    warn = [e for e in result["audit_trail"]
            if e["action"] == "evidence_ignored_warning"]
    assert len(warn) == 1
    assert "strait_of_hormuz" in warn[0]["corridors"]

    with fetch_p, patch("agents.gri_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(MOCK_LLM_OUTPUT)
        result2 = gri_node(MOCK_STATE)   # hormuz scored 0.85 → no warning
    assert not [e for e in result2["audit_trail"]
                if e["action"] == "evidence_ignored_warning"]


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


# ── Assessment failure fails LOUD (debugger.md #21) ───────────────────────────

def test_llm_retry_recovers_from_transient_failure():
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("transient 500"), _make_llm_response(llm_out)]
        result = gri_node(MOCK_STATE)
    assert result["assessment_failed"] is False
    assert result["corridor_risk"]              # scorecard recovered on retry
    entry = next(e for e in result["audit_trail"] if e["action"] == "llm_assessment")
    assert entry["attempts"] == 2
    assert entry["llm_failure"] is None


def test_llm_failure_sets_assessment_failed_and_audits_reason():
    fetch_p, client_p, _ = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.side_effect = RuntimeError("network down")
        result = gri_node(MOCK_STATE)
    assert result["assessment_failed"] is True
    assert result["corridor_risk"] == {}
    entry = next(e for e in result["audit_trail"] if e["action"] == "llm_assessment")
    assert "network down" in entry["llm_failure"]
    # the empty scorecard is now a block-severity constitution violation, so the
    # coordinator's integrity aggregator can never miss it
    assert any(v["rule_id"] == "GRI-09" for v in result["constitution_flags"])


def test_unknown_corridor_scorecard_is_a_failure_not_calm():
    # a scorecard whose keys survive no filter (e.g. display names instead of
    # ids) must be treated as a failed assessment, not scored-everything-zero
    bad = {"corridor_risk": {"Strait of Hormuz": {"score": 0.9}},
           "novel_corridor_alerts": [], "overall_assessment": "",
           "low_trust_signals_flagged": 0}
    fetch_p, client_p, _ = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(bad)
        result = gri_node(MOCK_STATE)
    assert result["assessment_failed"] is True
    entry = next(e for e in result["audit_trail"] if e["action"] == "llm_assessment")
    assert "no known corridors" in entry["llm_failure"]
    assert entry["attempts"] == 2               # it retried before giving up


# ── Root-cause grouping (validated LLM judgment) ──────────────────────────────

def test_root_cause_groups_validated_and_returned():
    llm_out = dict(MOCK_LLM_OUTPUT)
    llm_out["root_cause_groups"] = [
        # valid — but self-link and unknown corridor must be stripped from driven
        {"origin": "strait_of_hormuz",
         "driven": ["bab_el_mandeb", "strait_of_hormuz", "atlantis_channel"],
         "reasoning": "Attacks declared in support of the Hormuz blockade.",
         "key_signals": ["Hormuz shipping lanes disrupted"]},
        # unknown origin → dropped
        {"origin": "atlantis_channel", "driven": ["suez_canal"]},
        # origin scored 0.20 (< 0.4) — a calm corridor can't be a root cause
        {"origin": "suez_canal", "driven": ["malacca_strait"]},
    ]
    fetch_p, client_p, _ = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)

    rc = result["root_causes"]
    assert len(rc) == 1
    assert rc[0]["origin"] == "strait_of_hormuz"
    assert rc[0]["driven"] == ["bab_el_mandeb"]
    assert any(e.get("action") == "root_cause_grouping"
               for e in result["audit_trail"])


def test_missing_root_cause_groups_is_empty_list():
    fetch_p, client_p, llm_out = _patch_gri()
    with fetch_p, client_p as mock_client:
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_out)
        result = gri_node(MOCK_STATE)
    assert result["root_causes"] == []
    assert not any(e.get("action") == "root_cause_grouping"
                   for e in result["audit_trail"])
