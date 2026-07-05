"""
Unit tests for the memory layer:
  - memory/working_memory.py  (in-process scratchpad, no network)
  - memory/episodic_store.py  (Supabase-backed; client is mocked, no network)
  - memory/semantic_store.py  (Pinecone-backed; index + embed mocked, no network)
"""
from unittest.mock import MagicMock

import pytest

from memory.working_memory import WorkingMemory
from memory.episodic_store import EpisodicStore
import memory.semantic_store as semantic_mod
from memory.semantic_store import SemanticStore
from memory.procedural_store import ProceduralStore
import memory.distillation as distill_mod
from memory.distillation import Distiller
from memory.xmemory import XMemory


# ══════════════════════════════════════════════════════════════════════════════
# WorkingMemory
# ══════════════════════════════════════════════════════════════════════════════

def test_set_and_get():
    wm = WorkingMemory(budget=1000)
    assert wm.set("a", "hello", 10) is True
    assert wm.get("a") == "hello"


def test_get_missing_returns_none():
    wm = WorkingMemory()
    assert wm.get("nope") is None


def test_budget_remaining_tracks_usage():
    wm = WorkingMemory(budget=100)
    wm.set("a", "x", 30)
    wm.set("b", "y", 20)
    assert wm.budget_remaining() == 50


def test_oversized_item_rejected():
    wm = WorkingMemory(budget=100)
    assert wm.set("big", "x", 101) is False
    assert wm.get("big") is None
    assert wm.budget_remaining() == 100


def test_item_equal_to_budget_accepted():
    wm = WorkingMemory(budget=100)
    assert wm.set("exact", "x", 100) is True
    assert wm.budget_remaining() == 0


def test_fifo_eviction_when_over_budget():
    wm = WorkingMemory(budget=100)
    wm.set("first", "1", 60)
    wm.set("second", "2", 30)
    # adding 50 must evict "first" (oldest) to fit
    wm.set("third", "3", 50)
    assert wm.get("first") is None
    assert wm.get("second") == "2"
    assert wm.get("third") == "3"


def test_eviction_frees_enough_multiple():
    wm = WorkingMemory(budget=100)
    wm.set("a", "1", 40)
    wm.set("b", "2", 40)
    # needs 80 free → both a and b evicted
    wm.set("c", "3", 90)
    assert wm.get("a") is None
    assert wm.get("b") is None
    assert wm.get("c") == "3"


def test_overwrite_same_key_frees_old_cost():
    wm = WorkingMemory(budget=100)
    wm.set("a", "big", 80)
    wm.set("a", "small", 10)   # replaces, not appends
    assert wm.get("a") == "small"
    assert wm.budget_remaining() == 90


def test_delete_frees_cost():
    wm = WorkingMemory(budget=100)
    wm.set("a", "x", 40)
    wm.delete("a")
    assert wm.get("a") is None
    assert wm.budget_remaining() == 100


def test_delete_missing_is_noop():
    wm = WorkingMemory(budget=100)
    wm.set("a", "x", 40)
    wm.delete("ghost")
    assert wm.budget_remaining() == 60


def test_clear_resets_everything():
    wm = WorkingMemory(budget=100)
    wm.set("a", "x", 40)
    wm.set("b", "y", 30)
    wm.clear()
    assert len(wm) == 0
    assert wm.budget_remaining() == 100


def test_snapshot_returns_key_value_only():
    wm = WorkingMemory(budget=100)
    wm.set("a", "x", 10)
    wm.set("b", {"n": 1}, 20)
    snap = wm.snapshot()
    assert snap == {"a": "x", "b": {"n": 1}}


def test_len_reflects_entry_count():
    wm = WorkingMemory(budget=100)
    wm.set("a", "x", 10)
    wm.set("b", "y", 10)
    assert len(wm) == 2


# ══════════════════════════════════════════════════════════════════════════════
# EpisodicStore  (Supabase client mocked)
# ══════════════════════════════════════════════════════════════════════════════

def _store_with_mock_client():
    """Return (store, mock_client) with the lazy client pre-injected."""
    store = EpisodicStore()
    mock_client = MagicMock()
    store._client = mock_client
    return store, mock_client


def _make_insert_response(rows):
    resp = MagicMock()
    resp.data = rows
    return resp


# ── store() ────────────────────────────────────────────────────────────────────

def test_store_returns_error_when_unconfigured(monkeypatch):
    # no SUPABASE_URL/KEY → _get_client returns None
    import config.settings as settings
    monkeypatch.setattr(settings, "SUPABASE_URL", None)
    monkeypatch.setattr(settings, "SUPABASE_KEY", None)
    store = EpisodicStore()
    res = store.store("t", "agent", {"k": "v"})
    assert res["status"] == "error"


def test_store_returns_ok_and_id():
    store, client = _store_with_mock_client()
    (client.table.return_value
           .insert.return_value
           .execute.return_value) = _make_insert_response([{"id": "uuid-123"}])
    res = store.store("risk_assessment", "gri_agent", {"score": 0.9}, outcome="success")
    assert res == {"status": "ok", "id": "uuid-123"}


def test_store_builds_correct_row():
    store, client = _store_with_mock_client()
    (client.table.return_value
           .insert.return_value
           .execute.return_value) = _make_insert_response([{"id": "x"}])
    store.store("procurement_bid", "spot_agent", {"reason": "no cargo"}, outcome="failure")
    client.table.assert_called_with("episodic_events")
    inserted_row = client.table.return_value.insert.call_args[0][0]
    assert inserted_row == {
        "event_type": "procurement_bid",
        "agent": "spot_agent",
        "payload": {"reason": "no cargo"},
        "outcome": "failure",
    }


def test_store_outcome_defaults_to_none():
    store, client = _store_with_mock_client()
    (client.table.return_value
           .insert.return_value
           .execute.return_value) = _make_insert_response([{"id": "x"}])
    store.store("event", "agent", {})
    inserted_row = client.table.return_value.insert.call_args[0][0]
    assert inserted_row["outcome"] is None


def test_store_returns_error_on_exception():
    store, client = _store_with_mock_client()
    client.table.side_effect = RuntimeError("network down")
    res = store.store("t", "agent", {})
    assert res["status"] == "error"
    assert "network down" in res["error"]


def test_store_handles_empty_response_data():
    store, client = _store_with_mock_client()
    (client.table.return_value
           .insert.return_value
           .execute.return_value) = _make_insert_response([])
    res = store.store("t", "agent", {})
    assert res["status"] == "ok"
    assert res["id"] is None


# ── query() ─────────────────────────────────────────────────────────────────────

def test_query_applies_all_filters():
    store, client = _store_with_mock_client()
    chain = client.table.return_value.select.return_value
    chain.eq.return_value = chain
    chain.order.return_value = chain
    chain.limit.return_value = chain
    chain.execute.return_value = _make_insert_response([{"id": "1"}])

    rows = store.query(agent="gri_agent", event_type="risk", outcome="failure", limit=5)

    assert rows == [{"id": "1"}]
    eq_calls = {c.args for c in chain.eq.call_args_list}
    assert ("agent", "gri_agent") in eq_calls
    assert ("event_type", "risk") in eq_calls
    assert ("outcome", "failure") in eq_calls
    chain.limit.assert_called_with(5)


def test_query_returns_empty_on_exception():
    store, client = _store_with_mock_client()
    client.table.side_effect = RuntimeError("boom")
    assert store.query(agent="x") == []


def test_query_returns_empty_when_unconfigured(monkeypatch):
    import config.settings as settings
    monkeypatch.setattr(settings, "SUPABASE_URL", None)
    monkeypatch.setattr(settings, "SUPABASE_KEY", None)
    store = EpisodicStore()
    assert store.query(agent="x") == []


# ── recent() ────────────────────────────────────────────────────────────────────

def test_recent_orders_and_limits():
    store, client = _store_with_mock_client()
    chain = client.table.return_value.select.return_value
    chain.order.return_value = chain
    chain.limit.return_value = chain
    chain.execute.return_value = _make_insert_response([{"id": "a"}, {"id": "b"}])

    rows = store.recent(n=2)

    assert rows == [{"id": "a"}, {"id": "b"}]
    chain.order.assert_called_with("timestamp", desc=True)
    chain.limit.assert_called_with(2)


def test_recent_returns_empty_on_exception():
    store, client = _store_with_mock_client()
    client.table.side_effect = RuntimeError("boom")
    assert store.recent() == []


# ══════════════════════════════════════════════════════════════════════════════
# SemanticStore  (Pinecone index + embedding model mocked)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _mock_embed(monkeypatch):
    """Replace the real embedding model with a fixed 3-dim vector — no model load."""
    monkeypatch.setattr(semantic_mod, "_embed", lambda text: [0.1, 0.2, 0.3])


def _semantic_with_mock_index():
    store = SemanticStore()
    mock_index = MagicMock()
    store._index = mock_index
    return store, mock_index


def _match(id, score, metadata):
    """Attr-style match object (mimics pinecone response objects)."""
    m = MagicMock()
    m.id = id
    m.score = score
    m.metadata = metadata
    return m


# ── store() ────────────────────────────────────────────────────────────────────

def test_semantic_store_returns_error_when_unconfigured(monkeypatch):
    import config.settings as settings
    monkeypatch.setattr(settings, "PINECONE_API_KEY", None)
    store = SemanticStore()
    res = store.store("some text")
    assert res["status"] == "error"


def test_semantic_store_returns_error_when_embed_unavailable(monkeypatch):
    monkeypatch.setattr(semantic_mod, "_embed", lambda text: None)
    store, _ = _semantic_with_mock_index()
    res = store.store("some text")
    assert res["status"] == "error"
    assert "embedding" in res["error"]


def test_semantic_store_ok_and_autogenerates_id():
    store, index = _semantic_with_mock_index()
    res = store.store("Hormuz tension", metadata={"corridor": "strait_of_hormuz"})
    assert res["status"] == "ok"
    assert res["id"]  # a uuid was generated
    index.upsert.assert_called_once()


def test_semantic_store_uses_supplied_id():
    store, index = _semantic_with_mock_index()
    res = store.store("text", id="fixed-id")
    assert res["id"] == "fixed-id"


def test_semantic_store_folds_text_into_metadata():
    store, index = _semantic_with_mock_index()
    store.store("Iran blockade", metadata={"corridor": "strait_of_hormuz"})
    vectors = index.upsert.call_args.kwargs["vectors"]
    vec = vectors[0]
    assert vec["values"] == [0.1, 0.2, 0.3]
    assert vec["metadata"]["text"] == "Iran blockade"
    assert vec["metadata"]["corridor"] == "strait_of_hormuz"


def test_semantic_store_returns_error_on_exception():
    store, index = _semantic_with_mock_index()
    index.upsert.side_effect = RuntimeError("pinecone 500")
    res = store.store("text")
    assert res["status"] == "error"
    assert "pinecone 500" in res["error"]


# ── query() ─────────────────────────────────────────────────────────────────────

def test_semantic_query_sorts_by_score_desc():
    # Regression guard: backend may return unsorted matches.
    store, index = _semantic_with_mock_index()
    index.query.return_value = {"matches": [
        _match("b", 0.10, {"text": "low"}),
        _match("a", 0.90, {"text": "high"}),
        _match("c", 0.50, {"text": "mid"}),
    ]}
    hits = store.query("q", top_k=3)
    assert [h["score"] for h in hits] == [0.90, 0.50, 0.10]
    assert hits[0]["metadata"]["text"] == "high"


def test_semantic_query_parses_dict_matches():
    store, index = _semantic_with_mock_index()
    index.query.return_value = {"matches": [
        {"id": "a", "score": 0.8, "metadata": {"text": "hi"}},
    ]}
    hits = store.query("q")
    assert hits == [{"id": "a", "score": 0.8, "metadata": {"text": "hi"}}]


def test_semantic_query_handles_missing_metadata():
    store, index = _semantic_with_mock_index()
    index.query.return_value = {"matches": [{"id": "a", "score": 0.8, "metadata": None}]}
    hits = store.query("q")
    assert hits[0]["metadata"] == {}


def test_semantic_query_returns_empty_on_exception():
    store, index = _semantic_with_mock_index()
    index.query.side_effect = RuntimeError("boom")
    assert store.query("q") == []


def test_semantic_query_returns_empty_when_unconfigured(monkeypatch):
    import config.settings as settings
    monkeypatch.setattr(settings, "PINECONE_API_KEY", None)
    store = SemanticStore()
    assert store.query("q") == []


# ── delete() ────────────────────────────────────────────────────────────────────

def test_semantic_delete_ok():
    store, index = _semantic_with_mock_index()
    res = store.delete("some-id")
    assert res == {"status": "ok", "id": "some-id"}
    index.delete.assert_called_with(ids=["some-id"])


def test_semantic_delete_error_on_exception():
    store, index = _semantic_with_mock_index()
    index.delete.side_effect = RuntimeError("nope")
    res = store.delete("x")
    assert res["status"] == "error"


# ══════════════════════════════════════════════════════════════════════════════
# ProceduralStore  (Supabase client mocked)
# ══════════════════════════════════════════════════════════════════════════════

def _proc_with_mock_client():
    store = ProceduralStore()
    client = MagicMock()
    store._client = client
    return store, client


def _resp(data):
    r = MagicMock()
    r.data = data
    return r


# ── store_skill() ───────────────────────────────────────────────────────────────

def test_store_skill_returns_error_when_unconfigured(monkeypatch):
    import config.settings as settings
    monkeypatch.setattr(settings, "SUPABASE_URL", None)
    monkeypatch.setattr(settings, "SUPABASE_KEY", None)
    store = ProceduralStore()
    assert store.store_skill("s", "agent", {})["status"] == "error"


def test_store_skill_upserts_on_skill_name():
    store, client = _proc_with_mock_client()
    (client.table.return_value
           .upsert.return_value
           .execute.return_value) = _resp([{"id": "sk-1"}])
    res = store.store_skill("hormuz_response", "coordinator", {"steps": ["a"]})
    assert res == {"status": "ok", "id": "sk-1"}
    # upsert called with the conflict key = skill_name
    _, kwargs = client.table.return_value.upsert.call_args
    assert kwargs.get("on_conflict") == "skill_name"
    row = client.table.return_value.upsert.call_args[0][0]
    assert row == {"skill_name": "hormuz_response", "agent": "coordinator",
                   "template": {"steps": ["a"]}}


def test_store_skill_error_on_exception():
    store, client = _proc_with_mock_client()
    client.table.side_effect = RuntimeError("db down")
    res = store.store_skill("s", "a", {})
    assert res["status"] == "error"
    assert "db down" in res["error"]


# ── increment_use() ─────────────────────────────────────────────────────────────

def test_increment_use_success_bumps_both():
    store, client = _proc_with_mock_client()
    # first .select(...).eq(...).execute() → existing counters
    (client.table.return_value
           .select.return_value
           .eq.return_value
           .execute.return_value) = _resp([{"use_count": 3, "success_count": 1}])
    res = store.increment_use("s", success=True)
    assert res["use_count"] == 4
    assert res["success_count"] == 2
    # the update payload matches
    update_arg = client.table.return_value.update.call_args[0][0]
    assert update_arg == {"use_count": 4, "success_count": 2}


def test_increment_use_failure_bumps_use_only():
    store, client = _proc_with_mock_client()
    (client.table.return_value
           .select.return_value
           .eq.return_value
           .execute.return_value) = _resp([{"use_count": 3, "success_count": 1}])
    res = store.increment_use("s", success=False)
    assert res["use_count"] == 4
    assert res["success_count"] == 1


def test_increment_use_skill_not_found():
    store, client = _proc_with_mock_client()
    (client.table.return_value
           .select.return_value
           .eq.return_value
           .execute.return_value) = _resp([])
    res = store.increment_use("ghost")
    assert res["status"] == "error"
    assert "not found" in res["error"]


def test_increment_use_handles_null_counters():
    store, client = _proc_with_mock_client()
    (client.table.return_value
           .select.return_value
           .eq.return_value
           .execute.return_value) = _resp([{"use_count": None, "success_count": None}])
    res = store.increment_use("s", success=True)
    assert res["use_count"] == 1
    assert res["success_count"] == 1


# ── get_skill() ─────────────────────────────────────────────────────────────────

def test_get_skill_returns_row():
    store, client = _proc_with_mock_client()
    (client.table.return_value
           .select.return_value
           .eq.return_value
           .limit.return_value
           .execute.return_value) = _resp([{"skill_name": "s", "use_count": 2}])
    assert store.get_skill("s") == {"skill_name": "s", "use_count": 2}


def test_get_skill_returns_none_when_missing():
    store, client = _proc_with_mock_client()
    (client.table.return_value
           .select.return_value
           .eq.return_value
           .limit.return_value
           .execute.return_value) = _resp([])
    assert store.get_skill("ghost") is None


def test_get_skill_returns_none_on_exception():
    store, client = _proc_with_mock_client()
    client.table.side_effect = RuntimeError("boom")
    assert store.get_skill("s") is None


# ── list_skills() ───────────────────────────────────────────────────────────────

def test_list_skills_orders_by_use_count_and_filters_agent():
    store, client = _proc_with_mock_client()
    chain = client.table.return_value.select.return_value
    chain.eq.return_value = chain
    chain.order.return_value = chain
    chain.limit.return_value = chain
    chain.execute.return_value = _resp([{"skill_name": "a"}, {"skill_name": "b"}])

    rows = store.list_skills(agent="coordinator", limit=10)

    assert rows == [{"skill_name": "a"}, {"skill_name": "b"}]
    chain.eq.assert_called_with("agent", "coordinator")
    chain.order.assert_called_with("use_count", desc=True)
    chain.limit.assert_called_with(10)


def test_list_skills_returns_empty_on_exception():
    store, client = _proc_with_mock_client()
    client.table.side_effect = RuntimeError("boom")
    assert store.list_skills() == []


# ══════════════════════════════════════════════════════════════════════════════
# Distiller  (LLM client mocked; stores are fakes)
# ══════════════════════════════════════════════════════════════════════════════

def _llm_response(content_dict):
    import json as _json
    msg = MagicMock()
    msg.content = _json.dumps(content_dict)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


GOOD_DISTILL = {
    "summary": "Hormuz closed; spot failed; West Africa covered the gap.",
    "key_events": [
        {"event_type": "risk_assessment", "agent": "gri_agent",
         "payload": {"corridor": "strait_of_hormuz", "score": 0.9}, "outcome": "success"},
        {"event_type": "procurement_bid", "agent": "spot_market_agent",
         "payload": {"reason": "no cargo"}, "outcome": "failure"},
    ],
    "candidate_skill": {
        "skill_name": "hormuz_closure_response", "agent": "coordinator",
        "template": {"trigger": "hormuz risk>0.8", "steps": ["source WAF"]},
    },
    "confidence": 0.85,
}


# ── distill() ───────────────────────────────────────────────────────────────────

def test_distill_returns_empty_on_llm_exception(monkeypatch):
    monkeypatch.setattr(distill_mod._client.chat.completions, "create",
                        MagicMock(side_effect=RuntimeError("llm down")))
    d = Distiller()
    res = d.distill({"query": "x"})
    assert res == {"summary": "", "key_events": [], "candidate_skill": None, "confidence": 0.0}


def test_distill_parses_good_response(monkeypatch):
    monkeypatch.setattr(distill_mod._client.chat.completions, "create",
                        MagicMock(return_value=_llm_response(GOOD_DISTILL)))
    d = Distiller()
    res = d.distill({"query": "x"})
    assert res["summary"].startswith("Hormuz")
    assert len(res["key_events"]) == 2
    assert res["candidate_skill"]["skill_name"] == "hormuz_closure_response"
    assert res["confidence"] == 0.85


def test_distill_normalizes_missing_and_garbage_fields(monkeypatch):
    junk = {"confidence": "not-a-number"}  # missing summary/key_events/skill
    monkeypatch.setattr(distill_mod._client.chat.completions, "create",
                        MagicMock(return_value=_llm_response(junk)))
    d = Distiller()
    res = d.distill({"query": "x"})
    assert res["summary"] == ""
    assert res["key_events"] == []
    assert res["candidate_skill"] is None
    assert res["confidence"] == 0.0


# ── persist() ───────────────────────────────────────────────────────────────────

def _fake_stores():
    epi = MagicMock()
    epi.store.return_value = {"status": "ok", "id": "shared-id"}
    sem = MagicMock()
    sem.store.return_value = {"status": "ok", "id": "shared-id"}
    proc = MagicMock()
    proc.store_skill.return_value = {"status": "ok", "id": "sk-1"}
    return epi, sem, proc


def test_persist_writes_events_to_episodic_and_semantic():
    epi, sem, proc = _fake_stores()
    d = Distiller()
    report = d.persist(GOOD_DISTILL, episodic=epi, semantic=sem, procedural=proc)
    assert report["episodic_written"] == 2
    assert report["semantic_written"] == 2
    assert epi.store.call_count == 2
    assert sem.store.call_count == 2


def test_persist_shares_episodic_id_with_semantic():
    epi, sem, proc = _fake_stores()
    d = Distiller()
    d.persist(GOOD_DISTILL, episodic=epi, semantic=sem, procedural=proc)
    # semantic.store called with id=the episodic id
    for call in sem.store.call_args_list:
        assert call.kwargs.get("id") == "shared-id"


def test_persist_writes_skill_when_confident():
    epi, sem, proc = _fake_stores()
    d = Distiller()
    report = d.persist(GOOD_DISTILL, episodic=epi, semantic=sem, procedural=proc)
    assert report["skill_written"] is True
    proc.store_skill.assert_called_once()


def test_persist_skips_skill_when_below_confidence():
    epi, sem, proc = _fake_stores()
    low = {**GOOD_DISTILL, "confidence": 0.5}
    d = Distiller()
    report = d.persist(low, episodic=epi, semantic=sem, procedural=proc)
    assert report["skill_written"] is False
    assert "0.50" in report["skill_skipped_reason"]
    proc.store_skill.assert_not_called()


def test_persist_reports_no_candidate_skill():
    epi, sem, proc = _fake_stores()
    no_skill = {**GOOD_DISTILL, "candidate_skill": None}
    d = Distiller()
    report = d.persist(no_skill, episodic=epi, semantic=sem, procedural=proc)
    assert report["skill_skipped_reason"] == "no candidate_skill"


def test_persist_skips_none_store_legs():
    d = Distiller()
    # all stores None → nothing written, no crash
    report = d.persist(GOOD_DISTILL, episodic=None, semantic=None, procedural=None)
    assert report["episodic_written"] == 0
    assert report["semantic_written"] == 0
    assert report["skill_written"] is False


def test_persist_normalizes_null_string_outcome():
    epi, sem, proc = _fake_stores()
    traj = {"summary": "", "confidence": 0.0, "candidate_skill": None,
            "key_events": [{"event_type": "e", "agent": "a", "payload": {}, "outcome": "null"}]}
    d = Distiller()
    d.persist(traj, episodic=epi, semantic=sem, procedural=proc)
    # episodic.store called with outcome=None (not the string "null")
    assert epi.store.call_args.kwargs.get("outcome") is None


def test_persist_collects_written_ids():
    epi, sem, proc = _fake_stores()
    d = Distiller()
    report = d.persist(GOOD_DISTILL, episodic=epi, semantic=sem, procedural=proc)
    assert report["written_ids"] == ["shared-id", "shared-id"]


# ══════════════════════════════════════════════════════════════════════════════
# XMemory  (facade — underlying stores replaced with fakes)
# ══════════════════════════════════════════════════════════════════════════════

def _xmemory_with_fakes():
    xm = XMemory()
    xm.episodic = MagicMock()
    xm.semantic = MagicMock()
    xm.procedural = MagicMock()
    xm.distiller = MagicMock()
    return xm


# ── Working scratchpad delegation ───────────────────────────────────────────────

def test_scratch_roundtrip_and_snapshot():
    xm = XMemory()
    assert xm.scratch_set("k", "v", 5) is True
    assert xm.scratch_get("k") == "v"
    assert xm.working_snapshot() == {"k": "v"}


def test_reset_working_clears_scratchpad():
    xm = XMemory()
    xm.scratch_set("k", "v", 5)
    xm.reset_working()
    assert xm.working_snapshot() == {}


# ── remember() dual write ───────────────────────────────────────────────────────

def test_remember_dual_writes_with_shared_id():
    xm = _xmemory_with_fakes()
    xm.episodic.store.return_value = {"status": "ok", "id": "abc"}
    xm.semantic.store.return_value = {"status": "ok", "id": "abc"}
    res = xm.remember("risk_assessment", "gri_agent", {"score": 0.9}, outcome="success")
    assert res == {"episodic_id": "abc", "semantic_id": "abc", "status": "ok"}
    # semantic used the episodic id
    assert xm.semantic.store.call_args.kwargs["id"] == "abc"


def test_remember_derives_text_from_payload_summary():
    xm = _xmemory_with_fakes()
    xm.episodic.store.return_value = {"status": "ok", "id": "abc"}
    xm.semantic.store.return_value = {"status": "ok", "id": "abc"}
    xm.remember("e", "a", {"summary": "the summary", "score": 0.1})
    assert xm.semantic.store.call_args[0][0] == "the summary"


def test_remember_explicit_text_overrides_payload():
    xm = _xmemory_with_fakes()
    xm.episodic.store.return_value = {"status": "ok", "id": "abc"}
    xm.semantic.store.return_value = {"status": "ok", "id": "abc"}
    xm.remember("e", "a", {"summary": "ignored"}, text="explicit text")
    assert xm.semantic.store.call_args[0][0] == "explicit text"


def test_remember_partial_status_on_episodic_failure():
    xm = _xmemory_with_fakes()
    xm.episodic.store.return_value = {"status": "error", "error": "db down"}
    xm.semantic.store.return_value = {"status": "ok", "id": "sem-only"}
    res = xm.remember("e", "a", {"x": 1})
    assert res["status"] == "partial"
    assert res["episodic_id"] is None
    assert res["semantic_id"] == "sem-only"


# ── recall_events() decay ranking ───────────────────────────────────────────────

def test_recall_events_uses_payload_event_type_for_decay():
    # Two same-age rows; the war_conflict one (180d half-life) must outrank the
    # weather one (7d) at equal intensity — proves payload event_type drives decay.
    from datetime import datetime, timezone, timedelta
    old = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    xm = _xmemory_with_fakes()
    xm.episodic.query.return_value = [
        {"event_type": "risk_assessment", "timestamp": old,
         "payload": {"event_type": "weather_disruption", "score": 0.8}},
        {"event_type": "risk_assessment", "timestamp": old,
         "payload": {"event_type": "war_conflict", "score": 0.8}},
    ]
    rows = xm.recall_events(agent="gri_agent", decay=True)
    assert rows[0]["payload"]["event_type"] == "war_conflict"
    assert rows[0]["decayed_relevance"] > rows[1]["decayed_relevance"]


def test_recall_events_decay_false_preserves_order():
    xm = _xmemory_with_fakes()
    original = [{"event_type": "e", "timestamp": "", "payload": {"score": 0.1}},
                {"event_type": "e", "timestamp": "", "payload": {"score": 0.9}}]
    xm.episodic.query.return_value = list(original)
    rows = xm.recall_events(decay=False)
    assert [r["payload"]["score"] for r in rows] == [0.1, 0.9]
    assert "decayed_relevance" not in rows[0]


def test_recall_events_empty_returns_empty():
    xm = _xmemory_with_fakes()
    xm.episodic.query.return_value = []
    assert xm.recall_events() == []


def test_recall_similar_delegates_to_semantic():
    xm = _xmemory_with_fakes()
    xm.semantic.query.return_value = [{"id": "1", "score": 0.9, "metadata": {}}]
    hits = xm.recall_similar("hormuz", top_k=2)
    assert hits == [{"id": "1", "score": 0.9, "metadata": {}}]
    xm.semantic.query.assert_called_with("hormuz", top_k=2, filter=None)


# ── Procedural delegation ───────────────────────────────────────────────────────

def test_record_skill_use_delegates():
    xm = _xmemory_with_fakes()
    xm.procedural.increment_use.return_value = {"status": "ok", "use_count": 1}
    xm.record_skill_use("s", success=True)
    xm.procedural.increment_use.assert_called_with("s", success=True)


# ── distill_run() learning loop ─────────────────────────────────────────────────

def test_distill_run_builds_trajectory_when_none():
    xm = _xmemory_with_fakes()
    xm.episodic.recent.return_value = [{"id": "e1"}]
    xm.distiller.distill.return_value = {"summary": "s", "key_events": [],
                                         "candidate_skill": None, "confidence": 0.0}
    xm.distiller.persist.return_value = {"episodic_written": 0}
    xm.scratch_set("k", "v", 5)
    xm.distill_run()
    # trajectory passed to distill contains working snapshot + recent episodic
    traj = xm.distiller.distill.call_args[0][0]
    assert traj["working_memory"] == {"k": "v"}
    assert traj["episodic_events"] == [{"id": "e1"}]


def test_distill_run_uses_supplied_trajectory():
    xm = _xmemory_with_fakes()
    xm.distiller.distill.return_value = {"summary": "", "key_events": [],
                                         "candidate_skill": None, "confidence": 0.0}
    xm.distiller.persist.return_value = {}
    supplied = {"working_memory": {}, "episodic_events": []}
    xm.distill_run(trajectory=supplied)
    assert xm.distiller.distill.call_args[0][0] is supplied
    # persist routed into the facade's own stores
    _, kwargs = xm.distiller.persist.call_args
    assert kwargs["episodic"] is xm.episodic
    assert kwargs["semantic"] is xm.semantic
    assert kwargs["procedural"] is xm.procedural
