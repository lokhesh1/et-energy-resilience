"""
Tests for agents/distiller/consolidation_agent.py — the memory janitor.

Deterministic housekeeping, so these are exact-behaviour tests (no LLM):
  * MERGE   — near-identical semantic entries collapse to the oldest; only the
    duplicate is tombstoned, and only in semantic (episodic never touched);
  * PRUNE   — faded events (decayed_relevance below floor) are tombstoned in
    semantic; live events are kept; merge + prune never double-delete;
  * PROMOTE — skills over the use/success bar get stamped "proven"; under-bar and
    already-proven skills are left alone (idempotent);
  * the append-only invariant: episodic is never deleted;
  * the node wrapper reports the pass in the audit.
"""
from unittest.mock import MagicMock, patch

import pytest

import agents.distiller.consolidation_agent as ca
from agents.distiller.consolidation_agent import consolidate, consolidation_node


def _ev(id, ts, relevance=1.0, summary=None):
    return {"id": id, "timestamp": ts, "decayed_relevance": relevance,
            "payload": {"summary": summary or f"event {id}"}}


def _skill(name, uses, successes, status=None):
    template = {"trigger": "x", "steps": ["a"]}
    if status:
        template["status"] = status
    return {"skill_name": name, "agent": "distiller", "template": template,
            "use_count": uses, "success_count": successes}


class _Mem:
    """A fake XMemory exposing only what the agent touches, recording mutations."""
    def __init__(self, similar=None):
        self._similar = similar or {}          # text/id → matches list
        self.deleted = []                      # semantic ids tombstoned
        self.promoted = []                     # (name, template) upserts
        self.semantic = MagicMock()
        self.semantic.delete.side_effect = lambda i: (self.deleted.append(i)
                                                      or {"status": "ok", "id": i})
        self.procedural = MagicMock()
        self.procedural.store_skill.side_effect = self._store_skill

    def _store_skill(self, name, agent, template):
        self.promoted.append((name, template))
        return {"status": "ok", "id": name}

    def recall_similar(self, text, top_k=5, filter=None):
        return self._similar.get(text, [])


@pytest.fixture
def mem(monkeypatch):
    m = _Mem()
    monkeypatch.setattr(ca, "_xmemory", m)
    return m


# ── MERGE ────────────────────────────────────────────────────────────────────────

def test_merge_tombstones_the_later_duplicate_only(mem):
    # e2 is a near-duplicate of the older e1 → e2's vector is tombstoned, e1 kept.
    mem._similar = {
        "event e1": [{"id": "e1", "score": 1.0}],
        "event e2": [{"id": "e1", "score": 0.97}, {"id": "e2", "score": 1.0}],
    }
    events = [_ev("e1", "2026-07-01T00:00:00+00:00"),
              _ev("e2", "2026-07-02T00:00:00+00:00")]
    report = consolidate(events=events, skills=[])
    assert report["merged"] == 1
    assert mem.deleted == ["e2"]           # the newer duplicate, not the canonical


def test_merge_keeps_distinct_events(mem):
    mem._similar = {
        "event e1": [{"id": "e1", "score": 1.0}],
        "event e2": [{"id": "e2", "score": 1.0}, {"id": "e1", "score": 0.40}],
    }
    events = [_ev("e1", "2026-07-01T00:00:00+00:00"),
              _ev("e2", "2026-07-02T00:00:00+00:00")]
    report = consolidate(events=events, skills=[])
    assert report["merged"] == 0
    assert mem.deleted == []


def test_merge_below_threshold_is_not_a_duplicate(mem):
    # 0.94 < 0.95 threshold → kept.
    mem._similar = {
        "event a": [{"id": "a", "score": 1.0}],
        "event b": [{"id": "a", "score": 0.94}, {"id": "b", "score": 1.0}],
    }
    events = [_ev("a", "2026-07-01T00:00:00+00:00", summary="a"),
              _ev("b", "2026-07-02T00:00:00+00:00", summary="b")]
    assert consolidate(events=events, skills=[])["merged"] == 0


# ── PRUNE ────────────────────────────────────────────────────────────────────────

def test_prune_tombstones_faded_events(mem):
    events = [_ev("live", "2026-07-09T00:00:00+00:00", relevance=0.8),
              _ev("dead", "2026-01-01T00:00:00+00:00", relevance=0.01)]
    report = consolidate(events=events, skills=[])
    assert report["pruned"] == 1
    assert mem.deleted == ["dead"]         # only the faded one


def test_merge_and_prune_do_not_double_delete(mem):
    # An event that is BOTH a duplicate and faded is tombstoned once, counted once.
    mem._similar = {
        "event e1": [{"id": "e1", "score": 1.0}],
        "event e2": [{"id": "e1", "score": 0.99}, {"id": "e2", "score": 1.0}],
    }
    events = [_ev("e1", "2026-07-01T00:00:00+00:00", relevance=0.9),
              _ev("e2", "2026-07-02T00:00:00+00:00", relevance=0.01)]
    report = consolidate(events=events, skills=[])
    assert report["merged"] == 1
    assert report["pruned"] == 0           # e2 already tombstoned by merge
    assert mem.deleted == ["e2"]           # exactly once


# ── PROMOTE ──────────────────────────────────────────────────────────────────────

def test_promote_stamps_proven_on_qualifying_skill(mem):
    skills = [_skill("hormuz_playbook", uses=5, successes=4)]   # 0.8 >= 0.6
    report = consolidate(events=[], skills=skills)
    assert report["promoted"] == 1
    assert report["promoted_skills"] == ["hormuz_playbook"]
    name, template = mem.promoted[0]
    assert template["status"] == "proven"
    assert template["trigger"] == "x"       # original body preserved


def test_promote_skips_low_use_and_low_ratio(mem):
    skills = [_skill("too_few", uses=2, successes=2),        # under min uses
              _skill("too_weak", uses=5, successes=1)]       # 0.2 ratio < 0.6
    report = consolidate(events=[], skills=skills)
    assert report["promoted"] == 0
    assert mem.promoted == []


def test_promote_is_idempotent_on_already_proven(mem):
    skills = [_skill("veteran", uses=10, successes=9, status="proven")]
    report = consolidate(events=[], skills=skills)
    assert report["promoted"] == 0
    assert mem.promoted == []               # not re-stamped


# ── report + node ────────────────────────────────────────────────────────────────

def test_report_counts_examined(mem):
    report = consolidate(events=[_ev("a", "2026-07-01T00:00:00+00:00")],
                         skills=[_skill("s", 1, 1)])
    assert report["examined_events"] == 1
    assert report["examined_skills"] == 1


def test_node_reports_pass_in_audit(mem):
    mem.recall_events = MagicMock(return_value=[])
    mem.list_skills = MagicMock(return_value=[])
    out = consolidation_node({})
    entry = out["audit_trail"][0]
    assert out["current_agent"] == "consolidation_agent"
    assert entry["agent"] == "consolidation_agent"
    assert entry["action"] == "consolidate"
    assert "merged" in entry and "pruned" in entry and "promoted" in entry


def test_consolidate_pulls_from_memory_when_not_injected(mem):
    mem.recall_events = MagicMock(return_value=[_ev("x", "2026-07-01T00:00:00+00:00")])
    mem.list_skills = MagicMock(return_value=[_skill("s", 4, 4)])
    report = consolidate()
    mem.recall_events.assert_called_once()
    mem.list_skills.assert_called_once()
    assert report["examined_events"] == 1
    assert report["promoted"] == 1
