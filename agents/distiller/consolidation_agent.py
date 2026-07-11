"""
Consolidation Agent — the memory janitor of the distiller pod.

The Experience Distiller keeps ADDING lessons; left alone, memory rots three ways:
near-identical lessons pile up, long-dead events keep surfacing in recall, and
unproven skills sit indistinguishable from battle-tested ones. This agent does the
periodic housekeeping — fully DETERMINISTIC, no LLM (it's bookkeeping, not judgment):

  1. MERGE  — de-duplicate the semantic layer. Two lessons that mean almost the same
     thing should not both surface as separate precedents. Keep the canonical
     (oldest) one; tombstone the rest.
  2. PRUNE  — retire faded memories. An event whose decayed relevance has fallen
     below the floor (its crisis is long over and unreinforced) is tombstoned from
     the SEMANTIC layer so recall stops surfacing it as "current".
  3. PROMOTE — mark skills that have proven themselves. A recipe with enough uses and
     a good success ratio is stamped `status: "proven"`, so agents can trust it over
     an unproven candidate.

Append-only invariant (critical): EPISODIC memory is the tamper-proof audit trail —
"we believed X on day 1" stays true even after X faded or was superseded, so this
agent NEVER deletes an episodic row. Only the SEMANTIC index (the "what's true now?"
recall layer) is tombstoned. This is exactly the decay/supersession split in
CLAUDE.md: don't rewrite history, change what recall surfaces as present.

Idempotent by construction: a tombstoned vector delete is harmless to repeat, and an
already-"proven" skill is skipped — so running consolidation twice changes nothing
the second time. Meant to run on a cadence (like the twin loop), not per query.
Best-effort throughout: the underlying stores return status dicts / [] on failure,
so nothing here raises.
"""
import json
from datetime import datetime, timezone

from graph.eib_state import EnergyIntelligenceBoard
from memory.xmemory import XMemory

_xmemory = XMemory()

# Two lessons whose cosine similarity is at/above this are treated as the same
# precedent — the later one is a duplicate and tombstoned from semantic recall.
MERGE_SIMILARITY_THRESHOLD = 0.95

# An event whose decayed relevance has fallen below this is effectively dead: its
# crisis is long over and unreinforced, so it should stop surfacing in recall.
PRUNE_RELEVANCE_FLOOR = 0.05

# A skill earns "proven" once it has been applied enough times with a good hit rate.
PROMOTE_MIN_USES = 3
PROMOTE_MIN_SUCCESS_RATIO = 0.6

# How many recent events one consolidation pass considers.
_EVENT_SCAN_LIMIT = 100


def _text_of(row: dict) -> str:
    """The text a memory was embedded under — same derivation as the write path
    (summary → text → json), so a merge query re-embeds the identical string."""
    payload = row.get("payload") or {}
    if isinstance(payload, dict):
        return (payload.get("summary") or payload.get("text")
                or json.dumps(payload, default=str, sort_keys=True))
    return str(payload)


def _merge(events: list[dict], tombstoned: set) -> int:
    """De-duplicate the semantic layer. Process oldest-first and keep the first of
    each near-identical cluster; a later event that is highly similar to an
    already-kept one is a duplicate → tombstone its semantic vector. Episodic
    untouched. Returns the number merged."""
    kept: set = set()
    merged = 0
    # Oldest first, so the canonical survivor is the earliest occurrence.
    for row in sorted(events, key=lambda r: r.get("timestamp", "")):
        rid = row.get("id")
        if not rid or rid in tombstoned:
            continue
        matches = _xmemory.recall_similar(_text_of(row), top_k=5) or []
        is_dup = any(
            m.get("id") in kept and float(m.get("score", 0.0)) >= MERGE_SIMILARITY_THRESHOLD
            for m in matches
        )
        if is_dup:
            _xmemory.semantic.delete(rid)
            tombstoned.add(rid)
            merged += 1
        else:
            kept.add(rid)
    return merged


def _prune(events: list[dict], tombstoned: set) -> int:
    """Tombstone the semantic vectors of faded events (decayed_relevance below the
    floor). The episodic row stays — history is never rewritten. Returns pruned."""
    pruned = 0
    for row in events:
        rid = row.get("id")
        if not rid or rid in tombstoned:
            continue
        if float(row.get("decayed_relevance", 1.0)) < PRUNE_RELEVANCE_FLOOR:
            _xmemory.semantic.delete(rid)
            tombstoned.add(rid)
            pruned += 1
    return pruned


def _promote(skills: list[dict]) -> list[str]:
    """Stamp `status: "proven"` on skills that have earned it. Already-proven skills
    are skipped (idempotent). Returns the names promoted this pass."""
    promoted: list[str] = []
    for skill in skills:
        template = skill.get("template") or {}
        if template.get("status") == "proven":
            continue
        uses = int(skill.get("use_count", 0) or 0)
        successes = int(skill.get("success_count", 0) or 0)
        if uses < PROMOTE_MIN_USES:
            continue
        if (successes / uses) < PROMOTE_MIN_SUCCESS_RATIO:
            continue
        name = skill.get("skill_name", "unnamed_skill")
        res = _xmemory.procedural.store_skill(
            name, skill.get("agent", "distiller"), {**template, "status": "proven"}
        )
        if res.get("status") == "ok":
            promoted.append(name)
    return promoted


def consolidate(events: list[dict] | None = None,
                skills: list[dict] | None = None) -> dict:
    """Run one housekeeping pass: merge duplicates, prune faded events, promote proven
    skills. `events`/`skills` are injectable for testing; otherwise pulled from
    memory (decayed episodic recall + the skill cookbook). Never raises."""
    if events is None:
        events = _xmemory.recall_events(limit=_EVENT_SCAN_LIMIT, decay=True) or []
    if skills is None:
        skills = _xmemory.list_skills() or []

    tombstoned: set = set()  # shared so merge + prune never double-delete/double-count
    merged = _merge(events, tombstoned)
    pruned = _prune(events, tombstoned)
    promoted = _promote(skills)

    return {
        "examined_events": len(events),
        "examined_skills": len(skills),
        "merged":          merged,
        "pruned":          pruned,
        "promoted":        len(promoted),
        "promoted_skills": promoted,
    }


def consolidation_node(state: EnergyIntelligenceBoard) -> dict:
    """Agent-shaped entry point. Consolidation is store-wide housekeeping, not
    per-query work, so it ignores most of state and reports the pass in the audit.
    Meant to run on a cadence (async), independent of the answer pipeline."""
    now = datetime.now(timezone.utc).isoformat()
    report = consolidate()
    audit = [{
        "agent":     "consolidation_agent",
        "action":    "consolidate",
        **report,
        "timestamp": now,
    }]
    return {
        "current_agent": "consolidation_agent",
        "audit_trail":   audit,
    }
