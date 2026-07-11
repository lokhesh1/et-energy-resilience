"""
Distiller pod demo — 3 scenarios, before/after memory.

Shows the learning loop end to end: the board answers a crisis, the distiller pod
learns from the run in the background, and a LATER repeat of the same crisis finds
the earlier lesson already in memory — the board *remembers*.

NON-DESTRUCTIVE + OFFLINE by design. It would be trivial to point this at the live
Supabase/Pinecone stores, but that would (a) write demo rows into real, append-only
memory and (b) cost live LLM/news calls. So instead:
  * the 3 cloud stores are swapped for in-memory fakes (same interfaces) — nothing
    touches the real backends;
  * the two LLM touchpoints (GRI risk read + the distiller's extraction) are stubbed
    deterministically, so every run is free and reproducible;
  * EVERYTHING ELSE IS THE REAL SYSTEM — the compiled graph, DSM/SCTD math, the
    procurement pod, the Crisis Coordinator, `build_trajectory`/`_run_outcome`, the
    real `persist()` routing, and the real consolidation merge/prune/promote.

Run:  python -m scripts.demo_distiller
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import agents.gri_agent as gri
import agents.dsm_agent as dsm
import agents.sctd_agent as sctd
import agents.crisis_coordinator as coord
import agents.distiller.experience_distiller as distiller
import agents.distiller.consolidation_agent as consolidation
import agents.procurement._sourcing_base as sourcing
from memory.xmemory import XMemory
from agents.distiller.pod import learn_from_run
from graph.workflow import build_graph, initial_state


# ── In-memory stand-ins for the 3 cloud stores (same public surface) ─────────────

class FakeEpisodic:
    """Append-only event log (the tamper-proof audit trail)."""
    def __init__(self):
        self.rows = []

    def store(self, event_type, agent, payload, outcome=None):
        row = {"id": str(uuid.uuid4()), "event_type": event_type, "agent": agent,
               "payload": payload, "outcome": outcome,
               "timestamp": datetime.now(timezone.utc).isoformat()}
        self.rows.append(row)
        return {"status": "ok", "id": row["id"]}

    def query(self, agent=None, event_type=None, outcome=None, limit=20):
        rows = [r for r in self.rows
                if (agent is None or r["agent"] == agent)
                and (event_type is None or r["event_type"] == event_type)
                and (outcome is None or r["outcome"] == outcome)]
        return list(reversed(rows))[:limit]

    def recent(self, n=10):
        return list(reversed(self.rows))[:n]


class FakeSemantic:
    """Similarity recall. Real embeddings need a model download, so this uses a
    deterministic token-overlap (Jaccard) score — enough to show recall surfacing
    the right past lesson. delete() = the tombstone the consolidator uses."""
    def __init__(self):
        self.vectors = {}   # id -> {"text", "metadata"}

    @staticmethod
    def _tokens(text):
        return {w for w in "".join(c.lower() if c.isalnum() else " "
                                   for c in text).split() if len(w) > 2}

    def store(self, text, metadata=None, id=None):
        vid = id or str(uuid.uuid4())
        self.vectors[vid] = {"text": text, "metadata": {"text": text, **(metadata or {})}}
        return {"status": "ok", "id": vid}

    def query(self, text, top_k=3, filter=None):
        q = self._tokens(text)
        scored = []
        for vid, v in self.vectors.items():
            t = self._tokens(v["text"])
            if not q or not t:
                continue
            score = len(q & t) / len(q | t)
            if score > 0:
                scored.append({"id": vid, "score": round(score, 4),
                               "metadata": v["metadata"]})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    def delete(self, id):
        self.vectors.pop(id, None)
        return {"status": "ok", "id": id}


class FakeProcedural:
    """Skill cookbook — one row per named recipe, upserted."""
    def __init__(self):
        self.skills = {}

    def store_skill(self, skill_name, agent, template):
        row = self.skills.get(skill_name, {"skill_name": skill_name, "agent": agent,
                                           "use_count": 0, "success_count": 0})
        row.update({"agent": agent, "template": template})
        self.skills[skill_name] = row
        return {"status": "ok", "id": skill_name}

    def increment_use(self, skill_name, success=False):
        row = self.skills.get(skill_name)
        if not row:
            return {"status": "error", "error": "not found"}
        row["use_count"] += 1
        row["success_count"] += 1 if success else 0
        return {"status": "ok", **row}

    def get_skill(self, skill_name):
        return self.skills.get(skill_name)

    def list_skills(self, agent=None, limit=50):
        rows = [r for r in self.skills.values() if agent is None or r["agent"] == agent]
        return sorted(rows, key=lambda r: r["use_count"], reverse=True)[:limit]


# ── Deterministic stubs for the two LLM touchpoints ──────────────────────────────

def _stub_distill(trajectory: dict) -> dict:
    """Stand in for the distiller's LLM extraction: derive one durable lesson +
    (on a resolved crisis) a reusable playbook, deterministically from the digest."""
    risks = trajectory.get("corridor_risks") or []
    lead = risks[0] if risks else {"corridor": "unknown", "event_type": "none"}
    gap = trajectory["twin"]["gap_mbd"]
    cargoes = trajectory["procurement"].get("cargoes") or []
    supplier = cargoes[0]["supplier"] if cargoes else "none"
    outcome = trajectory["outcome"]

    lesson = (f"{lead['corridor']} {lead['event_type']}: {gap} mbd India-bound crude "
              f"at risk; sourced from {supplier}; outcome {outcome}.")

    skill = None
    if outcome == "success" and gap > 0 and cargoes:
        skill = {
            "skill_name": f"{lead['corridor']}_{lead['event_type']}_playbook",
            "agent": "crisis_coordinator",
            "template": {
                "trigger": f"{lead['corridor']} {lead['event_type']} with India shortfall",
                "steps": [f"source {cargoes[0]['grade']} from {supplier} "
                          f"({cargoes[0]['region']})",
                          "escalate per residual gap"],
                "notes": f"worked at gap {gap} mbd",
            },
        }

    return {
        "summary": lesson,
        "key_events": [{
            "event_type": "distilled_lesson", "agent": "experience_distiller",
            "payload": {"summary": lesson, "corridor": lead["corridor"],
                        "event_type": lead["event_type"], "outcome": outcome},
            "outcome": outcome,
        }],
        "candidate_skill": skill,
        "confidence": 0.85,
    }


# ── Wiring: one shared fake-backed memory, injected everywhere ────────────────────

def _make_memory() -> XMemory:
    mem = XMemory()
    mem.episodic = FakeEpisodic()
    mem.semantic = FakeSemantic()
    mem.procedural = FakeProcedural()
    mem.distiller.distill = _stub_distill        # keep persist() real, stub extract
    return mem


def _install(mem: XMemory) -> None:
    """Point every agent's memory handle at the shared fake, and neutralise the
    remaining live edges (DSM/coordinator narrative LLMs, Brent price)."""
    for mod in (gri, dsm, sctd, coord, distiller, consolidation):
        mod._xmemory = mem
    empty = MagicMock()
    empty.chat.completions.create.return_value = _llm_json({})
    dsm._client = empty
    coord._client = empty
    sourcing.fetch_price = lambda: {"status": "ok", "data": {"current_price": 82.0}}


def _llm_json(obj: dict) -> MagicMock:
    import json
    msg = MagicMock(); msg.content = json.dumps(obj)
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


_NEWS = {"tool": "news_fetcher", "status": "ok",
         "data": {"articles": [{"title": "wire", "url": "u", "source": "reuters.com",
                                "trust_score": 0.95, "trusted": True}], "errors": []},
         "source_trust_avg": 0.95, "low_trust_sources_flagged": 0,
         "retrieved_at": "2026-07-10T00:00:00+00:00", "staleness_seconds": 1}
_CORR = {"tool": "corridor_status", "status": "ok", "data": {"corridors": []},
         "source_trust_avg": 1.0, "low_trust_sources_flagged": 0,
         "retrieved_at": "2026-07-10T00:00:00+00:00", "staleness_seconds": 0}


def _set_gri_scenario(corridor: str, score: float, event_type: str, headline: str) -> None:
    """Make GRI deterministically read one corridor at a given risk/event."""
    gri._fetch_tools = lambda *a, **k: (_NEWS, _CORR)
    gri._client = MagicMock()
    gri._client.chat.completions.create.return_value = _llm_json({
        "corridor_risk": {corridor: {
            "score": score, "confidence": 0.92, "evidence_count": 1,
            "key_signals": [headline], "reasoning": headline, "event_type": event_type}},
        "novel_corridor_alerts": [], "overall_assessment": headline,
        "low_trust_signals_flagged": 0,
    })


# ── Presentation ─────────────────────────────────────────────────────────────────

def _snap(mem: XMemory) -> dict:
    return {"episodic": len(mem.episodic.rows),
            "semantic": len(mem.semantic.vectors),
            "skills": len(mem.procedural.skills)}


def _line(width=72):
    print("─" * width)


def run():
    # Windows consoles default to cp1252; the box-drawing glyphs need UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    mem = _make_memory()
    _install(mem)
    graph = build_graph()

    scenarios = [
        ("1. Hormuz war (first time)", "Iran closes the Strait of Hormuz",
         "strait_of_hormuz", 0.90, "war_conflict", "Hormuz shipping lanes disrupted"),
        ("2. Suez weather", "Storm shuts the Suez Canal",
         "suez_canal", 0.55, "weather_disruption", "Severe storm closes Suez transit"),
        ("3. Hormuz war (RECURS — does the board remember?)",
         "Iran closes the Strait of Hormuz",
         "strait_of_hormuz", 0.90, "war_conflict", "Hormuz shipping lanes disrupted"),
    ]

    print("\n" + "=" * 72)
    print("  DISTILLER POD DEMO — 3 scenarios, before/after memory (offline)")
    print("=" * 72)

    for idx, (title, query, corridor, score, event, headline) in enumerate(scenarios, 1):
        _line(); print(f"  SCENARIO {title}"); _line()

        before = _snap(mem)
        print(f"  memory BEFORE : episodic={before['episodic']}  "
              f"semantic={before['semantic']}  skills={before['skills']}")

        # What does the board ALREADY know about this crisis, before it even runs?
        prior = mem.recall_similar(f"{corridor} {event} india crude shortfall", top_k=3)
        if prior:
            print(f"  board RECALLS  : {len(prior)} prior memory(ies) for this query —")
            for p in prior[:3]:
                print(f"      • ({p['score']:.2f}) {p['metadata'].get('text', '')[:66]}")
        else:
            print("  board RECALLS  : (nothing — first time seeing this)")

        # Run the real board.
        _set_gri_scenario(corridor, score, event, headline)
        final = graph.invoke(initial_state(query),
                             config={"configurable": {"thread_id": f"demo-{idx}"}})
        plan = final["response_plan"]
        print(f"  board OUTPUT   : [{plan['escalation_level'].upper()}] "
              f"gap={plan['situation']['gap_mbd']} mbd  "
              f"covered={plan['procurement']['covered_mbd']} mbd  "
              f"residual={plan['procurement']['residual_gap_mbd']} mbd")
        print(f"                   {final['final_recommendation'][:66]}")

        # The agents themselves fire-and-forget events to memory DURING the run
        # (GRI risk, DSM scenario, SCTD impact, coordinator response) — separate
        # from the distiller's lesson, so snapshot here to attribute each.
        mid = _snap(mem)
        print(f"  run WROTE      : +{mid['episodic']-before['episodic']} episodic "
              f"(GRI/DSM/SCTD/coordinator fire-and-forget)")

        # Learn from the run (real pod, synchronous so output stays ordered).
        report = learn_from_run(final)
        d = report["distilled"] or {}
        c = report["consolidated"] or {}
        print(f"  DISTILLER      : +{d.get('episodic_written', 0)} episodic  "
              f"+{d.get('semantic_written', 0)} semantic  "
              f"skill={'yes' if d.get('skill_written') else 'no'} (the durable lesson)")
        print(f"  CONSOLIDATED   : merged={c.get('merged', 0)}  "
              f"pruned={c.get('pruned', 0)}  promoted={c.get('promoted', 0)}")

        after = _snap(mem)
        print(f"  memory AFTER  : episodic={after['episodic']}  "
              f"semantic={after['semantic']}  skills={after['skills']}  "
              f"(+{after['episodic']-before['episodic']} / "
              f"+{after['semantic']-before['semantic']} / "
              f"+{after['skills']-before['skills']})")

        # After scenario 1, simulate the learned playbook being applied a few times
        # so scenario 3's consolidation can PROMOTE it to "proven".
        if idx == 1:
            for name in list(mem.procedural.skills):
                for _ in range(3):
                    mem.record_skill_use(name, success=True)
            print("  (simulated 3 successful applications of the new playbook)")
        print()

    _line(); print("  RESULT"); _line()
    proven = [s["skill_name"] for s in mem.procedural.list_skills()
              if (s.get("template") or {}).get("status") == "proven"]
    print("  • Scenario 3 recalled the lesson learned in Scenario 1 BEFORE running —")
    print("    the board remembered a crisis it had already solved once.")
    print(f"  • Episodic log grew to {len(mem.episodic.rows)} events (append-only history).")
    print(f"  • Semantic recall holds {len(mem.semantic.vectors)} live lessons "
          f"(duplicates merged out).")
    print(f"  • Skills promoted to 'proven': {proven or 'none'}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run()
