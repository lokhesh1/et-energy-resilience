"""
Distiller — turns a completed run's raw trajectory into durable learnings.

A run produces a lot of noise (news fetched, scores, bids, failures). Distillation
uses an LLM to read that trajectory and extract only what's worth keeping:
  - key_events    → written to episodic (log) + semantic (searchable by meaning)
  - candidate_skill → a reusable recipe for the procedural cookbook (if one emerged)

This is the "system learns over time" engine. It is deliberately conservative: a
skill is only proposed when the LLM is confident, and it enters the cookbook
UNPROVEN (use_count=0) — it earns trust later via ProceduralStore.increment_use.

Design: LLM-resilient (bad/failed response → empty-but-valid result, never raises).
Stores are injected into persist(), not imported — keeps this decoupled + testable.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from openai import OpenAI

from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DISTILLER_MODEL

_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# Only propose a skill when the LLM is at least this confident it generalises.
SKILL_CONFIDENCE_THRESHOLD = 0.7

_SYSTEM_PROMPT = """You are the Experience Distiller for an energy crisis intelligence board.

You read the raw trajectory of one completed run and extract only durable learnings.
Be conservative — do NOT invent a reusable skill from a one-off coincidence.

Rules:
- key_events: only the handful of events future runs would benefit from recalling.
- candidate_skill: propose one ONLY if a genuinely reusable strategy emerged;
  otherwise return null. A skill needs a clear trigger and repeatable steps.
- confidence reflects how strongly the candidate_skill generalises (0.0-1.0).
- Respond with valid JSON only — no prose outside the JSON object."""


def _empty_result() -> dict:
    return {"summary": "", "key_events": [], "candidate_skill": None, "confidence": 0.0}


def _build_user_prompt(trajectory: dict) -> str:
    return f"""RUN TRAJECTORY (working memory snapshot + episodic events):
{json.dumps(trajectory, indent=2, default=str)[:12000]}

Extract the durable learnings. Return this exact JSON schema:
{{
  "summary": "<one paragraph: what happened this run>",
  "key_events": [
    {{
      "event_type": "<short label>",
      "agent": "<agent name>",
      "payload": {{"...": "..."}},
      "outcome": "success | failure | null"
    }}
  ],
  "candidate_skill": {{
    "skill_name": "<snake_case unique name>",
    "agent": "<owning agent>",
    "template": {{
      "trigger": "<when to apply>",
      "steps": ["<step>", "..."],
      "notes": "<caveats>"
    }}
  }},
  "confidence": <float 0.0-1.0>
}}
Set candidate_skill to null if no reusable strategy emerged."""


class Distiller:
    def __init__(self, model: str = DISTILLER_MODEL) -> None:
        self._model = model

    # ── Extract ────────────────────────────────────────────────────────────────

    def distill(self, trajectory: dict) -> dict:
        """
        Ask the LLM to extract learnings from a run trajectory.
        Never raises — on any failure returns an empty-but-valid result.
        """
        try:
            resp = _client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(trajectory)},
                ],
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
        except Exception:
            return _empty_result()

        # Normalise to guarantee shape regardless of what the LLM returned.
        result = _empty_result()
        result["summary"] = parsed.get("summary", "") or ""
        result["key_events"] = parsed.get("key_events") or []
        result["candidate_skill"] = parsed.get("candidate_skill") or None
        try:
            result["confidence"] = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            result["confidence"] = 0.0
        return result

    # ── Route into memory ──────────────────────────────────────────────────────

    def persist(
        self,
        distilled: dict,
        episodic=None,
        semantic=None,
        procedural=None,
    ) -> dict:
        """
        Route distilled learnings into the memory stores. Stores are injected
        (any may be None → that leg is skipped). Never raises.

        Returns a report: counts of what was written + skill decision.
        """
        report = {"episodic_written": 0, "semantic_written": 0,
                  "skill_written": False, "skill_skipped_reason": None,
                  "written_ids": []}

        # 1. Key events → episodic (log) + semantic (searchable), sharing an id.
        for ev in distilled.get("key_events", []):
            event_type = ev.get("event_type", "distilled")
            agent      = ev.get("agent", "distiller")
            payload    = ev.get("payload", {}) or {}
            outcome    = ev.get("outcome")
            if isinstance(outcome, str) and outcome.lower() == "null":
                outcome = None

            shared_id = None
            if episodic is not None:
                res = episodic.store(event_type, agent, payload, outcome=outcome)
                if res.get("status") == "ok":
                    report["episodic_written"] += 1
                    shared_id = res.get("id")

            if semantic is not None:
                text = payload.get("summary") or payload.get("text") or json.dumps(payload, default=str)
                meta = {"event_type": event_type, "agent": agent}
                if outcome is not None:
                    meta["outcome"] = outcome
                res = semantic.store(text, metadata=meta, id=shared_id)
                if res.get("status") == "ok":
                    report["semantic_written"] += 1
                    if res.get("id"):
                        report["written_ids"].append(res["id"])

        # 2. Candidate skill → procedural, but only if confident enough.
        skill = distilled.get("candidate_skill")
        confidence = distilled.get("confidence", 0.0)
        if skill and procedural is not None:
            if confidence >= SKILL_CONFIDENCE_THRESHOLD:
                res = procedural.store_skill(
                    skill.get("skill_name", "unnamed_skill"),
                    skill.get("agent", "distiller"),
                    skill.get("template", {}) or {},
                )
                report["skill_written"] = res.get("status") == "ok"
            else:
                report["skill_skipped_reason"] = (
                    f"confidence {confidence:.2f} < {SKILL_CONFIDENCE_THRESHOLD}"
                )
        elif skill is None:
            report["skill_skipped_reason"] = "no candidate_skill"

        return report
