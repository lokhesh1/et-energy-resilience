import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from openai import OpenAI

from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, GRI_MODEL
from graph.eib_state import EnergyIntelligenceBoard, StigmergyMarker
from eib_guardrails.constitution_checker import check as constitution_check
from tools.corridor_status import get_corridor_status
from tools.news_fetcher import fetch_news, build_search_query
from memory.xmemory import XMemory

KNOWN_CORRIDORS = {
    "strait_of_hormuz", "suez_canal", "malacca_strait", "bab_el_mandeb",
    "turkish_straits", "danish_straits", "cape_of_good_hope", "panama_canal",
}

_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# Shared long-term memory facade. Lazy inside — no cloud connection on import.
_xmemory = XMemory()

# Persist only notable risks (matches the pheromone-deposit threshold) so the
# episodic log stays signal, not noise.
_MEMORY_PERSIST_THRESHOLD = 0.6

_SYSTEM_PROMPT = """You are a Geopolitical Risk Intelligence (GRI) analyst for Indian energy supply chains.

Your task has TWO phases — do them in order.

PHASE 1 — SCENARIO EXTRACTION (from the QUERY):
Read the user's QUERY. If it describes a disruption, blockade, closure, conflict, sanctions, attack, or any threat to one or more corridors, those corridors are SCENARIO-DISRUPTED. Score them to match the described severity:
  - blockade / closure / shut down / military action → 0.85-1.0
  - sanctions / embargo → 0.7-0.9
  - attack / piracy / infrastructure failure → 0.7-0.9
  - tension / escalation / exercises → 0.4-0.6
  - general disruption / risk → 0.5-0.7
If the query names MULTIPLE corridors, score ALL of them — not just one.
If the query describes no disruption (e.g. "current status?"), skip this phase.

PHASE 2 — NEWS CORROBORATION:
Check the provided news signals. For each corridor:
  - If news supports the scenario → cite articles in key_signals, optionally raise the score.
  - If news contradicts the scenario → note it in reasoning, optionally lower (but not below 0.5 for an explicit scenario).
  - For corridors NOT in the scenario and with no relevant news → default score (chokepoint=0.2, non-chokepoint=0.1).

Respond with valid JSON only — no prose outside the JSON object."""


def _build_user_prompt(query: str, articles: list[dict], corridors: list[dict]) -> str:
    signals = "\n".join(
        f"[{a['trust_score']:.2f}] {a['title']} | {a.get('source', 'unknown')}"
        for a in articles[:20]
    )
    baselines = "\n".join(
        f"{c['id']}: {c['baseline_flow_mbd']} mbd | chokepoint={c['chokepoint']} | factors={c['risk_factors']}"
        for c in corridors
    )
    return f"""QUERY: {query}

The 8 known corridor IDs are: strait_of_hormuz, suez_canal, malacca_strait, bab_el_mandeb, turkish_straits, danish_straits, cape_of_good_hope, panama_canal.

NEWS SIGNALS (trust_score | title | source):
{signals or "(no articles found)"}

CORRIDOR BASELINES:
{baselines}

INSTRUCTIONS:
1. PHASE 1: Read the QUERY. Identify which of the 8 corridors it explicitly describes as disrupted. Score each one by the described severity (see system prompt). If MULTIPLE corridors are named, score ALL of them.
2. PHASE 2: Check news signals for corroboration. Cite matching articles in key_signals.
3. For corridors not in the scenario and with no news, use default scores (chokepoint=0.2, non-chokepoint=0.1).
4. Any corridor name NOT in the 8 known ones → novel_corridor_alerts only.
5. Classify each corridor's dominant risk driver as one of: war_conflict | sanctions | political_tension | weather_disruption | market_spike | piracy | infrastructure_failure | none
6. You MUST include ALL 8 corridors in corridor_risk — not just the disrupted ones.

Return this exact JSON schema:
{{
  "corridor_risk": {{
    "<corridor_id>": {{
      "score": <float 0.0-1.0>,
      "confidence": <float 0.0-1.0>,
      "evidence_count": <int — number of key_signals>,
      "key_signals": ["<exact article title from news above, or empty list>"],
      "reasoning": "<one sentence explaining the score>",
      "event_type": "<one of the 8 types above>"
    }}
  }},
  "novel_corridor_alerts": ["<name>"],
  "overall_assessment": "<2-3 sentences>",
  "low_trust_signals_flagged": <int>
}}"""


def _deposit_pheromones(corridor_risk: dict) -> list[StigmergyMarker]:
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "type":         "risk",
            "target":       cid,
            "intensity":    round(float(val.get("score", val) if isinstance(val, dict) else val), 4),
            "deposited_by": "gri_agent",
            "timestamp":    now,
            "decay_rate":   0.1,
        }
        for cid, val in corridor_risk.items()
        if float(val.get("score", val) if isinstance(val, dict) else val) >= 0.6
    ]


def _fetch_tools(query: str) -> tuple[dict, dict]:
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_news     = ex.submit(fetch_news, query)
        f_corridor = ex.submit(get_corridor_status)
        return f_news.result(), f_corridor.result()


def gri_node(state: EnergyIntelligenceBoard) -> dict:
    query    = state.get("query", "")
    now      = datetime.now(timezone.utc).isoformat()

    # ── 1. Parallel tool fetch ─────────────────────────────────────────────
    # The news search runs on corridor keywords derived from the query — never
    # on the raw conversational text (see tools.news_fetcher.build_search_query).
    search_query = build_search_query(query)
    news_result, corridor_result = _fetch_tools(search_query)
    articles  = news_result["data"]["articles"]
    corridors = corridor_result["data"]["corridors"]

    # ── 2. Constitution check on tool outputs ──────────────────────────────
    tool_check = constitution_check("gri", {
        "risk_signals":             articles,
        "low_trust_signals_flagged": news_result["low_trust_sources_flagged"],
    })

    audit: list[dict] = [{
        "agent":             "gri_agent",
        "action":            "tool_fetch",
        "news_query":        search_query,
        "news_status":       news_result["status"],
        "corridor_status":   corridor_result["status"],
        "article_count":     len(articles),
        "constitution_check": tool_check,
        "timestamp":         now,
    }]

    # ── 3. LLM risk assessment (chain-of-evidence) ─────────────────────────
    try:
        response = _client.chat.completions.create(
            model=GRI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(query, articles, corridors)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        llm_output = json.loads(response.choices[0].message.content)
    except Exception as e:
        llm_output = {
            "corridor_risk":          {},
            "novel_corridor_alerts":  [],
            "overall_assessment":     f"LLM failed: {e}",
            "low_trust_signals_flagged": news_result["low_trust_sources_flagged"],
        }

    # ── 4. Constitution check on LLM output ───────────────────────────────
    llm_check = constitution_check("gri", {**llm_output, "risk_signals": articles})
    audit.append({
        "agent":             "gri_agent",
        "action":            "llm_assessment",
        "constitution_check": llm_check,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    })

    # ── 5. Filter corridor_risk to known corridors only ────────────────────
    raw_corridor_risk = llm_output.get("corridor_risk", {})
    corridor_risk = {
        cid: round(float(v.get("score", v) if isinstance(v, dict) else v), 4)
        for cid, v in raw_corridor_risk.items()
        if cid in KNOWN_CORRIDORS
    }

    # ── 6a. Carry event_type forward for DSM + decay ──────────────────────
    # corridor_risk collapses to {cid: float}; DSM's scenario model is event-type
    # driven, so surface the classification separately in shared state.
    corridor_events = {
        cid: (v.get("event_type", "none") if isinstance(v, dict) else "none")
        for cid, v in raw_corridor_risk.items()
        if cid in KNOWN_CORRIDORS
    }

    # ── 6b. Deposit stigmergy pheromones ───────────────────────────────────
    markers = _deposit_pheromones(raw_corridor_risk)

    # ── 7. Persist notable risk signals to long-term memory ────────────────
    # Dual-write (episodic + semantic) via the xMemory facade. The geopolitical
    # event_type lives in the payload so decay half-lives apply on recall.
    try:
        for cid, v in raw_corridor_risk.items():
            if cid not in KNOWN_CORRIDORS:
                continue
            entry = v if isinstance(v, dict) else {}
            score = float(entry.get("score", v) if isinstance(v, dict) else v)
            if score < _MEMORY_PERSIST_THRESHOLD:
                continue
            # Searchable text = what actually happened (the cited signals), not the
            # scoring rationale — that's what future "have we seen this?" recall needs.
            signals = entry.get("key_signals") or []
            memory_text = (
                "; ".join(signals) if signals
                else entry.get("reasoning") or f"{cid} elevated risk {score:.2f}"
            )
            _xmemory.remember(
                event_type="risk_assessment",
                agent="gri_agent",
                payload={
                    "corridor":    cid,
                    "score":       round(score, 4),
                    "event_type":  entry.get("event_type", "none"),
                    "key_signals": signals,
                    "reasoning":   entry.get("reasoning", ""),
                    "query":       query,
                },
                outcome="success",
                text=memory_text,
            )
    except Exception:
        pass  # memory is best-effort; never break the node

    return {
        "current_agent":    "gri_agent",
        "risk_signals":     articles,
        "corridor_risk":    corridor_risk,
        "corridor_events":  corridor_events,
        "stigmergy_markers": markers,
        "audit_trail":      audit,
        "constitution_flags": llm_check.get("violations", []),
    }
