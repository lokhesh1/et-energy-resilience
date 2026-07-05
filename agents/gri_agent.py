import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from openai import OpenAI

from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, GRI_MODEL
from graph.eib_state import EnergyIntelligenceBoard, StigmergyMarker
from eib_guardrails.constitution_checker import check as constitution_check
from tools.corridor_status import get_corridor_status
from tools.news_fetcher import fetch_news
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

Your role is to assess evidence and derive risk scores — not predict or anticipate.
Rules:
- Cite only the news signals provided. Do not add external knowledge as evidence.
- If a corridor has no relevant signals, assign the default score (chokepoint=0.2, non-chokepoint=0.1).
- evidence_count must exactly match the number of items in key_signals.
- Respond with valid JSON only — no prose outside the JSON object."""


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

NEWS SIGNALS (trust_score | title | source):
{signals}

CORRIDOR BASELINES:
{baselines}

CHAIN OF EVIDENCE INSTRUCTIONS:
1. For each of the 8 known corridors, list relevant signals from above (key_signals).
2. Derive score from signal count × trust weight — show reasoning in one sentence.
3. Any corridor name NOT in the 8 known ones goes to novel_corridor_alerts only.
4. Classify the dominant risk driver for each corridor as one of:
   war_conflict | sanctions | political_tension | weather_disruption |
   market_spike | piracy | infrastructure_failure | none

Return this exact JSON schema:
{{
  "corridor_risk": {{
    "<corridor_id>": {{
      "score": <float 0.0-1.0>,
      "confidence": <float 0.0-1.0>,
      "evidence_count": <int matching key_signals length>,
      "key_signals": ["<exact article title>"],
      "reasoning": "<one sentence>",
      "event_type": "<one of the 8 valid types above>"
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
            "intensity":    round(float(val.get("score", val)), 4),
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
    news_result, corridor_result = _fetch_tools(query)
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
            temperature=0.1,
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
