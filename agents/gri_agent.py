import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from openai import OpenAI

from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, GRI_MODEL
from graph.eib_state import EnergyIntelligenceBoard, StigmergyMarker
from eib_guardrails.constitution_checker import check as constitution_check
from tools.corridor_status import get_corridor_status
from tools.news_fetcher import fetch_news, build_search_query, route_corridors
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

You score corridor risk from TWO independent sources of truth, then combine them.

PHASE 1 — SCENARIO (from the QUERY):
If the QUERY describes a disruption, blockade, closure, conflict, sanctions, attack, or any threat to one or more corridors, those corridors are SCENARIO-DISRUPTED. Score them to match the described severity:
  - blockade / closure / shut down / military action → 0.85-1.0
  - sanctions / embargo → 0.7-0.9
  - attack / piracy / infrastructure failure → 0.7-0.9
  - tension / escalation / exercises → 0.4-0.6
  - general disruption / risk → 0.5-0.7
If the query names MULTIPLE corridors, score ALL of them — not just one.
If the query describes no disruption (e.g. "current status?"), skip this phase — the news evidence alone decides (PHASE 2).

PHASE 2 — EVIDENCE (from the NEWS SIGNALS — do this for EVERY corridor, ALWAYS):
Assess each corridor's CURRENT real-world risk from the retrieved articles alone, even when the query asserts nothing. Evidence alone CAN and MUST establish risk: what trusted reporting DESCRIBES maps to the SAME severity bands as Phase 1 (an attack reported by trusted press scores like an attack stated in the query).
Weighting rules:
  - Trust: high-trust sources (>= 0.7) carry the signal; unrated/low-trust sources corroborate but never establish on their own.
  - Recency: weight the last 72h most (age is given per article). A breaking disruption may have only 1-3 articles — few + fresh + high-trust IS a real signal: score it, set confidence lower. Many independent domains reporting across days = high confidence.
  - Attribution: "attributed" reporting (officials, navies, agencies quoted) beats "analysis"/opinion; market roundups and price commentary alone NEVER establish a disruption.
  - Calm is evidence too: if trusted articles describe normal transit, score low and cite them.
  - 0 articles for a corridor = UNVERIFIED: default score (chokepoint=0.2, non-chokepoint=0.1) and say "no evidence retrieved this run" in reasoning — never "confirmed calm".

PHASE 3 — COMBINE:
final score per corridor = the HIGHER of the scenario score and the evidence score. event_type follows whichever source set the final score. A scenario can raise a quiet corridor (what-ifs); evidence raises a corridor the query never mentioned.

ROOT-CAUSE GROUPING (after scoring):
If the evidence shows several corridors' risks stem from ONE underlying event — reporting explicitly links attacks/closures in one corridor to a conflict centred on another, or traffic diverted away from a disrupted corridor is raising pressure elsewhere — report it in root_cause_groups: the ORIGIN corridor and the corridors it is DRIVING, citing the linking articles. Group ONLY what the cited evidence links; simultaneous but unrelated events stay ungrouped. Empty list if none.

Respond with valid JSON only — no prose outside the JSON object."""


# Evidence window for the LLM prompt. Slots are allocated PROPORTIONALLY to how
# much evidence each corridor has (with a small floor so no corridor with news
# is invisible): a crisis corridor with 26 articles must dominate the window,
# not be diluted to an equal share with 7 quiet corridors — equal-share
# balancing is a DISPLAY rule, not a judgment rule (debugger.md #18 round 3).
# Corridor-tagged articles outrank untagged sweep noise; each bucket serves its
# highest-trust sources first.
_PROMPT_ARTICLE_CAP = 32
_PROMPT_FLOOR_PER_CORRIDOR = 2


def _select_articles(articles: list[dict], cap: int = _PROMPT_ARTICLE_CAP) -> list[dict]:
    tagged: dict[str, list[dict]] = {}
    untagged: list[dict] = []
    for a in articles:
        tags = a.get("corridors") or []
        if tags:
            for tag in tags:
                tagged.setdefault(tag, []).append(a)
        else:
            untagged.append(a)
    trust = lambda a: float(a.get("trust_score") or 0)
    for bucket in tagged.values():
        bucket.sort(key=trust, reverse=True)
    untagged.sort(key=trust, reverse=True)

    # Floor first, then hand remaining slots to the bucket with the most
    # unserved articles (i.e. proportional to evidence volume).
    alloc = {t: min(len(b), _PROMPT_FLOOR_PER_CORRIDOR) for t, b in tagged.items()}
    remaining = cap - sum(alloc.values())
    while remaining > 0:
        candidates = [(len(b) - alloc[t], t) for t, b in tagged.items()
                      if alloc[t] < len(b)]
        if not candidates:
            break
        candidates.sort(reverse=True)
        alloc[candidates[0][1]] += 1
        remaining -= 1

    selected: list[dict] = []
    seen: set[int] = set()
    for tag in sorted(tagged):
        for a in tagged[tag][:alloc[tag]]:
            if id(a) not in seen and len(selected) < cap:
                seen.add(id(a))
                selected.append(a)
    for a in untagged:
        if len(selected) >= cap:
            break
        if id(a) not in seen:
            seen.add(id(a))
            selected.append(a)
    return selected


def _signal_line(a: dict) -> str:
    age = a.get("age_days")
    meta = [f"trust {float(a.get('trust_score', 0)):.2f}",
            f"age {age:.1f}d" if isinstance(age, (int, float)) else "age ?"]
    if a.get("attribution"):
        meta.append(a["attribution"])
    line = f"[{' | '.join(meta)}] {a['title']} | {a.get('source', 'unknown')}"
    if a.get("corridors"):
        line += f" | corridors: {', '.join(a['corridors'])}"
    return line


def _build_user_prompt(query: str, articles: list[dict], corridors: list[dict],
                       evidence_by_corridor: dict | None = None,
                       corridor_evidence: dict | None = None) -> str:
    signals = "\n".join(_signal_line(a) for a in _select_articles(articles))
    coverage = ""
    if evidence_by_corridor:
        rich = corridor_evidence or {}

        def _cov_line(cid: str, n: int) -> str:
            ce = rich.get(cid)
            if not ce:
                return f"{cid}: {n}"
            return (f"{cid}: {n} articles | {ce.get('independent_domains', 0)} domains | "
                    f"{ce.get('fresh_72h', 0)} fresh(72h) | top trust "
                    f"{ce.get('top_trust', 0.0):.2f} | weight {ce.get('evidence_weight', 0.0)}")

        parts = "\n".join(
            _cov_line(cid, n) for cid, n in
            sorted(evidence_by_corridor.items(), key=lambda kv: -kv[1]))
        coverage = f"\nEVIDENCE COVERAGE (articles retrieved per corridor this run):\n{parts}\n"
        if any(n == 0 for n in evidence_by_corridor.values()):
            coverage += ("A corridor with 0 articles is UNVERIFIED this run — keep its default "
                         "score and say 'no evidence retrieved this run' in its reasoning; "
                         "never describe it as confirmed calm.\n")
    baselines = "\n".join(
        f"{c['id']}: {c['baseline_flow_mbd']} mbd | chokepoint={c['chokepoint']} | factors={c['risk_factors']}"
        for c in corridors
    )
    return f"""QUERY: {query}

The 8 known corridor IDs are: strait_of_hormuz, suez_canal, malacca_strait, bab_el_mandeb, turkish_straits, danish_straits, cape_of_good_hope, panama_canal.

NEWS SIGNALS (trust_score | title | source):
{signals or "(no articles found)"}
{coverage}

CORRIDOR BASELINES:
{baselines}

INSTRUCTIONS:
1. PHASE 1: Read the QUERY. Identify which of the 8 corridors it explicitly describes as disrupted. Score each one by the described severity (see system prompt). If MULTIPLE corridors are named, score ALL of them.
2. PHASE 2: Score EVERY corridor from the news evidence using the weighting rules (trust, recency, attribution, independent domains). Evidence alone MUST raise a corridor's score when trusted recent reporting describes disruption — even if the QUERY asserts nothing. Cite the articles in key_signals.
3. PHASE 3: final score = MAX(scenario score, evidence score). Only corridors with neither a scenario nor news evidence use default scores (chokepoint=0.2, non-chokepoint=0.1).
4. Any corridor name NOT in the 8 known ones → novel_corridor_alerts only.
5. Classify each corridor's dominant risk driver as one of: war_conflict | sanctions | political_tension | weather_disruption | market_spike | piracy | infrastructure_failure | none
6. You MUST include ALL 8 corridors in corridor_risk — not just the disrupted ones.
7. If the cited evidence links several corridors' risks to one underlying event, fill root_cause_groups (origin + driven corridors, cite the linking titles); otherwise use an empty list.

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
  "root_cause_groups": [{{"origin": "<corridor_id>", "driven": ["<corridor_id>"], "reasoning": "<one sentence citing the linking evidence>", "key_signals": ["<exact article title>"]}}],
  "overall_assessment": "<2-3 sentences>",
  "low_trust_signals_flagged": <int>
}}"""


# A causal group's origin must itself carry elevated risk — a calm corridor
# cannot be the root cause of anything.
_ROOT_CAUSE_MIN_ORIGIN_RISK = 0.4


def _validate_root_causes(groups, corridor_risk: dict[str, float]) -> list[dict]:
    """Deterministic gate on the LLM's causal grouping: known corridors only,
    no self-links, origin must be elevated, one group per origin. The LLM
    proposes the causal story; this keeps it inside the world model."""
    out: list[dict] = []
    seen: set[str] = set()
    if not isinstance(groups, list):
        return out
    for g in groups:
        if not isinstance(g, dict):
            continue
        origin = g.get("origin")
        if origin not in KNOWN_CORRIDORS or origin in seen:
            continue
        if corridor_risk.get(origin, 0.0) < _ROOT_CAUSE_MIN_ORIGIN_RISK:
            continue
        driven = sorted({c for c in (g.get("driven") or [])
                         if c in KNOWN_CORRIDORS and c != origin})
        if not driven:
            continue
        seen.add(origin)
        out.append({
            "origin":      origin,
            "driven":      driven,
            "reasoning":   str(g.get("reasoning") or "")[:300],
            "key_signals": [s for s in (g.get("key_signals") or [])
                            if isinstance(s, str)][:5],
        })
    return out


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


def _fetch_tools(query: str, corridors: list[str] | None = None) -> tuple[dict, dict]:
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_news     = ex.submit(fetch_news, query, corridors=corridors)
        f_corridor = ex.submit(get_corridor_status)
        return f_news.result(), f_corridor.result()


# The scoring LLM call is retried once: a transient API failure must not become
# a confident "all corridors nominal" (debugger.md #21 — an empty scorecard was
# silently indistinguishable from a calm world).
_LLM_ATTEMPTS = 2


def _llm_assess(query: str, articles: list[dict], corridors: list[dict],
                evidence_by_corridor: dict, corridor_evidence: dict,
                attempts: int = _LLM_ATTEMPTS) -> tuple[dict, int, str | None]:
    """Run the corridor-risk scoring LLM, retrying on failure.

    Returns (llm_output, attempts_used, failure_reason). Success requires a
    scorecard naming at least one known corridor — an exception, unparseable
    JSON, or an empty/unknown-keyed scorecard are ALL failures: "the analyst
    never answered" must never be read as "the analyst said all-clear"."""
    failure: str | None = None
    user_prompt = _build_user_prompt(query, articles, corridors,
                                     evidence_by_corridor, corridor_evidence)
    attempt = 0
    for attempt in range(1, attempts + 1):
        try:
            response = _client.chat.completions.create(
                model=GRI_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            llm_output = json.loads(response.choices[0].message.content)
        except Exception as e:
            failure = f"{type(e).__name__}: {e}"
            continue
        raw_risk = llm_output.get("corridor_risk")
        if isinstance(raw_risk, dict) and any(cid in KNOWN_CORRIDORS for cid in raw_risk):
            return llm_output, attempt, None
        keys = sorted(raw_risk)[:10] if isinstance(raw_risk, dict) else type(raw_risk).__name__
        failure = f"LLM answered but scorecard has no known corridors (corridor_risk: {keys})"
    return {
        "corridor_risk":             {},
        "novel_corridor_alerts":     [],
        "overall_assessment":        f"assessment failed after {attempt} attempt(s): {failure}",
        "low_trust_signals_flagged": 0,
    }, attempt, failure


def gri_node(state: EnergyIntelligenceBoard) -> dict:
    query    = state.get("query", "")
    now      = datetime.now(timezone.utc).isoformat()

    # ── 1. Parallel tool fetch ─────────────────────────────────────────────
    # The news search runs on corridor keywords derived from the query — never
    # on the raw conversational text (see tools.news_fetcher.build_search_query).
    # Corridors the query names get targeted searches; a query naming none fans
    # out to all 8 so no corridor is left competing for a single article page.
    search_query = build_search_query(query)
    routed = route_corridors(query)
    news_result, corridor_result = _fetch_tools(search_query, routed or None)
    articles  = news_result["data"]["articles"]
    corridors = corridor_result["data"]["corridors"]
    evidence_by_corridor = news_result["data"].get("evidence_by_corridor", {}) or {}
    corridor_evidence = news_result["data"].get("corridor_evidence", {}) or {}

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
        "evidence_by_corridor": evidence_by_corridor,
        "corridor_evidence": corridor_evidence,
        "constitution_check": tool_check,
        "timestamp":         now,
    }]

    # ── 3. LLM risk assessment (chain-of-evidence; retried, fails LOUD) ────
    llm_output, llm_attempts, llm_failure = _llm_assess(
        query, articles, corridors, evidence_by_corridor, corridor_evidence)
    assessment_failed = llm_failure is not None
    if assessment_failed:
        llm_output["low_trust_signals_flagged"] = news_result["low_trust_sources_flagged"]

    # ── 4. Constitution check on LLM output ───────────────────────────────
    llm_check = constitution_check("gri", {**llm_output, "risk_signals": articles})
    audit.append({
        "agent":             "gri_agent",
        "action":            "llm_assessment",
        "attempts":          llm_attempts,
        "llm_failure":       llm_failure,       # None on success — the audit must
        "assessment_failed": assessment_failed,  # show WHY a scorecard is empty
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

    # ── 5b. Evidence-ignored safety net ────────────────────────────────────
    # Deterministic tripwire: a corridor with fresh, high-trust evidence scored
    # at baseline is either the LLM reading calm reporting correctly — or the
    # exact silent failure of debugger.md #18 round 3. Either way it must be
    # visible in the audit, never silent. Warning only; the score stands.
    ignored = {
        cid: {"score": corridor_risk.get(cid, 0.0),
              "fresh_72h": ce.get("fresh_72h", 0),
              "top_trust": ce.get("top_trust", 0.0),
              "evidence_weight": ce.get("evidence_weight", 0.0)}
        for cid, ce in corridor_evidence.items()
        if cid in KNOWN_CORRIDORS
        and ce.get("fresh_72h", 0) >= 2 and ce.get("top_trust", 0.0) >= 0.8
        and corridor_risk.get(cid, 0.0) < 0.4
    }
    if ignored:
        audit.append({
            "agent":     "gri_agent",
            "action":    "evidence_ignored_warning",
            "detail":    ("baseline score despite fresh high-trust evidence — "
                          "verify: calm reporting, or evidence-blind scoring"),
            "corridors": ignored,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ── 5c. Root-cause groups (LLM-judged, deterministically validated) ────
    # Which corridors' risks are ONE event: origin + knock-on. The coordinator
    # merges these with the twin's overloaded reroute edges for the narrative.
    root_causes = _validate_root_causes(
        llm_output.get("root_cause_groups"), corridor_risk)
    if root_causes:
        audit.append({
            "agent":     "gri_agent",
            "action":    "root_cause_grouping",
            "groups":    root_causes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

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
        "assessment_failed": assessment_failed,
        "root_causes":      root_causes,
        "stigmergy_markers": markers,
        "audit_trail":      audit,
        "constitution_flags": llm_check.get("violations", []),
    }
