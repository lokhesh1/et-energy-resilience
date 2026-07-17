"""
Crisis Coordinator — the board's fan-in and final voice.

Every other agent produces a *piece*: GRI the risk, DSM the scenarios, SCTD the
physical twin, the procurement pod a recommended cargo mix. The Coordinator is the
single node that reads the whole board and turns it into ONE actionable answer: a
structured `response_plan` (what is happening, what has been secured, what remains,
how loud to escalate) plus a `final_recommendation` sentence a human can act on.

Design: HYBRID, same discipline as DSM — every load-bearing number is deterministic
and the LLM is decoration only.
    - `response_plan` is assembled purely from state (twin gap, refinery statuses,
      the evaluator's mix). Delete the LLM entirely and the plan is byte-identical.
    - The LLM writes ONLY `final_recommendation`, and a deterministic template
      fallback produces a coherent recommendation if the model is unavailable.
      So the coordinator never depends on a network call to produce a safe answer.

Integrity-aggregator role: `constitution_flags` is a PLAIN state key, so each
sequential agent overwrites the previous one's — by the time control reaches here
only the bid_evaluator's flags survive in that key. The durable record of every
agent's violations is the append-only `audit_trail` (each agent embeds its own
`constitution_check`). The coordinator therefore reconstructs the run's block-level
flags FROM the audit trail, so an upstream integrity failure can never be silently
dropped from the final plan. This is the board's last gate.

Memory: recalls semantically-similar past crises to inform the narrative (best
effort — [] on any failure) and persists the response (fire-and-forget), like the
other agents. Neither can break the node.
"""
import json
from datetime import datetime, timezone

from openai import OpenAI

from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, COORDINATOR_MODEL
from graph.eib_state import EnergyIntelligenceBoard
from eib_guardrails.constitution_checker import check as constitution_check
from memory.xmemory import XMemory
from tools.spr_calculator import calculate_drawdown

_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# Shared long-term memory facade. Lazy inside — no cloud connection on import.
_xmemory = XMemory()

# Recompute tolerance for the plan↔twin arithmetic (numbers are rounded upstream).
_TOL = 0.02

# Escalation vocabulary, quietest → loudest. COORD-05 validates against this set;
# COORD-01 forbids a "routine" all-clear while a shortfall is still uncovered.
ESCALATION_LEVELS = ("routine", "watch", "elevated", "critical")

# Corridor risk at/above this reads "watch" even with zero projected shortfall:
# real tension (GRI's 0.4–0.6 band) must never be reported as "routine / nominal".
# Matches the band DSM's scenario threshold (0.5) only partially covers.
_WATCH_RISK_THRESHOLD = 0.4


# ── Deterministic plan assembly ─────────────────────────────────────────────────

def _collect_block_flags(state: EnergyIntelligenceBoard) -> list[dict]:
    """Reconstruct every block-severity constitution violation raised anywhere in
    the run. `constitution_flags` only holds the LAST writer's (plain key, serially
    overwritten), so the authoritative source is the append-only audit_trail, where
    each agent embedded its own `constitution_check`. De-duplicated by (agent, rule).
    """
    seen: set[tuple] = set()
    flags: list[dict] = []

    def _add(agent: str, v: dict) -> None:
        if v.get("severity") != "block":
            return
        key = (agent, v.get("rule_id"), v.get("message"))
        if key in seen:
            return
        seen.add(key)
        flags.append({"agent": agent, "rule_id": v.get("rule_id"),
                      "message": v.get("message")})

    for entry in state.get("audit_trail", []) or []:
        agent = entry.get("agent", "unknown")
        cc = entry.get("constitution_check")
        if isinstance(cc, dict):
            for v in cc.get("violations", []) or []:
                _add(agent, v)

    # Whatever is still in the plain key (the bid_evaluator's, typically) too.
    for v in state.get("constitution_flags", []) or []:
        _add(v.get("agent", "bid_evaluator"), v)

    return flags


def _escalation_level(gap: float, residual: float, covers_gap: bool,
                      critical_count: int, stressed_count: int,
                      top_risk_score: float = 0.0) -> str:
    """Deterministic severity dial. An UNCOVERED shortfall is the worst case and
    always reads 'critical' — this is what makes COORD-01 (no all-clear over an
    open gap) hold by construction. Elevated corridor risk with no projected
    shortfall reads 'watch', never 'routine' — tension short of disruption is
    still not an all-clear."""
    if residual > _TOL or (gap > 0 and not covers_gap):
        return "critical"
    if critical_count > 0:
        return "critical"
    if stressed_count > 0 or gap > 0:
        return "elevated"
    if top_risk_score >= _WATCH_RISK_THRESHOLD:
        return "watch"
    return "routine"


def _committed_actions(mix: dict) -> list[dict]:
    """The cargoes the evaluator committed, reduced to what an operator needs to
    act — carries sanctions_status so the coordinator's own constitution can
    re-verify nothing blocked slipped into the plan (COORD-02)."""
    actions = []
    for c in mix.get("components", []) or []:
        actions.append({
            "supplier":         c.get("supplier"),
            "supplier_id":      c.get("supplier_id"),
            "region":           c.get("region"),
            "grade":            c.get("grade"),
            "volume_mbd":       c.get("volume_mbd"),
            "price_per_bbl":    c.get("price_per_bbl"),
            "delivery_corridor": c.get("delivery_corridor"),
            "transit_days":     c.get("transit_days_to_india"),
            "sanctions_status": c.get("sanctions_status", "clear"),
            # Disclose transit risk: without these the narrative says "secure via
            # strait_of_hormuz" and "monitor disruption on strait_of_hormuz" in the
            # same plan with no acknowledgment they are the same corridor.
            "delivery_risk_fraction": c.get("delivery_risk_fraction", 0.0),
            "effective_volume_mbd":   c.get("effective_volume_mbd",
                                            c.get("volume_mbd")),
        })
    return actions


def _spr_bridge(residual: float, scenarios: list[dict]) -> dict | None:
    """Size an SPR drawdown against the residual gap the market mix leaves open
    (best-effort — None on any failure, the plan just omits the bridge). The
    longest active scenario duration, if any, bounds `covers_duration`."""
    if residual <= _TOL:
        return None
    try:
        duration = max((float(s.get("duration_days") or 0) for s in scenarios),
                       default=0.0) or None
        spr = calculate_drawdown(residual, duration_days=duration)
        d = spr.get("data", {})
        return {
            "drawdown_mbd":    d.get("drawdown_mbd"),
            "days_of_cover":   d.get("days_of_cover"),
            "bridge_fraction": d.get("bridge_fraction"),
            "unbridged_mbd":   d.get("unbridged_mbd"),
            "duration_days":   d.get("duration_days"),
            "covers_duration": d.get("covers_duration"),
            "adequacy":        d.get("adequacy"),
        }
    except Exception:
        return None


def _priority_actions(escalation: str, residual: float, actions: list[dict],
                      disrupted: list[str], block_flags: list[dict],
                      spr_bridge: dict | None = None,
                      watch_risks: list[dict] | None = None) -> list[str]:
    """Human-readable next steps, deterministic from the plan. Ordered by urgency:
    the uncovered gap first, then the cargoes to secure, then what to watch."""
    out: list[str] = []
    if residual > _TOL:
        msg = (
            f"UNCOVERED: {round(residual, 3)} mbd of the shortfall is not met by "
            f"market bids — escalate for SPR drawdown / demand curtailment."
        )
        if spr_bridge and spr_bridge.get("days_of_cover"):
            msg += (f" SPR can bridge {spr_bridge['drawdown_mbd']} mbd for "
                    f"~{spr_bridge['days_of_cover']} days")
            if spr_bridge.get("adequacy") == "partial_bridge":
                msg += (f" ({spr_bridge['unbridged_mbd']} mbd exceeds the max "
                        f"drawdown rate — curtailment still required)")
            msg += "."
        out.append(msg)
    for a in actions:
        line = (
            f"Secure {a['volume_mbd']} mbd {a.get('grade') or 'crude'} from "
            f"{a['supplier']} ({a.get('region')}) via {a.get('delivery_corridor')} "
            f"at ${a['price_per_bbl']}/bbl."
        )
        fraction = float(a.get("delivery_risk_fraction", 0.0) or 0.0)
        if fraction > 0:
            line += (f" CAUTION: corridor {round(fraction * 100)}% disrupted — "
                     f"expected delivery {a.get('effective_volume_mbd')} mbd, "
                     f"risk priced into the ranking.")
        out.append(line)
    if disrupted:
        out.append(f"Monitor active disruption on: {', '.join(sorted(disrupted))}.")
    if block_flags and escalation != "routine":
        out.append(
            f"Resolve {len(block_flags)} upstream integrity flag(s) before execution."
        )
    if not out:
        if watch_risks:
            info = ", ".join(
                f"{r['corridor']} ({r['event_type']}, {r['score']:.2f})"
                for r in watch_risks
            )
            out.append(f"No procurement action required — no shortfall projected. "
                       f"Monitor elevated corridor risk: {info}.")
        else:
            out.append("No action required — no shortfall projected.")
    return out


def _build_response_plan(state: EnergyIntelligenceBoard,
                         block_flags: list[dict], now: str) -> dict:
    """Assemble the full plan from state. Every number here is copied or recomputed
    from an upstream deterministic value — nothing is invented."""
    twin = state.get("twin_state", {}) or {}
    mix  = state.get("recommended_mix", {}) or {}
    corridor_risk = state.get("corridor_risk", {}) or {}
    corridor_events = state.get("corridor_events", {}) or {}

    gap = round(float(twin.get("total_india_shortfall_mbd", 0.0) or 0.0), 4)
    # Coverage counts EXPECTED delivery (risk-discounted), not barrels bought: a
    # cargo through a 30%-choked corridor covers only 70% of its volume. Falls
    # back to the nominal total for mixes that predate the effective field.
    covered = round(float(
        mix.get("effective_volume_mbd", mix.get("total_volume_mbd", 0.0)) or 0.0), 4)
    residual = round(max(0.0, gap - covered), 4)
    covers_gap = bool(mix.get("covers_gap", gap <= 0))

    refineries = twin.get("refineries", []) or []
    critical = [r["name"] for r in refineries if r.get("status") == "critical"]
    stressed = [r["name"] for r in refineries if r.get("status") == "stressed"]

    disrupted = [
        c.get("id") for c in (twin.get("corridors", []) or [])
        if float(c.get("disruption_fraction", 0.0) or 0.0) > 0.0
    ]

    # Top corridor risks, strongest first (score may be a float or a dict).
    def _score(v):
        return float(v.get("score", v) if isinstance(v, dict) else v)
    top_risks = sorted(
        ({"corridor": cid, "score": round(_score(v), 4),
          "event_type": corridor_events.get(cid, "none")}
         for cid, v in corridor_risk.items()),
        key=lambda r: r["score"], reverse=True,
    )[:3]

    actions = _committed_actions(mix)
    watch_risks = [r for r in top_risks if r["score"] >= _WATCH_RISK_THRESHOLD]
    escalation = _escalation_level(
        gap, residual, covers_gap,
        int(twin.get("critical_count", 0) or 0),
        int(twin.get("stressed_count", 0) or 0),
        top_risk_score=top_risks[0]["score"] if top_risks else 0.0,
    )
    spr_bridge = _spr_bridge(residual, state.get("scenarios", []) or [])

    unresolved = [f"{f['agent']}/{f['rule_id']}: {f['message']}" for f in block_flags]
    if residual > _TOL:
        unresolved.append(f"{residual} mbd shortfall uncovered by market supply.")

    return {
        "escalation_level": escalation,
        "situation": {
            "top_corridor_risks":  top_risks,
            "scenarios_modelled":  len(state.get("scenarios", []) or []),
            "gap_mbd":             gap,
            "critical_refineries": critical,
            "stressed_refineries": stressed,
            "disrupted_corridors": disrupted,
            # Evidence base: how many live news articles GRI actually saw. Zero
            # means the run was BLIND — an all-clear must be caveated, because
            # "no disruption found" and "no evidence looked at" are not the same.
            "news_articles":       len(state.get("risk_signals", []) or []),
        },
        "procurement": {
            "covered_mbd":       covered,
            "coverage_ratio":    mix.get("coverage_ratio"),
            "covers_gap":        covers_gap,
            "residual_gap_mbd":  residual,
            "committed_actions": actions,
            "est_daily_cost_usd": mix.get("est_daily_cost_usd"),
            "spr_bridge":        spr_bridge,
        },
        "priority_actions": _priority_actions(escalation, residual, actions,
                                              disrupted, block_flags, spr_bridge,
                                              watch_risks=watch_risks),
        "unresolved_issues": unresolved,
        "generated_at": now,
    }


# ── Deterministic narrative fallback ────────────────────────────────────────────

def _template_recommendation(plan: dict) -> str:
    """A coherent recommendation built purely from the plan — used verbatim when
    the LLM is unavailable, and as the seed the LLM is asked to phrase."""
    sit = plan["situation"]
    proc = plan["procurement"]
    esc = plan["escalation_level"].upper()

    if sit["gap_mbd"] <= 0:
        # A blind run (zero news articles) must never hand out a confident
        # all-clear: "no disruption found" ≠ "no evidence looked at".
        caveat = ""
        if sit.get("news_articles") == 0:
            caveat = (" Caution: zero news articles were retrieved this run — "
                      "this assessment is baseline-only and low confidence.")
        elevated = [r for r in sit.get("top_corridor_risks", [])
                    if r.get("score", 0) >= _WATCH_RISK_THRESHOLD]
        if elevated:
            info = ", ".join(
                f"{r['corridor']} ({r['event_type']}, risk {r['score']:.2f})"
                for r in elevated
            )
            return (f"{esc}: No India-bound crude shortfall projected, but risk "
                    f"is elevated on {info}. Monitor closely; no procurement "
                    f"action required at this time.{caveat}")
        return (f"{esc}: No India-bound crude shortfall projected. "
                f"Corridors nominal; no procurement action required.{caveat}")

    lead = sit["top_corridor_risks"][0] if sit["top_corridor_risks"] else None
    driver = (f"{lead['corridor']} ({lead['event_type']}, risk {lead['score']})"
              if lead else "corridor disruption")
    crit = (f" {len(sit['critical_refineries'])} refinery(ies) critical."
            if sit["critical_refineries"] else "")

    if proc["residual_gap_mbd"] > _TOL:
        tail = (f"Market bids cover {proc['covered_mbd']} mbd; "
                f"{proc['residual_gap_mbd']} mbd remains UNCOVERED — escalate for "
                f"strategic reserve / demand-side measures.")
        bridge = proc.get("spr_bridge")
        if bridge and bridge.get("days_of_cover"):
            tail += (f" SPR can bridge {bridge['drawdown_mbd']} mbd for "
                     f"~{bridge['days_of_cover']} days.")
    else:
        tail = (f"Procurement secures {proc['covered_mbd']} mbd "
                f"({len(proc['committed_actions'])} cargo(es)), closing the gap.")

    risky = [a for a in proc["committed_actions"]
             if float(a.get("delivery_risk_fraction", 0.0) or 0.0) > 0]
    if risky:
        worst = max(float(a["delivery_risk_fraction"]) for a in risky)
        tail += (f" Note: {len(risky)} committed cargo(es) transit partially "
                 f"disrupted corridors (up to {round(worst * 100)}% choked) — "
                 f"coverage counts expected delivery, not barrels bought.")

    return (f"{esc}: {driver} puts {sit['gap_mbd']} mbd of India-bound crude at "
            f"risk.{crit} {tail}")


_SYSTEM_PROMPT = """You are the Crisis Coordinator for an energy supply-chain board.
You are given a fully-computed response plan (JSON) and a deterministic draft
recommendation. Do NOT change, add, or question any number. Rephrase the draft into
ONE clear, decision-ready recommendation for an energy-security officer, preserving
every figure exactly. Respond with valid JSON only: {"recommendation": "<text>"}."""


def _narrate(plan: dict, precedents: list[dict]) -> str:
    """LLM phrasing of the deterministic draft. Any failure → the draft verbatim.
    The numbers are already final; the model only improves the prose."""
    draft = _template_recommendation(plan)
    try:
        context = json.dumps({"plan": plan, "draft": draft,
                              "precedents": [p.get("text") for p in precedents]})
        response = _client.chat.completions.create(
            model=COORDINATOR_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        parsed = json.loads(response.choices[0].message.content)
        text = parsed.get("recommendation")
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass
    return draft


# ── Node ────────────────────────────────────────────────────────────────────────

def coordinator_node(state: EnergyIntelligenceBoard) -> dict:
    now = datetime.now(timezone.utc).isoformat()

    # ── 1. Aggregate the run's block-level integrity flags (from the audit trail) ─
    block_flags = _collect_block_flags(state)

    # ── 2. Deterministic response plan (load-bearing; no LLM) ──────────────────
    plan = _build_response_plan(state, block_flags, now)

    # ── 3. Recall similar past crises to inform the narrative (best-effort) ────
    precedents: list[dict] = []
    try:
        lead = plan["situation"]["top_corridor_risks"]
        query_text = (
            f"{state.get('query', '')} "
            f"{lead[0]['corridor'] if lead else ''} "
            f"{lead[0]['event_type'] if lead else ''} "
            f"gap {plan['situation']['gap_mbd']} mbd"
        ).strip()
        precedents = _xmemory.recall_similar(query_text, top_k=3) or []
    except Exception:
        precedents = []
    plan["precedents"] = [
        {"text": p.get("text"), "score": p.get("score")} for p in precedents
    ]

    # ── 4. Final recommendation (LLM phrasing, deterministic template fallback) ─
    recommendation = _narrate(plan, precedents)

    # ── 5. Coordinator's own constitution gate (the board's last check) ────────
    check_result = constitution_check("coordinator", {
        "response_plan":        plan,
        "twin_state":           state.get("twin_state", {}) or {},
        "upstream_block_flags": block_flags,
    })

    # ── 6. Persist the response (fire-and-forget) ──────────────────────────────
    try:
        _xmemory.remember(
            event_type="crisis_response",
            agent="crisis_coordinator",
            payload={
                "escalation":       plan["escalation_level"],
                "gap_mbd":          plan["situation"]["gap_mbd"],
                "covered_mbd":      plan["procurement"]["covered_mbd"],
                "residual_gap_mbd": plan["procurement"]["residual_gap_mbd"],
                "committed":        len(plan["procurement"]["committed_actions"]),
            },
            outcome="success",
            text=recommendation,
        )
    except Exception:
        pass  # memory is best-effort; never break the node

    audit = [{
        "agent":              "crisis_coordinator",
        "action":             "coordinate",
        "escalation_level":   plan["escalation_level"],
        "gap_mbd":            plan["situation"]["gap_mbd"],
        "covered_mbd":        plan["procurement"]["covered_mbd"],
        "residual_gap_mbd":   plan["procurement"]["residual_gap_mbd"],
        "committed_cargoes":  len(plan["procurement"]["committed_actions"]),
        "upstream_block_flags": len(block_flags),
        "precedents_recalled": len(precedents),
        "constitution_check": check_result,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    }]

    return {
        "current_agent":      "crisis_coordinator",
        "response_plan":      plan,
        "final_recommendation": recommendation,
        "retrieved_memories": precedents,
        "audit_trail":        audit,
        "constitution_flags": check_result.get("violations", []),
    }
