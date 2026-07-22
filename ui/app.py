"""Streamlit dashboard for the Energy Intelligence Board.

Multi-page layout (st.navigation):
  1. Board        — map + metrics + warnings + news + chat
  2. Simulation   — animated voyage simulation map
  3. Procurement  — recommended mix + SPR + economic impact
  4. Actions      — priority actions + escalation + recommendation
  5. Observability — trust & novelty (delegates to ui/observability.py)

Session state is shared across pages automatically.
Talks to the FastAPI backend over HTTP; never imports the graph directly.

Run:  streamlit run ui/app.py
"""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from streamlit_folium import st_folium

from ui.common import (
    EIB_API_URL, TONE_CSS,
    api, fetch_twin, init_state, send_message, clear_conversation,
    render_metric_tiles, render_sidebar, empty_state_message,
)
from ui.map_view import build_folium_map
from ui.observability import render as render_observability
from ui.sim_map import build_sim_map_html, build_voyages


# ── Page: Board ──────────────────────────────────────────────────────────────

def page_board() -> None:
    left, right = st.columns([3, 2])
    with left:
        _render_map()
        _render_metrics()
        _render_run_warnings()
        _render_news_sources()
    with right:
        _render_chat()


def _render_map() -> None:
    components = st.session_state.get("last_components", [])
    comp = next((c for c in components if c.get("type") == "map"), None)
    if comp and (comp.get("geojson") or {}).get("features"):
        geojson = comp["geojson"]
        source = "this run"
    else:
        twin = fetch_twin()
        twin_state = (twin or {}).get("twin_state", {}) or {}
        geojson = twin_state.get("geojson", {}) or {}
        source = "live twin (background refresh)"
    m = build_folium_map(geojson)
    st_folium(m, use_container_width=True, height=420, returned_objects=[])
    st.caption(f"Map source: {source}")


def _render_metrics() -> None:
    components = st.session_state.get("last_components", [])
    comp = next((c for c in components if c.get("type") == "metrics"
                 and c.get("title") != "Economic impact"), None)
    if not comp:
        return
    render_metric_tiles(comp.get("items", []))


def _render_run_warnings() -> None:
    summary = st.session_state.get("last_summary")
    if not summary:
        return
    assessment = summary.get("assessment") or {}
    if assessment.get("failed"):
        reason = assessment.get("failure_reason") or "no usable scorecard returned"
        st.error(
            "Risk scoring FAILED this run — the figures above are unassessed "
            f"defaults, not an all-clear. Re-run the board. ({reason})"
        )
    elif assessment.get("evidence_ignored_corridors"):
        cids = ", ".join(assessment["evidence_ignored_corridors"])
        st.warning(
            f"Scores contradict fresh high-trust evidence for: {cids} — "
            "verify in Observability tab."
        )
    sit = ((summary.get("response_plan") or {}).get("situation") or {})
    drivers = sit.get("disruption_drivers") or []
    causes = sit.get("root_causes") or []
    origin_of = {}
    for g in causes:
        for d in g.get("driven", []):
            origin_of.setdefault(d.get("corridor"), g.get("origin"))
    shown = [d for d in drivers if d.get("gap_contribution_mbd")] or drivers
    if shown:
        parts = [
            f"**{d['corridor']}** (~{d.get('gap_contribution_mbd', 0)} mbd of the gap, "
            f"risk {d.get('risk_score', 0)}"
            + (f", knock-on of {origin_of[d['corridor']]}"
               if d["corridor"] in origin_of else "")
            + ")"
            for d in shown
        ]
        st.markdown("Disrupted corridors by impact: " + " · ".join(parts))
    if causes and causes[0].get("reasoning"):
        st.caption(f"Root cause — {causes[0]['origin']}: {causes[0]['reasoning']}")


def _render_news_sources() -> None:
    summary = st.session_state.get("last_summary")
    if not summary:
        return
    news = summary.get("news_evidence") or {}
    articles = news.get("articles") or []
    count = news.get("article_count", 0)
    with st.expander(f"News evidence — {count} articles (click to inspect sources)"):
        by_corridor = news.get("by_corridor") or {}
        if by_corridor:
            nonzero = [f"{c}: {n}" for c, n in
                       sorted(by_corridor.items(), key=lambda kv: -kv[1]) if n]
            zero = sorted(c for c, n in by_corridor.items() if not n)
            if nonzero:
                st.caption("Evidence per corridor — " + " · ".join(nonzero))
            if zero:
                st.caption("No articles retrieved this run for: " + ", ".join(zero)
                           + " — unverified, not confirmed calm.")
        if not articles:
            st.caption("No articles retrieved — this assessment is baseline-only, "
                       "low confidence.")
        for a in articles:
            title = a.get("title") or "(untitled)"
            url = a.get("url") or ""
            line = f"[{title}]({url})" if url else title
            meta = f" — {a.get('source', 'unknown')}"
            trust = a.get("trust_score")
            if a.get("trust_rated") is False:
                meta += " · unrated source"
            elif trust is not None:
                meta += f" · trust {float(trust):.2f}"
            tags = a.get("corridors") or []
            if tags:
                meta += f" · {', '.join(tags)}"
            st.markdown(f"- {line}{meta}")


def _render_chat() -> None:
    st.markdown("**Ask the Board**")

    chat_box = st.container(height=480)
    with chat_box:
        if not st.session_state["messages"]:
            st.caption("Type a crisis scenario or click a preset to start.")
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                mode = msg.get("mode")
                if msg["role"] == "assistant" and mode:
                    if mode == "run_board":
                        st.caption("Board run")
                    elif mode == "answer_from_last_run":
                        st.caption("Follow-up — from last run")
                st.markdown(msg["content"])

    follow_ups = st.session_state.get("last_follow_ups", [])
    if follow_ups:
        fu_cols = st.columns(min(len(follow_ups), 4))
        for i, fu in enumerate(follow_ups):
            with fu_cols[i % len(fu_cols)]:
                if st.button(fu, key=f"fu_{i}", use_container_width=True):
                    st.session_state["pending_message"] = fu
                    st.rerun()

    user_input = st.chat_input("Ask the board...")
    if user_input:
        st.session_state["pending_message"] = user_input
        st.rerun()


# ── Page: Voyage Simulation ──────────────────────────────────────────────────

def page_simulation() -> None:
    st.subheader("Voyage Simulation")

    summary = st.session_state.get("last_summary")
    components = st.session_state.get("last_components", [])
    mix_comp = next((c for c in components if c.get("type") == "mix_table"), None)
    mix_rows = (mix_comp.get("rows") or []) if mix_comp else []

    twin_state = _get_twin_state()

    html = build_sim_map_html(mix_rows=mix_rows, twin_state=twin_state,
                              summary=summary, height=600)
    st.components.v1.html(html, height=620, scrolling=False)

    if mix_rows:
        st.markdown("**Active voyages**")
        voyage_data = build_voyages(mix_rows, twin_state, summary)
        rows_display = []
        for v in voyage_data.get("voyages", []):
            if v.get("type") == "baseline":
                continue
            rows_display.append({
                "Supplier": v.get("supplier", ""),
                "Grade": v.get("grade", ""),
                "Volume (mbd)": v.get("volume_mbd", 0),
                "Barrels/day": f"{v.get('barrels_per_day', 0):,}",
                "Corridor": (v.get("delivery_corridor") or "").replace("_", " "),
                "Transit (days)": v.get("transit_days", 0),
                "Status": v.get("status", "clear").upper(),
            })
        if rows_display:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows_display),
                         use_container_width=True, hide_index=True)

        for rv in voyage_data.get("reroutes", []):
            from_c = rv.get("from_corridor", "").replace("_", " ")
            to_c = rv.get("to_corridor", "").replace("_", " ")
            vol = rv.get("volume_mbd", 0)
            added = rv.get("added_transit_days", "?")
            tag = " **OVERLOADED**" if rv.get("overloaded") else ""
            st.warning(f"Reroute: {from_c} → {to_c} — {vol} mbd, +{added} days{tag}")
    elif not summary:
        empty_state_message()
    else:
        st.caption("No committed cargoes this run — showing baseline traffic.")


# ── Page: Procurement Mix ────────────────────────────────────────────────────

def page_procurement() -> None:
    st.subheader("Procurement Mix")

    components = st.session_state.get("last_components", [])
    mix_comp = next((c for c in components if c.get("type") == "mix_table"), None)

    if not mix_comp and not st.session_state.get("last_summary"):
        empty_state_message()
        return

    if mix_comp:
        _render_mix_table(mix_comp)

    econ_comp = next((c for c in components
                      if c.get("type") == "metrics" and c.get("title") == "Economic impact"), None)
    if econ_comp:
        st.markdown("---")
        st.markdown("**Economic impact**")
        render_metric_tiles(econ_comp.get("items", []))

    recovery_comp = next((c for c in components if c.get("type") == "recovery_table"), None)
    if recovery_comp:
        st.markdown("---")
        _render_recovery_table(recovery_comp)

    tradeoff_comp = next((c for c in components if c.get("type") == "policy_tradeoff"), None)
    if tradeoff_comp:
        _render_policy_tradeoff(tradeoff_comp)


def _render_mix_table(comp: dict) -> None:
    st.markdown("**Recommended procurement mix**")
    rows = comp.get("rows", [])
    if rows:
        import pandas as pd

        display_keys = [
            ("supplier",              "Supplier"),
            ("grade",                 "Grade"),
            ("volume_mbd",            "Volume (mbd)"),
            ("effective_volume_mbd",  "Expected delivery (mbd)"),
            ("price_per_bbl",         "Price ($/bbl)"),
            ("transit_days",          "Transit (days)"),
            ("delivery_risk_fraction", "Corridor risk"),
            ("sanctions_status",      "Sanctions"),
        ]
        df_rows: list[dict] = []
        for r in rows:
            row: dict = {}
            for key, label in display_keys:
                val = r.get(key, "")
                if key == "price_per_bbl" and isinstance(val, (int, float)):
                    val = f"${val:.2f}"
                elif key in ("volume_mbd", "effective_volume_mbd") and isinstance(val, (int, float)):
                    val = f"{val:.3f}"
                elif key == "delivery_risk_fraction":
                    val = f"{round(float(val) * 100)}%" if isinstance(val, (int, float)) and val else "—"
                row[label] = val
            df_rows.append(row)
        st.dataframe(
            pd.DataFrame(df_rows), use_container_width=True, hide_index=True,
        )

    spr = comp.get("spr_bridge")
    if spr:
        draw = spr.get("drawdown_mbd", 0)
        days = spr.get("days_of_cover", 0)
        st.warning(
            f"SPR bridge: {draw} mbd for {days} days "
            f"(partial drawdown at max sustainable rate)",
        )


def _render_recovery_table(comp: dict) -> None:
    st.markdown("**Recovery levers (ranked by net benefit)**")
    rows = comp.get("rows", [])
    if rows:
        import pandas as pd
        display_keys = [
            ("lever",              "Lever"),
            ("description",        "Description"),
            ("avoided_loss_usd",   "Avoided loss ($)"),
            ("lever_cost_usd",     "Cost ($)"),
            ("net_benefit_usd",    "Net benefit ($)"),
            ("time_to_effect_days", "Time (days)"),
        ]
        df_rows: list[dict] = []
        for r in rows:
            row: dict = {}
            for key, label in display_keys:
                val = r.get(key, "")
                if key in ("avoided_loss_usd", "lever_cost_usd", "net_benefit_usd"):
                    if isinstance(val, (int, float)):
                        val = f"${val:,.0f}"
                row[label] = val
            df_rows.append(row)
        st.dataframe(
            pd.DataFrame(df_rows), use_container_width=True, hide_index=True,
        )


def _render_policy_tradeoff(comp: dict) -> None:
    data = comp.get("data", {})
    if data:
        fiscal = data.get("subsidy_fiscal_cost_usd", 0)
        cpi = data.get("passthrough_cpi_bps", 0)
        st.info(
            f"**Policy tradeoff — subsidy vs pass-through:** "
            f"Subsidize fuel = ${fiscal / 1e9:.2f} bn fiscal cost; "
            f"pass through = +{cpi} bps CPI impact."
        )


# ── Page: Priority Actions ───────────────────────────────────────────────────

def page_actions() -> None:
    st.subheader("Priority Actions")

    summary = st.session_state.get("last_summary")
    if not summary:
        empty_state_message()
        return

    plan = summary.get("response_plan", {}) or {}

    escalation = summary.get("escalation_level") or plan.get("escalation", "routine")
    esc_colors = {"critical": "#ef4444", "elevated": "#f59e0b",
                  "watch": "#3b82f6", "routine": "#22c55e"}
    col = esc_colors.get(escalation, "#6b7280")
    st.markdown(
        f'<div style="display:inline-block;padding:6px 16px;border-radius:20px;'
        f'background:{col}22;color:{col};font-weight:600;font-size:15px;'
        f'border:1px solid {col}44;margin-bottom:16px">'
        f'Escalation: {escalation.upper()}</div>',
        unsafe_allow_html=True,
    )

    actions = plan.get("priority_actions", [])
    if actions:
        for i, action in enumerate(actions, 1):
            st.markdown(f"{i}. {action}")
    else:
        st.caption("No priority actions for this run.")

    recommendation = summary.get("final_recommendation") or plan.get("final_recommendation")
    if recommendation:
        st.markdown("---")
        st.markdown("**Board recommendation**")
        st.markdown(recommendation)


# ── Page: Observability ──────────────────────────────────────────────────────

def page_observability() -> None:
    render_observability(EIB_API_URL, st.session_state.get("last_summary"))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_twin_state() -> dict:
    """Get twin_state for the simulation map.

    Priority: stored twin_snapshot (full corridor_risks/impacts/routes from
    the last board run) > live twin endpoint > empty dict.
    """
    snap = st.session_state.get("last_twin_snapshot")
    if snap:
        return snap
    twin = fetch_twin()
    return (twin or {}).get("twin_state", {}) or {}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Energy Intelligence Board",
        page_icon="⚡",
        layout="wide",
    )
    init_state()

    pending = st.session_state.get("pending_message")
    if pending:
        st.session_state["pending_message"] = None
        with st.spinner("Running the board..."):
            send_message(pending)

    render_sidebar()

    pages = [
        st.Page(page_board,         title="Board",          icon="📊"),
        st.Page(page_simulation,    title="Simulation",     icon="🚢"),
        st.Page(page_procurement,   title="Procurement",    icon="📦"),
        st.Page(page_actions,       title="Actions",        icon="⚡"),
        st.Page(page_observability, title="Observability",  icon="🔍"),
    ]
    nav = st.navigation(pages)
    nav.run()


main()
