"""Streamlit dashboard for the Energy Intelligence Board.

Two tabs — Board (map + metrics + mix + chat) and Observability (trust &
novelty).  Talks to the FastAPI backend over HTTP; never imports the graph
directly (the twin loop lives in uvicorn's lifespan).

Run:  streamlit run ui/app.py
"""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import requests
import streamlit as st
from streamlit_folium import st_folium

from ui.map_view import build_folium_map
from ui.observability import render as render_observability

EIB_API_URL = os.environ.get("EIB_API_URL", "http://localhost:8000")

_PRESETS = [
    ("Hormuz Blockade",
     "Iran closes the Strait of Hormuz following a military escalation."),
    ("Suez Diversion",
     "Houthi attacks force tankers to reroute via Cape of Good Hope."),
    ("Sanctions Shock",
     "New US sanctions on Iranian crude exporters effective in 30 days."),
]

_TONE_CSS = {
    "critical": "background-color:#fef2f2;color:#dc2626;border-left:3px solid #ef4444;",
    "elevated": "background-color:#fffbeb;color:#b45309;border-left:3px solid #f59e0b;",
    "ok":       "background-color:#f0fdf4;color:#15803d;border-left:3px solid #22c55e;",
}


# ── API helpers ────────────────────────────────────────────────────────────────

def _api(method: str, path: str, **kw) -> dict | None:
    timeout = kw.pop("timeout", 120)
    try:
        r = getattr(requests, method)(
            f"{EIB_API_URL}{path}", timeout=timeout, **kw,
        )
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return None
    except Exception:
        return None


@st.cache_data(ttl=15)
def _fetch_twin() -> dict | None:
    return _api("get", "/twin", timeout=8)


# ── Session state ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict = {
        "session_id":      None,
        "messages":        [],
        "last_summary":    None,
        "last_components": [],
        "last_follow_ups": [],
        "pending_message": None,
        "learn":           True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _clear_conversation() -> None:
    st.session_state["session_id"] = None
    st.session_state["messages"] = []
    st.session_state["last_summary"] = None
    st.session_state["last_components"] = []
    st.session_state["last_follow_ups"] = []


# ── Chat logic ─────────────────────────────────────────────────────────────────

def _send_message(message: str) -> None:
    """POST /chat and update session state with the response."""
    st.session_state["messages"].append({"role": "user", "content": message})

    resp = _api("post", "/chat", timeout=300, json={
        "session_id": st.session_state["session_id"],
        "message":    message,
        "learn":      st.session_state["learn"],
    })

    if resp:
        st.session_state["session_id"] = resp.get("session_id")
        st.session_state["messages"].append({
            "role":    "assistant",
            "content": resp.get("reply", ""),
            "mode":    resp.get("mode", "run_board"),
        })
        if resp.get("run_summary") is not None:
            st.session_state["last_summary"] = resp["run_summary"]
        st.session_state["last_components"] = resp.get("components") or []
        st.session_state["last_follow_ups"] = resp.get("follow_ups") or []
    else:
        st.session_state["messages"].append({
            "role":    "assistant",
            "content": ("Could not reach the board.  "
                        f"Is the API running at `{EIB_API_URL}`?"),
            "mode":    "error",
        })


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Energy Intelligence Board")
        st.caption("Multi-agent crisis response")

        st.markdown("**Scenarios**")
        for name, query in _PRESETS:
            if st.button(name, use_container_width=True, key=f"pre_{name}"):
                st.session_state["pending_message"] = query
                st.rerun()

        st.divider()
        st.session_state["learn"] = st.toggle(
            "Learn from runs", value=st.session_state.get("learn", True),
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Refresh twin", use_container_width=True):
                with st.spinner("Refreshing..."):
                    _api("post", "/twin/refresh", json={})
                _fetch_twin.clear()
                st.rerun()
        with c2:
            if st.button("New chat", use_container_width=True):
                _clear_conversation()
                st.rerun()

        st.divider()
        twin = _fetch_twin()
        if twin:
            status = twin.get("status", "cold")
            dot = {"ok": "🟢", "stale": "🟡"}.get(status, "⚪")
            st.markdown(f"**Twin:** {dot} {status}")
            ts = twin.get("last_refreshed_at")
            if ts:
                st.caption(f"Last refresh: {ts[:19]}")
        else:
            st.caption(f"API offline — {EIB_API_URL}")


# ── Board tab — left column ───────────────────────────────────────────────────

def _render_map() -> None:
    # The Board tab must be internally consistent: metrics, mix and actions all
    # describe the LAST RUN, so the map must too — its geojson arrives in the
    # run's `map` component. The live-twin snapshot (GET /twin — an independent
    # GRI read on its own clock) is only the pre-first-run fallback: the two can
    # honestly disagree around the DSM modelling threshold, and an unlabelled
    # mismatch (all-green map beside "11 stressed refineries") reads as a bug.
    components = st.session_state.get("last_components", [])
    comp = next((c for c in components if c.get("type") == "map"), None)
    if comp and (comp.get("geojson") or {}).get("features"):
        geojson = comp["geojson"]
        source = "this run"
    else:
        twin = _fetch_twin()
        twin_state = (twin or {}).get("twin_state", {}) or {}
        geojson = twin_state.get("geojson", {}) or {}
        source = "live twin (background refresh — independent of the chat run)"
    m = build_folium_map(geojson)
    st_folium(m, use_container_width=True, height=420, returned_objects=[])
    st.caption(f"Map source: {source}")


def _render_metrics() -> None:
    components = st.session_state.get("last_components", [])
    comp = next((c for c in components if c.get("type") == "metrics"), None)
    if not comp:
        return

    items = comp.get("items", [])
    if not items:
        return

    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        tone = item.get("tone", "ok")
        css = _TONE_CSS.get(tone, _TONE_CSS["ok"])
        value = item["value"]
        unit = item.get("unit") or ""
        display = f"{value} {unit}".strip() if unit else str(value)
        label = item["label"]
        with col:
            # Labels may wrap to two lines ("Stressed refineries") — reserve two
            # lines in EVERY tile so values sit on one baseline and tile bottoms
            # stay even, instead of the row going ragged.
            st.markdown(
                f'<div style="padding:10px 12px;border-radius:6px;min-height:80px;{css}">'
                f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:.04em;opacity:.7;line-height:14px;min-height:28px;">'
                f'{label}</div>'
                f'<div style="font-size:22px;font-weight:700;margin-top:2px;">'
                f'{display}</div></div>',
                unsafe_allow_html=True,
            )


def _render_run_warnings() -> None:
    """Loud surfaces for silent-failure signals: a failed scoring step or the
    scores-contradict-evidence tripwire must be visible on the Board tab, not
    buried in the audit chain — and the disrupted corridors are named with their
    per-corridor share of the gap so the cause is readable at a glance."""
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
            "verify in Observability → audit trail."
        )
    sit = ((summary.get("response_plan") or {}).get("situation") or {})
    drivers = sit.get("disruption_drivers") or []
    causes = sit.get("root_causes") or []
    origin_of = {}
    for g in causes:
        for d in g.get("driven", []):
            origin_of.setdefault(d.get("corridor"), g.get("origin"))
    # Zero-contribution corridors (disrupted but no refinery depends on them,
    # e.g. turkish_straits) add noise to an "by impact" line — drop them. If the
    # twin provided no decomposition at all, keep every name rather than none.
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
    """The 'News evidence' tile made inspectable: clickable sources + the
    per-corridor evidence coverage (a 0 = unverified this run, not calm)."""
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
                # No rating ≠ verified-bad: don't print a number that reads
                # like a real (dis)trust judgement.
                meta += " · unrated source"
            elif trust is not None:
                meta += f" · trust {float(trust):.2f}"
            tags = a.get("corridors") or []
            if tags:
                meta += f" · {', '.join(tags)}"
            st.markdown(f"- {line}{meta}")


def _render_mix_table() -> None:
    components = st.session_state.get("last_components", [])
    comp = next((c for c in components if c.get("type") == "mix_table"), None)
    if not comp:
        return

    st.markdown("**Recommended procurement mix**")
    rows = comp.get("rows", [])
    if rows:
        import pandas as pd

        display_keys = [
            ("supplier",         "Supplier"),
            ("grade",            "Grade"),
            ("volume_mbd",       "Volume (mbd)"),
            ("effective_volume_mbd", "Expected delivery (mbd)"),
            ("price_per_bbl",    "Price ($/bbl)"),
            ("transit_days",     "Transit (days)"),
            ("delivery_risk_fraction", "Corridor risk"),
            ("sanctions_status", "Sanctions"),
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
        draw = spr.get("draw_mbd", 0)
        days = spr.get("days_of_cover", 0)
        st.warning(
            f"SPR bridge: {draw} mbd for {days} days "
            f"(partial drawdown at max sustainable rate)",
        )


def _render_economic_impact() -> None:
    components = st.session_state.get("last_components", [])
    econ_comp = next((c for c in components
                      if c.get("type") == "metrics" and c.get("title") == "Economic impact"), None)
    if not econ_comp:
        return
    items = econ_comp.get("items", [])
    if not items:
        return
    st.markdown("**Economic impact**")
    cols = st.columns(min(len(items), 4))
    for i, item in enumerate(items):
        tone = item.get("tone", "ok")
        css = _TONE_CSS.get(tone, _TONE_CSS["ok"])
        value = item["value"]
        unit = item.get("unit") or ""
        display = f"{value} {unit}".strip() if unit else str(value)
        label = item["label"]
        with cols[i % len(cols)]:
            st.markdown(
                f'<div style="padding:10px 12px;border-radius:6px;min-height:80px;{css}">'
                f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:.04em;opacity:.7;line-height:14px;min-height:28px;">'
                f'{label}</div>'
                f'<div style="font-size:22px;font-weight:700;margin-top:2px;">'
                f'{display}</div></div>',
                unsafe_allow_html=True,
            )


def _render_recovery_table() -> None:
    components = st.session_state.get("last_components", [])
    comp = next((c for c in components if c.get("type") == "recovery_table"), None)
    if not comp:
        return
    st.markdown("**Recovery levers (ranked by net benefit)**")
    rows = comp.get("rows", [])
    if rows:
        import pandas as pd
        display_keys = [
            ("lever",           "Lever"),
            ("description",     "Description"),
            ("avoided_loss_usd", "Avoided loss ($)"),
            ("lever_cost_usd",  "Cost ($)"),
            ("net_benefit_usd", "Net benefit ($)"),
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

    tradeoff = next((c for c in components if c.get("type") == "policy_tradeoff"), None)
    if tradeoff:
        data = tradeoff.get("data", {})
        if data:
            fiscal = data.get("subsidy_fiscal_cost_usd", 0)
            cpi = data.get("passthrough_cpi_bps", 0)
            st.info(
                f"**Policy tradeoff — subsidy vs pass-through:** "
                f"Subsidize fuel = ${fiscal / 1e9:.2f} bn fiscal cost; "
                f"pass through = +{cpi} bps CPI impact."
            )


def _render_priority_actions() -> None:
    summary = st.session_state.get("last_summary")
    if not summary:
        return
    plan = summary.get("response_plan", {}) or {}
    actions = plan.get("priority_actions", [])
    if not actions:
        return

    st.markdown("**Priority actions**")
    for i, action in enumerate(actions, 1):
        st.markdown(f"{i}. {action}")


# ── Board tab — right column (chat) ──────────────────────────────────────────

def _render_chat() -> None:
    st.markdown("**Ask the Board**")

    chat_box = st.container(height=480)
    with chat_box:
        if not st.session_state["messages"]:
            st.caption(
                "Type a crisis scenario or click a preset to start.",
            )
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Energy Intelligence Board",
        page_icon="⚡",
        layout="wide",
    )
    _init_state()

    # Process any queued message (from preset / follow-up / chat input).
    pending = st.session_state.get("pending_message")
    if pending:
        st.session_state["pending_message"] = None
        with st.spinner("Running the board..."):
            _send_message(pending)

    _render_sidebar()

    tab_board, tab_obs = st.tabs(["Board", "Observability"])

    with tab_board:
        left, right = st.columns([3, 2])
        with left:
            _render_map()
            _render_metrics()
            _render_run_warnings()
            _render_news_sources()
            _render_economic_impact()
            _render_recovery_table()
            _render_mix_table()
            _render_priority_actions()
        with right:
            _render_chat()

    with tab_obs:
        render_observability(EIB_API_URL, st.session_state["last_summary"])


main()
