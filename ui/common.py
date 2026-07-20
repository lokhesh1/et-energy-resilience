"""Shared helpers for the multi-page Streamlit dashboard.

Extracted from ui/app.py so that Board, Simulation, Procurement, Priority
Actions and Observability pages can share state, API calls, styling and
session logic without duplication.

No agent/graph imports — talks to the FastAPI backend over HTTP only.
"""
from __future__ import annotations

import os
import requests
import streamlit as st

EIB_API_URL = os.environ.get("EIB_API_URL", "http://localhost:8000")

_PRESETS = [
    ("Hormuz Blockade",
     "Iran closes the Strait of Hormuz following a military escalation."),
    ("Suez Diversion",
     "Houthi attacks force tankers to reroute via Cape of Good Hope."),
    ("Sanctions Shock",
     "New US sanctions on Iranian crude exporters effective in 30 days."),
]

TONE_CSS = {
    "critical": "background-color:#fef2f2;color:#dc2626;border-left:3px solid #ef4444;",
    "elevated": "background-color:#fffbeb;color:#b45309;border-left:3px solid #f59e0b;",
    "ok":       "background-color:#f0fdf4;color:#15803d;border-left:3px solid #22c55e;",
}


# ── API helpers ───────────────────────────────────────────────────────────────

def api(method: str, path: str, **kw) -> dict | None:
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
def fetch_twin() -> dict | None:
    return api("get", "/twin", timeout=8)


# ── Session state ─────────────────────────────────────────────────────────────

def init_state() -> None:
    defaults: dict = {
        "session_id":       None,
        "messages":         [],
        "last_summary":     None,
        "last_components":  [],
        "last_follow_ups":  [],
        "last_twin_snapshot": None,
        "pending_message":  None,
        "learn":            True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clear_conversation() -> None:
    st.session_state["session_id"] = None
    st.session_state["messages"] = []
    st.session_state["last_summary"] = None
    st.session_state["last_components"] = []
    st.session_state["last_follow_ups"] = []
    st.session_state["last_twin_snapshot"] = None


# ── Chat logic ────────────────────────────────────────────────────────────────

def send_message(message: str) -> None:
    """POST /chat and update session state with the response."""
    st.session_state["messages"].append({"role": "user", "content": message})

    resp = api("post", "/chat", timeout=300, json={
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
        if resp.get("twin_snapshot") is not None:
            st.session_state["last_twin_snapshot"] = resp["twin_snapshot"]
        st.session_state["last_components"] = resp.get("components") or []
        st.session_state["last_follow_ups"] = resp.get("follow_ups") or []
    else:
        st.session_state["messages"].append({
            "role":    "assistant",
            "content": ("Could not reach the board.  "
                        f"Is the API running at `{EIB_API_URL}`?"),
            "mode":    "error",
        })


# ── Shared renderers ─────────────────────────────────────────────────────────

def render_metric_tiles(items: list[dict]) -> None:
    """Render a row of tone-colored metric tiles from a metrics component."""
    if not items:
        return
    cols = st.columns(min(len(items), 6))
    for i, item in enumerate(items):
        tone = item.get("tone", "ok")
        css = TONE_CSS.get(tone, TONE_CSS["ok"])
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


def render_sidebar() -> None:
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
                    api("post", "/twin/refresh", json={})
                fetch_twin.clear()
                st.rerun()
        with c2:
            if st.button("New chat", use_container_width=True):
                clear_conversation()
                st.rerun()

        st.divider()
        twin = fetch_twin()
        if twin:
            status = twin.get("status", "cold")
            dot = {"ok": "🟢", "stale": "🟡"}.get(status, "⚪")
            st.markdown(f"**Twin:** {dot} {status}")
            ts = twin.get("last_refreshed_at")
            if ts:
                st.caption(f"Last refresh: {ts[:19]}")
        else:
            st.caption(f"API offline — {EIB_API_URL}")


def empty_state_message() -> None:
    """Standard 'no board run yet' placeholder."""
    st.info("No board run yet — start from a scenario preset or the Board chat.")
