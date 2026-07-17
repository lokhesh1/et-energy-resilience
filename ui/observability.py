"""Observability panel — trust & novelty view for the Energy Intelligence Board.

Renders constitution flags, retrieved memories, pheromone field, stigmergy
markers, audit-chain verification, and the agents roster.  Called from the
Observability tab in ui/app.py.
"""
from __future__ import annotations

import requests
import streamlit as st


def render(api_base: str, last_summary: dict | None) -> None:
    """Top-level render for the observability tab."""
    if not last_summary:
        st.info("Run a query from the Board tab to see observability data.")
        st.divider()
        _render_audit(api_base)
        _render_agents(api_base)
        return

    col1, col2 = st.columns(2)
    with col1:
        _render_constitution_flags(last_summary)
        _render_pheromone_field(last_summary)
    with col2:
        _render_memories(last_summary)
        _render_stigmergy(last_summary)

    st.divider()
    _render_audit(api_base)
    st.divider()
    _render_agents(api_base)


# ── Constitution flags ─────────────────────────────────────────────────────────

def _render_constitution_flags(summary: dict) -> None:
    st.markdown("**Constitution flags**")
    flags = summary.get("constitution_flags", [])
    if not flags:
        st.success("No constitution violations detected.")
        return

    for f in flags:
        if isinstance(f, dict):
            severity = f.get("severity", "warn")
            rule = f.get("rule_id", "unknown")
            msg = f.get("message", str(f))
        else:
            severity, rule, msg = "warn", "?", str(f)

        if severity == "block":
            st.error(f"**{rule}** BLOCK — {msg}")
        else:
            st.warning(f"**{rule}** WARN — {msg}")


# ── Retrieved memories ─────────────────────────────────────────────────────────

def _render_memories(summary: dict) -> None:
    st.markdown("**Retrieved memories**")
    memories = summary.get("retrieved_memories", [])
    if not memories:
        st.info("No memories retrieved for this run.")
        return

    for mem in memories:
        if isinstance(mem, dict):
            text = mem.get("content") or mem.get("text") or str(mem)
            score = mem.get("score")
            suffix = f"  (score {score:.2f})" if score else ""
            st.markdown(f"- {text}{suffix}")
        else:
            st.markdown(f"- {mem}")


# ── Pheromone field ────────────────────────────────────────────────────────────

def _render_pheromone_field(summary: dict) -> None:
    st.markdown("**Pheromone field**")
    field = summary.get("pheromone_field", {})
    if not field:
        st.info("No pheromone signals active.")
        return

    import pandas as pd

    data = sorted(field.items(), key=lambda x: -float(x[1]))
    df = pd.DataFrame(data, columns=["Corridor", "Intensity"])
    st.bar_chart(df, x="Corridor", y="Intensity", horizontal=True)


# ── Stigmergy markers ─────────────────────────────────────────────────────────

def _render_stigmergy(summary: dict) -> None:
    st.markdown("**Stigmergy markers** (top 5)")
    stig = summary.get("stigmergy", {})
    markers = stig.get("top_markers", [])
    count = stig.get("marker_count", 0)

    if not markers:
        st.info(f"No stigmergy markers ({count} total).")
        return

    st.caption(f"{count} markers total")
    import pandas as pd

    df = pd.DataFrame(markers)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Audit chain ───────────────────────────────────────────────────────────────

def _render_audit(api_base: str) -> None:
    st.markdown("**Audit chain verification**")
    st.caption("Tamper-evident hash-chained log — verify integrity end-to-end.")
    if st.button("Verify audit chain", key="btn_audit_verify"):
        try:
            r = requests.get(f"{api_base}/audit/verify", timeout=10)
            if r.status_code == 403:
                st.error("Access denied — insufficient permissions.")
                return
            r.raise_for_status()
            result = r.json()
            status = result.get("status", "")

            if status == "skipped":
                st.info(f"Audit log skipped: {result.get('reason', 'unknown')}")
            elif status == "failed":
                st.error(f"Verification failed: {result.get('error', 'unknown')}")
            elif result.get("valid") is True:
                entries = result.get("entries", 0)
                st.success(f"Chain intact — {entries} entries verified.")
            elif result.get("valid") is False:
                bad = result.get("first_bad_seq", "?")
                entries = result.get("entries", 0)
                st.error(
                    f"Chain BROKEN at sequence {bad} "
                    f"(out of {entries} entries).",
                )
            else:
                st.json(result)
        except requests.ConnectionError:
            st.error(f"Cannot reach API at {api_base}")
        except Exception as exc:
            st.error(f"Verification error: {exc}")


# ── Agents roster ─────────────────────────────────────────────────────────────

def _render_agents(api_base: str) -> None:
    st.markdown("**Board agents**")
    try:
        r = requests.get(f"{api_base}/agents", timeout=5)
        r.raise_for_status()
        agents = r.json().get("agents", [])
        for a in agents:
            st.markdown(f"**{a['name']}** — {a['role']}")
    except Exception:
        st.info("Could not load agents roster — is the API running?")
