"""
Audit Logger — the durable, tamper-evident sink for the board's audit trail.

The in-run `audit_trail` is a state list: every agent appends its actions and
constitution checks, the coordinator reconstructs integrity flags from it — and
then the run ends and it is gone. This module makes the trail durable: after each
board run the whole trail is flushed, in order, into an append-only SQLite table.

Tamper-EVIDENT, not just append-only: each row stores
    row_hash = sha256(prev_hash ‖ run_id ‖ agent ‖ action ‖ entry_json ‖ logged_at)
so the rows form a hash chain (each fingerprint commits to the entire history
before it). Editing or deleting any row breaks every fingerprint after it, and
`verify_chain()` pinpoints the first broken link. Same idea a blockchain uses,
minus the consensus machinery.

Discipline (same as memory / learning):
  * OFF the hot path — `log_run()` is called once per run with the final state,
    after the answer is already produced (`graph/workflow.run_board_with_learning`).
  * Best-effort — returns a status dict, NEVER raises; a broken audit DB must not
    take down the board. A failure is loud in the return value, silent to the user.
  * Append-only — there is no update/delete API, by design.

Verify from the shell (the 15-second demo):
    python -m eib_guardrails.audit_logger          # walks + verifies the chain
"""
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config.settings import AUDIT_DB_PATH, AUDIT_LOG_ENABLED

_GENESIS = "0" * 64  # prev_hash of the very first row

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    agent      TEXT NOT NULL,
    action     TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    logged_at  TEXT NOT NULL,
    prev_hash  TEXT NOT NULL,
    row_hash   TEXT NOT NULL
);
"""


def _canonical(entry: dict) -> str:
    """Deterministic JSON for hashing — key order and separators fixed so the
    same entry always produces the same fingerprint."""
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)


def _row_hash(prev_hash: str, run_id: str, agent: str, action: str,
              entry_json: str, logged_at: str) -> str:
    payload = "\x1f".join([prev_hash, run_id, agent, action, entry_json, logged_at])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def log_run(final_state: dict, run_id: str | None = None,
            db_path: str | None = None) -> dict:
    """Flush one completed run's audit_trail into the chained log.

    Called once per board run with the FINAL state (all agents' entries, in
    order). Returns a status dict and never raises — the audit sink is
    best-effort by the same rule as memory writes."""
    if not AUDIT_LOG_ENABLED:
        return {"status": "skipped", "reason": "disabled"}

    trail = (final_state or {}).get("audit_trail", []) or []
    if not trail:
        return {"status": "skipped", "reason": "empty_trail"}

    rid = run_id or uuid.uuid4().hex
    path = db_path or AUDIT_DB_PATH

    try:
        conn = _connect(path)
        try:
            with conn:  # one transaction: the tip read and the appends are atomic
                cur = conn.execute(
                    "SELECT row_hash FROM audit_log ORDER BY seq DESC LIMIT 1")
                row = cur.fetchone()
                prev = row[0] if row else _GENESIS

                written = 0
                for entry in trail:
                    if not isinstance(entry, dict):
                        entry = {"raw": entry}
                    agent = str(entry.get("agent", "unknown"))
                    action = str(entry.get("action", "unknown"))
                    entry_json = _canonical(entry)
                    logged_at = datetime.now(timezone.utc).isoformat()
                    h = _row_hash(prev, rid, agent, action, entry_json, logged_at)
                    conn.execute(
                        "INSERT INTO audit_log "
                        "(run_id, agent, action, entry_json, logged_at, prev_hash, row_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (rid, agent, action, entry_json, logged_at, prev, h),
                    )
                    prev = h
                    written += 1
        finally:
            conn.close()
        return {"status": "ok", "run_id": rid, "entries_written": written,
                "chain_tip": prev}
    except Exception as e:  # never let the audit sink take down the board
        return {"status": "failed", "run_id": rid, "error": str(e)}


def verify_chain(db_path: str | None = None) -> dict:
    """Walk the whole log recomputing every fingerprint. Any edited, deleted, or
    reordered row breaks the chain at the first affected link."""
    path = db_path or AUDIT_DB_PATH
    try:
        conn = _connect(path)
        try:
            rows = conn.execute(
                "SELECT seq, run_id, agent, action, entry_json, logged_at, "
                "prev_hash, row_hash FROM audit_log ORDER BY seq").fetchall()
        finally:
            conn.close()
    except Exception as e:
        return {"status": "failed", "error": str(e)}

    prev = _GENESIS
    for seq, rid, agent, action, entry_json, logged_at, prev_hash, row_hash in rows:
        expected = _row_hash(prev, rid, agent, action, entry_json, logged_at)
        if prev_hash != prev or row_hash != expected:
            return {"status": "ok", "valid": False, "entries": len(rows),
                    "first_bad_seq": seq}
        prev = row_hash
    return {"status": "ok", "valid": True, "entries": len(rows), "chain_tip": prev}


def read_log(run_id: str | None = None, limit: int = 100,
             db_path: str | None = None) -> list[dict]:
    """Read entries back (newest last), optionally for one run. Read-only."""
    path = db_path or AUDIT_DB_PATH
    try:
        conn = _connect(path)
        try:
            if run_id:
                rows = conn.execute(
                    "SELECT seq, run_id, agent, action, entry_json, logged_at "
                    "FROM audit_log WHERE run_id = ? ORDER BY seq LIMIT ?",
                    (run_id, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT seq, run_id, agent, action, entry_json, logged_at "
                    "FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
                rows = rows[::-1]
        finally:
            conn.close()
    except Exception:
        return []
    return [{"seq": s, "run_id": r, "agent": a, "action": ac,
             "entry": json.loads(ej), "logged_at": la}
            for s, r, a, ac, ej, la in rows]


if __name__ == "__main__":  # python -m eib_guardrails.audit_logger
    report = verify_chain()
    print(json.dumps(report, indent=2))
