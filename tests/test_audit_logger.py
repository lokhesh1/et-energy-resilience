"""
Tests for eib_guardrails/audit_logger.py — the durable, tamper-evident audit sink.

Covers:
  * flushing a run's audit_trail into the chained SQLite log (order preserved,
    one row per entry, chain tip returned);
  * the hash chain spanning RUNS, not just rows — run 2 links onto run 1's tip;
  * tamper evidence: editing, deleting, or reordering a row breaks verification
    at the first affected link;
  * the best-effort discipline: disabled → skipped, empty trail → skipped,
    unwritable path → failed status, never an exception;
  * `read_log` round-trip;
  * the workflow wiring: `run_board_with_learning` flushes the final state.
"""
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

import eib_guardrails.audit_logger as al
from eib_guardrails.audit_logger import log_run, verify_chain, read_log
from graph.workflow import run_board_with_learning


def _trail(n=3, agent="gri"):
    return [{"agent": agent, "action": f"step_{i}", "detail": i} for i in range(n)]


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "audit_test.db")


# ── Logging + chaining ──────────────────────────────────────────────────────────

def test_log_run_writes_all_entries_in_order(db):
    report = log_run({"audit_trail": _trail(3)}, run_id="run1", db_path=db)
    assert report["status"] == "ok"
    assert report["entries_written"] == 3
    assert report["chain_tip"]

    entries = read_log(db_path=db)
    assert [e["entry"]["action"] for e in entries] == ["step_0", "step_1", "step_2"]
    assert all(e["run_id"] == "run1" for e in entries)


def test_chain_spans_runs_and_verifies(db):
    log_run({"audit_trail": _trail(2)}, run_id="run1", db_path=db)
    tip1 = verify_chain(db_path=db)["chain_tip"]
    log_run({"audit_trail": _trail(2)}, run_id="run2", db_path=db)

    # run2's first row must link onto run1's tip — one chain over the whole log
    conn = sqlite3.connect(db)
    prev_of_run2 = conn.execute(
        "SELECT prev_hash FROM audit_log WHERE run_id='run2' ORDER BY seq LIMIT 1"
    ).fetchone()[0]
    conn.close()
    assert prev_of_run2 == tip1

    report = verify_chain(db_path=db)
    assert report["valid"] is True and report["entries"] == 4


def test_log_run_mints_a_run_id_when_none_given(db):
    report = log_run({"audit_trail": _trail(1)}, db_path=db)
    assert report["status"] == "ok" and len(report["run_id"]) == 32


# ── Tamper evidence ─────────────────────────────────────────────────────────────

def test_edited_row_breaks_the_chain(db):
    log_run({"audit_trail": _trail(4)}, run_id="r", db_path=db)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE audit_log SET entry_json = '{\"agent\":\"gri\",\"forged\":true}' "
                 "WHERE seq = 2")
    conn.commit(); conn.close()

    report = verify_chain(db_path=db)
    assert report["valid"] is False
    assert report["first_bad_seq"] == 2


def test_deleted_row_breaks_the_chain(db):
    log_run({"audit_trail": _trail(4)}, run_id="r", db_path=db)
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM audit_log WHERE seq = 2")
    conn.commit(); conn.close()

    report = verify_chain(db_path=db)
    assert report["valid"] is False
    assert report["first_bad_seq"] == 3   # the row after the hole no longer links


def test_untampered_log_verifies_clean(db):
    log_run({"audit_trail": _trail(5)}, run_id="r", db_path=db)
    assert verify_chain(db_path=db)["valid"] is True


# ── Best-effort discipline ──────────────────────────────────────────────────────

def test_disabled_skips(db, monkeypatch):
    monkeypatch.setattr(al, "AUDIT_LOG_ENABLED", False)
    report = log_run({"audit_trail": _trail(2)}, db_path=db)
    assert report == {"status": "skipped", "reason": "disabled"}


def test_empty_trail_skips(db):
    assert log_run({"audit_trail": []}, db_path=db)["status"] == "skipped"
    assert log_run({}, db_path=db)["status"] == "skipped"
    assert log_run(None, db_path=db)["status"] == "skipped"


def test_unwritable_path_fails_without_raising():
    report = log_run({"audit_trail": _trail(1)}, db_path="\0illegal/nope.db")
    assert report["status"] == "failed" and "error" in report


def test_non_dict_entries_survive(db):
    report = log_run({"audit_trail": ["just a string", 42]}, db_path=db)
    assert report["status"] == "ok" and report["entries_written"] == 2
    assert verify_chain(db_path=db)["valid"] is True


# ── Workflow wiring ─────────────────────────────────────────────────────────────

def test_runner_flushes_the_final_state_to_the_audit_log():
    fake_final = {"audit_trail": _trail(2), "final_recommendation": "x"}
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = fake_final
    with patch("graph.workflow.build_graph", return_value=fake_graph), \
         patch("graph.workflow.learn_async"), \
         patch("graph.workflow.log_run") as lr:
        run_board_with_learning("q")
    lr.assert_called_once_with(fake_final)
