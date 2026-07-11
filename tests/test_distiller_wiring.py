"""
Tests for the async distiller wiring — agents/distiller/pod.py + the
`run_board_with_learning` runner in graph/workflow.py.

The pod runs OFF the response path, so these verify:
  * `learn_from_run` runs distiller then consolidation and reports both legs;
  * a failure in one leg is isolated (recorded, doesn't stop the other);
  * `consolidate=False` skips consolidation;
  * `learn_async` runs on a daemon thread, doesn't block, and actually does the work;
  * `run_board_with_learning` returns the board's answer and fires learning in the
    background — and honours `learn=False`.
"""
from unittest.mock import MagicMock, patch

import agents.distiller.pod as pod
from agents.distiller.pod import learn_from_run, learn_async
from graph.workflow import run_board_with_learning


# ── learn_from_run (synchronous cycle) ───────────────────────────────────────────

def test_learn_from_run_runs_both_legs_and_reports():
    distill = MagicMock(return_value={"audit_trail": [{"agent": "experience_distiller",
                                                       "episodic_written": 2}]})
    consol = MagicMock(return_value={"audit_trail": [{"agent": "consolidation_agent",
                                                      "merged": 1}]})
    with patch.object(pod, "experience_distiller_node", distill), \
         patch.object(pod, "consolidation_node", consol):
        report = learn_from_run({"query": "x"})

    distill.assert_called_once()
    consol.assert_called_once()
    assert report["distilled"]["episodic_written"] == 2
    assert report["consolidated"]["merged"] == 1
    assert report["errors"] == []


def test_learn_from_run_skips_consolidation_when_disabled():
    distill = MagicMock(return_value={"audit_trail": [{}]})
    consol = MagicMock()
    with patch.object(pod, "experience_distiller_node", distill), \
         patch.object(pod, "consolidation_node", consol):
        report = learn_from_run({"query": "x"}, consolidate=False)

    consol.assert_not_called()
    assert report["consolidated"] is None


def test_learn_from_run_isolates_a_failing_leg():
    # Distillation blows up; consolidation must still run and the error is recorded.
    distill = MagicMock(side_effect=RuntimeError("boom"))
    consol = MagicMock(return_value={"audit_trail": [{"merged": 0}]})
    with patch.object(pod, "experience_distiller_node", distill), \
         patch.object(pod, "consolidation_node", consol):
        report = learn_from_run({"query": "x"})

    consol.assert_called_once()                       # not stopped by distill failure
    assert report["distilled"] is None
    assert any("distill: boom" in e for e in report["errors"])


# ── learn_async (background thread) ──────────────────────────────────────────────

def test_learn_async_runs_in_background_daemon_thread():
    distill = MagicMock(return_value={"audit_trail": [{}]})
    consol = MagicMock(return_value={"audit_trail": [{}]})
    with patch.object(pod, "experience_distiller_node", distill), \
         patch.object(pod, "consolidation_node", consol):
        thread = learn_async({"query": "x"})
        assert thread.daemon is True                  # never blocks process exit
        thread.join(timeout=5)                         # wait only in the test

    assert not thread.is_alive()
    distill.assert_called_once()
    consol.assert_called_once()


# ── run_board_with_learning (the runner) ─────────────────────────────────────────

def test_runner_returns_answer_and_fires_learning():
    fake_final = {"response_plan": {"escalation_level": "critical"},
                  "final_recommendation": "do X"}
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = fake_final
    with patch("graph.workflow.build_graph", return_value=fake_graph), \
         patch("graph.workflow.learn_async") as la:
        out = run_board_with_learning("Iran closes Hormuz")

    assert out is fake_final                           # answer returned immediately
    la.assert_called_once_with(fake_final, consolidate=True)


def test_runner_can_disable_learning():
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"final_recommendation": "y"}
    with patch("graph.workflow.build_graph", return_value=fake_graph), \
         patch("graph.workflow.learn_async") as la:
        run_board_with_learning("q", learn=False)

    la.assert_not_called()


def test_runner_passes_query_into_initial_state():
    captured = {}
    fake_graph = MagicMock()
    fake_graph.invoke.side_effect = lambda state, config: captured.update(state) or state
    with patch("graph.workflow.build_graph", return_value=fake_graph), \
         patch("graph.workflow.learn_async"):
        run_board_with_learning("Suez blocked", scenario_params={"k": 1})

    assert captured["query"] == "Suez blocked"
    assert captured["scenario_params"] == {"k": 1}
