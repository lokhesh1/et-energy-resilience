"""
Distiller pod driver — runs the learning cycle OFF the response path.

A LangGraph node cannot schedule itself, and the board's answer must not wait on
memory writes, so distillation is deliberately NOT a node in the main graph. Instead
this driver runs the pod AFTER a completed run, on its own thread:

    experience_distiller (learn this run) → consolidation (tidy the store)

`learn_from_run` is the synchronous cycle (testable, returns a combined report);
`learn_async` fires it on a daemon thread and returns immediately so the caller —
the query pipeline — hands the user their answer without blocking on learning.

Fully best-effort: both pod nodes already never raise, and each leg is additionally
wrapped so a failure in distillation can't stop consolidation (or crash the thread).
Decoupled cadences: the distiller is meaningful once per run; consolidation is store-
wide housekeeping and can be skipped here (`consolidate=False`) to run on its own
slower clock instead.
"""
import threading

from graph.eib_state import EnergyIntelligenceBoard
from agents.distiller.experience_distiller import experience_distiller_node
from agents.distiller.consolidation_agent import consolidation_node


def learn_from_run(state: EnergyIntelligenceBoard, *, consolidate: bool = True) -> dict:
    """Synchronous learning cycle for one completed run: distill, then optionally
    consolidate. Never raises — each leg is isolated so one failing does not stop
    the other. Returns a combined report of both legs (+ any errors)."""
    report: dict = {"distilled": None, "consolidated": None, "errors": []}

    try:
        result = experience_distiller_node(state)
        report["distilled"] = (result.get("audit_trail") or [None])[0]
    except Exception as exc:  # pragma: no cover — node is already best-effort
        report["errors"].append(f"distill: {exc}")

    if consolidate:
        try:
            result = consolidation_node(state)
            report["consolidated"] = (result.get("audit_trail") or [None])[0]
        except Exception as exc:  # pragma: no cover
            report["errors"].append(f"consolidate: {exc}")

    return report


def learn_async(state: EnergyIntelligenceBoard, *, consolidate: bool = True) -> threading.Thread:
    """Fire the learning cycle on a background daemon thread and return at once.

    The query pipeline calls this right after producing the answer; it never waits.
    Daemon=True so a pending learn never blocks process exit. The Thread is returned
    so a caller (or a test) may `join()` when it genuinely needs to wait for the
    write to land."""
    thread = threading.Thread(
        target=learn_from_run,
        args=(state,),
        kwargs={"consolidate": consolidate},
        name="distiller-pod",
        daemon=True,
    )
    thread.start()
    return thread
