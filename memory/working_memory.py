"""
WorkingMemory — in-process scratchpad for a single agent run.

Holds key/value pairs with caller-declared token costs. When a new entry would
push the total over budget, the oldest entries are evicted (FIFO) to make room.
No persistence — each run creates a fresh instance.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class WorkingMemory:
    def __init__(self, budget: int = 4000) -> None:
        self._budget = budget
        self._used = 0
        # insertion-ordered: key -> {"value", "token_cost", "inserted_at"}
        self._store: dict[str, dict] = {}

    # ── Write ──────────────────────────────────────────────────────────────────

    def set(self, key: str, value: Any, token_cost: int) -> bool:
        """
        Store value under key with an estimated token_cost.

        If the key already exists, the old entry is removed first (cost freed).
        Oldest entries are evicted (FIFO) until there is room.
        Returns False if token_cost alone exceeds the total budget.
        """
        if token_cost > self._budget:
            return False

        # Free existing entry for the same key
        if key in self._store:
            self._used -= self._store[key]["token_cost"]
            del self._store[key]

        # Evict oldest entries until there is room
        while self._used + token_cost > self._budget:
            oldest_key = next(iter(self._store))
            self._used -= self._store[oldest_key]["token_cost"]
            del self._store[oldest_key]

        self._store[key] = {
            "value": value,
            "token_cost": token_cost,
            "inserted_at": datetime.now(timezone.utc).isoformat(),
        }
        self._used += token_cost
        return True

    # ── Read ───────────────────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        return entry["value"] if entry is not None else None

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete(self, key: str) -> None:
        if key in self._store:
            self._used -= self._store[key]["token_cost"]
            del self._store[key]

    def clear(self) -> None:
        self._store.clear()
        self._used = 0

    # ── Introspection ──────────────────────────────────────────────────────────

    def budget_remaining(self) -> int:
        return self._budget - self._used

    def snapshot(self) -> dict:
        """Return a plain dict of key → value (no metadata). Used by audit/distiller."""
        return {k: v["value"] for k, v in self._store.items()}

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(budget={self._budget}, used={self._used}, "
            f"remaining={self.budget_remaining()}, entries={len(self._store)})"
        )
