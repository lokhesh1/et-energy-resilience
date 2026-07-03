"""
Decay functions for xMemory.

Decay is event-type driven + signal-reinforcement based:
- Each event category has its own half-life (wars decay slowly, weather fast).
- Every new corroborating signal for a corridor resets the age counter,
  so actively reported crises stay hot regardless of calendar time.
"""
import math
from datetime import datetime, timezone


EVENT_HALF_LIVES: dict[str, float] = {
    "war_conflict":          180.0,
    "sanctions":              90.0,
    "political_tension":      30.0,
    "infrastructure_failure": 21.0,
    "market_spike":           14.0,
    "piracy":                 14.0,
    "weather_disruption":      7.0,
    "none":                   30.0,
}

DEFAULT_HALF_LIFE = 30.0


def compute_decay(
    intensity: float,
    age_days: float,
    event_type: str,
    last_reinforced_days: float = 0.0,
) -> float:
    """
    Return decayed intensity using exponential half-life decay.

    age_days             — days since the memory was first stored
    last_reinforced_days — days since the last corroborating signal;
                           resets effective age to this value (signal reinforcement)
    """
    half_life = EVENT_HALF_LIVES.get(event_type, DEFAULT_HALF_LIFE)
    effective_age = min(age_days, last_reinforced_days) if last_reinforced_days > 0 else age_days
    decayed = intensity * math.pow(2.0, -effective_age / half_life)
    return round(max(0.0, min(1.0, decayed)), 6)


def age_days(timestamp: str) -> float:
    """Return how many days have elapsed since an ISO-8601 timestamp."""
    try:
        ts = datetime.fromisoformat(timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        return max(0.0, delta.total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.0


def apply_decay(
    records: list[dict],
    timestamp_field: str = "timestamp",
    intensity_field: str = "intensity",
    event_type_field: str = "event_type",
    reinforced_field: str = "last_reinforced_at",
) -> list[dict]:
    """
    Walk a list of memory records and attach a `decayed_intensity` field to each.
    Records are returned sorted by decayed_intensity descending (most relevant first).
    """
    result = []
    for rec in records:
        raw_intensity   = float(rec.get(intensity_field, 1.0))
        event_type      = rec.get(event_type_field, "none")
        stored_at       = rec.get(timestamp_field, "")
        reinforced_at   = rec.get(reinforced_field, "")

        age             = age_days(stored_at)
        reinforced_age  = age_days(reinforced_at) if reinforced_at else 0.0

        decayed = compute_decay(raw_intensity, age, event_type, reinforced_age)
        result.append({**rec, "decayed_intensity": decayed})

    result.sort(key=lambda r: r["decayed_intensity"], reverse=True)
    return result
