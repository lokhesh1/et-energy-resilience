import xml.etree.ElementTree as ET
from pathlib import Path

CONSTITUTIONS_DIR = Path(__file__).parent.parent / "config" / "constitutions"

KNOWN_CORRIDORS = {
    "strait_of_hormuz", "suez_canal", "malacca_strait", "bab_el_mandeb",
    "turkish_straits", "danish_straits", "cape_of_good_hope", "panama_canal",
}


def load_constitution(agent_name: str) -> list[dict]:
    path = CONSTITUTIONS_DIR / f"{agent_name}_constitution.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    return [
        {"id": r.get("id"), "severity": r.get("severity"), "text": r.text.strip()}
        for r in root.findall("rule")
    ]


# ── GRI rule checks ────────────────────────────────────────────────────────────

def _check_gri(output: dict, rules: list[dict]) -> list[dict]:
    violations = []
    rule_map = {r["id"]: r for r in rules}

    # GRI-02: every risk signal must have trust_score
    for article in output.get("risk_signals", []):
        if "trust_score" not in article:
            violations.append({
                "rule_id": "GRI-02",
                "severity": rule_map["GRI-02"]["severity"],
                "message": f"Article missing trust_score: {article.get('title', 'unknown')}",
            })
            break

    # GRI-03: corridor risk scores must be in [0.0, 1.0]
    for cid, val in output.get("corridor_risk", {}).items():
        score = val.get("score", val) if isinstance(val, dict) else val
        try:
            if not (0.0 <= float(score) <= 1.0):
                violations.append({
                    "rule_id": "GRI-03",
                    "severity": rule_map["GRI-03"]["severity"],
                    "message": f"Score out of range for {cid}: {score}",
                })
        except (TypeError, ValueError):
            violations.append({
                "rule_id": "GRI-03",
                "severity": rule_map["GRI-03"]["severity"],
                "message": f"Non-numeric score for {cid}: {score}",
            })

    # GRI-04: novel corridors must not appear in corridor_risk
    for cid in output.get("corridor_risk", {}):
        if cid not in KNOWN_CORRIDORS:
            violations.append({
                "rule_id": "GRI-04",
                "severity": rule_map["GRI-04"]["severity"],
                "message": f"Unknown corridor in scoring: {cid} — move to novel_corridor_alerts",
            })

    # GRI-05: low-trust sources must be flagged
    actual_low = sum(
        1 for a in output.get("risk_signals", []) if not a.get("trusted", True)
    )
    if actual_low > 0 and output.get("low_trust_signals_flagged", 0) == 0:
        violations.append({
            "rule_id": "GRI-05",
            "severity": rule_map["GRI-05"]["severity"],
            "message": f"{actual_low} low-trust source(s) not flagged",
        })

    # GRI-06: evidence_count must match key_signals length
    for cid, val in output.get("corridor_risk", {}).items():
        if isinstance(val, dict):
            ec = val.get("evidence_count", None)
            ks = val.get("key_signals", [])
            if ec is not None and ec != len(ks):
                violations.append({
                    "rule_id": "GRI-06",
                    "severity": rule_map["GRI-06"]["severity"],
                    "message": f"{cid}: evidence_count={ec} but {len(ks)} key_signals cited",
                })

    # GRI-07: minimum 1 evidence per scored corridor
    for cid, val in output.get("corridor_risk", {}).items():
        if isinstance(val, dict) and val.get("evidence_count", 1) < 1:
            violations.append({
                "rule_id": "GRI-07",
                "severity": rule_map["GRI-07"]["severity"],
                "message": f"No evidence for corridor {cid} — assessment not permitted",
            })

    # GRI-09: the scorecard must not be empty of known corridors. An LLM failure
    # collapses to corridor_risk {} — without this rule that empty scorecard
    # passes every per-corridor check (they iterate over nothing) and the run
    # reads "all corridors nominal" downstream. Only checked when the payload IS
    # an assessment (the tool-fetch check carries no corridor_risk key).
    if "corridor_risk" in output:
        scored = set(output.get("corridor_risk", {}) or {})
        if not (scored & KNOWN_CORRIDORS):
            violations.append({
                "rule_id": "GRI-09",
                "severity": rule_map["GRI-09"]["severity"],
                "message": ("scorecard names no known corridor — assessment "
                            "failed; downstream must not read this as calm"),
            })

    # GRI-08: event_type must be present and valid
    valid_event_types = {
        "war_conflict", "sanctions", "political_tension", "weather_disruption",
        "market_spike", "piracy", "infrastructure_failure", "none",
    }
    for cid, val in output.get("corridor_risk", {}).items():
        if isinstance(val, dict):
            et = val.get("event_type")
            if et is None:
                violations.append({
                    "rule_id": "GRI-08",
                    "severity": rule_map["GRI-08"]["severity"],
                    "message": f"{cid}: missing event_type field",
                })
            elif et not in valid_event_types:
                violations.append({
                    "rule_id": "GRI-08",
                    "severity": rule_map["GRI-08"]["severity"],
                    "message": f"{cid}: invalid event_type '{et}'",
                })

    return violations


# ── DSM rule checks ────────────────────────────────────────────────────────────

_VALID_EVENT_TYPES = {
    "war_conflict", "sanctions", "political_tension", "weather_disruption",
    "market_spike", "piracy", "infrastructure_failure", "none",
}

# Numbers are rounded to 3 decimals, so recompute checks compare within a small
# epsilon rather than demanding exact equality (which rounding would break).
_RECOMPUTE_TOL = 0.011


def _check_dsm(output: dict, rules: list[dict]) -> list[dict]:
    """Validate DSM disruption scenarios. Numbers are deterministic, so these
    checks catch a broken model (would-be silent failure) rather than an LLM."""
    violations = []
    rule_map = {r["id"]: r for r in rules}

    def _flag(rule_id, corridor, message):
        violations.append({
            "rule_id": rule_id,
            "severity": rule_map[rule_id]["severity"],
            "corridor": corridor,
            "message": message,
        })

    for sc in output.get("scenarios", []):
        cid = sc.get("corridor", "unknown")

        # DSM-01: volume in [0, baseline]
        vol = sc.get("volume_at_risk_mbd")
        baseline = sc.get("baseline_flow_mbd")
        try:
            vol_f = float(vol)
            if vol_f < 0:
                _flag("DSM-01", cid, f"{cid}: negative volume_at_risk_mbd {vol}")
            elif baseline is not None and vol_f > float(baseline) + 1e-6:
                _flag("DSM-01", cid, f"{cid}: volume {vol} exceeds baseline {baseline}")
        except (TypeError, ValueError):
            _flag("DSM-01", cid, f"{cid}: non-numeric volume_at_risk_mbd {vol}")

        # DSM-02: duration > 0
        dur = sc.get("duration_days")
        try:
            if float(dur) <= 0:
                _flag("DSM-02", cid, f"{cid}: duration_days must be > 0, got {dur}")
        except (TypeError, ValueError):
            _flag("DSM-02", cid, f"{cid}: non-numeric duration_days {dur}")

        # DSM-04: disruption_fraction in [0, 1]
        frac = sc.get("disruption_fraction")
        try:
            if not (0.0 <= float(frac) <= 1.0):
                _flag("DSM-04", cid, f"{cid}: disruption_fraction out of range {frac}")
        except (TypeError, ValueError):
            _flag("DSM-04", cid, f"{cid}: non-numeric disruption_fraction {frac}")

        # DSM-03: india_exposure <= volume
        india = sc.get("india_exposure_mbd")
        try:
            if india is not None and vol is not None and float(india) > float(vol) + 1e-6:
                _flag("DSM-03", cid, f"{cid}: india_exposure {india} exceeds volume {vol}")
        except (TypeError, ValueError):
            pass

        # DSM-05: known corridor
        if cid not in KNOWN_CORRIDORS:
            _flag("DSM-05", cid, f"Unknown corridor in scenario: {cid}")

        # DSM-06: valid event_type
        et = sc.get("event_type")
        if et not in _VALID_EVENT_TYPES:
            _flag("DSM-06", cid, f"{cid}: invalid or missing event_type '{et}'")

        # DSM-07: volume == baseline × fraction (independent recompute).
        # Catches arithmetic drift that stays within the DSM-01 bound.
        try:
            if baseline is not None and frac is not None and vol is not None:
                expected_vol = float(baseline) * float(frac)
                if abs(float(vol) - expected_vol) > _RECOMPUTE_TOL:
                    _flag("DSM-07", cid,
                          f"{cid}: volume {vol} != baseline×fraction {expected_vol:.3f}")
        except (TypeError, ValueError):
            pass  # non-numeric primitives already flagged by DSM-01/04

        # DSM-08: india_exposure == volume × india_import_share (independent recompute)
        share = sc.get("india_import_share")
        try:
            if vol is not None and share is not None and india is not None:
                expected_india = float(vol) * float(share)
                if abs(float(india) - expected_india) > _RECOMPUTE_TOL:
                    _flag("DSM-08", cid,
                          f"{cid}: india_exposure {india} != volume×share {expected_india:.3f}")
        except (TypeError, ValueError):
            pass

    return violations


# ── Procurement rule checks ──────────────────────────────────────────────────

# Premium band (USD/bbl) — a bid outside this is almost certainly a bug/poisoned
# catalog, not a real quote (our catalog spans roughly -15..+4).
_PREMIUM_BAND = 40.0

# Recommended-mix coverage band: total offered volume vs the shortfall gap.
_COVERAGE_MIN = 0.8
_COVERAGE_MAX = 1.3

# Prices/volumes are rounded, so recompute checks allow a small epsilon.
_PROC_TOL = 0.02


def _check_procurement(output: dict, rules: list[dict]) -> list[dict]:
    """Validate procurement bids + the evaluator's recommended mix.

    Bids are deterministic, so these catch a broken bidder/evaluator or a poisoned
    catalog. Sanctions/price/grade are INDEPENDENTLY re-verified here rather than
    trusting the flags the bidder attached (mirrors the DSM-07/08 recompute idea).

    Expected `output` shape (any subset):
      {"bids": [bid, ...],
       "disrupted_corridors": [corridor_id, ...],
       "affected_refineries": [refinery_id, ...],
       "recommended_mix": {"gap_mbd": float, "total_volume_mbd": float,
                           "components": [bid, ...]}}
    """
    from tools import sanctions_check as _sanctions
    from tools import grade_lookup as _grades

    violations = []
    rule_map = {r["id"]: r for r in rules}

    def _flag(rule_id, bid_id, message):
        violations.append({
            "rule_id": rule_id,
            "severity": rule_map[rule_id]["severity"],
            "bid": bid_id,
            "message": message,
        })

    disrupted = set(output.get("disrupted_corridors", []))
    affected = output.get("affected_refineries", [])

    def _check_bid(bid: dict, in_mix: bool = False) -> None:
        bid_id = bid.get("supplier_id") or bid.get("id") or bid.get("supplier", "unknown")

        # PROC-01: independent SDN re-screen. A sanctioned supplier must be marked
        # blocked, and must never sit inside a recommended mix.
        screen = _sanctions.check_supplier(bid.get("supplier", ""))
        really_sanctioned = screen["status"] == "ok" and screen["data"]["sanctioned"]
        if really_sanctioned:
            if in_mix:
                _flag("PROC-01", bid_id,
                      f"{bid_id}: sanctioned supplier present in recommended mix "
                      f"(matched '{screen['data']['matched_entity']}')")
            elif bid.get("sanctions_status") != "blocked":
                _flag("PROC-01", bid_id,
                      f"{bid_id}: matches SDN ('{screen['data']['matched_entity']}') "
                      f"but sanctions_status is '{bid.get('sanctions_status')}', not 'blocked'")

        # PROC-02: 0 < volume_mbd <= max_volume_mbd
        vol = bid.get("volume_mbd")
        maxv = bid.get("max_volume_mbd")
        try:
            vol_f = float(vol)
            if vol_f <= 0:
                _flag("PROC-02", bid_id, f"{bid_id}: volume_mbd must be > 0, got {vol}")
            elif maxv is not None and vol_f > float(maxv) + 1e-9:
                _flag("PROC-02", bid_id,
                      f"{bid_id}: volume_mbd {vol} exceeds max_volume_mbd {maxv}")
        except (TypeError, ValueError):
            _flag("PROC-02", bid_id, f"{bid_id}: non-numeric volume_mbd {vol}")

        # PROC-03: price positive, premium in band, price == brent_ref + premium
        price = bid.get("price_per_bbl")
        premium = bid.get("price_premium_usd")
        brent = bid.get("brent_ref")
        try:
            if float(price) <= 0:
                _flag("PROC-03", bid_id, f"{bid_id}: price_per_bbl must be > 0, got {price}")
        except (TypeError, ValueError):
            _flag("PROC-03", bid_id, f"{bid_id}: non-numeric price_per_bbl {price}")
        try:
            if abs(float(premium)) > _PREMIUM_BAND:
                _flag("PROC-03", bid_id,
                      f"{bid_id}: price_premium_usd {premium} outside +/-{_PREMIUM_BAND} band")
        except (TypeError, ValueError):
            _flag("PROC-03", bid_id, f"{bid_id}: non-numeric price_premium_usd {premium}")
        try:
            if brent is not None and premium is not None and price is not None:
                expected = float(brent) + float(premium)
                if abs(float(price) - expected) > _PROC_TOL:
                    _flag("PROC-03", bid_id,
                          f"{bid_id}: price_per_bbl {price} != brent_ref+premium {expected:.3f}")
        except (TypeError, ValueError):
            pass  # primitives already flagged above

        # PROC-04 (warn): grade compatible with NO affected refinery.
        grade = bid.get("grade")
        if grade and affected:
            res = _grades.check_grade_against_refineries(grade, affected)
            if res["status"] == "ok" and not res["data"]["any_compatible"]:
                _flag("PROC-04", bid_id,
                      f"{bid_id}: grade '{grade}' incompatible with all "
                      f"{len(affected)} affected refineries")

        # PROC-05 (warn): delivery corridor is itself disrupted.
        corridor = bid.get("delivery_corridor")
        if corridor and corridor in disrupted:
            _flag("PROC-05", bid_id,
                  f"{bid_id}: routes through disrupted corridor '{corridor}'")

    for bid in output.get("bids", []):
        _check_bid(bid)

    # ── Evaluator-level: the recommended mix ──
    mix = output.get("recommended_mix")
    if isinstance(mix, dict):
        components = mix.get("components", [])

        # Re-screen mix members (PROC-01 in-mix teeth).
        for bid in components:
            _check_bid(bid, in_mix=True)

        # PROC-07: reported total == sum of component volumes.
        total = mix.get("total_volume_mbd")
        try:
            recomputed = sum(float(b.get("volume_mbd", 0)) for b in components)
            if total is not None and abs(float(total) - recomputed) > _PROC_TOL:
                _flag("PROC-07", "recommended_mix",
                      f"mix total_volume_mbd {total} != sum of components {recomputed:.3f}")
        except (TypeError, ValueError):
            _flag("PROC-07", "recommended_mix", "non-numeric component volume in mix")
        else:
            total = recomputed if total is None else total

        # PROC-06: coverage of the gap within band.
        gap = mix.get("gap_mbd")
        try:
            gap_f = float(gap)
            if gap_f > 0:
                ratio = float(total) / gap_f
                if not (_COVERAGE_MIN <= ratio <= _COVERAGE_MAX):
                    _flag("PROC-06", "recommended_mix",
                          f"mix covers {ratio:.2f}x the gap "
                          f"({total} vs {gap}) — outside [{_COVERAGE_MIN}, {_COVERAGE_MAX}]")
        except (TypeError, ValueError):
            pass

    return violations


# ── Coordinator rule checks ──────────────────────────────────────────────────

# Plan↔twin arithmetic is rounded upstream, so recompute allows a small epsilon.
_COORD_TOL = 0.02
_ESCALATION_LEVELS = {"routine", "watch", "elevated", "critical"}


def _check_coordinator(output: dict, rules: list[dict]) -> list[dict]:
    """Validate the coordinator's response_plan — the board's final gate.

    Guards the coordinator's OWN decisions (not numbers other constitutions own):
    a false all-clear over an open gap, a sanctioned cargo laundered into the plan,
    plan totals that disagree with the twin, and dropped upstream block-flags.

    Expected `output` shape:
      {"response_plan": {...}, "twin_state": {...},
       "upstream_block_flags": [{"agent","rule_id","message"}, ...]}
    """
    from tools import sanctions_check as _sanctions

    violations = []
    rule_map = {r["id"]: r for r in rules}

    def _flag(rule_id, target, message):
        violations.append({
            "rule_id": rule_id,
            "severity": rule_map[rule_id]["severity"],
            "agent": "crisis_coordinator",
            "target": target,
            "message": message,
        })

    plan = output.get("response_plan") or {}
    situation = plan.get("situation") or {}
    procurement = plan.get("procurement") or {}
    twin = output.get("twin_state") or {}

    escalation = plan.get("escalation_level")
    gap = situation.get("gap_mbd")
    covered = procurement.get("covered_mbd")
    residual = procurement.get("residual_gap_mbd")
    covers_gap = procurement.get("covers_gap", True)

    # COORD-01: no routine/quiet all-clear while a shortfall is uncovered.
    try:
        gap_f = float(gap)
        residual_f = float(residual)
        uncovered = residual_f > _COORD_TOL or (gap_f > 0 and not covers_gap)
        if uncovered and escalation != "critical":
            _flag("COORD-01", "escalation_level",
                  f"uncovered shortfall (residual {residual}, covers_gap {covers_gap}) "
                  f"but escalation_level is '{escalation}', not 'critical'")
    except (TypeError, ValueError):
        _flag("COORD-01", "escalation_level",
              f"non-numeric gap/residual (gap={gap}, residual={residual})")

    # COORD-02: no sanctions-blocked supplier in a committed action (independent
    # re-screen — never trust the flag the plan carries).
    for a in procurement.get("committed_actions", []) or []:
        name = a.get("supplier", "")
        aid = a.get("supplier_id") or name or "unknown"
        if a.get("sanctions_status") == "blocked":
            _flag("COORD-02", aid,
                  f"{aid}: committed action carries sanctions_status 'blocked'")
            continue
        screen = _sanctions.check_supplier(name)
        if screen["status"] == "ok" and screen["data"]["sanctioned"]:
            _flag("COORD-02", aid,
                  f"{aid}: committed supplier matches SDN "
                  f"('{screen['data']['matched_entity']}') — must not be in the plan")

    # COORD-03: plan totals must reconcile with the twin.
    twin_gap = twin.get("total_india_shortfall_mbd")
    try:
        if twin_gap is not None and gap is not None:
            if abs(float(gap) - float(twin_gap)) > _COORD_TOL:
                _flag("COORD-03", "gap_mbd",
                      f"plan gap_mbd {gap} != twin shortfall {twin_gap}")
    except (TypeError, ValueError):
        _flag("COORD-03", "gap_mbd", f"non-numeric gap (plan={gap}, twin={twin_gap})")
    try:
        if gap is not None and covered is not None and residual is not None:
            expected_residual = max(0.0, float(gap) - float(covered))
            if abs(float(residual) - expected_residual) > _COORD_TOL:
                _flag("COORD-03", "residual_gap_mbd",
                      f"residual_gap_mbd {residual} != max(0, gap-covered) "
                      f"{expected_residual:.3f}")
    except (TypeError, ValueError):
        _flag("COORD-03", "residual_gap_mbd",
              f"non-numeric residual (residual={residual}, covered={covered})")

    # COORD-04 (warn): upstream block-flags must be surfaced in the plan.
    upstream = output.get("upstream_block_flags", []) or []
    if upstream and not (plan.get("unresolved_issues") or []):
        _flag("COORD-04", "unresolved_issues",
              f"{len(upstream)} upstream block-flag(s) raised but plan lists no "
              f"unresolved_issues")

    # COORD-05 (warn): escalation vocabulary.
    if escalation not in _ESCALATION_LEVELS:
        _flag("COORD-05", "escalation_level",
              f"unknown escalation_level '{escalation}'")

    return violations


# ── Registry ───────────────────────────────────────────────────────────────────

_CHECKERS = {
    "gri": _check_gri,
    "dsm": _check_dsm,
    "procurement": _check_procurement,
    "coordinator": _check_coordinator,
}


def check(agent_name: str, output: dict) -> dict:
    try:
        rules = load_constitution(agent_name)
    except FileNotFoundError:
        return {"passed": True, "violations": [], "warning": f"No constitution found for {agent_name}"}

    checker = _CHECKERS.get(agent_name)
    violations = checker(output, rules) if checker else []
    blocked = any(v["severity"] == "block" for v in violations)

    return {
        "passed": not blocked,
        "violations": violations,
    }
