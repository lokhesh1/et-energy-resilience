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

    return violations


# ── Registry ───────────────────────────────────────────────────────────────────

_CHECKERS = {
    "gri": _check_gri,
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
