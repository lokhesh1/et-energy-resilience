"""
Economic Impact Modeller (EIM) — the board's economist.

Sits between the bid evaluator and the coordinator. Reads the full board state
(twin, scenarios, recommended mix, corridor risk) and produces:

  Phase 1 — DAMAGE: micro (premium spend, residual loss, refinery throughput
            losses, reroute freight, SPR refill exposure) + macro (Brent spike
            estimate, import bill delta, CPI/CAD-GDP pass-through) + the
            do-nothing counterfactual.

  Phase 2 — RECOVERY: every available lever (committed procurement, SPR drawdown,
            strategic over-buy, demand restraint, export curtailment) ranked by
            net_benefit = avoided_loss − lever_cost, plus a subsidy-vs-pass-through
            policy tradeoff presented separately. Recovery timeline: cumulative
            loss curve + days-to-normal.

Design: FULLY DETERMINISTIC — no LLM. Every number traces to a formula whose
inputs are upstream deterministic values or seeded params. The coordinator's
existing LLM handles any phrasing. Same discipline as SCTD.

Guardrails (placed where a deterministic node actually fails):
  ECON-LIVENESS  — gap > 0 but total exposure = 0 (signal died)
  ECON-CONTRACT  — missing twin/mix inputs (silently reading as 0 would understate)
  ECON-DOMINANCE — do_nothing < best lever (model or disruption may be wrong)

Parameters live in data/econ_params.json (ILLUSTRATIVE — see _note in the file);
baked-in fallback keeps the node running if the file is missing.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from graph.eib_state import EnergyIntelligenceBoard
from tools.price_feed import fetch_price
from tools.spr_calculator import calculate_drawdown

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "econ_params.json"

_DEFAULT_PARAMS = {
    "india_crude_imports_mbd": 4.8,
    "avg_refinery_grm_usd_per_bbl": 8.0,
    "opportunity_cost_usd_per_bbl": 4.0,
    "short_run_demand_elasticity": 0.05,
    "brent_spike_cap_usd": 40.0,
    "global_crude_demand_mbd": 102.0,
    "cpi_bps_per_10usd_brent": 30,
    "cad_pct_gdp_per_10usd_brent": 0.4,
    "india_product_export_mbd": 1.2,
    "avg_product_margin_usd_per_bbl": 12.0,
    "india_domestic_consumption_mbd": 5.5,
    "carry_cost_usd_per_bbl_per_day": 0.05,
    "price_escalation_rate_per_day": 0.002,
    "max_feasible_restraint_pct": 0.05,
    "gdp_oil_intensity_usd_per_bbl": 3.5,
    "freight_rate_usd_per_bbl_per_day": 0.12,
    "spr_refill_premium_usd_per_bbl": 5.0,
}

_BRENT_FALLBACK = 80.0

# mbd → bbl/day conversion
_MBD = 1_000_000


def _load_params() -> tuple[dict, bool]:
    try:
        with open(_PARAMS_PATH, encoding="utf-8") as f:
            return json.load(f), True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _DEFAULT_PARAMS, False


def _p(params: dict, key: str) -> float:
    return float(params.get(key, _DEFAULT_PARAMS.get(key, 0.0)))


def _fetch_brent() -> tuple[float, bool]:
    try:
        res = fetch_price()
        if res.get("status") == "ok":
            price = float(res["data"]["current_price"])
            if price > 0:
                return price, True
    except Exception:
        pass
    return _BRENT_FALLBACK, False


# ── Phase 1: Damage ────────────────────────────────────────────────────────────

def _loss_per_bbl(params: dict) -> float:
    return _p(params, "avg_refinery_grm_usd_per_bbl") + _p(params, "opportunity_cost_usd_per_bbl")


def _micro_damage(params: dict, twin: dict, mix: dict, scenarios: list[dict],
                   brent: float, duration: float) -> dict:
    gap = float(twin.get("total_india_shortfall_mbd", 0.0) or 0.0)
    loss_bbl = _loss_per_bbl(params)

    # Premium spend: Σ (bid_price - brent) × volume × duration
    premium_spend = 0.0
    for c in mix.get("components", []) or []:
        price = float(c.get("price_per_bbl", 0.0) or 0.0)
        vol = float(c.get("volume_mbd", 0.0) or 0.0)
        premium_spend += max(0.0, price - brent) * vol * _MBD * duration
    premium_spend = round(premium_spend)

    # Residual loss: uncovered volume × duration × loss/bbl
    covered = float(mix.get("effective_volume_mbd", mix.get("total_volume_mbd", 0.0)) or 0.0)
    residual = max(0.0, gap - covered)
    residual_loss = round(residual * _MBD * duration * loss_bbl)

    # Per-refinery throughput losses
    grm = _p(params, "avg_refinery_grm_usd_per_bbl")
    refinery_losses = []
    for r in twin.get("refineries", []) or []:
        at_risk = float(r.get("feed_at_risk_mbd", 0.0) or 0.0)
        if at_risk <= 0:
            continue
        loss = round(at_risk * _MBD * duration * grm)
        refinery_losses.append({
            "name": r.get("name", "unknown"),
            "feed_at_risk_mbd": round(at_risk, 4),
            "throughput_loss_usd": loss,
            "status": r.get("status", "normal"),
        })
    refinery_losses.sort(key=lambda x: x["throughput_loss_usd"], reverse=True)

    # Reroute freight cost
    freight_rate = _p(params, "freight_rate_usd_per_bbl_per_day")
    reroute_freight = 0.0
    for sc in scenarios:
        if sc.get("quarantined"):
            continue
        reroute = sc.get("reroute")
        if not reroute:
            continue
        added_days = float(reroute.get("added_transit_days", 0.0) or 0.0)
        india_exp = float(sc.get("india_exposure_mbd", 0.0) or 0.0)
        reroute_freight += india_exp * _MBD * added_days * freight_rate
    reroute_freight = round(reroute_freight)

    # SPR refill exposure
    spr_refill = 0.0
    refill_premium = _p(params, "spr_refill_premium_usd_per_bbl")
    try:
        spr = calculate_drawdown(gap, duration_days=duration)
        d = spr.get("data", {})
        drawdown_mbd = float(d.get("drawdown_mbd", 0.0) or 0.0)
        doc = float(d.get("days_of_cover", 0.0) or 0.0)
        actual_days = min(doc, duration) if doc > 0 else 0.0
        drawn_bbl = drawdown_mbd * _MBD * actual_days
        spr_refill = round(drawn_bbl * refill_premium)
    except Exception:
        pass

    return {
        "premium_spend_usd": premium_spend,
        "residual_loss_usd": residual_loss,
        "refinery_losses": refinery_losses,
        "reroute_freight_usd": reroute_freight,
        "spr_refill_exposure_usd": spr_refill,
    }


def _brent_spike(params: dict, scenarios: list[dict], brent: float) -> dict:
    net_loss_mbd = 0.0
    for sc in scenarios:
        if sc.get("quarantined"):
            continue
        vol = float(sc.get("volume_at_risk_mbd", 0.0) or 0.0)
        reroute = sc.get("reroute")
        if reroute:
            mult = float(reroute.get("freight_cost_mult", 1.0) or 1.0)
            recovered = vol * (1.0 / max(mult, 1.0))
            net_loss_mbd += vol - recovered
        else:
            net_loss_mbd += vol

    elasticity = _p(params, "short_run_demand_elasticity")
    global_demand = _p(params, "global_crude_demand_mbd")
    cap = _p(params, "brent_spike_cap_usd")

    if global_demand <= 0 or elasticity <= 0:
        return {"delta_usd": 0.0, "net_supply_loss_mbd": round(net_loss_mbd, 3),
                "basis": "elasticity_model", "capped": False}

    pct_loss = net_loss_mbd / global_demand
    raw_delta = round(brent * pct_loss / elasticity, 2)
    capped = raw_delta > cap
    delta = min(raw_delta, cap)

    return {
        "delta_usd": round(delta, 2),
        "raw_delta_usd": round(raw_delta, 2),
        "net_supply_loss_mbd": round(net_loss_mbd, 3),
        "pct_global_supply": round(pct_loss * 100, 2),
        "basis": "elasticity_model",
        "capped": capped,
    }


def _macro_damage(params: dict, spike: dict, duration: float, brent: float) -> dict:
    delta = float(spike.get("delta_usd", 0.0))
    imports = _p(params, "india_crude_imports_mbd")

    baseline_daily = imports * _MBD * brent
    spiked_daily = imports * _MBD * (brent + delta)
    import_bill_delta = round((spiked_daily - baseline_daily) * duration)

    cpi_coeff = _p(params, "cpi_bps_per_10usd_brent")
    cpi_bps = round(cpi_coeff * delta / 10.0, 1)

    cad_coeff = _p(params, "cad_pct_gdp_per_10usd_brent")
    cad_pct = round(cad_coeff * delta / 10.0, 2)

    return {
        "import_bill_delta_usd": import_bill_delta,
        "cpi_impact_bps": cpi_bps,
        "cad_gdp_impact_pct": cad_pct,
    }


def _do_nothing_cost(params: dict, gap: float, duration: float) -> float:
    if gap <= 0 or duration <= 0:
        return 0.0
    return round(gap * _MBD * duration * _loss_per_bbl(params))


# ── Phase 2: Recovery ──────────────────────────────────────────────────────────

def _lever_procurement(micro: dict, do_nothing: float, gap: float,
                       covered: float, duration: float,
                       loss_bbl: float) -> dict | None:
    if gap <= 0 or covered <= 0:
        return None
    avoided = round(min(covered, gap) * _MBD * duration * loss_bbl)
    cost = micro["premium_spend_usd"]
    return {
        "lever": "committed_procurement",
        "description": "Execute the recommended cargo mix",
        "avoided_loss_usd": avoided,
        "lever_cost_usd": cost,
        "net_benefit_usd": avoided - cost,
        "feasible": True,
        "time_to_effect_days": 0,
    }


def _lever_spr(gap: float, residual: float, duration: float,
               loss_bbl: float, refill_exposure: float) -> dict | None:
    draw_target = residual if residual > 0 else gap
    if draw_target <= 0:
        return None
    try:
        spr = calculate_drawdown(draw_target, duration_days=duration)
        d = spr.get("data", {})
        drawdown = float(d.get("drawdown_mbd", 0.0) or 0.0)
        if drawdown <= 0:
            return None
        doc = float(d.get("days_of_cover", 0.0) or 0.0)
        actual_days = min(doc, duration) if doc > 0 else 0.0
        avoided = round(drawdown * _MBD * actual_days * loss_bbl)
        return {
            "lever": "spr_drawdown",
            "description": f"Draw SPR at {drawdown} mbd for ~{round(actual_days)} days",
            "avoided_loss_usd": avoided,
            "lever_cost_usd": refill_exposure,
            "net_benefit_usd": avoided - refill_exposure,
            "feasible": True,
            "time_to_effect_days": 1,
            "drawdown_mbd": drawdown,
            "days_of_cover": round(doc, 1),
        }
    except Exception:
        return None


def _lever_overbuy(params: dict, gap: float, covered: float,
                   brent: float, spike_delta: float,
                   duration: float) -> dict | None:
    coverage_cap = 1.3
    headroom = max(0.0, coverage_cap * gap - covered) if gap > 0 else 0.0
    if headroom <= 0.001:
        return None
    escalation = _p(params, "price_escalation_rate_per_day")
    carry = _p(params, "carry_cost_usd_per_bbl_per_day")
    mid_duration = duration / 2.0
    price_saving = spike_delta * escalation * mid_duration
    cost_per_bbl = carry * mid_duration + (brent + spike_delta) * 0.01
    avoided = round(headroom * _MBD * price_saving)
    cost = round(headroom * _MBD * cost_per_bbl)
    if avoided <= 0:
        return None
    return {
        "lever": "strategic_overbuy",
        "description": f"Buy {round(headroom, 3)} mbd beyond gap before spike worsens",
        "avoided_loss_usd": avoided,
        "lever_cost_usd": cost,
        "net_benefit_usd": avoided - cost,
        "feasible": headroom > 0,
        "time_to_effect_days": 0,
        "overbuy_volume_mbd": round(headroom, 4),
    }


def _lever_demand_restraint(params: dict, brent: float, spike_delta: float,
                            duration: float, gap: float = 0.0) -> dict | None:
    if gap <= 0 or duration <= 0:
        return None
    max_pct = _p(params, "max_feasible_restraint_pct")
    if max_pct <= 0:
        return None
    imports = _p(params, "india_crude_imports_mbd")
    spiked_price = brent + spike_delta
    avoided = round(max_pct * imports * _MBD * spiked_price * duration)
    activity_cost_bbl = _p(params, "gdp_oil_intensity_usd_per_bbl")
    cost = round(max_pct * imports * _MBD * activity_cost_bbl * duration)
    return {
        "lever": "demand_restraint",
        "description": f"Reduce demand by {round(max_pct * 100, 1)}%",
        "avoided_loss_usd": avoided,
        "lever_cost_usd": cost,
        "net_benefit_usd": avoided - cost,
        "feasible": True,
        "time_to_effect_days": 3,
        "restraint_pct": max_pct,
    }


def _lever_export_curtailment(params: dict, brent: float,
                              duration: float, gap: float) -> dict | None:
    if gap <= 0 or duration <= 0:
        return None
    export_mbd = _p(params, "india_product_export_mbd")
    if export_mbd <= 0:
        return None
    curtailable = min(export_mbd, gap)
    avoided = round(curtailable * _MBD * brent * duration)
    margin = _p(params, "avg_product_margin_usd_per_bbl")
    cost = round(curtailable * _MBD * margin * duration)
    return {
        "lever": "export_curtailment",
        "description": f"Redirect {round(curtailable, 3)} mbd refined-product exports domestic",
        "avoided_loss_usd": avoided,
        "lever_cost_usd": cost,
        "net_benefit_usd": avoided - cost,
        "feasible": True,
        "time_to_effect_days": 7,
        "curtailable_mbd": round(curtailable, 4),
    }


def _subsidy_vs_passthrough(params: dict, spike_delta: float,
                            duration: float) -> dict | None:
    if spike_delta <= 0:
        return None
    consumption = _p(params, "india_domestic_consumption_mbd")
    fiscal_cost = round(consumption * _MBD * spike_delta * duration)
    cpi_coeff = _p(params, "cpi_bps_per_10usd_brent")
    cpi_bps = round(cpi_coeff * spike_delta / 10.0, 1)
    return {
        "subsidy_fiscal_cost_usd": fiscal_cost,
        "passthrough_cpi_bps": cpi_bps,
        "spike_delta_usd": round(spike_delta, 2),
        "duration_days": round(duration),
    }


def _recovery_timeline(gap: float, duration: float, loss_bbl: float,
                       mix: dict, scenarios: list[dict]) -> dict:
    if gap <= 0 or duration <= 0:
        return {"days_to_normal": 0, "cumulative_do_nothing_usd": 0,
                "cumulative_with_plan_usd": 0, "daily_loss_curve": []}

    components = mix.get("components", []) or []
    timed = []
    for c in components:
        try:
            days = float(c.get("transit_days_to_india", 0) or 0)
            vol = float(c.get("effective_volume_mbd", c.get("volume_mbd", 0)) or 0)
        except (TypeError, ValueError):
            continue
        if vol > 0:
            timed.append((days, vol))
    timed.sort()

    days_int = max(1, int(duration))
    sample_days = min(days_int, 90)

    curve = []
    cum_nothing = 0.0
    cum_plan = 0.0
    days_to_normal = sample_days

    for day in range(1, sample_days + 1):
        daily_nothing = gap * _MBD * loss_bbl
        cum_nothing += daily_nothing

        delivered = sum(vol for d, vol in timed if d <= day)
        remaining_gap = max(0.0, gap - delivered)
        daily_plan = remaining_gap * _MBD * loss_bbl
        cum_plan += daily_plan

        curve.append({
            "day": day,
            "daily_loss_do_nothing_usd": round(daily_nothing),
            "daily_loss_with_plan_usd": round(daily_plan),
            "cumulative_do_nothing_usd": round(cum_nothing),
            "cumulative_with_plan_usd": round(cum_plan),
        })

        if remaining_gap <= 0.001 and days_to_normal == sample_days:
            days_to_normal = day

    return {
        "days_to_normal": days_to_normal,
        "cumulative_do_nothing_usd": round(cum_nothing),
        "cumulative_with_plan_usd": round(cum_plan),
        "savings_usd": round(cum_nothing - cum_plan),
        "daily_loss_curve": curve,
    }


# ── Node ───────────────────────────────────────────────────────────────────────

def eim_node(state: EnergyIntelligenceBoard) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    params, from_file = _load_params()

    twin = state.get("twin_state", {}) or {}
    mix = state.get("recommended_mix", {}) or {}
    scenarios = [s for s in (state.get("scenarios", []) or [])
                 if not s.get("quarantined")]

    gap = float(twin.get("total_india_shortfall_mbd", 0.0) or 0.0)
    covered = float(
        mix.get("effective_volume_mbd", mix.get("total_volume_mbd", 0.0)) or 0.0)
    residual = max(0.0, gap - covered)

    duration = max(
        (float(s.get("duration_days", 0) or 0) for s in scenarios), default=0.0
    ) if scenarios else 0.0

    brent, brent_live = _fetch_brent()
    loss_bbl = _loss_per_bbl(params)

    # ── Guardrail: ECON-CONTRACT ──
    flags: list[dict] = []
    contract_missing = []
    if not twin:
        contract_missing.append("twin_state")
    if not mix and gap > 0:
        contract_missing.append("recommended_mix")
    if contract_missing:
        flags.append({
            "flag": "ECON-CONTRACT",
            "severity": "warn",
            "message": f"Missing inputs: {', '.join(contract_missing)} — "
                       f"economic figures may understate damage",
        })

    # ── Phase 1: Damage ──
    spike = _brent_spike(params, scenarios, brent)
    micro = _micro_damage(params, twin, mix, scenarios, brent, duration)
    macro = _macro_damage(params, spike, duration, brent)
    do_nothing = _do_nothing_cost(params, gap, duration)

    total_exposure = (
        micro["premium_spend_usd"]
        + micro["residual_loss_usd"]
        + micro["reroute_freight_usd"]
        + micro["spr_refill_exposure_usd"]
        + macro["import_bill_delta_usd"]
    )

    # ── Guardrail: ECON-LIVENESS ──
    if gap > 0.001 and total_exposure == 0 and duration > 0:
        flags.append({
            "flag": "ECON-LIVENESS",
            "severity": "warn",
            "message": "Gap > 0 but total economic exposure is zero — "
                       "signal may have been lost in the handoff",
        })

    # ── Phase 2: Recovery levers ──
    levers: list[dict] = []
    lp = _lever_procurement(micro, do_nothing, gap, covered, duration, loss_bbl)
    if lp:
        levers.append(lp)
    ls = _lever_spr(gap, residual, duration, loss_bbl,
                    micro["spr_refill_exposure_usd"])
    if ls:
        levers.append(ls)
    lo = _lever_overbuy(params, gap, covered, brent,
                        spike.get("delta_usd", 0.0), duration)
    if lo:
        levers.append(lo)
    ld = _lever_demand_restraint(params, brent, spike.get("delta_usd", 0.0),
                                 duration, gap=gap)
    if ld:
        levers.append(ld)
    le = _lever_export_curtailment(params, brent, duration, gap)
    if le:
        levers.append(le)

    levers.sort(key=lambda x: x["net_benefit_usd"], reverse=True)

    # ── Guardrail: ECON-DOMINANCE ──
    if do_nothing > 0 and levers:
        best = levers[0]["net_benefit_usd"]
        if best < 0:
            flags.append({
                "flag": "ECON-DOMINANCE",
                "severity": "warn",
                "message": "Every recovery lever costs more than it saves — "
                           "doing nothing may be economically rational, or the "
                           "disruption is too small to act on",
            })

    subsidy = _subsidy_vs_passthrough(params, spike.get("delta_usd", 0.0), duration)

    # ── Timeline ──
    timeline = _recovery_timeline(gap, duration, loss_bbl, mix, scenarios)

    plan_net_benefit = 0
    if do_nothing > 0:
        plan_loss = timeline.get("cumulative_with_plan_usd", 0)
        plan_net_benefit = round(do_nothing - plan_loss)

    economic_impact = {
        "total_exposure_usd": round(total_exposure),
        "do_nothing_cost_usd": do_nothing,
        "plan_net_benefit_usd": plan_net_benefit,
        "brent_current_usd": brent,
        "brent_source": "live" if brent_live else "fallback",
        "brent_spike_estimate": spike,
        "duration_days": round(duration),
        "micro": micro,
        "macro": macro,
        "recovery_actions": levers,
        "subsidy_vs_passthrough": subsidy,
        "recovery_timeline": timeline,
        "guardrail_flags": flags,
        "params_source": "file" if from_file else "defaults",
        "generated_at": now,
    }

    audit = [{
        "agent": "economic_impact",
        "action": "model_impact",
        "gap_mbd": round(gap, 4),
        "duration_days": round(duration),
        "total_exposure_usd": round(total_exposure),
        "do_nothing_cost_usd": do_nothing,
        "plan_net_benefit_usd": plan_net_benefit,
        "levers_evaluated": len(levers),
        "guardrail_flags": flags,
        "brent_usd": brent,
        "brent_source": "live" if brent_live else "fallback",
        "params_source": "file" if from_file else "defaults",
        "timestamp": now,
    }]

    return {
        "current_agent": "economic_impact",
        "economic_impact": economic_impact,
        "audit_trail": audit,
        "constitution_flags": [],
    }
