"""Animated voyage-simulation Leaflet map — pure HTML builder.

No Streamlit import (mirrors ui/map_view.py discipline).  Returns a
self-contained HTML string rendered by ``st.components.v1.html(...)`` in the
Voyage Simulation page.  ``st_folium`` cannot host continuous JS animation;
a raw component iframe can.

Data contract — all fields already exist in state:

* ``mix_rows``: committed_actions from the coordinator's response_plan
  (supplier, supplier_id, grade, volume_mbd, effective_volume_mbd,
  price_per_bbl, delivery_corridor, transit_days, delivery_risk_fraction,
  trade_terms).
* ``twin_state``: the SCTD projection (corridors, refineries, routes —
  same shape geospatial_mapper consumes).
* ``summary``: the full summarize_final dict (corridor_risks, escalation).

Offline seed read: ``data/sea_routes.json`` (waypoints + ports) and
``data/suppliers.json`` (load_port lookup by supplier_id).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"

_MBD_TO_BARRELS = 1_000_000


def _load_json(name: str) -> dict:
    p = _DATA / name
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _sea_routes() -> dict:
    return _load_json("sea_routes.json")


def _suppliers_by_id() -> dict[str, dict]:
    data = _load_json("suppliers.json")
    return {s["id"]: s for s in data.get("suppliers", []) if s.get("id")}


def _corridors_map() -> dict[str, dict]:
    raw = _load_json("corridors.json")
    if isinstance(raw, list):
        return {c["id"]: c for c in raw if c.get("id")}
    return {}


def _refineries_list() -> list[dict]:
    raw = _load_json("refineries.json")
    if isinstance(raw, dict):
        return raw.get("refineries", [])
    if isinstance(raw, list):
        return raw
    return []


# ── Voyage data assembly ─────────────────────────────────────────────────────

def _build_voyage(row: dict, sea: dict, suppliers: dict[str, dict]) -> dict | None:
    """Build one voyage dict from a committed-action row."""
    corridor = row.get("delivery_corridor")
    lane = (sea.get("lanes") or {}).get(corridor)
    if not lane:
        return None

    ports = sea.get("ports") or {}
    sid = row.get("supplier_id") or ""
    sup = suppliers.get(sid, {})
    load_port_name = sup.get("load_port") or row.get("supplier", "")
    load_pt = ports.get(load_port_name)

    discharge = ports.get("_default_india_discharge", {"lat": 22.35, "lon": 69.83})

    path = []
    if load_pt:
        path.append([load_pt["lat"], load_pt["lon"]])
    path.extend(lane)
    path.append([discharge["lat"], discharge["lon"]])

    volume = float(row.get("volume_mbd") or 0)
    effective = float(row.get("effective_volume_mbd") or volume)
    risk_frac = float(row.get("delivery_risk_fraction") or 0)

    return {
        "type": "cargo",
        "supplier": row.get("supplier", "Unknown"),
        "grade": row.get("grade", "—"),
        "volume_mbd": round(volume, 3),
        "effective_volume_mbd": round(effective, 3),
        "barrels_per_day": round(volume * _MBD_TO_BARRELS),
        "price_per_bbl": row.get("price_per_bbl"),
        "transit_days": row.get("transit_days") or 0,
        "delivery_corridor": corridor,
        "delivery_risk_fraction": round(risk_frac, 2),
        "trade_terms": row.get("trade_terms", "FOB"),
        "status": "risky" if risk_frac > 0 else "clear",
        "path": path,
    }


def _build_reroute_voyage(route: dict, sea: dict, corridors: dict) -> dict | None:
    """Build a reroute voyage from twin_state.routes."""
    from_c = route.get("from_corridor", "")
    to_c = route.get("to_corridor", "")
    alt_lane = (sea.get("lanes") or {}).get(to_c)
    orig_lane = (sea.get("lanes") or {}).get(from_c)
    if not alt_lane:
        return None

    from_coords = corridors.get(from_c, {})
    blocked_at = [from_coords.get("lat"), from_coords.get("lon")]
    if blocked_at[0] is None:
        blocked_at = None

    return {
        "type": "reroute",
        "from_corridor": from_c,
        "to_corridor": to_c,
        "volume_mbd": round(float(route.get("volume_mbd") or 0), 3),
        "added_transit_days": route.get("added_transit_days"),
        "freight_cost_mult": route.get("freight_cost_mult"),
        "overloaded": bool(route.get("overloaded", False)),
        "path": list(alt_lane),
        "original_path": list(orig_lane) if orig_lane else None,
        "blocked_at": blocked_at,
    }


def _build_corridor_pins(twin_state: dict, corridors: dict) -> list[dict]:
    """Corridor pins with status from twin_state or baseline."""
    corridor_risks = {}
    for cr in (twin_state.get("corridor_risks") or []):
        cid = cr.get("corridor") or cr.get("id")
        if cid:
            corridor_risks[cid] = cr

    pins = []
    for cid, c in corridors.items():
        if c.get("lat") is None:
            continue
        cr = corridor_risks.get(cid, {})
        risk = cr.get("risk_score") or cr.get("score", 0)
        status = "disrupted" if risk >= 0.4 else "open"
        pins.append({
            "id": cid,
            "name": c.get("name", cid),
            "lat": c["lat"],
            "lon": c["lon"],
            "risk_score": round(float(risk), 2),
            "status": status,
            "baseline_flow_mbd": c.get("baseline_flow_mbd"),
        })
    return pins


def _build_refinery_pins(twin_state: dict) -> list[dict]:
    """Refinery pins from twin_state impacts."""
    pins = []
    for r in (twin_state.get("impacts") or twin_state.get("refineries") or []):
        if r.get("lat") is None:
            continue
        pins.append({
            "id": r.get("id"),
            "name": r.get("name", ""),
            "lat": r["lat"],
            "lon": r["lon"],
            "status": r.get("status", "normal"),
            "capacity_mbd": r.get("capacity_mbd"),
            "feed_at_risk_mbd": r.get("feed_at_risk_mbd", 0),
        })
    return pins


def build_voyages(mix_rows: list[dict] | None,
                  twin_state: dict | None,
                  summary: dict | None) -> dict:
    """Assemble all voyage data for the simulation map.

    Returns a dict with keys: voyages, reroutes, corridor_pins, refinery_pins,
    is_baseline.  Pure function, no I/O beyond the data-file reads.
    """
    sea = _sea_routes()
    suppliers = _suppliers_by_id()
    corridors = _corridors_map()
    twin_state = twin_state or {}
    summary = summary or {}
    mix_rows = mix_rows or []

    voyages = []
    for row in mix_rows:
        v = _build_voyage(row, sea, suppliers)
        if v:
            voyages.append(v)

    reroutes = []
    for route in (twin_state.get("routes") or []):
        rv = _build_reroute_voyage(route, sea, corridors)
        if rv:
            reroutes.append(rv)

    corridor_pins = _build_corridor_pins(twin_state, corridors)
    refinery_pins = _build_refinery_pins(twin_state)
    is_baseline = len(voyages) == 0 and len(reroutes) == 0

    if is_baseline:
        voyages = _baseline_voyages(sea, corridors)

    return {
        "voyages": voyages,
        "reroutes": reroutes,
        "corridor_pins": corridor_pins,
        "refinery_pins": refinery_pins,
        "is_baseline": is_baseline,
    }


def _baseline_voyages(sea: dict, corridors: dict) -> list[dict]:
    """Ambient baseline ships when no board run has happened."""
    lanes = sea.get("lanes") or {}
    baseline = []
    for cid in ("strait_of_hormuz", "cape_of_good_hope", "malacca_strait"):
        lane = lanes.get(cid)
        if not lane:
            continue
        c = corridors.get(cid, {})
        baseline.append({
            "type": "baseline",
            "supplier": f"Baseline flow ({c.get('name', cid)})",
            "grade": "—",
            "volume_mbd": round(float(c.get("baseline_flow_mbd", 0)), 1),
            "effective_volume_mbd": round(float(c.get("baseline_flow_mbd", 0)), 1),
            "barrels_per_day": round(float(c.get("baseline_flow_mbd", 0)) * _MBD_TO_BARRELS),
            "price_per_bbl": None,
            "transit_days": 15,
            "delivery_corridor": cid,
            "delivery_risk_fraction": 0,
            "trade_terms": "—",
            "status": "clear",
            "path": list(lane),
        })
    return baseline


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_sim_map_html(mix_rows: list[dict] | None = None,
                       twin_state: dict | None = None,
                       summary: dict | None = None,
                       height: int = 650) -> str:
    """Return a self-contained HTML string with an animated Leaflet map.

    Rendered via ``st.components.v1.html(html, height=height)``.
    """
    data = build_voyages(mix_rows, twin_state, summary)
    data_json = json.dumps(data, default=str)

    return _HTML_TEMPLATE.replace("__DATA_JSON__", data_json).replace(
        "__HEIGHT__", str(height)
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Voyage Simulation</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body,#map{width:100%;height:__HEIGHT__px;background:#1a1a2e}
.leaflet-container{background:#1a1a2e}

/* Controls overlay */
.sim-controls{
  position:absolute;top:10px;right:10px;z-index:1000;
  background:rgba(26,26,46,0.92);border-radius:10px;padding:12px 16px;
  display:flex;gap:10px;align-items:center;
  font-family:'Segoe UI',system-ui,sans-serif;color:#e0e0e0;font-size:13px;
  box-shadow:0 4px 20px rgba(0,0,0,0.4);backdrop-filter:blur(8px);
  border:1px solid rgba(255,255,255,0.08);
}
.sim-controls button{
  background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.15);
  color:#e0e0e0;border-radius:6px;padding:5px 12px;cursor:pointer;
  font-size:13px;transition:all 0.2s;
}
.sim-controls button:hover{background:rgba(255,255,255,0.2)}
.sim-controls button.active{background:rgba(59,130,246,0.5);border-color:rgba(59,130,246,0.6)}
.sim-controls .speed-group{display:flex;gap:4px}

/* Legend */
.sim-legend{
  position:absolute;bottom:30px;left:10px;z-index:1000;
  background:rgba(26,26,46,0.92);border-radius:10px;padding:14px 18px;
  font-family:'Segoe UI',system-ui,sans-serif;color:#e0e0e0;font-size:12px;
  box-shadow:0 4px 20px rgba(0,0,0,0.4);backdrop-filter:blur(8px);
  border:1px solid rgba(255,255,255,0.08);min-width:170px;
}
.sim-legend h4{margin:0 0 8px;font-size:13px;font-weight:600;color:#93c5fd;letter-spacing:0.03em}
.sim-legend .row{display:flex;align-items:center;gap:8px;margin:5px 0}
.sim-legend .dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.sim-legend .line-sample{width:24px;height:3px;border-radius:2px;flex-shrink:0}

/* Baseline caption */
.baseline-caption{
  position:absolute;top:10px;left:50%;transform:translateX(-50%);z-index:1000;
  background:rgba(26,26,46,0.88);border-radius:8px;padding:8px 18px;
  font-family:'Segoe UI',system-ui,sans-serif;color:#93c5fd;font-size:13px;
  box-shadow:0 2px 12px rgba(0,0,0,0.3);border:1px solid rgba(147,197,253,0.2);
  display:none;
}

/* Tooltip overrides */
.leaflet-tooltip{
  background:rgba(26,26,46,0.95) !important;color:#e0e0e0 !important;
  border:1px solid rgba(255,255,255,0.12) !important;border-radius:8px !important;
  padding:10px 14px !important;font-family:'Segoe UI',system-ui,sans-serif !important;
  font-size:12px !important;box-shadow:0 4px 16px rgba(0,0,0,0.4) !important;
  max-width:280px !important;
}
.leaflet-tooltip .tt-title{font-weight:600;font-size:13px;margin-bottom:4px;color:#93c5fd}
.leaflet-tooltip .tt-row{display:flex;justify-content:space-between;gap:12px;margin:2px 0}
.leaflet-tooltip .tt-label{color:#9ca3af}
.leaflet-tooltip .tt-value{font-weight:500;text-align:right}
.leaflet-tooltip .tt-warn{color:#fbbf24}
.leaflet-tooltip .tt-risk{color:#ef4444}

/* Pulsing disrupted corridors */
@keyframes pulse-ring{
  0%{transform:scale(1);opacity:0.7}
  100%{transform:scale(2.5);opacity:0}
}
.pulse-marker{position:relative}
.pulse-marker .ring{
  position:absolute;top:50%;left:50%;width:16px;height:16px;
  margin:-8px 0 0 -8px;border-radius:50%;
  border:2px solid #ef4444;animation:pulse-ring 1.8s ease-out infinite;
}

/* Ship icon */
.ship-icon{transition:transform 0.1s linear}
</style>
</head>
<body>
<div id="map"></div>
<div class="sim-controls" id="controls">
  <button id="btn-play" onclick="togglePlay()" title="Play/Pause">&#9654; Play</button>
  <div class="speed-group">
    <button class="active" data-speed="1" onclick="setSpeed(1,this)">1&times;</button>
    <button data-speed="2" onclick="setSpeed(2,this)">2&times;</button>
    <button data-speed="4" onclick="setSpeed(4,this)">4&times;</button>
  </div>
  <span id="sim-clock" style="min-width:70px;text-align:center">Day 0</span>
</div>
<div class="sim-legend" id="legend">
  <h4>Voyage Simulation</h4>
  <div class="row"><div class="dot" style="background:#22c55e"></div><span>Safe route</span></div>
  <div class="row"><div class="dot" style="background:#f59e0b"></div><span>Reroute / risky</span></div>
  <div class="row"><div class="dot" style="background:#ef4444"></div><span>Disrupted corridor</span></div>
  <div class="row"><div class="line-sample" style="background:#ef4444;opacity:0.5"></div><span>Blocked lane</span></div>
  <div class="row"><div class="dot" style="background:#3b82f6"></div><span>Refinery (normal)</span></div>
  <div class="row"><div class="dot" style="background:#f97316"></div><span>Refinery (stressed)</span></div>
  <div class="row"><div class="dot" style="background:#dc2626"></div><span>Refinery (critical)</span></div>
</div>
<div class="baseline-caption" id="baseline-caption">
  Baseline traffic — run a scenario to simulate a disruption
</div>

<script>
(function(){
  // ── Data ──
  var D = __DATA_JSON__;

  // ── Map init ──
  var map = L.map('map',{center:[15,55],zoom:3,zoomControl:true,
    attributionControl:false,preferCanvas:true});
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{
    maxZoom:18,subdomains:'abcd'
  }).addTo(map);

  if(D.is_baseline) document.getElementById('baseline-caption').style.display='block';

  // ── Ship SVG ──
  function shipSvg(color,size){
    size=size||20;
    return '<svg xmlns="http://www.w3.org/2000/svg" width="'+size+'" height="'+size+'" viewBox="0 0 24 24" fill="none" stroke="'+color+'" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
      +'<path d="M2 21c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1 .6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>'
      +'<path d="M19.38 20A11.6 11.6 0 0 0 21 14l-9-4-9 4c0 2.9.94 5.34 2.81 7.76"/>'
      +'<path d="M19 13V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6"/>'
      +'<path d="M12 2v3"/>'
      +'</svg>';
  }

  // ── Helpers ──
  function dist(a,b){
    var dx=a[0]-b[0],dy=a[1]-b[1];return Math.sqrt(dx*dx+dy*dy);
  }
  function lerp(a,b,t){return[a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t];}
  function bearing(a,b){
    return Math.atan2(b[1]-a[1],b[0]-a[0])*180/Math.PI;
  }
  function cumDists(path){
    var d=[0];
    for(var i=1;i<path.length;i++) d.push(d[i-1]+dist(path[i-1],path[i]));
    return d;
  }
  function posOnPath(path,dists,frac){
    var total=dists[dists.length-1];
    if(total===0) return{pos:path[0],angle:0};
    var target=frac*total;
    for(var i=1;i<dists.length;i++){
      if(dists[i]>=target){
        var segLen=dists[i]-dists[i-1];
        var t=segLen>0?(target-dists[i-1])/segLen:0;
        var pos=lerp(path[i-1],path[i],t);
        var angle=bearing(path[i-1],path[i]);
        return{pos:pos,angle:angle};
      }
    }
    return{pos:path[path.length-1],angle:0};
  }

  // ── Corridor pins ──
  var corridorColors={open:'#22c55e',disrupted:'#ef4444'};
  D.corridor_pins.forEach(function(c){
    var col=corridorColors[c.status]||'#6b7280';
    var circle=L.circleMarker([c.lat,c.lon],{
      radius:8,fillColor:col,color:col,weight:2,opacity:0.9,fillOpacity:0.7
    }).addTo(map);
    circle.bindTooltip(
      '<div class="tt-title">'+c.name+'</div>'
      +'<div class="tt-row"><span class="tt-label">Status</span><span class="tt-value'
      +(c.status==='disrupted'?' tt-risk':'')+'">'
      +c.status.toUpperCase()+'</span></div>'
      +'<div class="tt-row"><span class="tt-label">Risk</span><span class="tt-value">'
      +c.risk_score+'</span></div>'
      +(c.baseline_flow_mbd?'<div class="tt-row"><span class="tt-label">Flow</span><span class="tt-value">'
      +c.baseline_flow_mbd+' mbd</span></div>':'')
    );
    if(c.status==='disrupted'){
      var pulseHtml='<div class="pulse-marker"><div class="ring"></div></div>';
      L.marker([c.lat,c.lon],{
        icon:L.divIcon({className:'',html:pulseHtml,iconSize:[16,16],iconAnchor:[8,8]}),
        interactive:false
      }).addTo(map);
    }
  });

  // ── Refinery pins ──
  var refColors={normal:'#3b82f6',stressed:'#f97316',critical:'#dc2626'};
  D.refinery_pins.forEach(function(r){
    var col=refColors[r.status]||'#3b82f6';
    L.circleMarker([r.lat,r.lon],{
      radius:5,fillColor:col,color:'#1e293b',weight:1.5,fillOpacity:0.85
    }).addTo(map).bindTooltip(
      '<div class="tt-title">'+r.name+'</div>'
      +'<div class="tt-row"><span class="tt-label">Status</span><span class="tt-value'
      +(r.status==='critical'?' tt-risk':r.status==='stressed'?' tt-warn':'')+'">'
      +r.status.toUpperCase()+'</span></div>'
      +(r.capacity_mbd?'<div class="tt-row"><span class="tt-label">Capacity</span><span class="tt-value">'
      +r.capacity_mbd+' mbd</span></div>':'')
      +(r.feed_at_risk_mbd?'<div class="tt-row"><span class="tt-label">At risk</span><span class="tt-value tt-warn">'
      +r.feed_at_risk_mbd+' mbd</span></div>':'')
    );
  });

  // ── Draw reroute blocked lanes ──
  D.reroutes.forEach(function(rv){
    if(rv.original_path && rv.original_path.length>1){
      L.polyline(rv.original_path,{color:'#ef4444',weight:2.5,opacity:0.35,
        dashArray:'8 6'}).addTo(map);
    }
    if(rv.blocked_at){
      L.circleMarker(rv.blocked_at,{radius:10,fillColor:'#ef4444',
        color:'#fca5a5',weight:2,fillOpacity:0.6}).addTo(map)
        .bindTooltip('<div class="tt-title">⊘ BLOCKED</div><div>'+rv.from_corridor+'</div>');
    }
    if(rv.path && rv.path.length>1){
      L.polyline(rv.path,{color:'#f59e0b',weight:2,opacity:0.4,
        dashArray:'6 4'}).addTo(map);
    }
  });

  // ── Draw voyage paths ──
  D.voyages.forEach(function(v){
    if(!v.path||v.path.length<2) return;
    var col=v.status==='risky'?'#f59e0b':v.type==='baseline'?'rgba(34,197,94,0.3)':'#22c55e';
    L.polyline(v.path,{color:col,weight:1.8,opacity:v.type==='baseline'?0.25:0.4}).addTo(map);
  });

  // ── Animated ships ──
  var ships=[];

  function makeTooltip(v){
    var html='<div class="tt-title">'+v.supplier+'</div>';
    html+='<div class="tt-row"><span class="tt-label">Grade</span><span class="tt-value">'+v.grade+'</span></div>';
    html+='<div class="tt-row"><span class="tt-label">Volume</span><span class="tt-value">'+v.volume_mbd+' mbd</span></div>';
    if(v.barrels_per_day) html+='<div class="tt-row"><span class="tt-label">Barrels/day</span><span class="tt-value">'+v.barrels_per_day.toLocaleString()+'</span></div>';
    if(v.effective_volume_mbd && v.effective_volume_mbd!==v.volume_mbd)
      html+='<div class="tt-row"><span class="tt-label">Expected delivery</span><span class="tt-value tt-warn">'+v.effective_volume_mbd+' mbd</span></div>';
    if(v.price_per_bbl) html+='<div class="tt-row"><span class="tt-label">Price</span><span class="tt-value">$'+v.price_per_bbl+'/bbl</span></div>';
    html+='<div class="tt-row"><span class="tt-label">Transit</span><span class="tt-value">'+v.transit_days+' days</span></div>';
    if(v.delivery_corridor) html+='<div class="tt-row"><span class="tt-label">Corridor</span><span class="tt-value">'+v.delivery_corridor.replace(/_/g,' ')+'</span></div>';
    if(v.delivery_risk_fraction>0)
      html+='<div class="tt-row"><span class="tt-label">Corridor risk</span><span class="tt-value tt-risk">'+Math.round(v.delivery_risk_fraction*100)+'%</span></div>';
    if(v.trade_terms && v.trade_terms!=='—')
      html+='<div class="tt-row"><span class="tt-label">Terms</span><span class="tt-value">'+v.trade_terms+'</span></div>';
    return html;
  }

  function makeRerouteTooltip(rv){
    var html='<div class="tt-title">Reroute: '+rv.from_corridor.replace(/_/g,' ')+' → '+rv.to_corridor.replace(/_/g,' ')+'</div>';
    html+='<div class="tt-row"><span class="tt-label">Volume</span><span class="tt-value">'+rv.volume_mbd+' mbd</span></div>';
    if(rv.added_transit_days) html+='<div class="tt-row"><span class="tt-label">Added time</span><span class="tt-value tt-warn">+'+rv.added_transit_days+' days</span></div>';
    if(rv.freight_cost_mult) html+='<div class="tt-row"><span class="tt-label">Freight mult.</span><span class="tt-value">'+rv.freight_cost_mult+'×</span></div>';
    if(rv.overloaded) html+='<div class="tt-row"><span class="tt-label">Status</span><span class="tt-value tt-risk">OVERLOADED</span></div>';
    return html;
  }

  // Create ship markers for voyages
  D.voyages.forEach(function(v){
    if(!v.path||v.path.length<2) return;
    var col=v.status==='risky'?'#f59e0b':v.type==='baseline'?'#22c55e':'#22c55e';
    var size=v.type==='baseline'?16:22;
    var marker=L.marker(v.path[0],{
      icon:L.divIcon({className:'ship-icon',html:shipSvg(col,size),
        iconSize:[size,size],iconAnchor:[size/2,size/2]})
    }).addTo(map);
    marker.bindTooltip(makeTooltip(v),{sticky:true});
    ships.push({marker:marker,path:v.path,dists:cumDists(v.path),
      transit:v.transit_days||15,offset:Math.random()*0.15});
  });

  // Create ship markers for reroutes
  D.reroutes.forEach(function(rv){
    if(!rv.path||rv.path.length<2) return;
    var col=rv.overloaded?'#ef4444':'#f59e0b';
    var marker=L.marker(rv.path[0],{
      icon:L.divIcon({className:'ship-icon',html:shipSvg(col,20),
        iconSize:[20,20],iconAnchor:[10,10]})
    }).addTo(map);
    marker.bindTooltip(makeRerouteTooltip(rv),{sticky:true});
    ships.push({marker:marker,path:rv.path,dists:cumDists(rv.path),
      transit:rv.added_transit_days||20,offset:Math.random()*0.15});
  });

  // ── Animation loop ──
  var playing=false, speed=1, simTime=0, lastTs=null;
  var maxTransit=Math.max.apply(null,ships.map(function(s){return s.transit}))||30;
  var clockEl=document.getElementById('sim-clock');
  var btnPlay=document.getElementById('btn-play');

  function animate(ts){
    if(!playing){lastTs=null;requestAnimationFrame(animate);return;}
    if(!lastTs) lastTs=ts;
    var dt=(ts-lastTs)/1000;
    lastTs=ts;
    simTime+=dt*speed;
    var day=simTime;
    clockEl.textContent='Day '+Math.floor(day);

    ships.forEach(function(s){
      var frac=((day+s.offset*s.transit)%s.transit)/s.transit;
      var r=posOnPath(s.path,s.dists,frac);
      s.marker.setLatLng(r.pos);
      var el=s.marker.getElement();
      if(el){
        var svg=el.querySelector('svg');
        if(svg) svg.style.transform='rotate('+(r.angle-90)+'deg)';
      }
    });
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);

  window.togglePlay=function(){
    playing=!playing;
    btnPlay.innerHTML=playing?'&#9646;&#9646; Pause':'&#9654; Play';
    if(playing) lastTs=null;
  };
  window.setSpeed=function(s,btn){
    speed=s;
    document.querySelectorAll('.speed-group button').forEach(function(b){b.classList.remove('active')});
    btn.classList.add('active');
  };

  // Auto-play after 500ms
  setTimeout(function(){window.togglePlay();},500);
})();
</script>
</body>
</html>
"""
