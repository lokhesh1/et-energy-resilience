"""
geospatial_mapper — turns the digital-twin's current state into GeoJSON that
Folium/Leaflet renders directly.

Pure + deterministic: no LLM, no network, no I/O. Given corridors, refinery
impacts, and reroutes, it emits a FeatureCollection where every feature's
`properties` is display-ready — a pre-formatted `tooltip` (hover) and
`marker_color` — so `ui/map_view.py` stays a thin renderer:

    GeoJsonTooltip(fields=["tooltip"])   # hover
    GeoJsonPopup(fields=["popup"])       # click

SCTD writes the returned `data["geojson"]` into twin_state["geojson"]; the UI
reads it back unchanged. One geometry+display source, tested once.

Degrades safely: a corridor/refinery missing lat/lon is skipped (counted in
data["skipped"]), never crashed — a missing pin is visible, not silent.
"""
from datetime import datetime, timezone

# ── Status → colour maps (Leaflet-friendly names) ───────────────────────────────

_CORRIDOR_COLORS = {
    "open":       "green",
    "restricted": "orange",
    "closed":     "red",
}
_REFINERY_COLORS = {
    "normal":   "green",
    "stressed": "orange",
    "critical": "red",
}
_DEFAULT_COLOR = "gray"


def _point(lon: float, lat: float) -> dict:
    # GeoJSON is [lon, lat] — the reverse of the usual spoken order.
    return {"type": "Point", "coordinates": [round(float(lon), 4), round(float(lat), 4)]}


def _line(lon1: float, lat1: float, lon2: float, lat2: float) -> dict:
    return {
        "type": "LineString",
        "coordinates": [
            [round(float(lon1), 4), round(float(lat1), 4)],
            [round(float(lon2), 4), round(float(lat2), 4)],
        ],
    }


def _corridor_feature(c: dict) -> dict | None:
    if c.get("lat") is None or c.get("lon") is None:
        return None
    status = c.get("status", "open")
    color = _CORRIDOR_COLORS.get(status, _DEFAULT_COLOR)
    name = c.get("name", c.get("id", "corridor"))
    risk = c.get("risk_score")
    flow = c.get("current_flow_mbd", c.get("baseline_flow_mbd"))
    tooltip = f"{name} — {status.upper()}"
    if risk is not None:
        tooltip += f" · risk {risk}"
    return {
        "type": "Feature",
        "geometry": _point(c["lon"], c["lat"]),
        "properties": {
            "kind":              "corridor",
            "id":                c.get("id"),
            "name":              name,
            "status":            status,
            "risk_score":        risk,
            "baseline_flow_mbd": c.get("baseline_flow_mbd"),
            "current_flow_mbd":  flow,
            "marker_color":      color,
            "tooltip":           tooltip,
        },
    }


def _refinery_feature(r: dict) -> dict | None:
    if r.get("lat") is None or r.get("lon") is None:
        return None
    status = r.get("status", "normal")
    color = _REFINERY_COLORS.get(status, _DEFAULT_COLOR)
    name = r.get("name", r.get("id", "refinery"))
    cap = r.get("capacity_mbd")
    feed = r.get("feed_at_risk_mbd", 0.0)
    pct = r.get("at_risk_pct")
    if pct is None and cap:
        pct = round(100.0 * float(feed) / float(cap)) if float(cap) else 0
    tooltip = f"{name} — {status.upper()}"
    if cap:
        tooltip += f" · {feed}/{cap} mbd at risk"
    return {
        "type": "Feature",
        "geometry": _point(r["lon"], r["lat"]),
        "properties": {
            "kind":             "refinery",
            "id":               r.get("id"),
            "name":             name,
            "operator":         r.get("operator"),
            "status":           status,
            "capacity_mbd":     cap,
            "feed_at_risk_mbd": feed,
            "at_risk_pct":      pct,
            "top_corridor":     r.get("top_corridor"),
            "marker_color":     color,
            "tooltip":          tooltip,
        },
    }


def _route_feature(route: dict, coord_lookup: dict[str, tuple]) -> dict | None:
    src = route.get("from_corridor")
    dst = route.get("to_corridor")
    src_c = coord_lookup.get(src)
    dst_c = coord_lookup.get(dst)
    if not src_c or not dst_c:
        return None
    overloaded = bool(route.get("overloaded", False))
    name = f"{src} → {dst}"
    tooltip = f"Reroute {name}"
    added = route.get("added_transit_days")
    if added is not None:
        tooltip += f" · +{added}d"
    if overloaded:
        tooltip += " · OVERLOADED"
    return {
        "type": "Feature",
        "geometry": _line(src_c[1], src_c[0], dst_c[1], dst_c[0]),  # (lon,lat) each end
        "properties": {
            "kind":               "reroute",
            "from_corridor":      src,
            "to_corridor":        dst,
            "added_transit_days": added,
            "freight_cost_mult":  route.get("freight_cost_mult"),
            "volume_mbd":         route.get("volume_mbd"),
            "overloaded":         overloaded,
            "marker_color":       "red" if overloaded else "blue",
            "tooltip":            tooltip,
        },
    }


def build_supply_chain_map(
    corridors: list[dict] | None = None,
    refineries: list[dict] | None = None,
    routes: list[dict] | None = None,
) -> dict:
    """Build a GeoJSON FeatureCollection of the current twin state.

    Returns the standard tool envelope (matches corridor_status). Missing
    coordinates skip that feature and increment data["skipped"] — loud, not silent.
    """
    retrieved_at = datetime.now(timezone.utc)
    corridors = corridors or []
    refineries = refineries or []
    routes = routes or []

    coord_lookup = {
        c["id"]: (c["lat"], c["lon"])
        for c in corridors
        if c.get("id") and c.get("lat") is not None and c.get("lon") is not None
    }

    features: list[dict] = []
    skipped = 0

    for c in corridors:
        f = _corridor_feature(c)
        if f:
            features.append(f)
        else:
            skipped += 1
    for r in refineries:
        f = _refinery_feature(r)
        if f:
            features.append(f)
        else:
            skipped += 1
    for route in routes:
        f = _route_feature(route, coord_lookup)
        if f:
            features.append(f)
        else:
            skipped += 1

    geojson = {"type": "FeatureCollection", "features": features}

    return {
        "tool":   "geospatial_mapper",
        "status": "ok",
        "data": {
            "geojson":         geojson,
            "feature_count":   len(features),
            "corridor_count":  sum(1 for f in features if f["properties"]["kind"] == "corridor"),
            "refinery_count":  sum(1 for f in features if f["properties"]["kind"] == "refinery"),
            "reroute_count":   sum(1 for f in features if f["properties"]["kind"] == "reroute"),
            "skipped":         skipped,
        },
        "source_trust_avg":          1.0,
        "low_trust_sources_flagged": 0,
        "retrieved_at":              retrieved_at.isoformat(),
        "staleness_seconds":         0,
    }
