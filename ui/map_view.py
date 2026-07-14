"""Thin Folium renderer for the digital-twin GeoJSON.

Consumes twin_state["geojson"] produced by tools/geospatial_mapper.py.  Every
feature already carries display-ready properties (tooltip, marker_color), so
this file is a renderer only — no data logic.

No Streamlit import — keeps the module unit-testable offline.  ui/app.py
calls build_folium_map() and hands the result to st_folium().
"""
from __future__ import annotations

import folium
from folium.features import GeoJsonTooltip

_INDIA_CENTER = (21.0, 78.0)
_DEFAULT_ZOOM = 4
_TILES = "cartodbpositron"


def feature_counts(geojson: dict) -> dict[str, int]:
    """Count features by properties.kind (corridor / refinery / reroute)."""
    counts: dict[str, int] = {}
    for f in (geojson or {}).get("features", []) or []:
        kind = (f.get("properties") or {}).get("kind", "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def build_folium_map(geojson: dict, zoom: int = _DEFAULT_ZOOM) -> folium.Map:
    """Build a Folium map from a twin-state GeoJSON FeatureCollection.

    Empty / missing features → bare map centred on India, never raises.
    """
    m = folium.Map(location=_INDIA_CENTER, zoom_start=zoom, tiles=_TILES)

    features = (geojson or {}).get("features", []) or []
    if not features:
        return m

    def style_fn(feature: dict) -> dict:
        props = feature.get("properties") or {}
        color = props.get("marker_color", "blue")
        geom_type = (feature.get("geometry") or {}).get("type", "")
        if geom_type == "LineString":
            return {"color": color, "weight": 3, "opacity": 0.8}
        return {
            "color": color,
            "fillColor": color,
            "fillOpacity": 0.85,
            "weight": 2,
            "radius": 7,
        }

    folium.GeoJson(
        geojson,
        name="Digital Twin",
        style_function=style_fn,
        marker=folium.CircleMarker(radius=7, fill=True, fill_opacity=0.85,
                                   weight=2),
        tooltip=GeoJsonTooltip(fields=["tooltip"], labels=False,
                               sticky=True, style="font-size: 13px;"),
    ).add_to(m)

    return m
