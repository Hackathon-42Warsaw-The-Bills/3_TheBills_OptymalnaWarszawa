"""Eksportuje graf z data/ do lekkiego formatu JSON dla wizualizacji (viz/).

Odcinki u->v i v->u tej samej drogi łączymy w jedną linię (z listą obu edge_id),
współrzędne zapisujemy różnicowo jako liczby całkowite (1e-5 stopnia ≈ 1 m).

Uruchomienie (po build_graph.py):  .venv/bin/python mapping/export_viz.py
"""

import json
import os
import re
from pathlib import Path

import geopandas as gpd
import osmium
from shapely.geometry import LineString
from shapely.ops import linemerge

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "viz"
PBF_PATH = DATA_DIR / "osm" / "mazowieckie-latest.osm.pbf"
# Dzielnice i przystanki ZTM (zbiór "dane_warszawa"), ścieżkę można nadpisać.
EXTERNAL_DIR = Path(os.environ.get("DANE_WARSZAWA", Path.home() / "Downloads" / "dane_warszawa"))
SCALE = 100_000
SIMPLIFY = 0.00002  # ≈ 2 m

# Klasy od najważniejszej; pierwsza pasująca wygrywa przy scalonych odcinkach.
CLASSES = [
    ("express", {"motorway", "trunk"}),
    ("primary", {"primary"}),
    ("secondary", {"secondary"}),
    ("tertiary", {"tertiary"}),
    ("local", {"unclassified", "residential", "living_street", "busway"}),
    ("service", {"service"}),
]


def road_class(highway: str) -> int:
    kinds = {h.removesuffix("_link") for h in str(highway).split(";")}
    for i, (_, members) in enumerate(CLASSES):
        if kinds & members:
            return i
    return len(CLASSES) - 1


def first_int(value, urban=50) -> int:
    if value is None or str(value) in ("nan", "<NA>"):
        return 0
    s = str(value)
    if "urban" in s:
        return urban
    m = re.search(r"\d+", s)
    return int(m.group()) if m else 0


def encode(geom) -> list[int]:
    out, px, py = [], 0, 0
    for x, y in geom.coords:
        ix, iy = round(x * SCALE), round(y * SCALE)
        out += [ix - px, iy - py]
        px, py = ix, iy
    return out


def has_bus_priority(row) -> bool:
    return any(
        str(row.get(c)) not in ("None", "nan", "<NA>", "no")
        for c in ("busway", "lanes:bus", "lanes:psv")
    ) or str(row.get("highway")) == "busway"


def export_roads() -> dict:
    e = gpd.read_parquet(DATA_DIR / "roads_edges.parquet")
    e["pair"] = [
        (min(u, v), max(u, v), round(length)) for u, v, length in zip(e.u, e.v, e["length"])
    ]
    names: dict[str, int] = {"": 0}
    edges = []
    for _, group in e.groupby("pair", sort=False):
        row = group.iloc[0]
        name = row["name"] if isinstance(row["name"], str) else ""
        name = name.split(";")[0]
        flags = (
            (1 if len(group) == 1 and str(row["oneway"]) == "True" else 0)
            | (2 if has_bus_priority(row) else 0)
            | (4 if str(row["bridge"]) not in ("None", "nan", "<NA>", "no") else 0)
            | (8 if str(row["tunnel"]) not in ("None", "nan", "<NA>", "no") else 0)
        )
        edges.append([
            road_class(row["highway"]),
            names.setdefault(name, len(names)),
            round(row["length"]),
            first_int(row["lanes"]),
            first_int(row["maxspeed"]),
            flags,
            ";".join(group["edge_id"]),
            encode(row.geometry.simplify(SIMPLIFY)),
        ])
    return {
        "classes": [c for c, _ in CLASSES],
        "names": list(names),
        "fields": ["cls", "name", "len", "lanes", "speed", "flags", "ids", "coords"],
        "edges": edges,
        "stats": {
            "directed_edges": len(e),
            "nodes": int(len(gpd.read_parquet(DATA_DIR / "roads_nodes.parquet"))),
            "km": round(sum(edge[2] for edge in edges) / 1000),
        },
    }


def export_trams() -> dict:
    e = gpd.read_parquet(DATA_DIR / "trams_edges.parquet")
    e["pair"] = [(min(u, v), max(u, v), round(length)) for u, v, length in zip(e.u, e.v, e["length"])]
    lines = [
        [round(g.iloc[0]["length"]), ";".join(g["edge_id"]), encode(g.iloc[0].geometry.simplify(SIMPLIFY))]
        for _, g in e.groupby("pair", sort=False)
    ]
    return {"edges": lines, "km": round(sum(l[0] for l in lines) / 1000)}


def export_river(bounds) -> list[list[int]]:
    # Dwa przejścia zamiast indeksu lokalizacji całego województwa (za dużo RAM):
    # najpierw drogi wodne Wisły, potem tylko ich węzły.
    ways = [
        [n.ref for n in w.nodes]
        for w in osmium.FileProcessor(str(PBF_PATH), osmium.osm.WAY)
        if w.tags.get("waterway") == "river" and w.tags.get("name") == "Wisła"
    ]
    wanted = {ref for refs in ways for ref in refs}
    loc = {
        n.id: (n.lon, n.lat)
        for n in osmium.FileProcessor(str(PBF_PATH), osmium.osm.NODE)
        .with_filter(osmium.filter.IdFilter(wanted))
    }
    min_lon, min_lat, max_lon, max_lat = bounds
    parts = []
    for refs in ways:
        pts = [loc[r] for r in refs if r in loc]
        if len(pts) > 1 and any(
            min_lon - 0.05 < x < max_lon + 0.05 and min_lat - 0.05 < y < max_lat + 0.05 for x, y in pts
        ):
            parts.append(LineString(pts))
    merged = linemerge(parts)
    geoms = getattr(merged, "geoms", [merged])
    return [encode(g.simplify(0.0002)) for g in geoms]


def export_districts() -> list:
    d = gpd.read_file(EXTERNAL_DIR / "warszawa_dzielnice.geojson")
    out = []
    for _, row in d.iterrows():
        geom = row.geometry.simplify(0.0002)
        rings = [p.exterior for p in getattr(geom, "geoms", [geom])]
        c = row.geometry.representative_point()
        out.append({
            "name": row["dzielnica"],
            "pop": int(row["ludnosc_gus_2025"]),
            "km2": float(row["pow_km2"]),
            "stops": int(row["przystanki_slupki"]),
            "label": [round(c.x * SCALE), round(c.y * SCALE)],
            "rings": [encode(r) for r in rings],
        })
    return out


def export_stops() -> list:
    s = gpd.read_file(EXTERNAL_DIR / "warszawa_przystanki_ztm.geojson")
    return [
        [round(r.geometry.x * SCALE), round(r.geometry.y * SCALE),
         r["nazwa_zespolu"], f'{r["zespol"]}-{r["slupek"]}', r["linie_tram"] or ""]
        for _, r in s.iterrows()
    ]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    boundary = gpd.read_file(DATA_DIR / "warszawa_boundary.geojson")
    outline = boundary.geometry.iloc[0].simplify(0.0002)
    rings = [p.exterior for p in getattr(outline, "geoms", [outline])]
    data = {
        "roads": export_roads(),
        "trams": export_trams(),
        "boundary": [encode(r) for r in rings],
        "river": export_river(boundary.total_bounds),
        "districts": export_districts(),
        "stops": export_stops(),
        "scale": SCALE,
        "source": "OpenStreetMap (Geofabrik, mazowieckie)",
    }
    out = OUT_DIR / "warszawa.json"
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"== {out}: {out.stat().st_size / 1e6:.1f} MB, "
          f"{len(data['roads']['edges'])} linii ulic, {len(data['trams']['edges'])} linii torów")


if __name__ == "__main__":
    main()
