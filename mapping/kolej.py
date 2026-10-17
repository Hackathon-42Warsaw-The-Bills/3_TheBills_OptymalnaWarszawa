"""Linie i stacje kolei w Warszawie i okolicy: metro, SKM, Koleje Mazowieckie, WKD.

Źródło: relacje tras OSM (type=route, route=subway/train) z lokalnego snapshotu Geofabrik
(data/osm/mazowieckie-latest.osm.pbf, ten sam co w build_graph.py). Bierzemy trasy
przewoźników z SYSTEMS; oba kierunki i warianty tej samej linii scalamy w jedną geometrię.

Wynik (dane/kolej/):
  - linie.geojson   — jedna linia na (system, ref): geometria torów, przystanki,
                      dlugosc_km (najdłuższy wariant, jeden kierunek)
  - stacje.geojson  — jedna stacja na nazwę i system: lista linii, w_warszawie
  - stacje.csv      — to samo bez geometrii (lon/lat)
  - wezly.geojson / wezly.csv — jedna stacja na nazwę (wszystkie systemy razem), np. węzeł
                      Warszawa Zachodnia = KM + SKM; do liczenia przesiadek i zasięgu stacji

Uruchomienie:  .venv/bin/python mapping/kolej.py
"""

import os
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import osmium
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge

ROOT = Path(__file__).resolve().parent.parent
PBF_PATH = ROOT / "data" / "osm" / "mazowieckie-latest.osm.pbf"
OUT_DIR = ROOT / "dane" / "kolej"
EXTERNAL_DIR = Path(os.environ.get("DANE_WARSZAWA", Path.home() / "Downloads" / "dane_warszawa"))

# (network, fragment operatora) -> system
SYSTEMS = [
    ("ZTM Warszawa", "Metro", "metro"),
    ("ZTM Warszawa", "Szybka Kolej Miejska", "SKM"),
    ("Koleje Mazowieckie", "", "KM"),
    ("Warszawska Kolej Dojazdowa", "", "WKD"),
]
# Różne pisownie tej samej stacji w OSM.
NAME_ALIASES = {"Ratusz-Arsenał": "Ratusz Arsenał"}
STOP_ROLES = {"stop", "stop_entry_only", "stop_exit_only"}
PLATFORM_ROLES = {"platform", "platform_entry_only", "platform_exit_only"}


def system_of(tags) -> str | None:
    if tags.get("type") != "route" or tags.get("route") not in ("train", "subway", "light_rail"):
        return None
    for network, operator, name in SYSTEMS:
        if tags.get("network") == network and operator in (tags.get("operator") or ""):
            return name
    return None


def read_routes():
    routes = []
    for r in osmium.FileProcessor(str(PBF_PATH), osmium.osm.RELATION).with_filter(osmium.filter.KeyFilter("route")):
        t = dict(r.tags)
        system = system_of(t)
        if not system:
            continue
        ways = [m.ref for m in r.members if m.type == "w" and m.role not in STOP_ROLES | PLATFORM_ROLES]
        # Nazwy stacji z punktów zatrzymania; perony tylko gdy trasa nie ma punktów zatrzymania.
        stops = [(m.type, m.ref) for m in r.members if m.role in STOP_ROLES and m.type == "n"]
        if not stops:
            stops = [(m.type, m.ref) for m in r.members if m.role in PLATFORM_ROLES and m.type in "nw"]
        routes.append({"id": r.id, "system": system, "ref": t.get("ref") or t.get("name", "?"),
                       "name": t.get("name", ""), "from": t.get("from", ""), "to": t.get("to", ""),
                       "ways": ways, "stops": stops})
    return routes


def read_geometry(way_ids: set, node_ids: set):
    """Geometrie torów (way -> LineString) i przystanków (node/way -> punkt, nazwa)."""
    lines, stops = {}, {}
    fp = osmium.FileProcessor(str(PBF_PATH), osmium.osm.NODE | osmium.osm.WAY).with_locations()
    for o in fp:
        if o.is_node() and ("n", o.id) in node_ids:
            stops[("n", o.id)] = (Point(o.location.lon, o.location.lat), o.tags.get("name", ""))
        elif o.is_way() and (o.id in way_ids or ("w", o.id) in node_ids):
            pts = [(n.location.lon, n.location.lat) for n in o.nodes if n.location.valid()]
            if len(pts) < 2:
                continue
            if o.id in way_ids:
                lines[o.id] = LineString(pts)
            if ("w", o.id) in node_ids:  # peron jako linia/obszar -> środek
                stops[("w", o.id)] = (LineString(pts).centroid, o.tags.get("name", ""))
    return lines, stops


def clean_name(name: str) -> str:
    # Nazwy peronów bywają w stylu "Warszawa Wschodnia, peron 3" — zostawiamy nazwę stacji.
    name = name.split(",")[0].split(" peron")[0].strip()
    name = NAME_ALIASES.get(name, name)
    return "" if name.lower().startswith("peron") else name


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== relacje tras")
    routes = read_routes()
    way_ids = {w for r in routes for w in r["ways"]}
    stop_ids = {s for r in routes for s in r["stops"]}
    print(f"   {len(routes)} tras, {len(way_ids)} odcinków torów, {len(stop_ids)} przystanków")
    print("== geometria")
    lines, stops = read_geometry(way_ids, stop_ids)

    warsaw = gpd.read_file(EXTERNAL_DIR / "warszawa_dzielnice.geojson").union_all()

    by_line = defaultdict(lambda: {"ways": set(), "names": set(), "stations": []})
    station_lines = defaultdict(set)
    station_pts = defaultdict(list)
    for r in routes:
        key = (r["system"], r["ref"])
        by_line[key]["ways"] |= set(r["ways"])
        by_line[key].setdefault("route_ways", []).append(r["ways"])
        by_line[key]["names"].add(r["name"])
        for s in r["stops"]:
            if s not in stops or not stops[s][1]:
                continue
            pt, name = stops[s]
            name = clean_name(name)
            if not name:
                continue
            if name not in by_line[key]["stations"]:
                by_line[key]["stations"].append(name)
            station_lines[(r["system"], name)].add(r["ref"])
            station_pts[(r["system"], name)].append(pt)

    line_rows = []
    for (system, ref), v in sorted(by_line.items()):
        geoms = [lines[w] for w in v["ways"] if w in lines]
        if not geoms:
            continue
        merged = linemerge(geoms)
        merged = merged if merged.geom_type == "MultiLineString" else MultiLineString([merged])
        one_dir = max(gpd.GeoSeries([lines[w] for w in rw if w in lines], crs="EPSG:4326").to_crs("EPSG:2180").length.sum()
                      for rw in v["route_ways"])
        line_rows.append({"system": system, "ref": ref, "nazwa": " | ".join(sorted(v["names"])),
                          "dlugosc_km": round(one_dir / 1000, 1),
                          "n_stacji": len(v["stations"]), "stacje": "; ".join(v["stations"]),
                          "geometry": merged})
    linie = gpd.GeoDataFrame(line_rows, geometry="geometry", crs="EPSG:4326")
    m = linie.to_crs("EPSG:2180")
    # Tory w Warszawie (oba kierunki scalone geometrycznie -> przybliżenie długości torowiska).
    linie["km_torow_w_warszawie"] = (m.intersection(gpd.GeoSeries([warsaw], crs="EPSG:4326").to_crs("EPSG:2180").iloc[0]).length / 1000).round(1)

    st_rows = []
    for (system, name), refs in station_lines.items():
        pts = station_pts[(system, name)]
        c = Point(sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts))
        st_rows.append({"nazwa": name, "system": system, "linie": ", ".join(sorted(refs)),
                        "n_linii": len(refs), "w_warszawie": warsaw.contains(c), "geometry": c})
    stacje = gpd.GeoDataFrame(st_rows, geometry="geometry", crs="EPSG:4326").sort_values(["system", "nazwa"])

    linie.to_file(OUT_DIR / "linie.geojson", driver="GeoJSON")
    stacje.to_file(OUT_DIR / "stacje.geojson", driver="GeoJSON")
    csv = stacje.drop(columns="geometry").assign(lon=stacje.geometry.x.round(6), lat=stacje.geometry.y.round(6))
    csv.to_csv(OUT_DIR / "stacje.csv", index=False, encoding="utf-8-sig")

    # Węzły: stacje o tej samej nazwie w różnych systemach (KM + SKM + WKD) to jeden punkt.
    wezly = stacje.copy()
    wezly["lon"], wezly["lat"] = wezly.geometry.x, wezly.geometry.y
    wezly["linie_sys"] = wezly["system"] + ": " + wezly["linie"]
    agg = wezly.groupby("nazwa").agg(systemy=("system", lambda x: ", ".join(sorted(set(x)))),
                                     linie=("linie_sys", " | ".join), lon=("lon", "mean"), lat=("lat", "mean"),
                                     w_warszawie=("w_warszawie", "max")).reset_index()
    agg["n_linii"] = agg["linie"].str.count(",") + agg["linie"].str.count(r"\|") + 1
    wezly = gpd.GeoDataFrame(agg, geometry=gpd.points_from_xy(agg["lon"], agg["lat"]), crs="EPSG:4326")
    wezly.drop(columns=["lon", "lat"]).to_file(OUT_DIR / "wezly.geojson", driver="GeoJSON")
    agg.round({"lon": 6, "lat": 6}).to_csv(OUT_DIR / "wezly.csv", index=False, encoding="utf-8-sig")

    print(f"== {len(linie)} linii -> dane/kolej/linie.geojson")
    print(linie[["system", "ref", "n_stacji", "dlugosc_km", "km_torow_w_warszawie"]].to_string(index=False))
    s = stacje[stacje["w_warszawie"]]
    print(f"== {len(stacje)} stacji (w Warszawie: {len(s)}) -> dane/kolej/stacje.geojson")
    print(s.groupby("system").size().to_string())
    print(f"== {len(wezly)} węzłów (w Warszawie: {wezly['w_warszawie'].sum()}) -> dane/kolej/wezly.geojson")
    print(wezly[wezly["w_warszawie"]].nlargest(8, "n_linii")[["nazwa", "systemy", "n_linii"]].to_string(index=False))


if __name__ == "__main__":
    main()
