"""Węzły przesiadkowe: dzisiejsze i planowane stacje metra, kolei i tramwaju.

Geometria pochodzi z OSM (ten sam snapshot co build_graph.py): istniejące stacje
(railway=station/halt), budowane (railway=construction) i planowane (railway=proposed).
Rok otwarcia to założenie scenariusza — tabela OPEN_YEAR niżej, łatwa do edycji.
Planowane torowiska zamieniamy na „wirtualne przystanki” co TRAM_STOP_STEP m.

Wynik: data/hubs.parquet — name, kind (metro|kolej|tramwaj), mode (rail|tram),
status (istniejący|budowa|plan), open_year, project, geometry (EPSG:4326).

Uruchomienie (po build_graph.py):  .venv/bin/python mapping/rail_plans.py
"""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
import pandas as pd
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PBF_PATH = DATA_DIR / "osm" / "mazowieckie-latest.osm.pbf"
EXTERNAL_DIR = Path(os.environ.get("DANE_WARSZAWA", Path.home() / "Downloads" / "dane_warszawa"))
CRS = "EPSG:2180"
TRAM_STOP_STEP = 450
CURRENT_YEAR = 2026

# Założenia scenariusza bazowego (stan wiedzy 2026-10). Źródła: komunikaty Metra Warszawskiego,
# Tramwajów Warszawskich i prasa — terminy często się przesuwają, stąd jawna tabela.
METRO_PROJECTS = {
    "M2 zachód (Karolin)": (2027, {"Lazurowa", "Chrzanów", "Karolin"}),
    "M3 Stadion–Gocław": (2033, {"Dworzec Wschodni", "Mińska", "Wiatraczna", "Ostrobramska",
                                 "Nowaka-Jeziorańskiego", "Gocław"}),
}
FAR_FUTURE = 2040  # M4 i inne linie bez finansowania — poza horyzontem 2036
RAIL_HALTS = {"Warszawa Ulrychów": 2027, "Warszawa Jelonki": 2030}
# Planowane torowiska bez nazwy przypisujemy do projektu po położeniu:
# (projekt, rok, (min_lon, min_lat, max_lon, max_lat))
TRAM_PROJECTS = [
    ("Tramwaj do Dworca Zachodniego", 2027, (20.955, 52.205, 20.980, 52.222)),
    ("Tramwaj przez Pole Mokotowskie", 2029, (20.980, 52.200, 21.005, 52.215)),
    ("Tramwaj na Gocław", 2030, (21.055, 52.215, 21.110, 52.245)),
    ("Tramwaj na Zieloną Białołękę", 2030, (20.985, 52.280, 21.080, 52.330)),
    ("Tramwaj Wilanów–Ursynów", 2031, (21.040, 52.140, 21.100, 52.185)),
    ("Tramwaj Dw. Zachodni–Wola", 2032, (20.940, 52.215, 20.975, 52.235)),
]
TRAM_DEFAULT = ("Inne planowane torowiska", 2034)


def status_of(tags) -> str | None:
    r = tags.get("railway")
    if r in ("station", "halt") and tags.get("station") != "tram":
        return "istniejący"
    if r == "construction" or "construction:railway" in tags:
        return "budowa"
    if r == "proposed" or "proposed:railway" in tags:
        return "plan"
    return None


def kind_of(tags) -> str:
    sub = "subway" in (tags.get("station"), tags.get("construction"), tags.get("proposed"),
                       tags.get("construction:railway"), tags.get("subway"))
    return "metro" if sub or tags.get("subway") == "yes" else "kolej"


def read_osm(bounds):
    min_lon, min_lat, max_lon, max_lat = bounds
    stations, tram_ways = [], []
    keys = ("railway", "construction:railway", "proposed:railway")
    fp = (osmium.FileProcessor(str(PBF_PATH), osmium.osm.NODE | osmium.osm.WAY)
          .with_locations().with_filter(osmium.filter.KeyFilter(*keys)))
    for o in fp:
        t = dict(o.tags)
        st = status_of(t)
        if o.is_node():
            loc = o.location
            if not (min_lon <= loc.lon <= max_lon and min_lat <= loc.lat <= max_lat):
                continue
            val = {t.get("railway"), t.get("construction:railway"), t.get("proposed:railway"), t.get("proposed")}
            if st and val & {"station", "halt"}:
                stations.append({"osm_id": o.id, "name": t.get("name", ""), "kind": kind_of(t), "status": st,
                                 "geometry": Point(loc.lon, loc.lat)})
        elif st in ("budowa", "plan") and "tram" in (t.get("construction:railway"), t.get("proposed:railway"),
                                                    t.get("construction"), t.get("proposed")):
            pts = [(n.location.lon, n.location.lat) for n in o.nodes if n.location.valid()]
            if len(pts) > 1:
                tram_ways.append({"osm_id": o.id, "name": t.get("name", ""), "status": st, "geometry": LineString(pts)})
    return gpd.GeoDataFrame(stations, crs="EPSG:4326"), gpd.GeoDataFrame(tram_ways, crs="EPSG:4326")


def station_year(row) -> tuple[int, str]:
    if row["status"] == "istniejący":
        return 0, ""
    for project, (year, names) in METRO_PROJECTS.items():
        if row["name"] in names and (row["kind"] == "metro" or row["status"] == "budowa"):
            return year, project
    if row["name"] in RAIL_HALTS:
        return RAIL_HALTS[row["name"]], "nowy przystanek kolejowy"
    return FAR_FUTURE, "M4/M5 i inne (bez finansowania)" if row["kind"] == "metro" else "kolej (plan)"


def tram_project(geom) -> tuple[str, int]:
    c = geom.centroid
    for name, year, (x0, y0, x1, y1) in TRAM_PROJECTS:
        if x0 <= c.x <= x1 and y0 <= c.y <= y1:
            return name, year
    return TRAM_DEFAULT


def tram_virtual_stops(ways: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ways = ways.copy()
    ways[["project", "open_year"]] = [tram_project(g) for g in ways.geometry]
    ways = ways.to_crs(CRS)
    rows = []
    for (project, year), grp in ways.groupby(["project", "open_year"]):
        # Jeden tor na kierunek -> rzadsze próbkowanie i deduplikacja po siatce 100 m.
        seen = set()
        for g in grp.geometry:
            for d in np.arange(0, g.length + 1, TRAM_STOP_STEP / 2):
                p = g.interpolate(d)
                cell = (round(p.x / TRAM_STOP_STEP), round(p.y / TRAM_STOP_STEP))
                if cell not in seen:
                    seen.add(cell)
                    rows.append({"name": f"{project} (przystanek)", "kind": "tramwaj", "mode": "tram",
                                 "status": "plan", "open_year": int(year), "project": project, "geometry": p})
    return gpd.GeoDataFrame(rows, crs=CRS)


def existing_tram_stops() -> gpd.GeoDataFrame:
    s = gpd.read_file(EXTERNAL_DIR / "warszawa_przystanki_ztm.geojson").to_crs(CRS)
    s = s[s["linie_tram"].fillna("") != ""]
    g = s.dissolve("zespol", aggfunc={"nazwa_zespolu": "first"})
    return gpd.GeoDataFrame({"name": g["nazwa_zespolu"].values, "kind": "tramwaj", "mode": "tram",
                             "status": "istniejący", "open_year": 0, "project": ""},
                            geometry=g.geometry.centroid.values, crs=CRS)


def main() -> None:
    boundary = gpd.read_file(DATA_DIR / "warszawa_boundary.geojson")
    stations, tram_ways = read_osm(boundary.total_bounds)
    stations = stations[stations.within(boundary.union_all().buffer(0.01))]
    stations[["open_year", "project"]] = [station_year(r) for _, r in stations.iterrows()]
    stations.loc[stations["project"].str.startswith("M"), "kind"] = "metro"
    stations["mode"] = "rail"
    # Stacje w OSM są często zdublowane (węzeł stacji + perony) — jedna na nazwę i rodzaj.
    stations = stations.to_crs(CRS).drop_duplicates(["name", "kind", "status"])
    stations = stations[stations["name"] != ""]

    hubs = pd.concat([
        stations[["name", "kind", "mode", "status", "open_year", "project", "geometry"]],
        existing_tram_stops(),
        tram_virtual_stops(tram_ways),
    ], ignore_index=True)
    hubs = gpd.GeoDataFrame(hubs, geometry="geometry", crs=CRS)
    hubs["open_year"] = hubs["open_year"].where(hubs["open_year"] > 0, CURRENT_YEAR - 1).astype(int)
    hubs.to_crs("EPSG:4326").to_parquet(DATA_DIR / "hubs.parquet")

    print(f"== zapisano {len(hubs)} węzłów -> data/hubs.parquet")
    print(hubs.groupby(["kind", "status"]).size().to_string())
    plan = hubs[hubs["status"] != "istniejący"]
    print(plan.groupby(["project", "open_year"]).size().rename("węzłów").to_string())


if __name__ == "__main__":
    main()
