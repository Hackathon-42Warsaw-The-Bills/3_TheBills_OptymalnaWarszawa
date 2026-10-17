"""Dane dla symulatora (frontend/symulator.html) — wszystko, czego silnik oceny potrzebuje w przeglądarce.

Wynik: viz/symulator_dane.json
  hexes   — klastry H3: obrys, dzielnica, ludność 2025–2036 (3 scenariusze), dP_norm, kolej
  graph   — graf ulic dopuszczonych dla autobusu (skierowany): węzły, krawędzie z geometrią,
            długością, klasą i prędkością w szczycie / poza szczytem (GTFS ZTM, a bez danych — wg klasy)
  people  — punkty ludności 100 m (GHSL), każdy z udziałem w ludności swojego klastra
            -> ludność punktu w roku t = udział × ludność zabudowy istniejącej klastra w roku t
  newbuild— nowe budynki (pozwolenia GUNB): punkt + liczba nowych mieszkańców w każdym roku
  stops   — przystanki ZTM z liniami (dane/linie_dostepne_na_przystanku.csv) i stacje kolei/metra
  rail    — linie metra/SKM/KM/WKD do wyświetlenia
  routes  — trasy linii autobusowych i tramwajowych ZTM (dane/warszawa_linie_trasy.geojson) z liczbą kursów
  boundary— granica miasta (do maski „poza Warszawą”)
  infra   — cele podróży z dane/warszawa_infrastruktura.geojson (szpitale, szkoły, handel, kultura, parki…)
  Wszystko przycięte do granic Warszawy.
  params  — parametry modelu (osoby/mieszkanie, dryf, lata)

Uruchomienie (po build_graph.py, gtfs_predkosci.py, prognoza_ludnosci.py, kolej.py):
  .venv/bin/python mapping/export_symulator.py
"""

import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import h3
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import historia_ludnosci as hl  # noqa: E402
import prognoza_ludnosci as pl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DANE = ROOT / "dane"
OUT = ROOT / "frontend" / "symulator_dane.json"
CRS = "EPSG:2180"
R5 = lambda x: round(float(x), 5)  # noqa: E731

# Autobus 12–18 m: filtr techniczny ulic (jak w model/routes.py z gałęzi mapowanie_ulic).
BUS_WEIGHT_T, BUS_HEIGHT_M, MIN_WIDTH_M = 18, 3.6, 5.5
ALWAYS_OK = {"yes", "designated"}
BUS_HIGHWAYS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "busway",
                "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
NO_STOPS = {"motorway", "trunk", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
CLASSES = ["express", "primary", "secondary", "tertiary", "local", "service"]
# Prędkość autobusu w szczycie / poza szczytem wg klasy, gdy brak danych z rozkładu (km/h).
DEFAULT_SPEED = {"express": (35, 50), "primary": (17, 24), "secondary": (17, 23), "tertiary": (18, 23),
                 "local": (18, 22), "service": (12, 15)}


def num(v):
    m = re.search(r"\d+(?:[.,]\d+)?", str(v)) if v is not None else None
    return float(m.group().replace(",", ".")) if m else None


def bus_ok(r) -> bool:
    kinds = set(str(r["highway"]).split(";"))
    if str(r.get("bus")) in ALWAYS_OK or str(r.get("psv")).split(";")[0] in ALWAYS_OK or "busway" in kinds:
        return True
    if not kinds & BUS_HIGHWAYS:
        return False
    w, mw, mh = num(r.get("width")), num(r.get("maxweight")), num(r.get("maxheight"))
    if (w is not None and w < MIN_WIDTH_M) or (mw is not None and mw < BUS_WEIGHT_T):
        return False
    if str(r.get("maxheight")) == "below_default":
        return False
    return not (mh is not None and str(r.get("maxheight")) not in ("default", "none") and mh < BUS_HEIGHT_M)


def road_class(highway: str) -> str:
    kinds = {h.removesuffix("_link") for h in str(highway).split(";")}
    for c, members in [("express", {"motorway", "trunk"}), ("primary", {"primary"}), ("secondary", {"secondary"}),
                       ("tertiary", {"tertiary"}), ("local", {"unclassified", "residential", "living_street", "busway"})]:
        if kinds & members:
            return c
    return "service"


def export_graph() -> dict:
    e = gpd.read_parquet(DATA_DIR / "roads_edges.parquet")
    e = e[e.apply(bus_ok, axis=1)].copy()
    # Tylko największa silnie spójna część sieci — autobus musi móc dojechać i wrócić.
    g = nx.DiGraph(list(zip(e["u"], e["v"])))
    core = max(nx.strongly_connected_components(g), key=len)
    e = e[e["u"].isin(core) & e["v"].isin(core)].copy()
    speeds_path = DATA_DIR / "predkosci_ulic.parquet"
    sp = pd.read_parquet(speeds_path).set_index("edge_id") if speeds_path.exists() else pd.DataFrame()
    e["cls"] = e["highway"].map(road_class)
    # Brak tagu w OSM to pd.NA (nie tekst), więc sprawdzamy isna() wprost.
    flat = (e["bridge"].isna() | (e["bridge"] == "no")) & (e["tunnel"].isna() | e["tunnel"].isin(["no", "building_passage"]))
    no_stop_road = e["highway"].astype(str).str.split(";").map(lambda ks: bool(set(ks) & NO_STOPS)).astype(bool)
    e["stop_ok"] = (~no_stop_road & flat & (e["length"] >= 30)).astype(bool)
    nodes = pd.unique(pd.concat([e["u"], e["v"]]))
    nidx = {n: i for i, n in enumerate(nodes)}
    ncoord = gpd.read_parquet(DATA_DIR / "roads_nodes.parquet").set_index("osmid").loc[nodes]
    names = {"": 0}
    edges = []
    for r in e.itertuples():
        peak, free = DEFAULT_SPEED[r.cls]
        measured = 0
        if r.edge_id in sp.index:
            peak, free, measured = sp.at[r.edge_id, "v_szczyt_kmh"], sp.at[r.edge_id, "v_swobodna_kmh"], 1
        name = r.name.split(";")[0] if isinstance(r.name, str) else ""
        coords = [[R5(x), R5(y)] for x, y in r.geometry.simplify(0.00002).coords]
        edges.append([nidx[r.u], nidx[r.v], round(r.length, 1), CLASSES.index(r.cls), names.setdefault(name, len(names)),
                      round(float(peak), 1), round(float(free), 1), measured, int(r.stop_ok), r.edge_id, coords])
    print(f"   graf autobusowy: {len(nodes)} węzłów, {len(edges)} krawędzi, z prędkością z rozkładu: "
          f"{sum(x[7] for x in edges)}")
    return {"nodes": [[R5(x), R5(y)] for x, y in zip(ncoord["x"], ncoord["y"])], "names": list(names),
            "classes": CLASSES,
            "fields": ["u", "v", "len_m", "cls", "name", "v_peak", "v_free", "measured", "stop_ok", "edge_id", "coords"],
            "edges": edges}


def export_people(hexes: gpd.GeoDataFrame, inputs: dict) -> tuple[list, list, dict]:
    """Punkty ludności (GHSL 100 m) i nowe budynki z liczbą mieszkańców rok po roku."""
    px = hl.ghsl_pixels(hexes.union_all())
    px["h3"] = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(px["lat"], px["lon"])]
    px = px[px["h3"].isin(set(hexes["h3"])) & (px["ghsl_2020"] > 0)]
    share = px["ghsl_2020"] / px.groupby("h3")["ghsl_2020"].transform("sum")
    pop25 = hexes.set_index("h3")["population"]
    px["p2025"] = share * px["h3"].map(pop25)
    px = px[px["p2025"] >= 1]
    hid = {h: i for i, h in enumerate(hexes["h3"])}
    people = [[R5(r.lon), R5(r.lat), hid[r.h3], round(r.p2025, 1)] for r in px.itertuples()]

    # Nowi mieszkańcy pozwoleń rok po roku: ten sam rachunek co w prognoza_ludnosci.forecast (scenariusz bazowy).
    cal = inputs["calibration"]
    ppl = cal["realizacja"] * cal["os_na_mieszkanie"]
    permits = gpd.read_parquet(DATA_DIR / "gunb_permits.parquet")
    p = inputs["permits"].merge(permits[["permit_id", "geometry"]], on="permit_id")
    last = int(p["rok"].max())
    rec = p[p["rok"].isin(pl.TREND_YEARS)].copy()
    rec_rate = rec.groupby("h3")["n_flats_adj"].transform("sum") / len(pl.TREND_YEARS)
    rec["share_in_hex"] = rec["n_flats_adj"] / rec.groupby("h3")["n_flats_adj"].transform("sum")
    newbuild = []
    for r in p.itertuples():
        series = []
        for t in pl.YEARS:
            occ = pl.occupied_share(np.array([r.rok]), t, pl.BASE_YEAR)[0] * r.n_flats_adj
            series.append(occ)
        newbuild.append({"xy": [R5(r.geometry.x), R5(r.geometry.y)], "rok": int(r.rok), "adres": r.adres,
                         "mieszkania": int(round(r.n_flats_adj)), "series": series, "h": hid.get(r.h3, -1)})
    # Kontynuacja trendu („wirtualne pozwolenia”) — w miejscu ostatnich pozwoleń klastra.
    by_id = {nb["adres"] + str(i): nb for i, nb in enumerate(newbuild)}
    for (idx, r), rate in zip(rec.iterrows(), rec_rate):
        nb = newbuild[p.index.get_loc(idx)]
        for k, t in enumerate(pl.YEARS):
            ys = np.arange(last + 1, t + 1)
            if len(ys):
                w = (pl.TREND_DECAY ** (ys - last) * pl.occupied_share(ys, t, pl.BASE_YEAR)).sum()
                nb["series"][k] += rate * w * r["share_in_hex"]
    for nb in newbuild:
        nb["series"] = [round(v * ppl) for v in nb["series"]]
    newbuild = [nb for nb in newbuild if nb["series"][-1] > 0]
    del by_id

    drift = hexes["dzielnica"].map(cal["dryf_dzielnic"]).fillna(cal["dryf_miasta"]).values
    params = {"years": pl.YEARS, "drift": [round(float(d), 5) for d in drift],
              "os_na_mieszkanie": round(cal["os_na_mieszkanie"], 3), "realizacja": round(cal["realizacja"], 3)}
    print(f"   punkty ludności: {len(people)} (suma {sum(x[3] for x in people):,.0f}), nowe budynki: {len(newbuild)}")
    return people, newbuild, params


def export_hexes(hexes: gpd.GeoDataFrame) -> list:
    prog = pd.read_csv(DANE / "prognoza_ludnosci.csv").set_index("h3")
    kolej = pd.read_csv(DANE / "kolej" / "klastry_kolej.csv").set_index("h3")
    out = []
    for r in hexes.itertuples():
        g = r.geometry
        rings = [p.exterior for p in getattr(g, "geoms", [g])]
        row = prog.loc[r.h3]
        k = kolej.loc[r.h3]
        out.append({
            "h3": r.h3, "d": r.dzielnica,
            "rings": [[[R5(x), R5(y)] for x, y in ring.coords] for ring in rings],
            "pop": [int(row["pop_2025"])] + [int(row[f"pop_{t}"]) for t in pl.YEARS[1:]],
            "pop_lo": [int(row["pop_2025"])] + [int(row[f"pop_{t}_niski"]) for t in pl.YEARS[1:]],
            "pop_hi": [int(row["pop_2025"])] + [int(row[f"pop_{t}_wysoki"]) for t in pl.YEARS[1:]],
            "dPn": float(row["dP_2036_norm"]),
            "metro": bool(k["ma_metro"]), "kolej": bool(k["ma_kolej"]),
            "dist_rail": int(min(k["dist_metro_m"], k["dist_kolej_m"])),
        })
    return out


def warsaw_boundary():
    """Granica miasta = suma dzielnic (EPSG:4326)."""
    return gpd.read_file(hl.EXTERNAL_DIR / "warszawa_dzielnice.geojson").to_crs("EPSG:4326").union_all()


def parts_of(geom, tol: float) -> list:
    """Linie po przycięciu i uproszczeniu -> lista list [lon, lat] (pomija okruchy)."""
    g = geom.simplify(tol)
    geoms = getattr(g, "geoms", [g])
    return [[[R5(x), R5(y)] for x, y in p.coords] for p in geoms
            if p.geom_type == "LineString" and len(p.coords) > 1 and p.length > 0.0005]


def export_stops(boundary) -> dict:
    # Przystanki ZTM z liniami (dane/linie_dostepne_na_przystanku.csv), tylko w granicach miasta.
    st = pd.read_csv(DANE / "linie_dostepne_na_przystanku.csv", dtype={"stop_id": str, "linia": str})
    routes = gpd.read_file(DANE / "warszawa_linie_trasy.geojson", columns=["route_short_name", "typ"]) \
        if (DANE / "warszawa_linie_trasy.geojson").exists() else pd.DataFrame(columns=["route_short_name", "typ"])
    tram_lines = set(routes.loc[routes["typ"] == "tram", "route_short_name"].astype(str))
    g = st.groupby("stop_id").agg(lon=("stop_lon", "first"), lat=("stop_lat", "first"), name=("stop_name", "first"),
                                  nr=("busstopNr", "first"), linie=("linia", lambda x: sorted(set(x), key=lambda v: (len(v), v))))
    inside = gpd.GeoSeries(gpd.points_from_xy(g["lon"], g["lat"]), crs="EPSG:4326").within(boundary).values
    g = g[inside]
    stations = pd.read_csv(DANE / "kolej" / "stacje.csv", encoding="utf-8-sig")
    stations = stations[stations["w_warszawie"]]
    hubs_path = DATA_DIR / "hubs.parquet"
    planned = []
    if hubs_path.exists():
        hubs = gpd.read_parquet(hubs_path)
        hubs = hubs[(hubs["status"] != "istniejący") & (hubs["open_year"] <= 2036) & hubs.within(boundary)]
        planned = [[R5(p.x), R5(p.y), n, k, int(y)] for p, n, k, y in
                   zip(hubs.geometry, hubs["name"], hubs["kind"], hubs["open_year"])]
    print(f"   przystanki ZTM w Warszawie: {len(g)} (z liniami), stacje: {len(stations)}, planowane: {len(planned)}")
    return {
        # [lon, lat, nazwa, czy tramwajowy, "linie,…", numer słupka]
        "ztm": [[R5(r.lon), R5(r.lat), r.name, int(bool(set(r.linie) & tram_lines)), ",".join(r.linie), str(r.nr).zfill(2)]
                for r in g.itertuples()],
        "rail": [[R5(r.lon), R5(r.lat), r.nazwa, r.system, r.linie] for r in stations.itertuples()],
        "planned": planned,
    }


def export_rail_lines(boundary) -> list:
    lines = gpd.read_file(DANE / "kolej" / "linie.geojson")
    out = []
    for r in lines.itertuples():
        parts = parts_of(r.geometry.intersection(boundary), 0.0001)
        if parts:
            out.append({"system": r.system, "ref": r.ref, "parts": parts})
    return out


def export_routes(boundary) -> list:
    """Trasy linii autobusowych i tramwajowych ZTM, przycięte do granic Warszawy.
    Oba kierunki jednej linii łączymy: geometria dłuższego, kursy = średnia na kierunek."""
    path = DANE / "warszawa_linie_trasy.geojson"
    if not path.exists():
        return []
    r = gpd.read_file(path)
    r = r[r["typ"].isin(["autobus", "tram"])]
    out = []
    for ref, grp in r.groupby("route_short_name", sort=False):
        best = grp.loc[grp["dlugosc_km"].idxmax()]
        clipped = best.geometry.intersection(boundary)
        parts = parts_of(clipped, 0.00008)
        if not parts:
            continue
        out.append({"ref": str(ref), "typ": best["typ"], "nazwa": best["route_long_name"],
                    "kursy": int(round(grp["kursy_kierunek"].mean())), "parts": parts})
    print(f"   linie ZTM w Warszawie: {sum(x['typ'] == 'autobus' for x in out)} autobusowych, "
          f"{sum(x['typ'] == 'tram' for x in out)} tramwajowych")
    return out


# Kategorie z dane/warszawa_infrastruktura.* -> grupy pokazywane w aplikacji.
INFRA_GROUPS = {"szpital": "szpital", "szkola": "szkola", "centrum_handlowe": "handel", "park": "park",
                "kino": "kultura", "teatr": "kultura", "muzeum": "kultura", "urzad_dzielnicy": "urzad",
                "targowisko": "targowisko", "miejsce_kultu": "kult"}


def export_infra(boundary) -> dict:
    """Cele podróży: punkty (środek obiektu) wszystkich kategorii + obrysy parków."""
    path = DANE / "warszawa_infrastruktura.geojson"
    if not path.exists():
        return {"points": [], "parks": []}
    g = gpd.read_file(path).to_crs("EPSG:4326")
    g = g[g.geometry.representative_point().within(boundary)]
    g["grupa"] = g["kategoria"].map(INFRA_GROUPS).fillna("inne")
    pts = g.geometry.representative_point()
    points = [[R5(pt.x), R5(pt.y), r.grupa, r.nazwa if isinstance(r.nazwa, str) else "", r.kategoria,
               round(float(r.pow_ha), 1) if r.pow_ha == r.pow_ha else 0]
              for r, pt in zip(g.itertuples(), pts)]
    parks = []
    for r in g[(g["kategoria"] == "park") & g.geom_type.isin(["Polygon", "MultiPolygon"])].itertuples():
        geom = r.geometry.simplify(0.00005)
        polys = [[[[R5(x), R5(y)] for x, y in poly.exterior.coords]] for poly in getattr(geom, "geoms", [geom])]
        parks.append({"nazwa": r.nazwa if isinstance(r.nazwa, str) else "park", "ha": round(float(r.pow_ha), 1), "polys": polys})
    print(f"   infrastruktura: {len(points)} obiektów ({g['grupa'].value_counts().to_dict()}), parki-obszary: {len(parks)}")
    return {"points": points, "parks": parks}


def export_boundary(boundary) -> list:
    g = boundary.simplify(0.0002)
    return [[[R5(x), R5(y)] for x, y in p.exterior.coords] for p in getattr(g, "geoms", [g])]


def main() -> None:
    hexes = gpd.read_file(DANE / "kontury_400m" / "warszawa_h3_400m_pelna.geojson")
    inputs = pl.load_inputs()
    print("== graf ulic")
    graph = export_graph()
    print("== ludność")
    people, newbuild, params = export_people(hexes, inputs)
    boundary = warsaw_boundary()
    print("== linie, przystanki, granica")
    data = {"params": params, "hexes": export_hexes(hexes), "graph": graph, "people": people,
            "newbuild": newbuild, "stops": export_stops(boundary), "rail": export_rail_lines(boundary),
            "routes": export_routes(boundary), "boundary": export_boundary(boundary), "infra": export_infra(boundary)}
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"== {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
