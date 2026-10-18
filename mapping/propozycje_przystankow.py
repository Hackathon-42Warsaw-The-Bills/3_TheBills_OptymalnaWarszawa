"""Propozycje nowych przystanków ZTM dla całej Warszawy (848 klastrów H3 z dane/klastry/clastry_all.csv).

Model — wszystkie odległości to dojście piesze po sieci chodników i ulic z OSM (bariery: tory, Wisła, trasy
ekspresowe, tereny zamknięte), a nie okręgi:
  popyt      — mieszkańcy 2025 (pop_2025 klastra) rozłożeni na budynki mieszkalne z OSM wg pow. użytkowej;
               nowi mieszkańcy do 2036 w nowych budynkach z pozwoleń GUNB (te same co w symulatorze);
               miejsca pracy: biura/usługi z OSM, 1 etat / 30 m², waga 0,5; cele podróży (szpitale, przychodnie,
               szkoły, uczelnie, domy opieki, handel…) jako równoważnik osób, obiekt w tym samym miejscu liczony raz.
  dostępność — A = f(dojście) × g(kursy/h). f = 1 do progu ZTM (plan transportowy: 300 / 400 / 500 m dla strefy
               śródmiejskiej / miejskiej / przedmieść), potem liniowo do 0 przy progu + 400 m; metro i kolej: 1 do
               1000 m, 0 przy 1500 m. g = min(1, √(kursy/h ÷ 6)) — kursy w szczycie 7–9 z GTFS (dzień roboczy);
               kolej 0,8, metro 1. Rok 2036: dodatkowo planowane stacje metra/kolei i przystanki tramwajowe
               otwierane do 2033 r. (data/hubs.parquet).
  kandydaci  — punkty co 25 m na ulicach przejezdnych dla autobusu (graf symulatora, stop_ok), ≥ 30 m od skrzyżowań.
               Typ A: ulicą jadą linie zatrzymujące się (zwykłe, okresowe, strefowe, lokalne, tramwaje) i żadna nie
               ma przystanku bliżej niż 300 m — wystarczy dostawić słupki; częstotliwość = suma kursów tych linii.
               Typ B: żadna taka linia tędy nie jedzie — potrzebna zmiana trasy / przedłużenie linii (zakładamy
               3 kursy/h); ulica utwardzona, publiczna, przelotowa (nie ślepa odnoga), nie strefa zamieszkania ani
               droga serwisowa; ≥ 300 m od jakiegokolwiek słupka.
  wybór      — zachłanne maksymalne pokrycie: zysk = 0,5·Σ w·max(0, a − A_dziś) + 0,5·Σ w·max(0, a − A_2036);
               najpierw etap A, potem etap B (z uwzględnieniem A); propozycje ≥ 350 m od siebie.
  obsługa B  — dla przystanku B: objazd istniejącej linii (zjazd z trasy → odcinek z przystankiem → powrót, bez
               zawracania; ≤ 3 km na kurs) albo przedłużenie od pętli (≤ 5 km); rekomendowana linia całodzienna
               z ≥ 2 kursami/h i najmniejszą liczbą dodatkowych wozokilometrów na dobę.

Wynik (dane/przystanki/):
  propozycje_przystankow.csv             — przystanki: punkt (lon/lat, WKT, GeoJSON), liczby, linie, uzasadnienie,
                                           zasięg dojścia (GeoJSON wielokąta)
  propozycje_przystankow.geojson         — te same przystanki jako punkty (do mapy)
  propozycje_przystankow_zasiegi.geojson — obszar dojścia pieszego w progu ZTM wokół każdej propozycji
  propozycje_przystankow_klastry.csv     — dla każdego klastra: mieszkańcy poza standardem dziś / po propozycjach

Cache (data/cache/, tworzone przy pierwszym uruchomieniu): przystanki_siec_piesza.npz, przystanki_cele_osm.parquet,
przystanki_slupki.parquet, przystanki_linie_slupki.parquet.

Uruchomienie (po clastry_all.py, export_symulator.py, rail_plans.py i przystanki_top30.py — ten ostatni tworzy
data/cache/budynki_mieszkalne.parquet):  .venv/bin/python mapping/propozycje_przystankow.py
"""

import heapq
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import h3
import networkx as nx
import numpy as np
import osmium
import pandas as pd
import shapely
from pyproj import Transformer
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DANE, DATA = ROOT / "dane", ROOT / "data"
CACHE = DATA / "cache"
PBF = DATA / "osm" / "mazowieckie-latest.osm.pbf"
GTFS = DATA / "gtfs" / "warsaw"
WALK, POI_CACHE = CACHE / "przystanki_siec_piesza.npz", CACHE / "przystanki_cele_osm.parquet"
STOPS_CACHE, LINES_CACHE = CACHE / "przystanki_slupki.parquet", CACHE / "przystanki_linie_slupki.parquet"
OUT = DANE / "przystanki" / "propozycje_przystankow"
TO_PL = Transformer.from_crs(4326, 2180, always_xy=True)
TO_WGS = Transformer.from_crs(2180, 4326, always_xy=True)

# --- sieć piesza
BBOX = (20.81, 52.07, 21.31, 52.40)   # Warszawa + ok. 2,5 km zapasu
MAX_SEG = 50.0                        # zagęszczenie odcinków — lepsze dowiązanie budynków i przystanków
WALKABLE = {"footway", "path", "pedestrian", "steps", "living_street", "residential", "service", "unclassified",
            "tertiary", "tertiary_link", "secondary", "secondary_link", "primary", "primary_link", "track",
            "cycleway", "road", "crossing", "bridleway"}
CAR_ONLY = {"motorway", "motorway_link", "trunk", "trunk_link"}
FOOT_OK = {"yes", "designated", "permissive", "destination"}
POI_TAGS = {("amenity", "clinic"): ("przychodnia", 150), ("amenity", "doctors"): ("przychodnia", 60),
            ("healthcare", "clinic"): ("przychodnia", 150), ("healthcare", "centre"): ("przychodnia", 150),
            ("amenity", "university"): ("uczelnia", 400), ("amenity", "college"): ("uczelnia", 200),
            ("amenity", "kindergarten"): ("przedszkole", 60), ("amenity", "nursing_home"): ("dom_opieki", 60),
            ("social_facility", "nursing_home"): ("dom_opieki", 60), ("social_facility", "group_home"): ("dom_opieki", 40),
            ("amenity", "library"): ("biblioteka", 30), ("leisure", "sports_centre"): ("sport", 40),
            ("amenity", "post_office"): ("poczta", 30), ("shop", "supermarket"): ("supermarket", 60)}

# --- GTFS
PEAK = (7 * 3600, 9 * 3600)
MODE = {"3": "autobus", "0": "tramwaj", "1": "metro", "2": "kolej"}

# --- model
R_ZONE = {"śródmiejska": 300, "miejska": 400, "przedmieścia": 500}
DECAY = 400
R_RAPID, R_RAPID_MAX, G_RAIL = 1000, 1500, 0.8
FREQ_FULL, FREQ_NEW_LINE = 6.0, 3.0
JOB_W, M2_PER_JOB, JOB_CAP = 0.5, 30, 8000
RES_TYPES = {"dom": 1.0, "yes": 0.5}
YES_AREA = (40, 3000)
# równoważnik osób dla celu podróży (dziennie, istotny dla komunikacji)
DEST_EQ = {"szpital": 300, "centrum_handlowe": 250, "uczelnia": 150, "urzad_dzielnicy": 150, "szkola": 100,
           "przychodnia": 80, "targowisko": 80, "dom_opieki": 40, "supermarket": 30, "park": 30, "kino": 30,
           "teatr": 30, "muzeum": 30, "poczta": 20, "sport": 20, "przedszkole": 15, "miejsce_kultu": 15,
           "biblioteka": 15}
DEST_PL = {"szpital": "szpital", "centrum_handlowe": "centrum handlowe", "uczelnia": "uczelnia",
           "urzad_dzielnicy": "urząd dzielnicy", "szkola": "szkoła", "przychodnia": "przychodnia",
           "targowisko": "targowisko", "dom_opieki": "dom opieki", "supermarket": "supermarket", "park": "park",
           "kino": "kino", "teatr": "teatr", "muzeum": "muzeum", "poczta": "poczta", "sport": "obiekt sportowy",
           "przedszkole": "przedszkole/żłobek", "miejsce_kultu": "kościół", "biblioteka": "biblioteka"}
SAMPLE_M, FROM_JUNCTION_M = 25, 30
ON_ROUTE = {"autobus": 15, "tram": 25}
SAME_LINE_MIN, ANY_STOP_MIN_B = 300, 300
MIN_SPACING = 350
MIN_GAIN = {"A": 100, "B": 200}
MAX_STOPS = {"A": 25, "B": 60}
STOPPING = {"zwykła", "zwykła okresowa", "strefowa", "lokalna", "tramwaj"}
DIVERTABLE = {"zwykła", "zwykła okresowa", "strefowa", "lokalna"}
UNPAVED = {"unpaved", "compacted", "ground", "gravel", "dirt", "fine_gravel", "sand", "grass", "grass_paver", "mud",
           "earth", "pebblestone", "woodchips", "unhewn_cobblestone", "rock", "stepping_stones"}
BAD_ACCESS = {"private", "no", "customers", "permit", "delivery", "employees", "forestry", "agricultural", "unknown"}
B_HIGHWAYS = {"primary", "secondary", "tertiary", "unclassified", "residential", "busway"}
PLANNED_MAX_YEAR = 2033
LIMIT = max(R_ZONE.values()) + DECAY        # zasięg liczenia dla kandydata (m pieszo)
ROUTE_LIMIT = 6000                          # zasięg szukania linii do zmiany trasy (m po ulicach)
MAX_DETOUR_KM, MAX_EXT_KM = 3.0, 5.0        # objazd na kurs (zjazd + powrót) / przedłużenie od pętli (tam i z powrotem)
PRIORITY = {"A": (200, 150), "B": (500, 300)}   # zysk dla priorytetu 1 / 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fmt(x, n=0):
    return f"{x:,.{n}f}".replace(",", " ").replace(".", ",")


def f_bus(d, R):
    return np.clip((R + DECAY - d) / DECAY, 0, 1)


def f_rapid(d):
    return np.clip((R_RAPID_MAX - d) / (R_RAPID_MAX - R_RAPID), 0, 1)


def g_freq(k):
    return np.minimum(1.0, np.sqrt(np.asarray(k, float) / FREQ_FULL))


def tokens(v) -> set:
    return set() if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA else set(str(v).split(";"))


def srt(xs) -> str:
    return " ".join(sorted(set(xs), key=lambda x: (len(x), x)))


def h3_of(lon, lat):
    return np.array([h3.latlng_to_cell(b, a, 8) for a, b in zip(lon, lat)])


# ---------------------------------------------------------------- sieć piesza i cele z OSM (cache)
def walkable(t) -> bool:
    h, foot = t.get("highway"), t.get("foot")
    if h is None or foot == "no":
        return False
    if h in CAR_ONLY:   # chodnik zmapowany osobno (sidewalk=separate) — sama jezdnia nie jest dla pieszych
        sw = t.get("sidewalk") or t.get("sidewalk:both") or ""
        return foot in FOOT_OK or sw in ("both", "left", "right", "yes")
    if h not in WALKABLE:
        return False
    if t.get("access") in ("private", "no") and foot not in FOOT_OK:
        return False
    return not (t.get("area") == "yes" and h not in ("pedestrian", "footway"))


def poi_kind(t):
    return next((val for (k, v), val in POI_TAGS.items() if t.get(k) == v), None)


def build_walk_network() -> None:
    x0, y0, x1, y1 = BBOX
    node_id: dict[int, int] = {}
    lon, lat, eu, ev, pois = [], [], [], [], []
    fp = (osmium.FileProcessor(str(PBF)).with_locations()
          .with_filter(osmium.filter.KeyFilter("highway", "amenity", "healthcare", "social_facility", "leisure", "shop")))
    for o in fp:
        t = o.tags
        if o.is_node():
            pk = poi_kind(t)
            if pk and o.location.valid() and x0 <= o.location.lon <= x1 and y0 <= o.location.lat <= y1:
                pois.append((o.location.lon, o.location.lat, pk[0], pk[1], t.get("name", ""), f"n{o.id}"))
            continue
        if not o.is_way():
            continue
        pk = poi_kind(t)
        if pk and "highway" not in t:
            pts = [(n.location.lon, n.location.lat) for n in o.nodes if n.location.valid()]
            if pts:
                cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    pois.append((cx, cy, pk[0], pk[1], t.get("name", ""), f"w{o.id}"))
            continue
        if not walkable(t):
            continue
        nodes = [(n.ref, n.location.lon, n.location.lat) for n in o.nodes if n.location.valid()]
        if len(nodes) < 2 or not any(x0 <= a <= x1 and y0 <= b <= y1 for _, a, b in nodes):
            continue
        prev = None
        for ref, a, b in nodes:
            if ref not in node_id:
                node_id[ref] = len(lon)
                lon.append(a)
                lat.append(b)
            cur = node_id[ref]
            if prev is not None and prev != cur:
                eu.append(prev)
                ev.append(cur)
            prev = cur
    xy = np.column_stack(TO_PL.transform(np.array(lon), np.array(lat)))
    eu, ev = np.array(eu), np.array(ev)
    L = np.hypot(*(xy[eu] - xy[ev]).T)
    k = np.ceil(L / MAX_SEG).astype(int)
    long_ = k > 1
    new_xy, nu, nv, nl = [], [], [], []
    nxt = len(xy)
    for u, v, kk, ll in zip(eu[long_], ev[long_], k[long_], L[long_]):
        new_xy.append(xy[u] + (xy[v] - xy[u]) * (np.arange(1, kk)[:, None] / kk))
        chain = np.concatenate([[u], np.arange(nxt, nxt + kk - 1), [v]])
        nxt += kk - 1
        nu.append(chain[:-1])
        nv.append(chain[1:])
        nl.append(np.full(kk, ll / kk))
    xy = np.vstack([xy] + new_xy)
    edges = np.column_stack([np.concatenate([eu[~long_]] + nu), np.concatenate([ev[~long_]] + nv),
                             np.concatenate([L[~long_]] + nl)])
    np.savez(WALK, xy=xy, edges=edges)
    p = pd.DataFrame(pois, columns=["lon", "lat", "rodzaj", "rownowaznik_osob", "nazwa", "osm"]).drop_duplicates("osm")
    gpd.GeoDataFrame(p, geometry=gpd.points_from_xy(p.lon, p.lat), crs=4326).to_parquet(POI_CACHE)
    log(f"sieć piesza z OSM: {len(xy)} węzłów, {len(edges)} krawędzi, {edges[:, 2].sum() / 1000:,.0f} km; cele z OSM: {len(p)}")


def load_walk_graph():
    if not WALK.exists() or not POI_CACHE.exists():
        build_walk_network()
    z = np.load(WALK)
    xy, e = z["xy"], z["edges"]
    u, v, w = e[:, 0].astype(np.int64), e[:, 1].astype(np.int64), e[:, 2]
    a, b = np.minimum(u, v), np.maximum(u, v)
    o = np.lexsort((w, b, a))
    a, b, w = a[o], b[o], w[o]
    k = np.ones(len(a), bool)
    k[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
    a, b, w = a[k], b[k], np.maximum(w[k], 0.01)
    n = len(xy)
    G = sparse.csr_matrix((np.concatenate([w, w]), (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    main = lab == np.bincount(lab).argmax()
    ids = np.flatnonzero(main)
    log(f"sieć piesza: {n} węzłów, główna spójna część {main.mean():.1%}")
    return G, xy, ids, cKDTree(xy[ids])


def snap(tree, ids, pts):
    d, i = tree.query(pts)
    return ids[i], d


def multi_source(G, src, limit=2500.0):
    if len(src) == 0:
        return np.full(G.shape[0], np.inf)
    return dijkstra(G, directed=True, indices=np.unique(src), limit=limit, min_only=True)


# ---------------------------------------------------------------- GTFS: kursy w szczycie na słupkach (cache)
def kategoria(n: str, rt: str) -> str:
    """Kategoria linii ZTM: czy zatrzymuje się na wszystkich przystankach."""
    if rt == "0":
        return "tramwaj"
    if rt in ("1", "2"):
        return MODE[rt]
    if n.startswith("N"):
        return "nocna"
    if n.startswith("E"):
        return "ekspresowa"
    if n.startswith(("C", "Z")):
        return "specjalna"
    if n.startswith("L"):
        return "lokalna"
    if n.isdigit() and len(n) == 3:
        return {"1": "zwykła", "2": "zwykła", "3": "zwykła okresowa", "4": "przyspieszona", "5": "przyspieszona",
                "7": "strefowa", "8": "strefowa przyspieszona", "9": "specjalna"}[n[0]]
    return "inna"


def build_gtfs_stops() -> None:
    cal = pd.read_csv(GTFS / "calendar_dates.txt", dtype=str)
    cal["dt"] = pd.to_datetime(cal["date"])
    wed = cal[(cal["dt"].dt.weekday == 2) & cal["service_id"].str.endswith(":PcS")].sort_values("dt")
    day = wed["service_id"].iloc[0].split(":")[0]          # środa — jak w cechy_klastrow.py
    routes = pd.read_csv(GTFS / "routes.txt", dtype=str)
    trips = pd.read_csv(GTFS / "trips.txt", dtype=str, usecols=["trip_id", "route_id", "service_id"])
    trips = trips[trips["service_id"].str.startswith(day)]
    st = pd.read_csv(GTFS / "stop_times.txt", dtype=str, usecols=["trip_id", "stop_id", "departure_time"], engine="pyarrow")
    st = st[st["trip_id"].str.startswith(day)]
    t = st["departure_time"].str.split(":", expand=True).astype(int)
    st = st[((t[0] * 3600 + t[1] * 60 + t[2]) >= PEAK[0]) & ((t[0] * 3600 + t[1] * 60 + t[2]) < PEAK[1])]
    st = st.merge(trips[["trip_id", "route_id"]], on="trip_id").merge(routes[["route_id", "route_short_name", "route_type"]], on="route_id")
    ls = st.groupby(["stop_id", "route_short_name", "route_type"]).size().rename("kursy").reset_index()
    ls["kursy_h"] = ls["kursy"] / ((PEAK[1] - PEAK[0]) / 3600)
    ls["kategoria"] = [kategoria(n, rt) for n, rt in zip(ls["route_short_name"], ls["route_type"])]
    ls = ls.rename(columns={"route_short_name": "linia"})
    stops = pd.read_csv(GTFS / "stops.txt", dtype={"stop_id": str})
    agg = ls.groupby("stop_id").agg(kursy_h=("kursy_h", "sum"), linie=("linia", srt), route_type=("route_type", "min"))
    s = stops.merge(agg, left_on="stop_id", right_index=True).rename(columns={"stop_name": "nazwa", "stop_lat": "lat", "stop_lon": "lon"})
    s["tryb"] = s["route_type"].map(MODE)
    s[["stop_id", "nazwa", "stop_code", "lon", "lat", "tryb", "kursy_h", "linie"]].to_parquet(STOPS_CACHE)
    ls.drop(columns="route_type").to_parquet(LINES_CACHE)
    log(f"GTFS {day}, szczyt 7–9: {len(s)} słupków z kursami, {ls['linia'].nunique()} linii")


# ---------------------------------------------------------------- klastry, popyt, obsługa
def clusters():
    df = pd.read_csv(DANE / "klastry" / "clastry_all.csv").copy()
    dens = df["pop_2025"] / df["area_km2"].clip(lower=0.05)
    df["strefa"] = np.where((df["dzielnica"] == "Śródmieście") | (df["odl_centrum_km"] <= 3.0), "śródmiejska",
                            np.where((dens >= 2500) | (df["odl_centrum_km"] <= 7.0), "miejska", "przedmieścia"))
    df["R"] = df["strefa"].map(R_ZONE)
    return df.set_index("h3")


def destinations() -> pd.DataFrame:
    inf = gpd.read_file(DANE / "infrastruktura" / "warszawa_infrastruktura.geojson")
    a = pd.DataFrame({"lon": inf["lon"], "lat": inf["lat"], "cel_typ": inf["kategoria"], "cel_nazwa": inf["nazwa"]})
    px = gpd.read_parquet(POI_CACHE)
    b = pd.DataFrame({"lon": px["lon"], "lat": px["lat"], "cel_typ": px["rodzaj"], "cel_nazwa": px["nazwa"]})
    d = pd.concat([a, b], ignore_index=True)
    d["cele"] = d["cel_typ"].map(DEST_EQ).fillna(15)
    # gabinet lekarski (amenity=doctors) i mała szkoła wyższa (college) ważą mniej niż przychodnia / uczelnia
    w_osm = pd.concat([pd.Series(np.nan, index=a.index), px["rownowaznik_osob"].reset_index(drop=True)], ignore_index=True)
    d.loc[(d["cel_typ"] == "przychodnia") & (w_osm < 100), "cele"] = 30
    d.loc[(d["cel_typ"] == "uczelnia") & (w_osm < 300), "cele"] = 60
    # ten sam obiekt kilka razy (np. szkoła podstawowa + liceum w jednym budynku) — jeden na kratkę 60 m i typ
    x, y = TO_PL.transform(d["lon"].values, d["lat"].values)
    d["k"] = d["cel_typ"] + "_" + (x // 60).astype(int).astype(str) + "_" + (y // 60).astype(int).astype(str)
    return d.sort_values("cele", ascending=False).drop_duplicates("k").drop(columns="k")


def demand(cl: pd.DataFrame, app: dict) -> pd.DataFrame:
    b = gpd.read_parquet(CACHE / "budynki_mieszkalne.parquet")
    b["h3"] = h3_of(b["lon"].values, b["lat"].values)
    b = b[b["h3"].isin(cl.index)]
    res = b[(b["kind"] == "dom") | ((b["kind"] == "yes") & b["pow_zabudowy"].between(*YES_AREA))].copy()
    res["w"] = res["m2"] * res["kind"].map(RES_TYPES)
    res["mieszk_2025"] = res["w"] / res.groupby("h3")["w"].transform("sum") * res["h3"].map(cl["pop_2025"])
    parts = [pd.DataFrame({"lon": res["lon"], "lat": res["lat"], "h3": res["h3"], "rodzaj": "mieszkancy",
                           "mieszk_2025": res["mieszk_2025"]})]
    # klastry z ludnością, ale bez budynków mieszkalnych w OSM — punkty GHSL 100 m z symulatora
    miss = set(cl.index[cl["pop_2025"] > 0]) - set(res["h3"])
    hexes = [hx["h3"] for hx in app["hexes"]]
    ppl = pd.DataFrame(app["people"], columns=["lon", "lat", "hid", "p"])
    ppl["h3"] = [hexes[i] for i in ppl["hid"]]
    ppl = ppl[ppl["h3"].isin(miss)]
    ppl["mieszk_2025"] = ppl["p"] / ppl.groupby("h3")["p"].transform("sum") * ppl["h3"].map(cl["pop_2025"])
    parts.append(pd.DataFrame({"lon": ppl["lon"], "lat": ppl["lat"], "h3": ppl["h3"], "rodzaj": "mieszkancy",
                               "mieszk_2025": ppl["mieszk_2025"]}))
    nb = pd.DataFrame([{"lon": x["xy"][0], "lat": x["xy"][1], "nowi_2036": x["series"][-1]} for x in app["newbuild"]])
    nb["h3"] = h3_of(nb["lon"].values, nb["lat"].values)
    parts.append(nb.assign(rodzaj="nowe_budynki"))
    wk = b[b["kind"] == "praca"]
    parts.append(pd.DataFrame({"lon": wk["lon"], "lat": wk["lat"], "h3": wk["h3"], "rodzaj": "praca",
                               "praca": np.minimum(wk["m2"] / M2_PER_JOB, JOB_CAP)}))
    de = destinations()
    de["h3"] = h3_of(de["lon"].values, de["lat"].values)
    parts.append(de.assign(rodzaj="cel"))
    d = pd.concat(parts, ignore_index=True)
    d = d[d["h3"].isin(cl.index)].copy()
    for c in ["mieszk_2025", "nowi_2036", "praca", "cele"]:
        d[c] = d[c].fillna(0.0)
    d["w_dzis"] = d["mieszk_2025"] + JOB_W * d["praca"] + d["cele"]
    d["w_2036"] = d["w_dzis"] + d["nowi_2036"]
    d = d[d["w_2036"] > 0].reset_index(drop=True)
    d["x"], d["y"] = TO_PL.transform(d["lon"].values, d["lat"].values)
    d["R"] = d["h3"].map(cl["R"]).values
    log(f"popyt: {len(d)} punktów; mieszkańcy 2025 {d['mieszk_2025'].sum():,.0f}, nowi 2036 {d['nowi_2036'].sum():,.0f}, "
        f"praca {d['praca'].sum():,.0f}, cele {d['cele'].sum():,.0f} os.-równ.")
    return d


def supply(app):
    s = pd.read_parquet(STOPS_CACHE)
    s = s[s["tryb"].isin(["autobus", "tramwaj"])].copy()
    s["x"], s["y"] = TO_PL.transform(s["lon"].values, s["lat"].values)
    rail = pd.DataFrame(app["stops"]["rail"], columns=["lon", "lat", "nazwa", "system", "linie"])
    rail["x"], rail["y"] = TO_PL.transform(rail["lon"].values, rail["lat"].values)
    hubs = gpd.read_parquet(DATA / "hubs.parquet")
    plan = hubs[(hubs["status"] != "istniejący") & (hubs["open_year"] <= PLANNED_MAX_YEAR)].copy()
    plan["lon"], plan["lat"] = plan.geometry.x, plan.geometry.y
    plan["x"], plan["y"] = TO_PL.transform(plan["lon"].values, plan["lat"].values)
    return s, rail, plan.reset_index(drop=True)


FREQ_CLASSES = [(6, 1e9), (4, 6), (2, 4), (1, 2), (0.01, 1)]   # kursy/h; g z dolnej granicy klasy (ostrożnie)


def access(G, tree, ids, d, s, rail, plan):
    """A_dziś, A_2036 i odległości piesze do przystanków / metra / kolei dla każdego punktu popytu."""
    node, off = snap(tree, ids, d[["x", "y"]].values)
    d["node"], d["off"] = node, off
    R = d["R"].values
    A = np.zeros(len(d))
    dist_any = np.full(len(d), np.inf)
    s_node, _ = snap(tree, ids, s[["x", "y"]].values)
    for lo, hi in FREQ_CLASSES:
        m = ((s["kursy_h"] >= lo) & (s["kursy_h"] < hi)).values
        dist = multi_source(G, s_node[m])[node] + off
        A = np.maximum(A, f_bus(dist, R) * g_freq(max(lo, 0.5)))
        dist_any = np.minimum(dist_any, dist)
    r_node, _ = snap(tree, ids, rail[["x", "y"]].values)
    metro = (rail["system"] == "metro").values
    dist_metro = multi_source(G, r_node[metro])[node] + off
    dist_rail = multi_source(G, r_node[~metro])[node] + off
    A = np.maximum(A, np.maximum(f_rapid(dist_metro), f_rapid(dist_rail) * G_RAIL))
    d["A_dzis"], d["dojscie_przystanek_m"] = A, dist_any
    p_node, _ = snap(tree, ids, plan[["x", "y"]].values)
    tram, pm = (plan["kind"] == "tramwaj").values, (plan["kind"] == "metro").values
    d["dist_plan_tram"] = multi_source(G, p_node[tram])[node] + off
    d["dist_plan_kolej"] = multi_source(G, p_node[~tram])[node] + off
    A36 = np.maximum(A, f_bus(d["dist_plan_tram"].values, R))
    A36 = np.maximum(A36, f_rapid(multi_source(G, p_node[pm])[node] + off))
    A36 = np.maximum(A36, f_rapid(multi_source(G, p_node[~tram & ~pm])[node] + off) * G_RAIL)
    d["A_2036"] = A36
    d["w_std_dzis"] = (dist_any <= R) | (dist_metro <= R_RAPID) | (dist_rail <= R_RAPID)
    return d


# ---------------------------------------------------------------- graf autobusowy, trasy linii, obsługa B
def route_segments():
    routes = gpd.read_file(DANE / "komunikacja" / "warszawa_linie_trasy.geojson")
    routes = routes[routes["typ"].isin(["autobus", "tram"])].reset_index(drop=True)
    seg = shapely.segmentize(routes.to_crs(2180).geometry.values, 50)
    xy, own = shapely.get_coordinates(seg, return_index=True)
    same = own[1:] == own[:-1]
    segs = shapely.linestrings(np.stack([xy[:-1][same], xy[1:][same]], axis=1))
    return routes, segs, own[:-1][same], shapely.STRtree(segs)


def lines_near(points_xy, routes, segs, seg_route, tree, dist_bus=ON_ROUTE["autobus"], dist_tram=ON_ROUTE["tram"]):
    cp = shapely.points(points_xy[:, 0], points_xy[:, 1])
    ci, si = tree.query(cp, predicate="dwithin", distance=max(dist_bus, dist_tram))
    typ = routes["typ"].values[seg_route[si]]
    ok = shapely.distance(cp[ci], segs[si]) <= np.where(typ == "tram", dist_tram, dist_bus)
    return pd.DataFrame({"c": ci[ok], "linia": routes["route_short_name"].values[seg_route[si[ok]]],
                         "route_idx": seg_route[si[ok]]})


def bus_graph(app):
    g = app["graph"]
    f = {k: i for i, k in enumerate(g["fields"])}
    E = g["edges"]
    ed = pd.DataFrame({"u": [e[f["u"]] for e in E], "v": [e[f["v"]] for e in E], "len_m": [e[f["len_m"]] for e in E],
                       "cls": [g["classes"][e[f["cls"]]] for e in E],
                       "ulica": [g["names"][e[f["name"]]] if e[f["name"]] is not None and e[f["name"]] >= 0 else "" for e in E],
                       "stop_ok": [e[f["stop_ok"]] for e in E], "edge_id": [e[f["edge_id"]] for e in E],
                       "coords": [e[f["coords"]] for e in E]})
    osm = pd.read_parquet(DATA / "roads_edges.parquet", columns=["edge_id", "highway", "surface", "access", "bus", "psv"])
    ed = ed.merge(osm, on="edge_id", how="left")
    hw = ed["highway"].map(tokens)
    psv_ok = ed["bus"].astype(str).isin(["yes", "designated"]) | ed["psv"].astype(str).isin(["yes", "designated"])
    unpaved = ed["surface"].map(lambda v: bool(tokens(v) & UNPAVED))
    restricted = ed["access"].map(lambda v: bool(tokens(v) & BAD_ACCESS))
    ed["ulica_ok_B"] = (hw.map(lambda t: bool(t & B_HIGHWAYS) and "living_street" not in t and "service" not in t) | psv_ok) \
        & ~unpaved & ~restricted
    nodes = np.array(g["nodes"])
    return ed, np.column_stack(TO_PL.transform(nodes[:, 0], nodes[:, 1]))


def routing_graph(ed, bnodes, routes, segs, seg_route, rtree):
    """Ulice, którymi można poprowadzić autobus (dziś jeździ albo ulica ok dla B) + węzły na trasach linii.

    Krawędzie ślepych odnóg (poza 2-rdzeniem grafu nieskierowanego) oznaczamy — autobus musiałby tam zawracać."""
    mids = np.array([TO_PL.transform(*np.array(c)[len(c) // 2]) for c in ed["coords"]])
    bus_r = routes["typ"].values == "autobus"
    hit = lines_near(mids, routes, segs, seg_route, rtree, dist_bus=12, dist_tram=-1)
    on_bus = np.zeros(len(ed), bool)
    on_bus[hit.loc[bus_r[hit["route_idx"].values], "c"].unique()] = True
    ok = ed["ulica_ok_B"].values | on_bus
    e = ed[ok]
    n = len(bnodes)
    G = sparse.csr_matrix((e["len_m"].values.astype(float) + 0.01, (e["u"].values, e["v"].values)), shape=(n, n))
    und = nx.Graph()
    und.add_edges_from(zip(e["u"].values, e["v"].values))
    und.remove_edges_from(nx.selfloop_edges(und))
    core = set(nx.k_core(und, 2).nodes)
    ed["przelotowa"] = ed["u"].isin(core) & ed["v"].isin(core) & ok
    nh = lines_near(bnodes, routes, segs, seg_route, rtree, dist_bus=15, dist_tram=-1)
    return G, nh[bus_r[nh["route_idx"].values]]


class Service:
    """Którą istniejącą linią obsłużyć przystanek B: objazd (zjazd z trasy → odcinek z przystankiem → powrót, bez
    zawracania na końcach odcinka) albo przedłużenie od pętli. Koszt = dodatkowe wozokm/dzień."""

    def __init__(self, ed, bnodes, routes, segs, seg_route, rtree, s):
        self.ed = ed
        self.G, nh = routing_graph(ed, bnodes, routes, segs, seg_route, rtree)
        ls = pd.read_parquet(LINES_CACHE)
        self.kat = ls.groupby("linia")["kategoria"].first().to_dict()
        self.freq = ls.groupby("linia")["kursy_h"].median().to_dict()
        self.trips = routes.groupby("route_short_name")["kursy_kierunek"].sum().to_dict()
        self.groups = [(routes.at[r, "route_short_name"], g["c"].values) for r, g in nh.groupby("route_idx")
                       if self.kat.get(routes.at[r, "route_short_name"]) in DIVERTABLE]
        ends = routes[(routes["typ"] == "autobus") & routes["route_short_name"].map(lambda l: self.kat.get(l) in DIVERTABLE)]
        exy = np.column_stack(TO_PL.transform(ends["koniec_lon"].values, ends["koniec_lat"].values))
        dist, node = cKDTree(bnodes).query(exy)
        _, js = cKDTree(s[["x", "y"]].values).query(exy)
        self.ends = [(l, n, k, s["nazwa"].values[j]) for l, n, k, j, dd in
                     zip(ends["route_short_name"], node, ends["kursy_kierunek"], js, dist) if dd <= 80]
        self.cache = {}

    def options(self, e):
        if e in self.cache:
            return self.cache[e]
        r = self.ed.loc[e]
        u, v, elen = int(r.u), int(r.v), float(r.len_m)
        G2 = self.G.tolil(copy=True)
        G2[v, u] = 0
        G2 = G2.tocsr()
        G2.eliminate_zeros()
        d_in = dijkstra(G2.T.tocsr(), directed=True, indices=u, limit=ROUTE_LIMIT)
        d_out = dijkstra(G2, directed=True, indices=v, limit=ROUTE_LIMIT)
        best = {}
        for line, nodes in self.groups:
            km = (d_in[nodes].min() + elen + d_out[nodes].min()) / 1000
            if np.isfinite(km) and km <= MAX_DETOUR_KM and (line not in best or km < best[line]["km"]):
                best[line] = {"linia": line, "rodzaj": "zmiana trasy", "km": km, "wozokm": km * self.trips.get(line, 0)}
        ext = {}
        for line, n, k, petla in self.ends:
            km = (d_in[n] + elen + d_out[n]) / 1000
            if np.isfinite(km) and km <= MAX_EXT_KM and (line not in ext or km < ext[line]["km"]):
                ext[line] = {"linia": line, "rodzaj": "przedłużenie", "km": km, "wozokm": km * k, "petla": petla}
        opts = list(best.values()) + list(ext.values())
        for o in opts:
            o["kategoria"] = self.kat.get(o["linia"], "")
            o["kursy_h"] = self.freq.get(o["linia"], 0.0)
            # rekomendacja: linia całodzienna z ≥ 2 kursami/h, potem najmniej dodatkowych wozokm
            o["klucz"] = (not (o["kategoria"] != "zwykła okresowa" and o["kursy_h"] >= 2), o["wozokm"])
        opts.sort(key=lambda o: o["klucz"])
        self.cache[e] = opts
        return opts


# ---------------------------------------------------------------- kandydaci, pokrycie, wybór
def candidates(ed, cl, s, routes, segs, seg_route, rtree):
    pts = []
    for r in ed[ed["stop_ok"] == 1].itertuples():
        line = shapely.LineString(np.column_stack(TO_PL.transform(*np.array(r.coords).T)))
        for t in np.arange(FROM_JUNCTION_M, line.length - FROM_JUNCTION_M + 1e-6, SAMPLE_M):
            p = line.interpolate(t)
            pts.append((r.Index, p.x, p.y))
    c = pd.DataFrame(pts, columns=["e", "x", "y"])
    c = c.join(ed[["edge_id", "ulica", "cls", "u", "v", "highway", "surface", "ulica_ok_B", "przelotowa"]], on="e")
    # obie jezdnie / oba kierunki dają te same punkty — zostaje jeden na kratkę 10 m
    c["key"] = (c["x"] // 10).astype(int).astype(str) + "_" + (c["y"] // 10).astype(int).astype(str)
    c = c.drop_duplicates("key").drop(columns="key").reset_index(drop=True)
    c["lon"], c["lat"] = TO_WGS.transform(c["x"].values, c["y"].values)
    c["h3"] = h3_of(c["lon"].values, c["lat"].values)
    c = c[c["h3"].isin(cl.index)].reset_index(drop=True)
    # linie jadące tą ulicą; częstotliwość linii = jej kursy/h na najbliższym słupku tej linii
    pairs = lines_near(c[["x", "y"]].values, routes, segs, seg_route, rtree).drop(columns="route_idx").drop_duplicates()
    ls = pd.read_parquet(LINES_CACHE).merge(s[["stop_id", "x", "y"]], on="stop_id")
    pairs["kategoria"] = pairs["linia"].map(ls.groupby("linia")["kategoria"].first()).fillna("inna")
    pairs["kursy_h"], pairs["odl_slupek_linii"] = 0.0, np.inf
    for line, grp in pairs.groupby("linia"):
        lp = ls[ls["linia"] == line]
        if lp.empty:
            continue
        dist, j = cKDTree(lp[["x", "y"]].values).query(c.loc[grp["c"], ["x", "y"]].values)
        pairs.loc[grp.index, "kursy_h"] = lp["kursy_h"].values[j]
        pairs.loc[grp.index, "odl_slupek_linii"] = dist
    pairs["zatrzymuje"] = pairs["kategoria"].isin(STOPPING)
    st = pairs[pairs["zatrzymuje"]]
    c = c.join(st.groupby("c").agg(linie_zatrzymujace=("linia", srt), kursy_h=("kursy_h", "sum"),
                                   min_odl_slupek_linii=("odl_slupek_linii", "min"),
                                   tylko_tramwaj=("kategoria", lambda k: bool((k == "tramwaj").all()))))
    c = c.join(pairs[~pairs["zatrzymuje"]].groupby("c").agg(linie_przyspieszone=("linia", srt)))
    c["kursy_h"] = c["kursy_h"].fillna(0.0)
    dist, j = cKDTree(s[["x", "y"]].values).query(c[["x", "y"]].values)
    c["odl_slupek_m"], c["najblizszy_slupek"] = dist, s["nazwa"].values[j]
    c["typ"] = np.where(c["linie_zatrzymujace"].notna(), "A", "B")
    far_b = (c["typ"] == "B") & (c["odl_slupek_m"] >= ANY_STOP_MIN_B)
    feasible = np.where(c["typ"] == "A", c["min_odl_slupek_linii"].fillna(np.inf) >= SAME_LINE_MIN,
                        far_b & c["ulica_ok_B"] & c["przelotowa"])
    c["g"] = np.where(c["typ"] == "A", g_freq(np.maximum(c["kursy_h"], 0.5)), g_freq(FREQ_NEW_LINE))
    log(f"kandydaci: {len(c)} punktów na ulicach dla autobusu, dopuszczalnych {feasible.sum()} "
        f"(A: {((c['typ'] == 'A') & feasible).sum()}, B: {((c['typ'] == 'B') & feasible).sum()}); odrzucone B: ulica "
        f"gruntowa/prywatna/strefa zam./serwisowa {(far_b & ~c['ulica_ok_B']).sum()}, ślepa odnoga "
        f"{(far_b & c['ulica_ok_B'] & ~c['przelotowa']).sum()}")
    return c[feasible].reset_index(drop=True)


def coverage(G, tree, ids, d, c):
    """Dla każdego kandydata: punkty popytu do poprawy w zasięgu LIMIT m pieszo (indeksy + odległość)."""
    imp = np.flatnonzero((d["A_dzis"].values < 0.999) | (d["A_2036"].values < 0.999))
    pot_w = 0.5 * d["w_dzis"].values * (1 - d["A_dzis"].values) + 0.5 * d["w_2036"].values * (1 - d["A_2036"].values)
    near = cKDTree(d[["x", "y"]].values[imp]).query_ball_point(c[["x", "y"]].values, r=LIMIT)
    pot = np.array([pot_w[imp[n]].sum() if n else 0.0 for n in near])
    c = c[pot >= min(MIN_GAIN.values())].reset_index(drop=True)
    c["node"], c["off"] = snap(tree, ids, c[["x", "y"]].values)
    dn = d["node"].values[imp]
    order = np.argsort(dn, kind="stable")
    dn_sorted, imp_sorted = dn[order], imp[order]
    off = d["off"].values
    uniq_nodes = np.unique(c["node"].values)
    res = {}
    t0 = time.time()
    for k in range(0, len(uniq_nodes), 16):
        batch = uniq_nodes[k:k + 16]
        D = dijkstra(G, directed=True, indices=batch, limit=LIMIT)
        for r, nd in enumerate(batch):
            row = D[r]
            reach = np.flatnonzero(np.isfinite(row))
            lo = np.searchsorted(dn_sorted, reach, "left")
            hi = np.searchsorted(dn_sorted, reach, "right")
            has = hi > lo
            reach, lo, hi = reach[has], lo[has], hi[has]
            if len(reach) == 0:
                res[nd] = (np.empty(0, np.int64), np.empty(0, np.float32))
                continue
            di = imp_sorted[np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])]
            res[nd] = (di, (np.repeat(row[reach], hi - lo) + off[di]).astype(np.float32))
    log(f"pokrycie: {len(c)} kandydatów z potencjałem, {len(uniq_nodes)} węzłów ({time.time() - t0:.0f} s)")
    return c, res


def greedy(d, c, cov, A0, A1, typ, chosen_xy, service=None):
    w0, w1, R = d["w_dzis"].values, d["w_2036"].values, d["R"].values
    rejected = 0

    def gain(i):
        di, dist = cov[c.at[i, "node"]]
        a = f_bus(dist + c.at[i, "off"], R[di]) * c.at[i, "g"]
        return 0.5 * (w0[di] * np.clip(a - A0[di], 0, None)).sum() + 0.5 * (w1[di] * np.clip(a - A1[di], 0, None)).sum()

    heap = [(-gain(i), i) for i in np.flatnonzero(c["typ"].values == typ)]
    heapq.heapify(heap)
    chosen = []
    while heap and len(chosen) < MAX_STOPS[typ]:
        _, i = heapq.heappop(heap)
        p = c.loc[i, ["x", "y"]].values.astype(float)
        if chosen_xy and np.min(np.hypot(*(np.array(chosen_xy) - p).T)) < MIN_SPACING:
            continue
        g_now = gain(i)
        if heap and g_now < -heap[0][0] - 1e-9:      # leniwa ocena: zysk mógł spaść po wcześniejszych wyborach
            heapq.heappush(heap, (-g_now, i))
            continue
        if g_now < MIN_GAIN[typ]:
            break
        if service is not None and not service.options(int(c.at[i, "e"])):
            rejected += 1          # autobus nie dojedzie bez zawracania albo żadna linia nie ma rozsądnego objazdu
            continue
        di, dist = cov[c.at[i, "node"]]
        dist = dist + c.at[i, "off"]
        a = f_bus(dist, R[di]) * c.at[i, "g"]
        chosen.append({"i": i, "di": di, "dist": dist, "zysk": g_now})
        chosen_xy.append(p)
        A0[di] = np.maximum(A0[di], a)
        A1[di] = np.maximum(A1[di], a)
    log(f"etap {typ}: {len(chosen)} przystanków, zysk {sum(x['zysk'] for x in chosen):,.0f}"
        + (f"; odrzucone bez obsługi linią (zawracanie / objazd > {MAX_DETOUR_KM} km): {rejected}" if service else ""))
    return chosen


# ---------------------------------------------------------------- raport
def describe(ch, c, d, cl, ed, rail, plan, service, vtree, vn, Gw, xy):
    r = c.loc[ch["i"]]
    di, dist = ch["di"], ch["dist"]
    sub = d.iloc[di]
    inR = dist <= sub["R"].values
    newly = inR & ~sub["w_std_dzis"].values
    m, nw, jobs = sub["mieszk_2025"].values, sub["nowi_2036"].values, sub["praca"].values
    _, jj = vtree.query([r.x, r.y], k=80)
    street = r.ulica or vn[jj[0]]
    cross = next((vn[j] for j in jj if vn[j] not in ("", street)), "")
    cel = sub[(sub["rodzaj"] == "cel").values & inR]
    cel_txt = ", ".join(f"{DEST_PL.get(k, k)} ×{v}" if v > 1 else DEST_PL.get(k, k) for k, v in cel["cel_typ"].value_counts().items())
    cel_named = [n for n in cel.sort_values("cele", ascending=False)["cel_nazwa"].dropna().astype(str) if n and n != "nan"][:3]
    dzis = np.average(sub["dojscie_przystanek_m"].values[newly], weights=m[newly] + 1e-9) if newly.any() else np.nan
    po = np.average(dist[newly], weights=m[newly] + 1e-9) if newly.any() else np.nan
    k = cl.loc[r.h3]
    strefa, Rz = k["strefa"], int(k["R"])
    dr, jr = cKDTree(rail[["x", "y"]].values).query([r.x, r.y])
    dp, jp = cKDTree(plan[["x", "y"]].values).query([r.x, r.y])
    plan_txt = f"{plan.at[jp, 'project'] or plan.at[jp, 'name']} ({int(plan.at[jp, 'open_year'])}) — {fmt(dp)} m" if dp <= 1200 else ""
    opts = service.options(int(r.e)) if r.typ == "B" else []
    parts = []
    for o in opts[:3]:
        okres = ", linia okresowa (szczyt)" if o["kategoria"] == "zwykła okresowa" else ""
        info = f"{fmt(o['kursy_h'], 1)} kursów/h{okres}; +{fmt(o['km'], 1)} km/kurs, ok. +{fmt(o['wozokm'])} wozokm/dzień"
        parts.append(f"zmiana trasy linii {o['linia']} ({info})" if o["rodzaj"] == "zmiana trasy"
                     else f"przedłużenie linii {o['linia']} od pętli {o['petla']} ({info})")
    trasa = "; ".join(parts)
    tx = []
    if newly.any() and m[newly].sum() >= 1:
        tx.append(f"{fmt(m[newly].sum())} mieszkańców (2025) zyska przystanek w standardzie ZTM ({Rz} m pieszo, strefa {strefa}); "
                  f"dziś mają średnio {fmt(dzis)} m dojścia, po zmianie {fmt(po)} m")
    else:
        tx.append(f"{fmt(m[inR].sum())} mieszkańców w zasięgu {Rz} m pieszo (dziś w standardzie, ale z rzadką obsługą)")
    if nw[inR].sum() >= 20:
        tx.append(f"do 2036 r. w zasięgu dojdzie ok. {fmt(nw[inR].sum())} nowych mieszkańców (pozwolenia GUNB)")
    if jobs[inR].sum() >= 100:
        tx.append(f"ok. {fmt(jobs[inR].sum())} miejsc pracy w zasięgu")
    if cel_txt:
        tx.append("cele w zasięgu: " + cel_txt + (f" ({'; '.join(cel_named)})" if cel_named else ""))
    if r.typ == "A":
        tx.append(f"ulicą jeżdżą już linie {r.linie_zatrzymujace} ({fmt(r.kursy_h, 1)} kursów/h w szczycie), a najbliższy "
                  f"ich przystanek jest ≥ {fmt(r.min_odl_slupek_linii)} m — wystarczy dostawić słupki")
        if str(r.tylko_tramwaj) == "True":
            tx.append("przystanek tramwajowy — wymaga peronów (wyższy koszt)")
        if str(r.highway).split(";")[0] in ("primary", "secondary"):
            tx.append("ulica główna — przystanek wymaga przejścia dla pieszych / azylu")
    else:
        tx.append(f"żadna linia nie jedzie tą ulicą; najbliższy przystanek ({r.najblizszy_slupek}) {fmt(r.odl_slupek_m)} m w linii "
                  f"prostej; ulica utwardzona, publiczna i przelotowa. Obsługa: {trasa}")
    if k["udzial_65_"] >= 0.2:
        tx.append(f"w klastrze {fmt(k['udzial_65_'] * 100)}% osób 65+")
    if plan_txt:
        tx.append(f"w pobliżu planowana inwestycja: {plan_txt} — przystanek jako rozwiązanie na lata do jej otwarcia i dowóz do niej")
    tx.append(f"klaster {r.h3}: wskaźnik wykluczenia {fmt(k['pred_wykluczenie'], 2)} (miejsce {int(k['rank_wykluczenie'])}/848)")
    reach = np.flatnonzero(np.isfinite(dijkstra(Gw, directed=True, indices=int(r.node), limit=Rz)))
    poly = shapely.concave_hull(shapely.multipoints(xy[reach]), ratio=0.15).buffer(25).union(
        shapely.Point(r.x, r.y).buffer(30)).simplify(10)
    poly = shapely.transform(poly, lambda p: np.column_stack(TO_WGS.transform(p[:, 0], p[:, 1])))
    return {
        "typ": r.typ,
        "rodzaj": "dostawienie przystanku na istniejącej trasie" if r.typ == "A" else "nowy przystanek + zmiana trasy / przedłużenie linii",
        "nazwa_robocza": f"{street} / {cross}" if cross else street, "dzielnica": k["dzielnica"], "h3": r.h3,
        "lon": round(r.lon, 6), "lat": round(r.lat, 6), "punkt_wkt": f"POINT ({r.lon:.6f} {r.lat:.6f})",
        "geojson": json.dumps({"type": "Point", "coordinates": [round(r.lon, 6), round(r.lat, 6)]}),
        "zysk_dostepnosci": round(ch["zysk"]), "strefa_ztm": strefa, "prog_dojscia_m": Rz,
        "mieszkancy_2025_w_zasiegu": round(m[inR].sum()), "mieszkancy_2025_nowo_w_standardzie": round(m[newly].sum()),
        "dojscie_dzis_m": None if np.isnan(dzis) else round(dzis), "dojscie_po_m": None if np.isnan(po) else round(po),
        "nowi_mieszkancy_2036_w_zasiegu": round(nw[inR].sum()), "miejsca_pracy_w_zasiegu": round(jobs[inR].sum()),
        "cele_w_zasiegu": cel_txt,
        "linie_zatrzymujace_sie": r.linie_zatrzymujace if r.typ == "A" else "",
        "kursy_h_szczyt": round(r.kursy_h, 1) if r.typ == "A" else FREQ_NEW_LINE,
        "linie_przyspieszone_na_ulicy": r.linie_przyspieszone if isinstance(r.linie_przyspieszone, str) else "",
        "propozycja_obslugi": trasa if r.typ == "B" else f"nowy przystanek na trasie linii {r.linie_zatrzymujace}",
        "linia_rekomendowana": (opts[0]["linia"] if opts else "") if r.typ == "B" else r.linie_zatrzymujace,
        "dodatkowe_wozokm_dzien": round(opts[0]["wozokm"]) if opts else None,
        "najblizszy_przystanek": r.najblizszy_slupek, "odl_najblizszy_przystanek_m": round(r.odl_slupek_m),
        "najblizsza_stacja_kolej_metro": f"{rail.at[jr, 'nazwa']} ({rail.at[jr, 'system']}) — {fmt(dr)} m",
        "planowana_inwestycja_w_poblizu": plan_txt,
        "ulica_klasa": str(r.highway), "ulica_nawierzchnia": "" if str(r.surface) in ("nan", "<NA>", "None") else str(r.surface),
        "klaster_pred_wykluczenie": round(k["pred_wykluczenie"], 3), "klaster_rank_wykluczenie": int(k["rank_wykluczenie"]),
        "klaster_ludnosc_2025": int(k["pop_2025"]), "klaster_ludnosc_2036": int(k["pop_2036"]),
        "klaster_mieszkania_w_budowie": int(k["mieszkania_w_budowie"]), "klaster_udzial_65": k["udzial_65_"],
        "klaster_deficyt_kursow_2036": k["deficyt_2036"],
        "uzasadnienie": ". ".join(t[0].upper() + t[1:] for t in tx) + ".",
        "zasieg_geojson": json.dumps(shapely.geometry.mapping(poly)),
        "_node": int(r.node),
    }


def cluster_summary(d, cl, out, Gw):
    new_dist = multi_source(Gw, out["_node"].values)[d["node"].values] + d["off"].values
    R = d["R"].values
    std_po = d["w_std_dzis"].values | (new_dist <= R)
    std36 = d["w_std_dzis"].values | (d["dist_plan_tram"].values <= R) | (d["dist_plan_kolej"].values <= R_RAPID)
    std36_po = std36 | (new_dist <= R)
    m, nw = d["mieszk_2025"].values, d["nowi_2036"].values
    log(f"mieszkańcy poza standardem ZTM: dziś {m[~d['w_std_dzis']].sum():,.0f} → po propozycjach {m[~std_po].sum():,.0f}; "
        f"2036 (z nowymi i planowanymi stacjami) {(m + nw)[~std36].sum():,.0f} → {(m + nw)[~std36_po].sum():,.0f}")
    t = pd.DataFrame({"h3": d["h3"], "mieszkancy_2025": m, "poza_standardem_2025": m * ~d["w_std_dzis"].values,
                      "poza_standardem_po_propozycjach": m * ~std_po, "nowi_2036": nw,
                      "nowi_2036_poza_standardem": nw * ~std36, "nowi_2036_poza_standardem_po": nw * ~std36_po,
                      "_ax": d["A_dzis"].values * m})
    g = t.groupby("h3").sum()
    g["udzial_poza_standardem_2025"] = (g["poza_standardem_2025"] / g["mieszkancy_2025"].replace(0, np.nan)).round(3)
    g["dostepnosc_A_dzis"] = (g.pop("_ax") / g["mieszkancy_2025"].replace(0, np.nan)).round(3)
    k = cl[["dzielnica", "strefa", "R", "pred_wykluczenie", "rank_wykluczenie"]].rename(columns={"strefa": "strefa_ztm", "R": "prog_dojscia_m"})
    g = k.join(g, how="left")
    g["propozycje"] = out.groupby("h3")["id"].agg(" ".join).reindex(g.index).fillna("")
    cnt = ["mieszkancy_2025", "poza_standardem_2025", "poza_standardem_po_propozycjach", "nowi_2036",
           "nowi_2036_poza_standardem", "nowi_2036_poza_standardem_po"]
    g[cnt] = g[cnt].fillna(0).round().astype(int)
    return g.reset_index().sort_values("poza_standardem_2025", ascending=False)


def main() -> None:
    if not STOPS_CACHE.exists() or not LINES_CACHE.exists():
        build_gtfs_stops()
    app = json.loads((ROOT / "frontend" / "symulator_dane.json").read_text(encoding="utf-8"))
    cl = clusters()
    G, xy, ids, tree = load_walk_graph()
    d = demand(cl, app)
    s, rail, plan = supply(app)
    d = access(G, tree, ids, d, s, rail, plan)
    ed, bnodes = bus_graph(app)
    routes, segs, seg_route, rtree = route_segments()
    service = Service(ed, bnodes, routes, segs, seg_route, rtree, s)
    c = candidates(ed, cl, s, routes, segs, seg_route, rtree)
    c, cov = coverage(G, tree, ids, d, c)
    A0, A1 = d["A_dzis"].values.copy(), d["A_2036"].values.copy()
    chosen_xy = []
    chosen = greedy(d, c, cov, A0, A1, "A", chosen_xy) + greedy(d, c, cov, A0, A1, "B", chosen_xy, service)

    names = ed[(ed["ulica"] != "") & (ed["cls"] != "express")]
    vxy = [np.column_stack(TO_PL.transform(*np.array(co).T)) for co in names["coords"]]
    vtree = cKDTree(np.vstack(vxy))
    vn = np.repeat(names["ulica"].values, [len(a) for a in vxy])
    out = pd.DataFrame([describe(ch, c, d, cl, ed, rail, plan, service, vtree, vn, G, xy) for ch in chosen])
    out.insert(0, "id", out["typ"] + out.groupby("typ").cumcount().add(1).map("{:02d}".format))
    # pakiety: kilka przystanków B obsłużonych tą samą rekomendowaną linią — kandydaci na jedną nową trasę linii
    b = out[(out["typ"] == "B") & (out["linia_rekomendowana"] != "")]
    out["pakiet"] = ""
    for line, grp in b.groupby("linia_rekomendowana"):
        if len(grp) > 1:
            out.loc[grp.index, "pakiet"] = f"linia {line}: " + ", ".join(grp["id"])
    hi = out["typ"].map(lambda t: PRIORITY[t][0])
    mid = out["typ"].map(lambda t: PRIORITY[t][1])
    out.insert(1, "etap", np.where(out["typ"] == "A", "A — od razu (tylko słupki na istniejącej trasie)",
                                   "B — wymaga zmiany trasy / przedłużenia linii"))
    out.insert(2, "priorytet", np.where(out["zysk_dostepnosci"] >= hi, 1, np.where(out["zysk_dostepnosci"] >= mid, 2, 3)))
    klastry = cluster_summary(d, cl, out, G)
    out = out.drop(columns="_node")

    out.to_csv(OUT.with_suffix(".csv"), index=False, encoding="utf-8")
    skip = {"geojson", "zasieg_geojson", "punkt_wkt"}
    clean = lambda v: None if isinstance(v, float) and np.isnan(v) else v  # noqa: E731
    pts = [{"type": "Feature", "geometry": json.loads(r["geojson"]),
            "properties": {k: clean(v) for k, v in r.items() if k not in skip}} for r in out.to_dict("records")]
    zas = [{"type": "Feature", "geometry": json.loads(r["zasieg_geojson"]),
            "properties": {k: r[k] for k in ("id", "typ", "nazwa_robocza", "prog_dojscia_m")}} for r in out.to_dict("records")]
    OUT.with_suffix(".geojson").write_text(json.dumps({"type": "FeatureCollection", "features": pts}, ensure_ascii=False), encoding="utf-8")
    (DANE / "przystanki" / "propozycje_przystankow_zasiegi.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": zas}, ensure_ascii=False), encoding="utf-8")
    klastry.to_csv(DANE / "przystanki" / "propozycje_przystankow_klastry.csv", index=False, encoding="utf-8")
    log(f"== {OUT.relative_to(ROOT)}.csv: {len(out)} propozycji (A: {(out['typ'] == 'A').sum()}, B: {(out['typ'] == 'B').sum()}); "
        f"+ .geojson, _zasiegi.geojson, _klastry.csv")


if __name__ == "__main__":
    main()
